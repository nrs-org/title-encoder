"""v7 artist-wrapper data: teach the encoder that platform/channel wrappers
around an artist name carry no identity.

Library misses at scale: `Takane Lui - Topic` ranks other `- Topic` channels
first; `Niko Ch. 虎金妃笑虎 - FLOW GLOW` ranks other FLOW GLOW members first.
The training data so far has only clean MB artist names, so nothing taught
the model to discount wrappers (v2 fixed the same problem for track titles
with same-template negatives; this is the artist counterpart).

Sources (all leakage-filtered: artists come from raw/artist_names.csv):
- **Real**: MB artist -> YouTube channel title (channels.py). Titles that
  differ from every artist name become (title, name) positives, with a
  negative that shares the title's wrapper template (`Y - Topic`).
- **Mined templates**: each channel title with the artist's own names masked
  (`{a} - Topic`, `{b} / {a}`, `{a}【公式】`), kept if seen on >= 3 artists.
- **VTuber-style templates** (hand-written; rare among MB-linked channels but
  common in the library): `{n} Ch. {a} - {g}`, `{a} / {r}【{g}】`, ... with
  `{g}` a JP label / group name. The negative is the *same template and
  group* around another artist, so only the name discriminates.

Holdout: ~3% of artists (by hash) and ~10% of group names never appear in
training; artist_wrap_test.jsonl (per-query corpora, template-test format)
and channel_test.jsonl (real titles) are built from them. Group names that
are library affiliations or library artist names are dropped entirely, so
the library eval also sees only unseen groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sqlite3
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from .artist_name import affiliations
from .paths import build_dir
from .text import allowed, norm_key, script_class

CHANNEL = re.compile(r"/channel/(UC[A-Za-z0-9_-]{22})")
LATIN_NAME = re.compile(r"^[A-Za-z][A-Za-z.'\-]*(?: [A-Za-z][A-Za-z.'\-]*){0,3}$")

# VTuber / agency channel shapes seen in the library, abstracted. Slots:
# {a} CJK name, {r} Latin alias of the same artist, {n} one token of {r}
# (the "given name" hololive puts before `Ch.`), {g} group / agency.
VT_TEMPLATES = [
    "{n} Ch. {a}", "{n} Ch. {a} - {g}", "{n} ch. {a} - {g} -", "{n} Ch. {a} ‐ {g}",
    "{r} Ch. {g}", "{r} Ch. {g}-EN", "{r} Channel {g}", "{a} / {r}【{g}】", "{a}【{g}】",
    "{a}【{g}所属】", "{a} - {r} - {g}", "{a} / {r} Channel【{g}】", "{a}ちゃんねる /{g}",
    "{n}Ch. - {r} -/{g}", "《{r}》Project channel【{g}】", "{a} Official【{g}】",
    "{a}【{g}】Official", "{r}【{a}】", "{a} Channel / {r}", "{a} / {r}",
    "{a}チャンネル【{g}】", "{r} ch. {a}【{g}】",
]


def is_test(key: str, pct: int) -> bool:
    return zlib.crc32(key.encode()) % 100 < pct


class Artist:
    def __init__(self, entity: str, tier: str):
        self.entity, self.tier = entity, tier
        self.names: list[str] = []
        self._keys: set[str] = set()

    def add(self, name: str) -> None:
        name = unicodedata.normalize("NFKC", name).strip()
        k = norm_key(name)
        if k and k not in self._keys and allowed(name):
            self._keys.add(k)
            self.names.append(name)

    def keys(self) -> set[str]:
        return self._keys

    def cjk(self) -> list[str]:
        return [n for n in self.names if script_class(n) in ("kana", "han")]

    def latin(self) -> list[str]:
        return [n for n in self.names if LATIN_NAME.match(n)]


def mask(title: str, artist: Artist) -> str | None:
    """Template of a channel title: the artist's names replaced by {a} / {b}.
    None if no name occurs (or the title *is* a name)."""
    s = unicodedata.normalize("NFKC", title).strip()
    if norm_key(s) in artist.keys():
        return None
    found = 0
    for name in sorted(artist.names, key=len, reverse=True):
        if len(norm_key(name)) < 2 or found >= 2:
            continue
        m = re.search(re.escape(name), s, re.IGNORECASE)
        if m:
            s = s[: m.start()] + ("{a}" if found == 0 else "{b}") + s[m.end():]
            found += 1
    if not found or "{" in s.replace("{a}", "").replace("{b}", ""):
        return None
    if "{b}" in s and "{a}" in s and s.index("{b}") < s.index("{a}"):
        s = s.replace("{a}", "\0").replace("{b}", "{a}").replace("\0", "{b}")
    return s


def fill(template: str, artist: Artist, group: str, rng: random.Random) -> tuple[str, set[str]] | None:
    """Wrap the artist's names in a template; returns the text and the names used."""
    used: dict[str, str] = {}
    if "{n}" in template or "{r}" in template:
        lat, cjk = artist.latin(), artist.cjk()
        if not lat or ("{a}" in template and not cjk):
            return None
        used["r"] = rng.choice(lat)
        toks = used["r"].split()
        used["n"] = toks[-1] if len(toks) > 1 and rng.random() < 0.7 else toks[0]
        if "{a}" in template:
            used["a"] = rng.choice(cjk)
    elif "{a}" in template:
        if "{b}" in template:
            if len(artist.names) < 2:
                return None
            used["a"], used["b"] = rng.sample(artist.names, 2)
        else:
            used["a"] = rng.choice(artist.names)
    text = template
    for slot, value in used.items():
        text = text.replace("{" + slot + "}", value)
    text = text.replace("{g}", group)
    return text, {used[s] for s in ("a", "b", "r") if s in used}


def positive_for(artist: Artist, used: set[str], rng: random.Random) -> str:
    """A name of the artist: usually one *not* spelled out in the wrapper
    (cross-script when possible), sometimes the bare wrapped name."""
    others = [n for n in artist.names if n not in used]
    if others and rng.random() < 0.7:
        return rng.choice(others)
    return rng.choice(sorted(used) or artist.names)


def library_exclusions(db: Path) -> tuple[set[str], set[str]]:
    """(library artist alias keys, library affiliation fragment keys)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    names = [n for (n,) in con.execute(
        """select al.name from entry_alias al
           join entry_source es on es.source = al.source and es.identifier = al.identifier
           join entry e on e.id = es.entry_id where e.entry_type = 'artist'""")]
    con.close()
    return {norm_key(n) for n in names}, {a for a in affiliations(names) if len(a) >= 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--library-db", type=Path,
                        default=Path.home() / ".local/share/musiclib-rs/musiclib.db")
    parser.add_argument("--synth-per-artist", type=int, default=2)
    parser.add_argument("--en-synth-rate", type=float, default=0.3)
    parser.add_argument("--test-pct", type=int, default=3)
    parser.add_argument("--min-template-artists", type=int, default=3)
    parser.add_argument("--test-corpus", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = build_dir(args.build)
    rng = random.Random(args.seed)

    artists: dict[str, Artist] = {}
    with (base / "raw" / "artist_names.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            a = artists.setdefault(row["entity"], Artist(row["entity"], row["tier"]))
            a.add(row["name"])
    artists = {e: a for e, a in artists.items() if a.names}

    titles = {}
    with (base / "youtube_channels.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("missing") and row.get("title"):
                titles[row["channel_id"]] = unicodedata.normalize("NFKC", row["title"]).strip()
    links: dict[str, set[str]] = defaultdict(set)
    with (base / "raw" / "artist_channels.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m = CHANNEL.search(row["url"])
            if row["artist"] in artists and m and m.group(1) in titles:
                links[row["artist"]].add(titles[m.group(1)])

    lib_names, lib_affil = library_exclusions(args.library_db)
    print(f"library: {len(lib_names)} artist name keys, {len(lib_affil)} affiliations "
          f"(e.g. {sorted(lib_affil)[:8]})")

    def mentions_library_group(text: str) -> bool:
        k = norm_key(text)
        return any(a in k for a in lib_affil)

    groups_train, groups_test = [], []
    with (base / "raw" / "groups.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            g = unicodedata.normalize("NFKC", row["name"]).strip()
            k = norm_key(g)
            if not (2 <= len(g) <= 24) or not allowed(g) or k in lib_names or mentions_library_group(g):
                continue
            (groups_test if is_test("g:" + k, 10) else groups_train).append(g)
    groups_train, groups_test = sorted(set(groups_train)), sorted(set(groups_test))

    test_artists = {e for e in artists if is_test("a:" + e, args.test_pct)}

    # ── Templates from real channel titles (training artists only) ──────────
    tmpl_artists: dict[str, set[str]] = defaultdict(set)
    tmpl_cased: dict[str, Counter] = defaultdict(Counter)
    real: list[tuple[str, str, str | None]] = []  # (entity, title, template key)
    dropped_lib = 0
    for e, ts in links.items():
        for t in ts:
            if norm_key(t) in artists[e].keys():
                continue
            if mentions_library_group(t):
                dropped_lib += 1
                continue
            tm = mask(t, artists[e])
            key = tm.casefold() if tm else None
            if tm and e not in test_artists:
                tmpl_artists[key].add(e)
                tmpl_cased[key][tm] += 1
            real.append((e, t, key))
    mined = {k: tmpl_cased[k].most_common(1)[0][0] for k, v in tmpl_artists.items()
             if len(v) >= args.min_template_artists}
    mined_weights = {k: math.sqrt(len(tmpl_artists[k])) for k in mined}
    print(f"real titles: {len(real)} (dropped {dropped_lib} naming a library group); "
          f"mined templates: {len(mined)}")

    by_template: dict[str | None, list[tuple[str, str]]] = defaultdict(list)
    by_tier_titles: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e, t, key in real:
        if e in test_artists:
            continue
        by_template[key].append((e, t))
        by_tier_titles[artists[e].tier].append((e, t))

    rows: list[dict] = []
    counts = Counter()

    def add(anchor: str, positive: str, negative: str, kind: str) -> None:
        if norm_key(anchor) == norm_key(positive) or norm_key(negative) in (norm_key(anchor), norm_key(positive)):
            return
        rows.append({"anchor": anchor, "positive": positive, "negative": negative, "kind": kind})
        counts[kind] += 1

    # ── Real rows ─────────────────────────────────────────────────────────────
    for e, t, key in real:
        if e in test_artists:
            continue
        pool = by_template.get(key) if key in mined else None
        pool = pool if pool and len(pool) > 1 else by_tier_titles[artists[e].tier]
        for _ in range(3):
            ne, neg = rng.choice(pool)
            if ne != e:
                break
        else:
            continue
        name = rng.choice(artists[e].names)
        add(t, name, neg, "real")
        add(rng.choice(artists[e].names), t, neg, "real_rev")

    # ── Synthetic rows ────────────────────────────────────────────────────────
    mined_keys = list(mined)
    mined_w = [mined_weights[k] for k in mined_keys]
    train_by_tier: dict[str, list[Artist]] = defaultdict(list)
    for e, a in artists.items():
        if e not in test_artists:
            train_by_tier[a.tier].append(a)

    def pick_template(a: Artist, r: random.Random) -> str:
        if a.cjk() and a.latin() and r.random() < 0.5:
            return r.choice(VT_TEMPLATES)
        if r.random() < 0.25:
            return r.choice([t for t in VT_TEMPLATES if "{r}" not in t and "{n}" not in t])
        return mined[r.choices(mined_keys, mined_w)[0]]

    for tier, pool in train_by_tier.items():
        for a in pool:
            if len(a.names) < 2 and rng.random() < 0.5:
                continue
            if tier == "en" and rng.random() > args.en_synth_rate:
                continue
            for _ in range(args.synth_per_artist):
                template = pick_template(a, rng)
                g = rng.choice(groups_train)
                got = fill(template, a, g, rng)
                if got is None:
                    continue
                text, used = got
                neg = None
                for _ in range(5):
                    b = rng.choice(pool)
                    if b is not a and (nb := fill(template, b, g, rng)):
                        neg = nb[0]
                        break
                if neg is None:
                    continue
                kind = "vt" if template in VT_TEMPLATES else "mined"
                pos = positive_for(a, used, rng)
                if rng.random() < 0.5:
                    add(text, pos, neg, kind)
                else:
                    add(pos, text, neg, kind + "_rev")

    rng.shuffle(rows)
    with (base / "artist_wrap_triplets.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("anchor", "positive", "negative")},
                               ensure_ascii=False) + "\n")

    # ── Held-out tests ────────────────────────────────────────────────────────
    test_pool = [artists[e] for e in sorted(test_artists) if len(artists[e].names) >= 2]
    trng = random.Random(args.seed + 1)
    wrap_test = []
    for a in test_pool:
        template = pick_template(a, trng)
        g = trng.choice(groups_test)
        got = fill(template, a, g, trng)
        if got is None:
            continue
        text, used = got
        pos = positive_for(a, used, trng)
        if norm_key(pos) == norm_key(text):
            continue
        corpus = [pos]
        for b in trng.sample(test_pool, min(len(test_pool), args.test_corpus * 2)):
            if b is a or len(corpus) > 2 * args.test_corpus:
                continue
            if nb := fill(template, b, g, trng):
                corpus += [nb[0], trng.choice(b.names)]
        if len(corpus) > 1 and text not in corpus:
            wrap_test.append({"query": text, "positive": pos, "corpus": list(dict.fromkeys(corpus)),
                              "template": template})
    with (base / "artist_wrap_test.jsonl").open("w", encoding="utf-8") as f:
        for r in wrap_test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    chan_test = [{"anchor": t, "positives": artists[e].names, "entity": e}
                 for e, t, _ in real if e in test_artists]
    with (base / "channel_test.jsonl").open("w", encoding="utf-8") as f:
        for r in chan_test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    card = {
        "rows": len(rows), "by_kind": dict(counts), "mined_templates": len(mined),
        "top_templates": [mined[k] for k, _ in sorted(mined_weights.items(), key=lambda x: -x[1])[:30]],
        "vt_templates": VT_TEMPLATES, "groups_train": len(groups_train), "groups_test": len(groups_test),
        "test_artists": len(test_artists), "artist_wrap_test": len(wrap_test),
        "channel_test": len(chan_test), "real_titles_dropped_library_group": dropped_lib,
        "library_affiliations": sorted(lib_affil), "args": {k: str(v) for k, v in vars(args).items()},
    }
    (base / "artist_wrap_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in card.items() if k not in ("library_affiliations", "vt_templates")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
