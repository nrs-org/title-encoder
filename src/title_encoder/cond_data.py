"""Artist-conditioned triplets: input is `title [A] primary artist`.

Experiment for the cover-flood problem (docs/plan-title-encoder.md): a plain
title encoder puts every cover of a song next to each other, so a hit song's
covers can fill the top-k. Conditioned rows teach that the same title by a
*different* artist is a different entry:

  anchor    clean title [A] artist          (artist slot: credit name or an
  positive  video-wrapped / other name       alias of the primary artist, so
            of the same song, same artist    kanji/romaji variants still match)
  negative  the same work by an artist credit with NO artist in common (a
            cover), else another song by the same credit

Train-split credits only. Also writes cover_test.jsonl from held-out credits:
the query must find its own artist's version among every credit's version of
that work.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from .pairs import NOISY
from .paths import build_dir
from .synth import fill, held_out, load_bank, related
from .text import allowed, norm_key

SEP = " [A] "


def cond(title: str, artist: str) -> str:
    return f"{title}{SEP}{artist}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--test-mod", type=int, default=50)
    parser.add_argument("--en-ratio", type=float, default=0.25)
    parser.add_argument("--alias-share", type=float, default=0.4,
                        help="share of artist slots filled with an artist alias, not the credit")
    parser.add_argument("--video-share", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    csv.field_size_limit(sys.maxsize)
    base = build_dir(args.build)
    raw = base / "raw"
    rng = random.Random(args.seed)
    bank = load_bank(base / "templates.jsonl", 40)

    credit_artists: dict[str, list[str]] = defaultdict(list)
    with (raw / "credit_artists.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            credit_artists[row["artist_credit"]].append(row["artist"])
    artist_names: dict[str, list[str]] = defaultdict(list)
    with (raw / "artist_names.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if allowed(row["name"]) and (not row["locale"] or row["locale"].startswith(("ja", "en"))):
                artist_names[row["entity"]].append(row["name"])

    def artist_slot(credit_id: str, credit: str) -> str:
        artists = credit_artists.get(credit_id) or []
        if artists and rng.random() < args.alias_share:
            names = artist_names.get(artists[0])
            if names:
                return rng.choice(names)
        return credit

    def pick_template(tier: str) -> str:
        use = "en" if tier == "en" or rng.random() < 0.3 else "jp"
        templates, weights = bank[use]
        return rng.choices(templates, weights)[0]

    songs = []
    by_work: dict[str, list[dict]] = defaultdict(list)
    by_credit: dict[str, list[str]] = defaultdict(list)
    with (raw / "songs.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["name"], row["credit"] = row["name"].strip(), row["credit"].strip()
            if not allowed(row["name"]) or not allowed(row["credit"]) or NOISY.search(row["name"]):
                continue
            songs.append(row)
            by_work[row["work"]].append(row)
            if not held_out(f"ac:{row['artist_credit']}", args.test_mod):
                by_credit[row["artist_credit"]].append(row["name"])

    def disjoint(a: dict, b: dict) -> bool:
        return (a["artist_credit"] != b["artist_credit"]
                and not set(credit_artists.get(a["artist_credit"], []))
                & set(credit_artists.get(b["artist_credit"], [])))

    rows = {"jp": [], "en": []}
    covers_used = 0
    for row in songs:
        if held_out(f"ac:{row['artist_credit']}", args.test_mod):
            continue
        tier, name, credit, cid = row["tier"], row["name"], row["credit"], row["artist_credit"]
        if rng.random() < args.video_share:
            positive = cond(fill(pick_template(tier), name, credit), artist_slot(cid, credit))
        else:
            positive = cond(name, artist_slot(cid, credit))
        anchor = cond(name, artist_slot(cid, credit))
        if norm_key(anchor) == norm_key(positive):
            continue
        covers = [c for c in by_work[row["work"]] if disjoint(row, c)
                  and not held_out(f"ac:{c['artist_credit']}", args.test_mod)]
        if covers:
            other = rng.choice(covers)
            negative = cond(other["name"], artist_slot(other["artist_credit"], other["credit"]))
            covers_used += 1
        else:
            others = [n for n in by_credit[cid] if not related(n, name)]
            if not others:
                continue
            negative = cond(rng.choice(others), artist_slot(cid, credit))
        rows[tier].append({"anchor": anchor, "positive": positive, "negative": negative,
                           "kind": "cond", "tier": tier})
    rng.shuffle(rows["en"])
    out_rows = rows["jp"] + rows["en"][: int(len(rows["jp"]) * args.en_ratio)]
    rng.shuffle(out_rows)

    # Cover test: a held-out credit's video-wrapped version must retrieve its
    # own clean version among every credit's version of the same work.
    tests = []
    for work, versions in by_work.items():
        held = [v for v in versions if held_out(f"ac:{v['artist_credit']}", args.test_mod)]
        if not held or len({v["artist_credit"] for v in versions}) < 3:
            continue
        h = rng.choice(held)
        corpus_rows = [v for v in versions if v is h or disjoint(h, v)]
        if len(corpus_rows) < 3:
            continue
        corpus = [cond(v["name"], v["credit"]) for v in corpus_rows]
        query = cond(fill(pick_template(h["tier"]), h["name"], h["credit"]),
                     artist_slot(h["artist_credit"], h["credit"]))
        tests.append({"query": query, "positive": cond(h["name"], h["credit"]),
                      "corpus": corpus, "tier": h["tier"]})
    rng.shuffle(tests)
    tests = tests[:2000]

    for name, data in (("cond_triplets", out_rows), ("cover_test", tests)):
        with (base / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"cond triplets: {len(out_rows)} (jp {len(rows['jp'])}, cover negatives {covers_used}); "
          f"cover tests: {len(tests)}")
    for r in out_rows[:10]:
        print(f"  {r['anchor'][:40]!r:44s} + {r['positive'][:45]!r:49s} - {r['negative'][:40]!r}")


if __name__ == "__main__":
    main()
