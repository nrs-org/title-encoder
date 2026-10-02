"""Turn exported MB name groups into contrastive training pairs.

Group kinds (all from build_mb.py's raw CSVs):
  song      recordings of one work by one artist credit (audio + video + versions)
  recording a recording's name + aliases
  work      a work's name + aliases (translations / transliterations)
  artist    an artist's name + aliases + credited-as names
  rg        a release group's name + aliases + its releases' names

Every pair of distinct names inside a group is a positive (name <-> alias and
alias <-> alias alike), capped per group. Song groups additionally get a hard
negative: a name from another song group with the same artist credit. EN-tier
pairs are downsampled to --en-ratio of the JP-tier count. A deterministic ~2%
of groups (by hash) is held out as test; its names double as eval distractors.
"""

from __future__ import annotations

import argparse
import csv
import re
import hashlib
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .paths import build_dir
from .text import allowed, cross_script, norm_key

csv.field_size_limit(sys.maxsize)

# Classical movements/catalogue numbers and medleys: MB links several distinct
# pieces under one work or alias group there, so their "positives" are wrong.
NOISY = re.compile(
    r"\b(?:op\.|opus|bwv|hob\.|k\.\s?\d|kv\s?\d|rv\s?\d|d\.\s?\d)|sonat|concert|symphon"
    r"|movement|medley|mix tape|楽章|交響曲|協奏曲|ソナタ|メドレー",
    re.IGNORECASE,
)

ENTRY_TYPE = {"song": "track", "recording": "track", "work": "track",
              "artist": "artist", "rg": "release_group"}


@dataclass
class Group:
    kind: str
    gid: str
    tier: str
    names: list[str]
    artist_credit: str | None = None


def keep_locale(locale: str | None) -> bool:
    return not locale or locale.startswith(("ja", "en"))


def distinct(names: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for name in names:
        if allowed(name):
            seen.setdefault(norm_key(name), name.strip())
    return list(seen.values())


def read_alias_groups(path: Path, kind: str) -> list[Group]:
    groups: dict[str, Group] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not keep_locale(row["locale"]):
                continue
            g = groups.setdefault(row["entity"], Group(kind, row["entity"], row["tier"], []))
            g.names.append(row["name"])
    return [g for g in groups.values() if len(g.names) > 1]


def read_song_groups(path: Path) -> list[Group]:
    groups: dict[tuple[str, str], Group] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["work"], row["artist_credit"])
            g = groups.setdefault(key, Group("song", f"{key[0]}:{key[1]}", row["tier"], [],
                                             row["artist_credit"]))
            g.names.append(row["name"])
    return list(groups.values())


def is_test(group: Group, test_mod: int) -> bool:
    # Song groups hold out by artist credit so a held-out credit never trains.
    key = f"ac:{group.artist_credit}" if group.artist_credit else f"{group.kind}:{group.gid}"
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % test_mod == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--max-pairs-per-group", type=int, default=6)
    parser.add_argument("--en-ratio", type=float, default=0.25,
                        help="EN pairs kept per JP pair (0.25 -> EN is 20%% of the mix)")
    parser.add_argument("--test-mod", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = build_dir(args.build)
    raw = base / "raw"
    rng = random.Random(args.seed)

    groups = read_song_groups(raw / "song_recordings.csv")
    for file, kind in (("recording_names.csv", "recording"), ("work_names.csv", "work"),
                       ("artist_names.csv", "artist"), ("release_group_names.csv", "rg")):
        groups += read_alias_groups(raw / file, kind)
    for g in groups:
        g.names = distinct(g.names)
    groups = [g for g in groups if len(g.names) > 1]

    by_credit: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        if g.kind == "song" and not is_test(g, args.test_mod):
            by_credit[g.artist_credit].append(i)

    train: dict[str, list[dict]] = {"jp": [], "en": []}
    test: list[dict] = []
    distractors: list[dict] = []
    stats: Counter = Counter()
    for gi, g in enumerate(groups):
        pairs = list(itertools.combinations(g.names, 2))
        rng.shuffle(pairs)
        pairs = pairs[: args.max_pairs_per_group]
        held_out = is_test(g, args.test_mod)
        if held_out:
            distractors += [{"entry_type": ENTRY_TYPE[g.kind], "text": n} for n in g.names]
        for a, b in pairs:
            if NOISY.search(a) or NOISY.search(b):
                stats["dropped_noisy"] += 1
                continue
            if rng.random() < 0.5:
                a, b = b, a
            row = {"anchor": a, "positive": b, "kind": g.kind, "tier": g.tier,
                   "cross_script": cross_script(a, b), "group": f"{g.kind}:{g.gid}"}
            if held_out:
                test.append(row)
                continue
            if g.kind == "song":
                siblings = [j for j in by_credit[g.artist_credit] if j != gi]
                if siblings:
                    other = groups[rng.choice(siblings)]
                    banned = {norm_key(a), norm_key(b)}
                    options = [n for n in other.names if norm_key(n) not in banned]
                    if options:
                        row["negative"] = rng.choice(options)
            train[g.tier].append(row)
            stats[(g.tier, g.kind, "cross" if row["cross_script"] else "same")] += 1

    jp = train["jp"]
    en = train["en"]
    rng.shuffle(en)
    en = en[: int(len(jp) * args.en_ratio)]
    rows = jp + en
    rng.shuffle(rows)

    with_neg = [r for r in rows if "negative" in r]
    without_neg = [r for r in rows if "negative" not in r]
    for name, data in (("train_pairs", without_neg), ("train_triplets", with_neg),
                       ("test_pairs", test), ("distractors", distractors)):
        with (base / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    card = {
        "build": args.build,
        "groups": len(groups),
        "train_rows": len(rows),
        "train_pairs": len(without_neg),
        "train_triplets": len(with_neg),
        "jp_rows": len(jp),
        "en_rows_kept": len(en),
        "en_rows_available": len(train["en"]),
        "test_pairs": len(test),
        "distractors": len(distractors),
        "dropped_noisy": stats.pop("dropped_noisy", 0),
        "by_tier_kind_script": {"/".join(k): v for k, v in sorted(stats.items())},
        "args": vars(args),
    }
    (base / "pairs_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
