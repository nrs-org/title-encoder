"""Cross-script artist-name triplets with shared-name-part hard negatives.

The remaining artist misses match one part of a name and confuse people who
share it (`Toru Hebiishi` -> `Toru Kitajima`). For each training-split JP
artist with both a Latin and a CJK name:
  Latin -> CJK : negative = CJK name of another artist sharing a Latin token
  CJK -> Latin : negative = Latin name of another artist sharing a 2-char
                 CJK prefix or suffix (surname / given-name guess)
so the model must read every part of the name, not the easy one.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .pairs import distinct, is_test, read_alias_groups
from .paths import build_dir
from .text import script_class

WORD = re.compile(r"[a-zA-ZÀ-ɏ]{3,}")


def latin_tokens(name: str) -> set[str]:
    return {w.lower() for w in WORD.findall(name)}


def cjk_keys(name: str) -> set[str]:
    compact = re.sub(r"\s+", "", name)
    return {compact[:2], compact[-2:]} if len(compact) >= 3 else set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--test-mod", type=int, default=50)
    parser.add_argument("--per-artist", type=int, default=2)
    parser.add_argument("--max-bucket", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = build_dir(args.build)
    rng = random.Random(args.seed)
    artists = []
    for g in read_alias_groups(base / "raw" / "artist_names.csv", "artist"):
        if g.tier != "jp" or is_test(g, args.test_mod):
            continue
        names = distinct(g.names)
        latin = [n for n in names if script_class(n) == "latin"]
        cjk = [n for n in names if script_class(n) in ("kana", "han")]
        if latin and cjk:
            artists.append((latin, cjk))
    by_token: dict[str, list[int]] = defaultdict(list)
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, (latin, cjk) in enumerate(artists):
        for t in {t for n in latin for t in latin_tokens(n)}:
            by_token[t].append(i)
        for k in {k for n in cjk for k in cjk_keys(n)}:
            by_key[k].append(i)

    rows = []
    for i, (latin, cjk) in enumerate(artists):
        for _ in range(args.per_artist):
            a, c = rng.choice(latin), rng.choice(cjk)
            if rng.random() < 0.5:
                pool = [j for t in latin_tokens(a) if len(by_token[t]) <= args.max_bucket
                        for j in by_token[t] if j != i]
                if pool:
                    rows.append({"anchor": a, "positive": c,
                                 "negative": rng.choice(artists[rng.choice(pool)][1])})
            else:
                pool = [j for k in cjk_keys(c) if len(by_key[k]) <= args.max_bucket
                        for j in by_key[k] if j != i]
                if pool:
                    rows.append({"anchor": c, "positive": a,
                                 "negative": rng.choice(artists[rng.choice(pool)][0])})
    rng.shuffle(rows)
    with (base / "artist_triplets.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"artists with latin+cjk names: {len(artists)}, triplets: {len(rows)}")
    for r in rows[:12]:
        print(f"  {r['anchor'][:25]!r:28s} + {r['positive'][:25]!r:28s} - {r['negative'][:25]!r}")


if __name__ == "__main__":
    main()
