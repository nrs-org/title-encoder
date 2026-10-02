"""Marginal recall of a deterministic romaji-key channel over the encoder.

Prototype of plan phase 1's transliteration keys, measured before any Rust
is written. Every pool name gets a key: kana/kanji romanized (pykakasi,
Hepburn), lowercased, romanization variants folded (long vowels, tsu/tu,
shi/si, ...), tokens sorted so name order doesn't matter (`蛇石 徹` and
`Toru Hebiishi` share `hebiishi toru`). A query is retrieved by the channel
if a different pool item with the same key belongs to its entry; blocks
larger than --max-block are dropped, as the Rust blocker does.

Reports encoder r@20, romaji-channel recall, and their union, per stratum.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pykakasi

from .eval_library import add_distractors, load_library, make_embed_fn, search

FOLD = [("ou", "o"), ("oo", "o"), ("uu", "u"), ("aa", "a"), ("ee", "e"), ("ii", "i"),
        ("tsu", "tu"), ("shi", "si"), ("chi", "ti"), ("ji", "zi"), ("fu", "hu"),
        ("sha", "sya"), ("shu", "syu"), ("sho", "syo"), ("cha", "tya"), ("chu", "tyu"),
        ("cho", "tyo"), ("ja", "zya"), ("ju", "zyu"), ("jo", "zyo"), ("nn", "n")]
_KAKASI = pykakasi.kakasi()


def romaji_key(text: str) -> str:
    try:
        roman = " ".join(part["hepburn"] for part in _KAKASI.convert(text)).lower()
    except IndexError:  # pykakasi bug on some trailing CJK sequences
        roman = text.lower()
    roman = (roman.replace("ā", "a").replace("ī", "i").replace("ū", "u")
             .replace("ē", "e").replace("ō", "o"))
    tokens = []
    for tok in re.findall(r"[a-z0-9]+", roman):
        for a, b in FOLD:
            tok = tok.replace(a, b)
        tokens.append(tok)
    return " ".join(sorted(tokens))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="model spec, as in eval_library (path or path@cond)")
    parser.add_argument("--library-db", type=Path,
                        default=Path.home() / ".local/share/musiclib-rs/musiclib.db")
    parser.add_argument("--distractors", type=Path)
    parser.add_argument("--distractors-per-type", type=int, default=0)
    parser.add_argument("--artist", action="store_true")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max-block", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pools, queries = load_library(args.library_db, 3, 0, args.artist)
    add_distractors(pools, args.distractors, args.distractors_per_type, 0, args.artist)
    embed = make_embed_fn(args.model, args.device, 256)
    for etype in ("artist", "track"):
        pool = pools[etype]
        qs = [q for q in queries if q.entry_type == etype]
        keys = [romaji_key(t) for t in pool.texts]
        blocks: dict[str, list[int]] = defaultdict(list)
        for i, k in enumerate(keys):
            if len(k) >= 3:
                blocks[k].append(i)
        nn = search(embed(pool.texts, pool.artists), np.array([q.pool_idx for q in qs]),
                    args.k, args.device)
        totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for q, row in zip(qs, nn):
            res = [j for j in row if j != q.pool_idx][: args.k]
            enc = any(q.entry in pool.entries[j] for j in res)
            block = blocks.get(keys[q.pool_idx], [])
            rom = len(block) <= args.max_block and any(
                j != q.pool_idx and q.entry in pool.entries[j] for j in block)
            for stratum in q.strata:
                t = totals[stratum]
                t[0] += 1
                t[1] += enc
                t[2] += rom
                t[3] += enc or rom
        print(f"== {etype} (pool {len(pool.texts)})")
        for stratum, (n, enc, rom, both) in sorted(totals.items()):
            print(f"  {stratum:16s} n={n:5d}  encoder r@{args.k}={enc / n:.3f}  "
                  f"romaji={rom / n:.3f}  union={both / n:.3f}  (+{(both - enc) / n:.3f})")


if __name__ == "__main__":
    main()
