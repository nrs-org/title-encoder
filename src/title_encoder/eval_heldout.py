"""Run every held-out evaluator from train.py on finished models.

MB pairs, real video titles, template confusion, cover confusion, artist
wrappers (artist_wrap.py) and real channel titles. Prints @1 / @20 and
optionally writes a JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from .paths import build_dir
from .train import (artist_wrap_evaluator, channel_evaluator, mb_evaluator, template_evaluator,
                    video_evaluator)

KEYS = ("accuracy@1", "accuracy@10", "accuracy@20")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    base = build_dir(args.build)
    evaluators = [e for e in (mb_evaluator(base, 3000, 0), video_evaluator(base),
                              template_evaluator(base), template_evaluator(base, "cover"),
                              artist_wrap_evaluator(base), channel_evaluator(base))
                  if e is not None]
    report = {}
    for path in args.models:
        model = SentenceTransformer(path, device="cuda")
        model.max_seq_length = 48
        scores = {}
        for ev in evaluators:
            scores.update(ev(model))
        row = {k.replace("_cosine", ""): round(v, 4) for k, v in scores.items()
               if any(k.endswith(x) for x in KEYS)}
        report[path] = row
        print(f"\n== {path}")
        for k, v in row.items():
            print(f"  {k:36s} {v:.4f}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
