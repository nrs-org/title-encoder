"""Summarize eval_library reports (data/title-encoder/eval/*.json) as one table.

One row per (report, model): pool sizes, then recall@1 / recall@20 for all
queries, cross-script, tracks and artists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .paths import DATA_ROOT
COLUMNS = (("*/all", "all"), ("*/cross_script", "cross"), ("track/all", "tracks"),
           ("artist/all", "artists"))


def short(model: str) -> str:
    parts = model.rstrip("/").split("/")
    if parts[-1].startswith(("final", "pruned")):
        return "/".join(parts[-2:])
    return parts[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DATA_ROOT / "eval")
    args = parser.parse_args()
    header = "| report | model | track pool | " + " | ".join(f"{n} r@1/r@20" for _, n in COLUMNS) + " |"
    print(header)
    print("|" + "---|" * (3 + len(COLUMNS)))
    for path in sorted(args.dir.glob("*.json")):
        report = json.loads(path.read_text())
        pool = report.get("pool_sizes", {}).get("track", "?")
        for model, result in report["models"].items():
            m = result["metrics"]
            cells = [f"{m[k]['r@1']:.3f}/{m[k]['r@20']:.3f}" if k in m else "—" for k, _ in COLUMNS]
            print(f"| {path.stem} | {short(model)} | {pool} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
