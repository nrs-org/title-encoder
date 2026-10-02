"""Distractor titles for the scale curve (eval_library --distractors).

Every distinct exported MB track title and artist name (all tiers); track
titles carry their credit as "artist" for eval_library --artist. They only
pad the retrieval pool; they are never queries, so using names that also
appear in training doesn't leak eval labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .paths import build_dir
from .text import allowed, norm_key

SOURCES = (("songs.csv", "track"), ("song_recordings.csv", "track"),
           ("recording_names.csv", "track"), ("artist_names.csv", "artist"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="mb-v1")
    args = parser.parse_args()
    csv.field_size_limit(sys.maxsize)
    base = build_dir(args.build)
    seen: dict[str, set[str]] = {"track": set(), "artist": set()}
    with (base / "scale_distractors.jsonl").open("w", encoding="utf-8") as out:
        for fname, etype in SOURCES:
            with (base / "raw" / fname).open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    text, key = row["name"], norm_key(row["name"])
                    if allowed(text) and key and key not in seen[etype]:
                        seen[etype].add(key)
                        row_out = {"entry_type": etype, "text": text}
                        if row.get("credit"):
                            row_out["artist"] = row["credit"]
                        out.write(json.dumps(row_out, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in seen.items()})


if __name__ == "__main__":
    main()
