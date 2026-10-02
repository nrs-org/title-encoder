"""Fetch YouTube channel titles for MB artist -> channel links (v7 artist wrappers).

Input: raw/artist_channels.csv (sql/export_channels.sql), filtered to artists
in the leakage-excluded raw/artist_names.csv. channels.list(part=snippet)
takes 50 ids per call at 1 quota unit. Resumable: fetched ids in
youtube_channels.jsonl are skipped; missing channels are recorded too.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .fetch_youtube import api_key
from .paths import build_dir

CHANNEL = re.compile(r"/channel/(UC[A-Za-z0-9_-]{22})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--max-calls", type=int, default=2000)
    args = parser.parse_args()

    base = build_dir(args.build)
    out = base / "youtube_channels.jsonl"
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            done = {json.loads(line)["channel_id"] for line in f}

    with (base / "raw" / "artist_names.csv").open(encoding="utf-8", newline="") as f:
        artists = {row["entity"] for row in csv.DictReader(f)}
    ids: list[str] = []
    with (base / "raw" / "artist_channels.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["artist"] in artists and (m := CHANNEL.search(row["url"])):
                if m.group(1) not in done:
                    done.add(m.group(1))
                    ids.append(m.group(1))
    print(f"to fetch: {len(ids)} channels, {-(-len(ids) // 50)} calls")

    key = api_key()
    calls = 0
    with out.open("a", encoding="utf-8") as f:
        for start in range(0, len(ids), 50):
            if calls >= args.max_calls:
                break
            batch = ids[start : start + 50]
            url = "https://www.googleapis.com/youtube/v3/channels?" + urllib.parse.urlencode(
                {"part": "snippet", "id": ",".join(batch), "maxResults": 50, "key": key}
            )
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=30) as resp:
                        data = json.load(resp)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 403:
                        raise SystemExit(f"quota/forbidden after {calls} calls: {e.read()[:300]!r}")
                    time.sleep(2 ** attempt)
            else:
                raise SystemExit("repeated failures")
            calls += 1
            found = {}
            for item in data.get("items", []):
                sn = item["snippet"]
                found[item["id"]] = {
                    "channel_id": item["id"],
                    "title": sn.get("title", ""),
                    "custom_url": sn.get("customUrl"),
                    "country": sn.get("country"),
                }
            for cid in batch:
                f.write(json.dumps(found.get(cid, {"channel_id": cid, "missing": True}),
                                   ensure_ascii=False) + "\n")
            if calls % 100 == 0:
                f.flush()
                print(f"{calls} calls, {start + len(batch)}/{len(ids)}")
    print(f"done: {calls} calls")


if __name__ == "__main__":
    main()
