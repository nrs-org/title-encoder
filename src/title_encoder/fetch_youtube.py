"""Fetch YouTube titles for the MB-linked videos in raw/video_links.csv.

videos.list(part=snippet) takes 50 ids per call at 1 quota unit. Resumable:
already-fetched ids in youtube_videos.jsonl are skipped. Unavailable/private
videos are recorded with "missing": true so they aren't retried.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .paths import PROJECT_ROOT, build_dir

ID = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def api_key() -> str:
    if key := os.environ.get("YOUTUBE_API_KEY"):
        return key
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        raise SystemExit("YOUTUBE_API_KEY not set and .env does not exist")
    for line in env_file.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip().removeprefix("export ").strip() == "YOUTUBE_API_KEY":
            return value.strip().strip("'\"")
    raise SystemExit("YOUTUBE_API_KEY not set and not in .env")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--max-calls", type=int, default=2000)
    args = parser.parse_args()

    base = build_dir(args.build)
    out = base / "youtube_videos.jsonl"
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            done = {json.loads(line)["video_id"] for line in f}

    ids: list[str] = []
    with (base / "raw" / "video_links.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if m := ID.search(row["url"]):
                vid = m.group(1)
                if vid not in done:
                    done.add(vid)
                    ids.append(vid)
    print(f"to fetch: {len(ids)} videos, {-(-len(ids) // 50)} calls")

    key = api_key()
    calls = 0
    with out.open("a", encoding="utf-8") as f:
        for start in range(0, len(ids), 50):
            if calls >= args.max_calls:
                break
            batch = ids[start : start + 50]
            url = "https://www.googleapis.com/youtube/v3/videos?" + urllib.parse.urlencode(
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
                    "video_id": item["id"],
                    "title": sn.get("title", ""),
                    "channel_id": sn.get("channelId", ""),
                    "channel_title": sn.get("channelTitle", ""),
                    "default_audio_language": sn.get("defaultAudioLanguage"),
                }
            for vid in batch:
                f.write(json.dumps(found.get(vid, {"video_id": vid, "missing": True}),
                                   ensure_ascii=False) + "\n")
            if calls % 50 == 0:
                f.flush()
                print(f"{calls} calls, {start + len(batch)}/{len(ids)}")
    print(f"done: {calls} calls")


if __name__ == "__main__":
    main()
