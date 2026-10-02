"""Mine video-title templates from fetched YouTube titles of MB-linked videos.

For each video, locate the MB recording name and the artist credit (or one of
its member names) inside the video title and replace them with {t} / {a}.
Whatever remains is the channel's title template, e.g.
    "{a}「{t}」Official Music Video"   "【MV】{t} / {a}"   "{t} feat. {a} [MV]"
Templates are counted by distinct channels, so one prolific channel can't
dominate. Output: templates.jsonl (template, channels, videos, tier, example).
Also writes video_pairs.jsonl: real (recording name, video title) positives.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .paths import build_dir
from .text import allowed

ID = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")
CREDIT_SPLIT = re.compile(r"\s*(?:feat\.|ft\.|featuring|×|&|,|、|\bx\b|\bvs\.?|with)\s*", re.IGNORECASE)
DIGITS = re.compile(r"\d+")


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def find(hay: str, needle: str) -> re.Match | None:
    if len(needle) < 2:
        return None
    return re.search(re.escape(needle), hay, re.IGNORECASE)


def templatize(title: str, name: str, credit: str) -> str | None:
    title, name, credit = nfkc(title), nfkc(name), nfkc(credit)
    m = find(title, name)
    if not m:
        return None
    out = title[: m.start()] + "\x00T\x00" + title[m.end() :]
    artists = [credit] + [p for p in CREDIT_SPLIT.split(credit) if p and p != credit]
    for artist in sorted(set(artists), key=len, reverse=True):
        am = find(out, artist)
        if am and "\x00" not in am.group(0):
            out = out[: am.start()] + "\x00A\x00" + out[am.end() :]
            break
    # Normalize variable bits so near-identical templates merge.
    out = DIGITS.sub("0", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out.replace("\x00T\x00", "{t}").replace("\x00A\x00", "{a}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="mb-v1")
    args = parser.parse_args()
    base = build_dir(args.build)

    videos = {}
    with (base / "youtube_videos.jsonl").open(encoding="utf-8") as f:
        for line in f:
            v = json.loads(line)
            if not v.get("missing"):
                videos[v["video_id"]] = v

    channels: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    tiers: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    example: dict[str, str] = {}
    pairs = []
    matched = total = 0
    with (base / "raw" / "video_links.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m = ID.search(row["url"])
            v = videos.get(m.group(1)) if m else None
            if not v or not allowed(v["title"]) or not allowed(row["name"]):
                continue
            total += 1
            if nfkc(v["title"]).casefold() != nfkc(row["name"]).casefold():
                pairs.append({"anchor": row["name"], "positive": v["title"], "tier": row["tier"],
                              "credit": row["credit"], "channel_id": v["channel_id"],
                              "recording": row["recording"]})
            tpl = templatize(v["title"], row["name"], row["credit"])
            if tpl is None:
                continue
            matched += 1
            channels[tpl].add(v["channel_id"])
            counts[tpl] += 1
            tiers[tpl][row["tier"]] += 1
            example.setdefault(tpl, v["title"])

    rows = sorted(
        ({"template": t, "channels": len(channels[t]), "videos": counts[t],
          "tier": max(tiers[t], key=tiers[t].get), "example": example[t]} for t in counts),
        key=lambda r: (-r["channels"], -r["videos"]),
    )
    with (base / "templates.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (base / "video_pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"videos with title: {total}, templatized: {matched}, templates: {len(rows)}, "
          f"multi-channel templates: {sum(r['channels'] >= 3 for r in rows)}, "
          f"real noisy pairs: {len(pairs)}")
    for r in rows[:40]:
        print(f"{r['channels']:5d} ch {r['videos']:6d} v  {r['tier']}  {r['template']}")


if __name__ == "__main__":
    main()
