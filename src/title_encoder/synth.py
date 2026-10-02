"""Synthetic + real noisy video-title triplets.

Synthetic: wrap clean MB song titles in mined video-title templates. The hard
negative is the *same template* around a *different song by the same artist*,
which is exactly the `[Official MV] A` vs `[Official MV] B` confusion: the
only signal left is the core title.

Real: (recording name, its linked YouTube video title), with another video from
the same channel (different recording) as the hard negative — real channel
templates. Capped per channel so one uploader can't dominate.

Held-out split follows pairs.py (artist credit hash), so a test credit never
trains in any phase. Real pairs from held-out credits go to video_test_pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .pairs import NOISY
from .paths import build_dir
from .text import allowed, norm_key



def held_out(key: str, test_mod: int) -> bool:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % test_mod == 0


def load_bank(path: Path, max_literal: int) -> dict[str, tuple[list[str], list[float]]]:
    bank: dict[str, tuple[list[str], list[float]]] = {"jp": ([], []), "en": ([], [])}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = r["template"]
            literal = t.replace("{t}", "").replace("{a}", "")
            if "{t}" not in t or t == "{t}" or t.count("{t}") > 1 or t.count("{a}") > 1:
                continue
            if "0" in literal or len(literal.strip()) > max_literal or not allowed(t):
                continue
            bank[r["tier"]][0].append(t)
            bank[r["tier"]][1].append(float(r["channels"]))
    return bank


def fill(template: str, title: str, artist: str) -> str:
    return re.sub(r"\s+", " ", template.replace("{t}", title).replace("{a}", artist)).strip()


def related(a: str, b: str) -> bool:
    ka, kb = norm_key(a), norm_key(b)
    return ka == kb or ka in kb or kb in ka


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--test-mod", type=int, default=50)
    parser.add_argument("--en-ratio", type=float, default=0.25)
    parser.add_argument("--jp-en-template-share", type=float, default=0.3,
                        help="share of JP songs wrapped in an EN-tier template")
    parser.add_argument("--video-anchor-share", type=float, default=0.3,
                        help="share of synthetic rows whose anchor is itself wrapped")
    parser.add_argument("--max-literal", type=int, default=40)
    parser.add_argument("--max-per-channel", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = build_dir(args.build)
    rng = random.Random(args.seed)
    bank = load_bank(base / "templates.jsonl", args.max_literal)
    print("template bank:", {k: len(v[0]) for k, v in bank.items()})

    def pick(tier: str) -> str:
        use = "en" if tier == "en" or rng.random() < args.jp_en_template_share else "jp"
        templates, weights = bank[use]
        return rng.choices(templates, weights)[0]

    songs: list[dict] = []
    by_credit: dict[str, list[str]] = defaultdict(list)
    with (base / "raw" / "songs.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name, credit = row["name"].strip(), row["credit"].strip()
            if (not allowed(name) or not allowed(credit) or NOISY.search(name)
                    or held_out(f"ac:{row['artist_credit']}", args.test_mod)):
                continue
            songs.append(row)
            by_credit[row["artist_credit"]].append(name)

    synth = {"jp": [], "en": []}
    for row in songs:
        name, credit, tier = row["name"].strip(), row["credit"].strip(), row["tier"]
        others = [n for n in by_credit[row["artist_credit"]] if not related(n, name)]
        if not others:
            continue
        tpl = pick(tier)
        anchor = name
        if rng.random() < args.video_anchor_share:
            anchor_tpl = pick(tier)
            if anchor_tpl != tpl:
                anchor = fill(anchor_tpl, name, credit)
        synth[tier].append({
            "anchor": anchor,
            "positive": fill(tpl, name, credit),
            "negative": fill(tpl, rng.choice(others), credit),
            "kind": "synth", "tier": tier,
        })
    rng.shuffle(synth["en"])
    synth_rows = synth["jp"] + synth["en"][: int(len(synth["jp"]) * args.en_ratio)]
    rng.shuffle(synth_rows)

    # Real video pairs: negatives from the same channel.
    real = []
    with (base / "video_pairs.jsonl").open(encoding="utf-8") as f:
        real = [json.loads(line) for line in f]
    by_channel: dict[str, list[dict]] = defaultdict(list)
    for r in real:
        by_channel[r["channel_id"]].append(r)
    train_real, test_real = [], []
    for channel, rows in by_channel.items():
        rng.shuffle(rows)
        for r in rows[: args.max_per_channel]:
            if held_out(f"credit:{r['credit']}", args.test_mod):
                test_real.append({"anchor": r["anchor"], "positive": r["positive"],
                                  "tier": r["tier"], "group": f"video:{r['recording']}"})
                continue
            out = {"anchor": r["anchor"], "positive": r["positive"], "kind": "video",
                   "tier": r["tier"]}
            sibs = [s for s in rows if s["recording"] != r["recording"]
                    and not related(s["anchor"], r["anchor"])
                    and not held_out(f"credit:{s['credit']}", args.test_mod)]
            if sibs:
                out["negative"] = rng.choice(sibs)["positive"]
            train_real.append(out)

    # Template-confusion test (held-out credits only): every corpus entry of a
    # credit is the SAME template around a different song by that artist, so
    # only the core title can pick the right one.
    held: dict[str, list[dict]] = defaultdict(list)
    with (base / "raw" / "songs.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (held_out(f"ac:{row['artist_credit']}", args.test_mod) and allowed(row["name"])
                    and allowed(row["credit"]) and not NOISY.search(row["name"])):
                held[row["artist_credit"]].append(row)
    confusion = []
    for credit_id, rows in held.items():
        names: list[str] = []
        for row in rows:
            if not any(related(row["name"], n) for n in names):
                names.append(row["name"].strip())
        if len(names) < 3:
            continue
        rng.shuffle(names)
        names = names[:10]
        tier, credit = rows[0]["tier"], rows[0]["credit"].strip()
        tpl = pick(tier)
        corpus = [fill(tpl, n, credit) for n in names]
        query = names[0]
        if rng.random() < 0.5:
            other = pick(tier)
            if other != tpl:
                query = fill(other, names[0], credit)
        confusion.append({"query": query, "positive": corpus[0], "corpus": corpus,
                          "tier": tier, "template": tpl})
    rng.shuffle(confusion)
    confusion = confusion[:2000]

    outputs = {
        "synth_triplets": synth_rows,
        "video_triplets": [r for r in train_real if "negative" in r],
        "video_pairs_train": [r for r in train_real if "negative" not in r],
        "video_test_pairs": test_real,
        "template_test": confusion,
    }
    for name, rows in outputs.items():
        with (base / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    card = {k: len(v) for k, v in outputs.items()}
    card["synth_jp"] = len(synth["jp"])
    card["synth_en_available"] = len(synth["en"])
    (base / "synth_card.json").write_text(json.dumps({"counts": card, "args": vars(args)}, indent=2) + "\n")
    print(json.dumps(card, indent=2))
    for r in rng.sample(synth_rows, 12):
        print(f"  {r['tier']}  {r['anchor'][:38]!r:42s} + {r['positive'][:50]!r:54s} - {r['negative'][:50]!r}")


if __name__ == "__main__":
    main()
