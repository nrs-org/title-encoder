"""Model-mined hard negatives for artist names.

Embeds every training-split artist name (JP tier + sampled EN) with the
current model, and for each anchor takes its nearest neighbours that belong
to a *different* artist as negatives (skipping identical strings: two
artists with the same name can't be told apart by text). The positive is
another name of the anchor's own artist. These are exactly the confusions
the model makes today, so training on them targets the remaining errors.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .pairs import distinct, is_test, read_alias_groups
from .paths import build_dir
from .text import norm_key



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--out", default="mined_artist_triplets")
    parser.add_argument("--en-ratio", type=float, default=0.25)
    parser.add_argument("--neighbours", type=int, default=10)
    parser.add_argument("--per-artist", type=int, default=2)
    parser.add_argument("--test-mod", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = build_dir(args.build)
    rng = random.Random(args.seed)
    groups = [g for g in read_alias_groups(base / "raw" / "artist_names.csv", "artist")
              if not is_test(g, args.test_mod)]
    jp = [g for g in groups if g.tier == "jp"]
    en = [g for g in groups if g.tier == "en"]
    rng.shuffle(en)
    groups = jp + en[: int(len(jp) * args.en_ratio)]
    names, owner = [], []
    for gi, g in enumerate(groups):
        g.names = distinct(g.names)
        for n in g.names:
            names.append(n)
            owner.append(gi)
    owner_arr = np.array(owner)
    keys = [norm_key(n) for n in names]
    print(f"artists: {len(groups)}, names: {len(names)}")

    model = SentenceTransformer(args.model, device="cuda")
    model.max_seq_length = 48
    emb = torch.from_numpy(model.encode(names, batch_size=512, normalize_embeddings=True,
                                        convert_to_numpy=True)).cuda().half()

    multi = [gi for gi, g in enumerate(groups) if len(g.names) > 1]
    anchors = []
    starts: dict[int, list[int]] = {}
    for i, gi in enumerate(owner):
        starts.setdefault(gi, []).append(i)
    for gi in multi:
        picks = starts[gi][:]
        rng.shuffle(picks)
        anchors += [(i, gi) for i in picks[: args.per_artist]]

    rows = []
    chunk = 1024
    for s in range(0, len(anchors), chunk):
        batch = anchors[s : s + chunk]
        q = emb[torch.tensor([i for i, _ in batch]).cuda()]
        top = torch.topk(q @ emb.T, k=args.neighbours + 8, dim=1).indices.cpu().numpy()
        for (i, gi), row in zip(batch, top):
            wrong = [j for j in row if owner_arr[j] != gi and keys[j] != keys[i]]
            if not wrong:
                continue
            positives = [j for j in starts[gi] if j != i and keys[j] != keys[i]]
            if not positives:
                continue
            rows.append({"anchor": names[i], "positive": names[rng.choice(positives)],
                         "negative": names[rng.choice(wrong[: args.neighbours])]})
    rng.shuffle(rows)
    with (base / f"{args.out}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"mined triplets: {len(rows)}")
    for r in rows[:12]:
        print(f"  {r['anchor'][:28]!r:32s} + {r['positive'][:28]!r:32s} - {r['negative'][:28]!r}")


if __name__ == "__main__":
    main()
