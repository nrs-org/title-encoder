"""CPU embedding cost per title (the ship bar is <= 1 ms/title, 10 ms acceptable).

Embeds a fixed sample of real library titles on CPU, length-sorted and
batched (the one free win from the LaBSE speed investigation), at 1 thread and
at all cores. PyTorch CPU here; ONNX Runtime numbers come with the export step.
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


def library_titles(db: Path, n: int, seed: int) -> list[str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    names = [r[0] for r in con.execute("select distinct name from entry_alias")]
    con.close()
    random.Random(seed).shuffle(names)
    return names[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--library-db", type=Path,
                        default=Path.home() / ".local/share/musiclib-rs/musiclib.db")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", default=f"1,{os.cpu_count()}")
    args = parser.parse_args()

    texts = sorted(library_titles(args.library_db, args.n, 0), key=len)
    for name in args.models:
        model = SentenceTransformer(name, device="cpu")
        model.max_seq_length = 48
        params = sum(p.numel() for p in model.parameters())
        emb = model[0].auto_model.get_input_embeddings().weight.numel()
        for threads in (int(t) for t in args.threads.split(",")):
            torch.set_num_threads(threads)
            model.encode(texts[:128], batch_size=args.batch_size)
            start = time.perf_counter()
            model.encode(texts, batch_size=args.batch_size)
            ms = (time.perf_counter() - start) * 1000 / len(texts)
            print(f"{name}: {threads:2d} threads  {ms:6.2f} ms/title  "
                  f"(params {params / 1e6:.0f}M, non-embedding {(params - emb) / 1e6:.1f}M)")


if __name__ == "__main__":
    main()
