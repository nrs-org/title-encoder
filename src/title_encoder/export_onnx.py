"""Export a sentence-transformers title encoder to ONNX and benchmark it on CPU.

The graph is transformer -> mean pooling over the attention mask -> L2
normalize, i.e. exactly what SentenceTransformer.encode returns, with dynamic
batch and sequence axes. Verifies parity against PyTorch, then times
onnxruntime on CPU (tokenization included, length-sorted batches) at 1 thread
and at all cores. Optional Matryoshka truncation is applied after the graph.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sentence_transformers import SentenceTransformer

from .paths import build_dir


class Encoder(torch.nn.Module):
    def __init__(self, bert: torch.nn.Module) -> None:
        super().__init__()
        self.bert = bert

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)


def library_titles(db: Path, n: int) -> list[str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    names = [r[0] for r in con.execute("select distinct name from entry_alias")]
    con.close()
    random.Random(0).shuffle(names)
    return names[:n]


def corpus_titles(build: str, n: int) -> list[str]:
    """Deterministic parity/benchmark sample when the private library is absent."""
    names: dict[str, None] = {}
    for path in sorted(build_dir(build).glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for col in ("anchor", "positive", "negative", "text"):
                    if isinstance(row.get(col), str):
                        names.setdefault(row[col], None)
                        if len(names) >= n:
                            return list(names)
    return list(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, help="default: <model>/model.onnx")
    parser.add_argument("--library-db", type=Path,
                        default=Path.home() / ".local/share/musiclib-rs/musiclib.db")
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=48)
    args = parser.parse_args()

    st = SentenceTransformer(args.model, device="cpu")
    tok = st.tokenizer
    enc = Encoder(st[0].auto_model).eval()
    out = args.out or Path(args.model) / "model.onnx"
    sample = tok(["hello world", "テスト"], padding=True, return_tensors="pt")
    torch.onnx.export(
        enc, (sample["input_ids"], sample["attention_mask"]), str(out),
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"}, "embedding": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"exported {out} ({out.stat().st_size / 1e6:.0f} MB)")

    texts = (library_titles(args.library_db, args.n) if args.library_db.exists()
             else corpus_titles(args.build, args.n))
    if not texts:
        raise SystemExit("no validation texts found; provide --library-db or a populated --build")
    texts = sorted(texts, key=len)
    ref = st.encode(texts[:256], normalize_embeddings=True)

    def run(sess: ort.InferenceSession, batch: list[str]) -> np.ndarray:
        t = tok(batch, padding=True, truncation=True, max_length=args.max_seq_length,
                return_tensors="np")
        return sess.run(None, {"input_ids": t["input_ids"].astype(np.int64),
                               "attention_mask": t["attention_mask"].astype(np.int64)})[0]

    for threads in (1, os.cpu_count()):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(out), opts, providers=["CPUExecutionProvider"])
        got = np.concatenate([run(sess, texts[i : i + args.batch_size])
                              for i in range(0, 256, args.batch_size)])
        cos = (got * ref).sum(1).min()
        run(sess, texts[:128])
        start = time.perf_counter()
        for i in range(0, len(texts), args.batch_size):
            run(sess, texts[i : i + args.batch_size])
        ms = (time.perf_counter() - start) * 1000 / len(texts)
        print(f"onnxruntime {threads:2d} threads: {ms:.3f} ms/title  (min cosine vs torch {cos:.6f})")


if __name__ == "__main__":
    main()
