"""Contrastive fine-tune of a title encoder on pairs.py output.

Loss: InfoNCE (CachedMultipleNegativesRankingLoss: GradCache, so the in-batch
negative pool is --batch-size while memory follows --mini-batch-size), plus the
explicit hard negative on triplet rows, wrapped in MatryoshkaLoss so the
embedding can be truncated at deploy time. Batches never repeat a text
(NO_DUPLICATES), which keeps identical strings from becoming false negatives.

Tracks MB held-out retrieval (test_pairs.jsonl against positives+distractors)
during training; the library eval (eval_library.py) is run on the result.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (InformationRetrievalEvaluator, SentenceEvaluator,
                                             SequentialEvaluator)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

from .paths import build_dir, model_dir
from .text import norm_key


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def mb_evaluator(base: Path, max_queries: int, seed: int) -> InformationRetrievalEvaluator:
    """One held-out pair per group: the anchor must retrieve its positive among
    all held-out names. Anchor texts are kept out of the corpus so a query
    never just finds itself."""
    rng = random.Random(seed)
    test = load_jsonl(base / "test_pairs.jsonl")
    rng.shuffle(test)
    seen_groups: set[str] = set()
    picked = []
    for row in test:
        if row["group"] not in seen_groups:
            seen_groups.add(row["group"])
            picked.append(row)
        if len(picked) >= max_queries:
            break
    anchor_keys = {norm_key(r["anchor"]) for r in picked}
    corpus: dict[str, str] = {}
    by_key: dict[str, str] = {}

    def cid(text: str) -> str:
        key = norm_key(text)
        if key not in by_key:
            by_key[key] = f"c{len(by_key)}"
            corpus[by_key[key]] = text
        return by_key[key]

    for row in load_jsonl(base / "distractors.jsonl"):
        if norm_key(row["text"]) not in anchor_keys:
            cid(row["text"])
    queries, relevant = {}, {}
    for i, row in enumerate(picked):
        if norm_key(row["positive"]) in anchor_keys:
            continue
        qid = f"q{i}"
        queries[qid] = row["anchor"]
        relevant[qid] = {cid(row["positive"])}
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name="mb-test", accuracy_at_k=[1, 10, 20],
        precision_recall_at_k=[10, 20], mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10],
        show_progress_bar=False, batch_size=256,
    )


def video_evaluator(base: Path) -> InformationRetrievalEvaluator | None:
    """Held-out real (recording name -> its YouTube title) pairs, among the
    held-out titles plus every other fetched video title as distractors."""
    path = base / "video_test_pairs.jsonl"
    if not path.exists():
        return None
    test = load_jsonl(path)
    corpus: dict[str, str] = {}
    by_key: dict[str, str] = {}

    def cid(text: str) -> str:
        key = norm_key(text)
        if key not in by_key:
            by_key[key] = f"c{len(by_key)}"
            corpus[by_key[key]] = text
        return by_key[key]

    queries, relevant = {}, {}
    for i, row in enumerate(test):
        queries[f"q{i}"] = row["anchor"]
        relevant[f"q{i}"] = {cid(row["positive"])}
    for row in load_jsonl(base / "youtube_videos.jsonl"):
        if not row.get("missing"):
            cid(row["title"])
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name="video-test", accuracy_at_k=[1, 10, 20],
        precision_recall_at_k=[10, 20], mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10],
        show_progress_bar=False, batch_size=256,
    )


def template_evaluator(base: Path, name: str = "template") -> InformationRetrievalEvaluator | None:
    """Per-query small corpora: template confusion (same template around
    different songs of one artist) or cover confusion (one work, many artists)."""
    path = base / f"{name}_test.jsonl"
    if not path.exists():
        return None
    corpus: dict[str, str] = {}
    by_text: dict[str, str] = {}
    queries, relevant = {}, {}
    for i, row in enumerate(load_jsonl(path)):
        for text in row["corpus"]:
            if text not in by_text:
                by_text[text] = f"c{len(by_text)}"
                corpus[by_text[text]] = text
        queries[f"q{i}"] = row["query"]
        relevant[f"q{i}"] = {by_text[row["positive"]]}
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name=f"{name}-test", accuracy_at_k=[1, 5],
        precision_recall_at_k=[5], mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10],
        show_progress_bar=False, batch_size=256,
    )


class PerQueryEvaluator(SentenceEvaluator):
    """accuracy@k where each query is ranked against *its own* corpus only.

    The InformationRetrievalEvaluator merges every row's corpus into one
    pool; for artist_wrap_test that pool holds the query's own artist
    wrapped in other rows' templates, unmarked, so a model that recognizes
    the artist better scores *worse*."""

    def __init__(self, path: Path, name: str, ks: tuple[int, ...] = (1, 5)):
        super().__init__()
        self.rows = load_jsonl(path)
        self.name, self.ks = name, ks
        self.primary_metric = f"{name}_accuracy@1"

    def __call__(self, model, output_path=None, epoch=-1, steps=-1) -> dict[str, float]:
        texts = list(dict.fromkeys(t for r in self.rows for t in [r["query"], *r["corpus"]]))
        index = {t: i for i, t in enumerate(texts)}
        emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        hits = [0] * len(self.ks)
        for r in self.rows:
            q = emb[index[r["query"]]]
            scores = [(float(emb[index[t]] @ q), t) for t in r["corpus"]]
            ranked = [t for _, t in sorted(scores, key=lambda x: -x[0])]
            rank = ranked.index(r["positive"]) + 1
            for i, k in enumerate(self.ks):
                hits[i] += rank <= k
        metrics = {f"{self.name}_accuracy@{k}": h / len(self.rows) for k, h in zip(self.ks, hits)}
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics


def artist_wrap_evaluator(base: Path) -> PerQueryEvaluator | None:
    path = base / "artist_wrap_test.jsonl"
    return PerQueryEvaluator(path, "artist_wrap-test") if path.exists() else None


def channel_evaluator(base: Path) -> InformationRetrievalEvaluator | None:
    """Held-out real YouTube channel titles (artist_wrap.py) -> any name of
    their MB artist, among every fetched channel title (wrapper-sharing
    distractors) plus the held-out artists' names."""
    path = base / "channel_test.jsonl"
    if not path.exists():
        return None
    corpus: dict[str, str] = {}
    by_key: dict[str, str] = {}

    def cid(text: str) -> str:
        key = norm_key(text)
        if key not in by_key:
            by_key[key] = f"c{len(by_key)}"
            corpus[by_key[key]] = text
        return by_key[key]

    test = load_jsonl(path)
    anchors = {norm_key(r["anchor"]) for r in test}
    queries, relevant = {}, {}
    for i, row in enumerate(test):
        rel = {cid(p) for p in row["positives"] if norm_key(p) not in anchors}
        if rel:
            queries[f"q{i}"] = row["anchor"]
            relevant[f"q{i}"] = rel
    for row in load_jsonl(base / "youtube_channels.jsonl"):
        if not row.get("missing") and norm_key(row.get("title", "")) and norm_key(row["title"]) not in anchors:
            cid(row["title"])
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name="channel-test", accuracy_at_k=[1, 10, 20],
        precision_recall_at_k=[10, 20], mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10],
        show_progress_bar=False, batch_size=256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--run", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mini-batch-size", type=int, default=64,
                        help="GradCache chunk; the contrastive batch stays --batch-size")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-seq-length", type=int, default=48)
    parser.add_argument("--matryoshka", default="384,256,128,64",
                        help="comma-separated dims, first = full dim; '' disables")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-queries", type=int, default=3000)
    parser.add_argument("--data", action="append",
                        help="jsonl name in the build dir, optionally name:max_rows; repeatable "
                             "(default: train_pairs + train_triplets)")
    parser.add_argument("--limit", type=int, default=0, help="debug: cap rows per dataset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = build_dir(args.build)
    out = model_dir(args.run)

    model = SentenceTransformer(args.base_model, device="cuda")
    model.max_seq_length = args.max_seq_length

    rng = random.Random(args.seed)

    def dataset(spec: str) -> Dataset:
        name, _, cap = spec.partition(":")
        rows = load_jsonl(base / f"{name}.jsonl")
        rng.shuffle(rows)
        if cap:
            rows = rows[: int(cap)]
        if args.limit:
            rows = rows[: args.limit]
        columns = ["anchor", "positive"] + (["negative"] if all("negative" in r for r in rows) else [])
        return Dataset.from_dict({c: [r[c] for r in rows] for c in columns})

    train = {spec.partition(":")[0]: dataset(spec)
             for spec in (args.data or ["train_pairs", "train_triplets"])}
    print({k: (len(v), v.column_names) for k, v in train.items()})
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
    if args.matryoshka:
        loss = MatryoshkaLoss(model, loss, [int(d) for d in args.matryoshka.split(",")])

    evaluators = [mb_evaluator(base, args.eval_queries, args.seed)]
    for extra in (video_evaluator(base), template_evaluator(base),
                  template_evaluator(base, "cover"), artist_wrap_evaluator(base),
                  channel_evaluator(base)):
        if extra is not None:
            evaluators.append(extra)
    evaluator = SequentialEvaluator(evaluators)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        # Only the final model is kept: mid-run checkpoints (weights +
        # optimizer state, ~1-3 GB each) once filled the disk.
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        dataloader_drop_last=True,
    )
    print("baseline:", evaluator(model))
    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train, loss=loss, evaluator=evaluator,
    )
    trainer.train()
    final = evaluator(model)
    print("final:", final)
    model.save_pretrained(str(out / "final"))
    (out / "run.json").write_text(json.dumps({"args": vars(args), "final": final},
                                             indent=2) + "\n")


if __name__ == "__main__":
    main()
