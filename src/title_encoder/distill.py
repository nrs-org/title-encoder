"""Distill a fine-tuned teacher into a shallower student for CPU speed.

The student keeps the teacher's tokenizer, embeddings and width, and a subset
of its transformer layers (spread evenly, first and last kept), so it starts
close to the teacher. Training is MSE to the teacher's embeddings over every
title we have (labels not needed) plus the contrastive loss on labeled rows,
so the student keeps both the teacher's geometry and the task signal.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MSELoss
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

from .paths import build_dir, model_dir
from .text import norm_key
from .train import artist_wrap_evaluator, channel_evaluator, load_jsonl, mb_evaluator, video_evaluator


def keep_layers(n_total: int, n_keep: int) -> list[int]:
    if n_keep >= n_total:
        return list(range(n_total))
    return sorted({round(i * (n_total - 1) / (n_keep - 1)) for i in range(n_keep)})


def make_student(teacher_path: str, n_layers: int) -> SentenceTransformer:
    student = SentenceTransformer(teacher_path, device="cuda")
    bert = student[0].auto_model
    layers = bert.encoder.layer
    keep = keep_layers(len(layers), n_layers)
    bert.encoder.layer = torch.nn.ModuleList([layers[i] for i in keep])
    bert.config.num_hidden_layers = len(keep)
    print(f"student keeps teacher layers {keep}")
    return student


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--run", required=True)
    parser.add_argument("--distill-texts", type=int, default=400_000)
    parser.add_argument("--contrastive", action="append", default=[],
                        help="jsonl name[:max_rows] for the contrastive term; repeatable")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = build_dir(args.build)
    out = model_dir(args.run)
    rng = random.Random(args.seed)

    # Every distinct title from the training files (never test files).
    texts: dict[str, str] = {}
    for name in ("train_pairs", "train_triplets", "synth_triplets", "video_triplets",
                 "video_pairs_train", "artist_triplets", "cond_triplets",
                 "mined_artist_triplets", "mined_artist_triplets_r2", "artist_wrap_triplets"):
        path = base / f"{name}.jsonl"
        if not path.exists():
            continue
        for row in load_jsonl(path):
            for col in ("anchor", "positive", "negative"):
                if col in row:
                    texts.setdefault(norm_key(row[col]), row[col])
    pool = list(texts.values())
    rng.shuffle(pool)
    pool = pool[: args.distill_texts]
    print(f"distill texts: {len(pool)}")

    teacher = SentenceTransformer(args.teacher, device="cuda")
    teacher.max_seq_length = 48
    targets = teacher.encode(pool, batch_size=512, convert_to_numpy=True,
                             normalize_embeddings=False, show_progress_bar=True)
    del teacher
    torch.cuda.empty_cache()

    student = make_student(args.teacher, args.layers)
    student.max_seq_length = 48

    train = {"distill": Dataset.from_dict({"text": pool, "label": [t for t in targets.astype(np.float32)]})}
    losses = {"distill": MSELoss(student)}
    for spec in args.contrastive:
        name, _, cap = spec.partition(":")
        rows = load_jsonl(base / f"{name}.jsonl")
        rng.shuffle(rows)
        if cap:
            rows = rows[: int(cap)]
        cols = ["anchor", "positive"] + (["negative"] if all("negative" in r for r in rows) else [])
        train[name] = Dataset.from_dict({c: [r[c] for r in rows] for c in cols})
        losses[name] = CachedMultipleNegativesRankingLoss(student, mini_batch_size=args.mini_batch_size)
    print({k: len(v) for k, v in train.items()})

    evaluators = [mb_evaluator(base, 3000, args.seed)]
    for extra in (video_evaluator(base), artist_wrap_evaluator(base),
                  channel_evaluator(base)):
        if extra is not None:
            evaluators.append(extra)
    evaluator = SequentialEvaluator(evaluators)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.05,
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
    print("student before training:", evaluator(student))
    trainer = SentenceTransformerTrainer(model=student, args=targs, train_dataset=train,
                                         loss=losses, evaluator=evaluator)
    trainer.train()
    final = evaluator(student)
    print("final:", final)
    student.save_pretrained(str(out / "final"))
    (out / "run.json").write_text(json.dumps({"args": vars(args), "final": final}, indent=2) + "\n")


if __name__ == "__main__":
    main()
