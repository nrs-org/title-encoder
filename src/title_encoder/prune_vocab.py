"""Prune an XLM-R (SentencePiece Unigram) model's vocabulary to JP/EN.

Kept tokens: specials, every token the training corpus actually produces, and
every single-character piece in the scripts we serve (kana, CJK, Latin,
fullwidth, punctuation), so any JP/EN title still has a segmentation. The
embedding matrix is sliced to the kept ids and tokenizer.json is rewritten
with remapped ids. Compute per token is unchanged; size and load time drop.
Verifies that re-tokenized library titles map to the same pieces and that
embeddings match the unpruned model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .paths import build_dir
from .text import allowed

KEEP_CHAR = re.compile(
    "^▁?["
    " -ɏ"  # Latin incl. extended, ASCII punctuation
    " -⁯←-⏿①-➿"  # general punctuation, arrows, symbols
    "　-ヿㇰ-ㇿ"  # CJK punctuation, kana
    "㐀-䶿一-鿿豈-﫿"  # CJK ideographs
    "＀-￯"  # halfwidth/fullwidth forms
    "]$"
)


def corpus_texts(build: Path) -> list[str]:
    texts = set()
    for path in build.glob("*.jsonl"):
        if path.name.startswith(("test", "video_test", "template_test", "distractors")):
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for col in ("anchor", "positive", "negative", "text"):
                    if isinstance(row.get(col), str):
                        texts.add(row[col])
    # Every exported MB name too (all tiers, before pair sampling): more rare
    # names keep their original segmentation.
    csv.field_size_limit(1 << 30)
    for path in (build / "raw").glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for col in ("name", "credit"):
                    if row.get(col) and allowed(row[col]):
                        texts.add(row[col])
    return sorted(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("out", type=Path)
    parser.add_argument("--build", default="mb-v1")
    parser.add_argument("--library-db", type=Path,
                        default=Path.home() / ".local/share/musiclib-rs/musiclib.db")
    args = parser.parse_args()

    st = SentenceTransformer(args.model, device="cpu")
    tok = st.tokenizer
    spec = json.loads(tok.backend_tokenizer.to_str())
    assert spec["model"]["type"] == "Unigram", spec["model"]["type"]
    vocab = spec["model"]["vocab"]

    keep = set(tok.all_special_ids)
    texts = corpus_texts(build_dir(args.build))
    for start in range(0, len(texts), 10000):
        for ids in tok(texts[start : start + 10000], add_special_tokens=False)["input_ids"]:
            keep.update(ids)
    from_corpus = len(keep)
    for i, (piece, _) in enumerate(vocab):
        if KEEP_CHAR.match(piece):
            keep.add(i)
    kept = sorted(keep)
    remap = {old: new for new, old in enumerate(kept)}
    print(f"vocab {len(vocab)} -> {len(kept)} ({from_corpus} from {len(texts)} corpus texts)")

    spec["model"]["vocab"] = [vocab[i] for i in kept]
    spec["model"]["unk_id"] = remap[spec["model"]["unk_id"]]
    for added in spec.get("added_tokens", []):
        added["id"] = remap[added["id"]]
    post = spec.get("post_processor") or {}
    for processor in [post] + post.get("processors", []):
        for special in (processor.get("special_tokens") or {}).values():
            special["ids"] = [remap[i] for i in special["ids"]]
        for key in ("sep", "cls"):
            if isinstance(processor.get(key), list):
                processor[key][1] = remap[processor[key][1]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        shutil.rmtree(args.out)
    st.save_pretrained(str(args.out))
    (args.out / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    for stale in ("sentencepiece.bpe.model",):
        (args.out / stale).unlink(missing_ok=True)

    # Slice the embedding matrix in the saved HF weights (module 0 lives at
    # the model root in sentence-transformers' layout).
    from transformers import AutoModel

    bert = AutoModel.from_pretrained(str(args.out))
    old_emb = bert.get_input_embeddings().weight.data
    new_emb = torch.nn.Embedding(len(kept), old_emb.shape[1])
    new_emb.weight.data = old_emb[torch.tensor(kept)].clone()
    bert.set_input_embeddings(new_emb)
    bert.config.vocab_size = len(kept)
    bert.save_pretrained(str(args.out))

    # The original run verified against the held-out library. A standalone
    # reproduction may not have that private database, so fall back to corpus
    # strings; verification does not affect the produced tokenizer or weights.
    if args.library_db.exists():
        con = sqlite3.connect(f"file:{args.library_db}?mode=ro", uri=True)
        lib = [r[0] for r in con.execute("select distinct name from entry_alias limit 20000")
               if allowed(r[0])]
        con.close()
        verification_source = str(args.library_db)
    else:
        lib = texts[:20000]
        verification_source = "training corpus (library DB unavailable)"
    reloaded = SentenceTransformer(str(args.out), device="cpu")
    same = sum(
        tok.convert_ids_to_tokens(a) == reloaded.tokenizer.convert_ids_to_tokens(b)
        for a, b in zip(tok(lib)["input_ids"], reloaded.tokenizer(lib)["input_ids"])
    )
    sample = lib[:2000]
    cos = (st.encode(sample, normalize_embeddings=True)
           * reloaded.encode(sample, normalize_embeddings=True)).sum(1)
    size = sum(p.numel() for p in reloaded.parameters())
    print(f"verification source: {verification_source}; "
          f"identical tokenization: {same}/{len(lib)} ({same / len(lib):.2%}); "
          f"embedding cosine min {cos.min():.4f} p1 {np.percentile(cos, 1):.4f}; "
          f"params {size / 1e6:.1f}M")


if __name__ == "__main__":
    main()
