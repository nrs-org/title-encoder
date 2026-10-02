# JP music title encoder

Training and evaluation code for
[`gbnam8/jp-music-title-encoder`](https://huggingface.co/gbnam8/jp-music-title-encoder),
a small Japanese/English music-entity retrieval model. It maps alternate song,
artist, release, and platform-wrapped names into a shared embedding space. The
published model is a six-layer student distilled from a twelve-layer
`multilingual-e5-small` teacher.

> **Research prototype.** Most of this project was produced with an AI coding
> assistant under light human direction. The runs are real, but the code,
> data, and methodology have not received an independent audit.

## Published artifacts

The immutable revisions and weight hashes are recorded in
[`artifacts.json`](artifacts.json).

| artifact | immutable revision | purpose |
|---|---|---|
| [student](https://huggingface.co/gbnam8/jp-music-title-encoder) | `a91836768a89a105db4e36b480c06e5418270835` | recommended 6-layer model, pruned tokenizer, ONNX |
| [teacher](https://huggingface.co/gbnam8/jp-music-title-encoder-teacher) | `c89465f714d7669e0e433841dd877e31f5bff4b7` | final 12-layer v7 checkpoint |
| `gbnam8/jp-music-title-encoder-data` | private, revision not recorded publicly | exact training rows, evaluation data, cards, and logs |

The last row is a real reproducibility boundary. A user with access to the
private, immutable data snapshot can reconstruct the checkpoints with the
commands below. A user without it can regenerate an equivalent dataset from a
MusicBrainz mirror and the YouTube API, but live source changes mean the rows
will not be identical. Exact public reproduction requires publishing a
sanitized data snapshot and pinning its commit.

## Reproduce the checkpoints

Requirements: Linux, an NVIDIA CUDA GPU with at least 6 GB VRAM, Python
3.12–3.13, [`uv`](https://docs.astral.sh/uv/), and enough disk for the data,
intermediate teachers, and Hugging Face caches. The original runs used an RTX
3060 6 GB. The lockfile currently resolves PyTorch 2.11.0+cu128,
sentence-transformers 6.1.0, and Transformers 5.18.0.

```bash
git clone <this-repository-url> jp-music-title-encoder-training
cd jp-music-title-encoder-training
uv sync --frozen

# Requires access. Pin DATASET_REVISION; never train from an unpinned `main`.
uv run hf download gbnam8/jp-music-title-encoder-data \
  --repo-type dataset --revision "$DATASET_REVISION" \
  --local-dir data/title-encoder

./scripts/reproduce-v7.sh
```

The script pins the upstream e5 checkpoint, checks the expected data files,
trains v1 through v7, distills v7 into the six-layer student, prunes the
tokenizer, exports ONNX, and runs held-out evaluation. Outputs are written to
`data/title-encoder/models/`.

All stochastic project steps use seed `0`. CUDA reductions and nearest-neighbor
ties are not guaranteed bit-for-bit deterministic across GPU architectures and
library builds, so successful replication means matching evaluation within
normal numerical tolerance and matching the architecture/tokenizer contract;
do not expect identical weight bytes.

To keep data elsewhere:

```bash
export TITLE_ENCODER_DATA_ROOT=/mnt/title-encoder
./scripts/reproduce-v7.sh
```

See [docs/TRAINING.md](docs/TRAINING.md) for the data construction, loss,
phase-by-phase lineage, evaluation protocol, expected counts, and a manual
command ledger.

## Result and inference contract

At a 511k-candidate pool, the published student achieved recall@20 of 0.985
for artist-conditioned tracks, 0.828 for artists, 0.824 for cross-script
queries, and 0.934 overall. These are measurements on the author's held-out
library, not an independently reproduced benchmark.

Tracks are encoded as `title [A] primary artist`; other entities use their raw
name. There is no `query:` or `passage:` prefix. Inputs are truncated at 48
tokens. The 384-dimensional, L2-normalized vectors use dot product/cosine;
the first 256 dimensions may be retained and normalized again with little
measured loss.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "gbnam8/jp-music-title-encoder",
    revision="a91836768a89a105db4e36b480c06e5418270835",
)
vectors = model.encode(
    ["夜に駆ける [A] YOASOBI", "Yoru ni Kakeru [A] YOASOBI"],
    normalize_embeddings=True,
)
print(vectors @ vectors.T)
```

## Repository layout

- `src/title_encoder/`: data builders, training, distillation, evaluation,
  vocabulary pruning, and ONNX export.
- `sql/`: MusicBrainz export queries.
- `scripts/reproduce-v7.sh`: executable checkpoint recipe.
- `docs/TRAINING.md`: methods and reproducibility record.
- `artifacts.json`: immutable upstream/output revisions and SHA-256 hashes.
- `uv.lock`: exact Python dependency solution.

Data and model outputs are intentionally ignored by Git. The source is MIT
licensed; source datasets and APIs retain their own terms.
