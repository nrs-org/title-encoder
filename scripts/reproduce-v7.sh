#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${TITLE_ENCODER_DATA_ROOT:-${PROJECT_DIR}/data/title-encoder}"
BUILD_DIR="${DATA_DIR}/mb-v1"
MODEL_DIR="${DATA_DIR}/models"
UPSTREAM_DIR="${DATA_DIR}/upstream/multilingual-e5-small"
UPSTREAM_REVISION="614241f622f53c4eeff9890bdc4f31cfecc418b3"

cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src"
export TITLE_ENCODER_DATA_ROOT="$DATA_DIR"
mkdir -p "${DATA_DIR}/eval"

required=(
  train_pairs.jsonl train_triplets.jsonl test_pairs.jsonl distractors.jsonl
  synth_triplets.jsonl video_triplets.jsonl video_pairs_train.jsonl
  artist_triplets.jsonl cond_triplets.jsonl mined_artist_triplets.jsonl
  mined_artist_triplets_r2.jsonl artist_wrap_triplets.jsonl
  artist_wrap_test.jsonl channel_test.jsonl
)
for file in "${required[@]}"; do
  if [[ ! -s "${BUILD_DIR}/${file}" ]]; then
    echo "missing ${BUILD_DIR}/${file}" >&2
    echo "download the pinned training-data snapshot first; see README.md" >&2
    exit 1
  fi
done

outputs=(e5s-mb-v1 e5s-mb-v2 e5s-mb-v3 e5s-mb-v4c e5s-mb-v5 e5s-mb-v6 e5s-mb-v7 e5s-v7-L6)
for run in "${outputs[@]}"; do
  if [[ -e "${MODEL_DIR}/${run}" ]]; then
    echo "refusing to overwrite ${MODEL_DIR}/${run}" >&2
    exit 1
  fi
done

if [[ ! -s "${UPSTREAM_DIR}/model.safetensors" ]]; then
  uv run hf download intfloat/multilingual-e5-small \
    --revision "$UPSTREAM_REVISION" --local-dir "$UPSTREAM_DIR"
fi

uv run python -m title_encoder.train \
  --run e5s-mb-v1 --base-model "$UPSTREAM_DIR"

uv run python -m title_encoder.train --run e5s-mb-v2 \
  --base-model "${MODEL_DIR}/e5s-mb-v1/final" \
  --data synth_triplets --data video_triplets --data video_pairs_train \
  --data train_pairs:100000 --data train_triplets

uv run python -m title_encoder.train --run e5s-mb-v3 \
  --base-model "${MODEL_DIR}/e5s-mb-v2/final" --lr 2e-5 \
  --data artist_triplets --data synth_triplets:60000 --data video_triplets \
  --data train_triplets --data train_pairs:60000

uv run python -m title_encoder.train --run e5s-mb-v4c \
  --base-model "${MODEL_DIR}/e5s-mb-v3/final" --lr 2e-5 \
  --data cond_triplets --data artist_triplets --data synth_triplets:60000 \
  --data video_triplets --data train_triplets --data train_pairs:60000

uv run python -m title_encoder.train --run e5s-mb-v5 \
  --base-model "${MODEL_DIR}/e5s-mb-v4c/final" --lr 2e-5 \
  --data mined_artist_triplets --data cond_triplets:80000 --data artist_triplets \
  --data synth_triplets:50000 --data video_triplets --data train_triplets \
  --data train_pairs:50000

uv run python -m title_encoder.train --run e5s-mb-v6 \
  --base-model "${MODEL_DIR}/e5s-mb-v5/final" --lr 2e-5 \
  --data mined_artist_triplets_r2 --data mined_artist_triplets:30000 \
  --data cond_triplets:80000 --data artist_triplets:30000 \
  --data synth_triplets:50000 --data video_triplets --data train_triplets \
  --data train_pairs:50000

uv run python -m title_encoder.train --run e5s-mb-v7 \
  --base-model "${MODEL_DIR}/e5s-mb-v6/final" --lr 1.5e-5 \
  --data artist_wrap_triplets:180000 --data mined_artist_triplets_r2:50000 \
  --data mined_artist_triplets:20000 --data cond_triplets:70000 \
  --data artist_triplets:25000 --data synth_triplets:40000 \
  --data video_triplets --data train_triplets --data train_pairs:40000

uv run python -m title_encoder.distill \
  --teacher "${MODEL_DIR}/e5s-mb-v7/final" --layers 6 --run e5s-v7-L6 \
  --distill-texts 500000 \
  --contrastive artist_wrap_triplets:120000 \
  --contrastive mined_artist_triplets_r2 \
  --contrastive mined_artist_triplets:40000 \
  --contrastive cond_triplets:120000 \
  --contrastive artist_triplets \
  --contrastive synth_triplets:80000 \
  --contrastive video_triplets \
  --contrastive train_triplets \
  --contrastive train_pairs:80000

uv run python -m title_encoder.prune_vocab \
  "${MODEL_DIR}/e5s-v7-L6/final" "${MODEL_DIR}/e5s-v7-L6/pruned"
uv run python -m title_encoder.export_onnx "${MODEL_DIR}/e5s-v7-L6/pruned"
uv run python -m title_encoder.eval_heldout \
  "${MODEL_DIR}/e5s-mb-v7/final" "${MODEL_DIR}/e5s-v7-L6/pruned" \
  --out "${DATA_DIR}/eval/reproduced-v7-heldout.json"

echo "student: ${MODEL_DIR}/e5s-v7-L6/pruned"
echo "teacher: ${MODEL_DIR}/e5s-mb-v7/final"
