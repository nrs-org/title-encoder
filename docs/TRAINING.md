# Training and reproducibility record

This document is the methods appendix and command ledger for the v7 teacher
and v7-L6 student published on Hugging Face. It distinguishes two targets:

1. **Checkpoint replication** uses the immutable prepared JSONL/CSV snapshot.
   This is the only route expected to match the published run closely.
2. **Method reproduction** rebuilds data from MusicBrainz and YouTube. It
   exercises the same pipeline, but source drift prevents identical rows.

## Task and model

The model supplies candidates for music entity resolution. Given one stored
name, it retrieves alternate names for the same track, artist, release, or
release group. The target domain is Japanese and English music, especially
J-pop, anime, VTuber, Vocaloid, game, and idol catalogs.

The teacher starts from `intfloat/multilingual-e5-small` at revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`: 12 BERT layers, hidden size 384.
The final student retains teacher layers `[0, 2, 4, 7, 9, 11]`. Both use a
48-token maximum. The student tokenizer is pruned from the XLM-R Unigram
vocabulary to 75,458 tokens after training.

Track inputs are `title [A] primary artist` when an artist is known. Artist,
release, and release-group inputs are raw names. No e5 query/passage prefix is
used.

## Environment

The original machine was NixOS with an NVIDIA RTX 3060 (6 GB). Training uses
bf16, a logical batch of 256, and GradCache mini-batches of 64. The final v7
teacher took about 49 minutes and the v7-L6 distillation about 19 minutes on
that machine; the full lineage takes several GPU-hours.

Install exactly the locked environment:

```bash
uv sync --frozen
export PYTHONPATH="$PWD/src"
```

On the original NixOS host these were also required:

```bash
export LD_LIBRARY_PATH=/run/opengl-driver/lib
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

The lock resolves Python 3.12–3.13 packages including PyTorch 2.11.0+cu128,
sentence-transformers 6.1.0, Transformers 5.18.0, datasets 5.0.1,
Accelerate 1.15.0, NumPy 2.5.3, ONNX 1.23.1, and ONNX Runtime 1.30.0.

## Prepared data layout

Commands default to `data/title-encoder`; set `TITLE_ENCODER_DATA_ROOT` to
override it. The build name is `mb-v1`.

Important files in `mb-v1/` are:

| file | role in the published lineage |
|---|---|
| `train_pairs.jsonl`, `train_triplets.jsonl` | clean MB alias/song positives and same-credit hard negatives |
| `test_pairs.jsonl`, `distractors.jsonl` | hash-held-out MB retrieval evaluation |
| `video_pairs_train.jsonl`, `video_triplets.jsonl` | real MB-linked YouTube titles |
| `synth_triplets.jsonl` | template-wrapped song titles with same-template negatives |
| `artist_triplets.jsonl` | cross-script artist pairs with shared-name-part negatives |
| `cond_triplets.jsonl` | `title [A] artist` positives and cover negatives |
| `mined_artist_triplets.jsonl` | nearest wrong artists under v4c |
| `mined_artist_triplets_r2.jsonl` | second mining round under v5 |
| `artist_wrap_triplets.jsonl` | YouTube/platform artist wrapper examples |
| `*_test.jsonl`, `channel_test.jsonl` | held-out video, template, cover, wrapper, and channel tests |
| `raw/*.csv`, `*_card.json` | source exports and data provenance/counts |

Rows have `anchor`, `positive`, and optionally `negative` string fields.
Evaluation files add group IDs or per-query corpora. The published export used
MusicBrainz replication sequence 189407. Its leakage inputs contained 19,289
library MBIDs and 12,283 library artist-name keys.

Observed build sizes included:

- 345k initial training rows: about 276k JP and 69k EN, 67k triplets, and
  about 143k cross-script JP pairs.
- 299,382 synthetic triplets; 8,385 real video triplets; 4,777 video pairs.
- 41,403 artist triplets; 204,866 conditioned triplets.
- 79,234 first-round and 79,145 second-round mined artist triplets.
- 261k available artist-wrapper triplets, of which 180k train v7 and 120k
  replay during distillation.

## Leakage and splits

The external `musiclib.db` library is the final evaluation set. Before the MB
export, the pipeline excludes every MB entity present in that library, every
artist credited on those entities, every artist sharing a library artist's
normalized name, and all recordings and works associated with excluded
artists. The export reported 17,651 artists, 1,865,431 recordings, and 215,610
works excluded.

Training/test assignment is deterministic hashing, normally 1/50 held out.
The YouTube split follows the MusicBrainz credit hash. Artist-wrapper data
holds out about 3% of artists and 10% of group names. Random sampling and row
shuffling use seed 0.

The library test is private because it contains source identifiers. Therefore
the public result is not independently auditable today. The MB and synthetic
held-out sets can be included in a sanitized data release without publishing
the library database.

## Data construction from live sources

This route requires a local
[`metabrainz/musicbrainz-docker`](https://github.com/metabrainz/musicbrainz-docker)
database container named `musicbrainz-docker-db-1`, a compatible musiclib
SQLite database, and a YouTube Data API key. It reproduces the method rather
than the exact 2026-10-02 snapshot.

```bash
export YOUTUBE_API_KEY=...

# Leakage-filtered MB export and initial pairs.
uv run python -m title_encoder.build_mb --library-db /path/to/musiclib.db
uv run python -m title_encoder.pairs

# Real videos, mined templates, and synthetic title wrappers.
uv run python -m title_encoder.fetch_youtube
uv run python -m title_encoder.mine_templates
uv run python -m title_encoder.synth

# Cross-script artists and artist-conditioned cover data.
uv run python -m title_encoder.artist_negs
uv run python -m title_encoder.cond_data --library-db /path/to/musiclib.db
```

The 2026 snapshot fetched 19,273 videos (386 `videos.list` calls) and mined
5,465 templates. The fetchers are append-only/resumable and record unavailable
objects so they are not retried.

For v7, export artist-channel links and Japanese label/group names:

```bash
docker cp sql/export_channels.sql musicbrainz-docker-db-1:/tmp/export_channels.sql
docker exec musicbrainz-docker-db-1 psql -U musicbrainz -d musicbrainz_db \
  -X -q -f /tmp/export_channels.sql
docker cp musicbrainz-docker-db-1:/tmp/te_channels.csv \
  data/title-encoder/mb-v1/raw/artist_channels.csv

docker cp sql/export_groups.sql musicbrainz-docker-db-1:/tmp/export_groups.sql
docker exec musicbrainz-docker-db-1 psql -U musicbrainz -d musicbrainz_db \
  -X -q -f /tmp/export_groups.sql
docker cp musicbrainz-docker-db-1:/tmp/te_groups.csv \
  data/title-encoder/mb-v1/raw/groups.csv

uv run python -m title_encoder.channels
uv run python -m title_encoder.artist_wrap --library-db /path/to/musiclib.db
```

The original channel fetch yielded about 33.3k channel titles (675 calls),
146 mined wrapper templates, and 22 hand-written VTuber wrapper shapes.

## Teacher objective

Each training phase uses
`CachedMultipleNegativesRankingLoss(mini_batch_size=64)`. A logical batch of
256 supplies in-batch negatives while GradCache controls memory. Triplet rows
also supply their explicit hard negative. `BatchSamplers.NO_DUPLICATES`
prevents identical strings in one batch from becoming false negatives.

The contrastive loss is wrapped in `MatryoshkaLoss` at dimensions
`[384, 256, 128, 64]`. AdamW is configured by Sentence Transformers with one
epoch, bf16, a 0.1 warmup ratio, no intermediate checkpoint saves, and seed 0.
Evaluation runs every 500 steps. Unless stated otherwise, learning rate is
5e-5; later continuation phases use lower rates.

## Exact teacher lineage

[`scripts/reproduce-v7.sh`](../scripts/reproduce-v7.sh) is the executable
record. The phase changes are:

| phase | starts from | learning rate | added signal / sampled inputs |
|---|---|---:|---|
| v1 | pinned e5-small | 5e-5 | all clean MB pairs and triplets |
| v2 | v1 | 5e-5 | 299,382 synthetic, 8,385 video triplets, 4,777 video pairs, 100k clean pairs, all clean triplets |
| v3 | v2 | 2e-5 | 41,403 cross-script artists, 60k synthetic, video and clean replay |
| v4c | v3 | 2e-5 | all 204,866 conditioned cover rows plus v3 replay |
| v5 | v4c | 2e-5 | 79,234 v4c-mined artist confusions plus capped replay |
| v6 | v5 | 2e-5 | 79,145 v5-mined confusions plus first-round and other replay |
| v7 | v6 | 1.5e-5 | 180k wrapper rows plus v6 replay |

To regenerate the two mined inputs instead of consuming the snapshot:

```bash
uv run python -m title_encoder.mine_negatives \
  --model data/title-encoder/models/e5s-mb-v4c/final \
  --out mined_artist_triplets
uv run python -m title_encoder.mine_negatives \
  --model data/title-encoder/models/e5s-mb-v5/final \
  --out mined_artist_triplets_r2
```

GPU nearest-neighbor ties can make regenerated files differ. Consume the
snapshot for checkpoint replication.

## Distillation, pruning, and ONNX

The student begins as a copy of the v7 teacher with layers
`[0, 2, 4, 7, 9, 11]`. It trains for one epoch with two simultaneous signals:

- MSE to teacher embeddings for 500,000 distinct training texts.
- The same cached multiple-negative loss on 120k wrapper, all second-round
  mined, 40k first-round mined, 120k conditioned, all artist, 80k synthetic,
  all video, all clean triplet, and 80k clean pair rows.

Distillation uses batch 256, mini-batch 64, bf16, learning rate 1e-4, warmup
ratio 0.05, and seed 0. The vocabulary pruner retains special tokens, every
piece observed in the corpus, and single-character pieces in served JP/EN
scripts. It slices the embedding matrix and verifies unchanged tokenization
and embeddings. ONNX export incorporates mean pooling and L2 normalization
and checks cosine parity against PyTorch.

The original artifacts had 39,968,640 student parameters, a 75,458-token
vocabulary, a 159,405,024-byte ONNX file, and minimum reported ONNX/PyTorch
cosine of 1.000000.

## Evaluation protocol and expected results

The private library evaluation produces queries only for entities with at
least two distinct aliases. A query must retrieve a different-text alias of
the same entity. Candidate pools are separated by entity type and optionally
padded with MusicBrainz distractors up to about 511k candidates. Tracks are
artist-conditioned per source. Known duplicate library entries count as
misses, so the measurement is a lower bound.

Published v7-L6 recall@20, pool size about 11k to 511k:

| stratum | 11k | 511k |
|---|---:|---:|
| tracks | 0.991 | 0.985 |
| artists | 0.909 | 0.828 |
| cross-script | 0.908 | 0.824 |
| all | 0.963 | 0.934 |

Held-out checks were approximately: MB accuracy@1 0.603, real video
accuracy@1 0.770, template accuracy@1 0.968, cover accuracy@1 0.997,
artist-wrapper accuracy@1 0.942, and real channel recall@20 0.881. Normal
floating-point and sampling differences should be investigated if they move
these figures by more than a few tenths of a percentage point.

## Reproducibility checklist

- Pin this source repository, the data snapshot, upstream e5 revision, and
  `uv.lock`.
- Preserve all `*_card.json` files and SHA-256 hashes of every JSONL/CSV.
- Record GPU, driver, CUDA, Python, PyTorch, and sentence-transformers versions.
- Run from an empty model-output directory; scripts refuse to overwrite it.
- Keep seed 0 and all default batch, sequence, Matryoshka, and warmup settings.
- Compare held-out metrics and model structure, not only weight bytes.
- Publish the sanitized data revision before claiming independent
  reproducibility. At present, this final item is unresolved.
