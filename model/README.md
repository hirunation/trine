# TRINE Models

This directory contains trained Stage-2 semantic models.

## Directory Structure

- `checkpoints/` — Training snapshots for warm-start or refinement
- `release/` — Production models, ready for inference

## Model File Formats

**.trine2** — Frozen model (~172.9 KB). Learned projection matrix and cascade network. Used with `trine_embed` and `trine_dedup`.

**.trine2a** — Accumulator state (~691.3 KB). Hebbian counters for warm-start training.

## Training

```bash
make
./build/trine_train --data training.jsonl --val validation.jsonl \
  --save model/release/model.trine2
```

Input: JSONL with `text_a`, `text_b` fields (validation requires `score`, 0.0-1.0).

### Best Configuration (v1.0.2)

```bash
./build/trine_train --data training.jsonl --val validation.jsonl \
  --epochs 10 --diagonal --threshold 0 --density 0.33 \
  --cells 512 --depth 4 --similarity-threshold 0.5 \
  --save model/release/model.trine2
```

Achieves: blend rho 0.2812 (+28.8% over Stage-1).

## Using Models

**Embedding:**
```bash
./build/trine_embed --stage2 --model model/release/model.trine2 \
  --depth 4 "text"
```

**Deduplication:**
```bash
./build/trine_dedup --semantic model/release/model.trine2 \
  --s2-depth 4 input.jsonl > output.jsonl
```

## Warm-Start Training

```bash
./build/trine_train --data extended.jsonl \
  --load-accum model/checkpoints/accum.trine2a \
  --epochs 5 --save model/release/model_v2.trine2 \
  --save-accum model/checkpoints/accum_v2.trine2a
```
