# TRINE Architecture Overview

**v1.0.3** -- Zero-dependency, zero-float (inference), deterministic text
fingerprinting in C. 240-dim ternary vectors via multi-scale shingling.
Stage-1 surface similarity. Stage-2 Hebbian semantic projection (+28.8%).

---

## 1. Data Flow

```
  text -> [canonicalize] -> [shingle encode] -> x0 in Z3^240
       -> [project K=3]  -> x1               -> [cascade x N] -> x_{N+1}
       -> [compare]      -> similarity in [0,1]

  Modules:  trine_canon.h -> trine_encode.h -> trine_project.h
         -> trine_learned_cascade.h -> trine_stage2.h -> trine_stage1.h
```

Every intermediate x_k is a valid 240-trit embedding. Depth is chosen at
query time. The inference path is zero-float; comparison uses float.

---

## 2. Encoding Pipeline (Stage-1)

### Channel Layout

```
  240 = 4 chains x 60 channels
  +----------+----------+----------+----------+
  | Ch0 0-59 | Ch1 60-  | Ch2 120- | Ch3 180- |
  | char 1+2 | char 3   | char 5   | word 1   |
  | (Edit)   | (Morph)  | (Phrase) | (Vocab)  |
  +----------+----------+----------+----------+
```

Each trit in Z_3 = {0,1,2}. Zero = absent feature. N-grams are hashed
(seeded FNV-1a) to K slots per chain. Values accumulate via mod-3 addition.
Input is case-folded. ~4.7M encodes/sec, arbitrary-length input.

### Comparison and Indexing

Lens-weighted cosine: per-chain weights emphasize different aspects (UNIFORM,
DEDUP, CODE, LEGAL, etc.). Length calibration: `raw / sqrt(fill_a * fill_b)`.
Packed storage: 240 trits -> 48 bytes (5 trits/byte).

- **Linear scan** (`trine_stage1.h`): O(N), sufficient for <10K entries
- **Band-LSH** (`trine_route.h`): 4-band hash, 3 probes/band, 10-50x speedup
- **CS-IDF** (`trine_csidf.h`): corpus-specific IDF on trit channels
- **Multi-field** (`trine_field.h`): composite embeddings, per-field blending

---

## 3. Semantic Projection (Stage-2)

### Projection Modes

| Mode | Name       | Params  | Operation                                |
|------|------------|---------|------------------------------------------|
| 0    | SIGN       | 172,800 | Full 240x240 centered dot + sign quant   |
| 1    | DIAGONAL   | 720     | Per-channel keep/flip/zero (diagonal)    |
| 2    | SPARSE     | 172,800 | Like SIGN, W=0 entries skipped           |
| 3    | BLOCK_DIAG | 43,200  | 4 independent 60x60 chain-local matmuls  |

All modes: K=3 copies, per-channel majority vote. All integer arithmetic.

### K=3 Majority Vote

Naive mod-3 washes out to uniform. Instead, 3 weight matrices produce
independent outputs; per-channel majority trit wins (tie: first copy wins):

```
  W[0]*x -> y0    W[1]*x -> y1    W[2]*x -> y2
       \               |               /
        +---- majority(y0[i], y1[i], y2[i]) ----> y[i]
```

### Cascade Mixing

ENDO-based mixing network using the 27-element ternary endomorphism table:

```
  Per tick: out = copy(in)    // residual
    for each cell: out[dst] = (out[dst] + ENDO[endo_id][in[src]]) % 3
```

Each cell: read 1 channel, apply 1 of 27 endomorphisms, accumulate into 1
output channel. n_cells=0 -> identity. Default: 512 cells, max depth 64.

### Depth and Comparison

Two depth modes: cascade (project once, cascade N times) or stacked
(re-project each tick). `trine_s2_encode_depths()` extracts all depths.

Stage-2 comparison: standard cosine, gated (skip zero-gate channels),
chain-blend (per-chain S1/S2 alpha), adaptive blend (10 S1-buckets,
per-bucket alpha: `result = alpha(s1)*s1 + (1-alpha(s1))*s2`).

---

## 4. Block-Diagonal Design

The 240-dim space decomposes into 4 chains. Cross-chain correlation is
structurally absent in the encoding. Block-diagonal projection enforces this:

```
  Full 240x240 (172,800 params):     Block 4x60x60 (43,200 params):
  +----+----+----+----+              +----+              +
  |    |    |    |    |              |C0  |  .    .    . |
  +----+----+----+----+              +----+----+         |
  |    |    |    |    |    -->       |    |C1  |  .    . |
  +----+----+----+----+              |    +----+----+    |
  |    |    |    |    |              |    |    |C2  |  . |
  +----+----+----+----+              |    |    +----+----+
  |    |    |    |    |              |    |    |    |C3  |
  +----+----+----+----+              +----+----+----+----+
```

75% parameter reduction. No cross-chain weights. K=3 majority per chain.
Chain independence holds end-to-end: encode, project, compare, train.

---

## 5. Training Pipeline

### Hebbian Accumulation

Self-supervised from Stage-1 partial order (no external labels):

```
  s1 = compare(a, b);  sign = (s1 > threshold) ? +1 : -1
  counter[k][i][j] += sign * a[j] * b[i]   // outer product, int32 saturating
```

Weighted mode: magnitude = f(|s1 - threshold|). Source weights rebalance
datasets. Block accumulators: within-chain outer products only.

### Freeze

```
  W[i][j] = counter > +T ? 2 (keep)  :  counter < -T ? 1 (flip)  :  0 (zero)
```

T: explicit or auto-tuned to target density (default 33%). Optional sparse
freeze: top-K per output row.

### Self-Supervised Deepening

```
  round 0: train on S1 order -> freeze -> model_0
  round 1: encode through model_0 -> S2 order -> train -> model_1
  round N: iterate to WQO convergence (rank lattice P >= S >= K)
```

### Workflow

```
  create(config) -> observe_text() x N -> freeze() -> model
                    |                      |
                    v                      v
              save .trine2a          save .trine2
```

---

## 6. File Formats

| Ext     | Magic | Contents                      | Size          |
|---------|-------|-------------------------------|---------------|
| .trine  | TRN1  | Stage-1 cascade model         | varies        |
| .trine2 | TR2\0 | Stage-2 model (proj+cascade)  | 43-173 KB     |
| .trine2a| TR2A  | Hebbian accumulators          | 173-691 KB    |
| .tridx  | TRS1  | Stage-1 embedding index       | varies        |
| .trrt   | TRRT  | Routed index (LSH+data)       | varies        |

### .trine2 (Stage-2 Model, 72-byte header + payload)

```
  0:4   magic       4:4   version     8:4   flags (diag|ident|block)
  12:4  proj_k      16:4  proj_dim    20:4  cascade_cells
  24:4  depth       28:8  topo_seed   36:4  sim_threshold
  40:4  density     44:20 reserved    64:8  header_checksum
  72:W  weights     72+W:8  payload_checksum
  W = K*240*240 (full: 172,800) or K*4*60*60 (block: 43,200)
```

### .trine2a (Accumulator, 56-byte header + payload)

```
  0:4   magic       4:4   version     8:4   flags
  12:4  proj_k      16:4  proj_dim    20:4  pairs_observed
  24:4  threshold   28:4  density     32:24 reserved
  56:8  header_checksum    64:P  int32 counters   64+P:8  payload_cksum
  P = K*240*240*4 (full: 691,200) or K*4*60*60*4 (block: 172,800)
```

### .trrt (Routed Index)

```
  0:4 magic  4:4 version(3)  8:4 endian_marker  12:4 flags(csidf|fields)
  16:8 checksum  24:... payload (config + entries + bucket tables)
```

All formats: FNV-1a checksums, backward-compatible loaders.

---

## 7. Dependency Graph

```
  trine_algebra.h / oicos.h            (ENDO table, Z3 ops)
         |
  trine_encode.h -----> trine_canon.h  (standalone, no deps)
         |
  trine_stage1.h ---+-> trine_csidf.h
         |          +-> trine_route.h --> trine_field.h
         |
  trine_project.h                       (standalone)
         |
  trine_learned_cascade.h               (standalone)
         |
  trine_stage2.h                        (depends: trine_error.h)
         |
  trine_hebbian.h --> trine_accumulator.h, trine_freeze.h
         |            trine_self_deepen.h
         v
  trine_s2_persist.h, trine_accumulator_persist.h
```

| Module                       | Hard Dependencies                     |
|------------------------------|---------------------------------------|
| `trine_encode.h`             | none                                  |
| `trine_canon.h`              | none                                  |
| `trine_project.h`            | none                                  |
| `trine_learned_cascade.h`    | none                                  |
| `trine_stage1.h`             | `trine_error.h`                       |
| `trine_stage2.h`             | `trine_error.h`                       |
| `trine_route.h`              | `trine_stage1.h`, `trine_csidf.h`, `trine_field.h` |
| `trine_accumulator.h`        | none                                  |
| `trine_hebbian.h`            | forward decl `trine_s2_model`         |
| `trine_s2_persist.h`         | forward decl `trine_s2_model`         |
| `trine_accumulator_persist.h`| `trine_accumulator.h`, `trine_hebbian.h` |

Core modules (encode, project, cascade, accumulator) have zero internal
dependencies. Hebbian training uses forward declarations to avoid cycles.

---

## 8. Invariants

1. **C1**: Encoder is immutable. Learning happens in comparison geometry.
2. **C2**: Every cascade depth is a valid embedding. Depth = resolution.
3. **C3**: Training is self-supervised from Stage-1 partial order.
4. **C4**: Persistence IS the index. Add doc = Hebbian update. Query = inference.
5. **Zero-float inference**: encode + project + cascade = all integer.
6. **Determinism**: same input + same model = same output, always.
