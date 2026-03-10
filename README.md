# TRINE v1.0.3

**Ternary Resonance Interference Network Embedding** -- a deterministic, zero-dependency text embedding library in C with learned semantic projection.

TRINE produces 240-dimensional ternary (0/1/2) fingerprints using multi-scale n-gram shingling across 4 semantic chains.

- **Stage-1** encodes surface-form similarity via algebraic hashing. No neural networks, no training data, no external dependencies. The same input always produces the same embedding. ~4.7M encodes/sec.
- **Stage-2** adds a learned semantic projection layer trained via Hebbian accumulation. Supports diagonal gating and block-diagonal projection, both with K=3 majority vote, to project Stage-1 fingerprints into a semantic space. Block-diagonal mode captures cross-channel correlations within each chain via 4 independent 60x60 blocks (43,200 parameters). +28.8% over Stage-1 baseline on blended similarity. ~1M encodes/sec at depth 0. Zero floats in the inference path.

Zero dependencies beyond libc and libm. C99. Pure integer inference.

---

## Quick Start (CLI)

```bash
# Build everything
make

# Stage-1: surface fingerprinting
./build/trine_embed model.trine -r shingle "Hello, world!"

# Stage-1: compare two texts
./build/trine_embed model.trine --compare "hello world" "hello there" -r shingle --lens dedup --detail

# Stage-2: train a semantic model from JSONL pairs
./build/trine_train --data data/splits/train.jsonl --epochs 1 --diagonal \
  --density 0.15 --similarity-threshold 0.90 --depth 0 \
  --save model.trine2 --val data/splits/val.jsonl

# Stage-2: embed with a trained model
./build/trine_embed model.trine --stage2 --model model.trine2 "Hello, world!"

# Stage-2: embed with depth control
./build/trine_embed model.trine --stage2 --model model.trine2 --depth 4 "Hello, world!"

# Stage-2: train a block-diagonal model (cross-channel correlations within chains)
./build/trine_train --data data/splits/train.jsonl --epochs 1 --block-diagonal \
  --density 0.15 --similarity-threshold 0.90 --depth 0 \
  --save model.trine2 --val data/splits/val.jsonl

# Stage-2: embed with block-diagonal model
./build/trine_embed model.trine --stage2 --model model.trine2 "Hello, world!"

# Stage-2: embed with block-diagonal projection (random, no trained model)
./build/trine_embed model.trine --stage2 --block-diagonal "Hello, world!"

# Semantic dedup (blended S1+S2)
./build/trine_dedup check --semantic model.trine2 "heart attack" "myocardial infarction"

# Semantic dedup (S2-only)
./build/trine_dedup check --semantic model.trine2 --s2-only "heart attack" "myocardial infarction"

# Scan for near-duplicates
./build/trine_dedup scan input.txt --threshold 0.7 --lens dedup

# Batch dedup with persistent index
./build/trine_dedup batch input.txt --save-index index.tridx
./build/trine_dedup stats --load-index index.tridx
```

## Quick Start (C API)

### Stage-1

```c
#include "trine_stage1.h"

int main(void) {
    uint8_t a[240], b[240];
    trine_s1_encode("hello world", 11, a);
    trine_s1_encode("hello there", 11, b);

    trine_s1_lens_t lens = TRINE_S1_LENS_DEDUP;
    float sim = trine_s1_compare(a, b, &lens);
    printf("Similarity: %.3f\n", sim);
    return 0;
}
```

### Stage-2

```c
#include "trine_stage2.h"
#include "trine_s2_persist.h"

int main(void) {
    /* Load a trained model */
    trine_s2_model_t *model = trine_s2_load("model.trine2");

    /* Encode with Stage-2 projection (depth 0 = projection only) */
    uint8_t a[240], b[240];
    trine_s2_encode(model, "heart attack", 12, 0, a);
    trine_s2_encode(model, "myocardial infarction", 21, 0, b);

    /* Compare using the uniform lens */
    float sim = trine_s2_compare(a, b, NULL);
    printf("Stage-2 similarity: %.3f\n", sim);

    trine_s2_free(model);
    return 0;
}
```

Compile against the static library:

```bash
cc -O2 -o my_tool my_tool.c -Lbuild -ltrine -lm
```

## Quick Start (Python)

```python
from pytrine import TrineEncoder, TrineIndex, Lens

# Stage-1: encode and compare
encoder = TrineEncoder()
index = TrineIndex(threshold=0.70, lens=Lens.DEDUP)

index.add("hello world", tag="doc1")
index.add("hello there", tag="doc2")

result = index.query("hello world!")
print(f"Match: {result.tag}, similarity: {result.similarity:.3f}")
```

```python
from pytrine import Stage2Model, Stage2Encoder

# Stage-2: load a trained model
model = Stage2Model.load("model.trine2")
enc = Stage2Encoder(model, depth=0)

# Encode and compare
emb = enc.encode("Hello, world!")
sim = enc.similarity("heart attack", "myocardial infarction")

# Blended S1+S2 score (alpha=0.65 means 65% S1 + 35% S2)
blended = enc.blend("heart attack", "myocardial infarction", alpha=0.65)

# Adaptive alpha (auto-tuned from model statistics)
adaptive_alpha = model.adaptive_alpha()
blended = enc.blend("heart attack", "myocardial infarction", alpha=adaptive_alpha)
```

Install: `cd bindings/python && make lib && pip install -e .`

## Quick Start (Rust)

```rust
use trine::{Embedding, Config, Index, Lens};
use trine::stage2::{Stage2Model, stage2_compare, stage2_blend};

// Stage-1
let emb = Embedding::encode("hello world");
let mut index = Index::new(Config::default())?;
index.add(&emb, Some("doc1"))?;

// Stage-2
let model = Stage2Model::load("model.trine2")?;
let a = model.encode("hello world", 0);
let b = model.encode("hello earth", 0);
let sim = stage2_compare(&a, &b, &Lens::UNIFORM);
let blended = stage2_blend("hello world", "hello earth", &model, 0.65, &Lens::UNIFORM, 0);
```

Build: `cd bindings/rust && cargo build`

## TrineDB REST API

```bash
# Start server
./build/trinedb -p 7319

# Embed text
curl -s -X POST http://localhost:7319/embed -d '{"text":"hello world"}'

# Add to index and query
curl -s -X POST http://localhost:7319/index/add -d '{"text":"hello world","tag":"doc1"}'
curl -s -X POST http://localhost:7319/query -d '{"text":"hello world!"}'
```

## Docker

```bash
cd deploy/docker && docker-compose up
# TrineDB available at http://localhost:7319
```

---

## Architecture

TRINE uses a two-stage architecture. Stage-1 is a fixed, deterministic encoder. Stage-2 is a learned semantic projection trained by Hebbian accumulation.

```
text
  |
  v
Stage 1: trine_encode_shingle()         ->  x0 in Z3^240   (surface fingerprint)
  |         4 chains x 60 channels
  |         ~4.7M encodes/sec
  v
Stage 2: Projection (K=3 majority vote)  ->  x1 in Z3^240   (semantic rotation)
  |         diagonal or block-diagonal gating
  |         ~1M encodes/sec (depth=0)
  v
Cascade: ENDO-based mixing ticks        ->  x2..xN in Z3^240  (depth-N embedding)
  |         each tick = valid embedding
  v
Comparison: lens-weighted cosine         ->  similarity in [0, 1]
  |         or blended: alpha*S1 + (1-alpha)*S2
```

Every intermediate vector is a valid 240-trit embedding. Depth is chosen at query time.

### The 4 Chains

TRINE encodes text at four structural scales simultaneously. Each chain captures a different axis of textual similarity:

| Chain | Dimensions | Content | Captures |
|-------|-----------|---------|----------|
| Edit | 0-59 | Char unigrams + bigrams | Character-level edits |
| Morph | 60-119 | Char trigrams | Morphological structure |
| Phrase | 120-179 | Char 5-grams | Word-fragment patterns |
| Vocab | 180-239 | Word unigrams | Vocabulary fingerprint |

### Stage-2 Diagonal Gating

The learned projection uses per-channel diagonal gates with K=3 majority vote:

```
gate = 2 -> keep:  y[i] = x[i]        (positive correlation)
gate = 1 -> flip:  y[i] = (3 - x[i]) % 3  (Z3 negation)
gate = 0 -> zero:  y[i] = 1            (uninformative -> neutral)
```

Three independent diagonal matrices vote per channel. The majority trit wins. This is a pure integer operation -- zero floats in the inference path.

### Block-Diagonal Projection (v1.0.3)

Block-diagonal mode extends diagonal gating by allowing cross-channel correlations *within* each chain. Instead of a single 240x240 diagonal, the projection uses 4 independent 60x60 dense blocks -- one per chain (Edit, Morph, Phrase, Vocab). Each block captures pairwise channel interactions that diagonal gating cannot express, while preserving chain independence.

```
Chain 0 (Edit):   60x60 block  ->  3,600 weights  x K=3  =  10,800
Chain 1 (Morph):  60x60 block  ->  3,600 weights  x K=3  =  10,800
Chain 2 (Phrase): 60x60 block  ->  3,600 weights  x K=3  =  10,800
Chain 3 (Vocab):  60x60 block  ->  3,600 weights  x K=3  =  10,800
                                                    Total =  43,200 parameters
```

Each block is a ternary matrix (values in {0, 1, 2}). K=3 independent blocks per chain vote via majority rule, identical to diagonal mode. Inference remains pure integer -- zero floats.

Train with `--block-diagonal` instead of `--diagonal`:

```bash
./build/trine_train --block-diagonal --save model.trine2 train.jsonl
./build/trine_embed model.trine --stage2 --model model.trine2 "hello world"
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Dimensions | 240 (4 chains x 60) |
| Stage-1 encode throughput | ~4.7M embeddings/sec |
| Stage-1 compare (raw cosine) | ~6.9M comparisons/sec |
| Stage-1 compare (lens-weighted) | ~4.1M comparisons/sec |
| Stage-2 encode throughput | ~1M embeddings/sec (depth=0, diagonal) |
| Stage-2 blend rho | 0.2812 (+28.8% over Stage-1 baseline of 0.2184) |
| Blend alpha | 0.65 (65% S1 + 35% S2), adaptive per model |
| Stage-2 model size (diagonal) | 172,880 bytes (.trine2 format) |
| Stage-2 parameters (diagonal) | 720 (3 x 240) |
| Stage-2 parameters (block-diagonal) | 43,200 (3 x 4 x 60 x 60) |
| Dedup F1 (code) | 0.985 |
| Dedup F1 (legal) | 0.990 |
| Dedup F1 (text) | 0.851 |
| Tests | 612 (434 C + 94 Python + 44 Rust + 40 TrineDB) |
| Index insert | ~10.6M inserts/sec |
| Routed query | Sublinear via Band-LSH (10-50x speedup) |

---

## The TRINE Contract

### What TRINE is for

**Stage-1 (surface-form similarity):**

- Near-duplicate detection -- same content with minor edits, reordering, abbreviation
- Typo and case normalization -- "calculateTotal" vs "calculate_total" (code lens: 0.835)
- Format variant matching -- "ERROR: timeout after 30 seconds" vs "ERROR: timed out after 30 sec" (dedup lens: 0.709)
- Template variant detection -- boilerplate with field substitutions
- Code identifier matching -- camelCase/snake_case/kebab-case variants

**Stage-2 (semantic projection):**

- Improved similarity ranking for semantically related text pairs
- Blended scoring that combines surface-form and semantic signals
- Learned per-channel gating that amplifies informative dimensions and suppresses noise
- Self-supervised training from text pair co-occurrence statistics

### Honest boundaries

Stage-2 improves semantic correlation by +28.8% over Stage-1 alone. Diagonal mode operates via per-channel gating and cannot capture cross-channel correlations. Block-diagonal mode (v1.0.3) captures cross-channel correlations *within* each 60-channel chain but not *across* chains. For tasks requiring deep semantic understanding (cross-lingual matching, complex paraphrase, negation detection), a neural embedding model remains appropriate as a downstream reranker.

### Expected separation gaps

| Domain | Near-match example | Score | Disjoint example | Score | Gap |
|--------|-------------------|-------|-------------------|-------|-----|
| Code (code lens) | `calculateTotal` vs `calculate_total` | 0.835 | vs "weather is sunny" | 0.463 | **0.372** |
| Dedup (dedup lens) | "db timeout after 30s" variants | 0.709 | vs unrelated text | 0.507 | **0.202** |
| Legal (legal lens) | "governed by laws of NY" variants | 0.519 | vs unrelated | ~0.48 | **~0.04** |
| Medical (medical lens) | "myocardial infarction" vs "heart attack" | 0.536 | vs unrelated | 0.482 | **0.054** |

### Recommended 2-stage pipeline

```
+---------------------------------------------+
|  Stage 1: TRINE surface-form filter          |
|  - 4.7M encodes/sec, 76K QPS @ 10K          |
|  - Cuts candidates from N to ~40             |
|  - 99-100% recall for surface-form matches   |
|  - Cost: ~0 (deterministic, no GPU)          |
+---------------------------------------------+
                    |
                    v
+---------------------------------------------+
|  Stage 2: TRINE semantic projection          |
|  - 1M encodes/sec (depth=0)                 |
|  - +28.8% correlation via Hebbian learning   |
|  - Diagonal or block-diagonal (43.2K params) |
|  - Zero floats, zero GPU, 173 KB model       |
|  - Blended scoring: alpha*S1 + (1-a)*S2      |
+---------------------------------------------+
                    |
                    v  (optional, for deep semantics)
+---------------------------------------------+
|  Stage 3: Neural reranker (external)         |
|  - Runs on ~40 candidates, not 10K           |
|  - Catches synonyms, paraphrase              |
|  - Cost: proportional to candidates (small)  |
+---------------------------------------------+
```

---

## Lens System

Lenses weight the 4 chains differently for different use cases. TRINE ships with 11 presets.

### Generic Lenses

| Lens | Edit | Morph | Phrase | Vocab | Best for |
|------|------|-------|--------|-------|----------|
| `uniform` | 1.0 | 1.0 | 1.0 | 1.0 | General purpose |
| `edit` | 1.0 | 0.3 | 0.1 | 0.0 | Typo/edit detection |
| `morph` | 0.3 | 1.0 | 0.5 | 0.2 | Word form matching |
| `phrase` | 0.1 | 0.5 | 1.0 | 0.3 | Phrase similarity |
| `vocab` | 0.0 | 0.2 | 0.3 | 1.0 | Topic/vocabulary |
| `dedup` | 0.5 | 0.5 | 0.7 | 1.0 | Near-duplicate detection |

### Domain Lenses

| Lens | Edit | Morph | Phrase | Vocab | Domain |
|------|------|-------|--------|-------|--------|
| `code` | 1.0 | 0.8 | 0.4 | 0.2 | Source code |
| `legal` | 0.2 | 0.4 | 1.0 | 0.8 | Legal text |
| `medical` | 0.3 | 1.0 | 0.6 | 0.5 | Medical text |
| `support` | 0.2 | 0.4 | 0.7 | 1.0 | Customer support |
| `policy` | 0.1 | 0.3 | 1.0 | 0.8 | Regulatory text |

---

## Canonicalization Presets

Deterministic text transforms applied before encoding to improve near-duplicate detection on real-world corpora. All transforms are zero-allocation, operating in a caller-provided buffer.

| Preset | Transforms | Use Case |
|--------|-----------|----------|
| `NONE` | Passthrough | Raw text, no normalization |
| `GENERAL` | Whitespace collapse + trim | General cleanup |
| `SUPPORT` | Whitespace + timestamps + UUIDs + number bucketing | Logs, tickets, support |
| `CODE` | Whitespace + identifier normalization (camelCase/snake_case) | Source code |
| `POLICY` | Whitespace + number bucketing | Regulatory, legal |

```c
#include "trine_canon.h"

char buf[1024];
size_t out_len;
trine_canon_apply("ERROR 2024-01-15T10:30:00Z: timeout 1234", 41,
                  TRINE_CANON_SUPPORT, buf, sizeof(buf), &out_len);
// buf = "ERROR: timeout <N>"
```

---

## CS-IDF (Corpus-Specific IDF)

Auto-computed IDF weights derived from your actual corpus, replacing the static built-in table. Reduces noise floor by ~8-14% on domain-specific corpora where term frequency distributions differ from the general case.

- Weights are computed during index build -- no training step, no configuration
- Zero overhead vs static IDF (~4.1M comparisons/sec either way)
- Enable on a routed index after population:

```c
trine_route_enable_csidf(rt);  // switches from static IDF to corpus-derived weights
```

## Field-Aware Indexing

Embed title, body, code, or other fields separately and route queries by field weights. Useful when different fields carry different signal (e.g., a title match matters more than a body match).

- Each document stores per-field embeddings
- Queries specify field weights for weighted comparison
- ~3.0M field-aware comparisons/sec (2 fields)

```c
#include "trine_field.h"

trine_field_doc_t doc;
trine_field_doc_init(&doc);
trine_field_doc_set(&doc, "title", title_emb);
trine_field_doc_set(&doc, "body", body_emb);

float weights[] = {0.7, 0.3};  // title weight, body weight
```

---

## CLI Reference

### `trine_embed`

Encode text to embeddings, compare pairs, and run benchmarks. Supports both Stage-1 cascade models and Stage-2 semantic models.

```
Usage: trine_embed <model.trine> [options] <text>
       trine_embed <model.trine> --compare <text1> <text2>
       trine_embed <model.trine> --batch -f <file>
       trine_embed <model.trine> --benchmark N

Options:
  -r, --resolution LEVEL   screening|standard|deep|shingle (default: standard)
  -f FILE                  Read text from file
  -v, --verbose            Show cascade details
  -l, --lens SPEC          Lens: edit|morph|phrase|vocab|dedup|code|legal|
                             medical|support|policy|uniform|w,w,w,w
  --detail                 Show per-chain cosine breakdown in compare mode
  --compare TEXT1 TEXT2     Compare two texts (cosine + hamming)
  --benchmark N            Run N embeddings and report timing
  --benchmark-compare N    Run N comparisons and report timing

Stage-2 options:
  --stage2                 Use Stage-2 encoding (no .trine model needed)
  --model PATH             Load a trained .trine2 model (requires --stage2)
  --depth N                Cascade depth for Stage-2 (default: 4)
  --block-diagonal         Use block-diagonal projection (random, no trained model)
  --s2-random SEED         Use random projection/cascade (for experimentation)
```

### `trine_dedup`

Near-duplicate detection pipeline with JSONL ingest and JSON output. Supports blended Stage-1 + Stage-2 scoring.

```
Usage: trine_dedup <command> [options]

Commands:
  check <text1> <text2>    Compare two texts
  scan < input.txt         Stream dedup (stdin)
  batch -f <file>          Batch dedup with index
  build-index -f <file>    Alias for batch
  stats -f <file>          Corpus duplicate density analysis

Options:
  -t, --threshold <float>  Similarity threshold (default: 0.60)
  -l, --lens <name>        Select lens preset
  --json                   Machine-readable JSON output
  --input-format <fmt>     Input: plain, jsonl, auto (default: auto)
  --save-index <path>      Save index to .tridx file
  --load-index <path>      Load index from .tridx file
  --routed                 Use Band-LSH routing for faster dedup
  --recall <mode>          fast | balanced (default) | strict

Stage-2 options:
  --semantic PATH          Load Stage-2 .trine2 model for semantic dedup
  --s2-model PATH          Alias for --semantic
  --s2-depth N             Cascade depth for Stage-2 (default: 0)
  --blend ALPHA            Blend factor: alpha*S1 + (1-alpha)*S2 (default: 0.65)
  --s2-only                Use Stage-2 similarity only (no S1 blending)
```

**JSONL input format:** Each line is `{"id": "...", "text": "...", "meta": {...}}`. The `text` field is encoded; `id` becomes the index tag. Auto-detected if first line starts with `{`.

### `trine_train`

Hebbian Stage-2 training CLI. Trains a ternary projection model from JSONL text pairs, with optional validation and model persistence.

```
Usage: trine_train [options] <data.jsonl>
       trine_train [options] --data <data.jsonl> [--val <val.jsonl>]

Options:
  --data PATH              Training data (JSONL with text_a, text_b, score)
  --val PATH               Validation data (same format)
  --epochs N               Number of training epochs (default: 10)
  --density D              Target density for auto-threshold (default: 0.33)
  --similarity-threshold S Stage-1 similarity threshold (default: 0.5)
  --depth N                Cascade depth (default: 4)
  --deepen N               Self-supervised deepening rounds (default: 0)
  --diagonal               Use diagonal gating instead of full projection
  --block-diagonal         Use block-diagonal projection (4 x 60x60 blocks per chain)
  --sparse K               Sparse cross-channel projection (top-K per row)
  --stacked                Stacked depth (re-apply projection instead of cascade)
  --save PATH              Save frozen model to .trine2 binary file
  --load PATH              Load a saved .trine2 model (skip training)
  --save-accum PATH        Save accumulator state to .trine2a file
  --load-accum PATH        Load accumulator state for warm-start training
```

**Best known configuration:**

```bash
./build/trine_train \
  --data data/splits/train.jsonl \
  --val data/splits/val.jsonl \
  --epochs 1 --diagonal \
  --density 0.15 --similarity-threshold 0.90 --depth 0 \
  --save model.trine2
```

### `trine_bench`

Throughput benchmarks for all pipeline stages.

```
Usage: trine_bench [options]

Options:
  --quick          Run reduced-scale benchmarks
  --encode-only    Only run encode benchmarks
  --route-only     Only run routing benchmarks
  --stage2-only    Only run Stage-2 pipeline benchmarks
```

### `trine_recall`

Routing recall validation and bucket health diagnostics.

```
Usage: trine_recall [options]

Options:
  --validate       Run recall validation with verdict
  --health         Bucket health analysis (histogram)
  --json           Machine-readable JSON output
  --size N         Corpus size (default: 10000)
  --queries N      Number of test queries (default: 1000)
  --recall <mode>  fast | balanced | strict
```

---

## API Reference

### Stage-1 API (`trine_stage1.h`)

The primary interface for encoding, comparison, indexing, and packed storage.

**Encoding:**

```c
int trine_s1_encode(const char *text, size_t len, uint8_t out[240]);
int trine_s1_encode_batch(const char **texts, const size_t *lens, int count, uint8_t *out);
```

**Comparison:**

```c
float trine_s1_compare(const uint8_t a[240], const uint8_t b[240], const trine_s1_lens_t *lens);
trine_s1_result_t trine_s1_check(const uint8_t candidate[240], const uint8_t reference[240],
                                  const trine_s1_config_t *config);
int trine_s1_compare_batch(const uint8_t candidate[240], const uint8_t *refs, int ref_count,
                            const trine_s1_config_t *config, float *best_sim);
```

**Index (in-memory, batch dedup):**

```c
trine_s1_index_t *trine_s1_index_create(const trine_s1_config_t *config);
int               trine_s1_index_add(trine_s1_index_t *idx, const uint8_t emb[240], const char *tag);
trine_s1_result_t trine_s1_index_query(const trine_s1_index_t *idx, const uint8_t candidate[240]);
int               trine_s1_index_count(const trine_s1_index_t *idx);
void              trine_s1_index_free(trine_s1_index_t *idx);
int               trine_s1_index_save(const trine_s1_index_t *idx, const char *path);
trine_s1_index_t *trine_s1_index_load(const char *path);
```

**Packed trit storage (240 trits to 48 bytes, 5x reduction):**

```c
int   trine_s1_pack(const uint8_t trits[240], uint8_t packed[48]);
int   trine_s1_unpack(const uint8_t packed[48], uint8_t trits[240]);
float trine_s1_compare_packed(const uint8_t a[48], const uint8_t b[48], const trine_s1_lens_t *lens);
```

### Routed Index API (`trine_route.h`)

Band-LSH routing overlay for sublinear query on large corpora. Each of the 4 chains is hashed to a bucket via FNV-1a. Queries only compare entries sharing at least one bucket key, with multi-probe for near-miss recovery.

- 10-50x fewer comparisons per query
- 97%+ recall at cosine >= 0.8

```c
trine_route_t    *trine_route_create(const trine_s1_config_t *config);
int               trine_route_add(trine_route_t *rt, const uint8_t emb[240], const char *tag);
trine_s1_result_t trine_route_query(const trine_route_t *rt, const uint8_t candidate[240],
                                     trine_route_stats_t *stats);
int               trine_route_count(const trine_route_t *rt);
void              trine_route_free(trine_route_t *rt);
int               trine_route_save(const trine_route_t *rt, const char *path);
trine_route_t    *trine_route_load(const char *path);
```

### Stage-2 API (`trine_stage2.h`)

Semantic embedding layer. The forward pass (encode + project + cascade) is zero-float. Comparison uses float via the Stage-1 lens system.

**Lifecycle:**

```c
trine_s2_model_t *trine_s2_create_identity(void);     /* pass-through model       */
trine_s2_model_t *trine_s2_create_random(uint32_t n_cells, uint64_t seed);
trine_s2_model_t *trine_s2_create_from_parts(const void *proj, uint32_t n_cells, uint64_t topo_seed);
void              trine_s2_free(trine_s2_model_t *model);
```

**Forward pass (zero-float):**

```c
int trine_s2_encode(const trine_s2_model_t *model,
                     const char *text, size_t len,
                     uint32_t depth, uint8_t out[240]);

int trine_s2_encode_from_trits(const trine_s2_model_t *model,
                                const uint8_t stage1[240],
                                uint32_t depth, uint8_t out[240]);

int trine_s2_encode_depths(const trine_s2_model_t *model,
                            const char *text, size_t len,
                            uint32_t max_depth, uint8_t *out);
```

**Comparison:**

```c
float trine_s2_compare(const uint8_t a[240], const uint8_t b[240], const void *lens);

/* Gate-aware comparison: skips channels where diagonal gates are zero */
float trine_s2_compare_gated(const trine_s2_model_t *model,
                              const uint8_t a[240], const uint8_t b[240]);

/* Per-chain blend: alpha[c]*S1 + (1-alpha[c])*S2 per chain */
float trine_s2_compare_chain_blend(const uint8_t s1_a[240], const uint8_t s1_b[240],
                                    const uint8_t s2_a[240], const uint8_t s2_b[240],
                                    const float alpha[4]);
```

**Adaptive blend alpha:**

```c
/* Compute adaptive blend alpha from model statistics.
 * Returns alpha in [0, 1] tuned to the model's trained density and gate distribution. */
float trine_s2_adaptive_alpha(const trine_s2_model_t *model);
```

**Introspection & Configuration:**

```c
int  trine_s2_info(const trine_s2_model_t *model, trine_s2_info_t *info);
void trine_s2_set_projection_mode(trine_s2_model_t *model, int mode);
int  trine_s2_get_projection_mode(const trine_s2_model_t *model);
void trine_s2_set_stacked_depth(trine_s2_model_t *model, int enable);
int  trine_s2_get_stacked_depth(const trine_s2_model_t *model);
```

### Stage-2 Persistence API (`trine_s2_persist.h`)

Save and load trained Stage-2 models in the `.trine2` binary format.

```c
int                    trine_s2_save(const trine_s2_model_t *model, const char *path,
                                      const trine_s2_save_config_t *config);
trine_s2_model_t      *trine_s2_load(const char *path);
int                    trine_s2_validate(const char *path);
```

---

## Build

```bash
make              # Build library + all tools (Stage-1 + Stage-2)
make libtrine.a   # Static library only
make test         # Build + run 434 C tests (Stage-1 + Stage-2)
make test_s1      # 163 Stage-1 tests
make test_s2      # 271 Stage-2 tests (all phases)
make test_s2_p3   # 64 Stage-2 Phase 3 tests (projection/cascade/pipeline)
make test_s2_p4   # 78 Stage-2 Phase 4 tests (hebbian/freeze/self-deepen)
make test_s2_p7   # 17 Stage-2 Phase 7 tests (persistence)
make bench        # Build + run throughput benchmarks
make clean        # Remove build artifacts
```

Requirements: C99 compiler, POSIX environment. No external dependencies beyond libc and libm.

---

## Properties

- **240 dimensions** -- 4 chains x 60 channels each
- **Deterministic** -- same input always produces the same embedding
- **Zero external dependencies** -- only libc + libm
- **Zero floats in Stage-2 inference** -- projection and cascade are pure integer
- **Embeddable** -- static library (`libtrine.a`) includes Stage-1 + Stage-2
- **Packed storage** -- 240 trits pack to 48 bytes (5x reduction)
- **Learned projection** -- Hebbian-trained diagonal or block-diagonal gating, 173 KB model
- **Block-diagonal mode** -- 4 x 60x60 blocks capture cross-channel correlations within chains (43,200 parameters)

---

## File Formats

### `.tridx` (v2) -- Stage-1 Index

Serialized linear-scan index with FNV-1a integrity checksum. Magic: `TRS1`. Includes endianness marker for cross-platform safety.

### `.trrt` (v3) -- Routed Index

Band-LSH routed index with bucket topology, recall mode, and FNV-1a checksum. Magic: `TRRT`. Backward compatible with v1 files.

### `.trine` (v1) -- Cascade Model

Stage-1 cascade model. 64-byte header + ROM (snap arena) + snap state. Used by `trine_embed` for cascade-based embedding at screening/standard/deep resolutions.

### `.trine2` (v1) -- Stage-2 Model

Trained Stage-2 projection model. 72-byte packed header + K x DIM x DIM weights + 8-byte FNV-1a payload checksum. For block-diagonal models, the weight layout is K x 4 x 60 x 60.

```
Offset  Field                  Size
------  -----                  ----
0       magic ("TR2\0")        4 bytes
4       version                4 bytes
8       flags                  4 bytes (bit 0 = diagonal, bit 1 = block-diagonal)
12      projection_k           4 bytes (3)
16      projection_dim         4 bytes (240)
20      cascade_cells          4 bytes
24      cascade_depth          4 bytes
28      topo_seed              8 bytes
36      similarity_threshold   4 bytes (float)
40      density                4 bytes (float)
44      reserved               20 bytes
64      header_checksum        8 bytes (FNV-1a)
72      projection weights     K * DIM * DIM = 172,800 bytes
172872  payload_checksum       8 bytes (FNV-1a)
```

Total: 172,880 bytes for K=3, DIM=240.

### `.trine2a` (v1) -- Accumulator State

Hebbian accumulator state for warm-start training, curriculum learning, and incremental updates. Contains the full K x DIM x DIM int32 counter matrices.

```
Offset  Field                  Size
------  -----                  ----
0       magic ("TR2A")         4 bytes
4       version                4 bytes
8       flags                  4 bytes (bit 0 = diagonal, bit 1 = block-diagonal)
12      projection_k           4 bytes (3)
16      projection_dim         4 bytes (240)
20      pairs_observed         4 bytes (uint32_t)
24      similarity_threshold   4 bytes (float)
28      freeze_target_density  4 bytes (float)
32      reserved               24 bytes
56      header_checksum        8 bytes (FNV-1a)
64      accumulators           K * DIM * DIM * 4 = 691,200 bytes
691264  payload_checksum       8 bytes (FNV-1a)
```

Total: 691,272 bytes (~675 KB) for K=3, DIM=240.

---

## Format Safety

All index and model formats include production hardening:

- **Endianness marker** -- rejects files written on incompatible architectures
- **Feature flags** -- reserved field for forward compatibility
- **FNV-1a checksum** -- 64-bit payload integrity verification on load
- **Version gating** -- unknown future versions are rejected with clear error
- **Backward compatibility** -- older format versions still load (checksum skipped where absent)

## Thread Safety

- **Encoding** (`trine_s1_encode`, `trine_s2_encode`) is stateless and fully thread-safe.
- **Stage-2 model** (`trine_s2_model_t`) is immutable after construction and thread-safe for concurrent encoding.
- **Comparison** functions are pure and thread-safe.
- **Index operations** (add/query on `trine_s1_index_t` and `trine_route_t`) are not thread-safe. Callers must synchronize access to shared index instances.

---

## File Structure

```
trine/
├── SPEC.md                              # Technical specification
├── CLAUDE.md                            # Claude Code project context
├── Makefile                             # Build orchestration
├── README.md                            # This file
│
├── src/                                 # All source code
│   ├── encode/                          # Stage-1 shingle encoder
│   │   ├── trine_encode.c/h
│   │   └── trine_idf.h
│   ├── compare/                         # Comparison & lenses
│   │   ├── trine_stage1.c/h
│   │   └── trine_csidf.c/h
│   ├── index/                           # Indexing & routing
│   │   ├── trine_route.c/h
│   │   └── trine_field.c/h
│   ├── canon/                           # Canonicalization
│   │   └── trine_canon.c/h
│   ├── algebra/                         # Ternary algebraic core
│   │   ├── oicos.h
│   │   ├── trine_algebra.h
│   │   └── trine_format.c/h
│   ├── model/                           # Cascade model runtime
│   │   └── trine.c/h
│   ├── pack/                            # Packed trit storage
│   │   └── trine_pack.c
│   ├── stage2/                          # Stage-2 semantic layer
│   │   ├── projection/                  # K=3 majority-vote ternary matmul
│   │   │   ├── trine_project.c/h
│   │   │   └── trine_majority.c
│   │   ├── cascade/                     # ENDO-based mixing network
│   │   │   ├── trine_learned_cascade.c/h
│   │   │   └── trine_topology_gen.c
│   │   ├── inference/                   # Unified forward pass API
│   │   │   └── trine_stage2.c/h
│   │   ├── hebbian/                     # Hebbian training harness
│   │   │   ├── trine_hebbian.c/h
│   │   │   ├── trine_accumulator.c/h
│   │   │   ├── trine_freeze.c/h
│   │   │   └── trine_self_deepen.c
│   │   └── persist/                     # .trine2 / .trine2a binary formats
│   │       ├── trine_s2_persist.c/h
│   │       └── trine_accumulator_persist.c/h
│   └── tools/                           # CLI programs
│       ├── trine_embed.c
│       ├── trine_dedup.c
│       ├── trine_train.c
│       ├── trine_bench.c
│       ├── trine_recall.c
│       └── trine_test_sim.c
│
├── tests/                               # Test suite
│   └── stage2/                          # Stage-2 unit & integration tests
│       ├── test_projection.c
│       ├── test_majority.c
│       ├── test_cascade.c
│       ├── test_full_pipeline.c
│       ├── test_hebbian.c
│       ├── test_freeze.c
│       ├── test_self_deepen.c
│       └── test_persistence.c
│
├── bindings/                            # Language bindings
│   ├── python/                          # pytrine (ctypes FFI)
│   │   ├── pytrine/
│   │   │   ├── trine.py                 # Stage-1 API
│   │   │   ├── stage2.py               # Stage-2 API
│   │   │   ├── langchain.py            # LangChain integration
│   │   │   └── llamaindex.py           # LlamaIndex integration
│   │   └── tests/
│   └── rust/                            # trine crate (FFI)
│       ├── src/
│       │   ├── lib.rs                   # Stage-1 API
│       │   ├── stage2.rs               # Stage-2 API
│       │   ├── ffi.rs                  # C FFI declarations
│       │   └── tests.rs
│       └── Cargo.toml
│
├── deploy/                              # Deployment
│   ├── trinedb/                         # REST server
│   │   ├── trinedb.c
│   │   └── test_trinedb.sh
│   └── docker/                          # Container
│       ├── Dockerfile
│       └── docker-compose.yml
│
├── data/                                # Datasets (7 datasets, 174K pairs)
│   ├── download/                        # Download scripts
│   ├── raw/                             # Downloaded, unprocessed
│   ├── prepared/                        # Standardized JSONL
│   └── splits/                          # Train/val/test partitions
│
├── bench/                               # Benchmark framework
│   ├── harness/                         # Benchmark runners
│   └── reports/                         # Results & analysis
│
└── build/
    ├── libtrine.a                       # Static library (S1 + S2)
    ├── trine_embed                      # CLI embed tool
    ├── trine_dedup                      # CLI dedup tool
    ├── trine_train                      # CLI training tool
    └── trine_bench                      # CLI benchmark tool
```

---

## License

MIT License -- See LICENSE file.
