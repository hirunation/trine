# TRINE

**Ternary Resonance Interference Network Embedding** -- a deterministic, zero-dependency text embedding library in C.

TRINE produces 240-dimensional ternary (0/1/2) fingerprints using multi-scale n-gram shingling across 4 semantic chains. No neural networks, no training data, no external dependencies. The same input always produces the same embedding. It is not a learned model -- it is algebraic hashing with mathematically defined chain structure.

---

## The TRINE Contract

TRINE is a **surface-form similarity engine**. It measures how much text two strings share at the character, morpheme, and word level. This section defines what it promises and what it does not.

### What TRINE is for

- **Near-duplicate detection** -- same content with minor edits, reordering, abbreviation
- **Typo and case normalization** -- "calculateTotal" vs "calculate_total" (code lens: 0.835)
- **Format variant matching** -- "ERROR: timeout after 30 seconds" vs "ERROR: timed out after 30 sec" (dedup lens: 0.709)
- **Template variant detection** -- boilerplate with field substitutions
- **Code identifier matching** -- camelCase/snake_case/kebab-case variants

### What TRINE is NOT for

TRINE does **not** understand meaning. It cannot:

- Match synonyms with different surface forms ("myocardial infarction" vs "heart attack" -- gap: 0.054)
- Match paraphrases ("the cat sat on the mat" vs "a feline rested on the rug")
- Match cross-lingual equivalents
- Understand negation ("is valid" vs "is not valid" may score high)

For semantic similarity, use a neural embedding model (sentence-transformers, OpenAI embeddings, etc.) as a Stage-2 reranker on TRINE's candidate set.

### Expected separation gaps

| Domain | Near-match example | Score | Disjoint example | Score | Gap |
|--------|-------------------|-------|-------------------|-------|-----|
| Code (code lens) | `calculateTotal` vs `calculate_total` | 0.835 | vs "weather is sunny" | 0.463 | **0.372** |
| Dedup (dedup lens) | "db timeout after 30s" variants | 0.709 | vs unrelated text | 0.507 | **0.202** |
| Legal (legal lens) | "governed by laws of NY" variants | 0.519 | vs unrelated | ~0.48 | **~0.04** |
| Medical (medical lens) | "myocardial infarction" vs "heart attack" | 0.536 | vs unrelated | 0.482 | **0.054** |

The medical and legal rows show the honest boundary: when near-duplicates share few surface n-grams, TRINE's separation narrows. This is by design -- TRINE measures surface overlap, not semantic equivalence.

### Recommended 2-stage architecture

```
+---------------------------------------------+
|  Stage 1: TRINE (surface-form filter)       |
|  - 4.7M encodes/sec, 76K QPS @ 10K         |
|  - Cuts candidates from N to ~40            |
|  - 99-100% recall for surface-form matches  |
|  - Cost: ~0 (deterministic, no GPU)         |
+---------------------------------------------+
|  Stage 2: Neural reranker (semantic)        |
|  - Runs on 40 candidates, not 10K           |
|  - Catches synonyms, paraphrase             |
|  - Cost: proportional to candidates (small) |
+---------------------------------------------+
```

This pattern gives you the best of both worlds: TRINE's speed for bulk filtering, neural quality for final ranking, at a fraction of the cost of running neural on the full corpus.

---

## Performance

| Operation | Throughput |
|-----------|-----------|
| Encode (short text) | ~4.7M embeddings/sec |
| Encode (medium text) | ~467K embeddings/sec |
| Compare (lens-weighted) | ~4.1M comparisons/sec |
| Compare (raw cosine) | ~6.9M comparisons/sec |
| Compare (CS-IDF weighted) | ~4.1M comparisons/sec (zero overhead vs static IDF) |
| Compare (field-aware, 2 fields) | ~3.0M comparisons/sec |
| Index insert | ~10.6M inserts/sec |
| Routed query | Sublinear via band-LSH (10-50x speedup over brute force) |

## Properties

- **240 dimensions** -- 4 chains x 60 channels each
- **Deterministic** -- same input always produces the same embedding
- **Zero external dependencies** -- only libc + libm
- **Embeddable** -- static library (`libtrine.a`) is < 100 KB
- **Packed storage** -- 240 trits pack to 48 bytes (5x reduction)

---

## Chains

TRINE encodes text at four structural scales simultaneously. Each chain captures a different axis of textual similarity:

| Chain | Dimensions | Content | Captures |
|-------|-----------|---------|----------|
| Edit | 0-59 | Char unigrams + bigrams | Character-level edits |
| Morph | 60-119 | Trigrams | Morphological structure |
| Phrase | 120-179 | 5-grams | Word-fragment patterns |
| Vocab | 180-239 | Word unigrams | Vocabulary fingerprint |

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

## Quick Start (C API)

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

Compile against the static library:

```bash
cc -O2 -o my_tool my_tool.c -Lbuild -ltrine -lm
```

## Quick Start (CLI)

```bash
# Embed and compare two texts
./build/trine_embed --compare "hello world" "hello there" --resolution shingle --lens dedup --detail

# Scan file for near-duplicates
./build/trine_dedup scan input.txt --threshold 0.7 --lens dedup

# Batch dedup with persistent index
./build/trine_dedup batch input.txt --save-index index.tridx
./build/trine_dedup stats --load-index index.tridx
```

## Quick Start (Python)

```python
from pytrine import TrineEncoder, TrineIndex, Lens

encoder = TrineEncoder()
index = TrineIndex(threshold=0.70, lens=Lens.DEDUP)

index.add("hello world", tag="doc1")
index.add("hello there", tag="doc2")

result = index.query("hello world!")
print(f"Match: {result.tag}, similarity: {result.similarity:.3f}")
```

Install: `pip install pytrine` (or `cd python && make lib && pip install -e .`)

## Quick Start (Rust)

```rust
use trine::{Embedding, Config, Index, Lens};

let emb = Embedding::encode("hello world");
let mut index = Index::new(Config::default())?;
index.add(&emb, Some("doc1"))?;
```

Build: `cd rust && cargo build`

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
cd docker && docker-compose up
# TrineDB available at http://localhost:7319
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

---

## CLI Tools

### `trine_embed`

Encode text to embeddings, compare pairs, and run benchmarks.

```
Usage: trine_embed [options]
  --encode <text>                  Encode and print embedding
  --compare <text1> <text2>        Compare two texts
  --lens <name>                    Select lens preset
  --resolution shingle             Use shingle encoder
  --detail                         Print per-chain breakdown
  --bench                          Run throughput benchmark
```

### `trine_dedup`

Near-duplicate detection pipeline with JSONL ingest and JSON output.

```
Usage: trine_dedup <command> [options]
  check <text1> <text2>            Compare two texts
  scan < input.txt                 Stream dedup (stdin)
  batch -f <file>                  Batch dedup with index
  build-index -f <file>            Alias for batch
  stats -f <file>                  Corpus duplicate density analysis
  --threshold <float>              Similarity threshold (default: 0.60)
  --lens <name>                    Select lens preset
  --json                           Machine-readable JSON output
  --input-format <fmt>             Input: plain, jsonl, auto (default: auto)
  --save-index <path>              Save index to .tridx file
  --load-index <path>              Load index from .tridx file
  --routed                         Use band-LSH routing for faster dedup
  --recall <mode>                  fast | balanced (default) | strict
```

**JSONL input format:** Each line is `{"id": "...", "text": "...", "meta": {...}}`. The `text` field is encoded; `id` becomes the index tag. Auto-detected if first line starts with `{`.

### `trine_recall`

Routing recall validation and bucket health diagnostics.

```
Usage: trine_recall [options]
  --validate                       Run recall validation with verdict
  --health                         Bucket health analysis (histogram)
  --json                           Machine-readable JSON output
  --size N                         Corpus size (default: 10000)
  --queries N                      Number of test queries (default: 1000)
  --recall <mode>                  fast | balanced | strict
```

---

## Build

```bash
make              # Build everything (libtrine.a + all tools)
make libtrine.a   # Static library only
make test         # Build + run 163-test harness
make bench        # Build + run benchmark suite
make clean        # Remove build artifacts
```

Requirements: C99 compiler, POSIX environment. No external dependencies beyond libc and libm.

---

## File Structure

```
hteb/
├── trine_stage1.h    — Stage-1 API (primary interface)
├── trine_stage1.c    — Stage-1 implementation (v2 format w/ checksums)
├── trine_route.h     — Band-LSH routing API + bucket diagnostics
├── trine_route.c     — Routing implementation (v3 format w/ checksums)
├── trine_encode.h    — Shingle encoder API
├── trine_encode.c    — Shingle encoder implementation
├── trine_canon.h     — Canonicalization presets API
├── trine_canon.c     — Deterministic text transforms (whitespace/timestamp/UUID/identifier/number)
├── trine_csidf.h     — Corpus-specific IDF API
├── trine_csidf.c     — CS-IDF weight computation
├── trine_field.h     — Field-aware indexing API
├── trine_field.c     — Per-field embedding and weighted comparison
├── trine_idf.h       — IDF weights (header-only)
├── trine.h           — Full model API (cascade-based)
├── trine.c           — Full model implementation
├── trine_embed.c     — CLI embed/compare tool
├── trine_dedup.c     — CLI dedup tool (JSONL ingest, JSON output)
├── trine_test_sim.c  — 163-test validation harness
├── trine_bench.c     — Benchmark suite
├── trine_recall.c    — Recall validation + bucket health diagnostics
├── Makefile          — Standalone build system
├── examples/         — Example programs
│   ├── basic_embed.c
│   ├── dedup_pipeline.c
│   └── routed_search.c
├── python/           — Python bindings (pytrine)
│   ├── pytrine/      — Package (ctypes FFI)
│   ├── tests/        — 57 tests
│   ├── Makefile
│   └── setup.py
├── rust/             — Rust crate (trine)
│   ├── src/          — FFI bindings + safe wrapper
│   ├── examples/
│   ├── Cargo.toml
│   └── build.rs
├── trinedb/          — REST API server (C + HTTP)
│   ├── trinedb.c     — Embedded HTTP server
│   ├── Makefile
│   └── test_trinedb.sh
├── docker/           — Container deployment
│   ├── Dockerfile
│   └── docker-compose.yml
└── build/
    ├── libtrine.a    — Static library (encode + stage1 + route + canon + csidf + field)
    ├── trine_embed   — CLI embed tool
    ├── trine_dedup   — CLI dedup tool
    └── model.trine   — Pre-built model (562 KB)
```

---

## Format Safety

Both `.tridx` (Stage-1) and `.trrt` (Routed) index formats include production hardening:

- **Endianness marker** -- rejects files written on incompatible architectures
- **Feature flags** -- reserved field for forward compatibility
- **FNV-1a checksum** -- 64-bit payload integrity verification on load
- **Version gating** -- unknown future versions are rejected with clear error
- **Backward compatibility** -- v1 files still load (checksum skipped)

## Thread Safety

- **Encoding** (`trine_s1_encode`) is stateless and fully thread-safe.
- **Comparison** functions are pure and thread-safe.
- **Index operations** (add/query on `trine_s1_index_t` and `trine_route_t`) are not thread-safe. Callers must synchronize access to shared index instances.

---

## License

MIT License -- See LICENSE file.
