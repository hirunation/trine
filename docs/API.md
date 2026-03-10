# TRINE C API Reference

TRINE v1.0.3 -- Public C API. 240-dimensional ternary text fingerprinting engine.

---

## Table of Contents

- [Error Codes](#error-codes)
- [Encoding](#encoding)
- [Comparison](#comparison)
- [Packed Trit Storage](#packed-trit-storage)
- [Length Calibration](#length-calibration)
- [SIMD](#simd)
- [Batch Operations](#batch-operations)
- [Canonicalization](#canonicalization)
- [CS-IDF Weighting](#cs-idf-weighting)
- [Indexing](#indexing)
- [Field-Aware Indexing](#field-aware-indexing)
- [Stage-2 Projection](#stage-2-projection)
- [Stage-2 Inference](#stage-2-inference)
- [Hebbian Training](#hebbian-training)
- [Hebbian Accumulator](#hebbian-accumulator)
- [Persistence](#persistence)
- [2-bit Trit Packing](#2-bit-trit-packing)

---

## Error Codes

**Header:** `src/encode/trine_error.h`

```c
typedef enum {
    TRINE_OK            =  0,
    TRINE_ERR_NULL      = -1,
    TRINE_ERR_ALLOC     = -2,
    TRINE_ERR_IO        = -3,
    TRINE_ERR_FORMAT    = -4,
    TRINE_ERR_CHECKSUM  = -5,
    TRINE_ERR_VERSION   = -6,
    TRINE_ERR_RANGE     = -7,
    TRINE_ERR_CAPACITY  = -8,
    TRINE_ERR_CORRUPT   = -9,
    TRINE_ERR_CONFIG    = -10
} trine_error_t;
```

| Function | Signature | Description | Thread-safe |
|----------|-----------|-------------|-------------|
| `trine_strerror` | `const char *trine_strerror(int code)` | Return human-readable error description for a `trine_error_t` code. | Yes (static strings) |

---

## Encoding

**Header:** `src/encode/trine_encode.h`

240 channels = 4 chains x 60. Values in {0, 1, 2}.

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_encode` | `void trine_encode(const char *text, size_t len, uint8_t channels[240])` | Position-based encoding, max 12 chars. Truncates with overflow hash. | void | Yes |
| `trine_decode` | `int trine_decode(const uint8_t channels[240], char *text, size_t max_len)` | Decode 240 trits back to text from forward chain. | Chars decoded; 0 on error, -1 if truncated | Yes |
| `trine_encode_info` | `void trine_encode_info(const char *text, size_t len, trine_encode_info_t *info)` | Compute encoding metadata without performing encoding. | void | Yes |
| `trine_encode_shingle` | `int trine_encode_shingle(const char *text, size_t len, uint8_t channels[240])` | Multi-scale n-gram shingling encoder. Arbitrary-length, case-insensitive. | 0 on success, -1 on OOM | Yes |
| `trine_encode_shingle_batch` | `int trine_encode_shingle_batch(const char *const *texts, const size_t *lens, size_t n, uint8_t *out)` | Batch shingle encode n texts. Reuses internal casefold buffer. `out` must be n*240 bytes. | 0 on success, -1 on OOM | Yes |

---

## Comparison

**Header:** `src/compare/trine_stage1.h`

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_s1_encode` | `int trine_s1_encode(const char *text, size_t len, uint8_t out[240])` | Convenience wrapper: text to 240-byte shingle embedding. | 0 on success | Yes |
| `trine_s1_encode_batch` | `int trine_s1_encode_batch(const char **texts, const size_t *lens, int count, uint8_t *out)` | Batch encode N texts. `out` must be N*240 bytes. | 0 on success | Yes |
| `trine_s1_compare` | `float trine_s1_compare(const uint8_t a[240], const uint8_t b[240], const trine_s1_lens_t *lens)` | Lens-weighted cosine similarity between two embeddings. | Similarity in [0.0, 1.0], -1.0 on error | Yes |
| `trine_s1_check` | `trine_s1_result_t trine_s1_check(const uint8_t candidate[240], const uint8_t reference[240], const trine_s1_config_t *config)` | Full dedup check: compare + threshold + length calibration. | `trine_s1_result_t` with is_duplicate, similarity, calibrated | Yes |
| `trine_s1_compare_batch` | `int trine_s1_compare_batch(const uint8_t candidate[240], const uint8_t *refs, int ref_count, const trine_s1_config_t *config, float *best_sim)` | Compare one candidate against N references. | Index of best match above threshold, -1 if none | Yes |

---

## Packed Trit Storage

**Header:** `src/compare/trine_stage1.h`

5 trits per byte: 240 trits pack into 48 bytes.

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_s1_pack` | `int trine_s1_pack(const uint8_t trits[240], uint8_t packed[48])` | Pack 240 trits into 48 bytes (base-243 encoding). | 0 on success | Yes |
| `trine_s1_unpack` | `int trine_s1_unpack(const uint8_t packed[48], uint8_t trits[240])` | Unpack 48 bytes back to 240 trits. | 0 on success | Yes |
| `trine_s1_compare_packed` | `float trine_s1_compare_packed(const uint8_t a[48], const uint8_t b[48], const trine_s1_lens_t *lens)` | Compare two packed embeddings (unpacks internally). | Lens-weighted cosine | Yes |

---

## Length Calibration

**Header:** `src/compare/trine_stage1.h`

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_s1_fill_ratio` | `float trine_s1_fill_ratio(const uint8_t emb[240])` | Fraction of non-zero channels in an embedding. | Value in [0.0, 1.0] | Yes |
| `trine_s1_calibrate` | `float trine_s1_calibrate(float raw_cosine, float fill_a, float fill_b)` | Adjust raw cosine by fill ratios: `raw / sqrt(fill_a * fill_b)`, clamped to [0.0, 1.0]. | Calibrated similarity | Yes |

---

## SIMD

**Header:** `src/compare/trine_simd.h`

SSE2-accelerated centered ternary operations. Maps {0,1,2} to {-1,0,+1}.

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_simd_available` | `int trine_simd_available(void)` | Check if SSE2 was compiled in. | 1 if available, 0 otherwise | Yes |
| `trine_simd_dot_sse2` | `int trine_simd_dot_sse2(const uint8_t *a, const uint8_t *b, int len)` | Centered dot product: sum((a[i]-1)*(b[i]-1)). | Integer dot product | Yes |
| `trine_simd_norm2_sse2` | `int trine_simd_norm2_sse2(const uint8_t *a, int len)` | Centered squared norm: sum((a[i]-1)^2). | Integer norm squared | Yes |
| `trine_simd_cosine_sse2` | `float trine_simd_cosine_sse2(const uint8_t *a, const uint8_t *b, int len)` | Centered cosine similarity. Returns 0.0 if either norm is zero. | Cosine similarity | Yes |
| `trine_simd_selftest` | `int trine_simd_selftest(void)` | Verify SSE2 results match scalar reference. | 0 on pass, -1 on failure | Yes |

---

## Batch Operations

**Header:** `src/compare/trine_batch_compare.h`

Cache-friendly one-vs-many comparison (block size 16, uniform cosine, no lens weighting).

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_batch_compare` | `int trine_batch_compare(const uint8_t *query, const uint8_t *corpus, size_t n, float *sims)` | Compare query against n corpus vectors, write all similarities. | 0 on success | Yes |
| `trine_batch_compare_topk` | `size_t trine_batch_compare_topk(const uint8_t *query, const uint8_t *corpus, size_t n, size_t top_k, size_t *top_k_idx, float *top_k_sim)` | Compare query against n corpus vectors, return top-k results sorted descending. | Number of results written (<= top_k) | Yes |

---

## Canonicalization

**Header:** `src/canon/trine_canon.h`

Deterministic text normalization applied before encoding. No malloc, no regex.

**Presets:** `TRINE_CANON_NONE` (0), `TRINE_CANON_SUPPORT` (1), `TRINE_CANON_CODE` (2), `TRINE_CANON_POLICY` (3), `TRINE_CANON_GENERAL` (4).

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_canon_apply` | `int trine_canon_apply(const char *text, size_t len, int preset, char *out, size_t out_cap, size_t *out_len)` | Apply a canonicalization preset to text. Output null-terminated. | 0 on success, -1 if out_cap insufficient | Yes |
| `trine_canon_preset_name` | `const char *trine_canon_preset_name(int preset)` | Human-readable preset name. Returns "UNKNOWN" for invalid values. | Static string pointer | Yes |
| `trine_canon_normalize_whitespace` | `void trine_canon_normalize_whitespace(char *buf, size_t *len)` | Collapse whitespace runs to single space, trim edges. In-place. | void | Yes |
| `trine_canon_strip_timestamps` | `void trine_canon_strip_timestamps(char *buf, size_t *len)` | Remove ISO-8601 dates, HH:MM:SS times, 10-digit Unix timestamps. In-place. | void | Yes |
| `trine_canon_strip_uuids` | `void trine_canon_strip_uuids(char *buf, size_t *len)` | Remove UUID patterns (8-4-4-4-12 hex). In-place. | void | Yes |
| `trine_canon_normalize_identifiers` | `void trine_canon_normalize_identifiers(char *buf, size_t *len)` | Normalize camelCase/snake_case to lowercase words. In-place. | void | Yes |
| `trine_canon_bucket_numbers` | `void trine_canon_bucket_numbers(char *buf, size_t *len)` | Replace digit runs with `<N>` placeholder. In-place. | void | Yes |

---

## CS-IDF Weighting

**Header:** `src/compare/trine_csidf.h`

Corpus-specific inverse document frequency. Per-channel weights computed from actual corpus statistics.

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_csidf_init` | `void trine_csidf_init(trine_csidf_t *csidf)` | Initialize tracker. Zeroes all counters and weights. | void | Yes |
| `trine_csidf_observe` | `void trine_csidf_observe(trine_csidf_t *csidf, const uint8_t emb[240])` | Increment DF for each non-zero channel. Call per document. | void | No (mutates state) |
| `trine_csidf_compute` | `int trine_csidf_compute(trine_csidf_t *csidf)` | Compute normalized IDF weights from accumulated DF counters. | 0 on success, -1 if doc_count is 0 | No (mutates state) |
| `trine_csidf_merge` | `int trine_csidf_merge(trine_csidf_t *dst, const trine_csidf_t *src)` | Merge another tracker into this one (for append mode). Does not recompute weights. | 0 on success, -1 on error | No (mutates dst) |
| `trine_csidf_cosine` | `float trine_csidf_cosine(const uint8_t a[240], const uint8_t b[240], const trine_csidf_t *csidf)` | CS-IDF weighted cosine similarity. Requires `csidf->computed == 1`. | Cosine similarity, 0.0 if zero magnitude | Yes (if computed) |
| `trine_csidf_cosine_lens` | `float trine_csidf_cosine_lens(const uint8_t a[240], const uint8_t b[240], const trine_csidf_t *csidf, const float lens[4])` | Per-chain IDF-weighted cosine combined via lens weights. | Cosine similarity | Yes (if computed) |
| `trine_csidf_write` | `int trine_csidf_write(const trine_csidf_t *csidf, void *fp)` | Serialize CS-IDF state to FILE stream (1924 bytes). | 0 on success, -1 on error | Yes |
| `trine_csidf_read` | `int trine_csidf_read(trine_csidf_t *csidf, void *fp)` | Deserialize CS-IDF state from FILE stream. Sets `computed = 1`. | 0 on success, -1 on error | No (mutates state) |

---

## Indexing

**Header:** `src/index/trine_route.h`

Band-LSH routing overlay for sub-linear query. 4 bands, multi-probe, recall presets.

**Recall presets:** `TRINE_RECALL_FAST` (0), `TRINE_RECALL_BALANCED` (1), `TRINE_RECALL_STRICT` (2).

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_route_create` | `trine_route_t *trine_route_create(const trine_s1_config_t *config)` | Create a routed index with given comparison config. | Pointer to index, NULL on failure | N/A |
| `trine_route_add` | `int trine_route_add(trine_route_t *rt, const uint8_t emb[240], const char *tag)` | Add an embedding with optional tag. | Assigned 0-based index, -1 on error | No |
| `trine_route_query` | `trine_s1_result_t trine_route_query(const trine_route_t *rt, const uint8_t candidate[240], trine_route_stats_t *stats)` | Find best match via LSH routing. `stats` may be NULL. | `trine_s1_result_t` | No |
| `trine_route_count` | `int trine_route_count(const trine_route_t *rt)` | Number of entries in the index. | Entry count | No |
| `trine_route_tag` | `const char *trine_route_tag(const trine_route_t *rt, int index)` | Get tag for an entry (may be NULL). | Tag string pointer | No |
| `trine_route_embedding` | `const uint8_t *trine_route_embedding(const trine_route_t *rt, int index)` | Get raw 240-byte embedding for an entry. | Pointer to embedding | No |
| `trine_route_free` | `void trine_route_free(trine_route_t *rt)` | Free the index and all entries. | void | No |
| `trine_route_save` | `int trine_route_save(const trine_route_t *rt, const char *path)` | Save index to .trrt binary file (v3 format). | 0 on success, -1 on error | No |
| `trine_route_load` | `trine_route_t *trine_route_load(const char *path)` | Load index from .trrt file. Supports v1/v2/v3. | Pointer to index, NULL on error | N/A |
| `trine_route_save_atomic` | `int trine_route_save_atomic(const trine_route_t *rt, const char *path)` | Atomic save via temp+fsync+rename. NULL path uses load path. | 0 on success, -1 on error | No |
| `trine_route_global_stats` | `void trine_route_global_stats(const trine_route_t *rt, trine_route_stats_t *stats)` | Global bucket statistics for the index. | void | No |
| `trine_route_set_recall` | `int trine_route_set_recall(trine_route_t *rt, int mode)` | Set recall preset (FAST/BALANCED/STRICT). Takes effect on next query. | 0 on success, -1 on invalid mode | No |
| `trine_route_get_recall` | `int trine_route_get_recall(const trine_route_t *rt)` | Get current recall mode. | Recall mode constant | No |
| `trine_route_bucket_sizes` | `int trine_route_bucket_sizes(const trine_route_t *rt, int band, int *sizes)` | Get per-bucket occupancy for a band. `sizes` must hold TRINE_ROUTE_BUCKETS ints. | 0 on success, -1 on invalid args | No |
| `trine_route_enable_csidf` | `int trine_route_enable_csidf(trine_route_t *rt)` | Enable CS-IDF tracking. Must be called before adding entries. | 0 on success, -1 on error | No |
| `trine_route_compute_csidf` | `int trine_route_compute_csidf(trine_route_t *rt)` | Compute CS-IDF weights from accumulated DF. Call after all adds. | 0 on success, -1 on error | No |
| `trine_route_get_csidf` | `const trine_csidf_t *trine_route_get_csidf(const trine_route_t *rt)` | Get computed CS-IDF weights (internal pointer, do not free). | Pointer or NULL | No |
| `trine_route_query_csidf` | `trine_s1_result_t trine_route_query_csidf(const trine_route_t *rt, const uint8_t candidate[240], trine_route_stats_t *stats)` | Query using CS-IDF weighted comparison. | `trine_s1_result_t` | No |

---

## Field-Aware Indexing

**Headers:** `src/index/trine_field.h`, `src/index/trine_route.h`

Multi-field documents with per-field embeddings and weighted scoring. Max 4 fields.

**Field presets:** `TRINE_FIELD_PRESET_CODE` (0), `TRINE_FIELD_PRESET_SUPPORT` (1), `TRINE_FIELD_PRESET_POLICY` (2).

### Field Configuration (`trine_field.h`)

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_field_config_init` | `void trine_field_config_init(trine_field_config_t *cfg)` | Initialize config: 3 fields (title/body/code), uniform weights, route on body. | void | Yes |
| `trine_field_config_preset` | `int trine_field_config_preset(trine_field_config_t *cfg, int preset)` | Apply a domain preset (CODE/SUPPORT/POLICY). | 0 on success, -1 on invalid preset | Yes |
| `trine_field_config_parse_fields` | `int trine_field_config_parse_fields(trine_field_config_t *cfg, const char *spec)` | Parse field spec string (e.g. "title,body,code" or "auto"). | 0 on success, -1 on error | Yes |
| `trine_field_config_parse_weights` | `int trine_field_config_parse_weights(trine_field_config_t *cfg, const char *spec)` | Parse weight spec (e.g. "title=1.0,body=0.8"). | 0 on success, -1 on error | Yes |
| `trine_field_config_write` | `int trine_field_config_write(const trine_field_config_t *cfg, void *fp)` | Serialize field config to FILE stream (152 bytes). | 0 on success, -1 on error | Yes |
| `trine_field_config_read` | `int trine_field_config_read(trine_field_config_t *cfg, void *fp)` | Deserialize field config from FILE stream. | 0 on success, -1 on error | No (mutates cfg) |

### Field Operations (`trine_field.h`)

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_field_encode` | `int trine_field_encode(const trine_field_config_t *cfg, const char **texts, const size_t *lens, trine_field_entry_t *out)` | Encode multi-field document. NULL/empty fields produce zero embeddings. | 0 on success | Yes |
| `trine_field_route_embedding` | `const uint8_t *trine_field_route_embedding(const trine_field_config_t *cfg, const trine_field_entry_t *entry)` | Get the routing field's 240-byte embedding. | Pointer to embedding | Yes |
| `trine_field_cosine` | `float trine_field_cosine(const trine_field_entry_t *a, const trine_field_entry_t *b, const trine_field_config_t *cfg)` | Field-weighted cosine: sum(w_f * cos(a_f, b_f)) / sum(w_f). | Weighted similarity | Yes |
| `trine_field_cosine_idf` | `float trine_field_cosine_idf(const trine_field_entry_t *a, const trine_field_entry_t *b, const trine_field_config_t *cfg, const float idf_weights[240])` | Field-weighted cosine with per-channel IDF. | Weighted similarity | Yes |
| `trine_field_extract_jsonl` | `int trine_field_extract_jsonl(const char *jsonl_buf, size_t jsonl_len, const trine_field_config_t *cfg, const char **out_texts, size_t *out_lens, char **out_tag)` | Extract fields from a JSONL line. `out_tag` must be freed by caller. | Number of fields found | Yes |

### Field-Aware Routing (`trine_route.h`)

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_route_enable_fields` | `int trine_route_enable_fields(trine_route_t *rt, const trine_field_config_t *fcfg)` | Enable field-aware mode. Must be called before adding entries. | 0 on success, -1 on error | No |
| `trine_route_add_fields` | `int trine_route_add_fields(trine_route_t *rt, const trine_field_entry_t *entry, const char *tag)` | Add a field-aware entry. Routes on primary field. | Assigned index, -1 on error | No |
| `trine_route_query_fields` | `trine_s1_result_t trine_route_query_fields(const trine_route_t *rt, const trine_field_entry_t *query, trine_route_stats_t *stats)` | Query with field-weighted scoring on routed candidates. | `trine_s1_result_t` | No |
| `trine_route_field_entry` | `const trine_field_entry_t *trine_route_field_entry(const trine_route_t *rt, int index)` | Get field entry for an index entry. | Pointer or NULL if fields not enabled | No |
| `trine_route_field_config` | `const trine_field_config_t *trine_route_field_config(const trine_route_t *rt)` | Get the field config for this index. | Pointer or NULL if fields not enabled | No |

---

## Stage-2 Projection

**Header:** `src/stage2/projection/trine_project.h`

K=3 majority-vote ternary matmul. All integer, zero float.

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_project_single` | `void trine_project_single(const uint8_t W[240][240], const uint8_t x[240], uint8_t y[240])` | Single-matrix ternary matmul: y[i] = (sum_j W[i][j]*x[j]) % 3. | void | Yes |
| `trine_project_majority` | `void trine_project_majority(const trine_projection_t *proj, const uint8_t x[240], uint8_t y[240])` | K=3 majority-vote projection. Tie-break: first projection wins. | void | Yes |
| `trine_projection_identity` | `void trine_projection_identity(trine_projection_t *proj)` | Initialize all K matrices to identity (pass-through). | void | Yes |
| `trine_projection_random` | `void trine_projection_random(trine_projection_t *proj, uint64_t seed)` | Fill all K matrices with pseudorandom trits (deterministic by seed). | void | Yes |
| `trine_project_single_sign` | `void trine_project_single_sign(const uint8_t W[240][240], const uint8_t x[240], uint8_t y[240])` | Sign-based projection: center to {-1,0,+1}, dot product, quantize by sign. | void | Yes |
| `trine_project_majority_sign` | `void trine_project_majority_sign(const trine_projection_t *proj, const uint8_t x[240], uint8_t y[240])` | K=3 majority-vote with sign-based projection. | void | Yes |
| `trine_project_diagonal_gate` | `void trine_project_diagonal_gate(const trine_projection_t *proj, const uint8_t x[240], uint8_t y[240])` | Diagonal gating: W[i][i]=2 keep, =1 flip, =0 zero. K=3 majority vote. | void | Yes |
| `trine_project_single_sparse_sign` | `void trine_project_single_sparse_sign(const uint8_t W[240][240], const uint8_t x[240], uint8_t y[240])` | Sparse sign projection: W=0 entries skipped, W=2 is +1, W=1 is -1. | void | Yes |
| `trine_project_majority_sparse_sign` | `void trine_project_majority_sparse_sign(const trine_projection_t *proj, const uint8_t x[240], uint8_t y[240])` | K=3 majority-vote with sparse sign projection. | void | Yes |
| `trine_project_block_diagonal` | `void trine_project_block_diagonal(const uint8_t W_block[4][60][60], const uint8_t x[240], uint8_t y[240])` | Block-diagonal: 4 independent 60x60 ternary matmuls (one per chain). | void | Yes |
| `trine_projection_majority_block` | `void trine_projection_majority_block(const uint8_t *W_blocks, int K, const uint8_t x[240], uint8_t y[240])` | K-way majority vote over block-diagonal projections. `W_blocks`: K*4*60*60 bytes. | void | Yes |
| `trine_projection_block_identity` | `void trine_projection_block_identity(uint8_t *W_blocks, int K)` | Initialize K block-diagonal projections as identity. | void | Yes |
| `trine_projection_block_random` | `void trine_projection_block_random(uint8_t *W_blocks, int K, uint64_t seed)` | Initialize K block-diagonal projections with pseudorandom trits. | void | Yes |

---

## Stage-2 Inference

**Header:** `src/stage2/inference/trine_stage2.h`

Forward pass: Stage-1 encode -> projection -> cascade. Inference path is zero float.

**Projection modes:** `TRINE_S2_PROJ_SIGN` (0), `TRINE_S2_PROJ_DIAGONAL` (1), `TRINE_S2_PROJ_SPARSE` (2), `TRINE_S2_PROJ_BLOCK_DIAG` (3).

### Lifecycle

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_s2_create_identity` | `trine_s2_model_t *trine_s2_create_identity(void)` | Create identity model (Stage-1 pass-through). | Pointer to model, NULL on failure | N/A |
| `trine_s2_create_random` | `trine_s2_model_t *trine_s2_create_random(uint32_t n_cells, uint64_t seed)` | Create random model with given cascade cells and seed. | Pointer to model, NULL on failure | N/A |
| `trine_s2_create_from_parts` | `trine_s2_model_t *trine_s2_create_from_parts(const void *proj, uint32_t n_cells, uint64_t topo_seed)` | Create model from pre-trained projection weights. | Pointer to model, NULL on failure | N/A |
| `trine_s2_create_block_diagonal` | `trine_s2_model_t *trine_s2_create_block_diagonal(const uint8_t *block_weights, int K, uint32_t n_cells, uint64_t topo_seed)` | Create model with block-diagonal projection (K * 4 * 60 * 60 bytes). | Pointer to model, NULL on failure | N/A |
| `trine_s2_free` | `void trine_s2_free(trine_s2_model_t *model)` | Free model and all internal structures. | void | N/A |

### Encoding

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_s2_encode` | `int trine_s2_encode(const trine_s2_model_t *model, const char *text, size_t len, uint32_t depth, uint8_t out[240])` | Full pipeline: shingle encode -> projection -> cascade at given depth. | 0 on success, -1 on error | Yes |
| `trine_s2_encode_from_trits` | `int trine_s2_encode_from_trits(const trine_s2_model_t *model, const uint8_t stage1[240], uint32_t depth, uint8_t out[240])` | Project + cascade from pre-computed Stage-1 trits. | 0 on success, -1 on error | Yes |
| `trine_s2_encode_depths` | `int trine_s2_encode_depths(const trine_s2_model_t *model, const char *text, size_t len, uint32_t max_depth, uint8_t *out, size_t out_size)` | Extract embedding at every depth 0..max_depth-1. `out`: max_depth*240 bytes. | 0 on success, -1 on error | Yes |
| `trine_s2_encode_batch` | `int trine_s2_encode_batch(const trine_s2_model_t *model, const char *const *texts, const size_t *lens, size_t n, int depth, uint8_t *out)` | Batch Stage-2 encode n texts. `out`: n*240 bytes. | 0 on success, -1 on error | Yes |

### Comparison

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_s2_compare` | `float trine_s2_compare(const uint8_t a[240], const uint8_t b[240], const void *lens)` | Compare two Stage-2 embeddings. `lens` may be NULL for uniform weights. | Similarity in [0.0, 1.0], -1.0 on error | Yes |
| `trine_s2_compare_gated` | `float trine_s2_compare_gated(const trine_s2_model_t *model, const uint8_t a[240], const uint8_t b[240])` | Gated compare: only channels with active diagonal gates contribute. | Similarity in [-1.0, 1.0], 0.0 if no active channels | Yes |
| `trine_s2_compare_chain_blend` | `float trine_s2_compare_chain_blend(const uint8_t s1_a[240], const uint8_t s1_b[240], const uint8_t s2_a[240], const uint8_t s2_b[240], const float alpha[4])` | Per-chain alpha blend of S1 and S2 similarities. alpha=1.0 is pure S1, alpha=0.0 is pure S2. | Blended similarity | Yes |
| `trine_s2_compare_adaptive_blend` | `float trine_s2_compare_adaptive_blend(const trine_s2_model_t *model, const uint8_t s1_a[240], const uint8_t s1_b[240], const uint8_t s2_a[240], const uint8_t s2_b[240])` | Adaptive blend: alpha selected from S1 similarity bucket lookup. | Blended similarity, 0.0 if not configured | Yes |

### Introspection

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_s2_info` | `int trine_s2_info(const trine_s2_model_t *model, trine_s2_info_t *info)` | Query model parameters (K, dims, cells, depth, identity flag). | 0 on success, -1 on null model | Yes |
| `trine_s2_set_projection_mode` | `void trine_s2_set_projection_mode(trine_s2_model_t *model, int mode)` | Set projection mode (sign/diagonal/sparse/block-diagonal). | void | No |
| `trine_s2_get_projection_mode` | `int trine_s2_get_projection_mode(const trine_s2_model_t *model)` | Get current projection mode. | Mode constant, -1 on null | Yes |
| `trine_s2_set_stacked_depth` | `void trine_s2_set_stacked_depth(trine_s2_model_t *model, int enable)` | Enable/disable stacked depth (re-project instead of cascade ticks). | void | No |
| `trine_s2_get_stacked_depth` | `int trine_s2_get_stacked_depth(const trine_s2_model_t *model)` | Check if stacked depth is enabled. | 1 if enabled, 0 otherwise | Yes |
| `trine_s2_set_adaptive_alpha` | `void trine_s2_set_adaptive_alpha(trine_s2_model_t *model, const float buckets[10])` | Set per-S1-bucket alpha values. Pass NULL to disable. | void | No |
| `trine_s2_get_projection` | `const void *trine_s2_get_projection(const trine_s2_model_t *model)` | Get read-only pointer to full projection weights. | Pointer, NULL on null model | Yes |
| `trine_s2_get_block_projection` | `const uint8_t *trine_s2_get_block_projection(const trine_s2_model_t *model)` | Get read-only pointer to block-diagonal weights (K*4*60*60 bytes). | Pointer, NULL if not block-diagonal | Yes |
| `trine_s2_get_cascade_cells` | `uint32_t trine_s2_get_cascade_cells(const trine_s2_model_t *model)` | Get cascade cell count. | Cell count, 0 on null model | Yes |
| `trine_s2_get_default_depth` | `uint32_t trine_s2_get_default_depth(const trine_s2_model_t *model)` | Get default inference depth. | Depth, 0 on null model | Yes |
| `trine_s2_is_identity` | `int trine_s2_is_identity(const trine_s2_model_t *model)` | Check if model is identity (pass-through). | 1 if identity, 0 otherwise | Yes |

---

## Hebbian Training

**Header:** `src/stage2/hebbian/trine_hebbian.h`

High-level training orchestrator: observe pairs, accumulate, freeze to model.

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_hebbian_create` | `trine_hebbian_state_t *trine_hebbian_create(const trine_hebbian_config_t *config)` | Create training state from config. | Pointer to state, NULL on failure | N/A |
| `trine_hebbian_free` | `void trine_hebbian_free(trine_hebbian_state_t *state)` | Free training state. Safe with NULL. | void | N/A |
| `trine_hebbian_observe` | `void trine_hebbian_observe(trine_hebbian_state_t *state, const uint8_t a[240], const uint8_t b[240], float similarity)` | Observe a training pair (raw trits + similarity). | void | No |
| `trine_hebbian_observe_text` | `void trine_hebbian_observe_text(trine_hebbian_state_t *state, const char *text_a, size_t len_a, const char *text_b, size_t len_b)` | Observe a pair from text (encode + compare + accumulate). | void | No |
| `trine_hebbian_train_file` | `int64_t trine_hebbian_train_file(trine_hebbian_state_t *state, const char *path, uint32_t epochs)` | Train from JSONL file for given number of epochs. | Pairs processed, -1 on error | No |
| `trine_hebbian_freeze` | `struct trine_s2_model *trine_hebbian_freeze(const trine_hebbian_state_t *state)` | Freeze accumulators to a Stage-2 model. Caller owns returned model. | Pointer to model, NULL on failure | No |
| `trine_hebbian_metrics` | `int trine_hebbian_metrics(const trine_hebbian_state_t *state, trine_hebbian_metrics_t *out)` | Get training metrics (pairs, counter stats, density, threshold). | 0 on success | No |
| `trine_hebbian_reset` | `void trine_hebbian_reset(trine_hebbian_state_t *state)` | Reset accumulators to zero, keep config. | void | No |
| `trine_hebbian_get_config` | `trine_hebbian_config_t trine_hebbian_get_config(const trine_hebbian_state_t *state)` | Get a read-only copy of the current config. | Config struct (by value) | No |
| `trine_hebbian_set_threshold` | `void trine_hebbian_set_threshold(trine_hebbian_state_t *state, float threshold)` | Update similarity threshold between epochs. | void | No |
| `trine_hebbian_get_accumulator` | `struct trine_accumulator *trine_hebbian_get_accumulator(trine_hebbian_state_t *state)` | Get internal accumulator pointer (for persistence). Do not free. | Pointer to accumulator | No |
| `trine_self_deepen` | `struct trine_s2_model *trine_self_deepen(trine_hebbian_state_t *state, const char *data_path, uint32_t n_rounds)` | Self-supervised deepening: freeze -> re-encode -> re-accumulate for n rounds. | Final model, NULL on error | No |

---

## Hebbian Accumulator

**Header:** `src/stage2/hebbian/trine_accumulator.h`

Low-level int32 counter matrices for Hebbian weight accumulation.

### Full-Matrix Accumulator (K x 240 x 240)

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_accumulator_create` | `trine_accumulator_t *trine_accumulator_create(void)` | Create K copies of 240x240 int32 matrices, zeroed. | Pointer, NULL on failure | N/A |
| `trine_accumulator_free` | `void trine_accumulator_free(trine_accumulator_t *acc)` | Free accumulator. Safe with NULL. | void | N/A |
| `trine_accumulator_update` | `void trine_accumulator_update(trine_accumulator_t *acc, const uint8_t a[240], const uint8_t b[240], int sign)` | Hebbian update: counter[k][i][j] += sign * a[j] * b[i]. Saturating. | void | No |
| `trine_accumulator_update_weighted` | `void trine_accumulator_update_weighted(trine_accumulator_t *acc, const uint8_t a[240], const uint8_t b[240], int sign, int32_t magnitude)` | Weighted Hebbian update with integer magnitude scaling. | void | No |
| `trine_accumulator_counters` | `int32_t (*trine_accumulator_counters(trine_accumulator_t *acc, uint32_t k))[240]` | Get mutable pointer to counter matrix for projection k. | Pointer, NULL if invalid | No |
| `trine_accumulator_counters_const` | `const int32_t (*trine_accumulator_counters_const(const trine_accumulator_t *acc, uint32_t k))[240]` | Get read-only pointer to counter matrix for projection k. | Pointer, NULL if invalid | Yes |
| `trine_accumulator_reset` | `void trine_accumulator_reset(trine_accumulator_t *acc)` | Reset all counters and total_updates to zero. | void | No |
| `trine_accumulator_stats` | `void trine_accumulator_stats(const trine_accumulator_t *acc, trine_accumulator_stats_t *stats)` | Get statistics: total_updates, max_abs, positive/negative/zero/saturated counts. | void | Yes |

### Block-Diagonal Accumulator (K x 4 x 60 x 60)

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_block_accumulator_create` | `trine_block_accumulator_t *trine_block_accumulator_create(int K)` | Create K copies of 4x60x60 counter matrices. | Pointer, NULL on failure | N/A |
| `trine_block_accumulator_free` | `void trine_block_accumulator_free(trine_block_accumulator_t *acc)` | Free block accumulator. | void | N/A |
| `trine_block_accumulator_update` | `void trine_block_accumulator_update(trine_block_accumulator_t *acc, const uint8_t a[240], const uint8_t b[240], int positive)` | Per-chain outer product update. Only cross-correlates within each chain. | void | No |
| `trine_block_accumulator_update_weighted` | `void trine_block_accumulator_update_weighted(trine_block_accumulator_t *acc, const uint8_t a[240], const uint8_t b[240], int positive, int magnitude)` | Weighted per-chain update with magnitude scaling. | void | No |
| `trine_block_accumulator_reset` | `void trine_block_accumulator_reset(trine_block_accumulator_t *acc)` | Reset all counters to zero. | void | No |
| `trine_block_accumulator_stats` | `void trine_block_accumulator_stats(const trine_block_accumulator_t *acc, int32_t *max_val, int32_t *min_val, uint64_t *nonzero)` | Get min/max counter values and nonzero count. | void | Yes |

---

## Persistence

### Stage-2 Model (.trine2)

**Header:** `src/stage2/persist/trine_s2_persist.h`

Binary format: 72-byte header + projection weights + checksums.

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_s2_save` | `int trine_s2_save(const struct trine_s2_model *model, const char *path, const trine_s2_save_config_t *config)` | Save model to .trine2 file. `config` may be NULL for defaults. | 0 on success, -1 on error | Yes |
| `trine_s2_load` | `struct trine_s2_model *trine_s2_load(const char *path)` | Load model from .trine2 file. Caller must free with `trine_s2_free()`. | Pointer to model, NULL on error | Yes |
| `trine_s2_validate` | `int trine_s2_validate(const char *path)` | Validate a .trine2 file without loading. | 0 if valid, -1 on error | Yes |

### Stage-1 Index (.trrt)

**Header:** `src/compare/trine_stage1.h`

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_s1_index_save` | `int trine_s1_index_save(const trine_s1_index_t *idx, const char *path)` | Save Stage-1 index to binary file (v2 format with checksums). | 0 on success, -1 on error | No |
| `trine_s1_index_load` | `trine_s1_index_t *trine_s1_index_load(const char *path)` | Load Stage-1 index from binary file. Supports v1/v2. | Pointer to index, NULL on error | N/A |

### Accumulator (.trine2a)

**Header:** `src/stage2/persist/trine_accumulator_persist.h`

Binary format for saving/loading Hebbian accumulator state (warm-start, curriculum learning).

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_accumulator_save` | `int trine_accumulator_save(const trine_accumulator_t *acc, const trine_hebbian_config_t *config, int64_t pairs_observed, const char *path)` | Save full accumulator to .trine2a file. | 0 on success, -1 on error | Yes |
| `trine_accumulator_load` | `trine_accumulator_t *trine_accumulator_load(const char *path, trine_hebbian_config_t *config_out, int64_t *pairs_out)` | Load accumulator from .trine2a. Optionally restores config and pair count. | Pointer, NULL on error | N/A |
| `trine_accumulator_validate` | `int trine_accumulator_validate(const char *path)` | Validate a .trine2a file without loading. | 0 if valid, -1 on error | Yes |
| `trine_accumulator_from_frozen` | `trine_accumulator_t *trine_accumulator_from_frozen(const void *projection_weights, int32_t reconstruction_scale)` | Reconstruct approximate accumulators from frozen .trine2 model weights. | Pointer, NULL on error | Yes |
| `trine_block_accumulator_save` | `int trine_block_accumulator_save(const trine_block_accumulator_t *acc, float similarity_threshold, float freeze_target_density, int projection_mode, const char *path)` | Save block-diagonal accumulator to .trine2a. | 0 on success, -1 on error | Yes |
| `trine_block_accumulator_load` | `trine_block_accumulator_t *trine_block_accumulator_load(const char *path, float *threshold_out, float *density_out, uint32_t *pairs_out)` | Load block-diagonal accumulator. Only succeeds if file has block-diag flag. | Pointer, NULL on error | N/A |

---

## 2-bit Trit Packing

**Header:** `src/stage2/persist/trine_pack.h`

Packs ternary weights (0/1/2) into 2 bits each (4 trits per byte, 4x storage reduction).

| Function | Signature | Description | Return | Thread-safe |
|----------|-----------|-------------|--------|-------------|
| `trine_pack_size` | `size_t trine_pack_size(size_t n_trits)` | Compute packed byte count: ceil(n_trits / 4). | Packed size in bytes | Yes |
| `trine_pack_trits` | `size_t trine_pack_trits(const uint8_t *trits, size_t n, uint8_t *packed)` | Pack n trits into 2-bit-per-trit format. Does not validate input. | Bytes written | Yes |
| `trine_unpack_trits` | `void trine_unpack_trits(const uint8_t *packed, size_t n_trits, uint8_t *trits)` | Unpack 2-bit packed bytes back to individual trit bytes. | void | Yes |
| `trine_pack_validate` | `int trine_pack_validate(const uint8_t *packed, size_t n_trits)` | Validate all packed trits are in {0, 1, 2}. | 0 if valid, -1 if any trit > 2 | Yes |
