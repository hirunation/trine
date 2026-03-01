/* ═══════════════════════════════════════════════════════════════════════
 * TRINE — Ternary Resonance Interference Network Embedding
 * Stage-1 Retriever/Deduper API v1.0.1
 * ═══════════════════════════════════════════════════════════════════════
 *
 * OVERVIEW
 *   Fast, deterministic near-duplicate detection and candidate retrieval.
 *   All operations use TRINE_SHINGLE (240-dim) embeddings only.
 *
 *   This layer wraps the shingle encoder (~4M embeddings/sec) with
 *   lens-weighted comparison, length-aware calibration, and a simple
 *   in-memory index for batch deduplication workflows.
 *
 * DESIGN
 *   Stage-1 is a lightweight pre-filter, not a vector database. The
 *   index uses linear scan (O(N) per query), which is sufficient for
 *   thousands of entries. For larger corpora, Stage-1 results feed
 *   into a proper ANN index (Stage-2, not yet implemented).
 *
 * THREAD SAFETY
 *   Encoding (trine_s1_encode) is stateless and thread-safe.
 *   Comparison functions are pure and thread-safe.
 *   Index operations (add/query) are NOT thread-safe; callers must
 *   synchronize access to a shared index.
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef TRINE_STAGE1_H
#define TRINE_STAGE1_H

#include <stdint.h>
#include <stddef.h>

#define TRINE_S1_VERSION "1.0.1"
#define TRINE_S1_DIMS 240
#define TRINE_S1_CHAINS 4
#define TRINE_S1_CHAIN_WIDTH 60
#define TRINE_S1_PACKED_SIZE 48  /* 240 trits / 5 trits per byte */
#define TRINE_S1_INDEX_MAGIC "TRS1"

/* Lens weights for comparison (same semantics as trine_lens_t) */
typedef struct {
    float weights[TRINE_S1_CHAINS];
} trine_s1_lens_t;

/* Predefined lens presets */
#define TRINE_S1_LENS_UNIFORM {{1.0f, 1.0f, 1.0f, 1.0f}}
#define TRINE_S1_LENS_DEDUP   {{0.5f, 0.5f, 0.7f, 1.0f}}
#define TRINE_S1_LENS_EDIT    {{1.0f, 0.3f, 0.1f, 0.0f}}
#define TRINE_S1_LENS_VOCAB   {{0.0f, 0.2f, 0.3f, 1.0f}}
#define TRINE_S1_LENS_CODE    {{1.0f, 0.8f, 0.4f, 0.2f}}
#define TRINE_S1_LENS_LEGAL   {{0.2f, 0.4f, 1.0f, 0.8f}}
#define TRINE_S1_LENS_MEDICAL {{0.3f, 1.0f, 0.6f, 0.5f}}
#define TRINE_S1_LENS_SUPPORT {{0.2f, 0.4f, 0.7f, 1.0f}}
#define TRINE_S1_LENS_POLICY  {{0.1f, 0.3f, 1.0f, 0.8f}}

/* Configuration for dedup operations */
typedef struct {
    float threshold;           /* Cosine threshold for "duplicate" (0.0-1.0) */
    trine_s1_lens_t lens;      /* Which lens to use for comparison */
    int calibrate_length;      /* 1 = apply length-aware calibration */
} trine_s1_config_t;

/* Default config: threshold=0.60, DEDUP lens, length calibration on */
#define TRINE_S1_CONFIG_DEFAULT { \
    .threshold = 0.60f, \
    .lens = TRINE_S1_LENS_DEDUP, \
    .calibrate_length = 1 \
}

/* Result of a single dedup check */
typedef struct {
    int is_duplicate;          /* 1 if similarity >= threshold */
    float similarity;          /* Raw lens-weighted cosine */
    float calibrated;          /* Length-calibrated score (if enabled) */
    int matched_index;         /* Index of best match (for batch), -1 if none */
} trine_s1_result_t;

/* In-memory embedding index for batch dedup */
typedef struct trine_s1_index trine_s1_index_t;

/* ═══════════════════════════════════════════════════════════════════════
 * Encoding API
 * ═══════════════════════════════════════════════════════════════════════ */

/* Encode text directly to a 240-byte shingle embedding.
 * This is a convenience wrapper — no model file needed.
 * Returns 0 on success. */
int trine_s1_encode(const char *text, size_t len, uint8_t out[240]);

/* Batch encode: encode N texts at once.
 * texts[i] points to text, lens[i] is the length.
 * out must be pre-allocated: N * 240 bytes.
 * Returns 0 on success. */
int trine_s1_encode_batch(const char **texts, const size_t *lens,
                           int count, uint8_t *out);

/* ═══════════════════════════════════════════════════════════════════════
 * Comparison API
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compare two 240-byte embeddings with lens weighting.
 * Returns lens-weighted cosine similarity (0.0-1.0), or -1.0 on error. */
float trine_s1_compare(const uint8_t a[240], const uint8_t b[240],
                        const trine_s1_lens_t *lens);

/* Full dedup check: encode + compare + threshold + calibration. */
trine_s1_result_t trine_s1_check(const uint8_t candidate[240],
                                  const uint8_t reference[240],
                                  const trine_s1_config_t *config);

/* Batch compare: compare one candidate against N references.
 * Returns the index of the best match, or -1 if none above threshold.
 * If best_sim is non-NULL, writes the best similarity score. */
int trine_s1_compare_batch(const uint8_t candidate[240],
                            const uint8_t *refs, int ref_count,
                            const trine_s1_config_t *config,
                            float *best_sim);

/* ═══════════════════════════════════════════════════════════════════════
 * Index API (in-memory, for batch dedup)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Create an empty index. Returns NULL on allocation failure. */
trine_s1_index_t *trine_s1_index_create(const trine_s1_config_t *config);

/* Add an embedding to the index. Returns the assigned index (0-based).
 * Returns -1 on allocation failure. Optional metadata tag (can be NULL). */
int trine_s1_index_add(trine_s1_index_t *idx, const uint8_t emb[240],
                        const char *tag);

/* Query: find best match in the index for a candidate.
 * Returns result with matched_index set to best match (-1 if no match above threshold). */
trine_s1_result_t trine_s1_index_query(const trine_s1_index_t *idx,
                                        const uint8_t candidate[240]);

/* Get the number of entries in the index. */
int trine_s1_index_count(const trine_s1_index_t *idx);

/* Get the tag for an entry (may be NULL). */
const char *trine_s1_index_tag(const trine_s1_index_t *idx, int index);

/* Free the index and all entries. */
void trine_s1_index_free(trine_s1_index_t *idx);

/* Save index to binary file. Format (v2):
 *   Magic:        "TRS1" (4 bytes)
 *   Version:      uint32_t = 2
 *   Endian check: uint32_t = 0x01020304 (byte order marker)
 *   Flags:        uint32_t = 0 (reserved for future use)
 *   Checksum:     uint64_t = FNV-1a over payload after header
 *   --- payload ---
 *   Count:        uint32_t
 *   Config:       threshold (float), lens weights (4 floats), calibrate_length (int32)
 *   For each entry:
 *     Embedding: 240 bytes (unpacked trits)
 *     Tag len:   uint32_t (0 if no tag)
 *     Tag:       tag_len bytes (no null terminator)
 *
 * v1 files (without endian/flags/checksum) are still loadable.
 * Returns 0 on success, -1 on error. */
int trine_s1_index_save(const trine_s1_index_t *idx, const char *path);

/* Load index from binary file. Creates a new index.
 * Returns NULL on error (message printed to stderr). */
trine_s1_index_t *trine_s1_index_load(const char *path);

/* ═══════════════════════════════════════════════════════════════════════
 * Packed Trit Storage (5 trits/byte, 240 → 48 bytes)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Pack 240 trits into 48 bytes. Each byte encodes 5 trits as:
 * byte = t0 + t1*3 + t2*9 + t3*27 + t4*81  (range 0-242)
 * Returns 0 on success. */
int trine_s1_pack(const uint8_t trits[240], uint8_t packed[48]);

/* Unpack 48 bytes back to 240 trits. Returns 0 on success. */
int trine_s1_unpack(const uint8_t packed[48], uint8_t trits[240]);

/* Compare two packed embeddings without unpacking.
 * Unpacks internally, computes lens-weighted cosine.
 * Slightly slower than comparing unpacked, but saves memory in storage. */
float trine_s1_compare_packed(const uint8_t a[48], const uint8_t b[48],
                               const trine_s1_lens_t *lens);

/* ═══════════════════════════════════════════════════════════════════════
 * Length Calibration
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compute fill ratio for a 240-byte embedding (fraction of non-zero channels).
 * Returns value in [0.0, 1.0]. */
float trine_s1_fill_ratio(const uint8_t emb[240]);

/* Calibrate raw cosine similarity based on fill ratios of both embeddings.
 * Adjusts for the fact that sparse embeddings (short texts) have lower
 * baseline cosine than dense embeddings (long texts).
 * Formula: calibrated = raw_cosine / sqrt(fill_a * fill_b)
 * Clamped to [0.0, 1.0]. */
float trine_s1_calibrate(float raw_cosine, float fill_a, float fill_b);

#endif /* TRINE_STAGE1_H */
