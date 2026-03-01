/* =====================================================================
 * TRINE CS-IDF — Corpus-Specific Inverse Document Frequency Weighting
 * Ternary Resonance Interference Network Embedding
 * =====================================================================
 *
 * OVERVIEW
 *   Computes per-channel document-frequency weights from an actual corpus
 *   at index build time, replacing the static/generic IDF weights in
 *   trine_idf.h. Channels that fire for many documents (boilerplate)
 *   get downweighted; rare-activation channels stay high-weight.
 *
 * ALGORITHM
 *   For each document embedding added:
 *     For each channel i in 0..239:
 *       if embedding[i] != 0: channel_df[i]++
 *
 *   After all docs:
 *     N = doc_count
 *     For each channel i:
 *       raw_idf[i] = log2f((float)N / (1.0f + (float)channel_df[i]))
 *     Normalize: weights[i] = raw_idf[i] / max(raw_idf)
 *     Clamp minimum to 0.01 (never fully zero-weight a channel)
 *
 * PROPERTIES
 *   - Deterministic: same corpus always produces identical weights
 *   - Compact: 1924 bytes total (4 + 240*4 + 240*4)
 *   - Zero dependencies beyond libc + libm
 *   - Weights are NOT stored in the embedding — applied at comparison time
 *
 * ===================================================================== */

#ifndef TRINE_CSIDF_H
#define TRINE_CSIDF_H

#include <stdint.h>

#define TRINE_CSIDF_DIMS       240
#define TRINE_CSIDF_MIN_WEIGHT 0.01f  /* Floor: never zero-weight a channel */

/* Corpus-specific IDF state */
typedef struct {
    uint32_t doc_count;                      /* Total documents observed */
    uint32_t channel_df[TRINE_CSIDF_DIMS];   /* Per-channel document frequency */
    float    weights[TRINE_CSIDF_DIMS];      /* Computed IDF weights (normalized) */
    int      computed;                        /* 1 if weights have been computed */
} trine_csidf_t;

/* Initialize a CS-IDF tracker. Zeroes all counters and weights. */
void trine_csidf_init(trine_csidf_t *csidf);

/* Observe a document embedding: increment DF for each non-zero channel.
 * Call this for every document added to the index during build. */
void trine_csidf_observe(trine_csidf_t *csidf, const uint8_t emb[240]);

/* Compute IDF weights from accumulated DF counters.
 * Must be called after all observations, before using weights for comparison.
 * Safe to call multiple times (e.g., after appending new documents).
 * Returns 0 on success, -1 if doc_count is 0. */
int trine_csidf_compute(trine_csidf_t *csidf);

/* Merge another CS-IDF tracker into this one (for append mode).
 * Adds the other's doc_count and channel_df to this one.
 * Does NOT recompute weights — call trine_csidf_compute() after merge.
 * Returns 0 on success, -1 on error. */
int trine_csidf_merge(trine_csidf_t *dst, const trine_csidf_t *src);

/* CS-IDF weighted cosine similarity over full 240 dims.
 * Formula: sum(w[i]*a[i]*b[i]) / (sqrt(sum(w[i]*a[i]^2)) * sqrt(sum(w[i]*b[i]^2)))
 * Returns 0.0 if either vector has zero weighted magnitude.
 * Requires weights to have been computed (csidf->computed == 1). */
float trine_csidf_cosine(const uint8_t a[240], const uint8_t b[240],
                          const trine_csidf_t *csidf);

/* CS-IDF + lens weighted cosine over full 240 dims.
 * Per-chain cosine with IDF weighting, combined via lens weights.
 * lens[4] = per-chain weights (chains with weight <= 0 are skipped). */
float trine_csidf_cosine_lens(const uint8_t a[240], const uint8_t b[240],
                               const trine_csidf_t *csidf,
                               const float lens[4]);

/* Serialize CS-IDF state to a file stream.
 * Writes: doc_count(4) + channel_df(240*4) + weights(240*4) = 1924 bytes.
 * Returns 0 on success, -1 on error. */
int trine_csidf_write(const trine_csidf_t *csidf, void *fp);

/* Deserialize CS-IDF state from a file stream.
 * Reads 1924 bytes. Sets computed = 1.
 * Returns 0 on success, -1 on error. */
int trine_csidf_read(trine_csidf_t *csidf, void *fp);

/* Get the serialized size of a CS-IDF section in bytes. */
#define TRINE_CSIDF_SERIAL_SIZE (4 + TRINE_CSIDF_DIMS * 4 + TRINE_CSIDF_DIMS * 4)

#endif /* TRINE_CSIDF_H */
