/* ═══════════════════════════════════════════════════════════════════════
 * TRINE — Ternary Resonance Interference Network Embedding
 * Batch Compare API — Cache-Friendly Block Processing
 * ═══════════════════════════════════════════════════════════════════════
 *
 * OVERVIEW
 *   High-throughput one-vs-many comparison using block processing for
 *   cache-friendly memory access. The query vector is loaded once into
 *   L1 cache, then compared against blocks of 16 corpus vectors at a
 *   time. This keeps the working set (query + 16 corpus vectors =
 *   240 + 16*240 = 4080 bytes) well within typical 32-64 KB L1d.
 *
 * COMPARISON
 *   Uses uniform cosine similarity across all 240 dimensions (no lens
 *   weighting). For lens-weighted comparison, use trine_s1_compare or
 *   trine_s1_compare_batch from trine_stage1.h.
 *
 * THREAD SAFETY
 *   All functions are stateless and thread-safe. Multiple threads may
 *   call these functions concurrently with different query/corpus data.
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef TRINE_BATCH_COMPARE_H
#define TRINE_BATCH_COMPARE_H

#include <stdint.h>
#include <stddef.h>

/* Block size for cache-friendly processing. 16 vectors * 240 bytes =
 * 3840 bytes, plus 240 bytes for the query = 4080 bytes total working
 * set, fitting comfortably in L1 data cache. */
#define TRINE_BATCH_BLOCK_SIZE 16

/* Compare one query vector against n stored vectors, writing results
 * to sims[].
 * Uses block processing for cache-friendly access.
 * query:  240-byte query vector
 * corpus: n * 240 byte contiguous embedding array
 * n:      number of corpus vectors
 * sims:   output array of n floats
 * Returns 0 on success. */
int trine_batch_compare(
    const uint8_t *query,
    const uint8_t *corpus,
    size_t n,
    float *sims);

/* Same but returns top-k indices and similarities.
 * top_k_idx: output array of top_k indices (sorted by sim, descending)
 * top_k_sim: output array of top_k similarities
 * Returns number of results written (<= top_k). */
size_t trine_batch_compare_topk(
    const uint8_t *query,
    const uint8_t *corpus,
    size_t n,
    size_t top_k,
    size_t *top_k_idx,
    float *top_k_sim);

#endif /* TRINE_BATCH_COMPARE_H */
