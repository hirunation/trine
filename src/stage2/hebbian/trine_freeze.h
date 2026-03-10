/* =====================================================================
 * TRINE Stage-2 — Freeze (Quantization)
 * =====================================================================
 *
 * Converts int16 Hebbian accumulators to ternary projection weights.
 *
 * Freeze rule (SPEC 2.4):
 *   W[k][i][j] = (counter[k][i][j] > +T)  ? 2
 *              : (counter[k][i][j] < -T)  ? 1
 *              : 0
 *
 * T (threshold) controls sparsity: larger T yields sparser weights.
 * Auto-threshold uses binary search to find T achieving target density.
 *
 * ===================================================================== */

#ifndef TRINE_FREEZE_H
#define TRINE_FREEZE_H

#include <stdint.h>
#include "trine_accumulator.h"

typedef struct { uint8_t W[3][240][240]; } trine_projection_t;

/* Freeze accumulators to ternary projection weights.
 * For each k, i, j:
 *   W[k][i][j] = (counter[k][i][j] > +threshold)  ? 2
 *              : (counter[k][i][j] < -threshold)  ? 1
 *              : 0
 *
 * threshold must be >= 0. If 0, all non-zero counters become 1 or 2. */
void trine_freeze_projection(const trine_accumulator_t *acc,
                              int32_t threshold,
                              trine_projection_t *out);

/* Find threshold that achieves a target weight density.
 * Uses binary search over [1, max_abs]. Returns the first threshold
 * where the fraction of non-zero weights drops to or below target_density.
 * target_density in (0.0, 1.0), e.g. 0.33 means ~1/3 of weights non-zero.
 * Returns the found threshold, or max_abs if density never drops low enough. */
int32_t trine_freeze_auto_threshold(const trine_accumulator_t *acc,
                                     float target_density);

/* Sparse freeze: for each output row, keep only the top-K entries by
 * absolute counter magnitude. All others are set to 0.
 * This enables cross-channel mixing while avoiding noise from weak entries.
 * top_k must be >= 1 and <= 240. Values outside range are clamped. */
void trine_freeze_sparse(const trine_accumulator_t *acc,
                           uint32_t top_k,
                           trine_projection_t *out);

/* Statistics about a frozen projection. */
typedef struct {
    uint32_t n_zero;       /* W[k][i][j] == 0 */
    uint32_t n_one;        /* W[k][i][j] == 1 */
    uint32_t n_two;        /* W[k][i][j] == 2 */
    float    density;      /* (n_one + n_two) / total */
} trine_freeze_stats_t;

void trine_freeze_stats(const trine_projection_t *proj,
                         trine_freeze_stats_t *stats);

/* =====================================================================
 * Block-Diagonal Freeze (v1.0.3)
 * =====================================================================
 *
 * Freezes block-diagonal accumulators (4 x 60x60 per copy) to ternary
 * weights.  Output layout matches accumulator: K * 4 * 60 * 60 uint8_t.
 * ===================================================================== */

/* Freeze block-diagonal accumulators to ternary weights.
 * Output: W_blocks[K * 4 * 60 * 60] ternary weights.
 * threshold: counters > +T become 2, < -T become 1, else 0. */
int trine_freeze_block(const trine_block_accumulator_t *acc,
                       int32_t threshold,
                       uint8_t *W_blocks);

/* Auto-threshold for target density */
int32_t trine_freeze_block_auto_threshold(const trine_block_accumulator_t *acc,
                                           float target_density);

#endif /* TRINE_FREEZE_H */
