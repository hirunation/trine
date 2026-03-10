/* =====================================================================
 * TRINE Stage-2 — Hebbian Accumulator
 * =====================================================================
 *
 * Int16 counter matrices for Hebbian weight accumulation.
 *
 * Maintains K=3 copies of 240x240 int16 counters.  For each training
 * pair (a, b) with Stage-1 similarity s1, the Hebbian update rule is:
 *
 *   counter[k][i][j] += sign * a[j] * b[i]
 *
 * where sign = +1 for similar pairs (s1 > threshold) and -1 for
 * dissimilar pairs.  After accumulation, counters are frozen to
 * ternary projection weights via trine_freeze.
 *
 * Total storage: 3 x 240 x 240 x 4 = 691,200 bytes (~675 KB).
 * Accumulation uses saturating arithmetic (INT32_MIN..INT32_MAX).
 *
 * ===================================================================== */

#ifndef TRINE_ACCUMULATOR_H
#define TRINE_ACCUMULATOR_H

#include <stdint.h>
#include <stddef.h>

#define TRINE_ACC_K    3     /* Projection copies (matches TRINE_PROJECT_K) */
#define TRINE_ACC_DIM  240   /* Dimensionality */

typedef struct trine_accumulator trine_accumulator_t;

/* Create accumulator: K copies of DIM x DIM int16 matrices.
 * All counters initialized to zero. Returns NULL on alloc failure. */
trine_accumulator_t *trine_accumulator_create(void);

/* Free accumulator. Safe to call with NULL. */
void trine_accumulator_free(trine_accumulator_t *acc);

/* Hebbian update for a training pair.
 * For each projection k, for each (i, j):
 *   counter[k][i][j] += sign * (int16_t)(a[j] * b[i])
 *
 * sign = +1 for similar pairs (s1 > threshold)
 * sign = -1 for dissimilar pairs (s1 <= threshold)
 *
 * a and b are 240-trit Stage-1 embeddings (values in {0,1,2}).
 * Accumulation saturates at INT16_MIN/INT16_MAX. */
void trine_accumulator_update(trine_accumulator_t *acc,
                               const uint8_t a[240], const uint8_t b[240],
                               int sign);

/* Weighted Hebbian update.  Same outer-product rule as update(), but
 * scales each delta by an integer magnitude factor.
 *   delta = sign * magnitude * centered_a[j] * centered_b[i]
 * magnitude must be >= 1 (clamped if less). */
void trine_accumulator_update_weighted(trine_accumulator_t *acc,
                                        const uint8_t a[240], const uint8_t b[240],
                                        int sign, int32_t magnitude);

/* Get pointer to counter matrix for projection k (0..K-1).
 * Returns pointer to DIM x DIM int32 array (row-major).
 * Caller must not free. Returns NULL if k >= K or acc is NULL. */
int32_t (*trine_accumulator_counters(trine_accumulator_t *acc, uint32_t k))[TRINE_ACC_DIM];

/* Get pointer to counter matrix (const version). */
const int32_t (*trine_accumulator_counters_const(const trine_accumulator_t *acc, uint32_t k))[TRINE_ACC_DIM];

/* Reset all counters to zero. Also resets total_updates. */
void trine_accumulator_reset(trine_accumulator_t *acc);

/* Statistics about current accumulator state. */
typedef struct {
    int64_t  total_updates;    /* Number of update() calls */
    int32_t  max_abs;          /* Max |counter| across all K x DIM x DIM */
    uint32_t n_positive;       /* Counters > 0 */
    uint32_t n_negative;       /* Counters < 0 */
    uint32_t n_zero;           /* Counters == 0 */
    uint32_t n_saturated;      /* Counters at INT32_MIN or INT32_MAX */
} trine_accumulator_stats_t;

void trine_accumulator_stats(const trine_accumulator_t *acc,
                              trine_accumulator_stats_t *stats);

/* =====================================================================
 * Block-Diagonal Accumulator (v1.0.3)
 * =====================================================================
 *
 * Accumulates into 4 independent 60x60 blocks (one per chain) instead
 * of a full 240x240 matrix.  This matches the block-diagonal projection
 * structure where chains do not cross-correlate.
 *
 * Counter layout: counters[k * 4 * 60 * 60 + c * 60 * 60 + i * 60 + j]
 * Total size: K * 4 * 60 * 60 * sizeof(int32_t) = K * 57,600 bytes.
 * ===================================================================== */

#define TRINE_ACCUM_MODE_FULL     0  /* Full 240x240 matrix */
#define TRINE_ACCUM_MODE_DIAGONAL 1  /* 240-element diagonal */
#define TRINE_ACCUM_MODE_BLOCK    2  /* 4 x 60x60 block-diagonal */

#define TRINE_BLOCK_CHAINS  4
#define TRINE_BLOCK_DIM    60   /* Dims per chain = 240 / 4 */

/* Block-diagonal accumulator: K copies of 4 x 60x60 counter matrices.
 * Total size: K * 4 * 60 * 60 * sizeof(int32_t) = K * 57,600 bytes */
typedef struct {
    int32_t *counters;      /* K * 4 * 60 * 60 counters */
    int K;                  /* Number of projection copies (typically 3) */
    int mode;               /* TRINE_ACCUM_MODE_BLOCK */
    uint64_t total_updates;
    uint32_t pairs_observed;
} trine_block_accumulator_t;

trine_block_accumulator_t *trine_block_accumulator_create(int K);
void trine_block_accumulator_free(trine_block_accumulator_t *acc);

/* Hebbian update for block-diagonal: for each chain c, accumulate
 * outer product of (a[c*60..c*60+59] - 1) and (b[c*60..c*60+59] - 1)
 * scaled by sign(similarity - threshold). Only cross-correlates within each chain. */
void trine_block_accumulator_update(
    trine_block_accumulator_t *acc,
    const uint8_t a[240], const uint8_t b[240],
    int sign);

/* Weighted variant with magnitude scaling */
void trine_block_accumulator_update_weighted(
    trine_block_accumulator_t *acc,
    const uint8_t a[240], const uint8_t b[240],
    int sign, int magnitude);

/* Reset all counters to zero */
void trine_block_accumulator_reset(trine_block_accumulator_t *acc);

/* Statistics */
void trine_block_accumulator_stats(const trine_block_accumulator_t *acc,
                                    int32_t *max_val, int32_t *min_val,
                                    uint64_t *nonzero);

#endif /* TRINE_ACCUMULATOR_H */
