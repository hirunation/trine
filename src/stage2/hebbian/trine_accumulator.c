/* =====================================================================
 * TRINE Stage-2 — Hebbian Accumulator (implementation)
 * =====================================================================
 *
 * K=3 copies of 240x240 int32 counter matrices with saturating
 * Hebbian update.  See trine_accumulator.h for API documentation.
 *
 * Storage layout: counters[k][i][j] where k is the projection copy,
 * i is the output (row) dimension, j is the input (column) dimension.
 * Row-major within each matrix.
 *
 * Trits {0,1,2} are centered to {-1,0,+1} before computing the
 * outer product, so it captures correlation (both 0 or both 2)
 * vs anti-correlation (one 0, other 2).  Trit value 1 is neutral.
 * Per-update delta is in {-1, 0, +1}.
 *
 * Int32 counters support up to ~2B updates without saturation.
 *
 * ===================================================================== */

#include "trine_accumulator.h"

#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* --------------------------------------------------------------------- */
/* Internal structure                                                     */
/* --------------------------------------------------------------------- */

struct trine_accumulator {
    int32_t  counters[TRINE_ACC_K][TRINE_ACC_DIM][TRINE_ACC_DIM];
    int64_t  total_updates;
};

/* --------------------------------------------------------------------- */
/* Saturating int32 addition                                              */
/* --------------------------------------------------------------------- */

static inline int32_t sat_add_i32(int32_t a, int32_t b)
{
    int64_t sum = (int64_t)a + (int64_t)b;
    if (sum > INT32_MAX) return INT32_MAX;
    if (sum < INT32_MIN) return INT32_MIN;
    return (int32_t)sum;
}

/* --------------------------------------------------------------------- */
/* Create / free / reset                                                  */
/* --------------------------------------------------------------------- */

trine_accumulator_t *trine_accumulator_create(void)
{
    trine_accumulator_t *acc = calloc(1, sizeof(*acc));
    /* calloc zeros both counters and total_updates */
    return acc;
}

void trine_accumulator_free(trine_accumulator_t *acc)
{
    free(acc);
}

void trine_accumulator_reset(trine_accumulator_t *acc)
{
    if (!acc) return;
    memset(acc->counters, 0, sizeof(acc->counters));
    acc->total_updates = 0;
}

/* --------------------------------------------------------------------- */
/* Hebbian update                                                         */
/* --------------------------------------------------------------------- */

void trine_accumulator_update(trine_accumulator_t *acc,
                               const uint8_t a[240], const uint8_t b[240],
                               int sign)
{
    uint32_t k, i, j;

    if (!acc) return;

    /* Normalize sign to +1 or -1 */
    int s = (sign >= 0) ? 1 : -1;

    /*
     * Hebbian rule (applied to all K projection copies):
     *   counter[k][i][j] += sign * centered_a[j] * centered_b[i]
     *
     * Trits {0,1,2} are centered to {-1,0,+1} so the outer product
     * captures correlation (both high or both low) vs anti-correlation.
     * centered[v] = v - 1, so: 0→-1, 1→0, 2→+1.
     * Products: (-1)(-1)=1, (-1)(+1)=-1, (+1)(+1)=1, anything*0=0.
     */
    for (k = 0; k < TRINE_ACC_K; k++) {
        for (i = 0; i < TRINE_ACC_DIM; i++) {
            int32_t bi = (int32_t)b[i] - 1;  /* center: {0,1,2} -> {-1,0,+1} */
            if (bi == 0) continue;  /* trit=1 is neutral, no contribution */
            int32_t s_bi = s * bi;
            for (j = 0; j < TRINE_ACC_DIM; j++) {
                int32_t aj = (int32_t)a[j] - 1;  /* center */
                if (aj == 0) continue;  /* trit=1 is neutral */
                int32_t delta = s_bi * aj;
                acc->counters[k][i][j] = sat_add_i32(acc->counters[k][i][j], delta);
            }
        }
    }

    acc->total_updates++;
}

void trine_accumulator_update_weighted(trine_accumulator_t *acc,
                                        const uint8_t a[240], const uint8_t b[240],
                                        int sign, int32_t magnitude)
{
    uint32_t k, i, j;

    if (!acc) return;
    if (magnitude < 1) magnitude = 1;

    int s = (sign >= 0) ? 1 : -1;
    int32_t s_mag = s * magnitude;

    for (k = 0; k < TRINE_ACC_K; k++) {
        for (i = 0; i < TRINE_ACC_DIM; i++) {
            int32_t bi = (int32_t)b[i] - 1;
            if (bi == 0) continue;
            int32_t s_bi = s_mag * bi;
            for (j = 0; j < TRINE_ACC_DIM; j++) {
                int32_t aj = (int32_t)a[j] - 1;
                if (aj == 0) continue;
                int32_t delta = s_bi * aj;
                acc->counters[k][i][j] = sat_add_i32(acc->counters[k][i][j], delta);
            }
        }
    }

    acc->total_updates++;
}

/* --------------------------------------------------------------------- */
/* Counter access                                                         */
/* --------------------------------------------------------------------- */

int32_t (*trine_accumulator_counters(trine_accumulator_t *acc, uint32_t k))[TRINE_ACC_DIM]
{
    if (!acc || k >= TRINE_ACC_K) return NULL;
    return acc->counters[k];
}

const int32_t (*trine_accumulator_counters_const(const trine_accumulator_t *acc, uint32_t k))[TRINE_ACC_DIM]
{
    if (!acc || k >= TRINE_ACC_K) return NULL;
    return acc->counters[k];
}

/* --------------------------------------------------------------------- */
/* Statistics                                                              */
/* --------------------------------------------------------------------- */

void trine_accumulator_stats(const trine_accumulator_t *acc,
                              trine_accumulator_stats_t *stats)
{
    uint32_t k, i, j;

    if (!stats) return;
    memset(stats, 0, sizeof(*stats));

    if (!acc) return;

    stats->total_updates = acc->total_updates;

    int32_t max_abs = 0;

    for (k = 0; k < TRINE_ACC_K; k++) {
        for (i = 0; i < TRINE_ACC_DIM; i++) {
            for (j = 0; j < TRINE_ACC_DIM; j++) {
                int32_t v = acc->counters[k][i][j];
                if (v > 0) {
                    stats->n_positive++;
                    if (v > max_abs) max_abs = v;
                } else if (v < 0) {
                    stats->n_negative++;
                    int32_t abs_v = (v == INT32_MIN) ? INT32_MAX : -v;
                    if (abs_v > max_abs) max_abs = abs_v;
                } else {
                    stats->n_zero++;
                }
                if (v == INT32_MAX || v == INT32_MIN) {
                    stats->n_saturated++;
                }
            }
        }
    }

    stats->max_abs = max_abs;
}

/* =====================================================================
 * Block-Diagonal Accumulator (v1.0.3)
 * =====================================================================
 *
 * K copies of 4 independent 60x60 counter matrices (one per chain).
 * Counter layout (flat): counters[k * 4 * 60 * 60 + c * 60 * 60 + i * 60 + j]
 *
 * Only accumulates within-chain correlations.  Chain c spans dims
 * [c*60 .. c*60+59] of the 240-dim embedding.
 * ===================================================================== */

#define BLOCK_ELEMS (TRINE_BLOCK_CHAINS * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM)
/* = 4 * 60 * 60 = 14,400 counters per projection copy */

/* --------------------------------------------------------------------- */
/* Block-diagonal: create / free / reset                                  */
/* --------------------------------------------------------------------- */

trine_block_accumulator_t *trine_block_accumulator_create(int K)
{
    if (K < 1) return NULL;

    trine_block_accumulator_t *acc = calloc(1, sizeof(*acc));
    if (!acc) return NULL;

    size_t n = (size_t)K * BLOCK_ELEMS;
    acc->counters = calloc(n, sizeof(int32_t));
    if (!acc->counters) {
        free(acc);
        return NULL;
    }

    acc->K = K;
    acc->mode = TRINE_ACCUM_MODE_BLOCK;
    acc->total_updates = 0;
    acc->pairs_observed = 0;

    return acc;
}

void trine_block_accumulator_free(trine_block_accumulator_t *acc)
{
    if (!acc) return;
    free(acc->counters);
    free(acc);
}

void trine_block_accumulator_reset(trine_block_accumulator_t *acc)
{
    if (!acc) return;
    size_t n = (size_t)acc->K * BLOCK_ELEMS;
    memset(acc->counters, 0, n * sizeof(int32_t));
    acc->total_updates = 0;
    acc->pairs_observed = 0;
}

/* --------------------------------------------------------------------- */
/* Block-diagonal: Hebbian update                                         */
/* --------------------------------------------------------------------- */

void trine_block_accumulator_update(
    trine_block_accumulator_t *acc,
    const uint8_t a[240], const uint8_t b[240],
    int sign)
{
    int k, c;
    uint32_t i, j;

    if (!acc || !acc->counters) return;

    /* Normalize: positive (sign > 0) → +1, negative (sign <= 0) → -1.
     * Matches both boolean (0/1) and signed (-1/+1) calling conventions. */
    int s = (sign > 0) ? 1 : -1;

    for (k = 0; k < acc->K; k++) {
        int32_t *base_k = acc->counters + k * BLOCK_ELEMS;

        for (c = 0; c < TRINE_BLOCK_CHAINS; c++) {
            int32_t *base_c = base_k + c * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;
            uint32_t chain_offset = (uint32_t)c * TRINE_BLOCK_DIM;

            for (i = 0; i < TRINE_BLOCK_DIM; i++) {
                int32_t bi = (int32_t)b[chain_offset + i] - 1;
                if (bi == 0) continue;  /* neutral trit, skip */
                int32_t s_bi = s * bi;

                for (j = 0; j < TRINE_BLOCK_DIM; j++) {
                    int32_t aj = (int32_t)a[chain_offset + j] - 1;
                    if (aj == 0) continue;  /* neutral trit, skip */
                    int32_t delta = s_bi * aj;
                    base_c[i * TRINE_BLOCK_DIM + j] =
                        sat_add_i32(base_c[i * TRINE_BLOCK_DIM + j], delta);
                }
            }
        }
    }

    acc->total_updates++;
    acc->pairs_observed++;
}

/* --------------------------------------------------------------------- */
/* Block-diagonal: weighted Hebbian update                                */
/* --------------------------------------------------------------------- */

void trine_block_accumulator_update_weighted(
    trine_block_accumulator_t *acc,
    const uint8_t a[240], const uint8_t b[240],
    int sign, int magnitude)
{
    int k, c;
    uint32_t i, j;

    if (!acc || !acc->counters) return;
    if (magnitude < 1) magnitude = 1;

    int s = (sign > 0) ? 1 : -1;
    int32_t s_mag = (int32_t)s * (int32_t)magnitude;

    for (k = 0; k < acc->K; k++) {
        int32_t *base_k = acc->counters + k * BLOCK_ELEMS;

        for (c = 0; c < TRINE_BLOCK_CHAINS; c++) {
            int32_t *base_c = base_k + c * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;
            uint32_t chain_offset = (uint32_t)c * TRINE_BLOCK_DIM;

            for (i = 0; i < TRINE_BLOCK_DIM; i++) {
                int32_t bi = (int32_t)b[chain_offset + i] - 1;
                if (bi == 0) continue;
                int32_t s_bi = s_mag * bi;

                for (j = 0; j < TRINE_BLOCK_DIM; j++) {
                    int32_t aj = (int32_t)a[chain_offset + j] - 1;
                    if (aj == 0) continue;
                    int32_t delta = s_bi * aj;
                    base_c[i * TRINE_BLOCK_DIM + j] =
                        sat_add_i32(base_c[i * TRINE_BLOCK_DIM + j], delta);
                }
            }
        }
    }

    acc->total_updates++;
    acc->pairs_observed++;
}

/* --------------------------------------------------------------------- */
/* Block-diagonal: statistics                                             */
/* --------------------------------------------------------------------- */

void trine_block_accumulator_stats(const trine_block_accumulator_t *acc,
                                    int32_t *max_val, int32_t *min_val,
                                    uint64_t *nonzero)
{
    int32_t  mx = 0;
    int32_t  mn = 0;
    uint64_t nz = 0;

    if (!acc || !acc->counters) {
        if (max_val)  *max_val  = 0;
        if (min_val)  *min_val  = 0;
        if (nonzero)  *nonzero  = 0;
        return;
    }

    size_t total = (size_t)acc->K * BLOCK_ELEMS;
    for (size_t idx = 0; idx < total; idx++) {
        int32_t v = acc->counters[idx];
        if (v > mx) mx = v;
        if (v < mn) mn = v;
        if (v != 0) nz++;
    }

    if (max_val)  *max_val  = mx;
    if (min_val)  *min_val  = mn;
    if (nonzero)  *nonzero  = nz;
}
