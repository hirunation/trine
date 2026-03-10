/* =====================================================================
 * TRINE Stage-2 — Freeze (Quantization) (implementation)
 * =====================================================================
 *
 * Quantizes int32 Hebbian accumulators to ternary (0/1/2) projection
 * weights.  See trine_freeze.h for API documentation.
 *
 * ===================================================================== */

#include "trine_freeze.h"
#include "trine_accumulator.h"

#include <string.h>
#include <limits.h>

/* --------------------------------------------------------------------- */
/* Freeze accumulators to ternary                                         */
/* --------------------------------------------------------------------- */

void trine_freeze_projection(const trine_accumulator_t *acc,
                              int32_t threshold,
                              trine_projection_t *out)
{
    uint32_t k, i, j;

    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!acc) return;

    /* Clamp threshold to non-negative */
    if (threshold < 0) threshold = 0;

    for (k = 0; k < TRINE_ACC_K; k++) {
        const int32_t (*mat)[TRINE_ACC_DIM] =
            trine_accumulator_counters_const(acc, k);
        if (!mat) continue;

        for (i = 0; i < TRINE_ACC_DIM; i++) {
            for (j = 0; j < TRINE_ACC_DIM; j++) {
                int32_t v = mat[i][j];
                if (v > threshold) {
                    out->W[k][i][j] = 2;
                } else if (v < -threshold) {
                    out->W[k][i][j] = 1;
                } else {
                    out->W[k][i][j] = 0;
                }
            }
        }
    }
}

/* --------------------------------------------------------------------- */
/* Sparse freeze: keep top-K per output row                               */
/* --------------------------------------------------------------------- */

void trine_freeze_sparse(const trine_accumulator_t *acc,
                           uint32_t top_k,
                           trine_projection_t *out)
{
    uint32_t k, i, j;

    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!acc) return;

    /* Clamp top_k */
    if (top_k < 1) top_k = 1;
    if (top_k > TRINE_ACC_DIM) top_k = TRINE_ACC_DIM;

    /* For each copy k, each output row i: find top-K by |counter| */
    for (k = 0; k < TRINE_ACC_K; k++) {
        const int32_t (*mat)[TRINE_ACC_DIM] =
            trine_accumulator_counters_const(acc, k);
        if (!mat) continue;

        for (i = 0; i < TRINE_ACC_DIM; i++) {
            /* Collect (abs_value, index) pairs for this row */
            uint32_t indices[TRINE_ACC_DIM];
            int32_t  abs_vals[TRINE_ACC_DIM];

            for (j = 0; j < TRINE_ACC_DIM; j++) {
                indices[j] = j;
                int32_t v = mat[i][j];
                abs_vals[j] = (v < 0) ? -v : v;
            }

            /* Partial selection sort: find top-K largest abs values.
             * O(K*DIM) is fine for K=8, DIM=240 (1,920 iterations/row). */
            for (uint32_t s = 0; s < top_k; s++) {
                uint32_t best = s;
                for (uint32_t t = s + 1; t < TRINE_ACC_DIM; t++) {
                    if (abs_vals[t] > abs_vals[best]) {
                        best = t;
                    }
                }
                if (best != s) {
                    /* Swap */
                    int32_t  tmp_v = abs_vals[s];
                    uint32_t tmp_i = indices[s];
                    abs_vals[s] = abs_vals[best];
                    indices[s]  = indices[best];
                    abs_vals[best] = tmp_v;
                    indices[best]  = tmp_i;
                }
            }

            /* Quantize top-K entries by sign */
            for (uint32_t s = 0; s < top_k; s++) {
                if (abs_vals[s] == 0) break;  /* no more signal */
                uint32_t col = indices[s];
                int32_t v = mat[i][col];
                out->W[k][i][col] = (v > 0) ? 2 : 1;
            }
        }
    }
}

/* --------------------------------------------------------------------- */
/* Helper: count non-zero weights at a given threshold                    */
/* --------------------------------------------------------------------- */

static uint32_t count_nonzero_at(const trine_accumulator_t *acc, int32_t t)
{
    uint32_t k, i, j;
    uint32_t n_nonzero = 0;

    for (k = 0; k < TRINE_ACC_K; k++) {
        const int32_t (*mat)[TRINE_ACC_DIM] =
            trine_accumulator_counters_const(acc, k);
        if (!mat) continue;

        for (i = 0; i < TRINE_ACC_DIM; i++) {
            for (j = 0; j < TRINE_ACC_DIM; j++) {
                int32_t v = mat[i][j];
                if (v > t || v < -t) {
                    n_nonzero++;
                }
            }
        }
    }

    return n_nonzero;
}

/* --------------------------------------------------------------------- */
/* Auto-threshold: binary search for target density                       */
/* --------------------------------------------------------------------- */

int32_t trine_freeze_auto_threshold(const trine_accumulator_t *acc,
                                     float target_density)
{
    if (!acc) return 0;

    /* Find max absolute value */
    trine_accumulator_stats_t st;
    trine_accumulator_stats(acc, &st);

    if (st.max_abs == 0) return 0;

    uint32_t total = (uint32_t)TRINE_ACC_K * TRINE_ACC_DIM * TRINE_ACC_DIM;
    uint32_t target_nonzero = (uint32_t)(target_density * (float)total);

    /* Binary search: find smallest threshold where nonzero <= target.
     * density decreases monotonically with threshold.
     * lo = 0 (max density), hi = max_abs (min density). */
    int32_t lo = 0;
    int32_t hi = st.max_abs;

    /* Quick check: at threshold=0, if density is already <= target, return 0 */
    if (count_nonzero_at(acc, 0) <= target_nonzero) return 0;

    /* At threshold=max_abs, density is 0 (nothing exceeds max_abs strictly).
     * So the answer exists in [1, max_abs]. */
    lo = 1;

    while (lo < hi) {
        int32_t mid = lo + (hi - lo) / 2;
        uint32_t nonzero = count_nonzero_at(acc, mid);

        if (nonzero <= target_nonzero) {
            hi = mid;  /* mid is a valid threshold, try lower */
        } else {
            lo = mid + 1;  /* density too high, need higher threshold */
        }
    }

    return lo;
}

/* --------------------------------------------------------------------- */
/* Frozen projection statistics                                           */
/* --------------------------------------------------------------------- */

void trine_freeze_stats(const trine_projection_t *proj,
                         trine_freeze_stats_t *stats)
{
    uint32_t k, i, j;

    if (!stats) return;
    memset(stats, 0, sizeof(*stats));

    if (!proj) return;

    for (k = 0; k < 3; k++) {
        for (i = 0; i < 240; i++) {
            for (j = 0; j < 240; j++) {
                uint8_t v = proj->W[k][i][j];
                if (v == 0) {
                    stats->n_zero++;
                } else if (v == 1) {
                    stats->n_one++;
                } else {
                    stats->n_two++;
                }
            }
        }
    }

    uint32_t total = 3u * 240u * 240u;
    stats->density = (float)(stats->n_one + stats->n_two) / (float)total;
}

/* =====================================================================
 * Block-Diagonal Freeze (v1.0.3)
 * =====================================================================
 *
 * Freeze K * 4 * 60x60 block-diagonal accumulators to ternary weights.
 * ===================================================================== */

/* --------------------------------------------------------------------- */
/* Freeze block-diagonal accumulators to ternary                          */
/* --------------------------------------------------------------------- */

int trine_freeze_block(const trine_block_accumulator_t *acc,
                       int32_t threshold,
                       uint8_t *W_blocks)
{
    if (!acc || !acc->counters || !W_blocks) return -1;

    /* Clamp threshold to non-negative */
    if (threshold < 0) threshold = 0;

    size_t total = (size_t)acc->K * TRINE_BLOCK_CHAINS
                   * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;

    for (size_t idx = 0; idx < total; idx++) {
        int32_t v = acc->counters[idx];
        if (v > threshold) {
            W_blocks[idx] = 2;
        } else if (v < -threshold) {
            W_blocks[idx] = 1;
        } else {
            W_blocks[idx] = 0;
        }
    }

    return 0;
}

/* --------------------------------------------------------------------- */
/* Helper: count non-zero block entries at a given threshold              */
/* --------------------------------------------------------------------- */

static uint64_t count_block_nonzero_at(const trine_block_accumulator_t *acc,
                                        int32_t t)
{
    uint64_t n_nonzero = 0;
    size_t total = (size_t)acc->K * TRINE_BLOCK_CHAINS
                   * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;

    for (size_t idx = 0; idx < total; idx++) {
        int32_t v = acc->counters[idx];
        if (v > t || v < -t) {
            n_nonzero++;
        }
    }

    return n_nonzero;
}

/* --------------------------------------------------------------------- */
/* Auto-threshold for block-diagonal: binary search for target density    */
/* --------------------------------------------------------------------- */

int32_t trine_freeze_block_auto_threshold(const trine_block_accumulator_t *acc,
                                           float target_density)
{
    if (!acc || !acc->counters) return 0;

    /* Find max absolute value */
    int32_t max_val, min_val;
    uint64_t nonzero;
    trine_block_accumulator_stats(acc, &max_val, &min_val, &nonzero);

    /* Compute max_abs from max_val and min_val */
    int32_t max_abs = max_val;
    int32_t abs_min = (min_val == INT32_MIN) ? INT32_MAX : -min_val;
    if (abs_min > max_abs) max_abs = abs_min;

    if (max_abs == 0) return 0;

    uint64_t total = (uint64_t)acc->K * TRINE_BLOCK_CHAINS
                     * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;
    uint64_t target_nonzero = (uint64_t)(target_density * (float)total);

    /* Quick check: at threshold=0, if density is already <= target, return 0 */
    if (count_block_nonzero_at(acc, 0) <= target_nonzero) return 0;

    /* Binary search: find smallest threshold where nonzero <= target.
     * At threshold=max_abs, density is 0. Answer in [1, max_abs]. */
    int32_t lo = 1;
    int32_t hi = max_abs;

    while (lo < hi) {
        int32_t mid = lo + (hi - lo) / 2;
        uint64_t nz = count_block_nonzero_at(acc, mid);

        if (nz <= target_nonzero) {
            hi = mid;  /* mid is valid, try lower */
        } else {
            lo = mid + 1;  /* density too high, need higher threshold */
        }
    }

    return lo;
}
