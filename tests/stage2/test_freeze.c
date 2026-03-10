/* =====================================================================
 * TRINE Stage-2 — Freeze/Quantization Tests (~12 tests)
 * =====================================================================
 *
 * Tests for freeze/quantization:
 *   1. Freeze with threshold=0: all non-zero counters become 1 or 2
 *   2. Freeze with high threshold: mostly zeros (sparse)
 *   3. Freeze with threshold=1: balanced output
 *   4. Freeze preserves sign direction: positive counters -> 2, negative -> 1
 *   5. Z3 closure: all frozen values in {0,1,2}
 *   6. auto_threshold returns reasonable values
 *   7. auto_threshold with high density -> low threshold
 *   8. auto_threshold with low density -> high threshold
 *   9. freeze_stats() counts match total (n_zero + n_one + n_two = 3*240*240)
 *  10. Determinism: same accumulators -> same frozen weights
 *  11. Zero accumulator -> all zero weights regardless of threshold
 *  12. Identity-like accumulator produces diagonal-heavy weights
 *
 * ===================================================================== */

#include "trine_freeze.h"
#include "trine_accumulator.h"
#include "trine_encode.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static int g_passed = 0;
static int g_failed = 0;
static int g_total  = 0;

static void check(const char *name, int cond)
{
    g_total++;
    if (cond) {
        g_passed++;
    } else {
        g_failed++;
        printf("  FAIL  freeze: %s\n", name);
    }
}

#define TOTAL_WEIGHTS (3u * 240u * 240u)

/* ── Test 1: Freeze with threshold=0 ───────────────────────────────── */
static void test_freeze_threshold_zero(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    /* Accumulate one pair so counters are non-zero */
    uint8_t a[240], b[240];
    trine_encode_shingle("threshold zero test", 19, a);
    trine_encode_shingle("threshold zero check", 20, b);
    trine_accumulator_update(acc, a, b, 1);

    trine_projection_t proj;
    trine_freeze_projection(acc, 0, &proj);

    /* With threshold=0, all non-zero counters must map to 1 or 2 */
    trine_freeze_stats_t stats;
    trine_freeze_stats(&proj, &stats);

    /* There should be non-zero weights (since we had non-zero counters) */
    check("t0_has_nonzero", stats.n_one + stats.n_two > 0);
    /* All values in {0,1,2} (Z3 closure) */
    check("t0_total", stats.n_zero + stats.n_one + stats.n_two == TOTAL_WEIGHTS);

    trine_accumulator_free(acc);
}

/* ── Test 2: Freeze with high threshold ─────────────────────────────── */
static void test_freeze_high_threshold(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    uint8_t a[240], b[240];
    trine_encode_shingle("high threshold", 14, a);
    trine_encode_shingle("high threshold test", 19, b);

    /* Only a few updates -- counters will be small */
    trine_accumulator_update(acc, a, b, 1);
    trine_accumulator_update(acc, a, b, 1);

    /* Get max_abs to set a very high threshold */
    trine_accumulator_stats_t acc_stats;
    trine_accumulator_stats(acc, &acc_stats);

    /* Freeze with threshold well above max_abs */
    trine_projection_t proj;
    trine_freeze_projection(acc, acc_stats.max_abs + 10, &proj);

    trine_freeze_stats_t stats;
    trine_freeze_stats(&proj, &stats);

    /* Everything should be zero -- threshold exceeds all counter values */
    check("high_t_all_zero", stats.n_one == 0 && stats.n_two == 0);
    check("high_t_total", stats.n_zero == TOTAL_WEIGHTS);

    trine_accumulator_free(acc);
}

/* ── Test 3: Freeze with threshold=1 ───────────────────────────────── */
static void test_freeze_threshold_one(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    uint8_t a[240], b[240];
    trine_encode_shingle("balanced freeze", 15, a);
    trine_encode_shingle("balanced test", 13, b);

    /* Several updates to build up counters */
    for (int i = 0; i < 5; i++)
        trine_accumulator_update(acc, a, b, 1);

    trine_projection_t proj_t0, proj_t1;
    trine_freeze_projection(acc, 0, &proj_t0);
    trine_freeze_projection(acc, 1, &proj_t1);

    trine_freeze_stats_t s0, s1;
    trine_freeze_stats(&proj_t0, &s0);
    trine_freeze_stats(&proj_t1, &s1);

    /* threshold=1 should be sparser than threshold=0 */
    uint32_t nonzero_0 = s0.n_one + s0.n_two;
    uint32_t nonzero_1 = s1.n_one + s1.n_two;
    check("t1_sparser_than_t0", nonzero_1 <= nonzero_0);

    trine_accumulator_free(acc);
}

/* ── Test 4: Sign direction ─────────────────────────────────────────── */
static void test_sign_direction(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    /* Create a simple pair where we know the centered outer product.
     * Trit 2 centers to +1, trit 0 centers to -1, trit 1 centers to 0. */
    uint8_t a[240], b[240];
    memset(a, 1, sizeof(a));  /* all neutral (trit=1 → centered=0) */
    memset(b, 1, sizeof(b));

    /* Set correlated positions: both trit=2 → centered (+1)(+1) = +1 */
    a[0] = 2;
    b[0] = 2;

    /* Positive update: counter[k][0][0] += +1 * (+1)*(+1) = +1 for all k */
    trine_accumulator_update(acc, a, b, 1);

    trine_projection_t proj;
    trine_freeze_projection(acc, 0, &proj);

    /* Positive counter -> should freeze to 2 */
    check("positive_to_2", proj.W[0][0][0] == 2);

    /* Reset and do negative update */
    trine_accumulator_reset(acc);
    trine_accumulator_update(acc, a, b, -1);

    trine_freeze_projection(acc, 0, &proj);

    /* Negative counter -> should freeze to 1 */
    check("negative_to_1", proj.W[0][0][0] == 1);

    trine_accumulator_free(acc);
}

/* ── Test 5: Z3 closure ─────────────────────────────────────────────── */
static void test_z3_closure(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    /* Feed multiple diverse pairs */
    const char *texts[] = {
        "alpha", "beta", "gamma", "delta", "epsilon",
        "one two three", "four five six", "seven eight nine"
    };
    for (int i = 0; i < 7; i++) {
        uint8_t a[240], b[240];
        trine_encode_shingle(texts[i], strlen(texts[i]), a);
        trine_encode_shingle(texts[i + 1], strlen(texts[i + 1]), b);
        trine_accumulator_update(acc, a, b, (i % 2 == 0) ? 1 : -1);
    }

    trine_projection_t proj;
    trine_freeze_projection(acc, 0, &proj);

    /* Every single weight must be in {0, 1, 2} */
    int ok = 1;
    for (uint32_t k = 0; k < 3 && ok; k++)
        for (uint32_t i = 0; i < 240 && ok; i++)
            for (uint32_t j = 0; j < 240 && ok; j++)
                if (proj.W[k][i][j] > 2) ok = 0;

    check("z3_all_values_valid", ok);

    trine_accumulator_free(acc);
}

/* ── Test 6: auto_threshold returns reasonable value ────────────────── */
static void test_auto_threshold_reasonable(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    /* Feed many diverse pairs to populate enough counters that
     * density at threshold=0 exceeds the target. */
    const char *texts[] = {
        "alpha bravo charlie", "delta echo foxtrot",
        "golf hotel india", "juliet kilo lima",
        "mike november oscar", "papa quebec romeo",
        "sierra tango uniform", "victor whiskey xray",
        "yankee zulu one two", "three four five six"
    };

    for (int i = 0; i < 10; i += 2) {
        uint8_t a[240], b[240];
        trine_encode_shingle(texts[i], strlen(texts[i]), a);
        trine_encode_shingle(texts[i + 1], strlen(texts[i + 1]), b);
        for (int j = 0; j < 50; j++)
            trine_accumulator_update(acc, a, b, 1);
    }

    trine_accumulator_stats_t st;
    trine_accumulator_stats(acc, &st);

    int32_t auto_t = trine_freeze_auto_threshold(acc, 0.33f);

    /* Auto threshold should be between 0 and max_abs (inclusive) */
    check("auto_t_non_negative", auto_t >= 0);
    check("auto_t_bounded", auto_t <= st.max_abs);

    trine_accumulator_free(acc);
}

/* ── Test 7: auto_threshold with high density -> low threshold ─────── */
static void test_auto_threshold_high_density(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    uint8_t a[240], b[240];
    trine_encode_shingle("density high test", 17, a);
    trine_encode_shingle("density high check", 18, b);
    for (int i = 0; i < 30; i++)
        trine_accumulator_update(acc, a, b, 1);

    /* target_density=0.9 means we want lots of non-zero -> low threshold */
    int32_t t_high_dens = trine_freeze_auto_threshold(acc, 0.9f);

    /* target_density=0.05 means we want very sparse -> high threshold */
    int32_t t_low_dens = trine_freeze_auto_threshold(acc, 0.05f);

    check("high_density_lower_threshold", t_high_dens <= t_low_dens);

    trine_accumulator_free(acc);
}

/* ── Test 8: auto_threshold with low density -> high threshold ─────── */
static void test_auto_threshold_low_density(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    uint8_t a[240], b[240];
    trine_encode_shingle("sparse target test", 18, a);
    trine_encode_shingle("sparse target check", 19, b);
    for (int i = 0; i < 30; i++)
        trine_accumulator_update(acc, a, b, 1);

    /* Very sparse target: threshold should be higher */
    int32_t t_sparse = trine_freeze_auto_threshold(acc, 0.01f);
    /* Moderate density target */
    int32_t t_moderate = trine_freeze_auto_threshold(acc, 0.5f);

    check("sparse_higher_threshold", t_sparse >= t_moderate);

    trine_accumulator_free(acc);
}

/* ── Test 9: freeze_stats() counts sum to total ─────────────────────── */
static void test_stats_sum(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    uint8_t a[240], b[240];
    trine_encode_shingle("stats sum test", 14, a);
    trine_encode_shingle("stats sum verify", 16, b);
    for (int i = 0; i < 10; i++)
        trine_accumulator_update(acc, a, b, (i % 2 == 0) ? 1 : -1);

    trine_projection_t proj;
    trine_freeze_projection(acc, 1, &proj);

    trine_freeze_stats_t stats;
    trine_freeze_stats(&proj, &stats);

    uint32_t sum = stats.n_zero + stats.n_one + stats.n_two;
    check("stats_sum_eq_total", sum == TOTAL_WEIGHTS);

    /* Density should match */
    float expected_density = (float)(stats.n_one + stats.n_two) / (float)TOTAL_WEIGHTS;
    check("stats_density", fabsf(stats.density - expected_density) < 1e-6f);

    trine_accumulator_free(acc);
}

/* ── Test 10: Determinism ───────────────────────────────────────────── */
static void test_determinism(void)
{
    trine_projection_t proj1, proj2;

    for (int trial = 0; trial < 2; trial++) {
        trine_accumulator_t *acc = trine_accumulator_create();

        uint8_t a[240], b[240];
        trine_encode_shingle("determinism freeze", 18, a);
        trine_encode_shingle("determinism check", 17, b);

        trine_accumulator_update(acc, a, b, 1);
        trine_accumulator_update(acc, b, a, -1);

        trine_freeze_projection(acc, 1, trial == 0 ? &proj1 : &proj2);
        trine_accumulator_free(acc);
    }

    check("freeze_determinism", memcmp(&proj1, &proj2, sizeof(proj1)) == 0);
}

/* ── Test 11: Zero accumulator -> all zero weights ──────────────────── */
static void test_zero_accumulator(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();
    /* No updates -- all counters are zero */

    trine_projection_t proj;

    /* threshold=0 */
    trine_freeze_projection(acc, 0, &proj);
    trine_freeze_stats_t s0;
    trine_freeze_stats(&proj, &s0);
    check("zero_acc_t0_all_zero", s0.n_zero == TOTAL_WEIGHTS);

    /* threshold=5 */
    trine_freeze_projection(acc, 5, &proj);
    trine_freeze_stats_t s5;
    trine_freeze_stats(&proj, &s5);
    check("zero_acc_t5_all_zero", s5.n_zero == TOTAL_WEIGHTS);

    /* Auto threshold with zero accumulators should return 0 */
    int32_t auto_t = trine_freeze_auto_threshold(acc, 0.33f);
    check("zero_acc_auto_t_zero", auto_t == 0);

    trine_accumulator_free(acc);
}

/* ── Test 12: Identity-like accumulator (diagonal positive) ─────────── */
static void test_diagonal_accumulator(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    /* Manually set diagonal counters to positive values by feeding
     * vectors with trit=2 at one position (centered=+1) through the
     * accumulator.  a = b = e_i (trit=2 at pos i, trit=1 elsewhere).
     * counter[k][i][i] += (+1)*(+1) = +1, off-diag stays 0. */
    for (int pos = 0; pos < 240; pos++) {
        uint8_t v[240];
        memset(v, 1, sizeof(v));  /* neutral (trit=1 → centered=0) */
        v[pos] = 2;               /* active (trit=2 → centered=+1) */
        trine_accumulator_update(acc, v, v, 1);
    }

    trine_projection_t proj;
    trine_freeze_projection(acc, 0, &proj);

    /* Check that diagonal elements are 2 (positive counters -> 2) */
    int diag_all_two = 1;
    for (uint32_t k = 0; k < 3 && diag_all_two; k++)
        for (uint32_t i = 0; i < 240 && diag_all_two; i++)
            if (proj.W[k][i][i] != 2) diag_all_two = 0;

    check("diagonal_all_two", diag_all_two);

    /* Off-diagonal should be mostly zero for this specific input */
    trine_freeze_stats_t stats;
    trine_freeze_stats(&proj, &stats);

    /* Exactly 3*240 = 720 diagonal elements should be 2 */
    check("diagonal_count", stats.n_two == 3u * 240u);

    trine_accumulator_free(acc);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Freeze Tests ===\n");

    test_freeze_threshold_zero();
    test_freeze_high_threshold();
    test_freeze_threshold_one();
    test_sign_direction();
    test_z3_closure();
    test_auto_threshold_reasonable();
    test_auto_threshold_high_density();
    test_auto_threshold_low_density();
    test_stats_sum();
    test_determinism();
    test_zero_accumulator();
    test_diagonal_accumulator();

    printf("\nFreeze: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
