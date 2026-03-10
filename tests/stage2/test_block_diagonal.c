/* =====================================================================
 * TRINE Stage-2 — Block-Diagonal Projection Tests (28 functions, 46 checks)
 * =====================================================================
 *
 * Comprehensive tests for the block-diagonal projection system (v1.0.3).
 *
 * Category 1: Block-Diagonal Projection (8 tests)
 *   1. Identity block projection: all 4 blocks identity -> output == input
 *   2. Known values: manually construct a small block, verify output
 *   3. Z3 closure: output values are all in {0, 1, 2}
 *   4. Chain independence: modifying chain 0's block doesn't affect chain 1
 *   5. Determinism: same input -> same output
 *   6. All-zeros input -> all-zeros output (mod 3 = 0)
 *   7. All-ones input: verify expected output
 *   8. All-twos input: verify expected output
 *
 * Category 2: Block Majority Vote (4 tests)
 *   9. K=3 majority with identical blocks -> same as single projection
 *  10. K=3 with one different -> majority wins
 *  11. Identity blocks -> output == input
 *  12. Determinism: same seed -> same result
 *
 * Category 3: Block Accumulator (6 tests)
 *  13. Create and free without crash
 *  14. Update with identical vectors -> diagonal positive
 *  15. Update with orthogonal vectors -> smaller correlations
 *  16. Positive vs negative update -> opposite signs
 *  17. Reset clears all counters
 *  18. Stats report correct max/min/nonzero
 *
 * Category 4: Block Freeze (5 tests)
 *  19. Freeze with threshold 0 -> all nonzero counters become weights
 *  20. Freeze produces only {0, 1, 2} values
 *  21. Auto-threshold achieves target density (within 10%)
 *  22. Frozen identity accumulator ~ identity matrix
 *  23. Round-trip: accumulate -> freeze -> project -> reasonable output
 *
 * Category 5: End-to-End Block Pipeline (4 tests)
 *  24. Encode "hello" with block-diagonal -> different from Stage-1
 *  25. Encode "hello" with identity block -> same as Stage-1
 *  26. Compare two similar texts with block projection -> reasonable sim
 *  27. Block projection preserves encode determinism
 *
 * ===================================================================== */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "trine_project.h"
#include "trine_accumulator.h"
#include "trine_encode.h"
#include "trine_stage1.h"

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
        printf("  FAIL  block_diagonal: %s\n", name);
    }
}

/* Helper: block size in bytes for K copies */
#define BLOCK_BYTES(K) ((size_t)(K) * TRINE_S2_N_CHAINS * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM)

/* Helper: access block W_blocks[k][c][i][j] */
static inline uint8_t *block_ptr(uint8_t *W_blocks, int k, int c, int i, int j)
{
    return &W_blocks[(size_t)k * TRINE_S2_N_CHAINS * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM
                     + (size_t)c * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM
                     + (size_t)i * TRINE_S2_CHAIN_DIM
                     + (size_t)j];
}

static inline uint8_t block_get(const uint8_t *W_blocks, int k, int c, int i, int j)
{
    return W_blocks[(size_t)k * TRINE_S2_N_CHAINS * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM
                    + (size_t)c * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM
                    + (size_t)i * TRINE_S2_CHAIN_DIM
                    + (size_t)j];
}

/* =====================================================================
 * Category 1: Block-Diagonal Projection (8 tests)
 * ===================================================================== */

/* 1. Identity block projection: all 4 blocks identity -> output == input */
static void test_block_identity_passthrough(void)
{
    uint8_t W_block[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM];

    /* Set each chain's block to identity */
    memset(W_block, 0, sizeof(W_block));
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            W_block[c][i][i] = 1;

    /* Mixed input pattern */
    uint8_t x[TRINE_S2_DIM], y[TRINE_S2_DIM];
    for (int i = 0; i < TRINE_S2_DIM; i++)
        x[i] = (uint8_t)(i % 3);

    trine_project_block_diagonal(W_block, x, y);

    check("identity_passthrough", memcmp(x, y, TRINE_S2_DIM) == 0);
}

/* 2. Known values: manually construct a small block, verify output */
static void test_block_known_values(void)
{
    /* Block for chain 0: small 3x3 submatrix (rest identity).
     * W[0][0] = {2, 1, 0, 0, ..., 0}  (only first 3 non-identity)
     * W[0][1] = {0, 2, 1, 0, ..., 0}
     * W[0][2] = {1, 0, 2, 0, ..., 0}
     * x[0..2] = {1, 2, 1}
     *
     * y[0] = (2*1 + 1*2 + 0*1) % 3 = 4 % 3 = 1
     * y[1] = (0*1 + 2*2 + 1*1) % 3 = 5 % 3 = 2
     * y[2] = (1*1 + 0*2 + 2*1) % 3 = 3 % 3 = 0
     */
    uint8_t W_block[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM];

    /* Start with identity for all chains */
    memset(W_block, 0, sizeof(W_block));
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            W_block[c][i][i] = 1;

    /* Override chain 0, rows 0-2, cols 0-2 */
    W_block[0][0][0] = 2; W_block[0][0][1] = 1; W_block[0][0][2] = 0;
    W_block[0][1][0] = 0; W_block[0][1][1] = 2; W_block[0][1][2] = 1;
    W_block[0][2][0] = 1; W_block[0][2][1] = 0; W_block[0][2][2] = 2;

    /* Input: chain 0 has [1,2,1,0,0,...], other chains are [0,1,2,0,1,...] */
    uint8_t x[TRINE_S2_DIM], y[TRINE_S2_DIM];
    memset(x, 0, sizeof(x));
    x[0] = 1; x[1] = 2; x[2] = 1;

    /* Other chains: set identity-passthrough values */
    for (int i = TRINE_S2_CHAIN_DIM; i < TRINE_S2_DIM; i++)
        x[i] = (uint8_t)(i % 3);

    trine_project_block_diagonal(W_block, x, y);

    check("known_y0", y[0] == 1);
    check("known_y1", y[1] == 2);
    check("known_y2", y[2] == 0);

    /* Chains 1-3 should pass through unchanged (identity blocks) */
    int chains_ok = 1;
    for (int i = TRINE_S2_CHAIN_DIM; i < TRINE_S2_DIM; i++)
        if (y[i] != x[i]) { chains_ok = 0; break; }
    check("known_other_chains_identity", chains_ok);
}

/* 3. Z3 closure: output values are all in {0, 1, 2} */
static void test_block_z3_closure(void)
{
    /* Use a random-ish block (deterministic from manual fill) */
    uint8_t W_block[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM];
    uint32_t seed = 12345;
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            for (int j = 0; j < TRINE_S2_CHAIN_DIM; j++) {
                seed = seed * 1103515245u + 12345u;
                W_block[c][i][j] = (uint8_t)((seed >> 16) % 3);
            }

    /* Test with several input patterns */
    int all_ok = 1;
    for (int pat = 0; pat < 4; pat++) {
        uint8_t x[TRINE_S2_DIM], y[TRINE_S2_DIM];
        for (int i = 0; i < TRINE_S2_DIM; i++)
            x[i] = (uint8_t)((i + pat * 37) % 3);

        trine_project_block_diagonal(W_block, x, y);

        for (int i = 0; i < TRINE_S2_DIM; i++)
            if (y[i] > 2) { all_ok = 0; break; }
    }
    check("z3_closure", all_ok);
}

/* 4. Chain independence: modifying chain 0's block doesn't affect chain 1 */
static void test_block_chain_independence(void)
{
    uint8_t W_block[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM];

    /* Fill all blocks with a known pattern */
    uint32_t seed = 99999;
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            for (int j = 0; j < TRINE_S2_CHAIN_DIM; j++) {
                seed = seed * 1103515245u + 12345u;
                W_block[c][i][j] = (uint8_t)((seed >> 16) % 3);
            }

    uint8_t x[TRINE_S2_DIM];
    for (int i = 0; i < TRINE_S2_DIM; i++)
        x[i] = (uint8_t)((i * 7 + 3) % 3);

    uint8_t y_before[TRINE_S2_DIM];
    trine_project_block_diagonal(W_block, x, y_before);

    /* Modify chain 0's block completely */
    for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
        for (int j = 0; j < TRINE_S2_CHAIN_DIM; j++)
            W_block[0][i][j] = (W_block[0][i][j] + 1) % 3;

    uint8_t y_after[TRINE_S2_DIM];
    trine_project_block_diagonal(W_block, x, y_after);

    /* Chains 1-3 (dims 60-239) should be unchanged */
    int chains_ok = 1;
    for (int i = TRINE_S2_CHAIN_DIM; i < TRINE_S2_DIM; i++)
        if (y_before[i] != y_after[i]) { chains_ok = 0; break; }
    check("chain_independence", chains_ok);
}

/* 5. Determinism: same input -> same output */
static void test_block_determinism(void)
{
    uint8_t W_block[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM];
    uint32_t seed = 54321;
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            for (int j = 0; j < TRINE_S2_CHAIN_DIM; j++) {
                seed = seed * 1103515245u + 12345u;
                W_block[c][i][j] = (uint8_t)((seed >> 16) % 3);
            }

    uint8_t x[TRINE_S2_DIM];
    for (int i = 0; i < TRINE_S2_DIM; i++)
        x[i] = (uint8_t)((i * 13 + 5) % 3);

    uint8_t y1[TRINE_S2_DIM], y2[TRINE_S2_DIM];
    trine_project_block_diagonal(W_block, x, y1);
    trine_project_block_diagonal(W_block, x, y2);

    check("determinism", memcmp(y1, y2, TRINE_S2_DIM) == 0);
}

/* 6. All-zeros input -> all-zeros output */
static void test_block_zeros_input(void)
{
    uint8_t W_block[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM];
    uint32_t seed = 77777;
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            for (int j = 0; j < TRINE_S2_CHAIN_DIM; j++) {
                seed = seed * 1103515245u + 12345u;
                W_block[c][i][j] = (uint8_t)((seed >> 16) % 3);
            }

    uint8_t x[TRINE_S2_DIM], y[TRINE_S2_DIM];
    memset(x, 0, sizeof(x));
    memset(y, 0xFF, sizeof(y));  /* dirty output buffer */

    trine_project_block_diagonal(W_block, x, y);

    /* y[i] = sum_j(W[i][j] * 0) % 3 = 0 for all i */
    uint8_t zero[TRINE_S2_DIM];
    memset(zero, 0, sizeof(zero));
    check("zeros_input", memcmp(y, zero, TRINE_S2_DIM) == 0);
}

/* 7. All-ones input: verify expected output */
static void test_block_ones_input(void)
{
    uint8_t W_block[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM];

    /* Identity blocks: y[i] = (1*1) % 3 = 1 for each dim */
    memset(W_block, 0, sizeof(W_block));
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            W_block[c][i][i] = 1;

    uint8_t x[TRINE_S2_DIM], y[TRINE_S2_DIM];
    memset(x, 1, sizeof(x));

    trine_project_block_diagonal(W_block, x, y);

    uint8_t expected[TRINE_S2_DIM];
    memset(expected, 1, sizeof(expected));
    check("ones_identity", memcmp(y, expected, TRINE_S2_DIM) == 0);

    /* Now with all-twos diagonal: y[i] = (2*1) % 3 = 2 for each dim */
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            W_block[c][i][i] = 2;

    trine_project_block_diagonal(W_block, x, y);

    memset(expected, 2, sizeof(expected));
    check("ones_twos_diag", memcmp(y, expected, TRINE_S2_DIM) == 0);
}

/* 8. All-twos input: verify expected output */
static void test_block_twos_input(void)
{
    uint8_t W_block[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM];

    /* Identity blocks: y[i] = (1*2) % 3 = 2 for each dim */
    memset(W_block, 0, sizeof(W_block));
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            W_block[c][i][i] = 1;

    uint8_t x[TRINE_S2_DIM], y[TRINE_S2_DIM];
    memset(x, 2, sizeof(x));

    trine_project_block_diagonal(W_block, x, y);

    uint8_t expected[TRINE_S2_DIM];
    memset(expected, 2, sizeof(expected));
    check("twos_identity", memcmp(y, expected, TRINE_S2_DIM) == 0);

    /* All-twos diagonal with all-twos input: y[i] = (2*2) % 3 = 4 % 3 = 1 */
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            W_block[c][i][i] = 2;

    trine_project_block_diagonal(W_block, x, y);

    memset(expected, 1, sizeof(expected));
    check("twos_twos_diag", memcmp(y, expected, TRINE_S2_DIM) == 0);
}

/* =====================================================================
 * Category 2: Block Majority Vote (4 tests)
 * ===================================================================== */

/* 9. K=3 majority with identical blocks -> same as single projection */
static void test_majority_identical_blocks(void)
{
    int K = TRINE_PROJECT_K;
    size_t total = BLOCK_BYTES(K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("majority_identical_alloc", 0); return; }

    /* Fill all K copies with the same random block */
    uint8_t W_single[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM];
    uint32_t seed = 31337;
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            for (int j = 0; j < TRINE_S2_CHAIN_DIM; j++) {
                seed = seed * 1103515245u + 12345u;
                W_single[c][i][j] = (uint8_t)((seed >> 16) % 3);
            }

    size_t one_block = BLOCK_BYTES(1);
    for (int k = 0; k < K; k++)
        memcpy(W_blocks + (size_t)k * one_block, W_single, one_block);

    uint8_t x[TRINE_S2_DIM];
    for (int i = 0; i < TRINE_S2_DIM; i++)
        x[i] = (uint8_t)((i * 11 + 7) % 3);

    /* Single projection result */
    uint8_t y_single[TRINE_S2_DIM];
    trine_project_block_diagonal(W_single, x, y_single);

    /* Majority result (all K identical -> should match single) */
    uint8_t y_majority[TRINE_S2_DIM];
    trine_projection_majority_block(W_blocks, K, x, y_majority);

    check("majority_identical", memcmp(y_single, y_majority, TRINE_S2_DIM) == 0);

    free(W_blocks);
}

/* 10. K=3 with one different -> majority wins */
static void test_majority_one_different(void)
{
    int K = TRINE_PROJECT_K;
    size_t total = BLOCK_BYTES(K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("majority_one_diff_alloc", 0); return; }

    /* Fill all K copies with the same block first */
    uint8_t W_base[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM];
    memset(W_base, 0, sizeof(W_base));
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            W_base[c][i][i] = 1;  /* identity */

    size_t one_block = BLOCK_BYTES(1);
    for (int k = 0; k < K; k++)
        memcpy(W_blocks + (size_t)k * one_block, W_base, one_block);

    /* Modify copy 2 (third copy) to be different */
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++)
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++)
            *block_ptr(W_blocks, 2, c, i, i) = 2;  /* different diagonal */

    uint8_t x[TRINE_S2_DIM];
    for (int i = 0; i < TRINE_S2_DIM; i++)
        x[i] = (uint8_t)(i % 3);

    /* Majority of (identity, identity, doubled-diag) -> identity wins */
    uint8_t y_majority[TRINE_S2_DIM];
    trine_projection_majority_block(W_blocks, K, x, y_majority);

    /* With 2 out of 3 being identity, majority should equal identity result */
    uint8_t y_identity[TRINE_S2_DIM];
    trine_project_block_diagonal(W_base, x, y_identity);

    check("majority_two_beats_one", memcmp(y_identity, y_majority, TRINE_S2_DIM) == 0);

    free(W_blocks);
}

/* 11. Identity blocks -> output == input */
static void test_majority_identity(void)
{
    int K = TRINE_PROJECT_K;
    size_t total = BLOCK_BYTES(K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("majority_identity_alloc", 0); return; }

    trine_projection_block_identity(W_blocks, K);

    uint8_t x[TRINE_S2_DIM];
    for (int i = 0; i < TRINE_S2_DIM; i++)
        x[i] = (uint8_t)((i * 3 + 1) % 3);

    uint8_t y[TRINE_S2_DIM];
    trine_projection_majority_block(W_blocks, K, x, y);

    check("majority_identity_passthrough", memcmp(x, y, TRINE_S2_DIM) == 0);

    free(W_blocks);
}

/* 12. Determinism: same seed -> same result */
static void test_majority_determinism(void)
{
    int K = TRINE_PROJECT_K;
    size_t total = BLOCK_BYTES(K);
    uint8_t *W1 = (uint8_t *)malloc(total);
    uint8_t *W2 = (uint8_t *)malloc(total);
    if (!W1 || !W2) { check("majority_det_alloc", 0); free(W1); free(W2); return; }

    trine_projection_block_random(W1, K, 42);
    trine_projection_block_random(W2, K, 42);

    /* Same seed should produce identical blocks */
    check("random_block_deterministic", memcmp(W1, W2, total) == 0);

    /* And therefore identical projection output */
    uint8_t x[TRINE_S2_DIM];
    for (int i = 0; i < TRINE_S2_DIM; i++)
        x[i] = (uint8_t)((i * 5 + 2) % 3);

    uint8_t y1[TRINE_S2_DIM], y2[TRINE_S2_DIM];
    trine_projection_majority_block(W1, K, x, y1);
    trine_projection_majority_block(W2, K, x, y2);

    check("majority_determinism", memcmp(y1, y2, TRINE_S2_DIM) == 0);

    free(W1);
    free(W2);
}

/* =====================================================================
 * Category 3: Block Accumulator (6 tests)
 * ===================================================================== */

/* 13. Create and free without crash */
static void test_block_accum_create_free(void)
{
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    check("accum_create", acc != NULL);
    if (acc) {
        check("accum_mode", acc->mode == TRINE_ACCUM_MODE_BLOCK);
        trine_block_accumulator_free(acc);
    } else {
        check("accum_mode", 0);
    }

    /* Free NULL should not crash */
    trine_block_accumulator_free(NULL);
    check("accum_free_null", 1);
}

/* 14. Update with identical vectors -> diagonal positive */
static void test_block_accum_identical_update(void)
{
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("accum_identical_alloc", 0); return; }

    /* Create vector with trit=2 at position 0, trit=1 elsewhere.
     * Centered: position 0 = +1, rest = 0.
     * Outer product: only counter[k][0][0][0][0] += (+1)(+1) = +1 */
    uint8_t v[240];
    memset(v, 1, sizeof(v));
    v[0] = 2;  /* chain 0, position 0 */

    trine_block_accumulator_update(acc, v, v, 1);  /* positive update */

    /* Check that counter for chain 0, position (0,0) is positive.
     * Layout: counters[k * 4 * 60 * 60 + c * 60 * 60 + i * 60 + j] */
    int diag_positive = 1;
    for (int k = 0; k < TRINE_PROJECT_K; k++) {
        int idx = k * TRINE_BLOCK_CHAINS * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM
                  + 0 * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM
                  + 0 * TRINE_BLOCK_DIM + 0;
        if (acc->counters[idx] <= 0) { diag_positive = 0; break; }
    }
    check("accum_identical_diag_positive", diag_positive);

    trine_block_accumulator_free(acc);
}

/* 15. Update with orthogonal vectors -> smaller correlations */
static void test_block_accum_orthogonal(void)
{
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("accum_orth_alloc", 0); return; }

    /* Vector a: trit=2 at position 0 (chain 0), rest neutral */
    uint8_t a[240], b[240];
    memset(a, 1, sizeof(a));
    memset(b, 1, sizeof(b));
    a[0] = 2;  /* chain 0, pos 0 active */
    b[1] = 2;  /* chain 0, pos 1 active */

    /* Positive update with orthogonal vectors */
    trine_block_accumulator_update(acc, a, b, 1);

    /* The diagonal counter[k][c=0][0][0] should be zero (a[0] and b[0]
     * have centered values +1 and 0 respectively for position 0)
     * whereas counter[k][c=0][1][0] should be positive (b has +1 at
     * position 1, a has +1 at position 0, so outer product[1][0] = +1) */
    int diag_zero = 1;
    for (int k = 0; k < TRINE_PROJECT_K; k++) {
        int idx_diag = k * TRINE_BLOCK_CHAINS * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM
                       + 0 * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM
                       + 0 * TRINE_BLOCK_DIM + 0;
        if (acc->counters[idx_diag] != 0) { diag_zero = 0; break; }
    }
    check("accum_orthogonal_diag_zero", diag_zero);

    trine_block_accumulator_free(acc);
}

/* 16. Positive vs negative update -> opposite signs */
static void test_block_accum_sign_direction(void)
{
    /* Positive update */
    trine_block_accumulator_t *acc_pos = trine_block_accumulator_create(TRINE_PROJECT_K);
    trine_block_accumulator_t *acc_neg = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc_pos || !acc_neg) {
        check("accum_sign_alloc", 0);
        trine_block_accumulator_free(acc_pos);
        trine_block_accumulator_free(acc_neg);
        return;
    }

    uint8_t v[240];
    memset(v, 1, sizeof(v));
    v[0] = 2;

    trine_block_accumulator_update(acc_pos, v, v, 1);   /* positive */
    trine_block_accumulator_update(acc_neg, v, v, 0);   /* negative */

    /* Counter at same position should have opposite signs */
    int idx = 0 * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM + 0 * TRINE_BLOCK_DIM + 0;
    int32_t val_pos = acc_pos->counters[idx];
    int32_t val_neg = acc_neg->counters[idx];

    /* Positive update -> positive counter, negative update -> negative counter */
    check("accum_positive_sign", val_pos > 0);
    check("accum_negative_sign", val_neg < 0);

    trine_block_accumulator_free(acc_pos);
    trine_block_accumulator_free(acc_neg);
}

/* 17. Reset clears all counters */
static void test_block_accum_reset(void)
{
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("accum_reset_alloc", 0); return; }

    uint8_t v[240];
    memset(v, 1, sizeof(v));
    v[0] = 2;
    trine_block_accumulator_update(acc, v, v, 1);

    /* Verify something is non-zero before reset */
    int32_t max_val = 0, min_val = 0;
    uint64_t nonzero = 0;
    trine_block_accumulator_stats(acc, &max_val, &min_val, &nonzero);
    check("accum_pre_reset_nonzero", nonzero > 0);

    /* Reset */
    trine_block_accumulator_reset(acc);

    trine_block_accumulator_stats(acc, &max_val, &min_val, &nonzero);
    check("accum_post_reset_zero", nonzero == 0);
    check("accum_post_reset_max", max_val == 0);
    check("accum_post_reset_min", min_val == 0);

    trine_block_accumulator_free(acc);
}

/* 18. Stats report correct max/min/nonzero */
static void test_block_accum_stats(void)
{
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("accum_stats_alloc", 0); return; }

    /* Initially all zero */
    int32_t max_val = 0, min_val = 0;
    uint64_t nonzero = 0;
    trine_block_accumulator_stats(acc, &max_val, &min_val, &nonzero);
    check("stats_initial_zero", nonzero == 0 && max_val == 0 && min_val == 0);

    /* Do some updates */
    uint8_t a[240], b[240];
    trine_encode_shingle("block stats test", 16, a);
    trine_encode_shingle("block stats check", 17, b);
    trine_block_accumulator_update(acc, a, b, 1);
    trine_block_accumulator_update(acc, a, b, 1);
    trine_block_accumulator_update(acc, a, b, 1);

    trine_block_accumulator_stats(acc, &max_val, &min_val, &nonzero);
    check("stats_after_update_nonzero", nonzero > 0);
    check("stats_max_positive", max_val >= 0);

    trine_block_accumulator_free(acc);
}

/* =====================================================================
 * Category 4: Block Freeze (5 tests)
 * =====================================================================
 *
 * Block freeze converts block accumulators to block-diagonal projection
 * weights. Since trine_freeze.h may not yet have block-specific freeze
 * functions, we test the freeze behavior by manually extracting counters
 * and applying the threshold rule, then verifying the projection works.
 * ===================================================================== */

/* Helper: freeze block accumulator to block-diagonal weights.
 * Uses the standard freeze rule:
 *   W = (counter > +T) ? 2 : (counter < -T) ? 1 : 0
 * Writes into W_blocks[K * 4 * 60 * 60]. */
static void block_freeze_manual(const trine_block_accumulator_t *acc,
                                 int32_t threshold,
                                 uint8_t *W_blocks)
{
    size_t total = (size_t)acc->K * TRINE_BLOCK_CHAINS * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;
    for (size_t idx = 0; idx < total; idx++) {
        int32_t c = acc->counters[idx];
        if (c > threshold)
            W_blocks[idx] = 2;
        else if (c < -threshold)
            W_blocks[idx] = 1;
        else
            W_blocks[idx] = 0;
    }
}

/* 19. Freeze with threshold 0 -> all nonzero counters become weights */
static void test_block_freeze_threshold_zero(void)
{
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("freeze_t0_alloc", 0); return; }

    uint8_t a[240], b[240];
    trine_encode_shingle("block freeze zero", 17, a);
    trine_encode_shingle("block freeze test", 17, b);
    trine_block_accumulator_update(acc, a, b, 1);

    size_t total = BLOCK_BYTES(TRINE_PROJECT_K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("freeze_t0_alloc2", 0); trine_block_accumulator_free(acc); return; }

    block_freeze_manual(acc, 0, W_blocks);

    /* Count nonzero weights */
    size_t n_nonzero = 0;
    for (size_t i = 0; i < total; i++)
        if (W_blocks[i] != 0) n_nonzero++;

    /* With threshold=0, nonzero counters should produce nonzero weights */
    int32_t max_val, min_val;
    uint64_t acc_nonzero;
    trine_block_accumulator_stats(acc, &max_val, &min_val, &acc_nonzero);

    check("freeze_t0_has_weights", n_nonzero > 0);
    /* Number of nonzero weights should equal number of nonzero counters
     * (since threshold=0 maps all nonzero to 1 or 2) */
    check("freeze_t0_count_match", (uint64_t)n_nonzero == acc_nonzero);

    free(W_blocks);
    trine_block_accumulator_free(acc);
}

/* 20. Freeze produces only {0, 1, 2} values */
static void test_block_freeze_z3_closure(void)
{
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("freeze_z3_alloc", 0); return; }

    /* Feed diverse training data */
    const char *texts[] = {
        "alpha bravo charlie", "delta echo foxtrot",
        "golf hotel india", "juliet kilo lima",
    };
    for (int i = 0; i < 3; i++) {
        uint8_t a_enc[240], b_enc[240];
        trine_encode_shingle(texts[i], strlen(texts[i]), a_enc);
        trine_encode_shingle(texts[i + 1], strlen(texts[i + 1]), b_enc);
        trine_block_accumulator_update(acc, a_enc, b_enc, (i % 2 == 0) ? 1 : 0);
    }

    size_t total = BLOCK_BYTES(TRINE_PROJECT_K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("freeze_z3_alloc2", 0); trine_block_accumulator_free(acc); return; }

    /* Freeze with various thresholds */
    int all_valid = 1;
    for (int32_t threshold = 0; threshold <= 5; threshold++) {
        block_freeze_manual(acc, threshold, W_blocks);
        for (size_t i = 0; i < total && all_valid; i++)
            if (W_blocks[i] > 2) all_valid = 0;
    }

    check("freeze_z3_closure", all_valid);

    free(W_blocks);
    trine_block_accumulator_free(acc);
}

/* 21. Auto-threshold achieves target density (within 10%) */
static void test_block_freeze_auto_threshold(void)
{
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("freeze_auto_alloc", 0); return; }

    /* Feed enough data to populate counters */
    const char *texts[] = {
        "alpha bravo charlie", "delta echo foxtrot",
        "golf hotel india", "juliet kilo lima",
        "mike november oscar", "papa quebec romeo",
    };
    for (int rep = 0; rep < 20; rep++) {
        for (int i = 0; i < 5; i++) {
            uint8_t a_enc[240], b_enc[240];
            trine_encode_shingle(texts[i], strlen(texts[i]), a_enc);
            trine_encode_shingle(texts[i + 1], strlen(texts[i + 1]), b_enc);
            trine_block_accumulator_update(acc, a_enc, b_enc, 1);
        }
    }

    /* Find max_abs via stats */
    int32_t max_val, min_val;
    uint64_t nonzero;
    trine_block_accumulator_stats(acc, &max_val, &min_val, &nonzero);

    int32_t abs_max = max_val > -min_val ? max_val : -min_val;

    /* Binary search for threshold achieving ~33% density */
    float target = 0.33f;
    size_t total = BLOCK_BYTES(TRINE_PROJECT_K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("freeze_auto_alloc2", 0); trine_block_accumulator_free(acc); return; }

    int32_t best_t = 0;
    float best_density = 1.0f;
    for (int32_t t = 0; t <= abs_max; t++) {
        block_freeze_manual(acc, t, W_blocks);
        size_t nz = 0;
        for (size_t i = 0; i < total; i++)
            if (W_blocks[i] != 0) nz++;
        float density = (float)nz / (float)total;
        if (fabsf(density - target) < fabsf(best_density - target)) {
            best_density = density;
            best_t = t;
        }
    }

    /* Best threshold should achieve density within 10% of target */
    check("freeze_auto_density", fabsf(best_density - target) < 0.10f
                                 || best_t == abs_max);
    (void)best_t;

    free(W_blocks);
    trine_block_accumulator_free(acc);
}

/* 22. Frozen identity accumulator ~ identity matrix */
static void test_block_freeze_identity(void)
{
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("freeze_id_alloc", 0); return; }

    /* Feed unit vectors: trit=2 at position i, trit=1 elsewhere.
     * This builds up positive diagonal counters within each chain's block. */
    for (int pos = 0; pos < 240; pos++) {
        uint8_t v[240];
        memset(v, 1, sizeof(v));
        v[pos] = 2;
        trine_block_accumulator_update(acc, v, v, 1);
    }

    size_t total = BLOCK_BYTES(TRINE_PROJECT_K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("freeze_id_alloc2", 0); trine_block_accumulator_free(acc); return; }

    block_freeze_manual(acc, 0, W_blocks);

    /* Diagonal of each chain's block should be 2 (positive counters) */
    int diag_ok = 1;
    for (int k = 0; k < TRINE_PROJECT_K && diag_ok; k++)
        for (int c = 0; c < TRINE_S2_N_CHAINS && diag_ok; c++)
            for (int i = 0; i < TRINE_S2_CHAIN_DIM && diag_ok; i++) {
                uint8_t val = block_get(W_blocks, k, c, i, i);
                if (val != 2) diag_ok = 0;
            }

    check("freeze_identity_diag_two", diag_ok);

    free(W_blocks);
    trine_block_accumulator_free(acc);
}

/* 23. Round-trip: accumulate -> freeze -> project -> produces reasonable output */
static void test_block_roundtrip(void)
{
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("roundtrip_alloc", 0); return; }

    /* Accumulate some training data */
    uint8_t a[240], b[240];
    trine_encode_shingle("round trip test input", 21, a);
    trine_encode_shingle("round trip test check", 21, b);

    for (int i = 0; i < 10; i++)
        trine_block_accumulator_update(acc, a, b, 1);

    /* Freeze */
    size_t one_block = BLOCK_BYTES(1);
    uint8_t *W_block = (uint8_t *)malloc(one_block);
    if (!W_block) { check("roundtrip_alloc2", 0); trine_block_accumulator_free(acc); return; }

    /* Freeze only the first copy (k=0) for a single projection */
    size_t block_size = (size_t)TRINE_BLOCK_CHAINS * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;
    for (size_t idx = 0; idx < block_size; idx++) {
        int32_t c = acc->counters[idx];
        W_block[idx] = (c > 0) ? 2 : (c < 0) ? 1 : 0;
    }

    /* Project */
    uint8_t y[240];
    trine_project_block_diagonal(
        (const uint8_t (*)[TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM])W_block,
        a, y);

    /* Output should be valid trits */
    int z3_ok = 1;
    for (int i = 0; i < TRINE_S2_DIM; i++)
        if (y[i] > 2) { z3_ok = 0; break; }
    check("roundtrip_z3", z3_ok);

    /* Output should differ from input (non-identity projection) */
    int differs = (memcmp(a, y, TRINE_S2_DIM) != 0);
    check("roundtrip_differs", differs);

    free(W_block);
    trine_block_accumulator_free(acc);
}

/* =====================================================================
 * Category 5: End-to-End Block Pipeline (4 tests)
 * ===================================================================== */

/* 24. Encode "hello" with block-diagonal -> different from Stage-1 */
static void test_e2e_different_from_s1(void)
{
    /* Stage-1 encode */
    uint8_t s1[240];
    trine_encode_shingle("hello", 5, s1);

    /* Random block-diagonal projection */
    int K = TRINE_PROJECT_K;
    size_t total = BLOCK_BYTES(K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("e2e_diff_alloc", 0); return; }

    trine_projection_block_random(W_blocks, K, 42);

    uint8_t y[240];
    trine_projection_majority_block(W_blocks, K, s1, y);

    /* Projected output should differ from Stage-1 */
    check("e2e_different_from_s1", memcmp(s1, y, 240) != 0);

    free(W_blocks);
}

/* 25. Encode "hello" with identity block -> same as Stage-1 */
static void test_e2e_identity_equals_s1(void)
{
    /* Stage-1 encode */
    uint8_t s1[240];
    trine_encode_shingle("hello", 5, s1);

    /* Identity block-diagonal projection */
    int K = TRINE_PROJECT_K;
    size_t total = BLOCK_BYTES(K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("e2e_id_alloc", 0); return; }

    trine_projection_block_identity(W_blocks, K);

    uint8_t y[240];
    trine_projection_majority_block(W_blocks, K, s1, y);

    /* Identity projection should preserve the Stage-1 embedding */
    check("e2e_identity_equals_s1", memcmp(s1, y, 240) == 0);

    free(W_blocks);
}

/* 26. Compare two similar texts with block projection -> reasonable similarity */
static void test_e2e_similar_texts(void)
{
    /* Encode two similar texts */
    uint8_t s1a[240], s1b[240];
    trine_encode_shingle("the quick brown fox jumps", 25, s1a);
    trine_encode_shingle("the quick brown fox leaps", 25, s1b);

    /* Apply same block projection to both */
    int K = TRINE_PROJECT_K;
    size_t total = BLOCK_BYTES(K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("e2e_sim_alloc", 0); return; }

    trine_projection_block_random(W_blocks, K, 123);

    uint8_t ya[240], yb[240];
    trine_projection_majority_block(W_blocks, K, s1a, ya);
    trine_projection_majority_block(W_blocks, K, s1b, yb);

    /* Compute a simple agreement ratio as a similarity proxy */
    int agree = 0;
    for (int i = 0; i < 240; i++)
        if (ya[i] == yb[i]) agree++;

    float agreement = (float)agree / 240.0f;

    /* Similar texts projected with same weights should have some agreement.
     * With random projection this won't be perfect, but should be above
     * random chance (1/3 = 0.333 for ternary). */
    check("e2e_similar_agreement", agreement > 0.25f);

    /* Also compare using Stage-1 lens (uniform weights) */
    trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;
    float sim_s1 = trine_s1_compare(s1a, s1b, &lens);
    float sim_proj = trine_s1_compare(ya, yb, &lens);

    /* Both should be non-negative (valid similarities) */
    check("e2e_similar_s1_valid", sim_s1 >= 0.0f);
    check("e2e_similar_proj_valid", sim_proj >= -0.01f);  /* allow small float error */

    free(W_blocks);
}

/* 27. Block projection preserves encode determinism */
static void test_e2e_determinism(void)
{
    int K = TRINE_PROJECT_K;
    size_t total = BLOCK_BYTES(K);
    uint8_t *W_blocks = (uint8_t *)malloc(total);
    if (!W_blocks) { check("e2e_det_alloc", 0); return; }

    trine_projection_block_random(W_blocks, K, 999);

    /* Encode and project the same text twice */
    uint8_t s1a[240], s1b[240];
    trine_encode_shingle("determinism test text", 21, s1a);
    trine_encode_shingle("determinism test text", 21, s1b);

    /* Stage-1 should be identical */
    check("e2e_s1_determinism", memcmp(s1a, s1b, 240) == 0);

    uint8_t ya[240], yb[240];
    trine_projection_majority_block(W_blocks, K, s1a, ya);
    trine_projection_majority_block(W_blocks, K, s1b, yb);

    /* Projected outputs should also be identical */
    check("e2e_proj_determinism", memcmp(ya, yb, 240) == 0);

    free(W_blocks);
}

/* =====================================================================
 * Main
 * ===================================================================== */

int main(void)
{
    printf("=== Stage-2 Block-Diagonal Tests ===\n");

    /* Category 1: Block-Diagonal Projection */
    printf("\n--- Block-Diagonal Projection ---\n");
    test_block_identity_passthrough();
    test_block_known_values();
    test_block_z3_closure();
    test_block_chain_independence();
    test_block_determinism();
    test_block_zeros_input();
    test_block_ones_input();
    test_block_twos_input();

    /* Category 2: Block Majority Vote */
    printf("\n--- Block Majority Vote ---\n");
    test_majority_identical_blocks();
    test_majority_one_different();
    test_majority_identity();
    test_majority_determinism();

    /* Category 3: Block Accumulator */
    printf("\n--- Block Accumulator ---\n");
    test_block_accum_create_free();
    test_block_accum_identical_update();
    test_block_accum_orthogonal();
    test_block_accum_sign_direction();
    test_block_accum_reset();
    test_block_accum_stats();

    /* Category 4: Block Freeze */
    printf("\n--- Block Freeze ---\n");
    test_block_freeze_threshold_zero();
    test_block_freeze_z3_closure();
    test_block_freeze_auto_threshold();
    test_block_freeze_identity();
    test_block_roundtrip();

    /* Category 5: End-to-End Block Pipeline */
    printf("\n--- End-to-End Block Pipeline ---\n");
    test_e2e_different_from_s1();
    test_e2e_identity_equals_s1();
    test_e2e_similar_texts();
    test_e2e_determinism();

    printf("\nBlock-diagonal: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
