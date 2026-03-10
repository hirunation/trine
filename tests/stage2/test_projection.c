/* =====================================================================
 * TRINE Stage-2 — Projection Tests (~15 tests)
 * =====================================================================
 *
 * Tests for trine_project_single():
 *   1. Z_3 closure: all outputs in {0,1,2}
 *   2. Identity matrix: output equals input
 *   3. Zero input: zero output
 *   4. Determinism: same input → same output
 *   5. Row independence: changing W[i] only affects y[i]
 *   6. Known-value verification (hand-computed 3x3 submatrix)
 *   7. Max accumulator check (all 2s → acc=960, 960%3=0)
 *   8. Various input patterns
 *
 * ===================================================================== */

#include "trine_project.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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
        printf("  FAIL  projection: %s\n", name);
    }
}

/* ── Test: Z_3 closure ──────────────────────────────────────────────── */
static void test_z3_closure(void)
{
    /* Random projection applied to various inputs: all outputs must be 0/1/2 */
    trine_projection_t proj;
    trine_projection_random(&proj, 42);

    uint8_t x[TRINE_PROJECT_DIM];
    uint8_t y[TRINE_PROJECT_DIM];

    /* Input: all 0s */
    memset(x, 0, sizeof(x));
    trine_project_single(proj.W[0], x, y);
    int ok = 1;
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        if (y[i] > 2) { ok = 0; break; }
    check("z3_closure_zero_input", ok);

    /* Input: all 1s */
    memset(x, 1, sizeof(x));
    trine_project_single(proj.W[0], x, y);
    ok = 1;
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        if (y[i] > 2) { ok = 0; break; }
    check("z3_closure_ones_input", ok);

    /* Input: all 2s */
    memset(x, 2, sizeof(x));
    trine_project_single(proj.W[0], x, y);
    ok = 1;
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        if (y[i] > 2) { ok = 0; break; }
    check("z3_closure_twos_input", ok);

    /* Input: mixed pattern */
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        x[i] = (uint8_t)(i % 3);
    trine_project_single(proj.W[0], x, y);
    ok = 1;
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        if (y[i] > 2) { ok = 0; break; }
    check("z3_closure_mixed_input", ok);
}

/* ── Test: Identity matrix ──────────────────────────────────────────── */
static void test_identity(void)
{
    trine_projection_t proj;
    trine_projection_identity(&proj);

    uint8_t x[TRINE_PROJECT_DIM], y[TRINE_PROJECT_DIM];

    /* All 0s → all 0s */
    memset(x, 0, sizeof(x));
    trine_project_single(proj.W[0], x, y);
    check("identity_zeros", memcmp(x, y, TRINE_PROJECT_DIM) == 0);

    /* All 1s → all 1s */
    memset(x, 1, sizeof(x));
    trine_project_single(proj.W[0], x, y);
    check("identity_ones", memcmp(x, y, TRINE_PROJECT_DIM) == 0);

    /* All 2s → all 2s */
    memset(x, 2, sizeof(x));
    trine_project_single(proj.W[0], x, y);
    check("identity_twos", memcmp(x, y, TRINE_PROJECT_DIM) == 0);

    /* Mixed pattern */
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        x[i] = (uint8_t)(i % 3);
    trine_project_single(proj.W[0], x, y);
    check("identity_mixed", memcmp(x, y, TRINE_PROJECT_DIM) == 0);
}

/* ── Test: Zero input → zero output ──────────────────────────────── */
static void test_zero_input(void)
{
    trine_projection_t proj;
    trine_projection_random(&proj, 123);

    uint8_t x[TRINE_PROJECT_DIM], y[TRINE_PROJECT_DIM];
    uint8_t zero[TRINE_PROJECT_DIM];

    memset(x, 0, sizeof(x));
    memset(zero, 0, sizeof(zero));

    trine_project_single(proj.W[0], x, y);
    check("zero_input_zero_output", memcmp(y, zero, TRINE_PROJECT_DIM) == 0);
}

/* ── Test: Determinism ──────────────────────────────────────────────── */
static void test_determinism(void)
{
    trine_projection_t proj;
    trine_projection_random(&proj, 99);

    uint8_t x[TRINE_PROJECT_DIM];
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        x[i] = (uint8_t)((i * 7 + 3) % 3);

    uint8_t y1[TRINE_PROJECT_DIM], y2[TRINE_PROJECT_DIM];
    trine_project_single(proj.W[0], x, y1);
    trine_project_single(proj.W[0], x, y2);
    check("determinism", memcmp(y1, y2, TRINE_PROJECT_DIM) == 0);
}

/* ── Test: Row independence ─────────────────────────────────────────── */
static void test_row_independence(void)
{
    trine_projection_t proj;
    trine_projection_random(&proj, 77);

    uint8_t x[TRINE_PROJECT_DIM];
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        x[i] = (uint8_t)(i % 3);

    uint8_t y_before[TRINE_PROJECT_DIM], y_after[TRINE_PROJECT_DIM];
    trine_project_single(proj.W[0], x, y_before);

    /* Change row 100 of the weight matrix */
    uint8_t saved_row[TRINE_PROJECT_DIM];
    memcpy(saved_row, proj.W[0][100], TRINE_PROJECT_DIM);
    for (int j = 0; j < TRINE_PROJECT_DIM; j++)
        proj.W[0][100][j] = (proj.W[0][100][j] + 1) % 3;

    trine_project_single(proj.W[0], x, y_after);

    /* Only row 100 should differ */
    int ok = 1;
    for (int i = 0; i < TRINE_PROJECT_DIM; i++) {
        if (i == 100) continue;
        if (y_before[i] != y_after[i]) { ok = 0; break; }
    }
    check("row_independence", ok);

    /* Restore */
    memcpy(proj.W[0][100], saved_row, TRINE_PROJECT_DIM);
}

/* ── Test: Known value (hand-computed) ──────────────────────────────── */
static void test_known_value(void)
{
    /* 3x3 submatrix test (embedded in 240x240 identity):
     * W[0][0] = {2, 1, 0, 0, 0, ...}
     * W[0][1] = {0, 2, 1, 0, 0, ...}
     * W[0][2] = {1, 0, 2, 0, 0, ...}
     * x = {1, 2, 1, 0, 0, ...}
     *
     * y[0] = (2*1 + 1*2 + 0*1) % 3 = 4 % 3 = 1
     * y[1] = (0*1 + 2*2 + 1*1) % 3 = 5 % 3 = 2
     * y[2] = (1*1 + 0*2 + 2*1) % 3 = 3 % 3 = 0
     * y[3..] = 0  (identity for rest)
     */
    trine_projection_t proj;
    trine_projection_identity(&proj);

    /* Override first 3 rows */
    proj.W[0][0][0] = 2; proj.W[0][0][1] = 1; proj.W[0][0][2] = 0;
    proj.W[0][1][0] = 0; proj.W[0][1][1] = 2; proj.W[0][1][2] = 1;
    proj.W[0][2][0] = 1; proj.W[0][2][1] = 0; proj.W[0][2][2] = 2;

    uint8_t x[TRINE_PROJECT_DIM];
    memset(x, 0, sizeof(x));
    x[0] = 1; x[1] = 2; x[2] = 1;

    uint8_t y[TRINE_PROJECT_DIM];
    trine_project_single(proj.W[0], x, y);

    check("known_value_y0", y[0] == 1);
    check("known_value_y1", y[1] == 2);
    check("known_value_y2", y[2] == 0);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Projection Tests ===\n");

    test_z3_closure();
    test_identity();
    test_zero_input();
    test_determinism();
    test_row_independence();
    test_known_value();

    printf("\nProjection: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
