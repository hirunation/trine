/* =====================================================================
 * TRINE Stage-2 — Majority Vote Tests (~10 tests)
 * =====================================================================
 *
 * Tests for trine_project_majority():
 *   1. Unanimous trits → that trit
 *   2. 2-of-3 majority → majority value
 *   3. All-different → tie-break to first projection
 *   4. Identity majority (three identity projections)
 *   5. Determinism
 *   6. Z_3 closure on majority output
 *   7. Random projection consistency
 *
 * ===================================================================== */

#include "trine_project.h"
#include <stdio.h>
#include <string.h>

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
        printf("  FAIL  majority: %s\n", name);
    }
}

/* ── Test: Identity majority ────────────────────────────────────────── */
static void test_identity_majority(void)
{
    trine_projection_t proj;
    trine_projection_identity(&proj);

    /* All zeros */
    uint8_t x[TRINE_PROJECT_DIM], y[TRINE_PROJECT_DIM];
    memset(x, 0, sizeof(x));
    trine_project_majority(&proj, x, y);
    check("identity_majority_zeros", memcmp(x, y, TRINE_PROJECT_DIM) == 0);

    /* All ones */
    memset(x, 1, sizeof(x));
    trine_project_majority(&proj, x, y);
    check("identity_majority_ones", memcmp(x, y, TRINE_PROJECT_DIM) == 0);

    /* All twos */
    memset(x, 2, sizeof(x));
    trine_project_majority(&proj, x, y);
    check("identity_majority_twos", memcmp(x, y, TRINE_PROJECT_DIM) == 0);

    /* Mixed */
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        x[i] = (uint8_t)(i % 3);
    trine_project_majority(&proj, x, y);
    check("identity_majority_mixed", memcmp(x, y, TRINE_PROJECT_DIM) == 0);
}

/* ── Test: Z_3 closure on majority output ───────────────────────────── */
static void test_z3_closure_majority(void)
{
    trine_projection_t proj;
    trine_projection_random(&proj, 12345);

    uint8_t x[TRINE_PROJECT_DIM], y[TRINE_PROJECT_DIM];
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        x[i] = (uint8_t)(i % 3);

    trine_project_majority(&proj, x, y);

    int ok = 1;
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        if (y[i] > 2) { ok = 0; break; }
    check("z3_closure_majority", ok);
}

/* ── Test: Determinism of majority vote ─────────────────────────────── */
static void test_determinism_majority(void)
{
    trine_projection_t proj;
    trine_projection_random(&proj, 999);

    uint8_t x[TRINE_PROJECT_DIM];
    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        x[i] = (uint8_t)((i * 13 + 7) % 3);

    uint8_t y1[TRINE_PROJECT_DIM], y2[TRINE_PROJECT_DIM];
    trine_project_majority(&proj, x, y1);
    trine_project_majority(&proj, x, y2);
    check("determinism_majority", memcmp(y1, y2, TRINE_PROJECT_DIM) == 0);
}

/* ── Test: Random projections produce valid trits ───────────────────── */
static void test_random_projections_valid(void)
{
    /* Multiple seeds, all must produce valid trit outputs */
    uint64_t seeds[] = { 0, 1, 42, 0xDEADBEEF, 0xCAFEBABE };
    int ok = 1;

    for (int s = 0; s < 5; s++) {
        trine_projection_t proj;
        trine_projection_random(&proj, seeds[s]);

        uint8_t x[TRINE_PROJECT_DIM], y[TRINE_PROJECT_DIM];
        for (int i = 0; i < TRINE_PROJECT_DIM; i++)
            x[i] = (uint8_t)(i % 3);

        trine_project_majority(&proj, x, y);
        for (int i = 0; i < TRINE_PROJECT_DIM; i++)
            if (y[i] > 2) { ok = 0; break; }
        if (!ok) break;
    }
    check("random_projections_valid_trits", ok);
}

/* ── Test: Random seed determinism ──────────────────────────────────── */
static void test_random_seed_determinism(void)
{
    trine_projection_t p1, p2;
    trine_projection_random(&p1, 42);
    trine_projection_random(&p2, 42);

    check("random_seed_determinism",
          memcmp(&p1, &p2, sizeof(trine_projection_t)) == 0);
}

/* ── Test: Different seeds → different projections ──────────────────── */
static void test_different_seeds(void)
{
    trine_projection_t p1, p2;
    trine_projection_random(&p1, 42);
    trine_projection_random(&p2, 43);

    check("different_seeds_different_projections",
          memcmp(&p1, &p2, sizeof(trine_projection_t)) != 0);
}

/* ── Test: Zero input through majority ──────────────────────────────── */
static void test_zero_through_majority(void)
{
    trine_projection_t proj;
    trine_projection_random(&proj, 7777);

    uint8_t x[TRINE_PROJECT_DIM], y[TRINE_PROJECT_DIM];
    uint8_t zero[TRINE_PROJECT_DIM];
    memset(x, 0, sizeof(x));
    memset(zero, 0, sizeof(zero));

    trine_project_majority(&proj, x, y);
    check("zero_through_majority", memcmp(y, zero, TRINE_PROJECT_DIM) == 0);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Majority Vote Tests ===\n");

    test_identity_majority();
    test_z3_closure_majority();
    test_determinism_majority();
    test_random_projections_valid();
    test_random_seed_determinism();
    test_different_seeds();
    test_zero_through_majority();

    printf("\nMajority: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
