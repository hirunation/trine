/* =====================================================================
 * TRINE Stage-2 — Cascade Tests (~10 tests)
 * =====================================================================
 *
 * Tests for trine_learned_cascade:
 *   1. Identity cascade (n_cells=0) → output = input
 *   2. Determinism (same topology + input + depth = same output)
 *   3. Depth consistency (depth-N = N individual ticks)
 *   4. Valid trits at every depth
 *   5. Random topology produces non-trivial mixing
 *   6. Reset/reuse produces identical results
 *   7. Topology generators produce valid cell params
 *   8. Layered topology
 *   9. Chain-local topology
 *
 * ===================================================================== */

#include "trine_learned_cascade.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Topology generators (from trine_topology_gen.c) */
extern void trine_topology_random(trine_learned_cascade_t *lc, uint64_t seed);
extern void trine_topology_layered(trine_learned_cascade_t *lc, uint64_t seed);
extern void trine_topology_chain_local(trine_learned_cascade_t *lc, uint64_t seed);

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
        printf("  FAIL  cascade: %s\n", name);
    }
}

static int all_valid_trits(const uint8_t *v, int n)
{
    for (int i = 0; i < n; i++)
        if (v[i] > 2) return 0;
    return 1;
}

/* ── Test: Identity cascade ─────────────────────────────────────────── */
static void test_identity_cascade(void)
{
    trine_cascade_config_t cfg = { .n_cells = 0, .max_depth = 32 };
    trine_learned_cascade_t *lc = trine_learned_cascade_create(&cfg);
    check("identity_create", lc != NULL);

    uint8_t in[TRINE_CASCADE_DIM], out[TRINE_CASCADE_DIM];
    for (int i = 0; i < TRINE_CASCADE_DIM; i++)
        in[i] = (uint8_t)(i % 3);

    trine_learned_cascade_tick(lc, in, out);
    check("identity_passthrough", memcmp(in, out, TRINE_CASCADE_DIM) == 0);

    trine_learned_cascade_free(lc);
}

/* ── Test: Determinism ──────────────────────────────────────────────── */
static void test_determinism(void)
{
    trine_cascade_config_t cfg = { .n_cells = 256, .max_depth = 32 };
    trine_learned_cascade_t *lc = trine_learned_cascade_create(&cfg);
    trine_topology_random(lc, 42);

    uint8_t in[TRINE_CASCADE_DIM];
    for (int i = 0; i < TRINE_CASCADE_DIM; i++)
        in[i] = (uint8_t)(i % 3);

    uint8_t out1[TRINE_CASCADE_DIM], out2[TRINE_CASCADE_DIM];
    trine_learned_cascade_tick(lc, in, out1);
    trine_learned_cascade_tick(lc, in, out2);
    check("determinism", memcmp(out1, out2, TRINE_CASCADE_DIM) == 0);

    trine_learned_cascade_free(lc);
}

/* ── Test: Depth consistency (N ticks = depth-N) ────────────────────── */
static void test_depth_consistency(void)
{
    trine_cascade_config_t cfg = { .n_cells = 128, .max_depth = 32 };
    trine_learned_cascade_t *lc = trine_learned_cascade_create(&cfg);
    trine_topology_random(lc, 77);

    uint8_t in[TRINE_CASCADE_DIM];
    for (int i = 0; i < TRINE_CASCADE_DIM; i++)
        in[i] = (uint8_t)((i * 7) % 3);

    /* Apply 4 individual ticks */
    uint8_t cur[TRINE_CASCADE_DIM], nxt[TRINE_CASCADE_DIM];
    memcpy(cur, in, TRINE_CASCADE_DIM);
    for (int d = 0; d < 4; d++) {
        trine_learned_cascade_tick(lc, cur, nxt);
        memcpy(cur, nxt, TRINE_CASCADE_DIM);
    }

    /* Compare with 4 fresh sequential ticks from same input */
    uint8_t cur2[TRINE_CASCADE_DIM], nxt2[TRINE_CASCADE_DIM];
    memcpy(cur2, in, TRINE_CASCADE_DIM);
    for (int d = 0; d < 4; d++) {
        trine_learned_cascade_tick(lc, cur2, nxt2);
        memcpy(cur2, nxt2, TRINE_CASCADE_DIM);
    }

    check("depth_consistency_4ticks", memcmp(cur, cur2, TRINE_CASCADE_DIM) == 0);

    trine_learned_cascade_free(lc);
}

/* ── Test: Valid trits at every depth ───────────────────────────────── */
static void test_valid_trits_all_depths(void)
{
    trine_cascade_config_t cfg = { .n_cells = 512, .max_depth = 32 };
    trine_learned_cascade_t *lc = trine_learned_cascade_create(&cfg);
    trine_topology_random(lc, 555);

    uint8_t cur[TRINE_CASCADE_DIM], nxt[TRINE_CASCADE_DIM];
    for (int i = 0; i < TRINE_CASCADE_DIM; i++)
        cur[i] = (uint8_t)(i % 3);

    int ok = 1;
    for (int d = 0; d < 16; d++) {
        trine_learned_cascade_tick(lc, cur, nxt);
        if (!all_valid_trits(nxt, TRINE_CASCADE_DIM)) { ok = 0; break; }
        memcpy(cur, nxt, TRINE_CASCADE_DIM);
    }
    check("valid_trits_16_depths", ok);

    trine_learned_cascade_free(lc);
}

/* ── Test: Random topology produces non-trivial mixing ──────────────── */
static void test_nontrivial_mixing(void)
{
    trine_cascade_config_t cfg = { .n_cells = 512, .max_depth = 32 };
    trine_learned_cascade_t *lc = trine_learned_cascade_create(&cfg);
    trine_topology_random(lc, 1234);

    uint8_t in[TRINE_CASCADE_DIM], out[TRINE_CASCADE_DIM];
    for (int i = 0; i < TRINE_CASCADE_DIM; i++)
        in[i] = (uint8_t)(i % 3);

    trine_learned_cascade_tick(lc, in, out);

    /* Output should differ from input (non-trivial mixing) */
    check("nontrivial_mixing", memcmp(in, out, TRINE_CASCADE_DIM) != 0);

    trine_learned_cascade_free(lc);
}

/* ── Test: Reset/reuse ──────────────────────────────────────────────── */
static void test_reuse(void)
{
    trine_cascade_config_t cfg = { .n_cells = 256, .max_depth = 32 };
    trine_learned_cascade_t *lc = trine_learned_cascade_create(&cfg);
    trine_topology_random(lc, 321);

    uint8_t in[TRINE_CASCADE_DIM], out1[TRINE_CASCADE_DIM], out2[TRINE_CASCADE_DIM];
    for (int i = 0; i < TRINE_CASCADE_DIM; i++)
        in[i] = (uint8_t)((i + 5) % 3);

    /* Use once */
    trine_learned_cascade_tick(lc, in, out1);

    /* Use a different input */
    uint8_t other[TRINE_CASCADE_DIM], tmp[TRINE_CASCADE_DIM];
    memset(other, 1, sizeof(other));
    trine_learned_cascade_tick(lc, other, tmp);

    /* Reuse with original input */
    trine_learned_cascade_tick(lc, in, out2);
    check("reuse_same_result", memcmp(out1, out2, TRINE_CASCADE_DIM) == 0);

    trine_learned_cascade_free(lc);
}

/* ── Test: Topology generators produce valid params ─────────────────── */
static void test_topology_valid(void)
{
    trine_cascade_config_t cfg = { .n_cells = 256, .max_depth = 32 };

    /* Random topology */
    {
        trine_learned_cascade_t *lc = trine_learned_cascade_create(&cfg);
        trine_topology_random(lc, 42);
        uint8_t  *endos = trine_learned_cascade_endos(lc);
        uint16_t *srcs  = trine_learned_cascade_srcs(lc);
        uint16_t *dsts  = trine_learned_cascade_dsts(lc);
        int ok = 1;
        for (uint32_t k = 0; k < 256; k++) {
            if (endos[k] >= 27 || srcs[k] >= 240 || dsts[k] >= 240)
                { ok = 0; break; }
        }
        check("topology_random_valid", ok);
        trine_learned_cascade_free(lc);
    }

    /* Layered topology */
    {
        trine_learned_cascade_t *lc = trine_learned_cascade_create(&cfg);
        trine_topology_layered(lc, 42);
        uint8_t  *endos = trine_learned_cascade_endos(lc);
        uint16_t *srcs  = trine_learned_cascade_srcs(lc);
        uint16_t *dsts  = trine_learned_cascade_dsts(lc);
        int ok = 1;
        for (uint32_t k = 0; k < 256; k++) {
            if (endos[k] >= 27 || srcs[k] >= 240 || dsts[k] >= 240)
                { ok = 0; break; }
        }
        check("topology_layered_valid", ok);
        trine_learned_cascade_free(lc);
    }

    /* Chain-local topology */
    {
        trine_learned_cascade_t *lc = trine_learned_cascade_create(&cfg);
        trine_topology_chain_local(lc, 42);
        uint8_t  *endos = trine_learned_cascade_endos(lc);
        uint16_t *srcs  = trine_learned_cascade_srcs(lc);
        uint16_t *dsts  = trine_learned_cascade_dsts(lc);
        int ok = 1;
        for (uint32_t k = 0; k < 256; k++) {
            if (endos[k] >= 27 || srcs[k] >= 240 || dsts[k] >= 240)
                { ok = 0; break; }
        }
        check("topology_chain_local_valid", ok);
        trine_learned_cascade_free(lc);
    }
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Cascade Tests ===\n");

    test_identity_cascade();
    test_determinism();
    test_depth_consistency();
    test_valid_trits_all_depths();
    test_nontrivial_mixing();
    test_reuse();
    test_topology_valid();

    printf("\nCascade: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
