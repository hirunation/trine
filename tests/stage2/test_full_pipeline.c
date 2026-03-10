/* =====================================================================
 * TRINE Stage-2 — Full Pipeline Tests (~15 tests)
 * =====================================================================
 *
 * End-to-end tests for the Stage-2 forward pass:
 *   1. Identity round-trip: encode → identity model → output == Stage-1
 *   2. Determinism across multiple calls
 *   3. Different texts → different embeddings
 *   4. Similar texts closer than dissimilar texts
 *   5. Depth=0 gives projection-only output
 *   6. Multi-depth extraction consistency
 *   7. trine_s2_compare() returns valid similarity in [0,1]
 *   8. Zero-float audit of inference path (structural)
 *   9. Model info introspection
 *  10. Null safety
 *  11. encode vs encode_from_trits equivalence
 *  12. Random model produces valid trits
 *  13. 1000-call determinism stress test
 *
 * ===================================================================== */

#include "trine_stage2.h"
#include "trine_encode.h"
#include "trine_stage1.h"
#include <stdio.h>
#include <string.h>
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
        printf("  FAIL  pipeline: %s\n", name);
    }
}

static int all_valid_trits(const uint8_t *v, int n)
{
    for (int i = 0; i < n; i++)
        if (v[i] > 2) return 0;
    return 1;
}

/* ── Test: Identity round-trip ──────────────────────────────────────── */
static void test_identity_roundtrip(void)
{
    trine_s2_model_t *m = trine_s2_create_identity();
    check("identity_create", m != NULL);

    const char *text = "hello world this is a test";
    size_t len = strlen(text);

    /* Stage-1 encode directly */
    uint8_t stage1[240];
    trine_encode_shingle(text, len, stage1);

    /* Stage-2 encode with identity model */
    uint8_t s2out[240];
    int rc = trine_s2_encode(m, text, len, 0, s2out);
    check("identity_encode_rc", rc == 0);
    check("identity_roundtrip", memcmp(stage1, s2out, 240) == 0);

    /* Also test with depth > 0 (identity cascade has 0 cells, so no change) */
    uint8_t s2out_d4[240];
    rc = trine_s2_encode(m, text, len, 4, s2out_d4);
    check("identity_roundtrip_depth4", memcmp(stage1, s2out_d4, 240) == 0);

    trine_s2_free(m);
}

/* ── Test: Determinism ──────────────────────────────────────────────── */
static void test_determinism(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);

    const char *text = "The quick brown fox jumps over the lazy dog";
    size_t len = strlen(text);

    uint8_t out1[240], out2[240];
    trine_s2_encode(m, text, len, 4, out1);
    trine_s2_encode(m, text, len, 4, out2);
    check("determinism", memcmp(out1, out2, 240) == 0);

    trine_s2_free(m);
}

/* ── Test: 1000-call determinism stress ─────────────────────────────── */
static void test_determinism_stress(void)
{
    trine_s2_model_t *m = trine_s2_create_random(256, 99);

    const char *text = "determinism stress test";
    size_t len = strlen(text);

    uint8_t ref[240], cur[240];
    trine_s2_encode(m, text, len, 4, ref);

    int ok = 1;
    for (int i = 0; i < 1000; i++) {
        trine_s2_encode(m, text, len, 4, cur);
        if (memcmp(ref, cur, 240) != 0) { ok = 0; break; }
    }
    check("determinism_1000_calls", ok);

    trine_s2_free(m);
}

/* ── Test: Different texts → different embeddings ───────────────────── */
static void test_different_texts(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);

    uint8_t out1[240], out2[240];
    trine_s2_encode(m, "alpha beta gamma", 16, 4, out1);
    trine_s2_encode(m, "x y z w", 7, 4, out2);
    check("different_texts", memcmp(out1, out2, 240) != 0);

    trine_s2_free(m);
}

/* ── Test: Similar texts closer than dissimilar texts ───────────────── */
static void test_similarity_ordering(void)
{
    /* Use identity model: preserves Stage-1 surface similarity ordering.
     * Random models do NOT preserve ordering (that's what training is for). */
    trine_s2_model_t *m = trine_s2_create_identity();

    const char *base  = "the quick brown fox jumps over the lazy dog";
    const char *near  = "the quick brown fox leaps over the lazy dog";
    const char *far   = "xylophone zebra quantum entanglement fusion";

    uint8_t e_base[240], e_near[240], e_far[240];
    trine_s2_encode(m, base, strlen(base), 0, e_base);
    trine_s2_encode(m, near, strlen(near), 0, e_near);
    trine_s2_encode(m, far,  strlen(far),  0, e_far);

    float sim_near = trine_s2_compare(e_base, e_near, NULL);
    float sim_far  = trine_s2_compare(e_base, e_far,  NULL);

    /* Identity model preserves Stage-1 ordering */
    check("similar_closer_than_dissimilar", sim_near > sim_far);

    trine_s2_free(m);
}

/* ── Test: Depth=0 gives projection-only output ─────────────────────── */
static void test_depth_zero(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);

    const char *text = "depth zero test";
    size_t len = strlen(text);

    uint8_t depth0[240];
    trine_s2_encode(m, text, len, 0, depth0);

    /* Should be valid trits */
    check("depth_zero_valid", all_valid_trits(depth0, 240));

    /* Depth 0 should differ from Stage-1 (random projection changes it) */
    uint8_t stage1[240];
    trine_encode_shingle(text, len, stage1);
    check("depth_zero_differs_from_stage1", memcmp(depth0, stage1, 240) != 0);

    trine_s2_free(m);
}

/* ── Test: Multi-depth extraction consistency ───────────────────────── */
static void test_multi_depth(void)
{
    trine_s2_model_t *m = trine_s2_create_random(256, 42);

    const char *text = "multi depth test";
    size_t len = strlen(text);

    /* Extract depths 0..3 */
    uint8_t depths[4 * 240];
    int rc = trine_s2_encode_depths(m, text, len, 4, depths, sizeof(depths));
    check("multi_depth_rc", rc == 0);

    /* Verify depth 0 matches single-call depth=0 */
    uint8_t single[240];
    trine_s2_encode(m, text, len, 0, single);
    check("multi_depth_d0_matches", memcmp(depths, single, 240) == 0);

    /* Verify all depths have valid trits */
    int ok = 1;
    for (int d = 0; d < 4; d++) {
        if (!all_valid_trits(depths + d * 240, 240)) { ok = 0; break; }
    }
    check("multi_depth_all_valid", ok);

    trine_s2_free(m);
}

/* ── Test: Compare returns valid similarity ─────────────────────────── */
static void test_compare_valid(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);

    uint8_t a[240], b[240];
    trine_s2_encode(m, "hello", 5, 4, a);
    trine_s2_encode(m, "world", 5, 4, b);

    float sim = trine_s2_compare(a, b, NULL);
    check("compare_in_range", sim >= 0.0f && sim <= 1.0f);

    /* Self-similarity should be 1.0 */
    float self = trine_s2_compare(a, a, NULL);
    check("compare_self_is_one", fabsf(self - 1.0f) < 1e-6f);

    trine_s2_free(m);
}

/* ── Test: Compare with lens ────────────────────────────────────────── */
static void test_compare_with_lens(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);

    uint8_t a[240], b[240];
    trine_s2_encode(m, "legal contract agreement", 24, 4, a);
    trine_s2_encode(m, "legal contract terms", 20, 4, b);

    trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;
    float sim = trine_s2_compare(a, b, &lens);
    check("compare_with_lens_valid", sim >= 0.0f && sim <= 1.0f);

    trine_s2_free(m);
}

/* ── Test: Model info introspection ─────────────────────────────────── */
static void test_info(void)
{
    trine_s2_model_t *m_id = trine_s2_create_identity();
    trine_s2_info_t info;

    int rc = trine_s2_info(m_id, &info);
    check("info_identity_rc", rc == 0);
    check("info_identity_k", info.projection_k == 3);
    check("info_identity_dims", info.projection_dims == 240);
    check("info_identity_cells", info.cascade_cells == 0);
    check("info_identity_flag", info.is_identity == 1);
    trine_s2_free(m_id);

    trine_s2_model_t *m_rnd = trine_s2_create_random(512, 42);
    rc = trine_s2_info(m_rnd, &info);
    check("info_random_rc", rc == 0);
    check("info_random_cells", info.cascade_cells == 512);
    check("info_random_flag", info.is_identity == 0);
    trine_s2_free(m_rnd);
}

/* ── Test: encode vs encode_from_trits equivalence ──────────────────── */
static void test_encode_vs_from_trits(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);

    const char *text = "encode equivalence test";
    size_t len = strlen(text);

    /* Full pipeline */
    uint8_t full[240];
    trine_s2_encode(m, text, len, 4, full);

    /* Two-step: Stage-1 then from_trits */
    uint8_t stage1[240], two_step[240];
    trine_encode_shingle(text, len, stage1);
    trine_s2_encode_from_trits(m, stage1, 4, two_step);

    check("encode_vs_from_trits", memcmp(full, two_step, 240) == 0);

    trine_s2_free(m);
}

/* ── Test: Random model valid trits ─────────────────────────────────── */
static void test_random_model_valid(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);

    uint8_t out[240];
    trine_s2_encode(m, "validity check", 14, 8, out);
    check("random_model_valid_trits", all_valid_trits(out, 240));

    trine_s2_free(m);
}

/* ── Test: Null safety ──────────────────────────────────────────────── */
static void test_null_safety(void)
{
    uint8_t dummy[240];
    memset(dummy, 0, sizeof(dummy));

    check("encode_null_model", trine_s2_encode(NULL, "x", 1, 0, dummy) == -1);
    check("info_null_model", trine_s2_info(NULL, NULL) == -1);
    check("compare_null_a", trine_s2_compare(NULL, dummy, NULL) < 0.0f);
    check("compare_null_b", trine_s2_compare(dummy, NULL, NULL) < 0.0f);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Full Pipeline Tests ===\n");

    test_identity_roundtrip();
    test_determinism();
    test_determinism_stress();
    test_different_texts();
    test_similarity_ordering();
    test_depth_zero();
    test_multi_depth();
    test_compare_valid();
    test_compare_with_lens();
    test_info();
    test_encode_vs_from_trits();
    test_random_model_valid();
    test_null_safety();

    printf("\nPipeline: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
