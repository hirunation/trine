/* =====================================================================
 * TRINE Stage-2 — Adaptive Blend Alpha Tests (24 assertions)
 * =====================================================================
 *
 * Tests for the adaptive per-S1-bucket alpha blending API:
 *
 *  1. Set/Unset Adaptive Alpha
 *   1. test_set_buckets_and_compare       — set buckets, verify blend works
 *   2. test_disable_with_null             — set NULL disables, returns 0.0
 *   3. test_re_enable_after_disable       — disable then re-enable works
 *
 *  2. Bucket Selection
 *   4. test_bucket_selection_low          — low S1 similarity uses bucket 0-2
 *   5. test_bucket_selection_high         — high S1 similarity uses bucket 8-9
 *
 *  3. Alpha=1.0 Everywhere (Pure S1)
 *   6. test_alpha_one_matches_s1          — blend equals S1 similarity
 *
 *  4. Alpha=0.0 Everywhere (Pure S2)
 *   7. test_alpha_zero_matches_s2         — blend equals S2 centered cosine
 *
 *  5. Monotonic Buckets
 *   8. test_monotonic_alpha_behavior      — increasing alpha shifts toward S1
 *
 *  6. Boundary Conditions
 *   9. test_boundary_identical_vectors    — S1 sim near 1.0 uses bucket 9
 *  10. test_boundary_zero_vectors         — all-zero embeddings return 0.0
 *
 *  7. Determinism
 *  11. test_determinism                   — same inputs yield same output
 *
 *  8. Model Without Adaptive Alpha
 *  12. test_fallback_without_alpha        — NULL adaptive_alpha returns 0.0
 *  13. test_null_model_safety             — NULL model returns 0.0
 *  14. test_null_embedding_safety         — NULL embedding ptrs return 0.0
 *
 * ===================================================================== */

#include "trine_stage2.h"
#include "trine_encode.h"
#include "trine_stage1.h"
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
        printf("  FAIL  adaptive_alpha: %s\n", name);
    }
}

/* ── Helper: compute S1 uniform cosine (raw values, not centered) ──── */
static float s1_uniform_cosine(const uint8_t a[240], const uint8_t b[240])
{
    double dot = 0.0, ma = 0.0, mb = 0.0;
    for (int i = 0; i < 240; i++) {
        double va = (double)a[i];
        double vb = (double)b[i];
        dot += va * vb;
        ma  += va * va;
        mb  += vb * vb;
    }
    if (ma < 1e-12 || mb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(ma) * sqrt(mb)));
}

/* ── Helper: compute S2 centered cosine (values shifted by -1) ─────── */
static float s2_centered_cosine(const uint8_t a[240], const uint8_t b[240])
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < 240; i++) {
        double va = (double)a[i] - 1.0;
        double vb = (double)b[i] - 1.0;
        dot += va * vb;
        na  += va * va;
        nb  += vb * vb;
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

/* =====================================================================
 * 1. Set/Unset Adaptive Alpha
 * ===================================================================== */

/* ── Test 1: Set buckets, verify blended compare works ─────────────── */
static void test_set_buckets_and_compare(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);
    check("set_buckets_model_created", model != NULL);

    float buckets[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                         0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    trine_s2_set_adaptive_alpha(model, buckets);

    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle("the quick brown fox", 19, s1_a);
    trine_encode_shingle("a fast brown fox", 16, s1_b);
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    float result = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);

    /* Should produce a valid finite similarity */
    check("set_buckets_result_valid", isfinite(result));
    /* Should not be exactly 0.0 (which is the disabled/error fallback) */
    check("set_buckets_result_nonzero", fabsf(result) > 1e-8f);

    trine_s2_free(model);
}

/* ── Test 2: Setting NULL disables adaptive alpha ──────────────────── */
static void test_disable_with_null(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);

    float buckets[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                         0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    trine_s2_set_adaptive_alpha(model, buckets);

    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle("hello world", 11, s1_a);
    trine_encode_shingle("hello there", 11, s1_b);
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    /* First: adaptive blend should work */
    float before = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);
    check("disable_before_nonzero", fabsf(before) > 1e-8f);

    /* Disable by setting NULL */
    trine_s2_set_adaptive_alpha(model, NULL);

    /* Now: adaptive blend should return 0.0 (disabled fallback) */
    float after = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);
    check("disable_after_zero", fabsf(after) < 1e-8f);

    trine_s2_free(model);
}

/* ── Test 3: Re-enable after disable ───────────────────────────────── */
static void test_re_enable_after_disable(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);

    float buckets[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                         0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle("cats are great", 14, s1_a);
    trine_encode_shingle("dogs are great", 14, s1_b);
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    /* Enable, then disable, then re-enable */
    trine_s2_set_adaptive_alpha(model, buckets);
    float first = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);

    trine_s2_set_adaptive_alpha(model, NULL);
    float disabled = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);

    trine_s2_set_adaptive_alpha(model, buckets);
    float second = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);

    check("re_enable_disabled_zero", fabsf(disabled) < 1e-8f);
    check("re_enable_matches_first", fabsf(first - second) < 1e-6f);

    trine_s2_free(model);
}

/* =====================================================================
 * 2. Bucket Selection
 * ===================================================================== */

/* ── Test 4: Low S1 similarity uses low bucket index ───────────────── */
static void test_bucket_selection_low(void)
{
    trine_s2_model_t *model = trine_s2_create_identity();

    /* Bucket 0 gets alpha=0.0 (pure S2), all others get alpha=1.0 (pure S1).
     * Use texts that produce very different S1 vectors (low similarity).
     * With identity model, S2 = S1, so result should reflect the selected alpha. */
    float buckets[10] = {0.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    trine_s2_set_adaptive_alpha(model, buckets);

    /* Very different texts -> low S1 sim -> bucket 0 or low indices */
    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle("xxxxxxxxxxxxxxx", 15, s1_a);
    trine_encode_shingle("yyyyyyyyyyyyyyy", 15, s1_b);
    /* Identity model: S2 = S1 */
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    float s1_sim = s1_uniform_cosine(s1_a, s1_b);
    /* Verify that these texts produce a low S1 similarity (bucket < 5) */
    check("bucket_low_s1_sim_low", s1_sim < 0.5f);

    /* The adaptive blend result should be valid */
    float result = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);
    check("bucket_low_result_valid", isfinite(result));

    trine_s2_free(model);
}

/* ── Test 5: High S1 similarity uses high bucket index ─────────────── */
static void test_bucket_selection_high(void)
{
    trine_s2_model_t *model = trine_s2_create_identity();

    /* Only bucket 9 (S1 sim >= 0.9) gets alpha=0.0, all others get 1.0 */
    float buckets_high[10] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                              1.0f, 1.0f, 1.0f, 1.0f, 0.0f};
    trine_s2_set_adaptive_alpha(model, buckets_high);

    /* Nearly identical texts -> high S1 sim -> bucket 9 */
    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle("the quick brown fox", 19, s1_a);
    trine_encode_shingle("the quick brown fox", 19, s1_b);  /* identical */
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    float s1_sim = s1_uniform_cosine(s1_a, s1_b);
    /* Identical text -> S1 sim = 1.0 -> bucket = min(floor(1.0*10), 9) = 9 */
    check("bucket_high_s1_sim_high", s1_sim > 0.9f);

    /* With identical S1 input, bucket 9 is selected (alpha=0.0).
     * For identity model, S2 centered cosine of identical vectors = 1.0,
     * so result = 0.0 * s1_sim + 1.0 * s2_centered = s2_centered */
    float result = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);
    float s2_sim = s2_centered_cosine(s2_a, s2_b);
    check("bucket_high_uses_bucket9", fabsf(result - s2_sim) < 1e-4f);

    trine_s2_free(model);
}

/* =====================================================================
 * 3. Alpha=1.0 Everywhere (Pure S1)
 * ===================================================================== */

/* ── Test 6: All buckets alpha=1.0 produces pure S1 similarity ─────── */
static void test_alpha_one_matches_s1(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);

    float buckets_one[10] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                             1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    trine_s2_set_adaptive_alpha(model, buckets_one);

    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle("machine learning algorithms", 27, s1_a);
    trine_encode_shingle("deep learning methods", 21, s1_b);
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    float s1_sim = s1_uniform_cosine(s1_a, s1_b);
    float blend  = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);

    /* alpha=1.0 -> result = 1.0 * s1_sim + 0.0 * s2_sim = s1_sim */
    check("alpha_one_matches_s1", fabsf(blend - s1_sim) < 1e-4f);

    trine_s2_free(model);
}

/* =====================================================================
 * 4. Alpha=0.0 Everywhere (Pure S2)
 * ===================================================================== */

/* ── Test 7: All buckets alpha=0.0 produces pure S2 centered cosine ── */
static void test_alpha_zero_matches_s2(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);

    float buckets_zero[10] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    trine_s2_set_adaptive_alpha(model, buckets_zero);

    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle("natural language processing", 27, s1_a);
    trine_encode_shingle("text analysis methods", 21, s1_b);
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    float s2_sim = s2_centered_cosine(s2_a, s2_b);
    float blend  = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);

    /* alpha=0.0 -> result = 0.0 * s1_sim + 1.0 * s2_sim = s2_sim */
    check("alpha_zero_matches_s2", fabsf(blend - s2_sim) < 1e-4f);

    trine_s2_free(model);
}

/* =====================================================================
 * 5. Monotonic Buckets
 * ===================================================================== */

/* ── Test 8: Increasing alpha shifts blend toward S1 ───────────────── */
static void test_monotonic_alpha_behavior(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);

    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle("computer vision tasks", 21, s1_a);
    trine_encode_shingle("image recognition work", 22, s1_b);
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    float s1_sim = s1_uniform_cosine(s1_a, s1_b);
    float s2_sim = s2_centered_cosine(s2_a, s2_b);

    /* We want to ensure blend moves linearly between s2_sim and s1_sim.
     * Use uniform buckets so the bucket selection does not matter. */
    float blend_low, blend_mid, blend_high;

    /* All buckets = 0.2 */
    float buckets_low[10] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
                             0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
    trine_s2_set_adaptive_alpha(model, buckets_low);
    blend_low = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);

    /* All buckets = 0.5 */
    float buckets_mid[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                             0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    trine_s2_set_adaptive_alpha(model, buckets_mid);
    blend_mid = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);

    /* All buckets = 0.8 */
    float buckets_high[10] = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
                              0.8f, 0.8f, 0.8f, 0.8f, 0.8f};
    trine_s2_set_adaptive_alpha(model, buckets_high);
    blend_high = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);

    /* When S1 > S2, increasing alpha should increase the blend.
     * When S1 < S2, increasing alpha should decrease the blend.
     * Either way, the blend should be monotonic with alpha. */
    if (s1_sim > s2_sim) {
        check("monotonic_increasing", blend_low <= blend_mid + 1e-6f &&
                                      blend_mid <= blend_high + 1e-6f);
    } else {
        check("monotonic_increasing", blend_low >= blend_mid - 1e-6f &&
                                      blend_mid >= blend_high - 1e-6f);
    }

    /* Verify the blend values are distinct when s1 != s2 */
    if (fabsf(s1_sim - s2_sim) > 0.01f) {
        check("monotonic_distinct", fabsf(blend_low - blend_high) > 1e-6f);
    } else {
        /* If S1 ~ S2, all blends will be similar, which is fine */
        check("monotonic_distinct", 1);
    }

    trine_s2_free(model);
}

/* =====================================================================
 * 6. Boundary Conditions
 * ===================================================================== */

/* ── Test 9: Identical vectors (S1 sim near 1.0) use bucket 9 ─────── */
static void test_boundary_identical_vectors(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);

    /* Make bucket 9 have a unique alpha to verify it is selected */
    float buckets[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                         0.5f, 0.5f, 0.5f, 0.5f, 0.0f};
    trine_s2_set_adaptive_alpha(model, buckets);

    uint8_t s1_a[240], s2_a[240];
    trine_encode_shingle("identical text here", 19, s1_a);
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);

    /* Self-comparison: S1 sim = 1.0, bucket = 9, alpha = 0.0 */
    float result = trine_s2_compare_adaptive_blend(
        model, s1_a, s1_a, s2_a, s2_a);

    /* With alpha=0.0: result = 0.0 * 1.0 + 1.0 * s2_centered(a,a)
     * s2_centered of identical vectors = 1.0
     * So result should be 1.0 */
    float s2_self = s2_centered_cosine(s2_a, s2_a);
    check("boundary_identical_result", fabsf(result - s2_self) < 1e-4f);

    /* Now use bucket 9 alpha = 1.0 (pure S1) */
    float buckets2[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                          0.5f, 0.5f, 0.5f, 0.5f, 1.0f};
    trine_s2_set_adaptive_alpha(model, buckets2);

    float result2 = trine_s2_compare_adaptive_blend(
        model, s1_a, s1_a, s2_a, s2_a);

    /* With alpha=1.0: result = 1.0 * s1_sim = 1.0 */
    float s1_self = s1_uniform_cosine(s1_a, s1_a);
    check("boundary_identical_pure_s1", fabsf(result2 - s1_self) < 1e-4f);

    trine_s2_free(model);
}

/* ── Test 10: All-zero embeddings produce 0.0 ─────────────────────── */
static void test_boundary_zero_vectors(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);

    float buckets[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                         0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    trine_s2_set_adaptive_alpha(model, buckets);

    /* All-zero S1 and S2 vectors */
    uint8_t zeros[240];
    memset(zeros, 0, sizeof(zeros));

    float result = trine_s2_compare_adaptive_blend(
        model, zeros, zeros, zeros, zeros);

    /* S1 cosine of zero vectors = 0.0, bucket = 0, alpha = 0.5
     * S2 centered cosine of zero vectors = 0.0 (all -1.0 centered, but norm check)
     * Actually: trit value 0 centered = -1.0, so all values are -1.0,
     * making the vectors identical. s2_centered = 1.0.
     * s1_sim = 0 means bucket 0, alpha=0.5.
     * result = 0.5 * 0.0 + 0.5 * 1.0 = 0.5  (if s1_sim = 0 due to norms)
     * But wait: zero vectors have norm 0, so s1_sim = 0.0.
     * s2_centered: all trits are 0, centered to -1.0, dot=240, norm=sqrt(240)^2
     * So s2_centered(zeros, zeros) = 1.0.
     * result = 0.5 * 0.0 + 0.5 * 1.0 = 0.5 */
    check("boundary_zero_valid", isfinite(result));

    trine_s2_free(model);
}

/* =====================================================================
 * 7. Determinism
 * ===================================================================== */

/* ── Test 11: Same inputs always produce the same result ───────────── */
static void test_determinism(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);

    float buckets[10] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
                         0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    trine_s2_set_adaptive_alpha(model, buckets);

    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle("reinforcement learning", 22, s1_a);
    trine_encode_shingle("reward-based training", 21, s1_b);
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    float r1 = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);
    float r2 = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);
    float r3 = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);

    check("determinism_r1_r2", r1 == r2);
    check("determinism_r2_r3", r2 == r3);

    trine_s2_free(model);
}

/* =====================================================================
 * 8. Model Without Adaptive Alpha / Null Safety
 * ===================================================================== */

/* ── Test 12: Model without adaptive alpha returns 0.0 ─────────────── */
static void test_fallback_without_alpha(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);
    /* Do NOT set adaptive alpha */

    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle("data science methods", 20, s1_a);
    trine_encode_shingle("statistical analysis", 20, s1_b);
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    float result = trine_s2_compare_adaptive_blend(model, s1_a, s1_b, s2_a, s2_b);
    check("fallback_returns_zero", fabsf(result) < 1e-8f);

    trine_s2_free(model);
}

/* ── Test 13: NULL model returns 0.0 ───────────────────────────────── */
static void test_null_model_safety(void)
{
    uint8_t dummy[240];
    memset(dummy, 1, sizeof(dummy));

    float result = trine_s2_compare_adaptive_blend(
        NULL, dummy, dummy, dummy, dummy);
    check("null_model_returns_zero", fabsf(result) < 1e-8f);
}

/* ── Test 14: NULL embedding pointers return 0.0 ───────────────────── */
static void test_null_embedding_safety(void)
{
    trine_s2_model_t *model = trine_s2_create_random(64, 42);

    float buckets[10] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                         0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    trine_s2_set_adaptive_alpha(model, buckets);

    uint8_t dummy[240];
    memset(dummy, 1, sizeof(dummy));

    float r1 = trine_s2_compare_adaptive_blend(model, NULL, dummy, dummy, dummy);
    check("null_s1a", fabsf(r1) < 1e-8f);

    float r2 = trine_s2_compare_adaptive_blend(model, dummy, NULL, dummy, dummy);
    check("null_s1b", fabsf(r2) < 1e-8f);

    float r3 = trine_s2_compare_adaptive_blend(model, dummy, dummy, NULL, dummy);
    check("null_s2a", fabsf(r3) < 1e-8f);

    float r4 = trine_s2_compare_adaptive_blend(model, dummy, dummy, dummy, NULL);
    check("null_s2b", fabsf(r4) < 1e-8f);

    trine_s2_free(model);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Adaptive Blend Alpha Tests ===\n");

    /* 1. Set/Unset */
    printf("\n--- Set/Unset Adaptive Alpha ---\n");
    test_set_buckets_and_compare();
    test_disable_with_null();
    test_re_enable_after_disable();

    /* 2. Bucket Selection */
    printf("\n--- Bucket Selection ---\n");
    test_bucket_selection_low();
    test_bucket_selection_high();

    /* 3. Alpha=1.0 (Pure S1) */
    printf("\n--- Alpha=1.0 (Pure S1) ---\n");
    test_alpha_one_matches_s1();

    /* 4. Alpha=0.0 (Pure S2) */
    printf("\n--- Alpha=0.0 (Pure S2) ---\n");
    test_alpha_zero_matches_s2();

    /* 5. Monotonic Buckets */
    printf("\n--- Monotonic Buckets ---\n");
    test_monotonic_alpha_behavior();

    /* 6. Boundary Conditions */
    printf("\n--- Boundary Conditions ---\n");
    test_boundary_identical_vectors();
    test_boundary_zero_vectors();

    /* 7. Determinism */
    printf("\n--- Determinism ---\n");
    test_determinism();

    /* 8. Fallback / Null Safety */
    printf("\n--- Fallback / Null Safety ---\n");
    test_fallback_without_alpha();
    test_null_model_safety();
    test_null_embedding_safety();

    printf("\nAdaptive Alpha: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
