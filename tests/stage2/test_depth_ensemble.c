/* =====================================================================
 * TRINE Stage-2 — Multi-Depth Encoding & Ensemble Tests (~25 tests)
 * =====================================================================
 *
 * Tests for multi-depth S2 encoding and ensemble behavior:
 *   1.  Depth monotonicity: depth 0 via encode matches encode_depths[0]
 *   2.  Increasing depths diverge for non-identity models
 *   3.  Depth determinism: identical output across repeated calls
 *   4.  Multi-depth array correctness (encode_depths)
 *   5.  Depth bounds: out-of-range returns error, in-range succeeds
 *   6.  Buffer overflow protection via out_size
 *   7.  Ensemble comparison: average similarity across depths
 *   8.  Depth with block-diagonal model
 *
 * ===================================================================== */

#include "trine_stage2.h"
#include "trine_encode.h"
#include "trine_stage1.h"
#include "trine_learned_cascade.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
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
        printf("  FAIL  depth_ensemble: %s\n", name);
    }
}

static int all_valid_trits(const uint8_t *v, int n)
{
    for (int i = 0; i < n; i++)
        if (v[i] > 2) return 0;
    return 1;
}

/* ── 1. Depth monotonicity: depth 0 single-call == encode_depths[0] ── */
static void test_depth_monotonicity(void)
{
    printf("  [depth monotonicity]\n");

    trine_s2_model_t *m = trine_s2_create_random(512, 42);

    const char *text = "depth monotonicity test sentence";
    size_t len = strlen(text);

    /* Single-call at depth 0 */
    uint8_t single_d0[240];
    int rc = trine_s2_encode(m, text, len, 0, single_d0);
    check("mono_encode_d0_rc", rc == 0);

    /* Multi-depth: extract depths 0..7 */
    uint8_t depths[8 * 240];
    rc = trine_s2_encode_depths(m, text, len, 8, depths, sizeof(depths));
    check("mono_depths_rc", rc == 0);

    /* depths[0] must exactly equal single-call depth=0 */
    check("mono_d0_matches_single", memcmp(single_d0, depths, 240) == 0);

    /* Also verify depths[d] matches trine_s2_encode at each depth */
    int all_match = 1;
    for (uint32_t d = 0; d < 8; d++) {
        uint8_t ref[240];
        trine_s2_encode(m, text, len, d, ref);
        if (memcmp(ref, depths + d * 240, 240) != 0) {
            all_match = 0;
            break;
        }
    }
    check("mono_all_depths_match_single", all_match);

    trine_s2_free(m);
}

/* ── 2. Increasing depths diverge ──────────────────────────────────── */
static void test_depths_diverge(void)
{
    printf("  [increasing depths diverge]\n");

    trine_s2_model_t *m = trine_s2_create_random(512, 77);

    const char *text = "divergence test for cascade depths";
    size_t len = strlen(text);

    uint8_t d0[240], d4[240], d8[240];
    trine_s2_encode(m, text, len, 0, d0);
    trine_s2_encode(m, text, len, 4, d4);
    trine_s2_encode(m, text, len, 8, d8);

    /* Non-identity random model: cascade should alter the vector */
    check("diverge_d0_vs_d4", memcmp(d0, d4, 240) != 0);
    check("diverge_d0_vs_d8", memcmp(d0, d8, 240) != 0);
    check("diverge_d4_vs_d8", memcmp(d4, d8, 240) != 0);

    trine_s2_free(m);
}

/* ── 3. Depth determinism ──────────────────────────────────────────── */
static void test_depth_determinism(void)
{
    printf("  [depth determinism]\n");

    trine_s2_model_t *m = trine_s2_create_random(256, 123);

    const char *text = "determinism under repeated encoding";
    size_t len = strlen(text);

    /* Run 100 times at depth 6 — must always match */
    uint8_t ref[240], cur[240];
    trine_s2_encode(m, text, len, 6, ref);

    int ok = 1;
    for (int i = 0; i < 100; i++) {
        trine_s2_encode(m, text, len, 6, cur);
        if (memcmp(ref, cur, 240) != 0) { ok = 0; break; }
    }
    check("determinism_depth6_100x", ok);

    /* Multi-depth determinism */
    uint8_t depths1[4 * 240], depths2[4 * 240];
    trine_s2_encode_depths(m, text, len, 4, depths1, sizeof(depths1));
    trine_s2_encode_depths(m, text, len, 4, depths2, sizeof(depths2));
    check("determinism_multi_depth", memcmp(depths1, depths2, sizeof(depths1)) == 0);

    trine_s2_free(m);
}

/* ── 4. Multi-depth array correctness ──────────────────────────────── */
static void test_multi_depth_array(void)
{
    printf("  [multi-depth array]\n");

    trine_s2_model_t *m = trine_s2_create_random(512, 55);

    const char *text = "multi depth array correctness";
    size_t len = strlen(text);

    /* Extract 16 depths */
    uint8_t depths[16 * 240];
    int rc = trine_s2_encode_depths(m, text, len, 16, depths, sizeof(depths));
    check("array_rc", rc == 0);

    /* Every depth slice must contain valid trits */
    int all_ok = 1;
    for (int d = 0; d < 16; d++) {
        if (!all_valid_trits(depths + d * 240, 240)) { all_ok = 0; break; }
    }
    check("array_all_valid_trits", all_ok);

    /* Adjacent depths should differ (non-identity cascade) */
    int any_differ = 0;
    for (int d = 0; d < 15; d++) {
        if (memcmp(depths + d * 240, depths + (d + 1) * 240, 240) != 0) {
            any_differ = 1;
            break;
        }
    }
    check("array_adjacent_differ", any_differ);

    trine_s2_free(m);
}

/* ── 5. Depth bounds ───────────────────────────────────────────────── */
static void test_depth_bounds(void)
{
    printf("  [depth bounds]\n");

    trine_s2_model_t *m = trine_s2_create_random(256, 99);
    uint8_t out[240];

    /* depth 0 always succeeds */
    check("bounds_depth0", trine_s2_encode(m, "test", 4, 0, out) == 0);

    /* depth 64 = TRINE_CASCADE_MAX_DEPTH is allowed */
    check("bounds_depth64", trine_s2_encode(m, "test", 4, 64, out) == 0);

    /* depth 65 > MAX_DEPTH returns error */
    check("bounds_depth65_err", trine_s2_encode(m, "test", 4, 65, out) == -1);

    /* Very large depth returns error */
    check("bounds_huge_depth", trine_s2_encode(m, "test", 4, 1000, out) == -1);

    /* encode_depths: max_depth=0 returns error */
    uint8_t buf[240];
    check("bounds_depths_zero", trine_s2_encode_depths(m, "t", 1, 0, buf, sizeof(buf)) == -1);

    /* encode_depths: max_depth=65 returns error */
    uint8_t big[65 * 240];
    check("bounds_depths_65", trine_s2_encode_depths(m, "t", 1, 65, big, sizeof(big)) == -1);

    trine_s2_free(m);
}

/* ── 6. Buffer overflow protection ─────────────────────────────────── */
static void test_buffer_overflow(void)
{
    printf("  [buffer overflow protection]\n");

    trine_s2_model_t *m = trine_s2_create_random(256, 42);

    /* Request 4 depths but provide buffer for only 3 */
    uint8_t small_buf[3 * 240];
    int rc = trine_s2_encode_depths(m, "overflow test", 13, 4, small_buf, sizeof(small_buf));
    check("overflow_too_small", rc == -1);

    /* Request 4 depths with exact-size buffer: should succeed */
    uint8_t exact_buf[4 * 240];
    rc = trine_s2_encode_depths(m, "overflow test", 13, 4, exact_buf, sizeof(exact_buf));
    check("overflow_exact_size", rc == 0);

    /* Oversized buffer is fine too */
    uint8_t big_buf[8 * 240];
    rc = trine_s2_encode_depths(m, "overflow test", 13, 4, big_buf, sizeof(big_buf));
    check("overflow_oversized", rc == 0);

    /* Verify first 4*240 bytes match between exact and oversized */
    check("overflow_content_match", memcmp(exact_buf, big_buf, 4 * 240) == 0);

    trine_s2_free(m);
}

/* ── 7. Ensemble comparison ────────────────────────────────────────── */
static void test_ensemble_comparison(void)
{
    printf("  [ensemble comparison]\n");

    trine_s2_model_t *m = trine_s2_create_random(512, 42);

    const char *text_a = "the quick brown fox jumps over the lazy dog";
    const char *text_b = "the quick brown fox leaps over the lazy dog";

    /* Encode at multiple depths for both texts */
    uint32_t n_depths = 8;
    uint8_t depths_a[8 * 240], depths_b[8 * 240];
    trine_s2_encode_depths(m, text_a, strlen(text_a), n_depths, depths_a, sizeof(depths_a));
    trine_s2_encode_depths(m, text_b, strlen(text_b), n_depths, depths_b, sizeof(depths_b));

    /* Average similarity across all depths (ensemble) */
    float sum = 0.0f;
    int valid_count = 0;
    for (uint32_t d = 0; d < n_depths; d++) {
        float sim = trine_s2_compare(depths_a + d * 240, depths_b + d * 240, NULL);
        if (sim >= 0.0f) {
            sum += sim;
            valid_count++;
        }
    }
    float ensemble_avg = (valid_count > 0) ? sum / (float)valid_count : 0.0f;

    /* Ensemble average should be a valid similarity value */
    check("ensemble_avg_valid", ensemble_avg >= 0.0f && ensemble_avg <= 1.0f);
    check("ensemble_all_depths_valid", valid_count == (int)n_depths);

    /* Single-depth similarity at depth 0 */
    float sim_d0 = trine_s2_compare(depths_a, depths_b, NULL);
    check("ensemble_d0_valid", sim_d0 >= 0.0f && sim_d0 <= 1.0f);

    /* Dissimilar text should have lower ensemble similarity */
    const char *text_c = "xylophone zebra quantum entanglement fusion reactor";
    uint8_t depths_c[8 * 240];
    trine_s2_encode_depths(m, text_c, strlen(text_c), n_depths, depths_c, sizeof(depths_c));

    float sum_far = 0.0f;
    for (uint32_t d = 0; d < n_depths; d++) {
        sum_far += trine_s2_compare(depths_a + d * 240, depths_c + d * 240, NULL);
    }
    float ensemble_far = sum_far / (float)n_depths;
    check("ensemble_near_gt_far", ensemble_avg > ensemble_far);

    trine_s2_free(m);
}

/* ── 8. Depth with block-diagonal model ────────────────────────────── */
static void test_block_diagonal_depth(void)
{
    printf("  [block-diagonal depth]\n");

    /* Create block-diagonal weights: K=3, 4 chains, 60x60 each */
    int K = 3;
    size_t bw_size = (size_t)K * 4 * 60 * 60;
    uint8_t *bw = calloc(bw_size, 1);
    if (!bw) {
        check("blockdiag_alloc", 0);
        return;
    }

    /* Fill with a simple pattern: cycling 0/1/2 */
    for (size_t i = 0; i < bw_size; i++) {
        bw[i] = (uint8_t)(i % 3);
    }

    trine_s2_model_t *m = trine_s2_create_block_diagonal(bw, K, 256, 88);
    check("blockdiag_create", m != NULL);

    if (m) {
        const char *text = "block diagonal depth test";
        size_t len = strlen(text);

        /* Encode at depth 0 and depth 4 */
        uint8_t d0[240], d4[240];
        int rc0 = trine_s2_encode(m, text, len, 0, d0);
        int rc4 = trine_s2_encode(m, text, len, 4, d4);

        check("blockdiag_d0_rc", rc0 == 0);
        check("blockdiag_d4_rc", rc4 == 0);
        check("blockdiag_d0_valid", all_valid_trits(d0, 240));
        check("blockdiag_d4_valid", all_valid_trits(d4, 240));

        /* Block-diagonal projection should modify the vector */
        uint8_t stage1[240];
        trine_encode_shingle(text, len, stage1);
        check("blockdiag_d0_differs_s1", memcmp(d0, stage1, 240) != 0);

        trine_s2_free(m);
    }

    free(bw);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Depth Ensemble Tests ===\n");

    test_depth_monotonicity();
    test_depths_diverge();
    test_depth_determinism();
    test_multi_depth_array();
    test_depth_bounds();
    test_buffer_overflow();
    test_ensemble_comparison();
    test_block_diagonal_depth();

    printf("\nDepth ensemble: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
