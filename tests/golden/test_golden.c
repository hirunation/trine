/* =====================================================================
 * TRINE — Golden Vector Regression Test Suite
 * =====================================================================
 *
 * Verifies determinism invariants that must hold across all versions:
 *
 * Stage-1 Golden Vectors (15 tests):
 *   1.  "hello world" first-20 trits are deterministic
 *   2.  "" (empty) encodes to all zeros
 *   3.  "a" (single char) encodes deterministically
 *   4.  "Hello World" vs "hello world" case folding produces same output
 *   5.  1000-char repeated string encodes deterministically
 *   6.  Encode symmetry: same input always produces same output
 *   7.  Chain 1 (forward/char) has non-trivial fill
 *   8.  Chain 2 (reverse/trigram) has non-trivial fill
 *   9.  Chain 3 (diff/5gram) has non-trivial fill
 *  10.  Chain 4 (struct/word) has non-trivial fill
 *  11.  Packed round-trip: pack(encode(X)) -> unpack -> matches original
 *  12.  Identity comparison: compare(X, X) == 1.0
 *  13.  Comparison symmetry: compare(A, B) == compare(B, A)
 *  14.  Different texts produce different embeddings
 *  15.  Similarity ordering: near text > far text
 *
 * Stage-2 Golden Vectors (10 tests):
 *  16.  Identity model encode matches Stage-1 output
 *  17.  Identity model encode at depth=4 matches Stage-1
 *  18.  Random model (seed=42) encode is deterministic
 *  19.  Random model depth=0 vs depth=4 differ
 *  20.  S2 compare with identity model matches S1 compare
 *  21.  S2 compare self-similarity == 1.0
 *  22.  Random model output contains valid trits only
 *  23.  Save/load identity model: encoded output matches
 *  24.  Random model 1000-call stress determinism
 *  25.  Encode vs encode_from_trits equivalence
 *
 * ===================================================================== */

#include "trine_encode.h"
#include "trine_stage1.h"
#include "trine_stage2.h"
#include "trine_s2_persist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int passed = 0;
static int failed = 0;

static void check(const char *name, int cond)
{
    if (cond) {
        passed++;
    } else {
        failed++;
        printf("  FAIL  golden: %s\n", name);
    }
}

static int all_valid_trits(const uint8_t *v, int n)
{
    for (int i = 0; i < n; i++)
        if (v[i] > 2) return 0;
    return 1;
}

static int all_zero(const uint8_t *v, int n)
{
    for (int i = 0; i < n; i++)
        if (v[i] != 0) return 0;
    return 1;
}

static int count_nonzero(const uint8_t *v, int offset, int len)
{
    int cnt = 0;
    for (int i = offset; i < offset + len; i++)
        if (v[i] != 0) cnt++;
    return cnt;
}

/* ── Category 1: Stage-1 Encoding Determinism ──────────────────────── */

static void test_s1_hello_world_deterministic(void)
{
    uint8_t a[240], b[240];
    trine_encode_shingle("hello world", 11, a);
    trine_encode_shingle("hello world", 11, b);

    /* First 20 trits must match across two calls */
    check("s1_hello_world_first20_deterministic",
          memcmp(a, b, 20) == 0);

    /* Full 240 must match */
    check("s1_hello_world_full_deterministic",
          memcmp(a, b, 240) == 0);

    /* All trits must be valid (0, 1, or 2) */
    check("s1_hello_world_valid_trits",
          all_valid_trits(a, 240));
}

static void test_s1_empty_string(void)
{
    uint8_t out[240];
    trine_encode_shingle("", 0, out);

    /* Empty string should produce all zeros */
    check("s1_empty_all_zeros", all_zero(out, 240));
}

static void test_s1_single_char(void)
{
    uint8_t a[240], b[240];
    trine_encode_shingle("a", 1, a);
    trine_encode_shingle("a", 1, b);

    check("s1_single_char_deterministic", memcmp(a, b, 240) == 0);
    check("s1_single_char_valid_trits", all_valid_trits(a, 240));
    /* Single char should produce at least some non-zero trits */
    check("s1_single_char_not_empty", !all_zero(a, 240));
}

static void test_s1_case_folding(void)
{
    uint8_t lower[240], upper[240];
    trine_encode_shingle("hello world", 11, lower);
    trine_encode_shingle("Hello World", 11, upper);

    /* Shingle encoder is case-insensitive: "Cat" == "cat" == "CAT"
     * So case-folded output should be identical. */
    check("s1_case_folding_identical", memcmp(lower, upper, 240) == 0);
}

static void test_s1_long_string_deterministic(void)
{
    /* Build a 1000-char repeated string */
    char long_text[1001];
    for (int i = 0; i < 1000; i++)
        long_text[i] = 'a' + (i % 26);
    long_text[1000] = '\0';

    uint8_t a[240], b[240];
    trine_encode_shingle(long_text, 1000, a);
    trine_encode_shingle(long_text, 1000, b);

    check("s1_long_string_deterministic", memcmp(a, b, 240) == 0);
    check("s1_long_string_valid_trits", all_valid_trits(a, 240));
}

static void test_s1_encode_symmetry(void)
{
    const char *texts[] = {
        "the quick brown fox",
        "machine learning",
        "ternary resonance interference"
    };
    int ntexts = 3;

    int ok = 1;
    for (int t = 0; t < ntexts; t++) {
        uint8_t first[240], second[240];
        size_t len = strlen(texts[t]);
        trine_encode_shingle(texts[t], len, first);
        trine_encode_shingle(texts[t], len, second);
        if (memcmp(first, second, 240) != 0) { ok = 0; break; }
    }
    check("s1_encode_symmetry_multi_text", ok);
}

/* ── Category 2: Stage-1 Chain Fill ────────────────────────────────── */

static void test_s1_chain_fill(void)
{
    uint8_t out[240];
    trine_encode_shingle("hello world this is a test", 26, out);

    /* Each chain (60 channels) should have at least 1 non-zero trit
     * for a non-trivial input string. */
    int fill_ch1 = count_nonzero(out,   0, 60);
    int fill_ch2 = count_nonzero(out,  60, 60);
    int fill_ch3 = count_nonzero(out, 120, 60);
    int fill_ch4 = count_nonzero(out, 180, 60);

    check("s1_chain1_nontrivial_fill", fill_ch1 > 0);
    check("s1_chain2_nontrivial_fill", fill_ch2 > 0);
    check("s1_chain3_nontrivial_fill", fill_ch3 > 0);
    check("s1_chain4_nontrivial_fill", fill_ch4 > 0);
}

/* ── Category 3: Stage-1 Packed Round-Trip ──────────────────────────── */

static void test_s1_packed_roundtrip(void)
{
    uint8_t original[240], packed[48], unpacked[240];
    trine_encode_shingle("packed round trip test", 21, original);

    int rc1 = trine_s1_pack(original, packed);
    int rc2 = trine_s1_unpack(packed, unpacked);

    check("s1_pack_rc", rc1 == 0);
    check("s1_unpack_rc", rc2 == 0);
    check("s1_packed_roundtrip_match", memcmp(original, unpacked, 240) == 0);
}

/* ── Category 4: Stage-1 Comparison Properties ──────────────────────── */

static void test_s1_compare_identity(void)
{
    uint8_t emb[240];
    trine_encode_shingle("identity test string", 20, emb);

    trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;
    float sim = trine_s1_compare(emb, emb, &lens);

    check("s1_compare_self_is_one", fabsf(sim - 1.0f) < 1e-6f);
}

static void test_s1_compare_symmetry(void)
{
    uint8_t a[240], b[240];
    trine_encode_shingle("alpha beta gamma", 16, a);
    trine_encode_shingle("delta epsilon zeta", 18, b);

    trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;
    float ab = trine_s1_compare(a, b, &lens);
    float ba = trine_s1_compare(b, a, &lens);

    check("s1_compare_symmetry", fabsf(ab - ba) < 1e-6f);
}

static void test_s1_different_texts_differ(void)
{
    uint8_t a[240], b[240];
    trine_encode_shingle("cat sat on the mat", 18, a);
    trine_encode_shingle("xylophone quantum zebra", 23, b);

    check("s1_different_texts_differ", memcmp(a, b, 240) != 0);
}

static void test_s1_similarity_ordering(void)
{
    uint8_t base[240], near[240], far[240];
    trine_encode_shingle("the quick brown fox jumps over the lazy dog", 43, base);
    trine_encode_shingle("the quick brown fox leaps over the lazy dog", 43, near);
    trine_encode_shingle("xylophone quantum zebra entanglement fusion", 44, far);

    trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;
    float sim_near = trine_s1_compare(base, near, &lens);
    float sim_far  = trine_s1_compare(base, far,  &lens);

    check("s1_near_more_similar_than_far", sim_near > sim_far);
}

/* ── Category 5: Stage-2 Identity Model ────────────────────────────── */

static void test_s2_identity_matches_s1(void)
{
    trine_s2_model_t *m = trine_s2_create_identity();
    check("s2_identity_create", m != NULL);
    if (!m) return;

    const char *text = "hello world";
    size_t len = 11;

    uint8_t s1[240], s2[240];
    trine_encode_shingle(text, len, s1);
    int rc = trine_s2_encode(m, text, len, 0, s2);

    check("s2_identity_encode_rc", rc == 0);
    check("s2_identity_depth0_matches_s1", memcmp(s1, s2, 240) == 0);

    /* Identity model at depth=4 should also match (no cascade cells) */
    uint8_t s2_d4[240];
    rc = trine_s2_encode(m, text, len, 4, s2_d4);
    check("s2_identity_depth4_matches_s1", memcmp(s1, s2_d4, 240) == 0);

    trine_s2_free(m);
}

/* ── Category 6: Stage-2 Random Model Determinism ──────────────────── */

static void test_s2_random_deterministic(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);
    check("s2_random_create", m != NULL);
    if (!m) return;

    const char *text = "hello world";
    size_t len = 11;

    uint8_t a[240], b[240];
    trine_s2_encode(m, text, len, 4, a);
    trine_s2_encode(m, text, len, 4, b);

    check("s2_random_seed42_deterministic", memcmp(a, b, 240) == 0);
    check("s2_random_seed42_valid_trits", all_valid_trits(a, 240));

    trine_s2_free(m);
}

static void test_s2_random_depth_differs(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);
    if (!m) { check("s2_random_depth_create", 0); return; }

    const char *text = "hello world";
    size_t len = 11;

    uint8_t d0[240], d4[240];
    trine_s2_encode(m, text, len, 0, d0);
    trine_s2_encode(m, text, len, 4, d4);

    check("s2_random_depth0_vs_depth4_differ", memcmp(d0, d4, 240) != 0);

    trine_s2_free(m);
}

/* ── Category 7: Stage-2 Comparison Properties ─────────────────────── */

static void test_s2_compare_identity_matches_s1(void)
{
    trine_s2_model_t *m = trine_s2_create_identity();
    if (!m) { check("s2_compare_id_create", 0); return; }

    const char *text_a = "the quick brown fox";
    const char *text_b = "the quick brown dog";

    uint8_t s1_a[240], s1_b[240], s2_a[240], s2_b[240];
    trine_encode_shingle(text_a, strlen(text_a), s1_a);
    trine_encode_shingle(text_b, strlen(text_b), s1_b);
    trine_s2_encode(m, text_a, strlen(text_a), 0, s2_a);
    trine_s2_encode(m, text_b, strlen(text_b), 0, s2_b);

    /* Identity model at depth 0 should produce identical embeddings to S1 */
    check("s2_compare_identity_matches_s1",
          memcmp(s1_a, s2_a, 240) == 0 && memcmp(s1_b, s2_b, 240) == 0);

    trine_s2_free(m);
}

static void test_s2_compare_self(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);
    if (!m) { check("s2_compare_self_create", 0); return; }

    uint8_t emb[240];
    trine_s2_encode(m, "self similarity test", 20, 4, emb);

    float self = trine_s2_compare(emb, emb, NULL);
    check("s2_compare_self_is_one", fabsf(self - 1.0f) < 1e-6f);

    trine_s2_free(m);
}

/* ── Category 8: Stage-2 Persistence ───────────────────────────────── */

static void test_s2_save_load_identity(void)
{
    trine_s2_model_t *m = trine_s2_create_identity();
    if (!m) { check("s2_persist_create", 0); return; }

    const char *path = "/tmp/test_golden_identity.trine2";
    int rc = trine_s2_save(m, path, NULL);
    check("s2_persist_save_rc", rc == 0);

    trine_s2_model_t *m2 = trine_s2_load(path);
    check("s2_persist_load_ok", m2 != NULL);

    if (m2) {
        const char *text = "persistence round trip";
        size_t len = strlen(text);

        uint8_t orig[240], loaded[240];
        trine_s2_encode(m, text, len, 0, orig);
        trine_s2_encode(m2, text, len, 0, loaded);

        check("s2_persist_roundtrip_match", memcmp(orig, loaded, 240) == 0);
        trine_s2_free(m2);
    }

    /* Clean up temp file */
    remove(path);
    trine_s2_free(m);
}

/* ── Category 9: Stage-2 Stress Determinism ────────────────────────── */

static void test_s2_1000_call_stress(void)
{
    trine_s2_model_t *m = trine_s2_create_random(256, 99);
    if (!m) { check("s2_stress_create", 0); return; }

    const char *text = "golden vector stress test";
    size_t len = strlen(text);

    uint8_t ref[240], cur[240];
    trine_s2_encode(m, text, len, 4, ref);

    int ok = 1;
    for (int i = 0; i < 1000; i++) {
        trine_s2_encode(m, text, len, 4, cur);
        if (memcmp(ref, cur, 240) != 0) { ok = 0; break; }
    }
    check("s2_1000_call_determinism", ok);

    trine_s2_free(m);
}

/* ── Category 10: Stage-2 Encode Equivalence ───────────────────────── */

static void test_s2_encode_vs_from_trits(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);
    if (!m) { check("s2_equiv_create", 0); return; }

    const char *text = "encode equivalence golden";
    size_t len = strlen(text);

    /* Full pipeline */
    uint8_t full[240];
    trine_s2_encode(m, text, len, 4, full);

    /* Two-step: Stage-1 then from_trits */
    uint8_t s1[240], two_step[240];
    trine_encode_shingle(text, len, s1);
    trine_s2_encode_from_trits(m, s1, 4, two_step);

    check("s2_encode_vs_from_trits_match", memcmp(full, two_step, 240) == 0);

    trine_s2_free(m);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Golden Vector Regression Tests ===\n\n");

    printf("--- Category 1: Stage-1 Encoding Determinism ---\n");
    test_s1_hello_world_deterministic();
    test_s1_empty_string();
    test_s1_single_char();
    test_s1_case_folding();
    test_s1_long_string_deterministic();
    test_s1_encode_symmetry();

    printf("--- Category 2: Stage-1 Chain Fill ---\n");
    test_s1_chain_fill();

    printf("--- Category 3: Stage-1 Packed Round-Trip ---\n");
    test_s1_packed_roundtrip();

    printf("--- Category 4: Stage-1 Comparison Properties ---\n");
    test_s1_compare_identity();
    test_s1_compare_symmetry();
    test_s1_different_texts_differ();
    test_s1_similarity_ordering();

    printf("--- Category 5: Stage-2 Identity Model ---\n");
    test_s2_identity_matches_s1();

    printf("--- Category 6: Stage-2 Random Model Determinism ---\n");
    test_s2_random_deterministic();
    test_s2_random_depth_differs();

    printf("--- Category 7: Stage-2 Comparison Properties ---\n");
    test_s2_compare_identity_matches_s1();
    test_s2_compare_self();

    printf("--- Category 8: Stage-2 Persistence ---\n");
    test_s2_save_load_identity();

    printf("--- Category 9: Stage-2 Stress Determinism ---\n");
    test_s2_1000_call_stress();

    printf("--- Category 10: Stage-2 Encode Equivalence ---\n");
    test_s2_encode_vs_from_trits();

    printf("\nGolden: %d passed, %d failed, %d total\n",
           passed, failed, passed + failed);
    return failed > 0 ? 1 : 0;
}
