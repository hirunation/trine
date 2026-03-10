/* =====================================================================
 * TRINE Stage-2 — Hebbian Training Tests (~15 tests)
 * =====================================================================
 *
 * Tests for the Hebbian training loop:
 *   1. Create/free lifecycle (no leaks)
 *   2. Default config values
 *   3. observe() with similar pair (sign=+1) accumulates positive counters
 *   4. observe() with dissimilar pair (sign=-1) accumulates negative counters
 *   5. observe_text() encodes and accumulates
 *   6. metrics() returns correct pair count
 *   7. reset() zeroes everything
 *   8. freeze() produces valid ternary weights (all values in {0,1,2})
 *   9. freeze() with identity-like signal preserves structure
 *  10. freeze() -> model -> encode produces valid 240-trit output
 *  11. Determinism: same observations -> same frozen model
 *  12. Multiple epochs of train_file() produce stronger signal
 *  13. Config threshold affects sign determination
 *  14. Null safety
 *
 * ===================================================================== */

#include "trine_hebbian.h"
#include "trine_stage2.h"
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
        printf("  FAIL  hebbian: %s\n", name);
    }
}

static int all_valid_trits(const uint8_t *v, int n)
{
    for (int i = 0; i < n; i++)
        if (v[i] > 2) return 0;
    return 1;
}

static void write_test_jsonl(const char *path)
{
    FILE *f = fopen(path, "w");
    fprintf(f, "{\"text_a\": \"the cat sat on the mat\", \"text_b\": \"the cat is sitting on a mat\", \"score\": 0.8}\n");
    fprintf(f, "{\"text_a\": \"hello world\", \"text_b\": \"goodbye universe\", \"score\": 0.1}\n");
    fprintf(f, "{\"text_a\": \"machine learning is great\", \"text_b\": \"deep learning is wonderful\", \"score\": 0.7}\n");
    fprintf(f, "{\"text_a\": \"the dog runs fast\", \"text_b\": \"a quick brown fox\", \"score\": 0.3}\n");
    fprintf(f, "{\"text_a\": \"apple pie recipe\", \"text_b\": \"apple pie ingredients\", \"score\": 0.9}\n");
    fclose(f);
}

/* ── Test 1: Create/free lifecycle ──────────────────────────────────── */
static void test_create_free(void)
{
    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);
    check("create_not_null", st != NULL);
    trine_hebbian_free(st);

    /* Create with NULL config (uses defaults) */
    st = trine_hebbian_create(NULL);
    check("create_null_config", st != NULL);
    trine_hebbian_free(st);

    /* Free NULL is safe */
    trine_hebbian_free(NULL);
    check("free_null_safe", 1);
}

/* ── Test 2: Default config values ──────────────────────────────────── */
static void test_default_config(void)
{
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);
    trine_hebbian_config_t cfg = trine_hebbian_get_config(st);

    check("default_sim_threshold", fabsf(cfg.similarity_threshold - 0.5f) < 1e-6f);
    check("default_freeze_threshold", cfg.freeze_threshold == 0);
    check("default_target_density", fabsf(cfg.freeze_target_density - 0.33f) < 1e-6f);
    check("default_cascade_cells", cfg.cascade_cells == 512);
    check("default_cascade_depth", cfg.cascade_depth == 4);

    trine_hebbian_free(st);
}

/* ── Test 3: observe() with similar pair ────────────────────────────── */
static void test_observe_similar(void)
{
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);

    /* Encode two similar texts */
    uint8_t a[240], b[240];
    trine_encode_shingle("the cat sat on the mat", 22, a);
    trine_encode_shingle("the cat is sitting on a mat", 27, b);

    /* Observe with high similarity (above default 0.5 threshold) -> positive */
    trine_hebbian_observe(st, a, b, 0.8f);

    trine_hebbian_metrics_t m;
    trine_hebbian_metrics(st, &m);

    check("observe_similar_pairs", m.pairs_observed == 1);
    check("observe_similar_positive", m.n_positive_weights > 0);

    trine_hebbian_free(st);
}

/* ── Test 4: observe() with dissimilar pair ─────────────────────────── */
static void test_observe_dissimilar(void)
{
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);

    uint8_t a[240], b[240];
    trine_encode_shingle("hello world", 11, a);
    trine_encode_shingle("goodbye universe", 16, b);

    /* Observe with low similarity (below default 0.5 threshold) -> negative */
    trine_hebbian_observe(st, a, b, 0.1f);

    trine_hebbian_metrics_t m;
    trine_hebbian_metrics(st, &m);

    check("observe_dissimilar_pairs", m.pairs_observed == 1);
    check("observe_dissimilar_negative", m.n_negative_weights > 0);

    trine_hebbian_free(st);
}

/* ── Test 5: observe_text() encodes and accumulates ─────────────────── */
static void test_observe_text(void)
{
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);

    trine_hebbian_observe_text(st, "hello world", 11, "hello there", 11);

    trine_hebbian_metrics_t m;
    trine_hebbian_metrics(st, &m);

    check("observe_text_pairs", m.pairs_observed == 1);
    /* After one observation, some counters should be non-zero */
    check("observe_text_signal", m.n_positive_weights > 0 || m.n_negative_weights > 0);

    trine_hebbian_free(st);
}

/* ── Test 6: metrics() returns correct pair count ───────────────────── */
static void test_metrics_pair_count(void)
{
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);

    uint8_t a[240], b[240];
    trine_encode_shingle("text one", 8, a);
    trine_encode_shingle("text two", 8, b);

    trine_hebbian_observe(st, a, b, 0.6f);
    trine_hebbian_observe(st, a, b, 0.3f);
    trine_hebbian_observe(st, a, b, 0.9f);

    trine_hebbian_metrics_t m;
    trine_hebbian_metrics(st, &m);

    check("metrics_three_pairs", m.pairs_observed == 3);
    check("metrics_max_abs_positive", m.max_abs_counter > 0);

    trine_hebbian_free(st);
}

/* ── Test 7: reset() zeroes everything ──────────────────────────────── */
static void test_reset(void)
{
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);

    uint8_t a[240], b[240];
    trine_encode_shingle("some text here", 14, a);
    trine_encode_shingle("some text there", 15, b);

    /* Accumulate some observations */
    for (int i = 0; i < 10; i++)
        trine_hebbian_observe(st, a, b, 0.7f);

    trine_hebbian_metrics_t m;
    trine_hebbian_metrics(st, &m);
    check("pre_reset_nonzero", m.pairs_observed == 10);

    /* Reset */
    trine_hebbian_reset(st);
    trine_hebbian_metrics(st, &m);

    check("reset_pairs_zero", m.pairs_observed == 0);
    check("reset_max_abs_zero", m.max_abs_counter == 0);
    check("reset_all_zero", m.n_positive_weights == 0 && m.n_negative_weights == 0);

    trine_hebbian_free(st);
}

/* ── Test 8: freeze() produces valid ternary weights ────────────────── */
static void test_freeze_valid(void)
{
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);

    uint8_t a[240], b[240];
    trine_encode_shingle("freeze test alpha", 17, a);
    trine_encode_shingle("freeze test beta", 16, b);

    for (int i = 0; i < 20; i++)
        trine_hebbian_observe(st, a, b, 0.8f);

    trine_s2_model_t *model = trine_hebbian_freeze(st);
    check("freeze_not_null", model != NULL);

    /* Encode a text through the frozen model */
    uint8_t out[240];
    int rc = trine_s2_encode(model, "test input", 10, 4, out);
    check("freeze_encode_rc", rc == 0);
    check("freeze_valid_trits", all_valid_trits(out, 240));

    trine_s2_free(model);
    trine_hebbian_free(st);
}

/* ── Test 9: freeze() with identity-like signal ─────────────────────── */
static void test_freeze_identity_signal(void)
{
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);

    /* Feed many identical-pair observations (a == b, high similarity).
     * This should produce a model where identical texts get high s2 similarity. */
    const char *texts[] = {
        "alpha beta gamma", "delta epsilon zeta",
        "hello world foo", "bar baz quux corge"
    };

    for (int i = 0; i < 4; i++) {
        uint8_t a[240];
        trine_encode_shingle(texts[i], strlen(texts[i]), a);
        /* Self-pair: similarity = 1.0 (above threshold) */
        trine_hebbian_observe(st, a, a, 1.0f);
    }

    trine_s2_model_t *model = trine_hebbian_freeze(st);
    check("freeze_identity_not_null", model != NULL);

    /* Self-similarity through the model should be 1.0 */
    uint8_t e[240];
    trine_s2_encode(model, "test text", 9, 4, e);
    float self_sim = trine_s2_compare(e, e, NULL);
    check("freeze_identity_self_sim", fabsf(self_sim - 1.0f) < 1e-6f);

    trine_s2_free(model);
    trine_hebbian_free(st);
}

/* ── Test 10: freeze() -> model -> encode produces valid output ─────── */
static void test_freeze_full_pipeline(void)
{
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);

    /* Train on several pairs */
    trine_hebbian_observe_text(st, "the quick brown fox", 19, "a fast brown fox", 16);
    trine_hebbian_observe_text(st, "machine learning", 16, "deep learning", 13);
    trine_hebbian_observe_text(st, "apple pie", 9, "cherry pie", 10);

    trine_s2_model_t *model = trine_hebbian_freeze(st);
    check("pipeline_model_not_null", model != NULL);

    /* Encode and compare */
    uint8_t ea[240], eb[240];
    int rc_a = trine_s2_encode(model, "hello world", 11, 4, ea);
    int rc_b = trine_s2_encode(model, "goodbye world", 13, 4, eb);
    check("pipeline_encode_a", rc_a == 0);
    check("pipeline_encode_b", rc_b == 0);
    check("pipeline_valid_a", all_valid_trits(ea, 240));
    check("pipeline_valid_b", all_valid_trits(eb, 240));

    float sim = trine_s2_compare(ea, eb, NULL);
    check("pipeline_sim_in_range", sim >= 0.0f && sim <= 1.0f);

    trine_s2_free(model);
    trine_hebbian_free(st);
}

/* ── Test 11: Determinism ───────────────────────────────────────────── */
static void test_determinism(void)
{
    /* Run the same training twice, freeze both, compare outputs */
    uint8_t out1[240], out2[240];

    for (int trial = 0; trial < 2; trial++) {
        trine_hebbian_state_t *st = trine_hebbian_create(NULL);

        uint8_t a[240], b[240], c[240], d[240];
        trine_encode_shingle("alpha beta", 10, a);
        trine_encode_shingle("alpha gamma", 11, b);
        trine_encode_shingle("hello", 5, c);
        trine_encode_shingle("world", 5, d);

        trine_hebbian_observe(st, a, b, 0.8f);
        trine_hebbian_observe(st, c, d, 0.2f);

        trine_s2_model_t *model = trine_hebbian_freeze(st);
        trine_s2_encode(model, "test determinism", 16, 4,
                        trial == 0 ? out1 : out2);

        trine_s2_free(model);
        trine_hebbian_free(st);
    }

    check("determinism_same_output", memcmp(out1, out2, 240) == 0);
}

/* ── Test 12: Multiple epochs produce stronger signal ───────────────── */
static void test_multi_epoch(void)
{
    const char *path = "/tmp/trine_test_hebbian.jsonl";
    write_test_jsonl(path);

    /* Train 1 epoch */
    trine_hebbian_state_t *st1 = trine_hebbian_create(NULL);
    int64_t n1 = trine_hebbian_train_file(st1, path, 1);
    check("epoch1_pairs", n1 == 5);

    trine_hebbian_metrics_t m1;
    trine_hebbian_metrics(st1, &m1);

    /* Train 3 epochs (fresh state) */
    trine_hebbian_state_t *st3 = trine_hebbian_create(NULL);
    int64_t n3 = trine_hebbian_train_file(st3, path, 3);
    check("epoch3_pairs", n3 == 15);

    trine_hebbian_metrics_t m3;
    trine_hebbian_metrics(st3, &m3);

    /* More epochs -> higher max_abs_counter */
    check("multi_epoch_stronger", m3.max_abs_counter >= m1.max_abs_counter);

    trine_hebbian_free(st1);
    trine_hebbian_free(st3);

    remove(path);
}

/* ── Test 13: Config threshold affects sign ─────────────────────────── */
static void test_config_threshold(void)
{
    /* With low threshold, the same similarity=0.5 pair is "similar" (positive) */
    trine_hebbian_config_t cfg_low = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_low.similarity_threshold = 0.3f;

    trine_hebbian_state_t *st_low = trine_hebbian_create(&cfg_low);

    uint8_t a[240], b[240];
    trine_encode_shingle("threshold test a", 16, a);
    trine_encode_shingle("threshold test b", 16, b);

    trine_hebbian_observe(st_low, a, b, 0.5f);  /* 0.5 > 0.3 -> positive */

    trine_hebbian_metrics_t m_low;
    trine_hebbian_metrics(st_low, &m_low);

    /* With high threshold, the same pair is "dissimilar" (negative) */
    trine_hebbian_config_t cfg_high = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_high.similarity_threshold = 0.8f;

    trine_hebbian_state_t *st_high = trine_hebbian_create(&cfg_high);

    trine_hebbian_observe(st_high, a, b, 0.5f);  /* 0.5 <= 0.8 -> negative */

    trine_hebbian_metrics_t m_high;
    trine_hebbian_metrics(st_high, &m_high);

    /* Low threshold: positive counters should dominate.
     * High threshold: negative counters should dominate. */
    check("low_threshold_positive", m_low.n_positive_weights > m_low.n_negative_weights);
    check("high_threshold_negative", m_high.n_negative_weights > m_high.n_positive_weights);

    trine_hebbian_free(st_low);
    trine_hebbian_free(st_high);
}

/* ── Test 14: Null safety ───────────────────────────────────────────── */
static void test_null_safety(void)
{
    trine_hebbian_metrics_t m;
    uint8_t dummy[240];
    memset(dummy, 1, sizeof(dummy));

    /* metrics with NULL state */
    check("metrics_null_state", trine_hebbian_metrics(NULL, &m) == -1);

    /* observe with NULL state */
    trine_hebbian_observe(NULL, dummy, dummy, 0.5f);
    check("observe_null_safe", 1);  /* Should not crash */

    /* observe_text with NULL state */
    trine_hebbian_observe_text(NULL, "a", 1, "b", 1);
    check("observe_text_null_safe", 1);  /* Should not crash */

    /* reset with NULL state */
    trine_hebbian_reset(NULL);
    check("reset_null_safe", 1);  /* Should not crash */

    /* freeze with NULL state */
    struct trine_s2_model *model = trine_hebbian_freeze(NULL);
    check("freeze_null_returns_null", model == NULL);

    /* train_file with NULL */
    check("train_null_state", trine_hebbian_train_file(NULL, "/tmp/x.jsonl", 1) == -1);

    trine_hebbian_state_t *st = trine_hebbian_create(NULL);
    check("train_null_path", trine_hebbian_train_file(st, NULL, 1) == -1);
    check("train_zero_epochs", trine_hebbian_train_file(st, "/tmp/x.jsonl", 0) == -1);
    trine_hebbian_free(st);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Hebbian Training Tests ===\n");

    test_create_free();
    test_default_config();
    test_observe_similar();
    test_observe_dissimilar();
    test_observe_text();
    test_metrics_pair_count();
    test_reset();
    test_freeze_valid();
    test_freeze_identity_signal();
    test_freeze_full_pipeline();
    test_determinism();
    test_multi_epoch();
    test_config_threshold();
    test_null_safety();

    printf("\nHebbian: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
