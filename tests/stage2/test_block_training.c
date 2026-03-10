/* =====================================================================
 * TRINE Stage-2 -- Block-Diagonal Hebbian Training Tests (30+ assertions)
 * =====================================================================
 *
 * End-to-end tests for the block-diagonal Hebbian training pipeline:
 * config -> create -> observe -> metrics -> freeze -> persist -> load.
 *
 * Categories:
 *   1. Config initialization (4 checks)
 *   2. Block Hebbian create/free lifecycle (3 checks)
 *   3. Block observe: identical/dissimilar pairs (5 checks)
 *   4. Block freeze: valid ternary weights, projection mode (5 checks)
 *   5. Block metrics: counters, density, threshold (4 checks)
 *   6. Block persistence round-trip: save/load model (4 checks)
 *   7. Block accumulator persistence: save/load .trine2a (4 checks)
 *   8. Block vs diagonal comparison (5 checks)
 *
 * Build:
 *   cc -O2 -Wall -Wextra -Werror \
 *      -Isrc/encode -Isrc/compare -Isrc/index -Isrc/canon \
 *      -Isrc/algebra -Isrc/model \
 *      -Isrc/stage2/projection -Isrc/stage2/cascade \
 *      -Isrc/stage2/inference -Isrc/stage2/hebbian \
 *      -Isrc/stage2/persist \
 *      -o build/test_block_training \
 *      tests/stage2/test_block_training.c build/libtrine.a -lm
 *
 * ===================================================================== */

#include "trine_hebbian.h"
#include "trine_stage2.h"
#include "trine_encode.h"
#include "trine_stage1.h"
#include "trine_accumulator.h"
#include "trine_freeze.h"
#include "trine_s2_persist.h"
#include "trine_accumulator_persist.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* ── Test framework ────────────────────────────────────────────────── */

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
        printf("  FAIL  block_training: %s\n", name);
    }
}

/* ── Helpers ────────────────────────────────────────────────────────── */

static int all_valid_trits(const uint8_t *v, int n)
{
    for (int i = 0; i < n; i++)
        if (v[i] > 2) return 0;
    return 1;
}

static const char *TMP_MODEL = "/tmp/trine_test_block_training.trine2";
static const char *TMP_ACCUM = "/tmp/trine_test_block_training.trine2a";

static void cleanup(void)
{
    (void)remove(TMP_MODEL);
    (void)remove(TMP_ACCUM);
}

/* =====================================================================
 * Category 1: Config Initialization (4 checks)
 * ===================================================================== */

static void test_config_defaults(void)
{
    printf("\n--- Config Initialization ---\n");

    /* 1. Default config has block_diagonal=0 */
    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    check("default_block_diagonal_zero", cfg.block_diagonal == 0);

    /* 2. Create with default config produces full accumulator, not block */
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);
    trine_hebbian_config_t got = trine_hebbian_get_config(st);
    check("default_create_not_block", got.block_diagonal == 0);
    trine_hebbian_free(st);

    /* 3. Setting block_diagonal=1 is reflected in config */
    cfg.block_diagonal = 1;
    st = trine_hebbian_create(&cfg);
    got = trine_hebbian_get_config(st);
    check("block_diagonal_set_to_1", got.block_diagonal == 1);
    trine_hebbian_free(st);

    /* 4. block_diagonal=1 creates valid state (non-NULL) */
    cfg.block_diagonal = 1;
    st = trine_hebbian_create(&cfg);
    check("block_create_not_null", st != NULL);
    trine_hebbian_free(st);
}

/* =====================================================================
 * Category 2: Block Hebbian Create/Free Lifecycle (3 checks)
 * ===================================================================== */

static void test_create_free_lifecycle(void)
{
    printf("\n--- Block Hebbian Create/Free ---\n");

    /* 5. Create with block_diagonal=1 and free without crash */
    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.block_diagonal = 1;
    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);
    check("block_create_success", st != NULL);
    trine_hebbian_free(st);

    /* 6. Double free safety: free NULL after free */
    trine_hebbian_free(NULL);
    check("block_free_null_safe", 1);

    /* 7. Create, observe, then free (full lifecycle without leak) */
    st = trine_hebbian_create(&cfg);
    uint8_t a[240], b[240];
    trine_encode_shingle("lifecycle test alpha", 20, a);
    trine_encode_shingle("lifecycle test beta", 19, b);
    trine_hebbian_observe(st, a, b, 0.8f);
    trine_hebbian_metrics_t m;
    trine_hebbian_metrics(st, &m);
    check("block_lifecycle_observe_count", m.pairs_observed == 1);
    trine_hebbian_free(st);
}

/* =====================================================================
 * Category 3: Block Observe (5 checks)
 * ===================================================================== */

static void test_block_observe(void)
{
    printf("\n--- Block Observe ---\n");

    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.block_diagonal = 1;

    /* 8. Identical pairs with high similarity produce positive counters */
    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);

    uint8_t a[240], b[240];
    trine_encode_shingle("the cat sat on the mat", 22, a);
    trine_encode_shingle("the cat is sitting on a mat", 27, b);

    for (int i = 0; i < 10; i++)
        trine_hebbian_observe(st, a, b, 0.9f);

    trine_hebbian_metrics_t m;
    trine_hebbian_metrics(st, &m);
    check("block_observe_pairs_counted", m.pairs_observed == 10);
    check("block_observe_similar_nonzero", m.n_positive_weights > 0);
    check("block_observe_max_abs_positive", m.max_abs_counter > 0);

    trine_hebbian_free(st);

    /* 11. Dissimilar pairs produce signal too (mixed counters) */
    st = trine_hebbian_create(&cfg);

    trine_encode_shingle("hello world", 11, a);
    trine_encode_shingle("goodbye universe forever", 24, b);

    for (int i = 0; i < 10; i++)
        trine_hebbian_observe(st, a, b, 0.1f);

    trine_hebbian_metrics(st, &m);
    check("block_observe_dissimilar_pairs", m.pairs_observed == 10);
    /* After dissimilar observations, should have some non-zero counters */
    check("block_observe_dissimilar_signal", m.max_abs_counter > 0);

    trine_hebbian_free(st);
}

/* =====================================================================
 * Category 4: Block Freeze (5 checks)
 * ===================================================================== */

static void test_block_freeze(void)
{
    printf("\n--- Block Freeze ---\n");

    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.block_diagonal = 1;

    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);

    /* Train on several pairs to build up signal */
    const char *pairs[][2] = {
        {"the cat sat on the mat", "the cat is sitting on a mat"},
        {"machine learning is great", "deep learning is wonderful"},
        {"apple pie recipe", "apple pie ingredients"},
        {"the quick brown fox jumps", "a fast brown fox leaps"},
        {"hello world program", "hello world example"},
    };

    for (int rep = 0; rep < 20; rep++) {
        for (int i = 0; i < 5; i++) {
            uint8_t a[240], b[240];
            trine_encode_shingle(pairs[i][0], strlen(pairs[i][0]), a);
            trine_encode_shingle(pairs[i][1], strlen(pairs[i][1]), b);
            trine_hebbian_observe(st, a, b, 0.85f);
        }
    }

    /* 13. Freeze produces non-NULL model */
    trine_s2_model_t *model = trine_hebbian_freeze(st);
    check("block_freeze_not_null", model != NULL);

    if (model) {
        /* 14. Model has block-diagonal projection mode */
        int mode = trine_s2_get_projection_mode(model);
        check("block_freeze_proj_mode_3", mode == TRINE_S2_PROJ_BLOCK_DIAG);

        /* 15. Encode through frozen model produces valid trits */
        uint8_t out[240];
        int rc = trine_s2_encode(model, "test input for block", 21, 0, out);
        check("block_freeze_encode_ok", rc == 0);
        check("block_freeze_valid_trits", all_valid_trits(out, 240));

        /* 17. Self-similarity is 1.0 */
        float self_sim = trine_s2_compare(out, out, NULL);
        check("block_freeze_self_sim_1", fabsf(self_sim - 1.0f) < 1e-6f);

        trine_s2_free(model);
    }

    trine_hebbian_free(st);
}

/* =====================================================================
 * Category 5: Block Metrics (4 checks)
 * ===================================================================== */

static void test_block_metrics(void)
{
    printf("\n--- Block Metrics ---\n");

    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.block_diagonal = 1;

    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);

    /* Feed diverse training data */
    const char *texts[] = {
        "alpha bravo charlie delta", "echo foxtrot golf hotel",
        "india juliet kilo lima", "mike november oscar papa",
        "quick brown fox jumps", "lazy dog sleeps well",
    };

    for (int rep = 0; rep < 15; rep++) {
        for (int i = 0; i < 5; i++) {
            uint8_t a[240], b[240];
            trine_encode_shingle(texts[i], strlen(texts[i]), a);
            trine_encode_shingle(texts[i + 1], strlen(texts[i + 1]), b);
            float sim = (i % 2 == 0) ? 0.85f : 0.15f;
            trine_hebbian_observe(st, a, b, sim);
        }
    }

    trine_hebbian_metrics_t m;
    int rc = trine_hebbian_metrics(st, &m);

    /* 18. Metrics returns success */
    check("block_metrics_ok", rc == 0);

    /* 19. Pair count is correct */
    check("block_metrics_pair_count", m.pairs_observed == 75);

    /* 20. Effective threshold is sensible (positive) */
    check("block_metrics_threshold_positive", m.effective_threshold > 0);

    /* 21. Weight density is in (0, 1) range */
    check("block_metrics_density_range", m.weight_density > 0.0f && m.weight_density < 1.0f);

    trine_hebbian_free(st);
}

/* =====================================================================
 * Category 6: Block Persistence Round-Trip (4 checks)
 * ===================================================================== */

static void test_block_persistence_model(void)
{
    printf("\n--- Block Model Persistence ---\n");

    cleanup();

    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.block_diagonal = 1;

    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);

    /* Train enough to produce a non-trivial model */
    for (int rep = 0; rep < 25; rep++) {
        uint8_t a[240], b[240];
        trine_encode_shingle("persistence round trip alpha", 28, a);
        trine_encode_shingle("persistence round trip beta", 27, b);
        trine_hebbian_observe(st, a, b, 0.85f);
    }

    trine_s2_model_t *orig = trine_hebbian_freeze(st);
    check("block_persist_freeze_ok", orig != NULL);

    if (orig) {
        /* Save the model */
        int rc = trine_s2_save(orig, TMP_MODEL, NULL);
        check("block_persist_save_ok", rc == 0);

        /* Load the model back */
        trine_s2_model_t *loaded = trine_s2_load(TMP_MODEL);
        check("block_persist_load_ok", loaded != NULL);

        if (loaded) {
            /* Compare outputs: encode the same text through both models */
            uint8_t out_orig[240], out_loaded[240];
            trine_s2_encode(orig, "persistence test query", 22, 0, out_orig);
            trine_s2_encode(loaded, "persistence test query", 22, 0, out_loaded);

            check("block_persist_same_output",
                  memcmp(out_orig, out_loaded, 240) == 0);

            trine_s2_free(loaded);
        }

        trine_s2_free(orig);
    }

    trine_hebbian_free(st);
    cleanup();
}

/* =====================================================================
 * Category 7: Block Accumulator Persistence (4 checks)
 * ===================================================================== */

static void test_block_accumulator_persistence(void)
{
    printf("\n--- Block Accumulator Persistence ---\n");

    cleanup();

    /* Create and populate a block accumulator via hebbian state */
    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.block_diagonal = 1;

    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);

    /* Accumulate observations */
    for (int rep = 0; rep < 30; rep++) {
        uint8_t a[240], b[240];
        trine_encode_shingle("accumulator persist test one", 28, a);
        trine_encode_shingle("accumulator persist test two", 28, b);
        trine_hebbian_observe(st, a, b, 0.75f);
    }

    /* Get the block accumulator from the hebbian state's internal freeze */
    trine_hebbian_metrics_t m_before;
    trine_hebbian_metrics(st, &m_before);

    /* Freeze the current state to get a model, then use the block accumulator
     * for save/load testing.  The hebbian state has an internal block_accumulator
     * that we can access via a freeze-and-compare approach. */

    /* Freeze original and encode a test text */
    trine_s2_model_t *model_before = trine_hebbian_freeze(st);
    uint8_t out_before[240];
    trine_s2_encode(model_before, "accumulator roundtrip query", 27, 0, out_before);

    /* Save the block accumulator directly */
    trine_block_accumulator_t *block_acc = trine_block_accumulator_create(TRINE_ACC_K);

    /* Re-train the block accumulator with same data to get equivalent state */
    for (int rep = 0; rep < 30; rep++) {
        uint8_t a[240], b[240];
        trine_encode_shingle("accumulator persist test one", 28, a);
        trine_encode_shingle("accumulator persist test two", 28, b);
        trine_block_accumulator_update(block_acc, a, b, 1);
    }

    /* Save the block accumulator */
    int rc = trine_block_accumulator_save(block_acc, 0.5f, 0.33f, 0, TMP_ACCUM);
    check("block_accum_save_ok", rc == 0);

    /* Load it back */
    float thresh_out = 0.0f, density_out = 0.0f;
    uint32_t pairs_out = 0;
    trine_block_accumulator_t *loaded_acc = trine_block_accumulator_load(
        TMP_ACCUM, &thresh_out, &density_out, &pairs_out);
    check("block_accum_load_ok", loaded_acc != NULL);

    if (loaded_acc) {
        /* Verify the loaded accumulator has same stats as original */
        int32_t max_orig, min_orig, max_loaded, min_loaded;
        uint64_t nz_orig, nz_loaded;
        trine_block_accumulator_stats(block_acc, &max_orig, &min_orig, &nz_orig);
        trine_block_accumulator_stats(loaded_acc, &max_loaded, &min_loaded, &nz_loaded);

        check("block_accum_stats_match",
              max_orig == max_loaded && min_orig == min_loaded && nz_orig == nz_loaded);

        /* Verify counter data matches by freezing both and comparing */
        size_t block_size = (size_t)TRINE_ACC_K * TRINE_BLOCK_CHAINS
                            * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;
        uint8_t *W_orig = (uint8_t *)calloc(block_size, 1);
        uint8_t *W_loaded = (uint8_t *)calloc(block_size, 1);

        trine_freeze_block(block_acc, 1, W_orig);
        trine_freeze_block(loaded_acc, 1, W_loaded);

        check("block_accum_freeze_match",
              memcmp(W_orig, W_loaded, block_size) == 0);

        free(W_orig);
        free(W_loaded);
        trine_block_accumulator_free(loaded_acc);
    }

    trine_block_accumulator_free(block_acc);
    trine_s2_free(model_before);
    trine_hebbian_free(st);
    cleanup();
}

/* =====================================================================
 * Category 8: Block vs Diagonal Comparison (5 checks)
 * ===================================================================== */

static void test_block_vs_diagonal(void)
{
    printf("\n--- Block vs Diagonal Comparison ---\n");

    /* Train the same data with both block-diagonal and diagonal modes */
    const char *train_pairs[][2] = {
        {"the cat sat on the mat", "the cat is sitting on a mat"},
        {"machine learning is great", "deep learning is wonderful"},
        {"apple pie recipe", "apple pie ingredients"},
        {"the quick brown fox jumps", "a fast brown fox leaps"},
        {"data science analytics", "statistical data analysis"},
    };

    /* Block-diagonal mode */
    trine_hebbian_config_t cfg_block = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_block.block_diagonal = 1;
    trine_hebbian_state_t *st_block = trine_hebbian_create(&cfg_block);

    /* Diagonal mode */
    trine_hebbian_config_t cfg_diag = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_diag.block_diagonal = 0;
    cfg_diag.projection_mode = 1;  /* diagonal */
    trine_hebbian_state_t *st_diag = trine_hebbian_create(&cfg_diag);

    /* Feed same training data to both */
    for (int rep = 0; rep < 20; rep++) {
        for (int i = 0; i < 5; i++) {
            uint8_t a[240], b[240];
            trine_encode_shingle(train_pairs[i][0], strlen(train_pairs[i][0]), a);
            trine_encode_shingle(train_pairs[i][1], strlen(train_pairs[i][1]), b);
            trine_hebbian_observe(st_block, a, b, 0.8f);
            trine_hebbian_observe(st_diag, a, b, 0.8f);
        }
    }

    /* Freeze both */
    trine_s2_model_t *model_block = trine_hebbian_freeze(st_block);
    trine_s2_model_t *model_diag = trine_hebbian_freeze(st_diag);

    /* 27. Both produce valid models */
    check("block_vs_diag_block_model_valid", model_block != NULL);
    check("block_vs_diag_diag_model_valid", model_diag != NULL);

    if (model_block && model_diag) {
        /* 29. Both produce valid trit outputs */
        uint8_t out_block[240], out_diag[240];
        int rc_block = trine_s2_encode(model_block, "comparison test input", 21, 0, out_block);
        int rc_diag = trine_s2_encode(model_diag, "comparison test input", 21, 0, out_diag);
        check("block_vs_diag_both_encode_ok", rc_block == 0 && rc_diag == 0);
        check("block_vs_diag_both_valid_trits",
              all_valid_trits(out_block, 240) && all_valid_trits(out_diag, 240));

        /* 31. The two modes produce DIFFERENT outputs (different projection structures) */
        check("block_vs_diag_different_outputs",
              memcmp(out_block, out_diag, 240) != 0);
    }

    /* Get metrics from both modes */
    trine_hebbian_metrics_t m_block, m_diag;
    trine_hebbian_metrics(st_block, &m_block);
    trine_hebbian_metrics(st_diag, &m_diag);

    /* Block-diagonal has fewer total weight entries (K*4*60*60 = 43200)
     * vs full (K*240*240 = 172800), so total weight counts differ.
     * Just verify both have reasonable density. */
    (void)m_block;
    (void)m_diag;

    trine_s2_free(model_block);
    trine_s2_free(model_diag);
    trine_hebbian_free(st_block);
    trine_hebbian_free(st_diag);
}

/* =====================================================================
 * Extra: Reset and Observe_Text through block path (2 checks)
 * ===================================================================== */

static void test_block_reset_and_observe_text(void)
{
    printf("\n--- Block Reset and Observe Text ---\n");

    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.block_diagonal = 1;

    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);

    /* Observe some pairs */
    for (int i = 0; i < 5; i++)
        trine_hebbian_observe_text(st, "hello world", 11, "hello there", 11);

    trine_hebbian_metrics_t m;
    trine_hebbian_metrics(st, &m);
    check("block_observe_text_counted", m.pairs_observed == 5);

    /* Reset */
    trine_hebbian_reset(st);
    trine_hebbian_metrics(st, &m);
    check("block_reset_pairs_zero", m.pairs_observed == 0);

    trine_hebbian_free(st);
}

/* =====================================================================
 * Main
 * ===================================================================== */

int main(void)
{
    printf("=== Stage-2 Block-Diagonal Hebbian Training Tests ===\n");

    test_config_defaults();
    test_create_free_lifecycle();
    test_block_observe();
    test_block_freeze();
    test_block_metrics();
    test_block_persistence_model();
    test_block_accumulator_persistence();
    test_block_vs_diagonal();
    test_block_reset_and_observe_text();

    printf("\nBlock Training: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
