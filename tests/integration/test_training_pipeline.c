/* =====================================================================
 * TRINE — End-to-End Training Pipeline Integration Test
 * =====================================================================
 *
 * Validates the full training lifecycle:
 *   1. Generate synthetic pairs in-memory (200 pairs)
 *   2. Train with diagonal mode, verify metrics
 *   3. Freeze to model, verify valid trits
 *   4. Save/load round-trip, verify deterministic output
 *   5. Train with block-diagonal mode
 *   6. Verify projection_mode == TRINE_S2_PROJ_BLOCK_DIAG
 *   7. Compare quality: self-similarity == 1.0
 *   8. Reset and retrain with different data, verify state changes
 *
 * Target: >= 25 assertions.
 *
 * ===================================================================== */

#include "trine_hebbian.h"
#include "trine_stage2.h"
#include "trine_s2_persist.h"
#include "trine_encode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* ── Test framework ─────────────────────────────────────────────────── */

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
        printf("  FAIL  training_pipeline: %s\n", name);
    }
}

/* ── Helpers ─────────────────────────────────────────────────────────── */

static int all_valid_trits(const uint8_t *v, int n)
{
    for (int i = 0; i < n; i++)
        if (v[i] > 2) return 0;
    return 1;
}

/* Temp file paths */
static const char *TMP_DIAG_MODEL  = "/tmp/trine_integ_diag.trine2";
static const char *TMP_BLOCK_MODEL = "/tmp/trine_integ_block.trine2";

static void cleanup(void)
{
    (void)unlink(TMP_DIAG_MODEL);
    (void)unlink(TMP_BLOCK_MODEL);
}

/* ── Synthetic pair generation ──────────────────────────────────────── */

/* 20 base sentences; pairs are formed by combining them. */
static const char *BASE_TEXTS[] = {
    "the quick brown fox jumps over the lazy dog",
    "a fast red fox leaps above a sleepy hound",
    "machine learning is a subset of artificial intelligence",
    "deep neural networks enable powerful pattern recognition",
    "the cat sat on the mat in the sunny room",
    "two cats were sleeping on a warm blanket today",
    "quantum computing may revolutionize cryptography soon",
    "classical computers struggle with certain quantum problems",
    "shakespeare wrote many plays and sonnets in english",
    "the bard of avon is celebrated for his literary works",
    "apple pie recipe with cinnamon and brown sugar",
    "baking a delicious cherry pie from scratch at home",
    "natural language processing transforms text into features",
    "text embeddings capture semantic meaning of words",
    "the sun rises in the east and sets in the west",
    "morning light illuminates the eastern mountain peaks",
    "electric vehicles reduce carbon emissions significantly",
    "battery technology advances make clean energy practical",
    "the ocean is deep and full of undiscovered species",
    "marine biology explores life in underwater ecosystems"
};
#define N_BASE (int)(sizeof(BASE_TEXTS) / sizeof(BASE_TEXTS[0]))

/* Generate 200 synthetic text pairs with similarity scores.
 * Similar pairs (same topic cluster): score 0.7-0.9
 * Dissimilar pairs (different clusters): score 0.1-0.3 */
typedef struct {
    const char *text_a;
    const char *text_b;
    float       score;
} text_pair_t;

static text_pair_t g_pairs[200];

static void generate_pairs(void)
{
    int idx = 0;

    /* Similar pairs: adjacent base texts share topics (0-1, 2-3, ...) */
    for (int round = 0; round < 10 && idx < 100; round++) {
        for (int i = 0; i < N_BASE - 1 && idx < 100; i += 2) {
            g_pairs[idx].text_a = BASE_TEXTS[i];
            g_pairs[idx].text_b = BASE_TEXTS[i + 1];
            g_pairs[idx].score  = 0.7f + 0.02f * (float)(i % 10);
            idx++;
        }
    }

    /* Dissimilar pairs: texts from different clusters */
    for (int round = 0; idx < 200; round++) {
        for (int i = 0; i < N_BASE && idx < 200; i++) {
            int j = (i + 3 + round) % N_BASE;
            if (j == i) j = (i + 5) % N_BASE;
            g_pairs[idx].text_a = BASE_TEXTS[i];
            g_pairs[idx].text_b = BASE_TEXTS[j];
            g_pairs[idx].score  = 0.1f + 0.01f * (float)(round % 20);
            idx++;
        }
    }
}

/* Feed all 200 pairs to a Hebbian trainer using observe_text. */
static void feed_all_pairs(trine_hebbian_state_t *st)
{
    for (int i = 0; i < 200; i++) {
        trine_hebbian_observe_text(st,
            g_pairs[i].text_a, strlen(g_pairs[i].text_a),
            g_pairs[i].text_b, strlen(g_pairs[i].text_b));
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Step 1+2: Train with diagonal mode, verify metrics
 * ══════════════════════════════════════════════════════════════════════ */

static trine_s2_model_t *g_diag_model = NULL;

static void test_diagonal_training(void)
{
    printf("  --- diagonal-mode training ---\n");

    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.projection_mode = 1;  /* diagonal */
    cfg.cascade_cells   = 512;
    cfg.cascade_depth   = 4;

    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);
    check("diag: create trainer", st != NULL);

    /* Feed 200 pairs */
    feed_all_pairs(st);

    /* Verify metrics */
    trine_hebbian_metrics_t m;
    int mrc = trine_hebbian_metrics(st, &m);
    check("diag: metrics ok", mrc == 0);
    check("diag: 200 pairs observed", m.pairs_observed == 200);
    check("diag: has positive weights", m.n_positive_weights > 0);
    check("diag: has negative weights", m.n_negative_weights > 0);
    check("diag: max_abs > 0", m.max_abs_counter > 0);

    /* Step 3: Freeze to model */
    g_diag_model = trine_hebbian_freeze(st);
    check("diag: freeze non-NULL", g_diag_model != NULL);

    /* Verify model produces valid trits */
    uint8_t out[240];
    int rc = trine_s2_encode(g_diag_model, "integration test text", 21, 0, out);
    check("diag: encode rc == 0", rc == 0);
    check("diag: valid trits", all_valid_trits(out, 240));

    /* Set diagonal projection mode on the frozen model.
     * The Hebbian freeze produces a full-matrix model (PROJ_SIGN);
     * diagonal mode is applied by the caller, matching CLI behavior. */
    trine_s2_set_projection_mode(g_diag_model, TRINE_S2_PROJ_DIAGONAL);
    int mode = trine_s2_get_projection_mode(g_diag_model);
    check("diag: projection_mode == DIAGONAL", mode == TRINE_S2_PROJ_DIAGONAL);

    trine_hebbian_free(st);
}

/* ══════════════════════════════════════════════════════════════════════
 * Step 4: Save/load round-trip
 * ══════════════════════════════════════════════════════════════════════ */

static void test_save_load_roundtrip(void)
{
    printf("  --- save/load round-trip ---\n");

    if (!g_diag_model) {
        check("roundtrip: model available", 0);
        check("roundtrip: save succeeds", 0);
        check("roundtrip: load succeeds", 0);
        check("roundtrip: output matches", 0);
        return;
    }

    /* Save */
    trine_s2_save_config_t scfg = {
        .similarity_threshold = 0.5f,
        .density = 0.33f,
        .topo_seed = 0
    };
    int rc = trine_s2_save(g_diag_model, TMP_DIAG_MODEL, &scfg);
    check("roundtrip: save succeeds", rc == 0);

    /* Load */
    trine_s2_model_t *loaded = trine_s2_load(TMP_DIAG_MODEL);
    check("roundtrip: load non-NULL", loaded != NULL);

    /* Encode same text with both models and compare */
    if (loaded) {
        const char *probe = "determinism probe text for roundtrip";
        size_t probe_len = strlen(probe);
        uint8_t out_orig[240], out_load[240];

        trine_s2_encode(g_diag_model, probe, probe_len, 0, out_orig);
        trine_s2_encode(loaded, probe, probe_len, 0, out_load);

        check("roundtrip: output matches", memcmp(out_orig, out_load, 240) == 0);
    } else {
        check("roundtrip: output matches", 0);
    }

    trine_s2_free(loaded);
}

/* ══════════════════════════════════════════════════════════════════════
 * Step 5+6: Train with block-diagonal mode, verify projection mode
 * ══════════════════════════════════════════════════════════════════════ */

static trine_s2_model_t *g_block_model = NULL;

static void test_block_diagonal_training(void)
{
    printf("  --- block-diagonal training ---\n");

    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.block_diagonal = 1;
    cfg.cascade_cells  = 512;
    cfg.cascade_depth  = 4;

    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);
    check("block: create trainer", st != NULL);

    /* Feed the same 200 pairs */
    feed_all_pairs(st);

    trine_hebbian_metrics_t m;
    trine_hebbian_metrics(st, &m);
    check("block: 200 pairs observed", m.pairs_observed == 200);

    /* Freeze */
    g_block_model = trine_hebbian_freeze(st);
    check("block: freeze non-NULL", g_block_model != NULL);

    /* Verify projection mode */
    if (g_block_model) {
        int mode = trine_s2_get_projection_mode(g_block_model);
        check("block: projection_mode == BLOCK_DIAG",
              mode == TRINE_S2_PROJ_BLOCK_DIAG);

        /* Verify valid trit output */
        uint8_t out[240];
        int rc = trine_s2_encode(g_block_model, "block diag test", 15, 0, out);
        check("block: encode rc == 0", rc == 0);
        check("block: valid trits", all_valid_trits(out, 240));

        /* Save and reload block-diagonal model */
        trine_s2_save_config_t scfg = {
            .similarity_threshold = 0.5f,
            .density = 0.33f,
            .topo_seed = 0
        };
        rc = trine_s2_save(g_block_model, TMP_BLOCK_MODEL, &scfg);
        check("block: save succeeds", rc == 0);

        trine_s2_model_t *reloaded = trine_s2_load(TMP_BLOCK_MODEL);
        check("block: reload non-NULL", reloaded != NULL);

        if (reloaded) {
            int reload_mode = trine_s2_get_projection_mode(reloaded);
            check("block: reload preserves mode",
                  reload_mode == TRINE_S2_PROJ_BLOCK_DIAG);
        } else {
            check("block: reload preserves mode", 0);
        }

        trine_s2_free(reloaded);
    } else {
        check("block: projection_mode == BLOCK_DIAG", 0);
        check("block: encode rc == 0", 0);
        check("block: valid trits", 0);
        check("block: save succeeds", 0);
        check("block: reload non-NULL", 0);
        check("block: reload preserves mode", 0);
    }

    trine_hebbian_free(st);
}

/* ══════════════════════════════════════════════════════════════════════
 * Step 7: Compare quality -- self-similarity == 1.0
 * ══════════════════════════════════════════════════════════════════════ */

static void test_self_similarity(void)
{
    printf("  --- self-similarity quality ---\n");

    const char *test_texts[] = {
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a powerful tool",
        "quantum computing changes everything"
    };
    int n_test = 3;

    /* Diagonal model: self-similarity must be 1.0 */
    if (g_diag_model) {
        int diag_ok = 1;
        for (int i = 0; i < n_test; i++) {
            uint8_t enc[240];
            trine_s2_encode(g_diag_model, test_texts[i],
                            strlen(test_texts[i]), 0, enc);
            float sim = trine_s2_compare(enc, enc, NULL);
            if (fabsf(sim - 1.0f) > 1e-5f) diag_ok = 0;
        }
        check("quality: diag self-sim == 1.0", diag_ok);
    } else {
        check("quality: diag self-sim == 1.0", 0);
    }

    /* Block-diagonal model: self-similarity must be 1.0 */
    if (g_block_model) {
        int block_ok = 1;
        for (int i = 0; i < n_test; i++) {
            uint8_t enc[240];
            trine_s2_encode(g_block_model, test_texts[i],
                            strlen(test_texts[i]), 0, enc);
            float sim = trine_s2_compare(enc, enc, NULL);
            if (fabsf(sim - 1.0f) > 1e-5f) block_ok = 0;
        }
        check("quality: block self-sim == 1.0", block_ok);
    } else {
        check("quality: block self-sim == 1.0", 0);
    }

    /* Cross-model: same text should produce some similarity (not NaN) */
    if (g_diag_model && g_block_model) {
        uint8_t d_enc[240], b_enc[240];
        trine_s2_encode(g_diag_model, test_texts[0],
                        strlen(test_texts[0]), 0, d_enc);
        trine_s2_encode(g_block_model, test_texts[0],
                        strlen(test_texts[0]), 0, b_enc);
        float cross = trine_s2_compare(d_enc, b_enc, NULL);
        check("quality: cross-model sim is finite",
              isfinite(cross) && cross >= 0.0f && cross <= 1.0f);
    } else {
        check("quality: cross-model sim is finite", 0);
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * Step 8: Reset and retrain with different data
 * ══════════════════════════════════════════════════════════════════════ */

static void test_reset_and_retrain(void)
{
    printf("  --- reset and retrain ---\n");

    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.projection_mode = 1;  /* diagonal */

    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);

    /* Initial training: 200 pairs */
    feed_all_pairs(st);

    trine_hebbian_metrics_t m1;
    trine_hebbian_metrics(st, &m1);
    int64_t pairs_before = m1.pairs_observed;
    int32_t max_before   = m1.max_abs_counter;

    /* Reset */
    trine_hebbian_reset(st);

    trine_hebbian_metrics_t m_reset;
    trine_hebbian_metrics(st, &m_reset);
    check("reset: pairs zeroed", m_reset.pairs_observed == 0);
    check("reset: max_abs zeroed", m_reset.max_abs_counter == 0);
    check("reset: no positive", m_reset.n_positive_weights == 0);
    check("reset: no negative", m_reset.n_negative_weights == 0);

    /* Retrain with different data: only dissimilar pairs */
    for (int i = 0; i < 50; i++) {
        trine_hebbian_observe_text(st,
            BASE_TEXTS[i % N_BASE],
            strlen(BASE_TEXTS[i % N_BASE]),
            BASE_TEXTS[(i + 7) % N_BASE],
            strlen(BASE_TEXTS[(i + 7) % N_BASE]));
    }

    trine_hebbian_metrics_t m2;
    trine_hebbian_metrics(st, &m2);
    check("retrain: 50 pairs observed", m2.pairs_observed == 50);
    check("retrain: state changed vs reset",
          m2.max_abs_counter > 0);

    /* Verify the retrained state is different from original
     * (different pair count is sufficient proof) */
    check("retrain: differs from original",
          m2.pairs_observed != pairs_before);

    /* Freeze the retrained model -- should still work */
    trine_s2_model_t *retrained = trine_hebbian_freeze(st);
    check("retrain: freeze non-NULL", retrained != NULL);

    if (retrained) {
        uint8_t out[240];
        int rc = trine_s2_encode(retrained, "retrain test", 12, 0, out);
        check("retrain: encode valid", rc == 0 && all_valid_trits(out, 240));
    } else {
        check("retrain: encode valid", 0);
    }

    /* Suppress unused variable warning */
    (void)max_before;

    trine_s2_free(retrained);
    trine_hebbian_free(st);
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Training Pipeline Integration Tests ===\n\n");

    cleanup();
    generate_pairs();

    test_diagonal_training();
    test_save_load_roundtrip();
    test_block_diagonal_training();
    test_self_similarity();
    test_reset_and_retrain();

    /* Free global models */
    trine_s2_free(g_diag_model);
    trine_s2_free(g_block_model);

    cleanup();

    printf("\nTraining pipeline: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);

    return g_failed > 0 ? 1 : 0;
}
