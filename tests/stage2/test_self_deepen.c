/* =====================================================================
 * TRINE Stage-2 — Self-Supervised Deepening Tests (~10 tests)
 * =====================================================================
 *
 * Tests for recursive partial-order bootstrapping:
 *   1. Basic deepening (1 round) produces a valid model
 *   2. Multi-round deepening produces a valid model
 *   3. Deepened model produces valid 240-trit output
 *   4. Deepened model self-similarity = 1.0
 *   5. Deepened model comparison is in [0, 1]
 *   6. Determinism: same data + same rounds = same output
 *   7. Deepening changes the model (not identical to round-0 freeze)
 *   8. Null safety
 *   9. Zero rounds returns NULL
 *  10. Missing file returns NULL
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
        printf("  FAIL  self_deepen: %s\n", name);
    }
}

static int all_valid_trits(const uint8_t *v, int n)
{
    for (int i = 0; i < n; i++)
        if (v[i] > 2) return 0;
    return 1;
}

static const char *TEST_JSONL = "/tmp/trine_test_self_deepen.jsonl";

static void write_test_data(void)
{
    FILE *f = fopen(TEST_JSONL, "w");
    fprintf(f, "{\"text_a\": \"the cat sat on the mat\", \"text_b\": \"the cat is sitting on a mat\", \"score\": 0.8}\n");
    fprintf(f, "{\"text_a\": \"hello world\", \"text_b\": \"goodbye universe\", \"score\": 0.1}\n");
    fprintf(f, "{\"text_a\": \"machine learning is great\", \"text_b\": \"deep learning is wonderful\", \"score\": 0.7}\n");
    fprintf(f, "{\"text_a\": \"the dog runs fast\", \"text_b\": \"a quick brown fox\", \"score\": 0.3}\n");
    fprintf(f, "{\"text_a\": \"apple pie recipe\", \"text_b\": \"apple pie ingredients\", \"score\": 0.9}\n");
    fprintf(f, "{\"text_a\": \"red blue green\", \"text_b\": \"red blue yellow\", \"score\": 0.6}\n");
    fprintf(f, "{\"text_a\": \"programming in c\", \"text_b\": \"coding in python\", \"score\": 0.4}\n");
    fprintf(f, "{\"text_a\": \"sunny weather today\", \"text_b\": \"rainy weather yesterday\", \"score\": 0.5}\n");
    fclose(f);
}

/* ── Helper: do initial training so state has signal ──────────────────── */
static trine_hebbian_state_t *train_initial(void)
{
    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.cascade_cells = 64;  /* Small for fast tests */
    cfg.cascade_depth = 2;
    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);
    if (!st) return NULL;
    trine_hebbian_train_file(st, TEST_JSONL, 3);
    return st;
}

/* ── Test 1: Basic deepening (1 round) ───────────────────────────────── */
static void test_deepen_one_round(void)
{
    trine_hebbian_state_t *st = train_initial();
    check("deepen1_state", st != NULL);

    struct trine_s2_model *model = trine_self_deepen(st, TEST_JSONL, 1);
    check("deepen1_model_not_null", model != NULL);

    if (model) trine_s2_free(model);
    trine_hebbian_free(st);
}

/* ── Test 2: Multi-round deepening ───────────────────────────────────── */
static void test_deepen_multi_round(void)
{
    trine_hebbian_state_t *st = train_initial();

    struct trine_s2_model *model = trine_self_deepen(st, TEST_JSONL, 3);
    check("deepen3_model_not_null", model != NULL);

    if (model) trine_s2_free(model);
    trine_hebbian_free(st);
}

/* ── Test 3: Deepened model produces valid trits ─────────────────────── */
static void test_deepen_valid_trits(void)
{
    trine_hebbian_state_t *st = train_initial();
    struct trine_s2_model *model = trine_self_deepen(st, TEST_JSONL, 2);

    uint8_t out[240];
    int rc = trine_s2_encode(model, "test valid trits", 16, 2, out);
    check("deepen_encode_rc", rc == 0);
    check("deepen_valid_trits", all_valid_trits(out, 240));

    trine_s2_free(model);
    trine_hebbian_free(st);
}

/* ── Test 4: Self-similarity = 1.0 ──────────────────────────────────── */
static void test_deepen_self_similarity(void)
{
    trine_hebbian_state_t *st = train_initial();
    struct trine_s2_model *model = trine_self_deepen(st, TEST_JSONL, 1);

    uint8_t emb[240];
    trine_s2_encode(model, "self similarity test", 20, 2, emb);
    float sim = trine_s2_compare(emb, emb, NULL);
    check("deepen_self_sim", fabsf(sim - 1.0f) < 1e-6f);

    trine_s2_free(model);
    trine_hebbian_free(st);
}

/* ── Test 5: Comparison in [0, 1] ───────────────────────────────────── */
static void test_deepen_compare_range(void)
{
    trine_hebbian_state_t *st = train_initial();
    struct trine_s2_model *model = trine_self_deepen(st, TEST_JSONL, 1);

    uint8_t ea[240], eb[240];
    trine_s2_encode(model, "hello world", 11, 2, ea);
    trine_s2_encode(model, "goodbye moon", 12, 2, eb);

    float sim = trine_s2_compare(ea, eb, NULL);
    check("deepen_sim_in_range", sim >= 0.0f && sim <= 1.0f);

    trine_s2_free(model);
    trine_hebbian_free(st);
}

/* ── Test 6: Determinism ────────────────────────────────────────────── */
static void test_deepen_determinism(void)
{
    uint8_t out1[240], out2[240];

    for (int trial = 0; trial < 2; trial++) {
        trine_hebbian_state_t *st = train_initial();
        struct trine_s2_model *model = trine_self_deepen(st, TEST_JSONL, 2);

        trine_s2_encode(model, "determinism check", 17, 2,
                        trial == 0 ? out1 : out2);

        trine_s2_free(model);
        trine_hebbian_free(st);
    }

    check("deepen_determinism", memcmp(out1, out2, 240) == 0);
}

/* ── Test 7: Deepening changes the model ─────────────────────────────── */
static void test_deepen_changes_model(void)
{
    /* Round-0 freeze (no deepening) */
    trine_hebbian_state_t *st0 = train_initial();
    struct trine_s2_model *model0 = trine_hebbian_freeze(st0);

    uint8_t out0[240];
    trine_s2_encode(model0, "change check", 12, 2, out0);
    trine_s2_free(model0);
    trine_hebbian_free(st0);

    /* Round-2 deepened */
    trine_hebbian_state_t *st2 = train_initial();
    struct trine_s2_model *model2 = trine_self_deepen(st2, TEST_JSONL, 2);

    uint8_t out2[240];
    trine_s2_encode(model2, "change check", 12, 2, out2);
    trine_s2_free(model2);
    trine_hebbian_free(st2);

    /* The deepened model should produce different embeddings */
    check("deepen_changes_output", memcmp(out0, out2, 240) != 0);
}

/* ── Test 8: Null safety ─────────────────────────────────────────────── */
static void test_deepen_null_safety(void)
{
    /* NULL state */
    struct trine_s2_model *m1 = trine_self_deepen(NULL, TEST_JSONL, 1);
    check("deepen_null_state", m1 == NULL);

    /* NULL path */
    trine_hebbian_state_t *st = train_initial();
    struct trine_s2_model *m2 = trine_self_deepen(st, NULL, 1);
    check("deepen_null_path", m2 == NULL);

    trine_hebbian_free(st);
}

/* ── Test 9: Zero rounds returns NULL ────────────────────────────────── */
static void test_deepen_zero_rounds(void)
{
    trine_hebbian_state_t *st = train_initial();
    struct trine_s2_model *m = trine_self_deepen(st, TEST_JSONL, 0);
    check("deepen_zero_rounds_null", m == NULL);
    trine_hebbian_free(st);
}

/* ── Test 10: Missing file returns NULL ──────────────────────────────── */
static void test_deepen_missing_file(void)
{
    trine_hebbian_state_t *st = train_initial();
    struct trine_s2_model *m = trine_self_deepen(st, "/tmp/nonexistent_file_trine.jsonl", 1);
    check("deepen_missing_file_null", m == NULL);
    trine_hebbian_free(st);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Self-Deepen Tests ===\n");

    write_test_data();

    test_deepen_one_round();
    test_deepen_multi_round();
    test_deepen_valid_trits();
    test_deepen_self_similarity();
    test_deepen_compare_range();
    test_deepen_determinism();
    test_deepen_changes_model();
    test_deepen_null_safety();
    test_deepen_zero_rounds();
    test_deepen_missing_file();

    remove(TEST_JSONL);

    printf("\nSelf-deepen: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
