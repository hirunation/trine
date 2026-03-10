/* =====================================================================
 * TRINE Stage-2 -- Sparse Cross-Channel Projection Tests (~20 tests)
 * =====================================================================
 *
 * Tests for sparse sign-based projection and sparse freeze:
 *
 *  Sparse Projection Function Tests:
 *   1. test_sparse_sign_zero_skip
 *   2. test_sparse_sign_basic
 *   3. test_sparse_majority_basic
 *   4. test_sparse_sign_all_nonzero
 *   5. test_sparse_sign_identity_diagonal
 *
 *  Sparse Freeze Tests:
 *   6. test_freeze_sparse_top1
 *   7. test_freeze_sparse_top8
 *   8. test_freeze_sparse_selects_strongest
 *   9. test_freeze_sparse_sign_correct
 *  10. test_freeze_sparse_zero_counter
 *  11. test_freeze_sparse_k_clamp
 *
 *  End-to-End Sparse Pipeline Tests:
 *  12. test_sparse_pipeline_creates_model
 *  13. test_sparse_pipeline_encode
 *  14. test_sparse_pipeline_compare
 *  15. test_sparse_mode_set
 *  16. test_sparse_vs_diagonal
 *
 *  Null Safety Tests:
 *  17. test_sparse_sign_null_safety
 *  18. test_freeze_sparse_null_safety
 *  19. test_sparse_config_defaults
 *
 *  Statistical Tests:
 *  20. test_freeze_sparse_density
 *
 * ===================================================================== */

#include "trine_project.h"
/* trine_freeze.h cannot be included alongside trine_project.h due to
 * conflicting forward-declared trine_projection_t typedefs.  Since
 * trine_project.h provides the canonical trine_projection_t, we
 * forward-declare the freeze functions we need here. */
#include "trine_accumulator.h"

/* Freeze API (from trine_freeze.h) */
void trine_freeze_projection(const trine_accumulator_t *acc,
                              int32_t threshold,
                              trine_projection_t *out);
void trine_freeze_sparse(const trine_accumulator_t *acc,
                           uint32_t top_k,
                           trine_projection_t *out);
typedef struct {
    uint32_t n_zero;
    uint32_t n_one;
    uint32_t n_two;
    float    density;
} trine_freeze_stats_t;
void trine_freeze_stats(const trine_projection_t *proj,
                         trine_freeze_stats_t *stats);

#include "trine_stage2.h"
#include "trine_hebbian.h"
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
        printf("  FAIL  sparse: %s\n", name);
    }
}

#define DIM TRINE_PROJECT_DIM  /* 240 */
#define K   TRINE_PROJECT_K    /* 3   */
#define TOTAL_WEIGHTS (K * DIM * DIM)

/* =====================================================================
 * Sparse Projection Function Tests
 * ===================================================================== */

/* -- Test 1: W=0 entries are skipped (not treated as -1) -------------- */
static void test_sparse_sign_zero_skip(void)
{
    /* Build a weight matrix that is all zeros except one entry.
     * In sign projection, W=0 centers to -1 and contributes.
     * In sparse sign projection, W=0 is skipped entirely. */
    uint8_t W_sparse[DIM][DIM];
    uint8_t W_sign[DIM][DIM];
    memset(W_sparse, 0, sizeof(W_sparse));
    memset(W_sign, 0, sizeof(W_sign));

    /* Set one nonzero entry: W[0][0] = 2 (positive) */
    W_sparse[0][0] = 2;
    W_sign[0][0]   = 2;

    /* Input: all trit=2 (centered = +1) */
    uint8_t x[DIM];
    memset(x, 2, sizeof(x));

    uint8_t y_sparse[DIM], y_sign[DIM];
    trine_project_single_sparse_sign(W_sparse, x, y_sparse);
    trine_project_single_sign(W_sign, x, y_sign);

    /* For row 0:
     * sparse: acc = 1 * (+1) = +1  ->  y=2
     * sign:   acc = (+1)(+1) + 239*(-1)(+1) = 1 - 239 = -238  ->  y=0
     * They must differ because sign treats W=0 as -1. */
    check("zero_skip_row0_sparse", y_sparse[0] == 2);
    check("zero_skip_row0_sign",   y_sign[0] == 0);
    check("zero_skip_differ",      y_sparse[0] != y_sign[0]);

    /* For rows with all-zero W (row 1..239):
     * sparse: acc = 0 (nothing to sum)  ->  y=1 (neutral)
     * sign:   acc = sum of -1 * (+1) = -240  ->  y=0 */
    check("zero_skip_empty_row_sparse", y_sparse[1] == 1);
    check("zero_skip_empty_row_sign",   y_sign[1] == 0);
}

/* -- Test 2: Basic sparse projection with few nonzero weights --------- */
static void test_sparse_sign_basic(void)
{
    uint8_t W[DIM][DIM];
    memset(W, 0, sizeof(W));

    /* Set a few entries in row 0:
     * W[0][0] = 2 (+1), W[0][1] = 1 (-1), W[0][2] = 2 (+1) */
    W[0][0] = 2;
    W[0][1] = 1;
    W[0][2] = 2;

    /* Input: x[0]=2 (+1), x[1]=2 (+1), x[2]=0 (-1) */
    uint8_t x[DIM];
    memset(x, 1, sizeof(x));  /* neutral */
    x[0] = 2;  /* centered = +1 */
    x[1] = 2;  /* centered = +1 */
    x[2] = 0;  /* centered = -1 */

    uint8_t y[DIM];
    trine_project_single_sparse_sign(W, x, y);

    /* Row 0: acc = (+1)(+1) + (-1)(+1) + (+1)(-1)
     *       = 1 - 1 - 1 = -1  ->  y=0 */
    check("sparse_basic_row0", y[0] == 0);

    /* Row 1 (all zeros): acc = 0  ->  y=1 (neutral) */
    check("sparse_basic_empty_row", y[1] == 1);

    /* All output trits must be in {0, 1, 2} */
    int all_valid = 1;
    for (int i = 0; i < DIM; i++)
        if (y[i] > 2) all_valid = 0;
    check("sparse_basic_z3", all_valid);
}

/* -- Test 3: K=3 majority vote with sparse sign projection ----------- */
static void test_sparse_majority_basic(void)
{
    trine_projection_t proj;
    memset(&proj, 0, sizeof(proj));

    /* Set up three copies with different sparse patterns for row 0.
     * Copy 0: W[0][0]=2  -> with x[0]=2: acc=+1 -> y=2
     * Copy 1: W[0][0]=2  -> with x[0]=2: acc=+1 -> y=2
     * Copy 2: W[0][0]=1  -> with x[0]=2: acc=-1 -> y=0
     * Majority of (2, 2, 0) = 2 */
    proj.W[0][0][0] = 2;
    proj.W[1][0][0] = 2;
    proj.W[2][0][0] = 1;

    uint8_t x[DIM];
    memset(x, 1, sizeof(x));
    x[0] = 2;

    uint8_t y[DIM];
    trine_project_majority_sparse_sign(&proj, x, y);

    /* Row 0: majority of (2, 2, 0) = 2 */
    check("sparse_majority_row0", y[0] == 2);

    /* Row 1 (all copies zero): all produce y=1, majority = 1 */
    check("sparse_majority_empty_row", y[1] == 1);

    /* All valid trits */
    int all_valid = 1;
    for (int i = 0; i < DIM; i++)
        if (y[i] > 2) all_valid = 0;
    check("sparse_majority_z3", all_valid);
}

/* -- Test 4: When all W entries are nonzero, sparse sign ~ sign ------- */
static void test_sparse_sign_all_nonzero(void)
{
    /* Fill W with all 2 (all +1 in centered space).
     * In sign projection: w_c = 2-1 = +1 for all j.
     * In sparse sign: w_c = +1 for all j (since W=2 -> +1).
     * Both should compute the same dot product. */
    uint8_t W[DIM][DIM];
    memset(W, 2, sizeof(W));

    uint8_t x[DIM];
    trine_encode_shingle("sparse vs sign test", 19, x);

    uint8_t y_sparse[DIM], y_sign[DIM];
    trine_project_single_sparse_sign(W, x, y_sparse);
    trine_project_single_sign(W, x, y_sign);

    /* When all W entries are 2, both projections interpret them identically:
     * sign: w_c = 2-1 = +1, sparse: W=2 -> +1.
     * Both compute the same accumulator, so outputs must match. */
    check("all_nonzero_match", memcmp(y_sparse, y_sign, DIM) == 0);
}

/* -- Test 5: Sparse with only diagonal nonzero approximates diagonal -- */
static void test_sparse_sign_identity_diagonal(void)
{
    trine_projection_t proj;
    memset(&proj, 0, sizeof(proj));

    /* Set diagonal to 2 (keep) for all K copies.
     * With only diagonal nonzero, sparse sign on each row i computes:
     *   acc = (+1) * (x[i]-1) = x[i]-1
     *   y[i] = (x[i]-1 > 0) ? 2 : (x[i]-1 < 0) ? 0 : 1
     * So: x=2 -> y=2, x=0 -> y=0, x=1 -> y=1
     * This is the identity mapping. */
    for (uint32_t k = 0; k < K; k++)
        for (int i = 0; i < DIM; i++)
            proj.W[k][i][i] = 2;

    uint8_t x[DIM];
    trine_encode_shingle("diagonal identity test", 22, x);

    uint8_t y[DIM];
    trine_project_majority_sparse_sign(&proj, x, y);

    /* Output should match input (identity-like behavior) */
    check("diagonal_sparse_identity", memcmp(x, y, DIM) == 0);
}

/* =====================================================================
 * Sparse Freeze Tests
 * ===================================================================== */

/* -- Test 6: top_k=1 gives exactly 1 nonzero per row ----------------- */
static void test_freeze_sparse_top1(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    /* Feed diverse pairs to populate many counters */
    const char *texts[] = {
        "alpha bravo", "charlie delta", "echo foxtrot",
        "golf hotel", "india juliet"
    };
    for (int i = 0; i < 4; i++) {
        uint8_t a[DIM], b[DIM];
        trine_encode_shingle(texts[i], strlen(texts[i]), a);
        trine_encode_shingle(texts[i + 1], strlen(texts[i + 1]), b);
        for (int r = 0; r < 10; r++)
            trine_accumulator_update(acc, a, b, 1);
    }

    trine_projection_t proj;
    trine_freeze_sparse(acc, 1, &proj);

    /* Each row in each copy should have at most 1 nonzero entry */
    int ok = 1;
    for (uint32_t k = 0; k < K && ok; k++) {
        for (int i = 0; i < DIM && ok; i++) {
            int count = 0;
            for (int j = 0; j < DIM; j++)
                if (proj.W[k][i][j] != 0) count++;
            if (count > 1) ok = 0;
        }
    }
    check("top1_at_most_one", ok);

    /* There should be some nonzero entries (not all-zero) */
    trine_freeze_stats_t stats;
    trine_freeze_stats(&proj, &stats);
    check("top1_has_nonzero", stats.n_one + stats.n_two > 0);

    trine_accumulator_free(acc);
}

/* -- Test 7: top_k=8 gives at most 8 nonzero per row ----------------- */
static void test_freeze_sparse_top8(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    const char *texts[] = {
        "the quick brown fox jumps over the lazy dog",
        "a fast brown fox leaps over a lazy hound",
        "machine learning algorithms process data",
        "deep neural networks analyze information"
    };
    for (int i = 0; i < 3; i++) {
        uint8_t a[DIM], b[DIM];
        trine_encode_shingle(texts[i], strlen(texts[i]), a);
        trine_encode_shingle(texts[i + 1], strlen(texts[i + 1]), b);
        for (int r = 0; r < 20; r++)
            trine_accumulator_update(acc, a, b, (i % 2 == 0) ? 1 : -1);
    }

    trine_projection_t proj;
    trine_freeze_sparse(acc, 8, &proj);

    int ok = 1;
    for (uint32_t k = 0; k < K && ok; k++) {
        for (int i = 0; i < DIM && ok; i++) {
            int count = 0;
            for (int j = 0; j < DIM; j++)
                if (proj.W[k][i][j] != 0) count++;
            if (count > 8) ok = 0;
        }
    }
    check("top8_at_most_eight", ok);

    trine_accumulator_free(acc);
}

/* -- Test 8: Kept entries have the largest absolute counters ----------- */
static void test_freeze_sparse_selects_strongest(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    /* Create a controlled scenario: make specific counter entries large.
     * We manually update with vectors that produce strong signal at
     * known positions. */
    uint8_t a[DIM], b[DIM];
    memset(a, 1, sizeof(a));
    memset(b, 1, sizeof(b));

    /* Position 0 and 5 will get large positive signal.
     * Trit 2 centered is +1. */
    a[0] = 2; b[0] = 2;
    a[5] = 2; b[5] = 2;

    /* Many updates to build strong signal at these positions */
    for (int r = 0; r < 50; r++)
        trine_accumulator_update(acc, a, b, 1);

    /* Also add weak signal at other positions */
    uint8_t c[DIM], d[DIM];
    trine_encode_shingle("weak signal text", 16, c);
    trine_encode_shingle("another weak text", 17, d);
    trine_accumulator_update(acc, c, d, 1);

    /* Freeze with top_k=2 */
    trine_projection_t proj;
    trine_freeze_sparse(acc, 2, &proj);

    /* Row 0 should have its strongest columns selected.
     * Since a[0]=2, b[0]=2, the outer product at [0][0] gets
     * centered_a[0]*centered_b[0] = (+1)(+1) = +1 per update.
     * After 50 updates: counter[k][0][0] = +50 for each k. */
    int row0_has_col0 = 0;
    for (uint32_t k = 0; k < K; k++)
        if (proj.W[k][0][0] != 0) row0_has_col0 = 1;

    check("strongest_includes_strong", row0_has_col0);

    trine_accumulator_free(acc);
}

/* -- Test 9: Positive counters -> W=2, negative counters -> W=1 ------- */
static void test_freeze_sparse_sign_correct(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    /* Create vectors that produce known counter signs.
     * a[0]=2, b[0]=2: centered outer product at [0][0] = (+1)(+1) = +1
     * With sign=+1 update: counter[k][0][0] += 1 (positive) */
    uint8_t a[DIM], b[DIM];
    memset(a, 1, sizeof(a));
    memset(b, 1, sizeof(b));
    a[0] = 2; b[0] = 2;

    for (int r = 0; r < 20; r++)
        trine_accumulator_update(acc, a, b, 1);

    trine_projection_t proj;
    trine_freeze_sparse(acc, 1, &proj);

    /* Counter at [k][0][0] should be positive -> W=2 */
    int pos_ok = 1;
    for (uint32_t k = 0; k < K; k++)
        if (proj.W[k][0][0] != 2) pos_ok = 0;
    check("sparse_sign_positive_to_2", pos_ok);

    /* Now do negative updates */
    trine_accumulator_reset(acc);
    for (int r = 0; r < 20; r++)
        trine_accumulator_update(acc, a, b, -1);

    trine_freeze_sparse(acc, 1, &proj);

    /* Counter at [k][0][0] should be negative -> W=1 */
    int neg_ok = 1;
    for (uint32_t k = 0; k < K; k++)
        if (proj.W[k][0][0] != 1) neg_ok = 0;
    check("sparse_sign_negative_to_1", neg_ok);

    trine_accumulator_free(acc);
}

/* -- Test 10: Zero counters are never selected ------------------------ */
static void test_freeze_sparse_zero_counter(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();
    /* No updates: all counters are zero */

    trine_projection_t proj;
    trine_freeze_sparse(acc, 8, &proj);

    /* All weights should be zero (sparse freeze skips zero counters) */
    trine_freeze_stats_t stats;
    trine_freeze_stats(&proj, &stats);
    check("zero_counter_all_zero", stats.n_zero == (uint32_t)TOTAL_WEIGHTS);
    check("zero_counter_no_nonzero", stats.n_one == 0 && stats.n_two == 0);

    trine_accumulator_free(acc);
}

/* -- Test 11: top_k > 240 gets clamped to 240 ------------------------ */
static void test_freeze_sparse_k_clamp(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    uint8_t a[DIM], b[DIM];
    trine_encode_shingle("clamp test text", 15, a);
    trine_encode_shingle("clamp verify text", 17, b);
    for (int r = 0; r < 10; r++)
        trine_accumulator_update(acc, a, b, 1);

    /* top_k=999 should get clamped to 240 */
    trine_projection_t proj;
    trine_freeze_sparse(acc, 999, &proj);

    /* Result should be the same as a threshold=0 freeze (all nonzero
     * counters kept), since top_k >= DIM means keep everything. */
    trine_projection_t proj_t0;
    trine_freeze_projection(acc, 0, &proj_t0);

    /* Both should have valid weights -- no crash from unclamped K */
    trine_freeze_stats_t stats_sparse, stats_t0;
    trine_freeze_stats(&proj, &stats_sparse);
    trine_freeze_stats(&proj_t0, &stats_t0);

    /* Clamped sparse should have at most as many nonzero as threshold=0 */
    uint32_t nz_sparse = stats_sparse.n_one + stats_sparse.n_two;
    uint32_t nz_t0     = stats_t0.n_one + stats_t0.n_two;
    check("k_clamp_no_crash", nz_sparse > 0);
    check("k_clamp_bounded", nz_sparse <= nz_t0);

    trine_accumulator_free(acc);
}

/* =====================================================================
 * End-to-End Sparse Pipeline Tests
 * ===================================================================== */

/* Helper: train a sparse model from text pairs */
static trine_s2_model_t *train_sparse_model(uint32_t sparse_k)
{
    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.sparse_k = sparse_k;
    cfg.similarity_threshold = 0.5f;

    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);
    if (!st) return NULL;

    const char *texts_a[] = {
        "machine learning algorithms", "natural language processing",
        "computer vision research", "reinforcement learning agent",
        "data science methods", "neural network architecture"
    };
    const char *texts_b[] = {
        "ML algorithm design", "NLP text analysis",
        "image recognition tasks", "RL policy optimization",
        "statistical analysis methods", "deep network layers"
    };

    for (int i = 0; i < 6; i++)
        trine_hebbian_observe_text(st,
                                    texts_a[i], strlen(texts_a[i]),
                                    texts_b[i], strlen(texts_b[i]));

    /* Add dissimilar pairs */
    trine_hebbian_observe_text(st, "apple pie", 9, "quantum physics", 15);
    trine_hebbian_observe_text(st, "ocean waves", 11, "mountain climbing", 17);

    trine_s2_model_t *model = trine_hebbian_freeze(st);
    trine_hebbian_free(st);
    return model;
}

/* -- Test 12: Create hebbian with sparse_k, train, freeze, verify ----- */
static void test_sparse_pipeline_creates_model(void)
{
    trine_s2_model_t *model = train_sparse_model(8);
    check("pipeline_creates_model", model != NULL);

    if (model) {
        trine_s2_info_t info;
        int rc = trine_s2_info(model, &info);
        check("pipeline_info_ok", rc == 0);
        check("pipeline_not_identity", info.is_identity == 0);
        check("pipeline_dims", info.projection_dims == 240);
        trine_s2_free(model);
    }
}

/* -- Test 13: Encode text through sparse model, verify valid trits ---- */
static void test_sparse_pipeline_encode(void)
{
    trine_s2_model_t *model = train_sparse_model(8);
    check("pipeline_encode_model", model != NULL);

    if (model) {
        uint8_t out[DIM];
        int rc = trine_s2_encode(model, "test encoding text", 18, 0, out);
        check("pipeline_encode_ok", rc == 0);

        /* All output trits must be in {0, 1, 2} */
        int all_valid = 1;
        for (int i = 0; i < DIM; i++)
            if (out[i] > 2) all_valid = 0;
        check("pipeline_encode_z3", all_valid);

        /* Output should not be all-neutral (all 1s) */
        int all_one = 1;
        for (int i = 0; i < DIM; i++)
            if (out[i] != 1) { all_one = 0; break; }
        check("pipeline_encode_not_trivial", !all_one);

        trine_s2_free(model);
    }
}

/* -- Test 14: Compare two texts through sparse model, verify range ---- */
static void test_sparse_pipeline_compare(void)
{
    trine_s2_model_t *model = train_sparse_model(8);
    check("pipeline_compare_model", model != NULL);

    if (model) {
        uint8_t ea[DIM], eb[DIM];
        trine_s2_encode(model, "machine learning", 16, 0, ea);
        trine_s2_encode(model, "deep learning", 13, 0, eb);

        float sim = trine_s2_compare(ea, eb, NULL);
        check("pipeline_compare_range", sim >= 0.0f && sim <= 1.0f);

        /* Self-similarity should be 1.0 */
        float self_sim = trine_s2_compare(ea, ea, NULL);
        check("pipeline_compare_self", fabsf(self_sim - 1.0f) < 1e-4f);

        trine_s2_free(model);
    }
}

/* -- Test 15: Model gets TRINE_S2_PROJ_SPARSE mode after sparse freeze */
static void test_sparse_mode_set(void)
{
    trine_s2_model_t *model = train_sparse_model(8);
    check("mode_set_model", model != NULL);

    if (model) {
        int mode = trine_s2_get_projection_mode(model);
        check("mode_set_sparse", mode == TRINE_S2_PROJ_SPARSE);
        trine_s2_free(model);
    }
}

/* -- Test 16: Sparse K=8 vs diagonal produce different outputs -------- */
static void test_sparse_vs_diagonal(void)
{
    /* Train with sparse K=8 (cross-channel mixing) */
    trine_s2_model_t *model_sparse = train_sparse_model(8);
    check("vs_diag_sparse_model", model_sparse != NULL);

    /* Train with diagonal mode (per-channel only) */
    trine_hebbian_config_t cfg_diag = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_diag.projection_mode = 1;  /* diagonal */
    cfg_diag.freeze_target_density = 0.15f;
    cfg_diag.similarity_threshold = 0.5f;

    trine_hebbian_state_t *st_diag = trine_hebbian_create(&cfg_diag);

    const char *texts_a[] = {
        "machine learning algorithms", "natural language processing",
        "computer vision research", "reinforcement learning agent",
        "data science methods", "neural network architecture"
    };
    const char *texts_b[] = {
        "ML algorithm design", "NLP text analysis",
        "image recognition tasks", "RL policy optimization",
        "statistical analysis methods", "deep network layers"
    };

    for (int i = 0; i < 6; i++)
        trine_hebbian_observe_text(st_diag,
                                    texts_a[i], strlen(texts_a[i]),
                                    texts_b[i], strlen(texts_b[i]));

    trine_hebbian_observe_text(st_diag, "apple pie", 9, "quantum physics", 15);
    trine_hebbian_observe_text(st_diag, "ocean waves", 11, "mountain climbing", 17);

    trine_s2_model_t *model_diag = trine_hebbian_freeze(st_diag);
    trine_s2_set_projection_mode(model_diag, TRINE_S2_PROJ_DIAGONAL);
    check("vs_diag_diag_model", model_diag != NULL);

    if (model_sparse && model_diag) {
        uint8_t out_sparse[DIM], out_diag[DIM];
        trine_s2_encode(model_sparse, "cross channel mixing", 20, 0, out_sparse);
        trine_s2_encode(model_diag,   "cross channel mixing", 20, 0, out_diag);

        /* Sparse (cross-channel) and diagonal (per-channel) should produce
         * different embeddings since sparse mixes across channels. */
        check("vs_diag_different", memcmp(out_sparse, out_diag, DIM) != 0);
    }

    trine_s2_free(model_sparse);
    trine_s2_free(model_diag);
    trine_hebbian_free(st_diag);
}

/* =====================================================================
 * Null Safety Tests
 * ===================================================================== */

/* -- Test 17: NULL inputs to sparse sign functions -------------------- */
static void test_sparse_sign_null_safety(void)
{
    uint8_t W[DIM][DIM];
    memset(W, 0, sizeof(W));
    W[0][0] = 2;

    uint8_t x[DIM];
    memset(x, 1, sizeof(x));

    uint8_t y[DIM];

    /* These functions take array pointers, so we cannot pass NULL directly
     * without undefined behavior. Instead, verify valid inputs produce
     * valid outputs as a regression guard. */
    memset(y, 0xFF, sizeof(y));
    trine_project_single_sparse_sign(W, x, y);

    /* y should have valid values (not 0xFF leftovers) */
    int all_valid = 1;
    for (int i = 0; i < DIM; i++)
        if (y[i] > 2) all_valid = 0;
    check("sparse_sign_valid_output", all_valid);

    /* Majority with valid projection */
    trine_projection_t proj;
    memset(&proj, 0, sizeof(proj));
    proj.W[0][0][0] = 2;

    memset(y, 0xFF, sizeof(y));
    trine_project_majority_sparse_sign(&proj, x, y);

    all_valid = 1;
    for (int i = 0; i < DIM; i++)
        if (y[i] > 2) all_valid = 0;
    check("sparse_majority_valid_output", all_valid);
}

/* -- Test 18: NULL accumulator or output ------------------------------ */
static void test_freeze_sparse_null_safety(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();
    trine_projection_t proj;

    /* NULL output: should not crash */
    trine_freeze_sparse(acc, 8, NULL);
    check("freeze_sparse_null_out", 1);  /* survived */

    /* NULL accumulator: should produce all-zero output */
    memset(&proj, 0xFF, sizeof(proj));
    trine_freeze_sparse(NULL, 8, &proj);

    trine_freeze_stats_t stats;
    trine_freeze_stats(&proj, &stats);
    check("freeze_sparse_null_acc_zero", stats.n_zero == (uint32_t)TOTAL_WEIGHTS);

    /* Both NULL: should not crash */
    trine_freeze_sparse(NULL, 8, NULL);
    check("freeze_sparse_both_null", 1);  /* survived */

    trine_accumulator_free(acc);
}

/* -- Test 19: Default config has sparse_k=0 --------------------------- */
static void test_sparse_config_defaults(void)
{
    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    check("config_default_sparse_k_zero", cfg.sparse_k == 0);

    /* Creating state with NULL config should also have sparse_k=0 */
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);
    check("config_null_create", st != NULL);

    if (st) {
        trine_hebbian_config_t got = trine_hebbian_get_config(st);
        check("config_null_sparse_k_zero", got.sparse_k == 0);
        trine_hebbian_free(st);
    }
}

/* =====================================================================
 * Statistical Tests
 * ===================================================================== */

/* -- Test 20: Sparse freeze K=8 density <= 8/240 ---------------------- */
static void test_freeze_sparse_density(void)
{
    trine_accumulator_t *acc = trine_accumulator_create();

    /* Feed many diverse pairs to fill most counter positions */
    const char *texts[] = {
        "alpha bravo charlie delta echo",
        "foxtrot golf hotel india juliet",
        "kilo lima mike november oscar",
        "papa quebec romeo sierra tango",
        "uniform victor whiskey xray yankee",
        "one two three four five six seven",
        "the quick brown fox jumps over",
        "a lazy dog sleeps in the sun"
    };

    for (int i = 0; i < 7; i++) {
        uint8_t a[DIM], b[DIM];
        trine_encode_shingle(texts[i], strlen(texts[i]), a);
        trine_encode_shingle(texts[i + 1], strlen(texts[i + 1]), b);
        for (int r = 0; r < 30; r++)
            trine_accumulator_update(acc, a, b, (i % 2 == 0) ? 1 : -1);
    }

    trine_projection_t proj;
    trine_freeze_sparse(acc, 8, &proj);

    trine_freeze_stats_t stats;
    trine_freeze_stats(&proj, &stats);

    /* Max nonzero per row = 8, total rows = K * DIM = 3 * 240 = 720.
     * Max possible nonzero = 720 * 8 = 5760.
     * Max density = 5760 / (3 * 240 * 240) = 5760 / 172800 = 0.0333...
     * So density should be at most ~0.034. */
    float max_density = (float)(K * DIM * 8) / (float)TOTAL_WEIGHTS;
    check("density_bounded", stats.density <= max_density + 1e-6f);

    /* There should be some nonzero entries */
    check("density_nonzero", stats.density > 0.0f);

    trine_accumulator_free(acc);
}

/* -- Main ------------------------------------------------------------- */

int main(void)
{
    printf("=== Stage-2 Sparse Projection Tests ===\n");

    /* Sparse Projection Function Tests */
    test_sparse_sign_zero_skip();
    test_sparse_sign_basic();
    test_sparse_majority_basic();
    test_sparse_sign_all_nonzero();
    test_sparse_sign_identity_diagonal();

    /* Sparse Freeze Tests */
    test_freeze_sparse_top1();
    test_freeze_sparse_top8();
    test_freeze_sparse_selects_strongest();
    test_freeze_sparse_sign_correct();
    test_freeze_sparse_zero_counter();
    test_freeze_sparse_k_clamp();

    /* End-to-End Sparse Pipeline Tests */
    test_sparse_pipeline_creates_model();
    test_sparse_pipeline_encode();
    test_sparse_pipeline_compare();
    test_sparse_mode_set();
    test_sparse_vs_diagonal();

    /* Null Safety Tests */
    test_sparse_sign_null_safety();
    test_freeze_sparse_null_safety();
    test_sparse_config_defaults();

    /* Statistical Tests */
    test_freeze_sparse_density();

    printf("\nSparse: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
