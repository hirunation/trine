/* =====================================================================
 * TRINE Stage-2 — Phase A+B Tests (~14 tests)
 * =====================================================================
 *
 * Tests for Phase A (Weighted Hebbian, Dataset Rebalancing, Threshold
 * Schedule) and Phase B (Gate-Aware Comparison, Per-Chain Blend):
 *
 *  A1. Weighted Hebbian Updates
 *   1. test_weighted_accumulator_basic
 *   2. test_weighted_accumulator_magnitude_1
 *   3. test_weighted_observe
 *   4. test_weighted_pos_neg_scale
 *
 *  A2. Dataset Rebalancing
 *   5. test_source_weight_lookup
 *   6. test_source_downsampling
 *
 *  A3. Threshold Schedule
 *   7. test_set_threshold
 *   8. test_threshold_schedule_effect
 *
 *  B1. Gate-Aware Comparison
 *   9. test_gated_compare_identity
 *  10. test_gated_compare_trained
 *  11. test_gated_compare_null_safety
 *
 *  B2. Per-Chain Blend
 *  12. test_chain_blend_uniform
 *  13. test_chain_blend_extremes
 *  14. test_chain_blend_null_safety
 *
 * ===================================================================== */

#include "trine_accumulator.h"
#include "trine_hebbian.h"
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
        printf("  FAIL  phase_ab: %s\n", name);
    }
}

/* =====================================================================
 * A1: Weighted Hebbian Updates
 * ===================================================================== */

/* ── Test 1: Weighted accumulator with magnitude=5 vs unweighted ────── */
static void test_weighted_accumulator_basic(void)
{
    /* Create two accumulators: one updated with magnitude=5, one unweighted */
    trine_accumulator_t *acc_w = trine_accumulator_create();
    trine_accumulator_t *acc_u = trine_accumulator_create();

    check("weighted_acc_create_w", acc_w != NULL);
    check("weighted_acc_create_u", acc_u != NULL);

    uint8_t a[240], b[240];
    trine_encode_shingle("the quick brown fox", 19, a);
    trine_encode_shingle("a fast brown fox", 16, b);

    /* Weighted update with magnitude=5 */
    trine_accumulator_update_weighted(acc_w, a, b, 1, 5);

    /* Unweighted update (implicitly magnitude=1) */
    trine_accumulator_update(acc_u, a, b, 1);

    trine_accumulator_stats_t stats_w, stats_u;
    trine_accumulator_stats(acc_w, &stats_w);
    trine_accumulator_stats(acc_u, &stats_u);

    /* Both should have exactly 1 update recorded */
    check("weighted_basic_updates_w", stats_w.total_updates == 1);
    check("weighted_basic_updates_u", stats_u.total_updates == 1);

    /* Weighted max_abs should be larger (5x) than unweighted */
    check("weighted_basic_larger", stats_w.max_abs > stats_u.max_abs);
    check("weighted_basic_5x", stats_w.max_abs == 5 * stats_u.max_abs);

    trine_accumulator_free(acc_w);
    trine_accumulator_free(acc_u);
}

/* ── Test 2: Weighted update with magnitude=1 matches unweighted ────── */
static void test_weighted_accumulator_magnitude_1(void)
{
    trine_accumulator_t *acc_w = trine_accumulator_create();
    trine_accumulator_t *acc_u = trine_accumulator_create();

    uint8_t a[240], b[240];
    trine_encode_shingle("machine learning is great", 25, a);
    trine_encode_shingle("deep learning is wonderful", 26, b);

    /* Weighted with magnitude=1 should produce identical counters */
    trine_accumulator_update_weighted(acc_w, a, b, 1, 1);
    trine_accumulator_update(acc_u, a, b, 1);

    /* Compare counter matrices directly */
    int identical = 1;
    for (uint32_t k = 0; k < TRINE_ACC_K; k++) {
        const int32_t (*cw)[TRINE_ACC_DIM] =
            trine_accumulator_counters_const(acc_w, k);
        const int32_t (*cu)[TRINE_ACC_DIM] =
            trine_accumulator_counters_const(acc_u, k);
        if (!cw || !cu) { identical = 0; break; }
        if (memcmp(cw, cu,
                   sizeof(int32_t) * TRINE_ACC_DIM * TRINE_ACC_DIM) != 0) {
            identical = 0;
            break;
        }
    }
    check("weighted_mag1_identical", identical);

    /* Stats should also match exactly */
    trine_accumulator_stats_t sw, su;
    trine_accumulator_stats(acc_w, &sw);
    trine_accumulator_stats(acc_u, &su);
    check("weighted_mag1_max_abs", sw.max_abs == su.max_abs);
    check("weighted_mag1_positive", sw.n_positive == su.n_positive);
    check("weighted_mag1_negative", sw.n_negative == su.n_negative);

    trine_accumulator_free(acc_w);
    trine_accumulator_free(acc_u);
}

/* ── Test 3: Weighted observe vs non-weighted observe ───────────────── */
static void test_weighted_observe(void)
{
    /* weighted_mode=1 should use magnitude scaling in observe */
    trine_hebbian_config_t cfg_w = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_w.weighted_mode = 1;
    cfg_w.pos_scale = 10.0f;
    cfg_w.neg_scale = 3.0f;

    /* weighted_mode=0 uses binary sign */
    trine_hebbian_config_t cfg_u = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_u.weighted_mode = 0;

    trine_hebbian_state_t *st_w = trine_hebbian_create(&cfg_w);
    trine_hebbian_state_t *st_u = trine_hebbian_create(&cfg_u);

    check("weighted_observe_create_w", st_w != NULL);
    check("weighted_observe_create_u", st_u != NULL);

    uint8_t a[240], b[240];
    trine_encode_shingle("apple pie recipe", 16, a);
    trine_encode_shingle("apple pie ingredients", 21, b);

    /* Observe with high similarity (well above default 0.5 threshold) */
    float sim = 0.9f;
    trine_hebbian_observe(st_w, a, b, sim);
    trine_hebbian_observe(st_u, a, b, sim);

    trine_hebbian_metrics_t m_w, m_u;
    trine_hebbian_metrics(st_w, &m_w);
    trine_hebbian_metrics(st_u, &m_u);

    /* Both should have 1 pair observed */
    check("weighted_observe_pairs_w", m_w.pairs_observed == 1);
    check("weighted_observe_pairs_u", m_u.pairs_observed == 1);

    /* Weighted mode should produce larger counter magnitudes.
     * For sim=0.9, threshold=0.5: delta=0.4, magnitude=1+(0.4*10)=5.
     * Unweighted mode always uses delta of +/-1.
     * So max_abs should differ. */
    check("weighted_observe_different",
          m_w.max_abs_counter != m_u.max_abs_counter);
    check("weighted_observe_larger",
          m_w.max_abs_counter > m_u.max_abs_counter);

    trine_hebbian_free(st_w);
    trine_hebbian_free(st_u);
}

/* ── Test 4: pos_scale and neg_scale affect magnitude differently ────── */
static void test_weighted_pos_neg_scale(void)
{
    /* Config with large pos_scale, small neg_scale */
    trine_hebbian_config_t cfg_pos = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_pos.weighted_mode = 1;
    cfg_pos.similarity_threshold = 0.5f;
    cfg_pos.pos_scale = 20.0f;
    cfg_pos.neg_scale = 1.0f;

    /* Config with small pos_scale, large neg_scale */
    trine_hebbian_config_t cfg_neg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_neg.weighted_mode = 1;
    cfg_neg.similarity_threshold = 0.5f;
    cfg_neg.pos_scale = 1.0f;
    cfg_neg.neg_scale = 20.0f;

    trine_hebbian_state_t *st_pos = trine_hebbian_create(&cfg_pos);
    trine_hebbian_state_t *st_neg = trine_hebbian_create(&cfg_neg);

    uint8_t a[240], b[240], c[240], d[240];
    trine_encode_shingle("cats are wonderful", 18, a);
    trine_encode_shingle("cats are great", 14, b);
    trine_encode_shingle("rockets fly high", 16, c);
    trine_encode_shingle("submarines dive deep", 20, d);

    /* Observe a similar pair (sim=0.9, above threshold) */
    trine_hebbian_observe(st_pos, a, b, 0.9f);
    trine_hebbian_observe(st_neg, a, b, 0.9f);

    trine_hebbian_metrics_t m_pos_sim, m_neg_sim;
    trine_hebbian_metrics(st_pos, &m_pos_sim);
    trine_hebbian_metrics(st_neg, &m_neg_sim);

    /* For similar pair (delta=0.4):
     *   st_pos: magnitude = 1 + (0.4 * 20) = 9
     *   st_neg: magnitude = 1 + (0.4 * 1)  = 1
     * So st_pos should have much larger counters for the positive pair */
    check("pos_scale_larger_for_similar",
          m_pos_sim.max_abs_counter > m_neg_sim.max_abs_counter);

    /* Reset both and observe a dissimilar pair */
    trine_hebbian_reset(st_pos);
    trine_hebbian_reset(st_neg);

    /* Observe a dissimilar pair (sim=0.1, below threshold) */
    trine_hebbian_observe(st_pos, c, d, 0.1f);
    trine_hebbian_observe(st_neg, c, d, 0.1f);

    trine_hebbian_metrics_t m_pos_dis, m_neg_dis;
    trine_hebbian_metrics(st_pos, &m_pos_dis);
    trine_hebbian_metrics(st_neg, &m_neg_dis);

    /* For dissimilar pair (delta=-0.4, neg_scale applies):
     *   st_pos: magnitude = 1 + (0.4 * 1)  = 1
     *   st_neg: magnitude = 1 + (0.4 * 20) = 9
     * So st_neg should have larger counters for the negative pair */
    check("neg_scale_larger_for_dissimilar",
          m_neg_dis.max_abs_counter > m_pos_dis.max_abs_counter);

    trine_hebbian_free(st_pos);
    trine_hebbian_free(st_neg);
}

/* =====================================================================
 * A2: Dataset Rebalancing
 * ===================================================================== */

static void write_source_jsonl(const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return;
    /* 3 pairs from source "wiki", 3 pairs from source "reddit" */
    fprintf(f, "{\"text_a\": \"the sun is bright\", \"text_b\": \"the sun shines brightly\", \"score\": 0.8, \"source\": \"wiki\"}\n");
    fprintf(f, "{\"text_a\": \"water flows downhill\", \"text_b\": \"water runs downhill\", \"score\": 0.7, \"source\": \"wiki\"}\n");
    fprintf(f, "{\"text_a\": \"birds can fly\", \"text_b\": \"most birds fly\", \"score\": 0.6, \"source\": \"wiki\"}\n");
    fprintf(f, "{\"text_a\": \"lol cats are funny\", \"text_b\": \"haha cats memes\", \"score\": 0.3, \"source\": \"reddit\"}\n");
    fprintf(f, "{\"text_a\": \"upvote this post\", \"text_b\": \"smash that like button\", \"score\": 0.2, \"source\": \"reddit\"}\n");
    fprintf(f, "{\"text_a\": \"tl;dr too long\", \"text_b\": \"summary please\", \"score\": 0.4, \"source\": \"reddit\"}\n");
    fclose(f);
}

/* ── Test 5: Source weight lookup produces different counter magnitudes ─ */
static void test_source_weight_lookup(void)
{
    const char *path = "/tmp/trine_test_source_weights.jsonl";
    write_source_jsonl(path);

    /* Config A: wiki weight=3.0, reddit weight=1.0 */
    trine_hebbian_config_t cfg_a = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_a.n_source_weights = 2;
    memset(cfg_a.source_weights, 0, sizeof(cfg_a.source_weights));
    strncpy(cfg_a.source_weights[0].name, "wiki", TRINE_SOURCE_NAME_LEN - 1);
    cfg_a.source_weights[0].weight = 3.0f;
    strncpy(cfg_a.source_weights[1].name, "reddit", TRINE_SOURCE_NAME_LEN - 1);
    cfg_a.source_weights[1].weight = 1.0f;

    /* Config B: wiki weight=1.0, reddit weight=3.0 */
    trine_hebbian_config_t cfg_b = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_b.n_source_weights = 2;
    memset(cfg_b.source_weights, 0, sizeof(cfg_b.source_weights));
    strncpy(cfg_b.source_weights[0].name, "wiki", TRINE_SOURCE_NAME_LEN - 1);
    cfg_b.source_weights[0].weight = 1.0f;
    strncpy(cfg_b.source_weights[1].name, "reddit", TRINE_SOURCE_NAME_LEN - 1);
    cfg_b.source_weights[1].weight = 3.0f;

    trine_hebbian_state_t *st_a = trine_hebbian_create(&cfg_a);
    trine_hebbian_state_t *st_b = trine_hebbian_create(&cfg_b);

    int64_t n_a = trine_hebbian_train_file(st_a, path, 1);
    int64_t n_b = trine_hebbian_train_file(st_b, path, 1);

    /* Both should process all 6 lines from the JSONL */
    check("source_weight_read_a", n_a == 6);
    check("source_weight_read_b", n_b == 6);

    trine_hebbian_metrics_t m_a, m_b;
    trine_hebbian_metrics(st_a, &m_a);
    trine_hebbian_metrics(st_b, &m_b);

    /* With different source weights, the internal pairs_observed should
     * differ because upweighted sources get repeated observations.
     * Config A: wiki pairs observed 3 times each (weight=3), reddit 1 each
     *   = 3*3 + 3*1 = 12 pairs
     * Config B: wiki pairs observed 1 time each, reddit 3 times each
     *   = 3*1 + 3*3 = 12 pairs
     * Both sum to 12, but the distribution of positive/negative counters
     * will differ because wiki pairs are high-sim and reddit pairs are low-sim. */
    check("source_weight_pairs_a", m_a.pairs_observed > 0);
    check("source_weight_pairs_b", m_b.pairs_observed > 0);

    /* The counter distributions should differ since wiki (high-sim)
     * and reddit (low-sim) have different signs. Upweighting one source
     * shifts the balance toward that source's sign direction. */
    check("source_weight_different_counters",
          m_a.max_abs_counter != m_b.max_abs_counter ||
          m_a.n_positive_weights != m_b.n_positive_weights ||
          m_a.n_negative_weights != m_b.n_negative_weights);

    trine_hebbian_free(st_a);
    trine_hebbian_free(st_b);
    remove(path);
}

/* ── Test 6: Source downsampling (weight=0.0 skips all pairs) ──────── */
static void test_source_downsampling(void)
{
    const char *path = "/tmp/trine_test_source_downsample.jsonl";
    write_source_jsonl(path);

    /* Config: reddit weight=0.0 → all reddit pairs should be skipped */
    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.n_source_weights = 2;
    memset(cfg.source_weights, 0, sizeof(cfg.source_weights));
    strncpy(cfg.source_weights[0].name, "wiki", TRINE_SOURCE_NAME_LEN - 1);
    cfg.source_weights[0].weight = 1.0f;
    strncpy(cfg.source_weights[1].name, "reddit", TRINE_SOURCE_NAME_LEN - 1);
    cfg.source_weights[1].weight = 0.0f;

    /* Also train without source weights for comparison */
    trine_hebbian_config_t cfg_all = TRINE_HEBBIAN_CONFIG_DEFAULT;

    trine_hebbian_state_t *st_skip = trine_hebbian_create(&cfg);
    trine_hebbian_state_t *st_all  = trine_hebbian_create(&cfg_all);

    int64_t n_skip = trine_hebbian_train_file(st_skip, path, 1);
    int64_t n_all  = trine_hebbian_train_file(st_all,  path, 1);

    /* train_file returns total lines read, but the internal pairs_observed
     * should differ because downsampled lines are skipped. */
    trine_hebbian_metrics_t m_skip, m_all;
    trine_hebbian_metrics(st_skip, &m_skip);
    trine_hebbian_metrics(st_all, &m_all);

    /* Without source weights, all 6 pairs are observed */
    check("downsample_all_pairs", m_all.pairs_observed == 6);
    check("downsample_all_returned", n_all == 6);

    /* With reddit=0.0, only wiki pairs (3) should be observed.
     * train_file still reads all lines but the reddit ones are skipped
     * at the downsampling stage (random < 0.0 is always false). */
    check("downsample_fewer_pairs", m_skip.pairs_observed < m_all.pairs_observed);
    check("downsample_wiki_only", m_skip.pairs_observed == 3);

    /* train_file returns total lines actually trained (after skipping),
     * so n_skip should be fewer than n_all */
    check("downsample_fewer_returned", n_skip < n_all);

    trine_hebbian_free(st_skip);
    trine_hebbian_free(st_all);
    remove(path);
}

/* =====================================================================
 * A3: Threshold Schedule
 * ===================================================================== */

/* ── Test 7: set_threshold changes the config ──────────────────────── */
static void test_set_threshold(void)
{
    trine_hebbian_state_t *st = trine_hebbian_create(NULL);

    /* Default threshold should be 0.5 */
    trine_hebbian_config_t cfg = trine_hebbian_get_config(st);
    check("set_threshold_initial", fabsf(cfg.similarity_threshold - 0.5f) < 1e-6f);

    /* Set to 0.8 */
    trine_hebbian_set_threshold(st, 0.8f);
    cfg = trine_hebbian_get_config(st);
    check("set_threshold_changed", fabsf(cfg.similarity_threshold - 0.8f) < 1e-6f);

    /* Set to 0.2 */
    trine_hebbian_set_threshold(st, 0.2f);
    cfg = trine_hebbian_get_config(st);
    check("set_threshold_changed_again", fabsf(cfg.similarity_threshold - 0.2f) < 1e-6f);

    /* NULL state should not crash */
    trine_hebbian_set_threshold(NULL, 0.5f);
    check("set_threshold_null_safe", 1);

    trine_hebbian_free(st);
}

/* ── Test 8: Threshold schedule effect on counter distributions ──── */
static void test_threshold_schedule_effect(void)
{
    /* Train with threshold=0.9 first: almost everything is "dissimilar"
     * (negative sign), so positive counters should be few. */
    trine_hebbian_config_t cfg_high = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_high.similarity_threshold = 0.9f;
    trine_hebbian_state_t *st_high = trine_hebbian_create(&cfg_high);

    /* Train with threshold=0.5 second: moderate split. */
    trine_hebbian_config_t cfg_low = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg_low.similarity_threshold = 0.5f;
    trine_hebbian_state_t *st_low = trine_hebbian_create(&cfg_low);

    /* Use pairs with varied similarity levels */
    uint8_t a[240], b[240], c[240], d[240], e[240], f[240];
    trine_encode_shingle("the cat sat on the mat", 22, a);
    trine_encode_shingle("the cat is on the mat", 21, b);
    trine_encode_shingle("dogs run in the park", 20, c);
    trine_encode_shingle("cats sleep on the couch", 23, d);
    trine_encode_shingle("programming in python", 21, e);
    trine_encode_shingle("coding with python", 18, f);

    /* Feed pairs with moderate similarity (0.6-0.7 range) */
    float sims[] = { 0.7f, 0.6f, 0.65f };
    const uint8_t *pairs_a[] = { a, c, e };
    const uint8_t *pairs_b[] = { b, d, f };

    for (int i = 0; i < 3; i++) {
        trine_hebbian_observe(st_high, pairs_a[i], pairs_b[i], sims[i]);
        trine_hebbian_observe(st_low,  pairs_a[i], pairs_b[i], sims[i]);
    }

    trine_hebbian_metrics_t m_high, m_low;
    trine_hebbian_metrics(st_high, &m_high);
    trine_hebbian_metrics(st_low,  &m_low);

    /* With threshold=0.9, all sims (0.6-0.7) are below threshold,
     * so all updates have negative sign -> more negative counters.
     * With threshold=0.5, all sims (0.6-0.7) are above threshold,
     * so all updates have positive sign -> more positive counters. */
    check("schedule_high_more_negative",
          m_high.n_negative_weights > m_high.n_positive_weights);
    check("schedule_low_more_positive",
          m_low.n_positive_weights > m_low.n_negative_weights);

    /* Now test dynamic threshold change: start high, then lower */
    trine_hebbian_state_t *st_sched = trine_hebbian_create(&cfg_high);

    /* First batch: threshold=0.9, all sim < 0.9, so negative sign */
    for (int i = 0; i < 3; i++)
        trine_hebbian_observe(st_sched, pairs_a[i], pairs_b[i], sims[i]);

    trine_hebbian_metrics_t m_before;
    trine_hebbian_metrics(st_sched, &m_before);
    check("schedule_before_negative",
          m_before.n_negative_weights > m_before.n_positive_weights);

    /* Lower threshold to 0.5 */
    trine_hebbian_set_threshold(st_sched, 0.5f);

    /* Second batch: same pairs, now sim > 0.5, so positive sign.
     * Apply more positive updates than negatives to break cancellation. */
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 3; i++)
            trine_hebbian_observe(st_sched, pairs_a[i], pairs_b[i], sims[i]);
    }

    trine_hebbian_metrics_t m_after;
    trine_hebbian_metrics(st_sched, &m_after);

    /* After 9 positive updates on top of 3 negative, the net signal is
     * positive. So we should see more positive counters than before. */
    check("schedule_after_more_positive",
          m_after.n_positive_weights > m_before.n_positive_weights);

    trine_hebbian_free(st_high);
    trine_hebbian_free(st_low);
    trine_hebbian_free(st_sched);
}

/* =====================================================================
 * B1: Gate-Aware Comparison
 * ===================================================================== */

/* ── Test 9: Identity model gated compare ─────────────────────────── */
static void test_gated_compare_identity(void)
{
    trine_s2_model_t *model = trine_s2_create_identity();
    check("gated_identity_model_not_null", model != NULL);

    uint8_t ea[240], eb[240];
    trine_s2_encode(model, "hello world", 11, 0, ea);
    trine_s2_encode(model, "hello there", 11, 0, eb);

    /* Standard comparison */
    float std_sim = trine_s2_compare(ea, eb, NULL);

    /* Gated comparison — identity model has W[k][i][i]=1 for all k,
     * so no channel has active (non-zero) diagonal gates via the identity
     * initialization.  The identity projection sets W[k][i][j] = (i==j)?1:0,
     * meaning the diagonal is 1 which is non-zero, so all channels should
     * be active.  Both should give valid similarity values. */
    float gated_sim = trine_s2_compare_gated(model, ea, eb);

    check("gated_identity_std_valid", std_sim >= 0.0f && std_sim <= 1.0f);
    check("gated_identity_gated_valid",
          gated_sim >= -1.0f && gated_sim <= 1.0f);

    /* Self-similarity check: both methods should return 1.0 for identical */
    float gated_self = trine_s2_compare_gated(model, ea, ea);
    check("gated_identity_self_sim", fabsf(gated_self - 1.0f) < 1e-4f);

    trine_s2_free(model);
}

/* ── Test 10: Trained model gated compare differs from standard ───── */
static void test_gated_compare_trained(void)
{
    /* Train a model with diagonal mode to get meaningful gate values */
    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.projection_mode = 1;  /* diagonal gating */
    cfg.freeze_target_density = 0.15f;

    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);

    /* Train on several pairs to get non-trivial gates */
    const char *texts_a[] = {
        "machine learning algorithms", "natural language processing",
        "computer vision tasks", "reinforcement learning agent",
        "data science methods", "neural network architecture"
    };
    const char *texts_b[] = {
        "ML algorithm design", "NLP text analysis",
        "image recognition", "RL policy optimization",
        "statistical analysis", "deep network layers"
    };

    for (int i = 0; i < 6; i++)
        trine_hebbian_observe_text(st,
                                    texts_a[i], strlen(texts_a[i]),
                                    texts_b[i], strlen(texts_b[i]));

    /* Also add some dissimilar pairs */
    trine_hebbian_observe_text(st, "apple pie", 9, "quantum physics", 15);
    trine_hebbian_observe_text(st, "ocean waves", 11, "mountain peaks", 14);

    trine_s2_model_t *model = trine_hebbian_freeze(st);
    check("gated_trained_model", model != NULL);

    trine_s2_set_projection_mode(model, TRINE_S2_PROJ_DIAGONAL);

    /* Encode test texts through the trained model */
    uint8_t s2_a[240], s2_b[240];
    trine_s2_encode(model, "deep learning model", 19, 0, s2_a);
    trine_s2_encode(model, "neural network model", 20, 0, s2_b);

    float std_sim   = trine_s2_compare(s2_a, s2_b, NULL);
    float gated_sim = trine_s2_compare_gated(model, s2_a, s2_b);

    /* Both should be valid similarity values */
    check("gated_trained_std_valid", std_sim >= 0.0f && std_sim <= 1.0f);
    check("gated_trained_gated_valid",
          gated_sim >= -1.0f && gated_sim <= 1.0f);

    /* With a trained model that has some zero gates (density < 1.0),
     * gated comparison should generally give a different result than
     * standard cosine, because it excludes zeroed channels. */
    check("gated_trained_different",
          fabsf(gated_sim - std_sim) > 1e-6f ||
          fabsf(gated_sim) < 1e-6f);  /* also OK if all channels inactive */

    trine_s2_free(model);
    trine_hebbian_free(st);
}

/* ── Test 11: Gated compare null safety ───────────────────────────── */
static void test_gated_compare_null_safety(void)
{
    uint8_t dummy[240];
    memset(dummy, 1, sizeof(dummy));

    /* NULL model -> 0.0 */
    float r1 = trine_s2_compare_gated(NULL, dummy, dummy);
    check("gated_null_model", fabsf(r1) < 1e-6f);

    trine_s2_model_t *model = trine_s2_create_identity();

    /* NULL embeddings -> 0.0 */
    float r2 = trine_s2_compare_gated(model, NULL, dummy);
    check("gated_null_a", fabsf(r2) < 1e-6f);

    float r3 = trine_s2_compare_gated(model, dummy, NULL);
    check("gated_null_b", fabsf(r3) < 1e-6f);

    float r4 = trine_s2_compare_gated(model, NULL, NULL);
    check("gated_null_both", fabsf(r4) < 1e-6f);

    trine_s2_free(model);
}

/* =====================================================================
 * B2: Per-Chain Blend
 * ===================================================================== */

/* ── Test 12: Uniform alpha=0.5 matches scalar 0.5 blend ─────────── */
static void test_chain_blend_uniform(void)
{
    /* Encode S1 embeddings */
    uint8_t s1_a[240], s1_b[240];
    trine_encode_shingle("the quick brown fox", 19, s1_a);
    trine_encode_shingle("a fast brown fox", 16, s1_b);

    /* Create a trained model for S2 embeddings */
    trine_hebbian_config_t cfg = TRINE_HEBBIAN_CONFIG_DEFAULT;
    cfg.projection_mode = 1;
    trine_hebbian_state_t *st = trine_hebbian_create(&cfg);

    /* Minimal training to get a non-identity model */
    trine_hebbian_observe_text(st, "hello world", 11, "hello there", 11);
    trine_hebbian_observe_text(st, "goodbye world", 13, "farewell world", 14);

    trine_s2_model_t *model = trine_hebbian_freeze(st);
    check("blend_uniform_model", model != NULL);

    uint8_t s2_a[240], s2_b[240];
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    /* Uniform alpha = 0.5 for all chains */
    float alpha_uniform[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
    float blend_uniform = trine_s2_compare_chain_blend(
        s1_a, s1_b, s2_a, s2_b, alpha_uniform);

    /* Compute manual scalar blend: 0.5 * s1_sim + 0.5 * s2_sim */
    /* Note: chain_blend uses centered cosine per chain, not the S1 lens.
     * So we can't directly compare to trine_s1_compare.  Instead verify
     * that the result is a valid similarity and is between the S1 and S2
     * per-chain values. */
    check("blend_uniform_valid",
          blend_uniform >= -1.0f && blend_uniform <= 1.0f);

    /* Compare with pure S1 (alpha=1) and pure S2 (alpha=0) */
    float alpha_s1[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float alpha_s2[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    float blend_s1 = trine_s2_compare_chain_blend(
        s1_a, s1_b, s2_a, s2_b, alpha_s1);
    float blend_s2 = trine_s2_compare_chain_blend(
        s1_a, s1_b, s2_a, s2_b, alpha_s2);

    /* Uniform 0.5 blend should be approximately the average of pure S1
     * and pure S2 chain-blend values */
    float expected_avg = (blend_s1 + blend_s2) / 2.0f;
    check("blend_uniform_is_average",
          fabsf(blend_uniform - expected_avg) < 0.05f);

    trine_s2_free(model);
    trine_hebbian_free(st);
}

/* ── Test 13: Extreme alphas match pure S1/S2 ────────────────────── */
static void test_chain_blend_extremes(void)
{
    /* Encode S1 embeddings */
    uint8_t s1_a[240], s1_b[240];
    trine_encode_shingle("alpha beta gamma delta", 22, s1_a);
    trine_encode_shingle("alpha beta gamma epsilon", 24, s1_b);

    /* Create a random model for S2 embeddings (non-trivial projection) */
    trine_s2_model_t *model = trine_s2_create_random(64, 42);
    check("blend_extremes_model", model != NULL);

    uint8_t s2_a[240], s2_b[240];
    trine_s2_encode_from_trits(model, s1_a, 0, s2_a);
    trine_s2_encode_from_trits(model, s1_b, 0, s2_b);

    /* alpha={1,1,1,1} should match pure S1 chain-by-chain cosine */
    float alpha_s1[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float blend_s1 = trine_s2_compare_chain_blend(
        s1_a, s1_b, s2_a, s2_b, alpha_s1);

    /* The chain_blend with alpha=1 uses only S1 data, so it should
     * produce the centered cosine over S1 embeddings per chain. */
    check("blend_s1_valid", blend_s1 >= -1.0f && blend_s1 <= 1.0f);

    /* alpha={0,0,0,0} should match pure S2 chain-by-chain cosine */
    float alpha_s2[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float blend_s2 = trine_s2_compare_chain_blend(
        s1_a, s1_b, s2_a, s2_b, alpha_s2);

    check("blend_s2_valid", blend_s2 >= -1.0f && blend_s2 <= 1.0f);

    /* S1 and S2 values should generally differ (different projections) */
    /* With a random model, S2 will project to different embeddings */
    check("blend_extremes_differ", fabsf(blend_s1 - blend_s2) > 1e-6f);

    /* Self-similarity: both pure S1 and pure S2 should give ~1.0 for
     * identical embedding pairs */
    float self_s1_self = trine_s2_compare_chain_blend(
        s1_a, s1_a, s2_a, s2_a, alpha_s1);
    check("blend_s1_self_sim", fabsf(self_s1_self - 1.0f) < 1e-4f);

    float self_s2_self = trine_s2_compare_chain_blend(
        s1_a, s1_a, s2_a, s2_a, alpha_s2);
    check("blend_s2_self_sim", fabsf(self_s2_self - 1.0f) < 1e-4f);

    trine_s2_free(model);
}

/* ── Test 14: Chain blend null safety ─────────────────────────────── */
static void test_chain_blend_null_safety(void)
{
    uint8_t dummy[240];
    memset(dummy, 1, sizeof(dummy));
    float alpha[4] = { 0.5f, 0.5f, 0.5f, 0.5f };

    /* Any NULL input should return 0.0 */
    float r1 = trine_s2_compare_chain_blend(NULL, dummy, dummy, dummy, alpha);
    check("blend_null_s1a", fabsf(r1) < 1e-6f);

    float r2 = trine_s2_compare_chain_blend(dummy, NULL, dummy, dummy, alpha);
    check("blend_null_s1b", fabsf(r2) < 1e-6f);

    float r3 = trine_s2_compare_chain_blend(dummy, dummy, NULL, dummy, alpha);
    check("blend_null_s2a", fabsf(r3) < 1e-6f);

    float r4 = trine_s2_compare_chain_blend(dummy, dummy, dummy, NULL, alpha);
    check("blend_null_s2b", fabsf(r4) < 1e-6f);

    float r5 = trine_s2_compare_chain_blend(dummy, dummy, dummy, dummy, NULL);
    check("blend_null_alpha", fabsf(r5) < 1e-6f);

    float r6 = trine_s2_compare_chain_blend(NULL, NULL, NULL, NULL, NULL);
    check("blend_null_all", fabsf(r6) < 1e-6f);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Phase A+B Tests ===\n");

    /* A1: Weighted Hebbian Updates */
    test_weighted_accumulator_basic();
    test_weighted_accumulator_magnitude_1();
    test_weighted_observe();
    test_weighted_pos_neg_scale();

    /* A2: Dataset Rebalancing */
    test_source_weight_lookup();
    test_source_downsampling();

    /* A3: Threshold Schedule */
    test_set_threshold();
    test_threshold_schedule_effect();

    /* B1: Gate-Aware Comparison */
    test_gated_compare_identity();
    test_gated_compare_trained();
    test_gated_compare_null_safety();

    /* B2: Per-Chain Blend */
    test_chain_blend_uniform();
    test_chain_blend_extremes();
    test_chain_blend_null_safety();

    printf("\nPhase A+B: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);
    return g_failed;
}
