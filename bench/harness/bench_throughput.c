/* =====================================================================
 * TRINE v1.0.3 — Comprehensive Throughput Benchmark Harness
 * =====================================================================
 *
 * Measures end-to-end throughput for all major TRINE operations:
 *
 *   1. S1 encode throughput   — trine_encode_shingle() at 3 text lengths
 *   2. S1 compare throughput  — trine_s1_compare() with uniform lens
 *   3. S2 encode throughput   — identity & random models at depth 0 & 4
 *   4. S2 compare throughput  — trine_s2_compare_chain_blend()
 *   5. Block-diagonal projection — trine_projection_majority_block()
 *   6. Index add/query        — trine_s1_index_{add,query}()
 *
 * Output: tab-separated rows:
 *   operation\tcount\ttime_ms\tops_per_sec
 *
 * Build:
 *   cc -O2 -Wall -Wextra -Werror -Isrc/encode -Isrc/compare -Isrc/index \
 *      -Isrc/canon -Isrc/algebra -Isrc/model -Isrc/stage2/projection    \
 *      -Isrc/stage2/cascade -Isrc/stage2/inference -Isrc/stage2/hebbian \
 *      -Isrc/stage2/persist \
 *      -o build/bench_throughput bench/harness/bench_throughput.c        \
 *      build/libtrine.a -lm
 *
 * Usage:
 *   ./build/bench_throughput            # full benchmark suite
 *   ./build/bench_throughput --quick    # shorter runs (~10x fewer iters)
 *
 * ===================================================================== */

#define _POSIX_C_SOURCE 200809L  /* clock_gettime */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "trine_encode.h"
#include "trine_stage1.h"
#include "trine_stage2.h"
#include "trine_project.h"

/* =====================================================================
 * Timing
 * ===================================================================== */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* =====================================================================
 * Anti-optimisation sinks
 * ===================================================================== */

static volatile int    sink_i;
static volatile float  sink_f;

/* =====================================================================
 * Result emission
 * ===================================================================== */

static void emit(const char *op, long long count, double time_ms,
                 double ops_per_sec)
{
    printf("%s\t%lld\t%.3f\t%.0f\n", op, count, time_ms, ops_per_sec);
}

/* =====================================================================
 * Adaptive iteration count
 *
 * Each benchmark targets at least MIN_RUNTIME_MS of wall-clock time.
 * We do a short calibration pass to estimate how many iterations we
 * need, then run the real measurement pass.
 * ===================================================================== */

#define MIN_RUNTIME_MS      100.0  /* at least 100 ms per benchmark         */
#define QUICK_RUNTIME_MS     50.0  /* at least 50 ms in quick mode          */
#define CALIBRATION_ITERS   1000   /* quick calibration probe               */

static int estimate_iters(double single_op_sec, int quick)
{
    if (single_op_sec <= 0.0) return 100000;
    double target_ms = quick ? QUICK_RUNTIME_MS : MIN_RUNTIME_MS;
    double target_sec = target_ms / 1000.0;
    int n = (int)(target_sec / single_op_sec);
    if (n < 1000) n = 1000;
    if (n > 50000000) n = 50000000;
    return n;
}

/* =====================================================================
 * Test data generation
 * ===================================================================== */

/* Short text: 20 chars */
static const char TEXT_SHORT[] = "the quick brown fox!";

/* Medium text: ~200 chars */
static const char TEXT_MEDIUM[] =
    "The quick brown fox jumps over the lazy dog near the river bank on "
    "a warm summer afternoon while the birds sing and the wind blows "
    "gently through the trees swaying in the golden sunlight above";

/* Long text buffer: ~2000 chars, filled at startup */
static char TEXT_LONG[2048];

static void generate_long_text(void)
{
    const char *sentences[] = {
        "The cellular automaton processes each snap in constant time. ",
        "Cascade waves propagate through the toroidal mesh topology. ",
        "Rank monotonicity ensures security isolation between strata. ",
        "The ternary algebra provides seven irreducible cell types. ",
        "Each endomorphism maps the 27-element space deterministically. ",
        "Fleet generation synthesizes billions of snaps on the GPU. ",
        "The golden vector stress benchmark validates correctness at scale. ",
        "Domain phase transitions follow the DTRIT tick sequence exactly. ",
        "Fiber numbers classify the algebraic structure of each mapping. ",
        "The identity particle preserves all input without modification. ",
        "Collapse and split operations create and destroy information. ",
        "The oscillator cell produces a deterministic clock signal. ",
        "Permutation-rank cells form the computational kernel fully. ",
        "Two-body composition yields the complete multiplication table. ",
        "The forge interface provides eleven algebraic exploration ops. ",
        "Snap step execution requires zero branches and fits tightly. ",
    };
    int n_sent = (int)(sizeof(sentences) / sizeof(sentences[0]));
    int pos = 0;
    int idx = 0;
    while (pos < 2000) {
        int slen = (int)strlen(sentences[idx % n_sent]);
        if (pos + slen >= (int)sizeof(TEXT_LONG) - 1) break;
        memcpy(TEXT_LONG + pos, sentences[idx % n_sent], (size_t)slen);
        pos += slen;
        idx++;
    }
    TEXT_LONG[pos] = '\0';
}

/* Fill a 240-byte embedding with pseudo-random trits using an LCG. */
static unsigned int fill_random_emb(uint8_t emb[240], unsigned int seed)
{
    for (int i = 0; i < 240; i++) {
        seed = seed * 1103515245u + 12345u;
        emb[i] = (uint8_t)((seed >> 16) % 3);
    }
    return seed;
}

/* =====================================================================
 * 1. S1 Encode Throughput
 * ===================================================================== */

static void bench_s1_encode(int quick)
{
    struct {
        const char *label;
        const char *text;
        size_t      len;
    } cases[3];

    cases[0].label = "s1_encode_short_20c";
    cases[0].text  = TEXT_SHORT;
    cases[0].len   = strlen(TEXT_SHORT);

    cases[1].label = "s1_encode_medium_200c";
    cases[1].text  = TEXT_MEDIUM;
    cases[1].len   = strlen(TEXT_MEDIUM);

    cases[2].label = "s1_encode_long_2000c";
    cases[2].text  = TEXT_LONG;
    cases[2].len   = strlen(TEXT_LONG);

    uint8_t channels[240];

    for (int t = 0; t < 3; t++) {
        /* Calibration */
        double cal_t0 = now_sec();
        for (int i = 0; i < CALIBRATION_ITERS; i++)
            trine_encode_shingle(cases[t].text, cases[t].len, channels);
        double cal_elapsed = now_sec() - cal_t0;
        double per_op = cal_elapsed / CALIBRATION_ITERS;

        int iters = estimate_iters(per_op, quick);
        if (iters < 1000) iters = 1000;

        /* Warmup */
        for (int i = 0; i < iters / 100; i++)
            trine_encode_shingle(cases[t].text, cases[t].len, channels);

        /* Timed run */
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            trine_encode_shingle(cases[t].text, cases[t].len, channels);
        double elapsed = now_sec() - t0;

        sink_i = (int)channels[0];

        double time_ms = elapsed * 1000.0;
        double ops = (double)iters / elapsed;
        emit(cases[t].label, (long long)iters, time_ms, ops);
    }
}

/* =====================================================================
 * 2. S1 Compare Throughput (uniform lens)
 * ===================================================================== */

static void bench_s1_compare(int quick)
{
    uint8_t emb_a[240], emb_b[240];
    trine_encode_shingle("The quick brown fox jumps over the lazy dog",
                         44, emb_a);
    trine_encode_shingle("The fast brown fox leaps over the tired dog",
                         44, emb_b);

    trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;

    /* Calibration */
    double cal_t0 = now_sec();
    for (int i = 0; i < CALIBRATION_ITERS; i++)
        sink_f = trine_s1_compare(emb_a, emb_b, &lens);
    double cal_elapsed = now_sec() - cal_t0;
    double per_op = cal_elapsed / CALIBRATION_ITERS;

    int iters = estimate_iters(per_op, quick);
    if (iters < 1000) iters = 1000;

    /* Warmup */
    for (int i = 0; i < iters / 100; i++)
        sink_f = trine_s1_compare(emb_a, emb_b, &lens);

    /* Timed run */
    float acc = 0.0f;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        acc += trine_s1_compare(emb_a, emb_b, &lens);
    double elapsed = now_sec() - t0;

    sink_f = acc;

    double time_ms = elapsed * 1000.0;
    double ops = (double)iters / elapsed;
    emit("s1_compare_uniform", (long long)iters, time_ms, ops);
}

/* =====================================================================
 * 3. S2 Encode Throughput
 *    - identity model, depth 0
 *    - identity model, depth 4
 *    - random model (512 cells), depth 0
 *    - random model (512 cells), depth 4
 * ===================================================================== */

static void bench_s2_encode_case(const char *label, trine_s2_model_t *model,
                                 const char *text, size_t len,
                                 uint32_t depth, int quick)
{
    uint8_t out[240];

    /* Calibration */
    double cal_t0 = now_sec();
    for (int i = 0; i < CALIBRATION_ITERS; i++)
        trine_s2_encode(model, text, len, depth, out);
    double cal_elapsed = now_sec() - cal_t0;
    double per_op = cal_elapsed / CALIBRATION_ITERS;

    int iters = estimate_iters(per_op, quick);
    if (iters < 1000) iters = 1000;

    /* Warmup */
    for (int i = 0; i < iters / 100; i++)
        trine_s2_encode(model, text, len, depth, out);

    /* Timed run */
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        trine_s2_encode(model, text, len, depth, out);
    double elapsed = now_sec() - t0;

    sink_i = (int)out[0];

    double time_ms = elapsed * 1000.0;
    double ops = (double)iters / elapsed;
    emit(label, (long long)iters, time_ms, ops);
}

static void bench_s2_encode(int quick)
{
    const char *text = "The quick brown fox jumps over the lazy dog";
    size_t len = strlen(text);

    /* Identity model */
    {
        trine_s2_model_t *m = trine_s2_create_identity();
        if (!m) {
            fprintf(stderr, "ERROR: failed to create identity model\n");
            return;
        }
        bench_s2_encode_case("s2_encode_identity_d0", m, text, len, 0, quick);
        bench_s2_encode_case("s2_encode_identity_d4", m, text, len, 4, quick);
        trine_s2_free(m);
    }

    /* Random model (512 cascade cells) */
    {
        trine_s2_model_t *m = trine_s2_create_random(512, 42);
        if (!m) {
            fprintf(stderr, "ERROR: failed to create random model\n");
            return;
        }
        bench_s2_encode_case("s2_encode_random_d0", m, text, len, 0, quick);
        bench_s2_encode_case("s2_encode_random_d4", m, text, len, 4, quick);
        trine_s2_free(m);
    }
}

/* =====================================================================
 * 4. S2 Compare Throughput (chain blend)
 * ===================================================================== */

static void bench_s2_compare(int quick)
{
    const char *text_a = "The quick brown fox jumps over the lazy dog";
    const char *text_b = "The fast brown fox leaps over the tired dog";

    /* Encode S1 embeddings */
    uint8_t s1_a[240], s1_b[240];
    trine_encode_shingle(text_a, strlen(text_a), s1_a);
    trine_encode_shingle(text_b, strlen(text_b), s1_b);

    /* Encode S2 embeddings via random model */
    trine_s2_model_t *m = trine_s2_create_random(512, 42);
    if (!m) {
        fprintf(stderr, "ERROR: failed to create S2 model for compare bench\n");
        return;
    }

    uint8_t s2_a[240], s2_b[240];
    trine_s2_encode(m, text_a, strlen(text_a), 0, s2_a);
    trine_s2_encode(m, text_b, strlen(text_b), 0, s2_b);

    float alpha[4] = {0.5f, 0.5f, 0.5f, 0.5f};

    /* Calibration */
    double cal_t0 = now_sec();
    for (int i = 0; i < CALIBRATION_ITERS; i++)
        sink_f = trine_s2_compare_chain_blend(s1_a, s1_b, s2_a, s2_b, alpha);
    double cal_elapsed = now_sec() - cal_t0;
    double per_op = cal_elapsed / CALIBRATION_ITERS;

    int iters = estimate_iters(per_op, quick);
    if (iters < 1000) iters = 1000;

    /* Warmup */
    for (int i = 0; i < iters / 100; i++)
        sink_f = trine_s2_compare_chain_blend(s1_a, s1_b, s2_a, s2_b, alpha);

    /* Timed run */
    float acc = 0.0f;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        acc += trine_s2_compare_chain_blend(s1_a, s1_b, s2_a, s2_b, alpha);
    double elapsed = now_sec() - t0;

    sink_f = acc;
    trine_s2_free(m);

    double time_ms = elapsed * 1000.0;
    double ops = (double)iters / elapsed;
    emit("s2_compare_chain_blend", (long long)iters, time_ms, ops);
}

/* =====================================================================
 * 5. Block-Diagonal Projection Throughput
 * ===================================================================== */

static void bench_block_diag(int quick)
{
    int K = TRINE_PROJECT_K;  /* 3 */
    size_t block_size = (size_t)K * TRINE_S2_N_CHAINS
                        * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM;

    uint8_t *W_blocks = (uint8_t *)malloc(block_size);
    if (!W_blocks) {
        fprintf(stderr, "ERROR: allocation failed for block-diag weights\n");
        return;
    }
    trine_projection_block_random(W_blocks, K, 12345);

    /* Prepare input vector */
    uint8_t x[240], y[240];
    trine_encode_shingle("The quick brown fox jumps over the lazy dog",
                         44, x);

    /* Calibration */
    double cal_t0 = now_sec();
    for (int i = 0; i < CALIBRATION_ITERS; i++)
        trine_projection_majority_block(W_blocks, K, x, y);
    double cal_elapsed = now_sec() - cal_t0;
    double per_op = cal_elapsed / CALIBRATION_ITERS;

    int iters = estimate_iters(per_op, quick);
    if (iters < 1000) iters = 1000;

    /* Warmup */
    for (int i = 0; i < iters / 100; i++)
        trine_projection_majority_block(W_blocks, K, x, y);

    /* Timed run */
    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        trine_projection_majority_block(W_blocks, K, x, y);
    double elapsed = now_sec() - t0;

    sink_i = (int)y[0];
    free(W_blocks);

    double time_ms = elapsed * 1000.0;
    double ops = (double)iters / elapsed;
    emit("block_diag_projection_K3", (long long)iters, time_ms, ops);
}

/* =====================================================================
 * 6. Index Add / Query Throughput (linear scan)
 * ===================================================================== */

static void bench_index(int quick)
{
    int n_entries = quick ? 1000 : 5000;
    int n_queries = quick ? 200 : 1000;

    /* Pre-generate embeddings */
    uint8_t *embeddings = (uint8_t *)malloc((size_t)n_entries * 240);
    if (!embeddings) {
        fprintf(stderr, "ERROR: allocation failed for index embeddings\n");
        return;
    }

    unsigned int seed = 77777;
    for (int i = 0; i < n_entries; i++)
        seed = fill_random_emb(embeddings + (size_t)i * 240, seed);

    /* Generate query embeddings */
    uint8_t *queries = (uint8_t *)malloc((size_t)n_queries * 240);
    if (!queries) {
        fprintf(stderr, "ERROR: allocation failed for query embeddings\n");
        free(embeddings);
        return;
    }
    seed = 54321;
    for (int i = 0; i < n_queries; i++)
        seed = fill_random_emb(queries + (size_t)i * 240, seed);

    /* --- Index add throughput --- */
    {
        trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
        trine_s1_index_t *idx = trine_s1_index_create(&config);
        if (!idx) {
            fprintf(stderr, "ERROR: index creation failed\n");
            free(embeddings);
            free(queries);
            return;
        }

        double t0 = now_sec();
        for (int i = 0; i < n_entries; i++)
            trine_s1_index_add(idx, embeddings + (size_t)i * 240, NULL);
        double elapsed = now_sec() - t0;

        double time_ms = elapsed * 1000.0;
        double ops = (double)n_entries / elapsed;
        emit("index_add", (long long)n_entries, time_ms, ops);

        /* --- Index query throughput (using populated index) --- */

        /* Warmup */
        for (int i = 0; i < 5; i++) {
            trine_s1_result_t r = trine_s1_index_query(idx, queries[0] ? queries : embeddings);
            sink_i = r.matched_index;
        }

        /* Ensure we run for at least MIN_RUNTIME_MS.  If a single pass
         * is too fast, we do multiple passes over the query array. */
        double cal_t0 = now_sec();
        for (int i = 0; i < n_queries && i < 100; i++) {
            trine_s1_result_t r = trine_s1_index_query(idx,
                queries + (size_t)i * 240);
            sink_i = r.matched_index;
        }
        double cal_elapsed = now_sec() - cal_t0;
        double per_query = cal_elapsed / (n_queries < 100 ? n_queries : 100);

        int total_queries = n_queries;
        int passes = 1;
        double target = MIN_RUNTIME_MS / 1000.0;
        if (per_query * (double)n_queries < target) {
            passes = (int)(target / (per_query * (double)n_queries));
            if (passes < 1) passes = 1;
            if (passes > 100) passes = 100;
            total_queries = n_queries * passes;
        }

        double qt0 = now_sec();
        for (int p = 0; p < passes; p++) {
            for (int i = 0; i < n_queries; i++) {
                trine_s1_result_t r = trine_s1_index_query(idx,
                    queries + (size_t)i * 240);
                sink_i = r.matched_index;
            }
        }
        double q_elapsed = now_sec() - qt0;

        double q_time_ms = q_elapsed * 1000.0;
        double q_ops = (double)total_queries / q_elapsed;

        char label[64];
        snprintf(label, sizeof(label), "index_query_%d_entries", n_entries);
        emit(label, (long long)total_queries, q_time_ms, q_ops);

        trine_s1_index_free(idx);
    }

    free(embeddings);
    free(queries);
}

/* =====================================================================
 * Main
 * ===================================================================== */

int main(int argc, char **argv)
{
    int quick = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--quick]\n", argv[0]);
            printf("  --quick   Run with shorter iterations\n\n");
            printf("Output: tab-separated columns:\n");
            printf("  operation  count  time_ms  ops_per_sec\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    /* Generate long text data */
    generate_long_text();

    /* Header */
    printf("operation\tcount\ttime_ms\tops_per_sec\n");

    /* 1. S1 encode throughput */
    bench_s1_encode(quick);

    /* 2. S1 compare throughput */
    bench_s1_compare(quick);

    /* 3. S2 encode throughput */
    bench_s2_encode(quick);

    /* 4. S2 compare throughput (chain blend) */
    bench_s2_compare(quick);

    /* 5. Block-diagonal projection throughput */
    bench_block_diag(quick);

    /* 6. Index add/query throughput */
    bench_index(quick);

    return 0;
}
