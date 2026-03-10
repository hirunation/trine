/* =====================================================================
 * TRINE Stage-2 — Projection Mode Comparison Benchmark
 * =====================================================================
 *
 * Compares all 4 projection modes across 3 dimensions:
 *
 *   1. Throughput (ops/sec) — projection-only, 10K+ iterations
 *   2. Majority vote overhead — K=1 vs K=3 for block-diagonal
 *   3. Quality — S1, identity S2, random diagonal, random block-diag
 *      on 20 text pairs
 *   4. Memory footprint — weight sizes per mode
 *
 * Modes:
 *   0 = Full sign projection     (K * 240 * 240 weights)
 *   1 = Diagonal gating           (K * 240 weights)
 *   2 = Sparse sign projection   (K * 240 * 240 weights)
 *   3 = Block-diagonal           (K * 4 * 60 * 60 weights)
 *
 * Build:
 *   cc -O2 -Wall -Wextra -Werror -Isrc/encode -Isrc/compare -Isrc/index \
 *      -Isrc/canon -Isrc/algebra -Isrc/model -Isrc/stage2/projection \
 *      -Isrc/stage2/cascade -Isrc/stage2/inference -Isrc/stage2/hebbian \
 *      -Isrc/stage2/persist -o build/bench_projection \
 *      bench/harness/bench_projection.c build/libtrine.a -lm
 *
 * ===================================================================== */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "trine_encode.h"
#include "trine_stage1.h"
#include "trine_project.h"
#include "trine_stage2.h"

/* ── Constants ─────────────────────────────────────────────────────── */

#define DIM         TRINE_PROJECT_DIM  /* 240 */
#define K           TRINE_PROJECT_K    /* 3   */
#define N_CHAINS    TRINE_S2_N_CHAINS  /* 4   */
#define CHAIN_DIM   TRINE_S2_CHAIN_DIM /* 60  */

#define BENCH_ITERS      50000
#define WARMUP_ITERS      5000
#define MAJORITY_ITERS   50000
#define N_PAIRS             20

/* ── Timing helper ─────────────────────────────────────────────────── */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Prevent dead-code elimination */
static volatile uint8_t bench_sink = 0;

/* ── LCG helper for deterministic random trits ────────────────────── */

static uint64_t lcg_state;

static void lcg_seed(uint64_t s) { lcg_state = s; }

static uint8_t lcg_trit(void)
{
    lcg_state = lcg_state * 6364136223846793005ULL
              + 1442695040888963407ULL;
    return (uint8_t)((lcg_state >> 32) % 3);
}

static void fill_random_trits(uint8_t *buf, size_t n)
{
    for (size_t i = 0; i < n; i++)
        buf[i] = lcg_trit();
}

/* ── Text pairs for quality comparison ─────────────────────────────── */

static const char *pair_a[N_PAIRS] = {
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "The cat sat on the mat",
    "Python is a programming language",
    "Climate change is a global challenge",
    "The stock market closed higher today",
    "Neural networks can learn complex patterns",
    "The president signed the executive order",
    "Quantum computing uses qubits for computation",
    "The restaurant serves Italian cuisine",
    "Blockchain technology enables decentralized systems",
    "The patient was diagnosed with pneumonia",
    "Renewable energy sources include solar and wind",
    "The court ruled in favor of the defendant",
    "Deep learning requires large datasets",
    "The airplane departed from the terminal",
    "Natural language processing analyzes text data",
    "The company reported quarterly earnings",
    "Genetic algorithms mimic biological evolution",
    "The bridge spans across the river"
};

static const char *pair_b[N_PAIRS] = {
    "A fast brown fox leaps over a sleepy dog",
    "AI includes machine learning as a subfield",
    "A cat was sitting upon the mat",
    "Python is a popular scripting language",
    "Global warming poses worldwide challenges",
    "The equity market ended the day with gains",
    "Artificial neural nets discover intricate patterns",
    "The executive order was signed by the president",
    "Quantum computers operate with quantum bits",
    "Italian food is served at the restaurant",
    "Decentralized systems are powered by blockchain",
    "The diagnosis for the patient was pneumonia",
    "Solar and wind are renewable energy types",
    "The defendant won the court ruling",
    "Large training sets are needed for deep learning",
    "The flight left the airport terminal",
    "Text analysis is done through NLP",
    "Quarterly earnings were reported by the firm",
    "Biological evolution inspires genetic algorithms",
    "The river is crossed by a bridge"
};

/* ── Section 1: Projection Throughput ──────────────────────────────── */

static void bench_throughput(void)
{
    printf("===================================================================\n");
    printf("  Section 1: Projection Throughput (raw projection, no encode)\n");
    printf("===================================================================\n");
    printf("  Iterations: %d  (warmup: %d)\n\n", BENCH_ITERS, WARMUP_ITERS);

    /* Prepare random input trit vector */
    uint8_t x[DIM], y[DIM];
    lcg_seed(12345);
    fill_random_trits(x, DIM);

    /* Prepare full projection weights (modes 0, 1, 2) */
    trine_projection_t proj;
    trine_projection_random(&proj, 42);

    /* Prepare block-diagonal weights (mode 3) */
    uint8_t block_w[K * N_CHAINS * CHAIN_DIM * CHAIN_DIM];
    trine_projection_block_random(block_w, K, 42);

    double elapsed, rate, us;
    const char *mode_names[4] = {
        "Mode 0: Full sign",
        "Mode 1: Diagonal gate",
        "Mode 2: Sparse sign",
        "Mode 3: Block-diagonal"
    };

    /* ── Mode 0: Full sign projection (K=3 majority) ─────────────── */
    for (int i = 0; i < WARMUP_ITERS; i++)
        trine_project_majority_sign(&proj, x, y);

    elapsed = 0.0;
    {
        double t0 = now_sec();
        for (int i = 0; i < BENCH_ITERS; i++)
            trine_project_majority_sign(&proj, x, y);
        elapsed = now_sec() - t0;
    }
    bench_sink = y[0];
    rate = (double)BENCH_ITERS / elapsed;
    us   = (elapsed / (double)BENCH_ITERS) * 1e6;
    printf("  %-28s %12.0f ops/sec  (%7.3f us/op)\n",
           mode_names[0], rate, us);

    /* ── Mode 1: Diagonal gating (K=3 majority) ──────────────────── */
    for (int i = 0; i < WARMUP_ITERS; i++)
        trine_project_diagonal_gate(&proj, x, y);

    {
        double t0 = now_sec();
        for (int i = 0; i < BENCH_ITERS; i++)
            trine_project_diagonal_gate(&proj, x, y);
        elapsed = now_sec() - t0;
    }
    bench_sink = y[0];
    rate = (double)BENCH_ITERS / elapsed;
    us   = (elapsed / (double)BENCH_ITERS) * 1e6;
    printf("  %-28s %12.0f ops/sec  (%7.3f us/op)\n",
           mode_names[1], rate, us);

    /* ── Mode 2: Sparse sign projection (K=3 majority) ───────────── */
    for (int i = 0; i < WARMUP_ITERS; i++)
        trine_project_majority_sparse_sign(&proj, x, y);

    {
        double t0 = now_sec();
        for (int i = 0; i < BENCH_ITERS; i++)
            trine_project_majority_sparse_sign(&proj, x, y);
        elapsed = now_sec() - t0;
    }
    bench_sink = y[0];
    rate = (double)BENCH_ITERS / elapsed;
    us   = (elapsed / (double)BENCH_ITERS) * 1e6;
    printf("  %-28s %12.0f ops/sec  (%7.3f us/op)\n",
           mode_names[2], rate, us);

    /* ── Mode 3: Block-diagonal (K=3 majority) ───────────────────── */
    for (int i = 0; i < WARMUP_ITERS; i++)
        trine_projection_majority_block(block_w, K, x, y);

    {
        double t0 = now_sec();
        for (int i = 0; i < BENCH_ITERS; i++)
            trine_projection_majority_block(block_w, K, x, y);
        elapsed = now_sec() - t0;
    }
    bench_sink = y[0];
    rate = (double)BENCH_ITERS / elapsed;
    us   = (elapsed / (double)BENCH_ITERS) * 1e6;
    printf("  %-28s %12.0f ops/sec  (%7.3f us/op)\n",
           mode_names[3], rate, us);

    printf("\n");
}

/* ── Section 2: Majority Vote Overhead (K=1 vs K=3 block-diag) ───── */

static void bench_majority_overhead(void)
{
    printf("===================================================================\n");
    printf("  Section 2: Majority Vote Overhead (block-diagonal)\n");
    printf("===================================================================\n");
    printf("  Iterations: %d  (warmup: %d)\n\n", MAJORITY_ITERS, WARMUP_ITERS);

    uint8_t x[DIM], y[DIM];
    lcg_seed(54321);
    fill_random_trits(x, DIM);

    /* K=1 block-diagonal weights */
    uint8_t block_w1[1 * N_CHAINS * CHAIN_DIM * CHAIN_DIM];
    trine_projection_block_random(block_w1, 1, 99);

    /* K=3 block-diagonal weights */
    uint8_t block_w3[K * N_CHAINS * CHAIN_DIM * CHAIN_DIM];
    trine_projection_block_random(block_w3, K, 99);

    double elapsed, rate, us;

    /* ── K=1 ──────────────────────────────────────────────────────── */
    for (int i = 0; i < WARMUP_ITERS; i++)
        trine_projection_majority_block(block_w1, 1, x, y);

    {
        double t0 = now_sec();
        for (int i = 0; i < MAJORITY_ITERS; i++)
            trine_projection_majority_block(block_w1, 1, x, y);
        elapsed = now_sec() - t0;
    }
    bench_sink = y[0];
    rate = (double)MAJORITY_ITERS / elapsed;
    us   = (elapsed / (double)MAJORITY_ITERS) * 1e6;
    double rate_k1 = rate;
    double us_k1   = us;
    printf("  Block-diag K=1 (no vote)   %12.0f ops/sec  (%7.3f us/op)\n",
           rate, us);

    /* ── K=3 ──────────────────────────────────────────────────────── */
    for (int i = 0; i < WARMUP_ITERS; i++)
        trine_projection_majority_block(block_w3, K, x, y);

    {
        double t0 = now_sec();
        for (int i = 0; i < MAJORITY_ITERS; i++)
            trine_projection_majority_block(block_w3, K, x, y);
        elapsed = now_sec() - t0;
    }
    bench_sink = y[0];
    rate = (double)MAJORITY_ITERS / elapsed;
    us   = (elapsed / (double)MAJORITY_ITERS) * 1e6;
    printf("  Block-diag K=3 (majority)  %12.0f ops/sec  (%7.3f us/op)\n",
           rate, us);

    double overhead = (us / us_k1 - 1.0) * 100.0;
    double speedup  = rate_k1 / rate;
    printf("\n  Overhead: K=3 is %.1fx slower (%.1f%% more time per op)\n\n",
           speedup, overhead);
}

/* ── Section 3: Quality Comparison on 20 Text Pairs ──────────────── */

static void bench_quality(void)
{
    printf("===================================================================\n");
    printf("  Section 3: Quality Comparison (%d text pairs)\n", N_PAIRS);
    printf("===================================================================\n");
    printf("  Columns: S1 = Stage-1 cosine, ID = identity S2,\n");
    printf("           DIAG = random diagonal, BDIAG = random block-diagonal\n\n");

    /* Create models */
    trine_s2_model_t *m_identity = trine_s2_create_identity();
    trine_s2_model_t *m_diag    = trine_s2_create_random(0, 7777);
    trine_s2_model_t *m_bdiag;

    /* Set diagonal mode on m_diag */
    trine_s2_set_projection_mode(m_diag, TRINE_S2_PROJ_DIAGONAL);

    /* Create block-diagonal model */
    {
        uint8_t bw[K * N_CHAINS * CHAIN_DIM * CHAIN_DIM];
        trine_projection_block_random(bw, K, 8888);
        m_bdiag = trine_s2_create_block_diagonal(bw, K, 0, 8888);
    }

    trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;

    /* Table header */
    printf("  %-3s  %-42s  %6s  %6s  %6s  %6s\n",
           "#", "Pair (truncated)", "S1", "ID", "DIAG", "BDIAG");
    printf("  ---  ------------------------------------------"
           "  ------  ------  ------  ------\n");

    double sum_s1 = 0.0, sum_id = 0.0, sum_diag = 0.0, sum_bdiag = 0.0;

    for (int p = 0; p < N_PAIRS; p++) {
        /* Stage-1 encode */
        uint8_t s1_a[DIM], s1_b[DIM];
        trine_s1_encode(pair_a[p], strlen(pair_a[p]), s1_a);
        trine_s1_encode(pair_b[p], strlen(pair_b[p]), s1_b);

        /* S1 similarity */
        float sim_s1 = trine_s1_compare(s1_a, s1_b, &lens);

        /* S2 identity (should equal S1 for depth=0) */
        uint8_t s2_a[DIM], s2_b[DIM];
        trine_s2_encode(m_identity, pair_a[p], strlen(pair_a[p]), 0, s2_a);
        trine_s2_encode(m_identity, pair_b[p], strlen(pair_b[p]), 0, s2_b);
        float sim_id = trine_s2_compare(s2_a, s2_b, NULL);

        /* S2 random diagonal */
        trine_s2_encode(m_diag, pair_a[p], strlen(pair_a[p]), 0, s2_a);
        trine_s2_encode(m_diag, pair_b[p], strlen(pair_b[p]), 0, s2_b);
        float sim_diag = trine_s2_compare(s2_a, s2_b, NULL);

        /* S2 random block-diagonal */
        trine_s2_encode(m_bdiag, pair_a[p], strlen(pair_a[p]), 0, s2_a);
        trine_s2_encode(m_bdiag, pair_b[p], strlen(pair_b[p]), 0, s2_b);
        float sim_bdiag = trine_s2_compare(s2_a, s2_b, NULL);

        /* Truncate pair label for display */
        char label[64];
        {
            char ta[20], tb[20];
            size_t la = strlen(pair_a[p]);
            size_t lb = strlen(pair_b[p]);
            if (la > 18) {
                memcpy(ta, pair_a[p], 16);
                ta[16] = '.'; ta[17] = '.'; ta[18] = '\0';
            } else {
                memcpy(ta, pair_a[p], la);
                ta[la] = '\0';
            }
            if (lb > 18) {
                memcpy(tb, pair_b[p], 16);
                tb[16] = '.'; tb[17] = '.'; tb[18] = '\0';
            } else {
                memcpy(tb, pair_b[p], lb);
                tb[lb] = '\0';
            }
            snprintf(label, sizeof(label), "%.18s / %.18s", ta, tb);
        }

        printf("  %2d   %-42s  %6.4f  %6.4f  %6.4f  %6.4f\n",
               p + 1, label, sim_s1, sim_id, sim_diag, sim_bdiag);

        sum_s1    += (double)sim_s1;
        sum_id    += (double)sim_id;
        sum_diag  += (double)sim_diag;
        sum_bdiag += (double)sim_bdiag;
    }

    printf("  ---  ------------------------------------------"
           "  ------  ------  ------  ------\n");
    printf("  %-3s  %-42s  %6.4f  %6.4f  %6.4f  %6.4f\n",
           "", "Mean",
           sum_s1 / N_PAIRS, sum_id / N_PAIRS,
           sum_diag / N_PAIRS, sum_bdiag / N_PAIRS);

    printf("\n");

    trine_s2_free(m_identity);
    trine_s2_free(m_diag);
    trine_s2_free(m_bdiag);
}

/* ── Section 4: Memory Footprint ──────────────────────────────────── */

static void bench_memory(void)
{
    printf("===================================================================\n");
    printf("  Section 4: Memory Footprint (weight sizes per mode)\n");
    printf("===================================================================\n\n");

    size_t full_sign  = (size_t)K * DIM * DIM;
    size_t diagonal   = (size_t)K * DIM;
    size_t sparse_sign = (size_t)K * DIM * DIM;  /* same storage, but ~1/3 zero */
    size_t block_diag = (size_t)K * N_CHAINS * CHAIN_DIM * CHAIN_DIM;

    printf("  %-30s  %10s  %10s  %12s\n",
           "Mode", "Weights", "Bytes", "vs Diagonal");
    printf("  ------------------------------"
           "  ----------  ----------  ------------\n");

    printf("  %-30s  %7d x %d  %10zu  %10.1fx\n",
           "Mode 0: Full sign",
           K, DIM * DIM, full_sign,
           (double)full_sign / (double)diagonal);

    printf("  %-30s  %7d x %d  %10zu  %10.1fx\n",
           "Mode 1: Diagonal gate",
           K, DIM, diagonal,
           1.0);

    printf("  %-30s  %7d x %d  %10zu  %10.1fx\n",
           "Mode 2: Sparse sign",
           K, DIM * DIM, sparse_sign,
           (double)sparse_sign / (double)diagonal);

    printf("  %-30s  %5d x %d  %10zu  %10.1fx\n",
           "Mode 3: Block-diagonal",
           K, N_CHAINS * CHAIN_DIM * CHAIN_DIM,
           block_diag,
           (double)block_diag / (double)diagonal);

    printf("\n  Notes:\n");
    printf("    - All weights are uint8_t (1 byte each, ternary {0,1,2})\n");
    printf("    - Mode 2 (sparse sign) has same allocated size as Mode 0,\n");
    printf("      but ~1/3 of entries are zero (absent); effective density varies\n");
    printf("    - Mode 1 (diagonal) uses only W[k][i][i]; rest of 240x240 unused\n");
    printf("    - Mode 3 (block-diagonal) has 4 independent 60x60 blocks per copy\n");
    printf("    - K=%d copies for majority vote in all modes\n", K);
    printf("\n");
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
    printf("\n");
    printf("###################################################################\n");
    printf("#                                                                 #\n");
    printf("#    TRINE Stage-2 — Projection Mode Comparison Benchmark         #\n");
    printf("#                                                                 #\n");
    printf("#    Modes: 0=Full sign, 1=Diagonal, 2=Sparse sign,              #\n");
    printf("#           3=Block-diagonal                                      #\n");
    printf("#                                                                 #\n");
    printf("###################################################################\n\n");

    bench_throughput();
    bench_majority_overhead();
    bench_quality();
    bench_memory();

    printf("###################################################################\n");
    printf("#  Benchmark complete.                                            #\n");
    printf("###################################################################\n\n");

    (void)bench_sink;
    return 0;
}
