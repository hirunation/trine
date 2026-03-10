/* =====================================================================
 * Diagonal Gating Throughput Benchmark
 * =====================================================================
 *
 * Measures Stage-2 encode throughput under four configurations:
 *   1. Stage-1 encode only (baseline)
 *   2. Stage-2 diagonal gating, depth=0 (projection only)
 *   3. Stage-2 diagonal gating, depth=4 (projection + 4 cascade ticks)
 *   4. Stage-2 sign-based matmul, depth=0 (full matmul comparison)
 *
 * Build:
 *   cc -O2 -Wall -Wextra -Isrc/encode -Isrc/compare -Isrc/index \
 *      -Isrc/canon -Isrc/algebra -Isrc/model -Isrc/stage2/projection \
 *      -Isrc/stage2/cascade -Isrc/stage2/inference -Isrc/stage2/hebbian \
 *      -o build/bench_s2_diag bench_s2_diag.c build/libtrine.a -lm
 *
 * ===================================================================== */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "trine_encode.h"
#include "trine_stage1.h"
#include "trine_stage2.h"

/* ── Timing helper ──────────────────────────────────────────────────── */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Prevent dead-code elimination */
static volatile int bench_sink = 0;

/* ── Benchmark runner ──────────────────────────────────────────────── */

static void bench_stage1_encode(const char *text, size_t len, int iters)
{
    uint8_t out[240];

    /* Warmup */
    for (int i = 0; i < iters / 100; i++)
        trine_encode_shingle(text, len, out);

    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        trine_encode_shingle(text, len, out);
    double elapsed = now_sec() - t0;

    bench_sink = (int)out[0];

    double rate = (double)iters / elapsed;
    double us   = (elapsed / (double)iters) * 1e6;
    printf("  Stage-1 encode only         %12.0f encodes/sec  (%6.2f us/encode)\n",
           rate, us);
}

static void bench_s2(const char *label, trine_s2_model_t *model,
                     const char *text, size_t len,
                     uint32_t depth, int iters)
{
    uint8_t out[240];

    /* Warmup */
    for (int i = 0; i < iters / 100; i++)
        trine_s2_encode(model, text, len, depth, out);

    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        trine_s2_encode(model, text, len, depth, out);
    double elapsed = now_sec() - t0;

    bench_sink = (int)out[0];

    double rate = (double)iters / elapsed;
    double us   = (elapsed / (double)iters) * 1e6;
    printf("  %-30s%12.0f encodes/sec  (%6.2f us/encode)\n",
           label, rate, us);
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
    const char *text = "Hello, world!";
    size_t len = strlen(text);
    int N = 100000;

    printf("===============================================================\n");
    printf("  TRINE Stage-2 Diagonal Gating Benchmark\n");
    printf("===============================================================\n");
    printf("  Text: \"%s\" (%zu bytes)\n", text, len);
    printf("  Iterations: %d\n", N);
    printf("  Model: 512 cells, seed=42\n");
    printf("===============================================================\n\n");

    /* 1. Stage-1 baseline */
    bench_stage1_encode(text, len, N);

    /* 2. Diagonal gating, depth=0 */
    {
        trine_s2_model_t *m = trine_s2_create_random(512, 42);
        trine_s2_set_projection_mode(m, TRINE_S2_PROJ_DIAGONAL);
        bench_s2("Diagonal gating (depth=0)", m, text, len, 0, N);
        trine_s2_free(m);
    }

    /* 3. Diagonal gating, depth=4 */
    {
        trine_s2_model_t *m = trine_s2_create_random(512, 42);
        trine_s2_set_projection_mode(m, TRINE_S2_PROJ_DIAGONAL);
        bench_s2("Diagonal gating (depth=4)", m, text, len, 4, N);
        trine_s2_free(m);
    }

    /* 4. Sign-based full matmul, depth=0 (comparison) */
    {
        trine_s2_model_t *m = trine_s2_create_random(512, 42);
        trine_s2_set_projection_mode(m, TRINE_S2_PROJ_SIGN);
        bench_s2("Sign-based matmul (depth=0)", m, text, len, 0, N);
        trine_s2_free(m);
    }

    /* 5. Sign-based full matmul, depth=4 (comparison) */
    {
        trine_s2_model_t *m = trine_s2_create_random(512, 42);
        trine_s2_set_projection_mode(m, TRINE_S2_PROJ_SIGN);
        bench_s2("Sign-based matmul (depth=4)", m, text, len, 4, N);
        trine_s2_free(m);
    }

    printf("\n===============================================================\n");
    printf("  Diagonal gating uses only W[k][i][i] (3x240 lookups)\n");
    printf("  Sign-based matmul uses full W[k][240][240] (3x240x240 MACs)\n");
    printf("===============================================================\n");

    (void)bench_sink;
    return 0;
}
