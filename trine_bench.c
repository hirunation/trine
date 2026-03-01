/* ═══════════════════════════════════════════════════════════════════════
 * TRINE Benchmark Suite v1.0.1
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Comprehensive throughput benchmarks for all TRINE embedding operations:
 *
 *   1. Encode throughput — trine_encode_shingle() at 3 text lengths
 *   2. Compare throughput — raw 240-dim cosine vs lens-weighted cosine
 *   3. IDF compare throughput — trine_idf_cosine() vs raw cosine
 *   4. Index throughput — insert + query at various index sizes
 *   5. End-to-end dedup — encode + insert-or-query pipeline
 *   6. Routed index — insert + query + brute-vs-routed comparison
 *
 * Build:
 *   cc -O2 -Wall -Wextra -o build/trine_bench \
 *      trine_bench.c trine_encode.c trine_stage1.c trine_route.c -lm
 *
 * Usage:
 *   ./build/trine_bench               # full benchmark
 *   ./build/trine_bench --quick       # 10x fewer iterations (CI mode)
 *   ./build/trine_bench --encode-only # only encode benchmarks
 *   ./build/trine_bench --route-only  # only routed index benchmarks
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#define _POSIX_C_SOURCE 200809L  /* clock_gettime, strdup */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "trine_encode.h"
#include "trine_idf.h"
#include "trine_stage1.h"
#include "trine_route.h"
#include "trine_csidf.h"
#include "trine_field.h"

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/*
 * Raw 240-dim cosine similarity over uint8_t trit vectors.
 * Treats {0,1,2} as real values. Returns 0.0 if either vector is zero.
 */
static double cosine_240(const uint8_t *a, const uint8_t *b)
{
    uint64_t dot = 0, ma = 0, mb = 0;
    for (int i = 0; i < 240; i++) {
        uint64_t va = a[i], vb = b[i];
        dot += va * vb;
        ma  += va * va;
        mb  += vb * vb;
    }
    if (ma == 0 || mb == 0) return 0.0;
    double denom = sqrt((double)ma) * sqrt((double)mb);
    return (double)dot / denom;
}

/*
 * Chain cosine over a 60-channel slice.
 */
static double chain_cosine(const uint8_t *a, const uint8_t *b,
                           int off, int w)
{
    uint64_t dot = 0, ma = 0, mb = 0;
    for (int i = off; i < off + w; i++) {
        uint64_t va = a[i], vb = b[i];
        dot += va * vb;
        ma  += va * va;
        mb  += vb * vb;
    }
    if (ma == 0 || mb == 0) return 0.0;
    double denom = sqrt((double)ma) * sqrt((double)mb);
    return (double)dot / denom;
}

/*
 * 4-chain lens-weighted cosine (independent per-chain cosine, weighted sum).
 */
static double lens_cosine(const uint8_t *a, const uint8_t *b,
                          const float w[4])
{
    double sum = 0.0, wsum = 0.0;
    for (int c = 0; c < 4; c++) {
        if (w[c] <= 0.0f) continue;
        double cos_c = chain_cosine(a, b, c * 60, 60);
        sum  += (double)w[c] * cos_c;
        wsum += (double)w[c];
    }
    return wsum > 0.0 ? sum / wsum : 0.0;
}

/*
 * Format a count with comma-separated thousands (e.g. 1,234,567).
 * Uses a rotating set of 4 static buffers so multiple calls can
 * appear in the same printf without clobbering each other.
 */
static const char *fmt_count(long long n)
{
    static char bufs[4][64];
    static int  which = 0;
    char *buf = bufs[which++ & 3];

    char raw[64];
    int len = snprintf(raw, sizeof(raw), "%lld", n < 0 ? -n : n);

    int commas = (len - 1) / 3;
    int out_len = len + commas;
    int neg = (n < 0) ? 1 : 0;

    if (neg) out_len++;

    if (out_len >= 64) out_len = 63;

    buf[out_len] = '\0';

    int src = len - 1;
    int dst = out_len - 1;
    int cnt = 0;

    while (src >= 0 && dst >= neg) {
        buf[dst--] = raw[src--];
        cnt++;
        if (cnt == 3 && src >= 0 && dst >= neg) {
            buf[dst--] = ',';
            cnt = 0;
        }
    }
    if (neg) buf[0] = '-';

    return buf;
}

/*
 * Prevent the compiler from optimizing away a computed result.
 * We write to a volatile global to create a side effect.
 */
static volatile double bench_sink_d;
static volatile float  bench_sink_f;
static volatile int    bench_sink_i;

/* Compare function for qsort on doubles */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test Data
 * ═══════════════════════════════════════════════════════════════════════ */

static const char *TEXT_SHORT  = "hello world";
static const char *TEXT_MEDIUM =
    "The quick brown fox jumps over the lazy dog near the river bank on a warm summer afternoon";
static const char *TEXT_LONG   = NULL;   /* generated at runtime */
static int         TEXT_LONG_LEN = 0;

static const char *TEXT_A = "The quick brown fox jumps over the lazy dog";
static const char *TEXT_B = "The fast brown fox leaps over the tired dog";

/* Generate a ~1000-char paragraph for long-text benchmarks. */
static char *generate_long_text(void)
{
    const char *sentences[] = {
        "The cellular automaton processes each snap in constant time. ",
        "Cascade waves propagate through the toroidal mesh topology. ",
        "Rank monotonicity ensures security isolation between strata. ",
        "The ternary algebra provides seven irreducible cell types. ",
        "Each endomorphism maps the 27-element space to itself deterministically. ",
        "Fleet generation synthesizes billions of snaps on the GPU. ",
        "The golden vector stress benchmark validates correctness at scale. ",
        "Domain phase transitions follow the DTRIT tick sequence exactly. ",
        "Fiber numbers classify the algebraic structure of each mapping. ",
        "The identity particle preserves all input without modification. ",
        "Collapse and split operations create and destroy information channels. ",
        "The oscillator cell produces a deterministic clock signal indefinitely. ",
        "Permutation-rank cells form the computational kernel of the system. ",
        "Two-body composition yields the complete 729-entry multiplication table. ",
        "The forge interface provides eleven operations for algebraic exploration. ",
        "Snap step execution requires zero branches and fits in 113 bytes. ",
    };
    int n_sent = (int)(sizeof(sentences) / sizeof(sentences[0]));

    /* Build ~1000 chars by cycling through sentences */
    int cap = 1200;
    char *buf = (char *)malloc((size_t)cap);
    if (!buf) return NULL;
    int pos = 0;
    int idx = 0;
    while (pos < 1000) {
        int slen = (int)strlen(sentences[idx % n_sent]);
        if (pos + slen >= cap - 1) break;
        memcpy(buf + pos, sentences[idx % n_sent], (size_t)slen);
        pos += slen;
        idx++;
    }
    buf[pos] = '\0';
    return buf;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 1: Encode Throughput
 * ═══════════════════════════════════════════════════════════════════════ */

static void bench_encode(int scale)
{
    printf("\n--- 1. Encode Throughput ---\n");

    struct {
        const char *label;
        const char *text;
        int len;
        int iters;
    } tests[3];

    tests[0].label = "Short text (11 chars)";
    tests[0].text  = TEXT_SHORT;
    tests[0].len   = (int)strlen(TEXT_SHORT);
    tests[0].iters = 1000000 / scale;

    tests[1].label = "Medium text (90 chars)";
    tests[1].text  = TEXT_MEDIUM;
    tests[1].len   = (int)strlen(TEXT_MEDIUM);
    tests[1].iters = 1000000 / scale;

    tests[2].label = "Long text (1000 chars)";
    tests[2].text  = TEXT_LONG;
    tests[2].len   = TEXT_LONG_LEN;
    tests[2].iters = 100000 / scale;

    uint8_t channels[240];

    for (int t = 0; t < 3; t++) {
        /* Warmup: 1% of iterations */
        int warmup = tests[t].iters / 100;
        if (warmup < 10) warmup = 10;
        for (int i = 0; i < warmup; i++) {
            trine_encode_shingle(tests[t].text, (size_t)tests[t].len, channels);
        }

        double t0 = now_sec();
        for (int i = 0; i < tests[t].iters; i++) {
            trine_encode_shingle(tests[t].text, (size_t)tests[t].len, channels);
        }
        double elapsed = now_sec() - t0;

        double per_us = (elapsed / tests[t].iters) * 1e6;
        long long throughput = (long long)(tests[t].iters / elapsed);

        /* Sink result to prevent dead-code elimination */
        bench_sink_i = (int)channels[0];

        printf("  %-24s %15s encodes/sec  (%6.2f us/encode)\n",
               tests[t].label, fmt_count(throughput), per_us);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 2: Compare Throughput
 * ═══════════════════════════════════════════════════════════════════════ */

static void bench_compare(int scale)
{
    printf("\n--- 2. Compare Throughput ---\n");

    /* Pre-encode two texts */
    uint8_t emb_a[240], emb_b[240];
    trine_encode_shingle(TEXT_A, strlen(TEXT_A), emb_a);
    trine_encode_shingle(TEXT_B, strlen(TEXT_B), emb_b);

    int n_raw  = 10000000 / scale;
    int n_lens = 10000000 / scale;

    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        bench_sink_d = cosine_240(emb_a, emb_b);
    }

    /* --- Raw cosine --- */
    {
        double t0 = now_sec();
        double acc = 0.0;
        for (int i = 0; i < n_raw; i++) {
            acc += cosine_240(emb_a, emb_b);
        }
        double elapsed = now_sec() - t0;
        bench_sink_d = acc;

        double per_ns = (elapsed / n_raw) * 1e9;
        long long throughput = (long long)(n_raw / elapsed);

        printf("  Raw cosine (240-dim):    %15s compares/sec  (%.0f ns/compare)\n",
               fmt_count(throughput), per_ns);
    }

    /* --- Lens cosine (4-chain weighted) --- */
    {
        float lens_w[4] = {0.5f, 0.5f, 0.7f, 1.0f};

        /* Warmup */
        for (int i = 0; i < 1000; i++) {
            bench_sink_d = lens_cosine(emb_a, emb_b, lens_w);
        }

        double t0 = now_sec();
        double acc = 0.0;
        for (int i = 0; i < n_lens; i++) {
            acc += lens_cosine(emb_a, emb_b, lens_w);
        }
        double elapsed = now_sec() - t0;
        bench_sink_d = acc;

        double per_ns = (elapsed / n_lens) * 1e9;
        long long throughput = (long long)(n_lens / elapsed);

        printf("  Lens cosine (4-chain):   %15s compares/sec  (%.0f ns/compare)\n",
               fmt_count(throughput), per_ns);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 3: IDF Compare Throughput
 * ═══════════════════════════════════════════════════════════════════════ */

static void bench_idf_compare(int scale)
{
    printf("\n--- 3. IDF Compare Throughput ---\n");

    uint8_t emb_a[240], emb_b[240];
    trine_encode_shingle(TEXT_A, strlen(TEXT_A), emb_a);
    trine_encode_shingle(TEXT_B, strlen(TEXT_B), emb_b);

    int n_idf = 10000000 / scale;

    /* Warmup */
    for (int i = 0; i < 1000; i++) {
        bench_sink_f = trine_idf_cosine(emb_a, emb_b, TRINE_IDF_WEIGHTS);
    }

    double t0 = now_sec();
    float acc = 0.0f;
    for (int i = 0; i < n_idf; i++) {
        acc += trine_idf_cosine(emb_a, emb_b, TRINE_IDF_WEIGHTS);
    }
    double elapsed = now_sec() - t0;
    bench_sink_f = acc;

    double per_ns = (elapsed / n_idf) * 1e9;
    long long throughput = (long long)(n_idf / elapsed);

    printf("  IDF cosine (240-dim):    %15s compares/sec  (%.0f ns/compare)\n",
           fmt_count(throughput), per_ns);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 4: Index Throughput
 * ═══════════════════════════════════════════════════════════════════════ */

static void bench_index(int scale)
{
    printf("\n--- 4. Index Throughput ---\n");

    int n_entries = 10000 / scale;
    if (n_entries < 100) n_entries = 100;  /* Minimum for meaningful results */

    int n_queries = 1000 / scale;
    if (n_queries < 10) n_queries = 10;

    /* Pre-generate embeddings for all entries */
    uint8_t *embeddings = (uint8_t *)malloc((size_t)n_entries * 240);
    if (!embeddings) {
        fprintf(stderr, "  ERROR: allocation failed for %d embeddings\n", n_entries);
        return;
    }

    /* Generate diverse texts and encode them */
    char textbuf[128];
    for (int i = 0; i < n_entries; i++) {
        int tlen = snprintf(textbuf, sizeof(textbuf),
                            "document %d about topic %d with content %d",
                            i, i % 37, i * 7);
        trine_encode_shingle(textbuf, (size_t)tlen, embeddings + (size_t)i * 240);
    }

    /* Encode a query vector */
    uint8_t query[240];
    trine_encode_shingle("document about topic with content", 33, query);

    /* --- Insert throughput --- */
    {
        trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
        trine_s1_index_t *idx = trine_s1_index_create(&config);
        if (!idx) {
            fprintf(stderr, "  ERROR: index creation failed\n");
            free(embeddings);
            return;
        }

        double t0 = now_sec();
        for (int i = 0; i < n_entries; i++) {
            trine_s1_index_add(idx, embeddings + (size_t)i * 240, NULL);
        }
        double elapsed = now_sec() - t0;

        double per_us = (elapsed / n_entries) * 1e6;
        long long throughput = (long long)(n_entries / elapsed);

        printf("  Insert (%s entries):  %15s inserts/sec  (%6.2f us/insert)\n",
               fmt_count((long long)n_entries), fmt_count(throughput), per_us);

        trine_s1_index_free(idx);
    }

    /* --- Query throughput at various sizes --- */
    int sizes[] = {10, 100, 1000, 10000};
    int n_sizes = 4;

    for (int s = 0; s < n_sizes; s++) {
        int size = sizes[s];
        if (size > n_entries) break;

        trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
        trine_s1_index_t *idx = trine_s1_index_create(&config);
        if (!idx) continue;

        /* Fill index to target size */
        for (int i = 0; i < size; i++) {
            trine_s1_index_add(idx, embeddings + (size_t)(i % n_entries) * 240, NULL);
        }

        /* Adjust query count: fewer queries for larger indices */
        int actual_queries = n_queries;
        if (size >= 10000) actual_queries = n_queries / 10;
        if (actual_queries < 5) actual_queries = 5;

        /* Warmup */
        for (int i = 0; i < 3; i++) {
            trine_s1_result_t r = trine_s1_index_query(idx, query);
            bench_sink_i = r.matched_index;
        }

        double t0 = now_sec();
        for (int i = 0; i < actual_queries; i++) {
            trine_s1_result_t r = trine_s1_index_query(idx, query);
            bench_sink_i = r.matched_index;
        }
        double elapsed = now_sec() - t0;

        double per_time = elapsed / actual_queries;
        long long throughput = (long long)(actual_queries / elapsed);

        /* Choose appropriate time unit */
        if (per_time < 1e-3) {
            printf("  Query @ %s entries: %15s queries/sec  (%6.2f us/query)\n",
                   fmt_count((long long)size), fmt_count(throughput),
                   per_time * 1e6);
        } else {
            printf("  Query @ %s entries: %15s queries/sec  (%6.2f ms/query)\n",
                   fmt_count((long long)size), fmt_count(throughput),
                   per_time * 1e3);
        }

        trine_s1_index_free(idx);
    }

    free(embeddings);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 5: End-to-End Dedup Pipeline
 * ═══════════════════════════════════════════════════════════════════════ */

static void bench_dedup(int scale)
{
    printf("\n--- 5. End-to-End Dedup ---\n");

    /* Note: the dedup pipeline is O(N^2) because each text queries a
     * growing linear-scan index. 10K texts is a practical upper bound
     * for benchmark runtime (~10-20 seconds). Use --quick for 1K. */
    int n_texts = 10000 / scale;
    if (n_texts < 100) n_texts = 100;

    /* 70% unique, 30% duplicate (repeat of an earlier text) */
    int n_unique = (n_texts * 70) / 100;

    /* Disable length calibration and use raw cosine threshold.
     * Calibration inflates scores for texts of similar length,
     * which causes false positives among generated random texts.
     * Raw threshold 0.85 reliably catches only exact duplicates
     * (raw cosine ~1.0) while random text pairs sit at ~0.45-0.55. */
    trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
    config.threshold = 0.85f;
    config.calibrate_length = 0;

    trine_s1_index_t *idx = trine_s1_index_create(&config);
    if (!idx) {
        fprintf(stderr, "  ERROR: index creation failed\n");
        return;
    }

    /* Pre-generate all texts */
    char **texts = (char **)malloc((size_t)n_texts * sizeof(char *));
    int  *lens   = (int *)malloc((size_t)n_texts * sizeof(int));
    if (!texts || !lens) {
        fprintf(stderr, "  ERROR: allocation failed\n");
        trine_s1_index_free(idx);
        free(texts);
        free(lens);
        return;
    }

    /* Generate unique texts using a seeded pseudo-random character
     * generator to ensure genuinely distinct content. Each text is
     * a randomized string with no shared template structure, so only
     * explicitly duplicated texts will cross the dedup threshold. */
    char buf[256];
    unsigned int gen_rng = 12345;
    for (int i = 0; i < n_unique; i++) {
        /* Random length between 40 and 120 */
        gen_rng = gen_rng * 1103515245 + 12345;
        int tlen = 40 + (int)(gen_rng % 81);

        /* Fill with pseudo-random lowercase + spaces */
        for (int j = 0; j < tlen; j++) {
            gen_rng = gen_rng * 1103515245 + 12345;
            int r = (int)(gen_rng % 30);
            if (r < 26)
                buf[j] = 'a' + (char)r;
            else
                buf[j] = ' ';
        }
        buf[tlen] = '\0';

        texts[i] = (char *)malloc((size_t)tlen + 1);
        if (texts[i]) {
            memcpy(texts[i], buf, (size_t)tlen + 1);
            lens[i] = tlen;
        } else {
            texts[i] = strdup("fallback");
            lens[i] = 8;
        }
    }

    /* Generate duplicates: repeat earlier texts */
    unsigned int rng_state = 42;
    for (int i = n_unique; i < n_texts; i++) {
        /* Simple LCG for reproducibility */
        rng_state = rng_state * 1103515245 + 12345;
        int src = (int)(rng_state % (unsigned int)n_unique);
        texts[i] = strdup(texts[src]);
        lens[i]  = lens[src];
    }

    /* --- Timed pipeline: encode + insert-or-query --- */
    int unique_found = 0;
    int dup_found = 0;
    uint8_t emb[240];

    double t0 = now_sec();
    for (int i = 0; i < n_texts; i++) {
        /* Encode */
        trine_encode_shingle(texts[i], (size_t)lens[i], emb);

        /* Query existing index for near-duplicate */
        trine_s1_result_t r = trine_s1_index_query(idx, emb);

        if (r.is_duplicate) {
            dup_found++;
        } else {
            /* New text: add to index */
            trine_s1_index_add(idx, emb, NULL);
            unique_found++;
        }
    }
    double elapsed = now_sec() - t0;

    double per_us = (elapsed / n_texts) * 1e6;
    long long throughput = (long long)(n_texts / elapsed);

    printf("  %s texts (70%% unique): %15s texts/sec  (%6.1f us/text)\n",
           fmt_count((long long)n_texts), fmt_count(throughput), per_us);
    printf("  Unique found: %d, Duplicates: %d\n", unique_found, dup_found);

    /* Cleanup */
    for (int i = 0; i < n_texts; i++)
        free(texts[i]);
    free(texts);
    free(lens);
    trine_s1_index_free(idx);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 6: Routed Index
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Fill a 240-byte embedding with pseudo-random trits using an LCG.
 * Returns the updated seed.
 */
static unsigned int fill_random_emb(uint8_t emb[240], unsigned int seed)
{
    for (int i = 0; i < 240; i++) {
        seed = seed * 1103515245 + 12345;
        emb[i] = (uint8_t)((seed >> 16) % 3);
    }
    return seed;
}

static void bench_routed(int scale)
{
    printf("\n--- 6. Routed Index ---\n");

    /* Sizes: full = 1K, 10K, 100K; quick = 100, 1K, 10K */
    int sizes[3];
    if (scale > 1) {
        sizes[0] = 100;
        sizes[1] = 1000;
        sizes[2] = 10000;
    } else {
        sizes[0] = 1000;
        sizes[1] = 10000;
        sizes[2] = 100000;
    }

    int n_queries = (scale > 1) ? 100 : 1000;

    /* --- 6a. Routed insert throughput --- */
    printf("\n  [Insert throughput]\n");

    for (int s = 0; s < 3; s++) {
        int n = sizes[s];

        trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
        trine_route_t *rt = trine_route_create(&config);
        if (!rt) {
            fprintf(stderr, "  ERROR: route index creation failed\n");
            return;
        }

        uint8_t emb[240];
        unsigned int seed = 77777 + (unsigned int)s;

        /* Warmup: insert 10 entries */
        for (int i = 0; i < 10; i++) {
            seed = fill_random_emb(emb, seed);
            trine_route_add(rt, emb, NULL);
        }
        trine_route_free(rt);

        /* Fresh index for timed run */
        rt = trine_route_create(&config);
        if (!rt) {
            fprintf(stderr, "  ERROR: route index creation failed\n");
            return;
        }

        seed = 99999 + (unsigned int)s;
        double t0 = now_sec();
        for (int i = 0; i < n; i++) {
            seed = fill_random_emb(emb, seed);
            trine_route_add(rt, emb, NULL);
        }
        double elapsed = now_sec() - t0;

        double per_us = (elapsed / n) * 1e6;
        long long throughput = (long long)(n / elapsed);

        printf("    %7s entries:  %15s inserts/sec  (%6.2f us/insert)\n",
               fmt_count((long long)n), fmt_count(throughput), per_us);

        trine_route_free(rt);
    }

    /* --- 6b. Routed query throughput --- */
    printf("\n  [Query throughput]\n");

    for (int s = 0; s < 3; s++) {
        int n = sizes[s];

        trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
        trine_route_t *rt = trine_route_create(&config);
        if (!rt) {
            fprintf(stderr, "  ERROR: route index creation failed\n");
            return;
        }

        /* Populate index */
        uint8_t emb[240];
        unsigned int seed = 12345 + (unsigned int)s;
        for (int i = 0; i < n; i++) {
            seed = fill_random_emb(emb, seed);
            trine_route_add(rt, emb, NULL);
        }

        /* Generate query embeddings */
        uint8_t *queries = (uint8_t *)malloc((size_t)n_queries * 240);
        if (!queries) {
            fprintf(stderr, "  ERROR: query allocation failed\n");
            trine_route_free(rt);
            return;
        }
        unsigned int qseed = 54321 + (unsigned int)s;
        for (int i = 0; i < n_queries; i++) {
            qseed = fill_random_emb(queries + (size_t)i * 240, qseed);
        }

        /* Warmup */
        for (int i = 0; i < 3 && i < n_queries; i++) {
            trine_route_stats_t stats;
            trine_s1_result_t r = trine_route_query(rt, queries + (size_t)i * 240, &stats);
            bench_sink_i = r.matched_index;
        }

        /* Allocate per-query latency array */
        double *latencies = (double *)calloc((size_t)n_queries, sizeof(double));
        if (!latencies) {
            fprintf(stderr, "  ERROR: latency allocation failed\n");
            free(queries);
            trine_route_free(rt);
            return;
        }

        /* Timed queries — measure each individually */
        double total_candidates = 0.0;
        double total_speedup = 0.0;

        for (int i = 0; i < n_queries; i++) {
            trine_route_stats_t stats;
            double qstart = now_sec();
            trine_s1_result_t r = trine_route_query(rt, queries + (size_t)i * 240, &stats);
            latencies[i] = now_sec() - qstart;
            bench_sink_i = r.matched_index;
            total_candidates += stats.candidates_checked;
            total_speedup += (double)stats.speedup;
        }

        /* Compute aggregate timing from individual measurements */
        double elapsed = 0.0;
        for (int i = 0; i < n_queries; i++) elapsed += latencies[i];

        double per_us = (elapsed / n_queries) * 1e6;
        long long throughput = (long long)(n_queries / elapsed);
        double avg_candidates = total_candidates / n_queries;
        double avg_speedup = total_speedup / n_queries;

        /* Sort latencies for percentile calculation */
        qsort(latencies, (size_t)n_queries, sizeof(double), cmp_double);
        double p50 = latencies[n_queries / 2] * 1e6;
        double p95 = latencies[(int)((double)n_queries * 0.95)] * 1e6;
        double p99 = latencies[(int)((double)n_queries * 0.99)] * 1e6;

        printf("    %7s entries:  %15s queries/sec  (%6.2f us/query)"
               "  avg_cand=%.0f  speedup=%.1fx\n",
               fmt_count((long long)n), fmt_count(throughput), per_us,
               avg_candidates, avg_speedup);
        printf("    %7s           p50=%.1fus  p95=%.1fus  p99=%.1fus\n",
               "", p50, p95, p99);

        free(latencies);

        free(queries);
        trine_route_free(rt);
    }

    /* --- 6c. Brute-force vs Routed comparison at 10K entries --- */
    printf("\n  [Brute-force vs Routed @ 10K entries]\n");

    {
        int n_compare = (scale > 1) ? 10000 : 10000;
        int n_cmp_queries = (scale > 1) ? 100 : 100;

        trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;

        /* Build brute-force index */
        trine_s1_index_t *brute = trine_s1_index_create(&config);
        /* Build routed index */
        trine_route_t *rt = trine_route_create(&config);

        if (!brute || !rt) {
            fprintf(stderr, "  ERROR: index creation failed\n");
            if (brute) trine_s1_index_free(brute);
            if (rt) trine_route_free(rt);
            return;
        }

        /* Populate both indices with identical data */
        uint8_t emb[240];
        unsigned int seed = 314159;
        for (int i = 0; i < n_compare; i++) {
            seed = fill_random_emb(emb, seed);
            trine_s1_index_add(brute, emb, NULL);
            trine_route_add(rt, emb, NULL);
        }

        /* Generate query embeddings */
        uint8_t *queries = (uint8_t *)malloc((size_t)n_cmp_queries * 240);
        if (!queries) {
            fprintf(stderr, "  ERROR: query allocation failed\n");
            trine_s1_index_free(brute);
            trine_route_free(rt);
            return;
        }
        unsigned int qseed = 271828;
        for (int i = 0; i < n_cmp_queries; i++) {
            qseed = fill_random_emb(queries + (size_t)i * 240, qseed);
        }

        /* Warmup both */
        for (int i = 0; i < 3 && i < n_cmp_queries; i++) {
            trine_s1_result_t r1 = trine_s1_index_query(brute, queries + (size_t)i * 240);
            bench_sink_i = r1.matched_index;
            trine_route_stats_t stats;
            trine_s1_result_t r2 = trine_route_query(rt, queries + (size_t)i * 240, &stats);
            bench_sink_i = r2.matched_index;
        }

        /* Allocate per-query latency arrays */
        double *brute_lat = (double *)calloc((size_t)n_cmp_queries, sizeof(double));
        double *route_lat = (double *)calloc((size_t)n_cmp_queries, sizeof(double));
        if (!brute_lat || !route_lat) {
            fprintf(stderr, "  ERROR: latency allocation failed\n");
            free(brute_lat);
            free(route_lat);
            free(queries);
            trine_s1_index_free(brute);
            trine_route_free(rt);
            return;
        }

        /* Brute-force timed — measure each query individually */
        for (int i = 0; i < n_cmp_queries; i++) {
            double qstart = now_sec();
            trine_s1_result_t r = trine_s1_index_query(brute, queries + (size_t)i * 240);
            brute_lat[i] = now_sec() - qstart;
            bench_sink_i = r.matched_index;
        }

        /* Routed timed — measure each query individually */
        for (int i = 0; i < n_cmp_queries; i++) {
            trine_route_stats_t stats;
            double qstart = now_sec();
            trine_s1_result_t r = trine_route_query(rt, queries + (size_t)i * 240, &stats);
            route_lat[i] = now_sec() - qstart;
            bench_sink_i = r.matched_index;
        }

        /* Compute aggregate times */
        double elapsed_brute = 0.0, elapsed_route = 0.0;
        for (int i = 0; i < n_cmp_queries; i++) {
            elapsed_brute += brute_lat[i];
            elapsed_route += route_lat[i];
        }

        long long brute_qps = (long long)(n_cmp_queries / elapsed_brute);
        long long route_qps = (long long)(n_cmp_queries / elapsed_route);
        double speedup_ratio = elapsed_brute / elapsed_route;

        /* Sort latencies for percentiles */
        qsort(brute_lat, (size_t)n_cmp_queries, sizeof(double), cmp_double);
        qsort(route_lat, (size_t)n_cmp_queries, sizeof(double), cmp_double);

        double brute_p50 = brute_lat[n_cmp_queries / 2] * 1e6;
        double brute_p95 = brute_lat[(int)((double)n_cmp_queries * 0.95)] * 1e6;
        double brute_p99 = brute_lat[(int)((double)n_cmp_queries * 0.99)] * 1e6;
        double route_p50 = route_lat[n_cmp_queries / 2] * 1e6;
        double route_p95 = route_lat[(int)((double)n_cmp_queries * 0.95)] * 1e6;
        double route_p99 = route_lat[(int)((double)n_cmp_queries * 0.99)] * 1e6;

        printf("    Brute-force:  %15s queries/sec  (%6.2f us/query)\n",
               fmt_count(brute_qps), (elapsed_brute / n_cmp_queries) * 1e6);
        printf("                  p50=%.1fus  p95=%.1fus  p99=%.1fus\n",
               brute_p50, brute_p95, brute_p99);
        printf("    Routed:       %15s queries/sec  (%6.2f us/query)\n",
               fmt_count(route_qps), (elapsed_route / n_cmp_queries) * 1e6);
        printf("                  p50=%.1fus  p95=%.1fus  p99=%.1fus\n",
               route_p50, route_p95, route_p99);
        printf("    Speedup:      %.2fx\n", speedup_ratio);

        free(brute_lat);
        free(route_lat);

        free(queries);
        trine_s1_index_free(brute);
        trine_route_free(rt);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 7: CS-IDF Compare Throughput (Phase 4)
 * ═══════════════════════════════════════════════════════════════════════ */

static void bench_csidf_compare(int scale)
{
    printf("\n--- 7. CS-IDF Compare Throughput ---\n");

    /* Build a small corpus to get realistic IDF weights */
    trine_csidf_t csidf;
    trine_csidf_init(&csidf);

    const char *corpus[] = {
        "the quick brown fox jumped over the lazy dog",
        "a completely unique and special phrase here today",
        "mathematical algorithms for sorting data efficiently",
        "quantum computing research paper abstract summary",
        "the old man and the sea by hemingway chapter one",
        "application crashes when user clicks login button",
        "implement dark mode feature for settings page",
        "fix null pointer dereference in database handler"
    };
    uint8_t corpus_emb[8][240];
    for (int d = 0; d < 8; d++) {
        trine_encode_shingle(corpus[d], strlen(corpus[d]), corpus_emb[d]);
        trine_csidf_observe(&csidf, corpus_emb[d]);
    }
    trine_csidf_compute(&csidf);

    uint8_t emb_a[240], emb_b[240];
    trine_encode_shingle(TEXT_A, strlen(TEXT_A), emb_a);
    trine_encode_shingle(TEXT_B, strlen(TEXT_B), emb_b);

    int n_iters = 5000000 / scale;

    /* --- CS-IDF cosine (flat 240-dim) --- */
    {
        for (int i = 0; i < 1000; i++)
            bench_sink_f = trine_csidf_cosine(emb_a, emb_b, &csidf);

        double t0 = now_sec();
        float acc = 0.0f;
        for (int i = 0; i < n_iters; i++) {
            acc += trine_csidf_cosine(emb_a, emb_b, &csidf);
        }
        double elapsed = now_sec() - t0;
        bench_sink_f = acc;

        double per_ns = (elapsed / n_iters) * 1e9;
        long long throughput = (long long)(n_iters / elapsed);

        printf("  CS-IDF cosine (240-dim):  %15s compares/sec  (%.0f ns/compare)\n",
               fmt_count(throughput), per_ns);
    }

    /* --- CS-IDF + lens cosine --- */
    {
        float lens[4] = {0.5f, 0.5f, 0.7f, 1.0f};

        for (int i = 0; i < 1000; i++)
            bench_sink_f = trine_csidf_cosine_lens(emb_a, emb_b, &csidf, lens);

        double t0 = now_sec();
        float acc = 0.0f;
        for (int i = 0; i < n_iters; i++) {
            acc += trine_csidf_cosine_lens(emb_a, emb_b, &csidf, lens);
        }
        double elapsed = now_sec() - t0;
        bench_sink_f = acc;

        double per_ns = (elapsed / n_iters) * 1e9;
        long long throughput = (long long)(n_iters / elapsed);

        printf("  CS-IDF+lens cosine:      %15s compares/sec  (%.0f ns/compare)\n",
               fmt_count(throughput), per_ns);
    }

    /* --- Static IDF cosine for comparison --- */
    {
        for (int i = 0; i < 1000; i++)
            bench_sink_f = trine_idf_cosine(emb_a, emb_b, TRINE_IDF_WEIGHTS);

        double t0 = now_sec();
        float acc = 0.0f;
        for (int i = 0; i < n_iters; i++) {
            acc += trine_idf_cosine(emb_a, emb_b, TRINE_IDF_WEIGHTS);
        }
        double elapsed = now_sec() - t0;
        bench_sink_f = acc;

        double per_ns = (elapsed / n_iters) * 1e9;
        long long throughput = (long long)(n_iters / elapsed);

        printf("  Static IDF (reference):  %15s compares/sec  (%.0f ns/compare)\n",
               fmt_count(throughput), per_ns);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 8: Field-Aware Scoring Throughput (Phase 4)
 * ═══════════════════════════════════════════════════════════════════════ */

static void bench_field_scoring(int scale)
{
    printf("\n--- 8. Field-Aware Scoring Throughput ---\n");

    trine_field_config_t cfg;
    trine_field_config_init(&cfg);
    cfg.field_count = 2;  /* title + body */

    /* Build two field entries */
    trine_field_entry_t a, b;
    const char *a_texts[] = {"Bug Report Critical Issue",
                             "Application crashes when user clicks the login button on main page"};
    size_t a_lens[] = {25, 65};
    const char *b_texts[] = {"Bug Report Minor Issue",
                             "Application hangs when user clicks the settings button on main page"};
    size_t b_lens[] = {22, 67};
    trine_field_encode(&cfg, a_texts, a_lens, &a);
    trine_field_encode(&cfg, b_texts, b_lens, &b);

    int n_iters = 5000000 / scale;

    /* --- Field-weighted cosine --- */
    {
        for (int i = 0; i < 1000; i++)
            bench_sink_f = trine_field_cosine(&a, &b, &cfg);

        double t0 = now_sec();
        float acc = 0.0f;
        for (int i = 0; i < n_iters; i++) {
            acc += trine_field_cosine(&a, &b, &cfg);
        }
        double elapsed = now_sec() - t0;
        bench_sink_f = acc;

        double per_ns = (elapsed / n_iters) * 1e9;
        long long throughput = (long long)(n_iters / elapsed);

        printf("  Field cosine (2 fields): %15s compares/sec  (%.0f ns/compare)\n",
               fmt_count(throughput), per_ns);
    }

    /* --- Field cosine with IDF weights --- */
    {
        float idf[240];
        for (int i = 0; i < 240; i++) idf[i] = 1.0f - 0.5f * (float)i / 240.0f;

        for (int i = 0; i < 1000; i++)
            bench_sink_f = trine_field_cosine_idf(&a, &b, &cfg, idf);

        double t0 = now_sec();
        float acc = 0.0f;
        for (int i = 0; i < n_iters; i++) {
            acc += trine_field_cosine_idf(&a, &b, &cfg, idf);
        }
        double elapsed = now_sec() - t0;
        bench_sink_f = acc;

        double per_ns = (elapsed / n_iters) * 1e9;
        long long throughput = (long long)(n_iters / elapsed);

        printf("  Field+IDF (2 fields):    %15s compares/sec  (%.0f ns/compare)\n",
               fmt_count(throughput), per_ns);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 9: CS-IDF Noise Floor (Phase 4)
 * ═══════════════════════════════════════════════════════════════════════ */

static void bench_csidf_noise_floor(int scale)
{
    printf("\n--- 9. CS-IDF Noise Floor ---\n");

    int n_docs = 200 / scale;
    if (n_docs < 20) n_docs = 20;
    int n_pairs = 500 / scale;
    if (n_pairs < 50) n_pairs = 50;

    /* Generate diverse documents with pseudo-random content */
    uint8_t *embs = (uint8_t *)malloc((size_t)n_docs * 240);
    if (!embs) {
        fprintf(stderr, "  ERROR: allocation failed\n");
        return;
    }

    trine_csidf_t csidf;
    trine_csidf_init(&csidf);

    char textbuf[256];
    unsigned int rng = 7654321;
    for (int i = 0; i < n_docs; i++) {
        /* Generate genuinely different text per doc */
        int tlen = 60 + (int)(rng % 61);
        rng = rng * 1103515245 + 12345;
        for (int j = 0; j < tlen; j++) {
            rng = rng * 1103515245 + 12345;
            int r = (int)(rng % 30);
            textbuf[j] = (r < 26) ? ('a' + (char)r) : ' ';
        }
        textbuf[tlen] = '\0';

        trine_encode_shingle(textbuf, (size_t)tlen, embs + (size_t)i * 240);
        trine_csidf_observe(&csidf, embs + (size_t)i * 240);
    }
    trine_csidf_compute(&csidf);

    /* Measure pairwise similarities: uniform vs CS-IDF */
    double sum_uni = 0.0, sum_idf = 0.0;
    double max_uni = 0.0, max_idf = 0.0;
    double min_uni = 2.0, min_idf = 2.0;

    rng = 1234567;
    for (int p = 0; p < n_pairs; p++) {
        rng = rng * 1103515245 + 12345;
        int i = (int)(rng % (unsigned int)n_docs);
        rng = rng * 1103515245 + 12345;
        int j = (int)(rng % (unsigned int)n_docs);
        if (i == j) { j = (j + 1) % n_docs; }

        const uint8_t *a = embs + (size_t)i * 240;
        const uint8_t *b = embs + (size_t)j * 240;

        double uni = cosine_240(a, b);
        double idf = (double)trine_csidf_cosine(a, b, &csidf);

        sum_uni += uni;
        sum_idf += idf;
        if (uni > max_uni) max_uni = uni;
        if (idf > max_idf) max_idf = idf;
        if (uni < min_uni) min_uni = uni;
        if (idf < min_idf) min_idf = idf;
    }

    double avg_uni = sum_uni / n_pairs;
    double avg_idf = sum_idf / n_pairs;

    printf("  Disjoint pairs: %d docs, %d pairs\n", n_docs, n_pairs);
    printf("  Uniform cosine:  avg=%.4f  min=%.4f  max=%.4f\n",
           avg_uni, min_uni, max_uni);
    printf("  CS-IDF cosine:   avg=%.4f  min=%.4f  max=%.4f\n",
           avg_idf, min_idf, max_idf);
    printf("  Noise reduction: %.1f%% (lower = better separation)\n",
           (1.0 - avg_idf / avg_uni) * 100.0);

    /* Count CS-IDF downweighted channels (weight < 0.5) */
    int downweighted = 0;
    for (int i = 0; i < 240; i++) {
        if (csidf.weights[i] < 0.5f) downweighted++;
    }
    printf("  Downweighted channels: %d/240 (weight < 0.5)\n", downweighted);

    free(embs);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Benchmark 10: CS-IDF Routed Query Latency (Phase 4)
 * ═══════════════════════════════════════════════════════════════════════ */

static void bench_csidf_routed(int scale)
{
    printf("\n--- 10. CS-IDF Routed Query Latency ---\n");

    int n_entries = (scale > 1) ? 1000 : 5000;
    int n_queries = (scale > 1) ? 100 : 500;

    trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
    trine_route_t *rt = trine_route_create(&config);
    if (!rt) {
        fprintf(stderr, "  ERROR: route index creation failed\n");
        return;
    }

    trine_route_enable_csidf(rt);

    /* Populate index */
    uint8_t emb[240];
    char textbuf[128];
    for (int i = 0; i < n_entries; i++) {
        int tlen = snprintf(textbuf, sizeof(textbuf),
                            "document %d about topic %d with content %d variation %d",
                            i, i % 37, i * 7, i % 13);
        trine_encode_shingle(textbuf, (size_t)tlen, emb);
        trine_route_add(rt, emb, NULL);
    }
    trine_route_compute_csidf(rt);

    /* Generate query embeddings */
    uint8_t *queries = (uint8_t *)malloc((size_t)n_queries * 240);
    if (!queries) {
        fprintf(stderr, "  ERROR: query allocation failed\n");
        trine_route_free(rt);
        return;
    }
    for (int i = 0; i < n_queries; i++) {
        int tlen = snprintf(textbuf, sizeof(textbuf),
                            "query about topic %d and content %d",
                            i % 37, i * 11);
        trine_encode_shingle(textbuf, (size_t)tlen, queries + (size_t)i * 240);
    }

    /* Allocate latency arrays */
    double *std_lat = (double *)calloc((size_t)n_queries, sizeof(double));
    double *idf_lat = (double *)calloc((size_t)n_queries, sizeof(double));
    if (!std_lat || !idf_lat) {
        fprintf(stderr, "  ERROR: latency allocation failed\n");
        free(std_lat); free(idf_lat); free(queries);
        trine_route_free(rt);
        return;
    }

    /* Warmup */
    for (int i = 0; i < 3 && i < n_queries; i++) {
        trine_route_stats_t stats;
        trine_s1_result_t r = trine_route_query(rt, queries + (size_t)i * 240, &stats);
        bench_sink_i = r.matched_index;
        r = trine_route_query_csidf(rt, queries + (size_t)i * 240, &stats);
        bench_sink_i = r.matched_index;
    }

    /* Standard query */
    for (int i = 0; i < n_queries; i++) {
        double qstart = now_sec();
        trine_s1_result_t r = trine_route_query(rt, queries + (size_t)i * 240, NULL);
        std_lat[i] = now_sec() - qstart;
        bench_sink_i = r.matched_index;
    }

    /* CS-IDF query */
    double total_cand = 0.0;
    for (int i = 0; i < n_queries; i++) {
        trine_route_stats_t stats;
        double qstart = now_sec();
        trine_s1_result_t r = trine_route_query_csidf(rt, queries + (size_t)i * 240, &stats);
        idf_lat[i] = now_sec() - qstart;
        bench_sink_i = r.matched_index;
        total_cand += stats.candidates_checked;
    }

    /* Compute aggregate stats */
    double elapsed_std = 0.0, elapsed_idf = 0.0;
    for (int i = 0; i < n_queries; i++) {
        elapsed_std += std_lat[i];
        elapsed_idf += idf_lat[i];
    }

    /* Sort for percentiles */
    qsort(std_lat, (size_t)n_queries, sizeof(double), cmp_double);
    qsort(idf_lat, (size_t)n_queries, sizeof(double), cmp_double);

    double std_p50 = std_lat[n_queries / 2] * 1e6;
    double std_p95 = std_lat[(int)((double)n_queries * 0.95)] * 1e6;
    double std_p99 = std_lat[(int)((double)n_queries * 0.99)] * 1e6;
    double idf_p50 = idf_lat[n_queries / 2] * 1e6;
    double idf_p95 = idf_lat[(int)((double)n_queries * 0.95)] * 1e6;
    double idf_p99 = idf_lat[(int)((double)n_queries * 0.99)] * 1e6;

    printf("  Index: %s entries, %d queries\n",
           fmt_count((long long)n_entries), n_queries);
    printf("  Standard query:  p50=%.1fus  p95=%.1fus  p99=%.1fus  avg=%.1fus\n",
           std_p50, std_p95, std_p99, (elapsed_std / n_queries) * 1e6);
    printf("  CS-IDF query:    p50=%.1fus  p95=%.1fus  p99=%.1fus  avg=%.1fus\n",
           idf_p50, idf_p95, idf_p99, (elapsed_idf / n_queries) * 1e6);
    printf("  Overhead:        %.1f%%\n",
           ((elapsed_idf / elapsed_std) - 1.0) * 100.0);
    printf("  Avg candidates:  %.1f\n", total_cand / n_queries);

    free(std_lat);
    free(idf_lat);
    free(queries);
    trine_route_free(rt);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    int quick = 0;
    int encode_only = 0;
    int route_only = 0;
    int phase4_only = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) {
            quick = 1;
        } else if (strcmp(argv[i], "--encode-only") == 0) {
            encode_only = 1;
        } else if (strcmp(argv[i], "--route-only") == 0) {
            route_only = 1;
        } else if (strcmp(argv[i], "--phase4-only") == 0) {
            phase4_only = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [--quick] [--encode-only] [--route-only] [--phase4-only]\n", argv[0]);
            printf("  --quick        Run 10x fewer iterations (fast CI mode)\n");
            printf("  --encode-only  Only run encode throughput benchmarks\n");
            printf("  --route-only   Only run routed index benchmarks\n");
            printf("  --phase4-only  Only run Phase 4 benchmarks (CS-IDF, fields)\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    int scale = quick ? 10 : 1;

    /* Generate long text */
    char *long_text = generate_long_text();
    if (!long_text) {
        fprintf(stderr, "ERROR: failed to generate long text\n");
        return 1;
    }
    TEXT_LONG = long_text;
    TEXT_LONG_LEN = (int)strlen(long_text);

    printf("\n");
    printf("===============================================================\n");
    printf("  TRINE Benchmark Suite v1.0.1%s\n",
           quick ? "  [QUICK MODE: 10x fewer iterations]" : "");
    printf("===============================================================\n");

    if (phase4_only) {
        bench_csidf_compare(scale);
        bench_field_scoring(scale);
        bench_csidf_noise_floor(scale);
        bench_csidf_routed(scale);
    } else if (route_only) {
        bench_routed(scale);
    } else {
        bench_encode(scale);

        if (!encode_only) {
            bench_compare(scale);
            bench_idf_compare(scale);
            bench_index(scale);
            bench_dedup(scale);
            bench_routed(scale);
            bench_csidf_compare(scale);
            bench_field_scoring(scale);
            bench_csidf_noise_floor(scale);
            bench_csidf_routed(scale);
        }
    }

    printf("\n");
    printf("===============================================================\n");
    printf("\n");

    free(long_text);
    return 0;
}
