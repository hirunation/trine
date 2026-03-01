/* ═══════════════════════════════════════════════════════════════════════
 * TRINE Routing Recall Validation v1.0.1
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Empirically measures recall and precision of the Band-LSH routing
 * layer (trine_route) against the brute-force baseline (trine_s1_index).
 *
 * Methodology:
 *   1. Generate a synthetic corpus of N random short texts
 *   2. Create near-duplicate pairs by word mutation (swap, synonym,
 *      add/remove, reorder)
 *   3. Insert all texts into both brute-force and routed indexes
 *   4. Query each near-duplicate against both indexes
 *   5. Compare results to measure recall, false negatives, and speedup
 *
 * Build:
 *   cc -O2 -Wall -Wextra -o build/trine_recall \
 *      trine_recall.c trine_encode.c trine_stage1.c trine_route.c -lm
 *
 * Usage:
 *   ./build/trine_recall                   # default (1000 entries, 100 queries)
 *   ./build/trine_recall --quick           # small corpus (100 entries, 20 queries)
 *   ./build/trine_recall --size 10000      # large corpus
 *   ./build/trine_recall --seed 99         # custom random seed
 *   ./build/trine_recall --validate        # pass/warn/fail verdict
 *   ./build/trine_recall --validate --json # machine-readable validate output
 *   ./build/trine_recall --health          # bucket health telemetry
 *   ./build/trine_recall --health --json   # machine-readable health output
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#define _POSIX_C_SOURCE 200809L  /* clock_gettime */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "trine_encode.h"
#include "trine_stage1.h"
#include "trine_route.h"

/* ═══════════════════════════════════════════════════════════════════════
 * Self-contained LCG PRNG (no rand()/srand())
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint64_t state;
} recall_rng_t;

static void rng_seed(recall_rng_t *rng, uint64_t seed)
{
    rng->state = seed;
}

static uint32_t rng_next(recall_rng_t *rng)
{
    rng->state = rng->state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng->state >> 33);
}

/* Return a random integer in [0, bound). */
static int rng_range(recall_rng_t *rng, int bound)
{
    if (bound <= 0) return 0;
    return (int)(rng_next(rng) % (uint32_t)bound);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Timing
 * ═══════════════════════════════════════════════════════════════════════ */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Vocabulary and Text Generation
 * ═══════════════════════════════════════════════════════════════════════ */

static const char *VOCAB[] = {
    "the",    "quick",   "brown",   "fox",     "jumps",
    "over",   "lazy",    "dog",     "near",    "river",
    "bank",   "under",   "bright",  "sun",     "warm",
    "summer", "cold",    "winter",  "red",     "blue",
    "green",  "small",   "large",   "fast",    "slow",
    "old",    "new",     "dark",    "light",   "deep",
    "high",   "low",     "long",    "short",   "wide",
    "open",   "close",   "first",   "last",    "next",
    "good",   "bad",     "hot",     "cool",    "soft",
    "hard",   "dry",     "wet",     "flat",    "round",
};
#define VOCAB_SIZE ((int)(sizeof(VOCAB) / sizeof(VOCAB[0])))

/* Synonym pairs for near-duplicate mutation. */
static const struct { int from; int to; } SYNONYMS[] = {
    {  1, 23 },  /* quick  -> fast  */
    { 23,  1 },  /* fast   -> quick */
    {  6, 24 },  /* lazy   -> slow  */
    { 24,  6 },  /* slow   -> lazy  */
    { 11, 29 },  /* under  -> deep  */
    { 18, 27 },  /* red    -> dark  */
    { 21, 22 },  /* small  -> large */
    { 22, 21 },  /* large  -> small */
    { 25, 26 },  /* old    -> new   */
    { 26, 25 },  /* new    -> old   */
    { 40, 41 },  /* good   -> bad   */
    { 41, 40 },  /* bad    -> good  */
    { 42, 43 },  /* hot    -> cool  */
    { 43, 42 },  /* cool   -> hot   */
};
#define SYNONYM_COUNT ((int)(sizeof(SYNONYMS) / sizeof(SYNONYMS[0])))

/* Generate a random text of word_count words. Writes to buf (must hold
 * at least word_count * 8 bytes). Returns length written. */
static int generate_text(recall_rng_t *rng, int word_count, char *buf,
                          int *word_indices, int max_words)
{
    int pos = 0;
    int wc = word_count;
    if (wc > max_words) wc = max_words;

    for (int i = 0; i < wc; i++) {
        int idx = rng_range(rng, VOCAB_SIZE);
        word_indices[i] = idx;
        const char *w = VOCAB[idx];
        int wlen = (int)strlen(w);
        if (i > 0) buf[pos++] = ' ';
        memcpy(buf + pos, w, (size_t)wlen);
        pos += wlen;
    }
    buf[pos] = '\0';
    return pos;
}

/* Reconstruct text from word indices. */
static int rebuild_text(const int *word_indices, int word_count, char *buf)
{
    int pos = 0;
    for (int i = 0; i < word_count; i++) {
        const char *w = VOCAB[word_indices[i]];
        int wlen = (int)strlen(w);
        if (i > 0) buf[pos++] = ' ';
        memcpy(buf + pos, w, (size_t)wlen);
        pos += wlen;
    }
    buf[pos] = '\0';
    return pos;
}

/* Mutation type for near-duplicate generation. */
enum mutation_type {
    MUT_SWAP,       /* Swap two adjacent words */
    MUT_SYNONYM,    /* Replace a word with a synonym */
    MUT_ADD,        /* Insert a random word */
    MUT_REMOVE,     /* Remove a word */
    MUT_COUNT
};

/* Create a near-duplicate by mutating word_indices.
 * Writes the mutated text to buf. Returns length. */
static int mutate_text(recall_rng_t *rng, const int *src_indices,
                        int src_count, char *buf, int *dst_indices,
                        int *dst_count, int max_words)
{
    int mutation = rng_range(rng, MUT_COUNT);
    int n = src_count;

    /* Copy source indices */
    memcpy(dst_indices, src_indices, (size_t)n * sizeof(int));

    switch (mutation) {
    case MUT_SWAP:
        if (n >= 2) {
            int pos = rng_range(rng, n - 1);
            int tmp = dst_indices[pos];
            dst_indices[pos] = dst_indices[pos + 1];
            dst_indices[pos + 1] = tmp;
        }
        break;

    case MUT_SYNONYM: {
        /* Find a word in the text that has a synonym */
        int attempts = 0;
        int replaced = 0;
        while (attempts < 10 && !replaced) {
            int pos = rng_range(rng, n);
            for (int s = 0; s < SYNONYM_COUNT; s++) {
                if (SYNONYMS[s].from == dst_indices[pos]) {
                    dst_indices[pos] = SYNONYMS[s].to;
                    replaced = 1;
                    break;
                }
            }
            attempts++;
        }
        if (!replaced) {
            /* Fallback: replace a random word */
            int pos = rng_range(rng, n);
            dst_indices[pos] = rng_range(rng, VOCAB_SIZE);
        }
        break;
    }

    case MUT_ADD:
        if (n < max_words - 1) {
            int pos = rng_range(rng, n + 1);
            /* Shift elements right */
            for (int i = n; i > pos; i--)
                dst_indices[i] = dst_indices[i - 1];
            dst_indices[pos] = rng_range(rng, VOCAB_SIZE);
            n++;
        }
        break;

    case MUT_REMOVE:
        if (n > 2) {
            int pos = rng_range(rng, n);
            /* Shift elements left */
            for (int i = pos; i < n - 1; i++)
                dst_indices[i] = dst_indices[i + 1];
            n--;
        }
        break;
    }

    *dst_count = n;
    return rebuild_text(dst_indices, n, buf);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Corpus Entry
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    char      text[128];
    int       text_len;
    uint8_t   emb[240];
    int       word_indices[16];
    int       word_count;
    int       is_duplicate;       /* 1 if this is a near-dup of another */
    int       original_index;     /* Index of the original (if is_duplicate) */
} corpus_entry_t;

/* ═══════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    /* --- Defaults --- */
    int corpus_size   = 1000;
    int query_count   = 100;
    float threshold   = 0.60f;
    uint64_t seed     = 42;
    int show_help     = 0;
    int recall_mode   = TRINE_RECALL_BALANCED;
    int all_modes     = 0;
    int validate_mode = 0;
    int health_mode   = 0;
    int json_output   = 0;

    /* --- Parse arguments --- */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            corpus_size = atoi(argv[++i]);
            if (corpus_size < 10) corpus_size = 10;
        } else if (strcmp(argv[i], "--queries") == 0 && i + 1 < argc) {
            query_count = atoi(argv[++i]);
            if (query_count < 1) query_count = 1;
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--quick") == 0) {
            corpus_size = 100;
            query_count = 20;
        } else if (strcmp(argv[i], "--recall") == 0 && i + 1 < argc) {
            const char *rm = argv[++i];
            if (strcmp(rm, "fast") == 0) {
                recall_mode = TRINE_RECALL_FAST;
            } else if (strcmp(rm, "balanced") == 0) {
                recall_mode = TRINE_RECALL_BALANCED;
            } else if (strcmp(rm, "strict") == 0) {
                recall_mode = TRINE_RECALL_STRICT;
            } else {
                fprintf(stderr, "Invalid recall mode '%s' "
                        "(use fast, balanced, strict)\n", rm);
                return 1;
            }
        } else if (strcmp(argv[i], "--all-modes") == 0) {
            all_modes = 1;
        } else if (strcmp(argv[i], "--validate") == 0) {
            validate_mode = 1;
        } else if (strcmp(argv[i], "--health") == 0) {
            health_mode = 1;
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            show_help = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    if (show_help) {
        printf("Usage: trine_recall [OPTIONS]\n");
        printf("  --size N       Corpus size (default: 1000)\n");
        printf("  --queries N    Number of query pairs to test (default: 100)\n");
        printf("  --threshold F  Cosine threshold (default: 0.60)\n");
        printf("  --seed N       Random seed (default: 42)\n");
        printf("  --quick        Use small corpus (100 entries, 20 queries)\n");
        printf("  --recall MODE  Set recall mode: fast, balanced (default), strict\n");
        printf("  --all-modes    Run all three recall modes and show comparison table\n");
        printf("  --validate     Run routing validation with pass/warn/fail verdict\n");
        printf("  --health       Run bucket health analysis with per-band telemetry\n");
        printf("  --json         Output results as JSON (use with --validate or --health)\n");
        printf("  --help         Show this help\n");
        return 0;
    }

    /* Clamp query_count to a sensible percentage of corpus */
    if (query_count > corpus_size / 2)
        query_count = corpus_size / 2;
    if (query_count < 1)
        query_count = 1;

    /* Recall mode label lookup */
    static const char *RECALL_LABELS[3] = { "FAST", "BALANCED", "STRICT" };
    static const char *recall_labels_lc[3] = { "fast", "balanced", "strict" };

    /* ═══════════════════════════════════════════════════════════════════
     * --health: Bucket Health Analysis
     * ═══════════════════════════════════════════════════════════════════ */
    if (health_mode) {
        recall_rng_t rng;
        rng_seed(&rng, seed);

        /* Generate and encode corpus */
        corpus_entry_t *corpus = (corpus_entry_t *)calloc(
            (size_t)corpus_size, sizeof(corpus_entry_t));
        if (!corpus) {
            fprintf(stderr, "ERROR: failed to allocate corpus\n");
            return 1;
        }

        for (int i = 0; i < corpus_size; i++) {
            int wc = 3 + rng_range(&rng, 6);
            corpus[i].text_len = generate_text(&rng, wc, corpus[i].text,
                                                corpus[i].word_indices, 16);
            corpus[i].word_count = wc;
            trine_s1_encode(corpus[i].text, (size_t)corpus[i].text_len,
                             corpus[i].emb);
        }

        /* Build routed index */
        trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
        config.threshold = threshold;
        trine_route_t *rt = trine_route_create(&config);
        if (!rt) {
            fprintf(stderr, "ERROR: failed to create routed index\n");
            free(corpus);
            return 1;
        }
        trine_route_set_recall(rt, recall_mode);

        for (int i = 0; i < corpus_size; i++) {
            trine_route_add(rt, corpus[i].emb, corpus[i].text);
        }

        /* Collect per-band bucket sizes */
        int nb = TRINE_ROUTE_BUCKETS;
        int *sizes = (int *)malloc((size_t)nb * sizeof(int));
        if (!sizes) {
            fprintf(stderr, "ERROR: allocation failed\n");
            trine_route_free(rt);
            free(corpus);
            return 1;
        }

        /* Per-band stats storage */
        double band_avg[TRINE_ROUTE_BANDS];
        int    band_max[TRINE_ROUTE_BANDS];
        int    band_empty[TRINE_ROUTE_BANDS];
        int    band_p95[TRINE_ROUTE_BANDS];
        int    band_p99[TRINE_ROUTE_BANDS];

        /* Global histogram across all bands: bucket size -> count
         * Histogram bins: 0, 1, 2-3, 4-7, 8-15, 16+ */
        #define HEALTH_HISTO_BINS 6
        int total_buckets = TRINE_ROUTE_BANDS * nb;
        int histo[HEALTH_HISTO_BINS];
        memset(histo, 0, sizeof(histo));

        for (int b = 0; b < TRINE_ROUTE_BANDS; b++) {
            trine_route_bucket_sizes(rt, b, sizes);

            /* Compute stats for this band */
            long sum = 0;
            int mx = 0;
            int empty = 0;

            /* We need sorted sizes for percentiles */
            int *sorted = (int *)malloc((size_t)nb * sizeof(int));
            if (!sorted) {
                fprintf(stderr, "ERROR: allocation failed\n");
                free(sizes);
                trine_route_free(rt);
                free(corpus);
                return 1;
            }

            for (int s = 0; s < nb; s++) {
                int c = sizes[s];
                sum += c;
                if (c > mx) mx = c;
                if (c == 0) empty++;
                sorted[s] = c;

                /* Accumulate histogram */
                if      (c == 0)  histo[0]++;
                else if (c == 1)  histo[1]++;
                else if (c <= 3)  histo[2]++;
                else if (c <= 7)  histo[3]++;
                else if (c <= 15) histo[4]++;
                else              histo[5]++;
            }

            band_avg[b] = (double)sum / (double)nb;
            band_max[b] = mx;
            band_empty[b] = empty;

            /* Simple insertion sort for percentiles (nb=4096, fast enough) */
            for (int i = 1; i < nb; i++) {
                int key = sorted[i];
                int j = i - 1;
                while (j >= 0 && sorted[j] > key) {
                    sorted[j + 1] = sorted[j];
                    j--;
                }
                sorted[j + 1] = key;
            }

            band_p95[b] = sorted[(int)((double)nb * 0.95)];
            band_p99[b] = sorted[(int)((double)nb * 0.99)];

            free(sorted);
        }

        if (json_output) {
            /* JSON health output */
            printf("{\"command\": \"health\", \"entries\": %d, \"bands\": %d, "
                   "\"buckets_per_band\": %d, \"bands_detail\": [",
                   corpus_size, TRINE_ROUTE_BANDS, nb);

            for (int b = 0; b < TRINE_ROUTE_BANDS; b++) {
                if (b > 0) printf(", ");
                printf("{\"band\": %d, \"avg\": %.1f, \"max\": %d, "
                       "\"empty\": %d, \"total\": %d, \"p95\": %d, \"p99\": %d}",
                       b, band_avg[b], band_max[b],
                       band_empty[b], nb, band_p95[b], band_p99[b]);
            }

            printf("], \"histogram\": [");
            static const char *histo_labels[HEALTH_HISTO_BINS] = {
                "0", "1", "2-3", "4-7", "8-15", "16+"
            };
            for (int h = 0; h < HEALTH_HISTO_BINS; h++) {
                if (h > 0) printf(", ");
                printf("{\"range\": \"%s\", \"count\": %d, \"pct\": %.1f}",
                       histo_labels[h], histo[h],
                       100.0 * (double)histo[h] / (double)total_buckets);
            }
            printf("]}\n");
        } else {
            /* Human-readable health output */
            printf("\n=== Bucket Health Analysis ===\n");
            printf("Entries:    %d\n", corpus_size);
            printf("Bands:      %d\n", TRINE_ROUTE_BANDS);
            printf("Buckets:    %d per band\n\n", nb);

            for (int b = 0; b < TRINE_ROUTE_BANDS; b++) {
                printf("Band %d: avg=%.1f  max=%-4d empty=%d/%d  p95=%-4d p99=%d\n",
                       b, band_avg[b], band_max[b],
                       band_empty[b], nb, band_p95[b], band_p99[b]);
            }

            /* Histogram */
            printf("\nBucket size histogram:\n");

            static const char *histo_labels[HEALTH_HISTO_BINS] = {
                "0", "1", "2-3", "4-7", "8-15", "16+"
            };

            /* Find max count for bar scaling */
            int histo_max = 0;
            for (int h = 0; h < HEALTH_HISTO_BINS; h++) {
                if (histo[h] > histo_max) histo_max = histo[h];
            }
            int bar_width = 24;

            for (int h = 0; h < HEALTH_HISTO_BINS; h++) {
                double pct = 100.0 * (double)histo[h] / (double)total_buckets;
                int bar_len = (histo_max > 0)
                    ? (int)((double)histo[h] / (double)histo_max * bar_width)
                    : 0;

                printf("  %-5s: ", histo_labels[h]);
                for (int x = 0; x < bar_len; x++) {
                    /* Full block for most, thin bar for tiny values */
                    if (x == 0 && bar_len == 0 && histo[h] > 0) {
                        printf("\xe2\x96\x8f"); /* thin bar (UTF-8) */
                    } else {
                        printf("\xe2\x96\x88"); /* full block (UTF-8) */
                    }
                }
                /* Show thin bar for nonzero entries that round to 0 bars */
                if (bar_len == 0 && histo[h] > 0) {
                    printf("\xe2\x96\x8f");
                }
                printf(" %d (%.1f%%)\n", histo[h], pct);
            }

            printf("\n");
        }

        free(sizes);
        trine_route_free(rt);
        free(corpus);
        return 0;
    }

    /* ═══════════════════════════════════════════════════════════════════
     * --validate: Routing Validation with Verdict
     * ═══════════════════════════════════════════════════════════════════ */
    if (validate_mode) {
        recall_rng_t rng;
        rng_seed(&rng, seed);

        /* Total entries = base corpus + query near-duplicates */
        int total_entries = corpus_size + query_count;

        corpus_entry_t *corpus = (corpus_entry_t *)calloc(
            (size_t)total_entries, sizeof(corpus_entry_t));
        if (!corpus) {
            fprintf(stderr, "ERROR: failed to allocate corpus\n");
            return 1;
        }

        /* Generate base corpus */
        for (int i = 0; i < corpus_size; i++) {
            int wc = 3 + rng_range(&rng, 6);
            corpus[i].text_len = generate_text(&rng, wc, corpus[i].text,
                                                corpus[i].word_indices, 16);
            corpus[i].word_count = wc;
            corpus[i].is_duplicate = 0;
            corpus[i].original_index = -1;
            trine_s1_encode(corpus[i].text, (size_t)corpus[i].text_len,
                             corpus[i].emb);
        }

        /* Generate near-duplicate queries */
        for (int i = 0; i < query_count; i++) {
            int idx = corpus_size + i;
            int orig = rng_range(&rng, corpus_size);
            int mut_indices[16];
            int mut_count = 0;

            corpus[idx].text_len = mutate_text(
                &rng, corpus[orig].word_indices, corpus[orig].word_count,
                corpus[idx].text, mut_indices, &mut_count, 16);
            memcpy(corpus[idx].word_indices, mut_indices,
                   (size_t)mut_count * sizeof(int));
            corpus[idx].word_count = mut_count;
            corpus[idx].is_duplicate = 1;
            corpus[idx].original_index = orig;
            trine_s1_encode(corpus[idx].text, (size_t)corpus[idx].text_len,
                             corpus[idx].emb);
        }

        /* Build both indexes */
        trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
        config.threshold = threshold;

        trine_s1_index_t *brute_idx = trine_s1_index_create(&config);
        trine_route_t *route_idx = trine_route_create(&config);
        if (!brute_idx || !route_idx) {
            fprintf(stderr, "ERROR: failed to create indexes\n");
            if (brute_idx) trine_s1_index_free(brute_idx);
            if (route_idx) trine_route_free(route_idx);
            free(corpus);
            return 1;
        }

        trine_route_set_recall(route_idx, recall_mode);

        for (int i = 0; i < corpus_size; i++) {
            trine_s1_index_add(brute_idx, corpus[i].emb, corpus[i].text);
            trine_route_add(route_idx, corpus[i].emb, corpus[i].text);
        }

        /* Run queries and collect metrics */
        int brute_matches = 0;
        int true_positives = 0;
        double total_candidates = 0.0;
        int overflow_count = 0;  /* queries that hit the candidate cap */

        /* Determine the candidate cap for this recall mode */
        int candidate_cap;
        switch (recall_mode) {
            case TRINE_RECALL_FAST:     candidate_cap = 200;  break;
            case TRINE_RECALL_STRICT:   candidate_cap = 2000; break;
            default:                    candidate_cap = 500;  break;
        }

        for (int i = 0; i < query_count; i++) {
            int qi = corpus_size + i;

            trine_s1_result_t brute_res = trine_s1_index_query(
                brute_idx, corpus[qi].emb);

            trine_route_stats_t qstats;
            memset(&qstats, 0, sizeof(qstats));
            trine_s1_result_t route_res = trine_route_query(
                route_idx, corpus[qi].emb, &qstats);

            total_candidates += qstats.candidates_checked;

            if (qstats.candidates_checked >= candidate_cap)
                overflow_count++;

            if (brute_res.is_duplicate) brute_matches++;
            if (brute_res.is_duplicate && route_res.is_duplicate)
                true_positives++;
        }

        /* Compute metrics */
        double recall_frac = (brute_matches > 0)
            ? (double)true_positives / (double)brute_matches
            : 1.0;
        double recall_pct = recall_frac * 100.0;
        double avg_candidates = total_candidates / (double)query_count;
        double overflow_rate = (double)overflow_count / (double)query_count;

        /* Bucket skew: max/avg ratio across all bands */
        int nb = TRINE_ROUTE_BUCKETS;
        int *sizes = (int *)malloc((size_t)nb * sizeof(int));
        double bucket_skew = 1.0;
        if (sizes) {
            double global_avg = 0.0;
            int global_max = 0;
            for (int b = 0; b < TRINE_ROUTE_BANDS; b++) {
                trine_route_bucket_sizes(route_idx, b, sizes);
                for (int s = 0; s < nb; s++) {
                    global_avg += (double)sizes[s];
                    if (sizes[s] > global_max) global_max = sizes[s];
                }
            }
            global_avg /= (double)(TRINE_ROUTE_BANDS * nb);
            if (global_avg > 0.0) {
                bucket_skew = (double)global_max / global_avg;
            }
            free(sizes);
        }

        /* Verdict logic:
         * PASS: recall >= 99% AND overflow_rate < 5%
         * WARN: recall >= 95% OR overflow_rate < 10%
         * FAIL: otherwise */
        const char *verdict;
        if (recall_pct >= 99.0 && overflow_rate < 0.05) {
            verdict = "PASS";
        } else if (recall_pct >= 95.0 || overflow_rate < 0.10) {
            verdict = "WARN";
        } else {
            verdict = "FAIL";
        }

        if (json_output) {
            printf("{\"command\": \"validate\", \"corpus_size\": %d, "
                   "\"queries\": %d, \"recall_mode\": \"%s\", "
                   "\"recall\": %.4f, \"avg_candidates\": %.1f, "
                   "\"overflow_rate\": %.4f, \"bucket_skew\": %.1f, "
                   "\"verdict\": \"%s\"}\n",
                   corpus_size, query_count,
                   RECALL_LABELS[recall_mode],
                   recall_frac, avg_candidates,
                   overflow_rate, bucket_skew, verdict);
        } else {
            printf("\n=== TRINE Routing Validation ===\n");
            printf("Corpus size:     %d\n", corpus_size);
            printf("Sample queries:  %d\n", query_count);
            printf("Recall mode:     %s\n\n", RECALL_LABELS[recall_mode]);

            printf("Recall:          %.1f%%\n", recall_pct);
            printf("Avg candidates:  %.1f\n", avg_candidates);
            printf("Overflow rate:   %.1f%% (queries hitting candidate cap)\n",
                   overflow_rate * 100.0);
            printf("Bucket skew:     %.1f (max/avg ratio)\n\n", bucket_skew);

            printf("Verdict: %s\n\n", verdict);
        }

        trine_s1_index_free(brute_idx);
        trine_route_free(route_idx);
        free(corpus);
        return (strcmp(verdict, "FAIL") == 0) ? 1 : 0;
    }

    /* ═══════════════════════════════════════════════════════════════════
     * Default mode: full recall benchmark
     * ═══════════════════════════════════════════════════════════════════ */

    /* Total entries = base corpus + near-duplicate queries */
    int total_entries = corpus_size + query_count;

    /* --- Print header --- */
    printf("\n");
    printf("TRINE Routing Recall Validation\n");
    printf("================================\n");
    printf("Corpus size:          %d\n", corpus_size);
    printf("Near-duplicate pairs: %d\n", query_count);
    printf("Threshold:            %.2f\n", (double)threshold);
    printf("Lens:                 DEDUP\n");
    printf("Seed:                 %llu\n", (unsigned long long)seed);
    printf("\n");

    /* --- Allocate corpus --- */
    corpus_entry_t *corpus = (corpus_entry_t *)calloc(
        (size_t)total_entries, sizeof(corpus_entry_t));
    if (!corpus) {
        fprintf(stderr, "ERROR: failed to allocate corpus (%d entries)\n",
                total_entries);
        return 1;
    }

    /* --- Initialize RNG --- */
    recall_rng_t rng;
    rng_seed(&rng, seed);

    /* --- Generate base corpus --- */
    printf("Generating corpus...\n");
    for (int i = 0; i < corpus_size; i++) {
        int word_count = 3 + rng_range(&rng, 6);  /* 3-8 words */
        corpus[i].text_len = generate_text(&rng, word_count,
                                            corpus[i].text,
                                            corpus[i].word_indices, 16);
        corpus[i].word_count = word_count;
        corpus[i].is_duplicate = 0;
        corpus[i].original_index = -1;

        /* Encode */
        trine_s1_encode(corpus[i].text, (size_t)corpus[i].text_len,
                         corpus[i].emb);
    }

    /* --- Generate near-duplicate entries --- */
    printf("Generating %d near-duplicate pairs...\n", query_count);
    for (int i = 0; i < query_count; i++) {
        int idx = corpus_size + i;
        int orig = rng_range(&rng, corpus_size);

        int mut_indices[16];
        int mut_count = 0;

        corpus[idx].text_len = mutate_text(
            &rng,
            corpus[orig].word_indices,
            corpus[orig].word_count,
            corpus[idx].text,
            mut_indices,
            &mut_count,
            16);

        memcpy(corpus[idx].word_indices, mut_indices,
               (size_t)mut_count * sizeof(int));
        corpus[idx].word_count = mut_count;
        corpus[idx].is_duplicate = 1;
        corpus[idx].original_index = orig;

        /* Encode */
        trine_s1_encode(corpus[idx].text, (size_t)corpus[idx].text_len,
                         corpus[idx].emb);
    }

    /* --- Build brute-force index with base corpus --- */
    printf("Building indexes (%d entries)...\n", corpus_size);

    trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
    config.threshold = threshold;

    trine_s1_index_t *brute_idx = trine_s1_index_create(&config);
    if (!brute_idx) {
        fprintf(stderr, "ERROR: failed to create brute-force index\n");
        free(corpus);
        return 1;
    }

    for (int i = 0; i < corpus_size; i++) {
        trine_s1_index_add(brute_idx, corpus[i].emb, corpus[i].text);
    }

    /* --- Determine which modes to run --- */
    static const int ALL_MODES[3] = {
        TRINE_RECALL_FAST, TRINE_RECALL_BALANCED, TRINE_RECALL_STRICT
    };
    static const char *MODE_LABELS[3] = { "FAST", "BALANCED", "STRICT" };
    static const char *mode_labels_lc[3] = { "fast", "balanced", "strict" };

    int modes_to_run[3];
    int num_modes = 0;

    if (all_modes) {
        modes_to_run[0] = TRINE_RECALL_FAST;
        modes_to_run[1] = TRINE_RECALL_BALANCED;
        modes_to_run[2] = TRINE_RECALL_STRICT;
        num_modes = 3;
    } else {
        modes_to_run[0] = recall_mode;
        num_modes = 1;
    }

    /* Storage for per-mode results (used in --all-modes comparison table) */
    double mode_recall[3]      = {0};
    double mode_speedup[3]     = {0};
    double mode_avg_cand[3]    = {0};
    double mode_avg_query[3]   = {0};
    int    mode_brute[3]       = {0};

    for (int m = 0; m < num_modes; m++) {
        int cur_mode = modes_to_run[m];

        /* Build routed index for this mode */
        trine_route_t *route_idx = trine_route_create(&config);
        if (!route_idx) {
            fprintf(stderr, "ERROR: failed to create routed index\n");
            trine_s1_index_free(brute_idx);
            free(corpus);
            return 1;
        }

        trine_route_set_recall(route_idx, cur_mode);

        for (int i = 0; i < corpus_size; i++) {
            trine_route_add(route_idx, corpus[i].emb, corpus[i].text);
        }

        /* --- Run queries: compare brute-force vs routed --- */
        if (!all_modes) {
            printf("Running %d queries...\n\n", query_count);
        } else if (m == 0) {
            printf("Running %d queries x 3 modes...\n\n", query_count);
        }

        int brute_matches = 0;
        int route_matches = 0;
        int true_positives = 0;
        int false_negatives = 0;
        int true_negatives = 0;
        int route_only = 0;

        double total_candidates = 0.0;
        double total_brute_time = 0.0;
        double total_route_time = 0.0;

        for (int i = 0; i < query_count; i++) {
            int qi = corpus_size + i;

            /* Brute-force query */
            double t0 = now_sec();
            trine_s1_result_t brute_res = trine_s1_index_query(
                brute_idx, corpus[qi].emb);
            double t1 = now_sec();
            total_brute_time += (t1 - t0);

            /* Routed query */
            trine_route_stats_t qstats;
            memset(&qstats, 0, sizeof(qstats));

            double t2 = now_sec();
            trine_s1_result_t route_res = trine_route_query(
                route_idx, corpus[qi].emb, &qstats);
            double t3 = now_sec();
            total_route_time += (t3 - t2);

            total_candidates += qstats.candidates_checked;

            if (brute_res.is_duplicate) brute_matches++;
            if (route_res.is_duplicate) route_matches++;

            if (brute_res.is_duplicate && route_res.is_duplicate) {
                true_positives++;
            } else if (brute_res.is_duplicate && !route_res.is_duplicate) {
                false_negatives++;
            } else if (!brute_res.is_duplicate && !route_res.is_duplicate) {
                true_negatives++;
            } else {
                route_only++;
            }
        }

        /* --- Compute metrics --- */
        double recall_pct = (brute_matches > 0)
            ? 100.0 * (double)true_positives / (double)brute_matches
            : 100.0;

        double avg_candidates = total_candidates / query_count;
        double avg_speedup = (avg_candidates > 0.0)
            ? (double)corpus_size / avg_candidates
            : 1.0;

        double avg_brute_us = (total_brute_time / query_count) * 1e6;
        double avg_route_us = (total_route_time / query_count) * 1e6;

        /* Store for comparison table */
        mode_recall[m]    = recall_pct;
        mode_speedup[m]   = avg_speedup;
        mode_avg_cand[m]  = avg_candidates;
        mode_avg_query[m] = avg_route_us;
        mode_brute[m]     = brute_matches;

        /* --- Single-mode report --- */
        if (!all_modes) {
            printf("Results:\n");
            printf("  Brute-force matches:  %d\n", brute_matches);
            printf("  Routed matches:       %d\n", route_matches);

            if (brute_matches > 0) {
                printf("  Recall:               %.1f%%  (%d/%d)\n",
                       recall_pct, true_positives, brute_matches);
            } else {
                printf("  Recall:               N/A  (no brute-force matches)\n");
            }

            printf("  False negatives:      %d\n", false_negatives);
            printf("  True negatives:       %d\n", true_negatives);

            if (route_only > 0)
                printf("  Route-only matches:   %d  (route found, brute missed)\n",
                       route_only);

            printf("\n");
            printf("Routing Performance (%s):\n", mode_labels_lc[cur_mode]);
            printf("  Avg candidates/query: %.1f / %d\n",
                   avg_candidates, corpus_size);
            printf("  Avg speedup:          %.1fx\n", avg_speedup);
            printf("  Avg query time:       %.1f us (routed) vs %.1f us (brute)\n",
                   avg_route_us, avg_brute_us);

            printf("\n");

            /* --- Verdict --- */
            if (brute_matches > 0 && recall_pct >= 95.0) {
                printf("PASS: Routing recall >= 95%% (%.1f%%)\n", recall_pct);
            } else if (brute_matches > 0 && recall_pct < 95.0) {
                printf("WARN: Routing recall below 95%% (%.1f%%)\n", recall_pct);
            } else {
                printf("NOTE: No brute-force matches found at threshold %.2f\n",
                       (double)threshold);
                printf("      Try lowering --threshold or increasing --size\n");
            }

            printf("\n");
        }

        trine_route_free(route_idx);
    }

    /* --- All-modes comparison table --- */
    if (all_modes) {
        printf("Recall by mode:\n");
        printf("  %-10s %-8s %-9s %-10s %-10s\n",
               "Mode", "Recall", "Speedup", "Avg Cand", "Avg Query");
        for (int m = 0; m < 3; m++) {
            printf("  %-10s %.1f%%    %.1fx%*s %.1f%*s %.1f us\n",
                   MODE_LABELS[m],
                   mode_recall[m],
                   mode_speedup[m],
                   (mode_speedup[m] >= 100.0) ? 2 :
                   (mode_speedup[m] >= 10.0) ? 3 : 4, "",
                   mode_avg_cand[m],
                   (mode_avg_cand[m] >= 100.0) ? 4 :
                   (mode_avg_cand[m] >= 10.0) ? 5 : 6, "",
                   mode_avg_query[m]);
        }

        printf("\n");

        /* Verdict based on balanced mode (index 1) */
        if (mode_brute[1] > 0 && mode_recall[1] >= 95.0) {
            printf("PASS: Balanced recall >= 95%% (%.1f%%)\n", mode_recall[1]);
        } else if (mode_brute[1] > 0 && mode_recall[1] < 95.0) {
            printf("WARN: Balanced recall below 95%% (%.1f%%)\n", mode_recall[1]);
        } else {
            printf("NOTE: No brute-force matches found at threshold %.2f\n",
                   (double)threshold);
            printf("      Try lowering --threshold or increasing --size\n");
        }

        printf("\n");
    }

    (void)ALL_MODES;
    (void)recall_labels_lc;
    (void)RECALL_LABELS;

    /* --- Cleanup --- */
    trine_s1_index_free(brute_idx);
    free(corpus);

    return 0;
}
