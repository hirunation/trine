/* =====================================================================
 * TRINE CORPUS BENCH — Real-World Corpus Benchmark Suite
 * Ternary Resonance Interference Network Embedding
 * =====================================================================
 *
 * Evaluates TRINE embedding quality and performance on real-world
 * corpora using JSONL input files with ground-truth annotations.
 *
 * Modes:
 *   --sts <file.jsonl>      STS correlation benchmark
 *   --dedup <file.jsonl>    Dedup precision/recall benchmark
 *   --routing <file.jsonl>  Routing performance benchmark
 *   --cost                  Cost equivalence calculator
 *
 * Build:
 *   cc -O2 -Wall -Wextra -Werror -o build/trine_corpus_bench \
 *       benchmarks/trine_corpus_bench.c trine_encode.c \
 *       trine_stage1.c trine_route.c trine_canon.c -lm
 *
 * ZERO external dependencies beyond libc + libm.
 *
 * ===================================================================== */

#define _POSIX_C_SOURCE 200809L  /* clock_gettime, strdup */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <float.h>

#include "../trine_encode.h"
#include "../trine_stage1.h"
#include "../trine_route.h"
#include "../trine_canon.h"

/* =====================================================================
 * Constants
 * ===================================================================== */

#define BENCH_VERSION       "1.0.1"
#define BENCH_MAX_LINE      65536       /* Max JSONL line length         */
#define BENCH_MAX_ENTRIES   50000       /* Max entries per corpus        */
#define BENCH_DIMS          240         /* Shingle embedding dimensions  */
#define BENCH_CHAINS        4           /* Number of encoding chains     */
#define BENCH_CHAIN_WIDTH   60          /* Channels per chain            */

/* Dedup threshold sweep points */
#define BENCH_NUM_THRESHOLDS 9
static const float BENCH_THRESHOLDS[BENCH_NUM_THRESHOLDS] = {
    0.50f, 0.55f, 0.60f, 0.65f, 0.70f, 0.75f, 0.80f, 0.85f, 0.90f
};

/* Routing subsample sizes for scaling curve */
#define BENCH_NUM_SUBSAMPLE 5
static const int BENCH_SUBSAMPLE_SIZES[BENCH_NUM_SUBSAMPLE] = {
    100, 500, 1000, 2000, 5000
};

/* Chain names (used in verbose output) */
static const char * const CHAIN_NAME[BENCH_CHAINS] __attribute__((unused)) = {
    "edit", "morph", "phrase", "vocab"
};

/* =====================================================================
 * Lens — Per-Chain Weighting
 * =====================================================================
 *
 * Each lens assigns a weight to each of the 4 encoding chains:
 *   Chain 0 (edit):   Character unigrams + bigrams
 *   Chain 1 (morph):  Character trigrams
 *   Chain 2 (phrase): Character 5-grams
 *   Chain 3 (vocab):  Word unigrams
 *
 * Lens-weighted cosine:
 *   combined = sum(w[i] * cos[i]) / sum(w[i])
 *
 * ===================================================================== */

typedef struct {
    float weights[BENCH_CHAINS];
    const char *name;
} bench_lens_t;

/* Preset lenses — same values as trine_stage1.h */
static const bench_lens_t LENS_UNIFORM = {{1.0f, 1.0f, 1.0f, 1.0f}, "uniform"};
static const bench_lens_t LENS_DEDUP   = {{0.5f, 0.5f, 0.7f, 1.0f}, "dedup"  };
static const bench_lens_t LENS_PHRASE  = {{0.1f, 0.3f, 1.0f, 0.5f}, "phrase" };
static const bench_lens_t LENS_VOCAB   = {{0.0f, 0.2f, 0.3f, 1.0f}, "vocab"  };
static const bench_lens_t LENS_CODE    = {{1.0f, 0.8f, 0.4f, 0.2f}, "code"   };
static const bench_lens_t LENS_LEGAL   = {{0.2f, 0.4f, 1.0f, 0.8f}, "legal"  };
static const bench_lens_t LENS_SUPPORT = {{0.2f, 0.4f, 0.7f, 1.0f}, "support"};
static const bench_lens_t LENS_POLICY  = {{0.1f, 0.3f, 1.0f, 0.8f}, "policy" };

/* All lenses to sweep during STS evaluation */
#define BENCH_NUM_LENSES 8
static const bench_lens_t *ALL_LENSES[BENCH_NUM_LENSES] = {
    &LENS_UNIFORM, &LENS_DEDUP, &LENS_PHRASE, &LENS_VOCAB,
    &LENS_CODE, &LENS_LEGAL, &LENS_SUPPORT, &LENS_POLICY
};

/* =====================================================================
 * Timing
 * ===================================================================== */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double now_usec(void)
{
    return now_sec() * 1e6;
}

/* =====================================================================
 * Cosine Similarity Functions
 * ===================================================================== */

/*
 * chain_cosine_60 — Cosine similarity over a single 60-channel chain.
 * Treats trit values as real-valued vector components {0, 1, 2}.
 * Returns 0.0 if either slice has zero magnitude.
 */
static double chain_cosine_60(const uint8_t *a, const uint8_t *b,
                               int offset)
{
    uint64_t dot_ab = 0;
    uint64_t mag_a  = 0;
    uint64_t mag_b  = 0;

    for (int i = offset; i < offset + BENCH_CHAIN_WIDTH; i++) {
        uint64_t va = a[i];
        uint64_t vb = b[i];
        dot_ab += va * vb;
        mag_a  += va * va;
        mag_b  += vb * vb;
    }

    if (mag_a == 0 || mag_b == 0) return 0.0;

    double denom = sqrt((double)mag_a) * sqrt((double)mag_b);
    if (denom == 0.0) return 0.0;

    double sim = (double)dot_ab / denom;
    if (sim > 1.0) sim = 1.0;
    if (sim < 0.0) sim = 0.0;
    return sim;
}

/*
 * lens_cosine — Lens-weighted cosine similarity over the full 240 dims.
 *
 * Computes cosine independently for each of the 4 chains (60 channels
 * each), then combines with lens weights:
 *   combined = sum(weight[i] * chain_cosine[i]) / sum(weight[i])
 */
static double lens_cosine(const uint8_t *a, const uint8_t *b,
                           const bench_lens_t *lens)
{
    double weighted_sum = 0.0;
    double weight_sum   = 0.0;

    for (int c = 0; c < BENCH_CHAINS; c++) {
        double w = (double)lens->weights[c];
        if (w <= 0.0) continue;

        double cos_c = chain_cosine_60(a, b, c * BENCH_CHAIN_WIDTH);
        weighted_sum += w * cos_c;
        weight_sum   += w;
    }

    if (weight_sum == 0.0) return 0.0;
    return weighted_sum / weight_sum;
}

/* =====================================================================
 * Minimal JSON Parser
 * =====================================================================
 *
 * Extracts top-level fields from a single JSON line.
 * Handles string, number, and boolean values.
 *
 * ===================================================================== */

/*
 * json_find_string — Find a string field value in a JSON line.
 * Returns pointer to start of value (after opening quote), sets *vlen.
 * Returns NULL if not found.
 */
static const char *json_find_string(const char *json, size_t json_len,
                                     const char *key, size_t *vlen)
{
    if (!json || !key || !vlen) return NULL;

    size_t klen = strlen(key);
    const char *end = json + json_len;
    const char *p = json;

    while (p < end) {
        const char *q1 = memchr(p, '"', (size_t)(end - p));
        if (!q1 || q1 + klen + 1 >= end) return NULL;

        if (memcmp(q1 + 1, key, klen) == 0 && q1[klen + 1] == '"') {
            const char *after_key = q1 + klen + 2;

            while (after_key < end &&
                   (*after_key == ' ' || *after_key == '\t'))
                after_key++;

            if (after_key >= end || *after_key != ':') {
                p = after_key;
                continue;
            }
            after_key++;

            while (after_key < end &&
                   (*after_key == ' ' || *after_key == '\t'))
                after_key++;

            if (after_key >= end || *after_key != '"') {
                p = after_key;
                continue;
            }
            after_key++;

            const char *vstart = after_key;
            const char *vp = vstart;
            while (vp < end) {
                if (*vp == '\\' && vp + 1 < end) {
                    vp += 2;
                    continue;
                }
                if (*vp == '"') {
                    *vlen = (size_t)(vp - vstart);
                    return vstart;
                }
                vp++;
            }
            return NULL;
        }

        q1++;
        while (q1 < end) {
            if (*q1 == '\\' && q1 + 1 < end) {
                q1 += 2;
                continue;
            }
            if (*q1 == '"') {
                q1++;
                break;
            }
            q1++;
        }
        p = q1;
    }

    return NULL;
}

/*
 * json_find_number — Find a numeric field value in a JSON line.
 * Searches for "key": <number> pattern. Parses the number into *out.
 * Returns 1 on success, 0 on failure.
 */
static int json_find_number(const char *json, size_t json_len,
                             const char *key, double *out)
{
    if (!json || !key || !out) return 0;

    size_t klen = strlen(key);
    const char *end = json + json_len;
    const char *p = json;

    while (p < end) {
        const char *q1 = memchr(p, '"', (size_t)(end - p));
        if (!q1 || q1 + klen + 1 >= end) return 0;

        if (memcmp(q1 + 1, key, klen) == 0 && q1[klen + 1] == '"') {
            const char *after_key = q1 + klen + 2;

            while (after_key < end &&
                   (*after_key == ' ' || *after_key == '\t'))
                after_key++;

            if (after_key >= end || *after_key != ':') {
                p = after_key;
                continue;
            }
            after_key++;

            while (after_key < end &&
                   (*after_key == ' ' || *after_key == '\t'))
                after_key++;

            if (after_key >= end) return 0;

            /* Parse number (integer or float) */
            char *numend = NULL;
            *out = strtod(after_key, &numend);
            if (numend > after_key) return 1;
            return 0;
        }

        q1++;
        while (q1 < end) {
            if (*q1 == '\\' && q1 + 1 < end) {
                q1 += 2;
                continue;
            }
            if (*q1 == '"') {
                q1++;
                break;
            }
            q1++;
        }
        p = q1;
    }

    return 0;
}

/*
 * json_find_bool — Find a boolean field value in a JSON line.
 * Searches for "key": true/false. Sets *out to 1 or 0.
 * Returns 1 on success, 0 on failure.
 */
static int json_find_bool(const char *json, size_t json_len,
                            const char *key, int *out)
{
    if (!json || !key || !out) return 0;

    size_t klen = strlen(key);
    const char *end = json + json_len;
    const char *p = json;

    while (p < end) {
        const char *q1 = memchr(p, '"', (size_t)(end - p));
        if (!q1 || q1 + klen + 1 >= end) return 0;

        if (memcmp(q1 + 1, key, klen) == 0 && q1[klen + 1] == '"') {
            const char *after_key = q1 + klen + 2;

            while (after_key < end &&
                   (*after_key == ' ' || *after_key == '\t'))
                after_key++;

            if (after_key >= end || *after_key != ':') {
                p = after_key;
                continue;
            }
            after_key++;

            while (after_key < end &&
                   (*after_key == ' ' || *after_key == '\t'))
                after_key++;

            if (after_key >= end) return 0;

            if (after_key + 4 <= end && memcmp(after_key, "true", 4) == 0) {
                *out = 1;
                return 1;
            }
            if (after_key + 5 <= end && memcmp(after_key, "false", 5) == 0) {
                *out = 0;
                return 1;
            }
            return 0;
        }

        q1++;
        while (q1 < end) {
            if (*q1 == '\\' && q1 + 1 < end) {
                q1 += 2;
                continue;
            }
            if (*q1 == '"') {
                q1++;
                break;
            }
            q1++;
        }
        p = q1;
    }

    return 0;
}

/*
 * json_unescape — Unescape a JSON string value in-place.
 * Returns the unescaped length.
 */
static size_t json_unescape(char *buf, size_t len)
{
    size_t r = 0, w = 0;
    while (r < len) {
        if (buf[r] == '\\' && r + 1 < len) {
            char c = buf[r + 1];
            switch (c) {
            case '"':  buf[w++] = '"';  r += 2; break;
            case '\\': buf[w++] = '\\'; r += 2; break;
            case 'n':  buf[w++] = '\n'; r += 2; break;
            case 't':  buf[w++] = '\t'; r += 2; break;
            case 'r':  buf[w++] = '\r'; r += 2; break;
            case '/':  buf[w++] = '/';  r += 2; break;
            default:   buf[w++] = buf[r++]; break;
            }
        } else {
            buf[w++] = buf[r++];
        }
    }
    buf[w] = '\0';
    return w;
}

/*
 * strdup_field — Extract a string field, heap-allocate, unescape.
 * Returns NULL if field not found. Caller must free.
 */
static char *strdup_field(const char *json, size_t json_len,
                           const char *key)
{
    size_t vlen = 0;
    const char *val = json_find_string(json, json_len, key, &vlen);
    if (!val || vlen == 0) return NULL;

    char *out = (char *)malloc(vlen + 1);
    if (!out) return NULL;
    memcpy(out, val, vlen);
    out[vlen] = '\0';
    json_unescape(out, vlen);
    return out;
}

/* =====================================================================
 * STS Pair Data
 * ===================================================================== */

typedef struct {
    char *id;
    char *text1;
    char *text2;
    double score;       /* Human score (0-5 scale) */
    char *label;        /* "high" / "medium" / "low" */
} sts_pair_t;

typedef struct {
    sts_pair_t *pairs;
    int count;
    int capacity;
} sts_corpus_t;

static void sts_corpus_init(sts_corpus_t *c)
{
    c->pairs = NULL;
    c->count = 0;
    c->capacity = 0;
}

static void sts_corpus_free(sts_corpus_t *c)
{
    for (int i = 0; i < c->count; i++) {
        free(c->pairs[i].id);
        free(c->pairs[i].text1);
        free(c->pairs[i].text2);
        free(c->pairs[i].label);
    }
    free(c->pairs);
    c->pairs = NULL;
    c->count = 0;
    c->capacity = 0;
}

static int sts_corpus_add(sts_corpus_t *c, const char *json, size_t len)
{
    if (c->count >= BENCH_MAX_ENTRIES) return 0;

    if (c->count >= c->capacity) {
        int newcap = c->capacity == 0 ? 256 : c->capacity * 2;
        if (newcap > BENCH_MAX_ENTRIES) newcap = BENCH_MAX_ENTRIES;
        sts_pair_t *newp = (sts_pair_t *)realloc(c->pairs,
                            (size_t)newcap * sizeof(sts_pair_t));
        if (!newp) return 0;
        c->pairs = newp;
        c->capacity = newcap;
    }

    sts_pair_t *p = &c->pairs[c->count];
    memset(p, 0, sizeof(*p));

    p->text1 = strdup_field(json, len, "text1");
    p->text2 = strdup_field(json, len, "text2");
    if (!p->text1 || !p->text2) {
        free(p->text1);
        free(p->text2);
        return 0;
    }

    p->id = strdup_field(json, len, "id");
    p->label = strdup_field(json, len, "label");

    if (!json_find_number(json, len, "score", &p->score)) {
        p->score = 0.0;
    }

    c->count++;
    return 1;
}

static int sts_corpus_load(sts_corpus_t *c, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "error: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }

    char line[BENCH_MAX_LINE];
    int lineno = 0;
    int loaded = 0;
    int skipped = 0;

    while (fgets(line, sizeof(line), f)) {
        lineno++;
        size_t llen = strlen(line);
        while (llen > 0 && (line[llen-1] == '\n' || line[llen-1] == '\r'))
            line[--llen] = '\0';
        if (llen == 0 || line[0] != '{') continue;

        if (sts_corpus_add(c, line, llen))
            loaded++;
        else
            skipped++;
    }

    fclose(f);
    fprintf(stderr, "[sts] loaded %d pairs from %s (%d skipped)\n",
            loaded, path, skipped);
    return loaded;
}

/* =====================================================================
 * Dedup Corpus Data
 * ===================================================================== */

typedef struct {
    char *id;
    char *text;
    char *group;
    int is_duplicate;
    uint8_t emb[BENCH_DIMS];
} dedup_entry_t;

typedef struct {
    dedup_entry_t *entries;
    int count;
    int capacity;
} dedup_corpus_t;

static void dedup_corpus_init(dedup_corpus_t *c)
{
    c->entries = NULL;
    c->count = 0;
    c->capacity = 0;
}

static void dedup_corpus_free(dedup_corpus_t *c)
{
    for (int i = 0; i < c->count; i++) {
        free(c->entries[i].id);
        free(c->entries[i].text);
        free(c->entries[i].group);
    }
    free(c->entries);
    c->entries = NULL;
    c->count = 0;
    c->capacity = 0;
}

static int dedup_corpus_add(dedup_corpus_t *c, const char *json, size_t len)
{
    if (c->count >= BENCH_MAX_ENTRIES) return 0;

    if (c->count >= c->capacity) {
        int newcap = c->capacity == 0 ? 256 : c->capacity * 2;
        if (newcap > BENCH_MAX_ENTRIES) newcap = BENCH_MAX_ENTRIES;
        dedup_entry_t *newe = (dedup_entry_t *)realloc(c->entries,
                               (size_t)newcap * sizeof(dedup_entry_t));
        if (!newe) return 0;
        c->entries = newe;
        c->capacity = newcap;
    }

    dedup_entry_t *e = &c->entries[c->count];
    memset(e, 0, sizeof(*e));

    e->text = strdup_field(json, len, "text");
    if (!e->text) return 0;

    e->id = strdup_field(json, len, "id");
    e->group = strdup_field(json, len, "group");

    if (!json_find_bool(json, len, "is_duplicate", &e->is_duplicate)) {
        e->is_duplicate = 0;
    }

    /* Encode immediately */
    trine_encode_shingle(e->text, strlen(e->text), e->emb);

    c->count++;
    return 1;
}

static int dedup_corpus_load(dedup_corpus_t *c, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "error: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }

    char line[BENCH_MAX_LINE];
    int lineno = 0;
    int loaded = 0;
    int skipped = 0;

    while (fgets(line, sizeof(line), f)) {
        lineno++;
        size_t llen = strlen(line);
        while (llen > 0 && (line[llen-1] == '\n' || line[llen-1] == '\r'))
            line[--llen] = '\0';
        if (llen == 0 || line[0] != '{') continue;

        if (dedup_corpus_add(c, line, llen))
            loaded++;
        else
            skipped++;
    }

    fclose(f);
    fprintf(stderr, "[dedup] loaded %d entries from %s (%d skipped)\n",
            loaded, path, skipped);
    return loaded;
}

/* =====================================================================
 * Spearman Rank Correlation
 * =====================================================================
 *
 * Computes Spearman's rho between two arrays of doubles.
 * Uses average rank for ties.
 *
 * ===================================================================== */

typedef struct {
    double value;
    int original_index;
} rank_entry_t;

static int rank_entry_cmp(const void *a, const void *b)
{
    const rank_entry_t *ra = (const rank_entry_t *)a;
    const rank_entry_t *rb = (const rank_entry_t *)b;
    if (ra->value < rb->value) return -1;
    if (ra->value > rb->value) return  1;
    return 0;
}

/*
 * compute_ranks — Assign average ranks (1-based) to values.
 * Allocates and returns a rank array. Caller must free.
 * Returns NULL on allocation failure.
 */
static double *compute_ranks(const double *values, int n)
{
    rank_entry_t *entries = (rank_entry_t *)malloc(
        (size_t)n * sizeof(rank_entry_t));
    double *ranks = (double *)malloc((size_t)n * sizeof(double));
    if (!entries || !ranks) {
        free(entries);
        free(ranks);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        entries[i].value = values[i];
        entries[i].original_index = i;
    }

    qsort(entries, (size_t)n, sizeof(rank_entry_t), rank_entry_cmp);

    /* Assign average ranks for ties */
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n && entries[j].value == entries[i].value)
            j++;

        /* All entries from i to j-1 are tied */
        double avg_rank = 0.0;
        for (int k = i; k < j; k++)
            avg_rank += (double)(k + 1);
        avg_rank /= (double)(j - i);

        for (int k = i; k < j; k++)
            ranks[entries[k].original_index] = avg_rank;

        i = j;
    }

    free(entries);
    return ranks;
}

/*
 * spearman_correlation — Compute Spearman's rank correlation coefficient.
 *
 * rho = 1 - (6 * sum(d_i^2)) / (n * (n^2 - 1))
 * where d_i = rank(x_i) - rank(y_i)
 *
 * Uses the Pearson correlation of ranks for tie correction.
 * Returns 0.0 if n < 2 or on allocation failure.
 */
static double spearman_correlation(const double *x, const double *y, int n)
{
    if (n < 2) return 0.0;

    double *rx = compute_ranks(x, n);
    double *ry = compute_ranks(y, n);
    if (!rx || !ry) {
        free(rx);
        free(ry);
        return 0.0;
    }

    /* Pearson correlation of the ranks */
    double mean_rx = 0.0, mean_ry = 0.0;
    for (int i = 0; i < n; i++) {
        mean_rx += rx[i];
        mean_ry += ry[i];
    }
    mean_rx /= n;
    mean_ry /= n;

    double cov = 0.0, var_rx = 0.0, var_ry = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = rx[i] - mean_rx;
        double dy = ry[i] - mean_ry;
        cov    += dx * dy;
        var_rx += dx * dx;
        var_ry += dy * dy;
    }

    free(rx);
    free(ry);

    double denom = sqrt(var_rx) * sqrt(var_ry);
    if (denom < 1e-12) return 0.0;

    return cov / denom;
}

/* =====================================================================
 * Percentile Helper
 * ===================================================================== */

static int double_cmp(const void *a, const void *b)
{
    double va = *(const double *)a;
    double vb = *(const double *)b;
    if (va < vb) return -1;
    if (va > vb) return  1;
    return 0;
}

/*
 * percentile — Compute the p-th percentile (0-100) of a sorted array.
 * Array must be pre-sorted. p is in [0, 100].
 */
static double percentile(const double *sorted, int n, double p)
{
    if (n <= 0) return 0.0;
    if (n == 1) return sorted[0];

    double idx = (p / 100.0) * (n - 1);
    int lo = (int)idx;
    int hi = lo + 1;
    if (hi >= n) hi = n - 1;

    double frac = idx - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

/* =====================================================================
 * Output Format
 * ===================================================================== */

static int g_json_output = 0;

static void print_separator(void)
{
    if (!g_json_output) {
        printf("-------------------------------------------------------------"
               "--------------------\n");
    }
}

static void print_header(const char *title)
{
    if (!g_json_output) {
        printf("\n");
        print_separator();
        printf("  %s\n", title);
        print_separator();
    }
}

/* =====================================================================
 * Mode 1: STS Benchmark
 * =====================================================================
 *
 * Load all sentence pairs, compute TRINE similarity with each lens,
 * compare against human scores, and report Spearman rank correlation.
 *
 * ===================================================================== */

static void run_sts_benchmark(const char *path)
{
    sts_corpus_t corpus;
    sts_corpus_init(&corpus);

    if (sts_corpus_load(&corpus, path) <= 0) {
        fprintf(stderr, "error: no valid STS pairs loaded\n");
        sts_corpus_free(&corpus);
        return;
    }

    int n = corpus.count;

    /* Encode all texts */
    uint8_t *emb1 = (uint8_t *)malloc((size_t)n * BENCH_DIMS);
    uint8_t *emb2 = (uint8_t *)malloc((size_t)n * BENCH_DIMS);
    double *human_scores = (double *)malloc((size_t)n * sizeof(double));
    double *trine_scores = (double *)malloc((size_t)n * sizeof(double));

    if (!emb1 || !emb2 || !human_scores || !trine_scores) {
        fprintf(stderr, "error: allocation failed for %d pairs\n", n);
        free(emb1); free(emb2); free(human_scores); free(trine_scores);
        sts_corpus_free(&corpus);
        return;
    }

    double t0 = now_sec();
    for (int i = 0; i < n; i++) {
        trine_encode_shingle(corpus.pairs[i].text1,
                              strlen(corpus.pairs[i].text1),
                              emb1 + (size_t)i * BENCH_DIMS);
        trine_encode_shingle(corpus.pairs[i].text2,
                              strlen(corpus.pairs[i].text2),
                              emb2 + (size_t)i * BENCH_DIMS);
        /* Normalize human score from 0-5 to 0-1 */
        human_scores[i] = corpus.pairs[i].score / 5.0;
    }
    double encode_time = now_sec() - t0;

    if (g_json_output) {
        printf("{\"mode\": \"sts\", \"pairs\": %d, \"encode_time_ms\": %.2f,\n",
               n, encode_time * 1000.0);
        printf(" \"lenses\": [\n");
    } else {
        print_header("STS BENCHMARK — Spearman Correlation");
        printf("  Pairs:       %d\n", n);
        printf("  Encode time: %.2f ms (%.0f pairs/sec)\n",
               encode_time * 1000.0,
               n > 0 ? n / encode_time : 0.0);
        printf("\n");
        printf("  %-12s  %10s  %10s  %10s  %10s\n",
               "Lens", "Spearman", "Mean-Err", "High-Corr", "Low-Corr");
        print_separator();
    }

    double best_rho = -2.0;
    const char *best_lens = "none";

    for (int li = 0; li < BENCH_NUM_LENSES; li++) {
        const bench_lens_t *lens = ALL_LENSES[li];

        /* Compute TRINE similarities with this lens */
        double t1 = now_sec();
        for (int i = 0; i < n; i++) {
            trine_scores[i] = lens_cosine(emb1 + (size_t)i * BENCH_DIMS,
                                           emb2 + (size_t)i * BENCH_DIMS,
                                           lens);
        }
        double compare_time = now_sec() - t1;
        (void)compare_time;

        /* Overall Spearman correlation */
        double rho = spearman_correlation(human_scores, trine_scores, n);

        /* Mean absolute error */
        double mae = 0.0;
        for (int i = 0; i < n; i++) {
            mae += fabs(human_scores[i] - trine_scores[i]);
        }
        mae /= n;

        /* Per-label correlation */
        double *high_h = NULL, *high_t = NULL;
        double *low_h = NULL, *low_t = NULL;
        int high_count = 0, low_count = 0;

        /* Count per-label entries */
        for (int i = 0; i < n; i++) {
            if (corpus.pairs[i].label) {
                if (strcmp(corpus.pairs[i].label, "high") == 0) high_count++;
                else if (strcmp(corpus.pairs[i].label, "low") == 0) low_count++;
            }
        }

        double rho_high = 0.0, rho_low = 0.0;

        if (high_count >= 2) {
            high_h = (double *)malloc((size_t)high_count * sizeof(double));
            high_t = (double *)malloc((size_t)high_count * sizeof(double));
            if (high_h && high_t) {
                int k = 0;
                for (int i = 0; i < n; i++) {
                    if (corpus.pairs[i].label &&
                        strcmp(corpus.pairs[i].label, "high") == 0) {
                        high_h[k] = human_scores[i];
                        high_t[k] = trine_scores[i];
                        k++;
                    }
                }
                rho_high = spearman_correlation(high_h, high_t, high_count);
            }
            free(high_h);
            free(high_t);
        }

        if (low_count >= 2) {
            low_h = (double *)malloc((size_t)low_count * sizeof(double));
            low_t = (double *)malloc((size_t)low_count * sizeof(double));
            if (low_h && low_t) {
                int k = 0;
                for (int i = 0; i < n; i++) {
                    if (corpus.pairs[i].label &&
                        strcmp(corpus.pairs[i].label, "low") == 0) {
                        low_h[k] = human_scores[i];
                        low_t[k] = trine_scores[i];
                        k++;
                    }
                }
                rho_low = spearman_correlation(low_h, low_t, low_count);
            }
            free(low_h);
            free(low_t);
        }

        if (rho > best_rho) {
            best_rho = rho;
            best_lens = lens->name;
        }

        if (g_json_output) {
            printf("  {\"lens\": \"%s\", \"spearman\": %.6f, "
                   "\"mae\": %.6f, \"rho_high\": %.6f, "
                   "\"rho_low\": %.6f}%s\n",
                   lens->name, rho, mae, rho_high, rho_low,
                   li < BENCH_NUM_LENSES - 1 ? "," : "");
        } else {
            printf("  %-12s  %10.6f  %10.6f  %10.6f  %10.6f\n",
                   lens->name, rho, mae, rho_high, rho_low);
        }
    }

    if (g_json_output) {
        printf(" ],\n");
        printf(" \"best_lens\": \"%s\", \"best_spearman\": %.6f,\n",
               best_lens, best_rho);

        /* Scatter data (first 200 pairs max for JSON brevity) */
        int scatter_n = n < 200 ? n : 200;
        printf(" \"scatter\": [\n");
        /* Recompute with best lens for scatter */
        for (int li = 0; li < BENCH_NUM_LENSES; li++) {
            if (strcmp(ALL_LENSES[li]->name, best_lens) == 0) {
                for (int i = 0; i < scatter_n; i++) {
                    trine_scores[i] = lens_cosine(
                        emb1 + (size_t)i * BENCH_DIMS,
                        emb2 + (size_t)i * BENCH_DIMS,
                        ALL_LENSES[li]);
                }
                break;
            }
        }
        for (int i = 0; i < scatter_n; i++) {
            printf("  [%.4f, %.4f]%s\n",
                   human_scores[i], trine_scores[i],
                   i < scatter_n - 1 ? "," : "");
        }
        printf(" ]\n}\n");
    } else {
        printf("\n");
        printf("  Best lens: %s (rho = %.6f)\n", best_lens, best_rho);

        /* Print scatter data (first 20 pairs as sample) */
        int scatter_n = n < 20 ? n : 20;
        printf("\n  Sample scatter (human vs TRINE, %s lens):\n",
               best_lens);
        printf("  %8s  %8s  %s\n", "Human", "TRINE", "ID");

        for (int li = 0; li < BENCH_NUM_LENSES; li++) {
            if (strcmp(ALL_LENSES[li]->name, best_lens) == 0) {
                for (int i = 0; i < scatter_n; i++) {
                    double ts = lens_cosine(
                        emb1 + (size_t)i * BENCH_DIMS,
                        emb2 + (size_t)i * BENCH_DIMS,
                        ALL_LENSES[li]);
                    printf("  %8.4f  %8.4f  %s\n",
                           human_scores[i], ts,
                           corpus.pairs[i].id ? corpus.pairs[i].id : "-");
                }
                break;
            }
        }
    }

    free(emb1);
    free(emb2);
    free(human_scores);
    free(trine_scores);
    sts_corpus_free(&corpus);
}

/* =====================================================================
 * Mode 2: Dedup Precision/Recall Benchmark
 * =====================================================================
 *
 * Load corpus with ground-truth groups and is_duplicate flags.
 * For each threshold, compute TP, FP, FN, precision, recall, F1.
 * Optionally test with canonicalization.
 *
 * ===================================================================== */

typedef struct {
    float threshold;
    int tp;
    int fp;
    int fn;
    double precision;
    double recall;
    double f1;
} dedup_threshold_result_t;

/*
 * same_group — Check if two entries belong to the same group.
 * Two entries with NULL group are NOT considered same group.
 */
static int same_group(const dedup_entry_t *a, const dedup_entry_t *b)
{
    if (!a->group || !b->group) return 0;
    return strcmp(a->group, b->group) == 0;
}

/*
 * run_dedup_sweep — Run precision/recall sweep at all thresholds.
 * Uses pre-computed embeddings from the corpus.
 * Results are written into the results array.
 */
static void run_dedup_sweep(const dedup_corpus_t *corpus,
                             const bench_lens_t *lens,
                             dedup_threshold_result_t *results)
{
    int n = corpus->count;

    for (int ti = 0; ti < BENCH_NUM_THRESHOLDS; ti++) {
        float thresh = BENCH_THRESHOLDS[ti];
        int tp = 0, fp = 0, fn = 0;

        /* For each entry marked is_duplicate, check if we find its match */
        for (int i = 0; i < n; i++) {
            if (!corpus->entries[i].is_duplicate) continue;

            /* This entry is a known duplicate. Find best match in corpus. */
            double best_sim = -1.0;
            int best_idx = -1;

            for (int j = 0; j < n; j++) {
                if (j == i) continue;

                double sim = lens_cosine(corpus->entries[i].emb,
                                          corpus->entries[j].emb,
                                          lens);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_idx = j;
                }
            }

            if (best_sim >= thresh) {
                if (best_idx >= 0 &&
                    same_group(&corpus->entries[i],
                               &corpus->entries[best_idx])) {
                    tp++;
                } else {
                    fp++;
                }
            } else {
                fn++;
            }
        }

        /* Also count non-duplicate entries that get false matched */
        for (int i = 0; i < n; i++) {
            if (corpus->entries[i].is_duplicate) continue;

            for (int j = i + 1; j < n; j++) {
                if (corpus->entries[j].is_duplicate) continue;

                double sim = lens_cosine(corpus->entries[i].emb,
                                          corpus->entries[j].emb,
                                          lens);
                if (sim >= thresh &&
                    !same_group(&corpus->entries[i],
                                &corpus->entries[j])) {
                    fp++;
                }
            }
        }

        results[ti].threshold = thresh;
        results[ti].tp = tp;
        results[ti].fp = fp;
        results[ti].fn = fn;

        double prec = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
        double rec  = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
        double f1   = (prec + rec > 0.0)
                        ? 2.0 * prec * rec / (prec + rec)
                        : 0.0;

        results[ti].precision = prec;
        results[ti].recall    = rec;
        results[ti].f1        = f1;
    }
}

static void run_dedup_benchmark(const char *path, int canon_preset)
{
    dedup_corpus_t corpus;
    dedup_corpus_init(&corpus);

    if (dedup_corpus_load(&corpus, path) <= 0) {
        fprintf(stderr, "error: no valid dedup entries loaded\n");
        dedup_corpus_free(&corpus);
        return;
    }

    /* If canonicalization requested, re-encode with canon */
    dedup_corpus_t canon_corpus;
    dedup_corpus_init(&canon_corpus);
    int has_canon = 0;

    if (canon_preset != TRINE_CANON_NONE) {
        has_canon = 1;
        /* Duplicate the corpus with canonicalized text */
        canon_corpus.entries = (dedup_entry_t *)malloc(
            (size_t)corpus.count * sizeof(dedup_entry_t));
        if (!canon_corpus.entries) {
            fprintf(stderr, "error: allocation failed for canon corpus\n");
            dedup_corpus_free(&corpus);
            return;
        }
        canon_corpus.count = corpus.count;
        canon_corpus.capacity = corpus.count;

        char canon_buf[BENCH_MAX_LINE];

        for (int i = 0; i < corpus.count; i++) {
            dedup_entry_t *src = &corpus.entries[i];
            dedup_entry_t *dst = &canon_corpus.entries[i];

            dst->id = src->id ? strdup(src->id) : NULL;
            dst->group = src->group ? strdup(src->group) : NULL;
            dst->is_duplicate = src->is_duplicate;

            size_t canon_len = 0;
            size_t text_len = strlen(src->text);
            trine_canon_apply(src->text, text_len, canon_preset,
                               canon_buf, sizeof(canon_buf), &canon_len);
            dst->text = strdup(canon_buf);

            trine_encode_shingle(canon_buf, canon_len, dst->emb);
        }
    }

    int n = corpus.count;
    int num_dups = 0;
    for (int i = 0; i < n; i++) {
        if (corpus.entries[i].is_duplicate) num_dups++;
    }

    if (g_json_output) {
        printf("{\"mode\": \"dedup\", \"entries\": %d, "
               "\"duplicates\": %d,\n", n, num_dups);
    } else {
        print_header("DEDUP BENCHMARK — Precision/Recall Sweep");
        printf("  Entries:     %d\n", n);
        printf("  Duplicates:  %d\n", num_dups);
        printf("  Originals:   %d\n", n - num_dups);
    }

    /* Run with DEDUP lens (primary) */
    dedup_threshold_result_t results[BENCH_NUM_THRESHOLDS];
    double t0 = now_sec();
    run_dedup_sweep(&corpus, &LENS_DEDUP, results);
    double sweep_time = now_sec() - t0;

    if (g_json_output) {
        printf(" \"lens\": \"dedup\", \"sweep_time_ms\": %.2f,\n",
               sweep_time * 1000.0);
        printf(" \"thresholds\": [\n");
    } else {
        printf("  Lens:        dedup\n");
        printf("  Sweep time:  %.2f ms\n\n", sweep_time * 1000.0);
        printf("  %9s  %6s  %6s  %6s  %9s  %9s  %9s\n",
               "Threshold", "TP", "FP", "FN",
               "Precision", "Recall", "F1");
        print_separator();
    }

    for (int ti = 0; ti < BENCH_NUM_THRESHOLDS; ti++) {
        dedup_threshold_result_t *r = &results[ti];
        if (g_json_output) {
            printf("  {\"threshold\": %.2f, \"tp\": %d, \"fp\": %d, "
                   "\"fn\": %d, \"precision\": %.6f, \"recall\": %.6f, "
                   "\"f1\": %.6f}%s\n",
                   r->threshold, r->tp, r->fp, r->fn,
                   r->precision, r->recall, r->f1,
                   ti < BENCH_NUM_THRESHOLDS - 1 ? "," : "");
        } else {
            printf("  %9.2f  %6d  %6d  %6d  %9.4f  %9.4f  %9.4f\n",
                   r->threshold, r->tp, r->fp, r->fn,
                   r->precision, r->recall, r->f1);
        }
    }

    /* Canonicalization comparison */
    if (has_canon) {
        dedup_threshold_result_t canon_results[BENCH_NUM_THRESHOLDS];
        run_dedup_sweep(&canon_corpus, &LENS_DEDUP, canon_results);

        if (g_json_output) {
            printf(" ],\n");
            printf(" \"canon_preset\": \"%s\",\n",
                   trine_canon_preset_name(canon_preset));
            printf(" \"canon_thresholds\": [\n");
        } else {
            printf("\n");
            printf("  With canonicalization: %s\n",
                   trine_canon_preset_name(canon_preset));
            printf("  %9s  %9s  %9s  %9s  %12s\n",
                   "Threshold", "Precision", "Recall", "F1",
                   "F1-Improve");
            print_separator();
        }

        for (int ti = 0; ti < BENCH_NUM_THRESHOLDS; ti++) {
            dedup_threshold_result_t *r = &canon_results[ti];
            double f1_delta = r->f1 - results[ti].f1;

            if (g_json_output) {
                printf("  {\"threshold\": %.2f, \"precision\": %.6f, "
                       "\"recall\": %.6f, \"f1\": %.6f, "
                       "\"f1_delta\": %.6f}%s\n",
                       r->threshold, r->precision, r->recall,
                       r->f1, f1_delta,
                       ti < BENCH_NUM_THRESHOLDS - 1 ? "," : "");
            } else {
                printf("  %9.2f  %9.4f  %9.4f  %9.4f  %+12.4f\n",
                       r->threshold, r->precision, r->recall,
                       r->f1, f1_delta);
            }
        }

        /* Free canon corpus */
        for (int i = 0; i < canon_corpus.count; i++) {
            free(canon_corpus.entries[i].id);
            free(canon_corpus.entries[i].text);
            free(canon_corpus.entries[i].group);
        }
        free(canon_corpus.entries);
    }

    if (g_json_output) {
        printf(" ]\n}\n");
    } else {
        printf("\n");
    }

    dedup_corpus_free(&corpus);
}

/* =====================================================================
 * Mode 3: Routing Performance Benchmark
 * =====================================================================
 *
 * Compare brute-force vs routed queries across recall modes.
 * Measure candidate reduction, recall vs brute-force, latency.
 *
 * ===================================================================== */

typedef struct {
    const char *mode_name;
    int recall_mode;
    double avg_candidates;
    double candidate_ratio;
    double recall_vs_brute;
    double p50_us;
    double p95_us;
    double p99_us;
    double total_time_ms;
} routing_result_t;

/*
 * run_routing_at_mode — Benchmark routing at a specific recall mode.
 * Builds a routed index, queries every entry, measures recall vs brute.
 */
static void run_routing_at_mode(const dedup_corpus_t *corpus,
                                 int recall_mode,
                                 const char *mode_name,
                                 const int *brute_matches,
                                 routing_result_t *result)
{
    int n = corpus->count;

    trine_s1_config_t config = TRINE_S1_CONFIG_DEFAULT;
    trine_route_t *rt = trine_route_create(&config);
    if (!rt) {
        fprintf(stderr, "error: failed to create routed index\n");
        return;
    }

    trine_route_set_recall(rt, recall_mode);

    /* Add all entries to routed index */
    for (int i = 0; i < n; i++) {
        trine_route_add(rt, corpus->entries[i].emb,
                         corpus->entries[i].id);
    }

    /* Query every entry and measure */
    double *latencies = (double *)malloc((size_t)n * sizeof(double));
    double total_candidates = 0.0;
    int recall_hits = 0;

    if (!latencies) {
        fprintf(stderr, "error: allocation failed for latencies\n");
        trine_route_free(rt);
        return;
    }

    double total_t0 = now_sec();

    for (int i = 0; i < n; i++) {
        trine_route_stats_t stats;
        memset(&stats, 0, sizeof(stats));

        double q_t0 = now_usec();
        trine_s1_result_t res = trine_route_query(rt,
                                                    corpus->entries[i].emb,
                                                    &stats);
        double q_t1 = now_usec();

        latencies[i] = q_t1 - q_t0;
        total_candidates += stats.candidates_checked;

        /* Check if routed match agrees with brute-force */
        if (res.matched_index == brute_matches[i]) {
            recall_hits++;
        }
    }

    double total_time = now_sec() - total_t0;

    /* Sort latencies for percentiles */
    qsort(latencies, (size_t)n, sizeof(double), double_cmp);

    result->mode_name       = mode_name;
    result->recall_mode     = recall_mode;
    result->avg_candidates  = total_candidates / n;
    result->candidate_ratio = total_candidates / ((double)n * n);
    result->recall_vs_brute = (double)recall_hits / n;
    result->p50_us          = percentile(latencies, n, 50.0);
    result->p95_us          = percentile(latencies, n, 95.0);
    result->p99_us          = percentile(latencies, n, 99.0);
    result->total_time_ms   = total_time * 1000.0;

    free(latencies);
    trine_route_free(rt);
}

/*
 * run_routing_scaling — Measure routing performance at various corpus sizes.
 */
typedef struct {
    int corpus_size;
    double avg_candidates;
    double candidate_ratio;
    double recall;
    double p50_us;
} routing_scaling_point_t;

static void run_routing_benchmark(const char *path)
{
    dedup_corpus_t corpus;
    dedup_corpus_init(&corpus);

    if (dedup_corpus_load(&corpus, path) <= 0) {
        fprintf(stderr, "error: no valid routing entries loaded\n");
        dedup_corpus_free(&corpus);
        return;
    }

    int n = corpus.count;

    if (g_json_output) {
        printf("{\"mode\": \"routing\", \"entries\": %d,\n", n);
    } else {
        print_header("ROUTING BENCHMARK — Band-LSH Performance");
        printf("  Entries: %d\n\n", n);
    }

    /* Step 1: Compute brute-force matches as ground truth */
    fprintf(stderr, "[routing] computing brute-force ground truth...\n");

    trine_s1_config_t bf_config = TRINE_S1_CONFIG_DEFAULT;
    trine_s1_index_t *bf_idx = trine_s1_index_create(&bf_config);
    if (!bf_idx) {
        fprintf(stderr, "error: failed to create brute-force index\n");
        dedup_corpus_free(&corpus);
        return;
    }

    for (int i = 0; i < n; i++) {
        trine_s1_index_add(bf_idx, corpus.entries[i].emb,
                            corpus.entries[i].id);
    }

    int *brute_matches = (int *)malloc((size_t)n * sizeof(int));
    if (!brute_matches) {
        fprintf(stderr, "error: allocation failed\n");
        trine_s1_index_free(bf_idx);
        dedup_corpus_free(&corpus);
        return;
    }

    double bf_t0 = now_sec();
    for (int i = 0; i < n; i++) {
        trine_s1_result_t res = trine_s1_index_query(bf_idx,
                                                       corpus.entries[i].emb);
        brute_matches[i] = res.matched_index;
    }
    double bf_time = now_sec() - bf_t0;

    trine_s1_index_free(bf_idx);

    if (!g_json_output) {
        printf("  Brute-force time: %.2f ms (%.0f queries/sec)\n\n",
               bf_time * 1000.0,
               n > 0 ? n / bf_time : 0.0);
    }

    /* Step 2: Test each recall mode */
    static const struct { int mode; const char *name; } recall_modes[] = {
        { TRINE_RECALL_FAST,     "FAST"     },
        { TRINE_RECALL_BALANCED, "BALANCED" },
        { TRINE_RECALL_STRICT,   "STRICT"   }
    };
    int num_modes = 3;

    routing_result_t mode_results[3];
    memset(mode_results, 0, sizeof(mode_results));

    for (int mi = 0; mi < num_modes; mi++) {
        fprintf(stderr, "[routing] testing %s mode...\n",
                recall_modes[mi].name);
        run_routing_at_mode(&corpus, recall_modes[mi].mode,
                             recall_modes[mi].name,
                             brute_matches, &mode_results[mi]);
    }

    if (g_json_output) {
        printf(" \"brute_force_ms\": %.2f,\n", bf_time * 1000.0);
        printf(" \"recall_modes\": [\n");
    } else {
        printf("  %-10s  %10s  %10s  %10s  %8s  %8s  %8s\n",
               "Mode", "AvgCands", "CandRatio", "Recall",
               "p50(us)", "p95(us)", "p99(us)");
        print_separator();
    }

    for (int mi = 0; mi < num_modes; mi++) {
        routing_result_t *r = &mode_results[mi];
        if (g_json_output) {
            printf("  {\"mode\": \"%s\", \"avg_candidates\": %.1f, "
                   "\"candidate_ratio\": %.4f, \"recall\": %.4f, "
                   "\"p50_us\": %.1f, \"p95_us\": %.1f, "
                   "\"p99_us\": %.1f, \"total_ms\": %.2f}%s\n",
                   r->mode_name, r->avg_candidates, r->candidate_ratio,
                   r->recall_vs_brute, r->p50_us, r->p95_us,
                   r->p99_us, r->total_time_ms,
                   mi < num_modes - 1 ? "," : "");
        } else {
            printf("  %-10s  %10.1f  %10.4f  %10.4f  %8.1f  %8.1f  %8.1f\n",
                   r->mode_name, r->avg_candidates, r->candidate_ratio,
                   r->recall_vs_brute, r->p50_us, r->p95_us, r->p99_us);
        }
    }

    /* Step 3: Scaling curve (BALANCED mode at various sizes) */
    if (g_json_output) {
        printf(" ],\n \"scaling\": [\n");
    } else {
        printf("\n");
        print_header("ROUTING SCALING — Candidate Ratio vs Corpus Size (BALANCED)");
        printf("  %10s  %10s  %10s  %10s  %8s\n",
               "Size", "AvgCands", "CandRatio", "Recall", "p50(us)");
        print_separator();
    }

    int scaling_count = 0;

    for (int si = 0; si < BENCH_NUM_SUBSAMPLE; si++) {
        int sub_size = BENCH_SUBSAMPLE_SIZES[si];
        if (sub_size > n) continue;

        /* Build sub-corpus (use first sub_size entries) */
        dedup_corpus_t sub;
        sub.entries = corpus.entries;  /* share, do not free */
        sub.count = sub_size;
        sub.capacity = sub_size;

        /* Compute brute-force for subcorpus */
        trine_s1_config_t sub_config = TRINE_S1_CONFIG_DEFAULT;
        trine_s1_index_t *sub_bf = trine_s1_index_create(&sub_config);
        if (!sub_bf) continue;

        for (int i = 0; i < sub_size; i++) {
            trine_s1_index_add(sub_bf, corpus.entries[i].emb,
                                corpus.entries[i].id);
        }

        int *sub_matches = (int *)malloc((size_t)sub_size * sizeof(int));
        if (!sub_matches) {
            trine_s1_index_free(sub_bf);
            continue;
        }

        for (int i = 0; i < sub_size; i++) {
            trine_s1_result_t res = trine_s1_index_query(sub_bf,
                corpus.entries[i].emb);
            sub_matches[i] = res.matched_index;
        }
        trine_s1_index_free(sub_bf);

        routing_result_t sub_result;
        memset(&sub_result, 0, sizeof(sub_result));

        fprintf(stderr, "[routing] scaling test at N=%d...\n", sub_size);
        run_routing_at_mode(&sub, TRINE_RECALL_BALANCED, "BALANCED",
                             sub_matches, &sub_result);

        free(sub_matches);

        if (g_json_output) {
            printf("  {\"size\": %d, \"avg_candidates\": %.1f, "
                   "\"candidate_ratio\": %.4f, \"recall\": %.4f, "
                   "\"p50_us\": %.1f}%s\n",
                   sub_size, sub_result.avg_candidates,
                   sub_result.candidate_ratio,
                   sub_result.recall_vs_brute, sub_result.p50_us,
                   (si < BENCH_NUM_SUBSAMPLE - 1 &&
                    BENCH_SUBSAMPLE_SIZES[si + 1] <= n) ? "," : "");
        } else {
            printf("  %10d  %10.1f  %10.4f  %10.4f  %8.1f\n",
                   sub_size, sub_result.avg_candidates,
                   sub_result.candidate_ratio,
                   sub_result.recall_vs_brute, sub_result.p50_us);
        }

        scaling_count++;
    }

    if (g_json_output) {
        printf(" ]\n}\n");
    } else if (scaling_count == 0) {
        printf("  (corpus too small for scaling tests)\n");
    }

    free(brute_matches);
    dedup_corpus_free(&corpus);
}

/* =====================================================================
 * Mode 4: Cost Equivalence Calculator
 * =====================================================================
 *
 * Compare cost of neural-only pipeline vs TRINE stage-1 pre-filter.
 *
 * Neural-only:
 *   - Embed all N documents: N * embedding_cost
 *   - Query each against all: N * N * db_cost_per_query
 *
 * TRINE stage-1 + neural stage-2:
 *   - TRINE encode: ~0 (deterministic, CPU-only, ~4M/sec)
 *   - TRINE stage-1 query: ~0 (CPU-only, linear scan)
 *   - Neural stage-2: ~40 candidates per query * embedding_cost
 *
 * ===================================================================== */

static void run_cost_calculator(int docs, int chunk_size,
                                 double embedding_cost,
                                 double db_cost)
{
    /* Total chunks */
    int chunks = docs * (1 + (chunk_size > 0 ? 1000 / chunk_size : 1));
    if (chunk_size <= 0) chunks = docs;

    /* Neural-only cost */
    double neural_embed_cost = chunks * embedding_cost;
    double neural_query_cost = chunks * chunks * db_cost;
    double neural_total = neural_embed_cost + neural_query_cost;

    /* TRINE stage-1: zero cost for encoding + queries (CPU-only) */
    double trine_encode_cost = 0.0;
    double trine_query_cost  = 0.0;

    /* TRINE stage-2: typically ~40 candidates per query need neural embed */
    int candidates_per_query = 40;
    double trine_stage2_cost = (double)chunks * candidates_per_query
                                * embedding_cost;
    double trine_total = trine_encode_cost + trine_query_cost
                          + trine_stage2_cost;

    /* Savings */
    double savings_pct = (neural_total > 0.0)
                          ? (1.0 - trine_total / neural_total) * 100.0
                          : 0.0;
    double savings_abs = neural_total - trine_total;

    if (g_json_output) {
        printf("{\"mode\": \"cost\",\n");
        printf(" \"inputs\": {\"docs\": %d, \"chunk_size\": %d, "
               "\"embedding_cost\": %.8f, \"db_cost\": %.8f},\n",
               docs, chunk_size, embedding_cost, db_cost);
        printf(" \"chunks\": %d,\n", chunks);
        printf(" \"neural_only\": {\"embed_cost\": %.6f, "
               "\"query_cost\": %.6f, \"total\": %.6f},\n",
               neural_embed_cost, neural_query_cost, neural_total);
        printf(" \"trine_hybrid\": {\"encode_cost\": %.6f, "
               "\"query_cost\": %.6f, \"stage2_cost\": %.6f, "
               "\"total\": %.6f},\n",
               trine_encode_cost, trine_query_cost,
               trine_stage2_cost, trine_total);
        printf(" \"savings\": {\"absolute\": %.6f, \"percent\": %.2f, "
               "\"candidates_per_query\": %d}\n",
               savings_abs, savings_pct, candidates_per_query);
        printf("}\n");
    } else {
        print_header("COST EQUIVALENCE CALCULATOR");
        printf("  Documents:            %d\n", docs);
        printf("  Chunk size:           %d chars\n", chunk_size);
        printf("  Total chunks:         %d\n", chunks);
        printf("  Embedding cost:       $%.8f per call\n", embedding_cost);
        printf("  DB query cost:        $%.8f per query\n", db_cost);
        printf("\n");

        printf("  NEURAL-ONLY PIPELINE\n");
        printf("    Embed all chunks:   $%.6f\n", neural_embed_cost);
        printf("    N*N DB queries:     $%.6f\n", neural_query_cost);
        printf("    Total:              $%.6f\n", neural_total);
        printf("\n");

        printf("  TRINE STAGE-1 + NEURAL STAGE-2\n");
        printf("    TRINE encode:       $%.6f (CPU-only, free)\n",
               trine_encode_cost);
        printf("    TRINE stage-1:      $%.6f (CPU-only, free)\n",
               trine_query_cost);
        printf("    Neural stage-2:     $%.6f (~%d candidates/query)\n",
               trine_stage2_cost, candidates_per_query);
        printf("    Total:              $%.6f\n", trine_total);
        printf("\n");

        printf("  SAVINGS\n");
        printf("    Absolute:           $%.6f\n", savings_abs);
        printf("    Percentage:         %.1f%%\n", savings_pct);
        printf("\n");

        /* Cost scaling table */
        printf("  SCALING TABLE (embedding=$%.6f, db=$%.8f)\n",
               embedding_cost, db_cost);
        printf("  %10s  %12s  %12s  %10s\n",
               "Docs", "Neural", "TRINE+NN", "Savings");
        print_separator();

        int scale_sizes[] = {100, 500, 1000, 5000, 10000, 50000};
        int num_scales = 6;

        for (int si = 0; si < num_scales; si++) {
            int nd = scale_sizes[si];
            double n_embed = nd * embedding_cost;
            double n_query = (double)nd * nd * db_cost;
            double n_total = n_embed + n_query;

            double t_s2 = (double)nd * candidates_per_query * embedding_cost;
            double t_total = t_s2;

            double sv = (n_total > 0.0)
                         ? (1.0 - t_total / n_total) * 100.0 : 0.0;

            printf("  %10d  $%11.4f  $%11.4f  %9.1f%%\n",
                   nd, n_total, t_total, sv);
        }
    }
}

/* =====================================================================
 * Lens Parsing
 * ===================================================================== */

static int parse_canon_preset(const char *name)
{
    if (!name) return TRINE_CANON_NONE;
    if (strcmp(name, "support") == 0) return TRINE_CANON_SUPPORT;
    if (strcmp(name, "code") == 0)    return TRINE_CANON_CODE;
    if (strcmp(name, "policy") == 0)  return TRINE_CANON_POLICY;
    if (strcmp(name, "general") == 0) return TRINE_CANON_GENERAL;
    if (strcmp(name, "none") == 0)    return TRINE_CANON_NONE;
    return -1;
}

/* =====================================================================
 * Usage
 * ===================================================================== */

static void print_usage(const char *progname)
{
    fprintf(stderr,
        "TRINE Corpus Benchmark v%s\n"
        "\n"
        "Usage:\n"
        "  %s --sts <file.jsonl>       STS correlation benchmark\n"
        "  %s --dedup <file.jsonl>     Dedup precision/recall benchmark\n"
        "  %s --routing <file.jsonl>   Routing performance benchmark\n"
        "  %s --cost --docs N          Cost equivalence calculator\n"
        "\n"
        "Options:\n"
        "  --json                       Machine-readable JSON output\n"
        "  --canon <preset>             Canon preset for dedup (support/code/policy/general)\n"
        "  --docs <N>                   Number of documents (cost mode)\n"
        "  --chunk-size <S>             Chunk size in chars (cost mode, default 500)\n"
        "  --embedding-cost <C>         Cost per embedding call (cost mode, default 0.0001)\n"
        "  --db-cost <D>                Cost per DB query (cost mode, default 0.00001)\n"
        "\n"
        "STS JSONL format:\n"
        "  {\"id\": \"...\", \"text1\": \"...\", \"text2\": \"...\", \"score\": 4.2, \"label\": \"high\"}\n"
        "\n"
        "Dedup/Routing JSONL format:\n"
        "  {\"id\": \"...\", \"text\": \"...\", \"group\": \"...\", \"is_duplicate\": false}\n"
        "\n",
        BENCH_VERSION, progname, progname, progname, progname);
}

/* =====================================================================
 * Main
 * ===================================================================== */

int main(int argc, char **argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Mode flags */
    const char *sts_path     = NULL;
    const char *dedup_path   = NULL;
    const char *routing_path = NULL;
    int cost_mode            = 0;

    /* Cost parameters */
    int cost_docs            = 1000;
    int cost_chunk_size      = 500;
    double cost_embed        = 0.0001;
    double cost_db           = 0.00001;

    /* Canon preset for dedup */
    int canon_preset         = TRINE_CANON_NONE;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sts") == 0 && i + 1 < argc) {
            sts_path = argv[++i];
        } else if (strcmp(argv[i], "--dedup") == 0 && i + 1 < argc) {
            dedup_path = argv[++i];
        } else if (strcmp(argv[i], "--routing") == 0 && i + 1 < argc) {
            routing_path = argv[++i];
        } else if (strcmp(argv[i], "--cost") == 0) {
            cost_mode = 1;
        } else if (strcmp(argv[i], "--json") == 0) {
            g_json_output = 1;
        } else if (strcmp(argv[i], "--canon") == 0 && i + 1 < argc) {
            i++;
            canon_preset = parse_canon_preset(argv[i]);
            if (canon_preset < 0) {
                fprintf(stderr, "error: unknown canon preset: %s\n",
                        argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--docs") == 0 && i + 1 < argc) {
            cost_docs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chunk-size") == 0 && i + 1 < argc) {
            cost_chunk_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--embedding-cost") == 0 && i + 1 < argc) {
            cost_embed = atof(argv[++i]);
        } else if (strcmp(argv[i], "--db-cost") == 0 && i + 1 < argc) {
            cost_db = atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    int modes_active = (sts_path != NULL) + (dedup_path != NULL) +
                       (routing_path != NULL) + cost_mode;
    if (modes_active == 0) {
        fprintf(stderr, "error: no benchmark mode specified\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!g_json_output) {
        printf("TRINE Corpus Benchmark v%s\n", BENCH_VERSION);
        printf("  Stage-1 API: v%s\n", TRINE_S1_VERSION);
        printf("  Route API:   v%s\n", TRINE_ROUTE_VERSION);
        printf("  Canon API:   v%s\n", TRINE_CANON_VERSION);
    }

    if (sts_path)     run_sts_benchmark(sts_path);
    if (dedup_path)   run_dedup_benchmark(dedup_path, canon_preset);
    if (routing_path) run_routing_benchmark(routing_path);
    if (cost_mode)    run_cost_calculator(cost_docs, cost_chunk_size,
                                          cost_embed, cost_db);

    return 0;
}
