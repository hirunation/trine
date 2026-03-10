/* bench_semantic.c — Stage-1 baseline benchmark on semantic datasets.
 *
 * Reads JSONL pair files, encodes both texts with TRINE Stage-1,
 * computes lens-weighted cosine similarity, and reports Spearman rho
 * for continuous-score datasets and F1 for binary-label datasets.
 *
 * Usage: bench_semantic <jsonl_file> [--lens NAME] [--threshold F]
 *
 * Output: per-dataset metrics in both human-readable and JSON format.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "trine_stage1.h"
#include "trine_encode.h"

#define MAX_TEXT 4096
#define MAX_PAIRS 200000
#define MAX_LINE 16384

/* Simple JSONL field extraction (no dependency on JSON library) */
static int extract_string(const char *json, const char *key, char *out, int max) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == ' ') p++;
    if (*p != '"') return -1;
    p++; /* skip opening quote */
    int i = 0;
    while (*p && *p != '"' && i < max - 1) {
        if (*p == '\\' && *(p+1)) {
            p++; /* skip escape */
            if (*p == 'n') out[i++] = '\n';
            else if (*p == 't') out[i++] = '\t';
            else if (*p == '\\') out[i++] = '\\';
            else if (*p == '"') out[i++] = '"';
            else out[i++] = *p;
        } else {
            out[i++] = *p;
        }
        p++;
    }
    out[i] = '\0';
    return i;
}

static int extract_float(const char *json, const char *key, float *out) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':') p++;
    *out = (float)atof(p);
    return 0;
}

/* Pair data */
typedef struct {
    float human_score;     /* Normalized 0-1 */
    float trine_score;     /* TRINE Stage-1 similarity */
    char label[16];        /* "similar" / "dissimilar" / "neutral" */
    char source[16];       /* dataset name */
} pair_t;

static pair_t pairs[MAX_PAIRS];
static int pair_count = 0;

/* Rank data for Spearman */
static void rank_array(const float *vals, double *ranks, int n) {
    int *indices = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) indices[i] = i;

    /* Simple insertion sort — fine for <200K */
    for (int i = 1; i < n; i++) {
        int key = indices[i];
        int j = i - 1;
        while (j >= 0 && vals[indices[j]] > vals[key]) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }

    /* Assign ranks with tie handling (average rank) */
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n - 1 && vals[indices[j]] == vals[indices[j + 1]])
            j++;
        double avg_rank = (i + j) / 2.0 + 1.0;
        for (int k = i; k <= j; k++)
            ranks[indices[k]] = avg_rank;
        i = j + 1;
    }
    free(indices);
}

static double spearman_rho(const float *x, const float *y, int n) {
    if (n < 2) return 0.0;
    double *rx = malloc(n * sizeof(double));
    double *ry = malloc(n * sizeof(double));
    rank_array(x, rx, n);
    rank_array(y, ry, n);

    double sum_d2 = 0.0;
    for (int i = 0; i < n; i++) {
        double d = rx[i] - ry[i];
        sum_d2 += d * d;
    }
    free(rx);
    free(ry);
    return 1.0 - (6.0 * sum_d2) / ((double)n * ((double)n * n - 1.0));
}

/* F1 computation for binary classification */
typedef struct {
    int tp, fp, fn, tn;
    double precision, recall, f1;
} binary_metrics_t;

static binary_metrics_t compute_f1(const pair_t *p, int n, float threshold) {
    binary_metrics_t m = {0};
    for (int i = 0; i < n; i++) {
        int predicted = p[i].trine_score >= threshold;
        int actual = strcmp(p[i].label, "similar") == 0;
        if (predicted && actual) m.tp++;
        else if (predicted && !actual) m.fp++;
        else if (!predicted && actual) m.fn++;
        else m.tn++;
    }
    m.precision = m.tp + m.fp > 0 ? (double)m.tp / (m.tp + m.fp) : 0.0;
    m.recall = m.tp + m.fn > 0 ? (double)m.tp / (m.tp + m.fn) : 0.0;
    m.f1 = m.precision + m.recall > 0 ? 2.0 * m.precision * m.recall / (m.precision + m.recall) : 0.0;
    return m;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <jsonl_file> [--lens NAME] [--threshold F] [--json]\n", argv[0]);
        return 1;
    }

    const char *jsonl_path = argv[1];
    float threshold = 0.60f;
    int json_output = 0;
    trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;
    const char *lens_name = "uniform";

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = 1;
        } else if (strcmp(argv[i], "--lens") == 0 && i + 1 < argc) {
            i++;
            lens_name = argv[i];
            if (strcmp(argv[i], "uniform") == 0) {
                trine_s1_lens_t l = TRINE_S1_LENS_UNIFORM; lens = l;
            } else if (strcmp(argv[i], "dedup") == 0) {
                trine_s1_lens_t l = TRINE_S1_LENS_DEDUP; lens = l;
            } else if (strcmp(argv[i], "vocab") == 0) {
                trine_s1_lens_t l = TRINE_S1_LENS_VOCAB; lens = l;
            } else if (strcmp(argv[i], "edit") == 0) {
                trine_s1_lens_t l = TRINE_S1_LENS_EDIT; lens = l;
            } else if (strcmp(argv[i], "code") == 0) {
                trine_s1_lens_t l = TRINE_S1_LENS_CODE; lens = l;
            } else if (strcmp(argv[i], "legal") == 0) {
                trine_s1_lens_t l = TRINE_S1_LENS_LEGAL; lens = l;
            } else {
                fprintf(stderr, "Unknown lens: %s\n", argv[i]);
                return 1;
            }
        }
    }

    /* Read JSONL file */
    FILE *fp = fopen(jsonl_path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", jsonl_path);
        return 1;
    }

    char line[MAX_LINE];
    char text_a[MAX_TEXT], text_b[MAX_TEXT];
    uint8_t emb_a[240], emb_b[240];
    int errors = 0;

    while (fgets(line, sizeof(line), fp) && pair_count < MAX_PAIRS) {
        if (extract_string(line, "text_a", text_a, MAX_TEXT) < 0) continue;
        if (extract_string(line, "text_b", text_b, MAX_TEXT) < 0) continue;
        float score;
        if (extract_float(line, "score", &score) < 0) continue;
        char label[16] = "neutral", source[16] = "unknown";
        extract_string(line, "label", label, sizeof(label));
        extract_string(line, "source", source, sizeof(source));

        /* Encode both texts */
        trine_s1_encode(text_a, strlen(text_a), emb_a);
        trine_s1_encode(text_b, strlen(text_b), emb_b);

        /* Compute similarity */
        float sim = trine_s1_compare(emb_a, emb_b, &lens);
        if (sim < 0.0f) { errors++; continue; }

        pairs[pair_count].human_score = score;
        pairs[pair_count].trine_score = sim;
        strncpy(pairs[pair_count].label, label, 15);
        pairs[pair_count].label[15] = '\0';
        strncpy(pairs[pair_count].source, source, 15);
        pairs[pair_count].source[15] = '\0';
        pair_count++;
    }
    fclose(fp);

    if (pair_count == 0) {
        fprintf(stderr, "No valid pairs found in %s\n", jsonl_path);
        return 1;
    }

    /* Overall Spearman rho */
    float *human = malloc(pair_count * sizeof(float));
    float *trine = malloc(pair_count * sizeof(float));
    for (int i = 0; i < pair_count; i++) {
        human[i] = pairs[i].human_score;
        trine[i] = pairs[i].trine_score;
    }
    double rho = spearman_rho(human, trine, pair_count);
    binary_metrics_t f1_all = compute_f1(pairs, pair_count, threshold);

    /* Per-source breakdown */
    char sources[10][16];
    int source_count = 0;
    for (int i = 0; i < pair_count; i++) {
        int found = 0;
        for (int j = 0; j < source_count; j++) {
            if (strcmp(sources[j], pairs[i].source) == 0) { found = 1; break; }
        }
        if (!found && source_count < 10) {
            strncpy(sources[source_count], pairs[i].source, 15);
            sources[source_count][15] = '\0';
            source_count++;
        }
    }

    if (json_output) {
        printf("{\n");
        printf("  \"file\": \"%s\",\n", jsonl_path);
        printf("  \"lens\": \"%s\",\n", lens_name);
        printf("  \"threshold\": %.3f,\n", threshold);
        printf("  \"total_pairs\": %d,\n", pair_count);
        printf("  \"overall_spearman_rho\": %.4f,\n", rho);
        printf("  \"overall_f1\": %.4f,\n", f1_all.f1);
        printf("  \"per_source\": {\n");
    } else {
        printf("=== TRINE Stage-1 Baseline ===\n");
        printf("File:      %s\n", jsonl_path);
        printf("Lens:      %s\n", lens_name);
        printf("Threshold: %.3f\n", threshold);
        printf("Pairs:     %d\n", pair_count);
        printf("Errors:    %d\n", errors);
        printf("\n--- Overall ---\n");
        printf("Spearman rho: %.4f\n", rho);
        printf("F1 (%.2f):    %.4f  (P=%.4f R=%.4f)\n", threshold, f1_all.f1, f1_all.precision, f1_all.recall);
        printf("\n--- Per Source ---\n");
    }

    for (int s = 0; s < source_count; s++) {
        /* Filter pairs for this source */
        float *sh = malloc(pair_count * sizeof(float));
        float *st = malloc(pair_count * sizeof(float));
        pair_t *sp = malloc(pair_count * sizeof(pair_t));
        int sn = 0;
        for (int i = 0; i < pair_count; i++) {
            if (strcmp(pairs[i].source, sources[s]) == 0) {
                sh[sn] = pairs[i].human_score;
                st[sn] = pairs[i].trine_score;
                sp[sn] = pairs[i];
                sn++;
            }
        }
        double srho = spearman_rho(sh, st, sn);
        binary_metrics_t sf1 = compute_f1(sp, sn, threshold);

        if (json_output) {
            printf("    \"%s\": {\"pairs\": %d, \"spearman_rho\": %.4f, \"f1\": %.4f, \"precision\": %.4f, \"recall\": %.4f}%s\n",
                   sources[s], sn, srho, sf1.f1, sf1.precision, sf1.recall,
                   s < source_count - 1 ? "," : "");
        } else {
            printf("%-10s  n=%5d  rho=%.4f  F1=%.4f  P=%.4f  R=%.4f\n",
                   sources[s], sn, srho, sf1.f1, sf1.precision, sf1.recall);
        }
        free(sh); free(st); free(sp);
    }

    if (json_output) {
        printf("  }\n}\n");
    }

    free(human); free(trine);
    return 0;
}
