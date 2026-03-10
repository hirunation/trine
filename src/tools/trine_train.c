/* =====================================================================
 * TRINE TRAIN — Hebbian Stage-2 Training CLI
 * =====================================================================
 *
 * Trains a Hebbian ternary Stage-2 model from JSONL text pairs.
 *
 * Usage:
 *   trine_train [options] <data.jsonl>
 *   trine_train [options] --data <data.jsonl> [--val <val.jsonl>]
 *
 * Build:
 *   cc -O2 -Wall -Wextra -Werror \
 *      -Isrc/encode -Isrc/compare -Isrc/index -Isrc/canon \
 *      -Isrc/algebra -Isrc/model -Isrc/stage2/projection \
 *      -Isrc/stage2/cascade -Isrc/stage2/inference \
 *      -Isrc/stage2/hebbian \
 *      -o build/trine_train src/tools/trine_train.c \
 *      src/stage2/hebbian/trine_accumulator.c \
 *      src/stage2/hebbian/trine_freeze.c \
 *      src/stage2/hebbian/trine_hebbian.c \
 *      src/stage2/hebbian/trine_self_deepen.c \
 *      build/libtrine.a -lm
 *
 * ===================================================================== */

#define _POSIX_C_SOURCE 200809L  /* clock_gettime, strdup */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "trine_encode.h"
#include "trine_stage1.h"
#include "trine_stage2.h"
#include "trine_hebbian.h"
#include "trine_s2_persist.h"
#include "trine_accumulator_persist.h"

/* =====================================================================
 * Constants
 * ===================================================================== */

#define TRAIN_VERSION    "1.1.0"
#define TRAIN_MAX_LINE   8192
#define TRAIN_MAX_TEXT   4096

/* =====================================================================
 * Timing helper
 * ===================================================================== */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* =====================================================================
 * JSONL parsing (strstr-based, same pattern as trine_hebbian.c)
 * ===================================================================== */

/* Extract a JSON string value following a key like "text_a".
 * Returns length of extracted string, or -1 if not found. */
static int extract_json_string(const char *line, const char *key,
                                char *buf, size_t buf_size)
{
    if (!line || !key || !buf || buf_size == 0) return -1;

    char pattern[128];
    int plen = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (plen < 0 || (size_t)plen >= sizeof(pattern)) return -1;

    const char *pos = strstr(line, pattern);
    if (!pos) return -1;

    pos += (size_t)plen;
    while (*pos == ' ' || *pos == '\t') pos++;
    if (*pos != ':') return -1;
    pos++;
    while (*pos == ' ' || *pos == '\t') pos++;

    if (*pos != '"') return -1;
    pos++;

    size_t out_len = 0;
    while (*pos && *pos != '"' && out_len < buf_size - 1) {
        if (*pos == '\\' && pos[1]) {
            pos++;
            switch (*pos) {
                case '"':  buf[out_len++] = '"';  break;
                case '\\': buf[out_len++] = '\\'; break;
                case 'n':  buf[out_len++] = '\n'; break;
                case 't':  buf[out_len++] = '\t'; break;
                case 'r':  buf[out_len++] = '\r'; break;
                case '/':  buf[out_len++] = '/';  break;
                default:   buf[out_len++] = *pos; break;
            }
        } else {
            buf[out_len++] = *pos;
        }
        pos++;
    }

    buf[out_len] = '\0';
    return (int)out_len;
}

/* Extract a JSON numeric value following a key like "score".
 * Returns the parsed float, or -1.0 on failure. */
static float extract_json_number(const char *line, const char *key)
{
    if (!line || !key) return -1.0f;

    char pattern[128];
    int plen = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (plen < 0 || (size_t)plen >= sizeof(pattern)) return -1.0f;

    const char *pos = strstr(line, pattern);
    if (!pos) return -1.0f;

    pos += (size_t)plen;
    while (*pos == ' ' || *pos == '\t') pos++;
    if (*pos != ':') return -1.0f;
    pos++;
    while (*pos == ' ' || *pos == '\t') pos++;

    char *end = NULL;
    float val = strtof(pos, &end);
    if (end == pos) return -1.0f;

    return val;
}

/* =====================================================================
 * Uniform cosine similarity (240-trit vectors)
 * ===================================================================== */

/* Centered cosine: treats trits {0,1,2} as {-1,0,+1}.
 * Correct metric for sign-based projection outputs where
 * 0 = negative, 1 = neutral, 2 = positive. */
static float centered_cosine_240(const uint8_t a[240], const uint8_t b[240])
{
    double dot = 0.0, ma = 0.0, mb = 0.0;
    for (int i = 0; i < 240; i++) {
        double va = (double)a[i] - 1.0;
        double vb = (double)b[i] - 1.0;
        dot += va * vb;
        ma  += va * va;
        mb  += vb * vb;
    }
    if (ma < 1e-12 || mb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(ma) * sqrt(mb)));
}

/* =====================================================================
 * Spearman rank correlation
 * ===================================================================== */

/* Comparison helper for qsort: sort by value ascending, preserve index */
typedef struct {
    double value;
    int    index;
} rank_entry_t;

static int rank_cmp(const void *a, const void *b)
{
    double va = ((const rank_entry_t *)a)->value;
    double vb = ((const rank_entry_t *)b)->value;
    if (va < vb) return -1;
    if (va > vb) return  1;
    return 0;
}

/* Compute ranks (1-based, with midrank for ties) into out_ranks.
 * values has n entries. */
static void compute_ranks(const double *values, int n, double *out_ranks)
{
    rank_entry_t *entries = (rank_entry_t *)malloc((size_t)n * sizeof(rank_entry_t));
    if (!entries) return;

    for (int i = 0; i < n; i++) {
        entries[i].value = values[i];
        entries[i].index = i;
    }

    qsort(entries, (size_t)n, sizeof(rank_entry_t), rank_cmp);

    /* Assign ranks with midrank for ties */
    int i = 0;
    while (i < n) {
        int j = i;
        /* Find extent of tied group */
        while (j < n - 1 && entries[j + 1].value == entries[j].value) {
            j++;
        }
        /* Midrank for positions i..j (1-based) */
        double midrank = (double)(i + 1 + j + 1) / 2.0;
        for (int k = i; k <= j; k++) {
            out_ranks[entries[k].index] = midrank;
        }
        i = j + 1;
    }

    free(entries);
}

/* Spearman rho = 1 - 6*sum(d_i^2) / (n*(n^2 - 1)) */
static double spearman_rho(const double *x, const double *y, int n)
{
    if (n < 2) return 0.0;

    double *rank_x = (double *)malloc((size_t)n * sizeof(double));
    double *rank_y = (double *)malloc((size_t)n * sizeof(double));
    if (!rank_x || !rank_y) {
        free(rank_x);
        free(rank_y);
        return 0.0;
    }

    compute_ranks(x, n, rank_x);
    compute_ranks(y, n, rank_y);

    double sum_d2 = 0.0;
    for (int i = 0; i < n; i++) {
        double d = rank_x[i] - rank_y[i];
        sum_d2 += d * d;
    }

    free(rank_x);
    free(rank_y);

    double nn = (double)n;
    return 1.0 - (6.0 * sum_d2) / (nn * (nn * nn - 1.0));
}

/* =====================================================================
 * Count lines in a JSONL file (for per-epoch pair count)
 * ===================================================================== */

static int64_t count_jsonl_pairs(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    char line[TRAIN_MAX_LINE];
    char text_a[TRAIN_MAX_TEXT];
    char text_b[TRAIN_MAX_TEXT];
    int64_t count = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;

        int len_a = extract_json_string(line, "text_a", text_a, sizeof(text_a));
        int len_b = extract_json_string(line, "text_b", text_b, sizeof(text_b));

        if (len_a > 0 && len_b > 0) count++;
    }

    fclose(fp);
    return count;
}

/* =====================================================================
 * Validation evaluation
 * ===================================================================== */

/* Evaluate on validation JSONL: compute Spearman rho for Stage-1 and
 * Stage-2 similarities against human scores.
 *
 * Returns the number of valid pairs evaluated, or -1 on error.
 * Writes Stage-1 and Stage-2 rho to *rho_s1 and *rho_s2. */
static int64_t evaluate_validation(const char *val_path,
                                    const trine_s2_model_t *model,
                                    uint32_t depth,
                                    double *rho_s1, double *rho_s2,
                                    int use_gated)
{
    FILE *fp = fopen(val_path, "r");
    if (!fp) return -1;

    /* First pass: count valid pairs */
    char line[TRAIN_MAX_LINE];
    char text_a[TRAIN_MAX_TEXT];
    char text_b[TRAIN_MAX_TEXT];
    int64_t n_pairs = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;

        int len_a = extract_json_string(line, "text_a", text_a, sizeof(text_a));
        int len_b = extract_json_string(line, "text_b", text_b, sizeof(text_b));
        float score = extract_json_number(line, "score");

        if (len_a > 0 && len_b > 0 && score >= 0.0f) n_pairs++;
    }

    if (n_pairs == 0) {
        fclose(fp);
        return 0;
    }

    /* Allocate arrays */
    double *human_scores = (double *)malloc((size_t)n_pairs * sizeof(double));
    double *s1_scores    = (double *)malloc((size_t)n_pairs * sizeof(double));
    double *s2_scores    = (double *)malloc((size_t)n_pairs * sizeof(double));

    if (!human_scores || !s1_scores || !s2_scores) {
        free(human_scores);
        free(s1_scores);
        free(s2_scores);
        fclose(fp);
        return -1;
    }

    /* Second pass: compute similarities */
    rewind(fp);
    int64_t idx = 0;
    trine_s1_lens_t uniform = TRINE_S1_LENS_UNIFORM;

    while (fgets(line, sizeof(line), fp) && idx < n_pairs) {
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            continue;

        int len_a = extract_json_string(line, "text_a", text_a, sizeof(text_a));
        int len_b = extract_json_string(line, "text_b", text_b, sizeof(text_b));
        float score = extract_json_number(line, "score");

        if (len_a <= 0 || len_b <= 0 || score < 0.0f) continue;

        human_scores[idx] = (double)score;

        /* Stage-1 similarity */
        uint8_t emb_a[240], emb_b[240];
        if (trine_encode_shingle(text_a, (size_t)len_a, emb_a) != 0) continue;
        if (trine_encode_shingle(text_b, (size_t)len_b, emb_b) != 0) continue;
        float sim_s1 = trine_s1_compare(emb_a, emb_b, &uniform);
        s1_scores[idx] = (double)sim_s1;

        /* Stage-2 similarity */
        uint8_t s2_a[240], s2_b[240];
        if (trine_s2_encode(model, text_a, (size_t)len_a, depth, s2_a) != 0 ||
            trine_s2_encode(model, text_b, (size_t)len_b, depth, s2_b) != 0) {
            s2_scores[idx] = (double)sim_s1;
        } else if (use_gated) {
            s2_scores[idx] = (double)trine_s2_compare_gated(model, s2_a, s2_b);
        } else {
            s2_scores[idx] = (double)centered_cosine_240(s2_a, s2_b);
        }

        idx++;
    }

    fclose(fp);

    /* Compute Spearman correlations */
    *rho_s1 = spearman_rho(human_scores, s1_scores, (int)idx);
    *rho_s2 = spearman_rho(human_scores, s2_scores, (int)idx);

    /* Try blended S1+S2 at different weights */
    {
        double *blended = (double *)malloc((size_t)idx * sizeof(double));
        if (blended) {
            double best_blend_rho = *rho_s1;
            double best_alpha = 1.0;
            double alphas[] = {0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6,
                               0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2,
                               0.15, 0.1, 0.05, 0.0};
            for (int a = 0; a < 20; a++) {
                double alpha = alphas[a];
                for (int64_t i2 = 0; i2 < idx; i2++)
                    blended[i2] = alpha * s1_scores[i2] + (1.0-alpha) * s2_scores[i2];
                double blend_rho = spearman_rho(human_scores, blended, (int)idx);
                if (blend_rho > best_blend_rho) {
                    best_blend_rho = blend_rho;
                    best_alpha = alpha;
                }
            }
            fprintf(stderr, "  Best blend: alpha=%.1f -> rho=%.4f (S1*%.0f%% + S2*%.0f%%)\n",
                    best_alpha, best_blend_rho,
                    best_alpha * 100.0, (1.0 - best_alpha) * 100.0);
            free(blended);
        }
    }

    free(human_scores);
    free(s1_scores);
    free(s2_scores);

    return idx;
}

/* =====================================================================
 * Number formatting helper (comma-separated thousands)
 * ===================================================================== */

static void format_number(int64_t n, char *buf, size_t buf_size)
{
    if (n < 0) {
        buf[0] = '-';
        format_number(-n, buf + 1, buf_size - 1);
        return;
    }

    char tmp[64];
    int len = snprintf(tmp, sizeof(tmp), "%lld", (long long)n);
    if (len <= 0 || (size_t)len >= sizeof(tmp)) {
        snprintf(buf, buf_size, "%lld", (long long)n);
        return;
    }

    /* Insert commas */
    int commas = (len - 1) / 3;
    int total = len + commas;
    if ((size_t)total >= buf_size) {
        snprintf(buf, buf_size, "%lld", (long long)n);
        return;
    }

    buf[total] = '\0';
    int src = len - 1;
    int dst = total - 1;
    int digits = 0;

    while (src >= 0) {
        buf[dst--] = tmp[src--];
        digits++;
        if (digits % 3 == 0 && src >= 0) {
            buf[dst--] = ',';
        }
    }
}

/* =====================================================================
 * Usage
 * ===================================================================== */

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "TRINE Training Tool v%s\n"
        "\n"
        "Usage:\n"
        "  %s [options] <data.jsonl>\n"
        "  %s [options] --data <data.jsonl> [--val <val.jsonl>]\n"
        "\n"
        "Options:\n"
        "  --epochs N            Number of training epochs (default: 10)\n"
        "  --threshold T         Freeze threshold (default: auto)\n"
        "  --density D           Target density for auto-threshold (default: 0.33)\n"
        "  --similarity-threshold S  Stage-1 similarity threshold for sign (default: 0.5)\n"
        "  --cells N             Cascade mixing cells (default: 512)\n"
        "  --depth N             Cascade depth (default: 4)\n"
        "  --deepen N            Self-supervised deepening rounds (default: 0)\n"
        "  --diagonal            Use diagonal gating instead of full projection\n"
        "  --block-diagonal      Use block-diagonal projection (4 x 60x60 blocks per chain)\n"
        "  --weighted            Enable weighted Hebbian updates (magnitude scaling)\n"
        "  --pos-scale F         Positive magnitude scale (default: 10.0)\n"
        "  --neg-scale F         Negative magnitude scale (default: 3.0)\n"
        "  --source-weights S    Per-source weights: \"sts:4,sick:3,snli:0.1,...\"\n"
        "  --threshold-schedule S  Comma-separated per-epoch thresholds: \"0.9,0.75,0.6\"\n"
        "  --sparse K            Sparse cross-channel projection (top-K per row)\n"
        "  --stacked             Stacked depth: re-project instead of cascade\n"
        "  --gated               Use gate-aware comparison in evaluation\n"
        "  --save PATH           Save frozen model to .trine2 binary file\n"
        "  --load PATH           Load a saved .trine2 model (skip training)\n"
        "  --save-accum PATH     Save accumulator state to .trine2a file\n"
        "  --load-accum PATH     Load accumulator state (warm-start training)\n"
        "  --output PATH         Save frozen model info to file (default: stdout)\n"
        "  --adaptive-alpha      Compute optimal per-S1-bucket alpha from validation data\n"
        "  --checkpoint-dir DIR  Save a .trine2 checkpoint after each epoch\n"
        "  --quiet               Suppress progress output\n"
        "  --verbose             Show detailed training progress\n"
        "  -h, --help            Show usage\n"
        "\n"
        "The training JSONL must have \"text_a\" and \"text_b\" fields per line.\n"
        "The validation JSONL must additionally have a \"score\" field (0.0-1.0).\n",
        TRAIN_VERSION, prog, prog);
}

/* =====================================================================
 * Main
 * ===================================================================== */

int main(int argc, char **argv)
{
    /* --- Defaults --- */
    const char *data_path   = NULL;
    const char *val_path    = NULL;
    const char *output_path = NULL;
    const char *save_path   = NULL;
    const char *load_path   = NULL;
    const char *save_accum_path = NULL;
    const char *load_accum_path = NULL;
    uint32_t epochs         = 10;
    int32_t  freeze_thresh  = 0;       /* 0 = auto */
    float    density        = 0.33f;
    float    sim_thresh     = 0.5f;
    uint32_t cells          = 512;
    uint32_t depth          = 4;
    uint32_t deepen_rounds  = 0;
    int      diagonal_mode  = 0;
    int      block_diag_mode = 0;
    int      weighted_mode  = 0;
    float    pos_scale      = 10.0f;
    float    neg_scale      = 3.0f;
    const char *source_weights_str = NULL;
    const char *threshold_schedule_str = NULL;
    uint32_t sparse_k       = 0;
    int      stacked_mode   = 0;
    int      gated_eval     = 0;
    int      adaptive_alpha = 0;
    const char *checkpoint_dir = NULL;
    int      quiet          = 0;
    int      verbose        = 0;

    /* --- Argument parsing --- */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            epochs = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            freeze_thresh = (int32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--density") == 0 && i + 1 < argc) {
            density = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--similarity-threshold") == 0 && i + 1 < argc) {
            sim_thresh = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--cells") == 0 && i + 1 < argc) {
            cells = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            depth = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--deepen") == 0 && i + 1 < argc) {
            deepen_rounds = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_path = argv[++i];
        } else if (strcmp(argv[i], "--val") == 0 && i + 1 < argc) {
            val_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            load_path = argv[++i];
        } else if (strcmp(argv[i], "--save-accum") == 0 && i + 1 < argc) {
            save_accum_path = argv[++i];
        } else if (strcmp(argv[i], "--load-accum") == 0 && i + 1 < argc) {
            load_accum_path = argv[++i];
        } else if (strcmp(argv[i], "--diagonal") == 0) {
            diagonal_mode = 1;
        } else if (strcmp(argv[i], "--block-diagonal") == 0) {
            block_diag_mode = 1;
        } else if (strcmp(argv[i], "--weighted") == 0) {
            weighted_mode = 1;
        } else if (strcmp(argv[i], "--pos-scale") == 0 && i + 1 < argc) {
            pos_scale = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--neg-scale") == 0 && i + 1 < argc) {
            neg_scale = strtof(argv[++i], NULL);
        } else if (strcmp(argv[i], "--source-weights") == 0 && i + 1 < argc) {
            source_weights_str = argv[++i];
        } else if (strcmp(argv[i], "--threshold-schedule") == 0 && i + 1 < argc) {
            threshold_schedule_str = argv[++i];
        } else if (strcmp(argv[i], "--sparse") == 0 && i + 1 < argc) {
            sparse_k = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--stacked") == 0) {
            stacked_mode = 1;
        } else if (strcmp(argv[i], "--gated") == 0) {
            gated_eval = 1;
        } else if (strcmp(argv[i], "--adaptive-alpha") == 0) {
            adaptive_alpha = 1;
        } else if (strcmp(argv[i], "--checkpoint-dir") == 0 && i + 1 < argc) {
            checkpoint_dir = argv[++i];
        } else if (strcmp(argv[i], "--quiet") == 0) {
            quiet = 1;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        } else {
            /* Positional argument: treat as data path */
            if (!data_path) {
                data_path = argv[i];
            } else {
                fprintf(stderr, "Error: unexpected argument '%s'\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        }
    }

    /* --- Mutual exclusivity check: --diagonal vs --block-diagonal --- */
    if (diagonal_mode && block_diag_mode) {
        fprintf(stderr, "Error: --diagonal and --block-diagonal are mutually exclusive\n");
        return 1;
    }

    /* --- Load mode: skip training, just load and evaluate --- */
    if (load_path) {
        if (!quiet) {
            fprintf(stderr, "TRINE Training v%s\n", TRAIN_VERSION);
            fprintf(stderr, "Mode:    load\n");
            fprintf(stderr, "Model:   %s\n", load_path);
            if (val_path) {
                fprintf(stderr, "Val:     %s\n", val_path);
            }
            fprintf(stderr, "\n");
        }

        /* Validate first */
        if (trine_s2_validate(load_path) != 0) {
            fprintf(stderr, "Error: '%s' failed validation\n", load_path);
            return 1;
        }

        trine_s2_model_t *model = trine_s2_load(load_path);
        if (!model) {
            fprintf(stderr, "Error: failed to load model from '%s'\n", load_path);
            return 1;
        }

        /* Apply projection mode from CLI flags */
        if (sparse_k > 0) {
            trine_s2_set_projection_mode(model, TRINE_S2_PROJ_SPARSE);
        } else if (block_diag_mode) {
            trine_s2_set_projection_mode(model, TRINE_S2_PROJ_BLOCK_DIAG);
        } else if (diagonal_mode) {
            trine_s2_set_projection_mode(model, TRINE_S2_PROJ_DIAGONAL);
        }
        if (stacked_mode) {
            trine_s2_set_stacked_depth(model, 1);
        }

        if (!quiet) {
            trine_s2_info_t info;
            if (trine_s2_info(model, &info) == 0) {
                fprintf(stderr, "Loaded: %ux%u projection (%u copies) + "
                                "%u-cell cascade%s\n",
                        info.projection_dims, info.projection_dims,
                        info.projection_k, info.cascade_cells,
                        info.is_identity ? " [identity]" : "");
            }
        }

        /* Validation (optional) */
        if (val_path) {
            if (!quiet) {
                fprintf(stderr, "\nValidation: %s\n", val_path);
            }

            double rho_s1 = 0.0, rho_s2 = 0.0;
            int64_t val_pairs = evaluate_validation(val_path, model, depth,
                                                     &rho_s1, &rho_s2,
                                                     gated_eval);

            if (val_pairs < 0) {
                fprintf(stderr, "  Error: cannot read validation file\n");
            } else if (val_pairs == 0) {
                fprintf(stderr, "  Warning: no valid pairs in validation file\n");
            } else if (!quiet) {
                char vpairs_buf[64];
                format_number(val_pairs, vpairs_buf, sizeof(vpairs_buf));
                fprintf(stderr, "  Pairs: %s\n", vpairs_buf);
                fprintf(stderr, "  Spearman rho (Stage-2): %.4f\n", rho_s2);
                fprintf(stderr, "  Spearman rho (Stage-1): %.4f\n", rho_s1);
                fprintf(stderr, "  Improvement: %+.4f\n", rho_s2 - rho_s1);
            }
        }

        trine_s2_free(model);
        return 0;
    }

    if (!data_path) {
        fprintf(stderr, "Error: no training data file specified\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (epochs == 0) {
        fprintf(stderr, "Error: --epochs must be >= 1\n");
        return 1;
    }

    /* --- Print header --- */
    if (!quiet) {
        fprintf(stderr, "TRINE Training v%s\n", TRAIN_VERSION);
        fprintf(stderr, "Data:    %s\n", data_path);
        fprintf(stderr, "Config:  sim_thresh=%.2f, freeze_thresh=%s, density=%.2f, "
                        "cells=%u, depth=%u\n",
                (double)sim_thresh,
                freeze_thresh > 0 ? "manual" : "auto",
                (double)density,
                (unsigned)cells,
                (unsigned)depth);
        fprintf(stderr, "Projection: %s\n",
                block_diag_mode ? "block-diagonal" :
                diagonal_mode   ? "diagonal" : "full");
        if (weighted_mode) {
            fprintf(stderr, "Weighted: pos_scale=%.1f, neg_scale=%.1f\n",
                    (double)pos_scale, (double)neg_scale);
        }
        if (sparse_k > 0) {
            fprintf(stderr, "Sparse:  top-%u per output row\n", (unsigned)sparse_k);
        }
        if (deepen_rounds > 0) {
            fprintf(stderr, "Deepen:  %u rounds\n", (unsigned)deepen_rounds);
        }
        if (val_path) {
            fprintf(stderr, "Val:     %s\n", val_path);
        }
        if (save_path) {
            fprintf(stderr, "Save:    %s\n", save_path);
        }
        if (save_accum_path) {
            fprintf(stderr, "SaveAcc: %s\n", save_accum_path);
        }
        if (load_accum_path) {
            fprintf(stderr, "LoadAcc: %s\n", load_accum_path);
        }
        fprintf(stderr, "\n");
    }

    /* --- Count pairs in the data file --- */
    int64_t pairs_per_epoch = count_jsonl_pairs(data_path);
    if (pairs_per_epoch < 0) {
        fprintf(stderr, "Error: cannot open '%s': %s\n", data_path, strerror(errno));
        return 1;
    }
    if (pairs_per_epoch == 0) {
        fprintf(stderr, "Error: no valid pairs found in '%s'\n", data_path);
        return 1;
    }

    /* --- Create checkpoint directory if requested --- */
    if (checkpoint_dir) {
        struct stat st;
        if (stat(checkpoint_dir, &st) != 0) {
            if (mkdir(checkpoint_dir, 0755) != 0) {
                fprintf(stderr, "Error: cannot create checkpoint directory '%s': %s\n",
                        checkpoint_dir, strerror(errno));
                return 1;
            }
            if (!quiet) {
                fprintf(stderr, "Checkpoint dir: %s (created)\n", checkpoint_dir);
            }
        } else if (!quiet) {
            fprintf(stderr, "Checkpoint dir: %s\n", checkpoint_dir);
        }
    }

    /* --- Create Hebbian training state --- */
    trine_hebbian_config_t config = {
        .similarity_threshold  = sim_thresh,
        .freeze_threshold      = freeze_thresh,
        .freeze_target_density = density,
        .cascade_cells         = cells,
        .cascade_depth         = depth,
        .projection_mode       = diagonal_mode ? 1 : 0,
        .weighted_mode         = weighted_mode,
        .pos_scale             = pos_scale,
        .neg_scale             = neg_scale,
        .source_weights        = {{{0}, 0.0f}},
        .n_source_weights      = 0,
        .sparse_k              = sparse_k,
        .block_diagonal        = block_diag_mode
    };

    /* Parse --source-weights "sts:4,sick:3,snli:0.1,..." */
    if (source_weights_str) {
        char sw_buf[512];
        strncpy(sw_buf, source_weights_str, sizeof(sw_buf) - 1);
        sw_buf[sizeof(sw_buf) - 1] = '\0';
        char *tok = strtok(sw_buf, ",");
        while (tok && config.n_source_weights < TRINE_MAX_SOURCES) {
            char *colon = strchr(tok, ':');
            if (colon) {
                *colon = '\0';
                /* Trim leading spaces */
                while (*tok == ' ') tok++;
                strncpy(config.source_weights[config.n_source_weights].name,
                        tok, TRINE_SOURCE_NAME_LEN - 1);
                config.source_weights[config.n_source_weights].name[TRINE_SOURCE_NAME_LEN - 1] = '\0';
                config.source_weights[config.n_source_weights].weight = strtof(colon + 1, NULL);
                config.n_source_weights++;
            }
            tok = strtok(NULL, ",");
        }
        if (!quiet && config.n_source_weights > 0) {
            fprintf(stderr, "Source weights:");
            for (int sw = 0; sw < config.n_source_weights; sw++) {
                fprintf(stderr, " %s=%.2f",
                        config.source_weights[sw].name,
                        (double)config.source_weights[sw].weight);
            }
            fprintf(stderr, "\n");
        }
    }

    /* Parse --threshold-schedule "0.9,0.75,0.6,0.45" */
    float threshold_schedule[64];
    int n_schedule = 0;
    if (threshold_schedule_str) {
        char ts_buf[256];
        strncpy(ts_buf, threshold_schedule_str, sizeof(ts_buf) - 1);
        ts_buf[sizeof(ts_buf) - 1] = '\0';
        char *tok = strtok(ts_buf, ",");
        while (tok && n_schedule < 64) {
            threshold_schedule[n_schedule++] = strtof(tok, NULL);
            tok = strtok(NULL, ",");
        }
        if (!quiet && n_schedule > 0) {
            fprintf(stderr, "Threshold schedule:");
            for (int ts = 0; ts < n_schedule; ts++) {
                fprintf(stderr, " %.2f", (double)threshold_schedule[ts]);
            }
            fprintf(stderr, "\n");
        }
    }

    trine_hebbian_state_t *state = trine_hebbian_create(&config);
    if (!state) {
        fprintf(stderr, "Error: failed to allocate training state\n");
        return 1;
    }

    /* --- Warm-start: load accumulator from .trine2a file --- */
    if (load_accum_path) {
        if (trine_accumulator_validate(load_accum_path) != 0) {
            fprintf(stderr, "Error: '%s' failed accumulator validation\n",
                    load_accum_path);
            trine_hebbian_free(state);
            return 1;
        }

        trine_hebbian_config_t restored_config;
        int64_t restored_pairs = 0;
        trine_accumulator_t *loaded_acc = trine_accumulator_load(
            load_accum_path, &restored_config, &restored_pairs);
        if (!loaded_acc) {
            fprintf(stderr, "Error: failed to load accumulator from '%s'\n",
                    load_accum_path);
            trine_hebbian_free(state);
            return 1;
        }

        /* Copy loaded counters into the training state's accumulator */
        trine_accumulator_t *target_acc = trine_hebbian_get_accumulator(state);
        if (target_acc) {
            for (uint32_t k = 0; k < TRINE_ACC_K; k++) {
                const int32_t (*src_mat)[TRINE_ACC_DIM] =
                    trine_accumulator_counters_const(loaded_acc, k);
                int32_t (*dst_mat)[TRINE_ACC_DIM] =
                    trine_accumulator_counters(target_acc, k);
                if (src_mat && dst_mat) {
                    memcpy(dst_mat, src_mat,
                           TRINE_ACC_DIM * TRINE_ACC_DIM * sizeof(int32_t));
                }
            }
        }

        trine_accumulator_free(loaded_acc);

        if (!quiet) {
            fprintf(stderr, "Warm-start: loaded %lld pairs from '%s'\n",
                    (long long)restored_pairs, load_accum_path);
        }
    }

    /* --- Training: one epoch at a time for progress reporting --- */
    double total_start = now_sec();
    int64_t total_pairs = 0;

    for (uint32_t e = 0; e < epochs; e++) {
        /* Apply threshold schedule (Phase A3) */
        if (n_schedule > 0) {
            int sched_idx = (int)e < n_schedule ? (int)e : n_schedule - 1;
            trine_hebbian_set_threshold(state, threshold_schedule[sched_idx]);
            if (verbose && !quiet) {
                fprintf(stderr, "  [schedule] epoch %u: threshold=%.3f\n",
                        e + 1, (double)threshold_schedule[sched_idx]);
            }
        }

        double epoch_start = now_sec();

        int64_t epoch_pairs = trine_hebbian_train_file(state, data_path, 1);
        if (epoch_pairs < 0) {
            fprintf(stderr, "Error: training failed at epoch %u\n", e + 1);
            trine_hebbian_free(state);
            return 1;
        }

        double epoch_elapsed = now_sec() - epoch_start;
        total_pairs += epoch_pairs;

        if (!quiet) {
            char pairs_buf[64];
            format_number(epoch_pairs, pairs_buf, sizeof(pairs_buf));

            double pairs_per_sec = (epoch_elapsed > 1e-9)
                                   ? (double)epoch_pairs / epoch_elapsed
                                   : 0.0;
            char rate_buf[64];
            format_number((int64_t)pairs_per_sec, rate_buf, sizeof(rate_buf));

            fprintf(stderr, "Epoch %2u/%u:  %s pairs  (%s pairs/sec)\n",
                    e + 1, epochs, pairs_buf, rate_buf);
        }

        /* --- Checkpoint: save model after each epoch --- */
        if (checkpoint_dir) {
            trine_s2_model_t *ckpt_model = trine_hebbian_freeze(state);
            if (ckpt_model) {
                /* Apply projection mode */
                if (sparse_k > 0) {
                    trine_s2_set_projection_mode(ckpt_model, TRINE_S2_PROJ_SPARSE);
                } else if (block_diag_mode) {
                    trine_s2_set_projection_mode(ckpt_model, TRINE_S2_PROJ_BLOCK_DIAG);
                } else if (diagonal_mode) {
                    trine_s2_set_projection_mode(ckpt_model, TRINE_S2_PROJ_DIAGONAL);
                }

                char ckpt_path[512];
                snprintf(ckpt_path, sizeof(ckpt_path), "%s/epoch_%03u.trine2",
                         checkpoint_dir, e + 1);

                trine_s2_save_config_t ckpt_cfg = {
                    .similarity_threshold = sim_thresh,
                    .density              = density,
                    .topo_seed            = 0
                };

                if (trine_s2_save(ckpt_model, ckpt_path, &ckpt_cfg) == 0) {
                    if (verbose && !quiet) {
                        fprintf(stderr, "  Checkpoint saved: %s\n", ckpt_path);
                    }
                } else {
                    fprintf(stderr, "  Warning: failed to save checkpoint '%s'\n",
                            ckpt_path);
                }
                trine_s2_free(ckpt_model);
            } else {
                fprintf(stderr, "  Warning: checkpoint freeze failed at epoch %u\n",
                        e + 1);
            }
        }
    }

    double total_elapsed = now_sec() - total_start;

    /* --- Print training summary --- */
    if (!quiet) {
        char total_buf[64];
        format_number(total_pairs, total_buf, sizeof(total_buf));
        fprintf(stderr, "\nTraining complete: %s total pairs  (%.1f sec)\n",
                total_buf, total_elapsed);
        fprintf(stderr, "Projection: %s\n",
                block_diag_mode ? "block-diagonal" :
                diagonal_mode   ? "diagonal" : "full");
    }

    /* --- Freeze or deepen --- */
    trine_s2_model_t *model = NULL;

    if (deepen_rounds > 0) {
        if (!quiet) {
            fprintf(stderr, "\nSelf-deepening: %u rounds...\n",
                    (unsigned)deepen_rounds);
        }

        double deepen_start = now_sec();
        model = trine_self_deepen(state, data_path, deepen_rounds);
        double deepen_elapsed = now_sec() - deepen_start;

        if (!model) {
            fprintf(stderr, "Error: self-deepening failed\n");
            trine_hebbian_free(state);
            return 1;
        }

        if (!quiet) {
            fprintf(stderr, "Deepening complete (%.1f sec)\n", deepen_elapsed);
        }

        if (sparse_k > 0) {
            trine_s2_set_projection_mode(model, TRINE_S2_PROJ_SPARSE);
        } else if (block_diag_mode) {
            trine_s2_set_projection_mode(model, TRINE_S2_PROJ_BLOCK_DIAG);
        } else if (diagonal_mode) {
            trine_s2_set_projection_mode(model, TRINE_S2_PROJ_DIAGONAL);
        }
        if (stacked_mode) {
            trine_s2_set_stacked_depth(model, 1);
        }
    } else {
        /* Get metrics before freeze (for reporting) */
        trine_hebbian_metrics_t metrics;
        trine_hebbian_metrics(state, &metrics);

        model = trine_hebbian_freeze(state);
        if (!model) {
            fprintf(stderr, "Error: freeze failed\n");
            trine_hebbian_free(state);
            return 1;
        }

        if (sparse_k > 0) {
            trine_s2_set_projection_mode(model, TRINE_S2_PROJ_SPARSE);
        } else if (block_diag_mode) {
            trine_s2_set_projection_mode(model, TRINE_S2_PROJ_BLOCK_DIAG);
        } else if (diagonal_mode) {
            trine_s2_set_projection_mode(model, TRINE_S2_PROJ_DIAGONAL);
        }
        if (stacked_mode) {
            trine_s2_set_stacked_depth(model, 1);
        }

        if (!quiet) {
            char max_buf[64];
            format_number((int64_t)metrics.max_abs_counter, max_buf, sizeof(max_buf));
            char pos_buf[64], neg_buf[64], zero_buf[64];
            format_number((int64_t)metrics.n_positive_weights, pos_buf, sizeof(pos_buf));
            format_number((int64_t)metrics.n_negative_weights, neg_buf, sizeof(neg_buf));
            format_number((int64_t)metrics.n_zero_weights, zero_buf, sizeof(zero_buf));

            fprintf(stderr, "  Max |counter|: %s\n", max_buf);
            fprintf(stderr, "  Weight density: %.2f (threshold: %s",
                    (double)metrics.weight_density,
                    freeze_thresh > 0 ? "manual" : "auto");
            fprintf(stderr, " -> %d)\n", (int)metrics.effective_threshold);
            fprintf(stderr, "  Positive: %s  Negative: %s  Zero: %s\n",
                    pos_buf, neg_buf, zero_buf);
        }
    }

    /* --- Print model info --- */
    if (!quiet) {
        trine_s2_info_t info;
        if (trine_s2_info(model, &info) == 0) {
            fprintf(stderr, "\nModel frozen: %ux%u projection (%u copies) + "
                            "%u-cell cascade\n",
                    info.projection_dims, info.projection_dims,
                    info.projection_k, info.cascade_cells);
        }
    }

    /* --- Save model (optional) --- */
    if (save_path) {
        trine_s2_save_config_t save_cfg = {
            .similarity_threshold = sim_thresh,
            .density              = density,
            .topo_seed            = 0  /* topology seed not tracked here */
        };

        int save_rc = trine_s2_save(model, save_path, &save_cfg);
        if (save_rc != 0) {
            fprintf(stderr, "Error: failed to save model to '%s'\n", save_path);
            trine_s2_free(model);
            trine_hebbian_free(state);
            return 1;
        }

        if (!quiet) {
            fprintf(stderr, "Model saved to: %s\n", save_path);
        }
    }

    /* --- Save accumulator state (optional) --- */
    if (save_accum_path) {
        trine_accumulator_t *acc = trine_hebbian_get_accumulator(state);
        if (!acc) {
            fprintf(stderr, "Error: cannot get accumulator for saving\n");
        } else {
            int acc_rc = trine_accumulator_save(acc, &config, total_pairs,
                                                 save_accum_path);
            if (acc_rc != 0) {
                fprintf(stderr, "Error: failed to save accumulator to '%s'\n",
                        save_accum_path);
            } else if (!quiet) {
                fprintf(stderr, "Accumulator saved to: %s\n", save_accum_path);
            }
        }
    }

    /* --- Validation (optional) --- */
    if (val_path) {
        if (!quiet) {
            fprintf(stderr, "\nValidation: %s\n", val_path);
        }

        double val_start = now_sec();
        double rho_s1 = 0.0, rho_s2 = 0.0;
        int64_t val_pairs = evaluate_validation(val_path, model, depth,
                                                 &rho_s1, &rho_s2,
                                                 gated_eval);
        double val_elapsed = now_sec() - val_start;

        if (val_pairs < 0) {
            fprintf(stderr, "  Error: cannot read validation file\n");
        } else if (val_pairs == 0) {
            fprintf(stderr, "  Warning: no valid pairs in validation file\n");
        } else {
            char vpairs_buf[64];
            format_number(val_pairs, vpairs_buf, sizeof(vpairs_buf));

            if (!quiet) {
                fprintf(stderr, "  Pairs: %s  (%.1f sec)\n",
                        vpairs_buf, val_elapsed);
                fprintf(stderr, "  Spearman rho (Stage-2): %.4f\n", rho_s2);
                fprintf(stderr, "  Spearman rho (Stage-1): %.4f\n", rho_s1);
                fprintf(stderr, "  Improvement: %+.4f\n", rho_s2 - rho_s1);
            }
        }
    }

    /* --- Adaptive alpha: compute per-S1-bucket optimal alpha --- */
    if (adaptive_alpha && val_path) {
        if (!quiet) {
            fprintf(stderr, "\nAdaptive alpha: computing per-S1-bucket alphas...\n");
        }

        FILE *afp = fopen(val_path, "r");
        if (!afp) {
            fprintf(stderr, "  Warning: cannot open validation file for adaptive alpha\n");
        } else {
            /* First pass: count valid pairs */
            char aline[TRAIN_MAX_LINE];
            char atext_a[TRAIN_MAX_TEXT], atext_b[TRAIN_MAX_TEXT];
            int64_t an_pairs = 0;

            while (fgets(aline, sizeof(aline), afp)) {
                if (aline[0] == '\n' || aline[0] == '\r' || aline[0] == '\0')
                    continue;
                int alen_a = extract_json_string(aline, "text_a", atext_a, sizeof(atext_a));
                int alen_b = extract_json_string(aline, "text_b", atext_b, sizeof(atext_b));
                float ascore = extract_json_number(aline, "score");
                if (alen_a > 0 && alen_b > 0 && ascore >= 0.0f) an_pairs++;
            }

            if (an_pairs > 0) {
                /* Allocate per-pair arrays */
                double *a_human  = (double *)malloc((size_t)an_pairs * sizeof(double));
                double *a_s1_sim = (double *)malloc((size_t)an_pairs * sizeof(double));
                double *a_s2_sim = (double *)malloc((size_t)an_pairs * sizeof(double));

                if (a_human && a_s1_sim && a_s2_sim) {
                    rewind(afp);
                    int64_t aidx = 0;
                    trine_s1_lens_t auniform = TRINE_S1_LENS_UNIFORM;

                    while (fgets(aline, sizeof(aline), afp) && aidx < an_pairs) {
                        if (aline[0] == '\n' || aline[0] == '\r' || aline[0] == '\0')
                            continue;
                        int alen_a = extract_json_string(aline, "text_a",
                                                          atext_a, sizeof(atext_a));
                        int alen_b = extract_json_string(aline, "text_b",
                                                          atext_b, sizeof(atext_b));
                        float ascore = extract_json_number(aline, "score");
                        if (alen_a <= 0 || alen_b <= 0 || ascore < 0.0f) continue;

                        a_human[aidx] = (double)ascore;

                        uint8_t ae_a[240], ae_b[240];
                        if (trine_encode_shingle(atext_a, (size_t)alen_a, ae_a) != 0)
                            continue;
                        if (trine_encode_shingle(atext_b, (size_t)alen_b, ae_b) != 0)
                            continue;
                        a_s1_sim[aidx] = (double)trine_s1_compare(ae_a, ae_b,
                                                                    &auniform);

                        uint8_t as2_a[240], as2_b[240];
                        if (trine_s2_encode(model, atext_a, (size_t)alen_a,
                                             depth, as2_a) != 0 ||
                            trine_s2_encode(model, atext_b, (size_t)alen_b,
                                             depth, as2_b) != 0) {
                            a_s2_sim[aidx] = a_s1_sim[aidx];
                        } else {
                            a_s2_sim[aidx] = (double)centered_cosine_240(as2_a,
                                                                          as2_b);
                        }
                        aidx++;
                    }

                    /* Per-S1-bucket alpha: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0] */
                    float buckets[10];
                    double alpha_cands[] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                            0.6, 0.7, 0.8, 0.9, 1.0};
                    int n_cands = 11;

                    for (int b = 0; b < 10; b++) {
                        double blo = b * 0.1;
                        double bhi = (b + 1) * 0.1;

                        /* Count pairs in this bucket */
                        int bcnt = 0;
                        for (int64_t j = 0; j < aidx; j++) {
                            if (a_s1_sim[j] >= blo && a_s1_sim[j] < bhi)
                                bcnt++;
                        }

                        if (bcnt < 3) {
                            buckets[b] = 0.5f;
                            continue;
                        }

                        double *bh = (double *)malloc((size_t)bcnt * sizeof(double));
                        double *bs = (double *)malloc((size_t)bcnt * sizeof(double));
                        if (!bh || !bs) {
                            free(bh);
                            free(bs);
                            buckets[b] = 0.5f;
                            continue;
                        }

                        double best_rho = -2.0;
                        double best_a = 0.5;

                        for (int ac = 0; ac < n_cands; ac++) {
                            double alpha = alpha_cands[ac];
                            int bi = 0;
                            for (int64_t j = 0; j < aidx; j++) {
                                if (a_s1_sim[j] >= blo && a_s1_sim[j] < bhi) {
                                    bh[bi] = a_human[j];
                                    bs[bi] = alpha * a_s1_sim[j] +
                                             (1.0 - alpha) * a_s2_sim[j];
                                    bi++;
                                }
                            }
                            double rho = spearman_rho(bh, bs, bcnt);
                            if (rho > best_rho) {
                                best_rho = rho;
                                best_a = alpha;
                            }
                        }

                        buckets[b] = (float)best_a;
                        free(bh);
                        free(bs);
                    }

                    trine_s2_set_adaptive_alpha(model, buckets);

                    if (!quiet) {
                        fprintf(stderr, "  Adaptive alpha buckets:");
                        for (int b = 0; b < 10; b++) {
                            fprintf(stderr, " [%.1f-%.1f):%.2f",
                                    b * 0.1, (b + 1) * 0.1,
                                    (double)buckets[b]);
                        }
                        fprintf(stderr, "\n");
                    }
                }

                free(a_human);
                free(a_s1_sim);
                free(a_s2_sim);
            }

            fclose(afp);
        }
    } else if (adaptive_alpha && !val_path) {
        fprintf(stderr, "Warning: --adaptive-alpha requires --val <val.jsonl>\n");
    }

    /* --- Output model summary --- */
    {
        FILE *out = stdout;
        if (output_path) {
            out = fopen(output_path, "w");
            if (!out) {
                fprintf(stderr, "Error: cannot open output file '%s': %s\n",
                        output_path, strerror(errno));
                trine_s2_free(model);
                trine_hebbian_free(state);
                return 1;
            }
        }

        trine_hebbian_metrics_t metrics;
        trine_hebbian_metrics(state, &metrics);

        trine_s2_info_t info;
        trine_s2_info(model, &info);

        fprintf(out, "{\n");
        fprintf(out, "  \"version\": \"%s\",\n", TRAIN_VERSION);
        fprintf(out, "  \"data\": \"%s\",\n", data_path);
        fprintf(out, "  \"epochs\": %u,\n", epochs);
        fprintf(out, "  \"total_pairs\": %lld,\n", (long long)total_pairs);
        fprintf(out, "  \"elapsed_sec\": %.2f,\n", total_elapsed);
        fprintf(out, "  \"config\": {\n");
        fprintf(out, "    \"similarity_threshold\": %.4f,\n", (double)sim_thresh);
        fprintf(out, "    \"freeze_threshold\": %d,\n", (int)metrics.effective_threshold);
        fprintf(out, "    \"freeze_target_density\": %.4f,\n", (double)density);
        fprintf(out, "    \"cascade_cells\": %u,\n", (unsigned)cells);
        fprintf(out, "    \"cascade_depth\": %u,\n", (unsigned)depth);
        fprintf(out, "    \"projection_mode\": \"%s\"\n",
                block_diag_mode ? "block-diagonal" :
                diagonal_mode   ? "diagonal" : "full");
        fprintf(out, "  },\n");
        fprintf(out, "  \"metrics\": {\n");
        fprintf(out, "    \"max_abs_counter\": %d,\n", (int)metrics.max_abs_counter);
        fprintf(out, "    \"weight_density\": %.4f,\n", (double)metrics.weight_density);
        fprintf(out, "    \"n_positive\": %u,\n", (unsigned)metrics.n_positive_weights);
        fprintf(out, "    \"n_negative\": %u,\n", (unsigned)metrics.n_negative_weights);
        fprintf(out, "    \"n_zero\": %u\n", (unsigned)metrics.n_zero_weights);
        fprintf(out, "  },\n");
        fprintf(out, "  \"model\": {\n");
        fprintf(out, "    \"projection_k\": %u,\n", info.projection_k);
        fprintf(out, "    \"projection_dims\": %u,\n", info.projection_dims);
        fprintf(out, "    \"cascade_cells\": %u,\n", info.cascade_cells);
        fprintf(out, "    \"is_identity\": %d\n", info.is_identity);
        fprintf(out, "  }");

        if (deepen_rounds > 0) {
            fprintf(out, ",\n  \"deepen_rounds\": %u", (unsigned)deepen_rounds);
        }

        fprintf(out, "\n}\n");

        if (output_path) {
            fclose(out);
            if (!quiet) {
                fprintf(stderr, "\nModel summary written to: %s\n", output_path);
            }
        }
    }

    /* --- Cleanup --- */
    trine_s2_free(model);
    trine_hebbian_free(state);

    if (verbose && !quiet) {
        fprintf(stderr, "\nDone.\n");
    }

    return 0;
}
