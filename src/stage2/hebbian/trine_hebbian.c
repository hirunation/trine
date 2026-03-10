/* =====================================================================
 * TRINE Stage-2 — Hebbian Training Orchestrator (implementation)
 * =====================================================================
 *
 * High-level training pipeline: text pairs -> accumulate -> freeze.
 * See trine_hebbian.h for API documentation.
 *
 * JSONL parsing delegates to the shared trine_jsonl utility
 * (trine_jsonl.h/c) for string and number extraction.
 *
 * ===================================================================== */

/* Enable getline() / ssize_t on POSIX systems */
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L
#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "trine_hebbian.h"
#include "trine_jsonl.h"
#include "trine_accumulator.h"
#include "trine_freeze.h"
#include "trine_stage2.h"
#include "trine_encode.h"
#include "trine_stage1.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>

/* --------------------------------------------------------------------- */
/* Internal state                                                         */
/* --------------------------------------------------------------------- */

struct trine_hebbian_state {
    trine_hebbian_config_t       config;
    trine_accumulator_t         *accumulator;       /* Full/diagonal mode */
    trine_block_accumulator_t   *block_accumulator;  /* Block-diagonal mode (v1.0.3) */
    int64_t                      pairs_observed;
};

/* --------------------------------------------------------------------- */
/* Create / free / reset                                                  */
/* --------------------------------------------------------------------- */

trine_hebbian_state_t *trine_hebbian_create(const trine_hebbian_config_t *config)
{
    trine_hebbian_state_t *st = calloc(1, sizeof(*st));
    if (!st) return NULL;

    if (config) {
        st->config = *config;
    } else {
        trine_hebbian_config_t defaults = TRINE_HEBBIAN_CONFIG_DEFAULT;
        st->config = defaults;
    }

    if (st->config.block_diagonal) {
        /* Block-diagonal accumulator: 4 x 60x60 per copy, K=3 */
        st->block_accumulator = trine_block_accumulator_create(TRINE_ACC_K);
        if (!st->block_accumulator) {
            free(st);
            return NULL;
        }
        st->accumulator = NULL;
    } else {
        /* Standard full 240x240 accumulator */
        st->accumulator = trine_accumulator_create();
        if (!st->accumulator) {
            free(st);
            return NULL;
        }
        st->block_accumulator = NULL;
    }

    st->pairs_observed = 0;
    return st;
}

void trine_hebbian_free(trine_hebbian_state_t *state)
{
    if (!state) return;
    if (state->accumulator)
        trine_accumulator_free(state->accumulator);
    if (state->block_accumulator)
        trine_block_accumulator_free(state->block_accumulator);
    free(state);
}

void trine_hebbian_reset(trine_hebbian_state_t *state)
{
    if (!state) return;
    if (state->block_accumulator)
        trine_block_accumulator_reset(state->block_accumulator);
    else if (state->accumulator)
        trine_accumulator_reset(state->accumulator);
    state->pairs_observed = 0;
}

trine_hebbian_config_t trine_hebbian_get_config(const trine_hebbian_state_t *state)
{
    if (state) return state->config;
    trine_hebbian_config_t defaults = TRINE_HEBBIAN_CONFIG_DEFAULT;
    return defaults;
}

/* --------------------------------------------------------------------- */
/* Observe (raw trits)                                                    */
/* --------------------------------------------------------------------- */

void trine_hebbian_observe(trine_hebbian_state_t *state,
                            const uint8_t a[240], const uint8_t b[240],
                            float similarity)
{
    if (!state || !a || !b) return;

    float delta = similarity - state->config.similarity_threshold;
    int sign = (delta > 0.0f) ? 1 : -1;

    if (state->block_accumulator) {
        /* Block-diagonal accumulator path */
        if (state->config.weighted_mode) {
            int32_t magnitude;
            if (delta > 0.0f) {
                magnitude = 1 + (int32_t)(delta * state->config.pos_scale);
            } else {
                magnitude = 1 + (int32_t)((-delta) * state->config.neg_scale);
            }
            trine_block_accumulator_update_weighted(state->block_accumulator,
                                                     a, b, sign, magnitude);
        } else {
            trine_block_accumulator_update(state->block_accumulator,
                                            a, b, sign);
        }
    } else {
        /* Standard full accumulator path */
        if (state->config.weighted_mode) {
            int32_t magnitude;
            if (delta > 0.0f) {
                magnitude = 1 + (int32_t)(delta * state->config.pos_scale);
            } else {
                magnitude = 1 + (int32_t)((-delta) * state->config.neg_scale);
            }
            trine_accumulator_update_weighted(state->accumulator, a, b,
                                               sign, magnitude);
        } else {
            trine_accumulator_update(state->accumulator, a, b, sign);
        }
    }
    state->pairs_observed++;
}

/* --------------------------------------------------------------------- */
/* Observe from text (encode + compare + accumulate)                      */
/* --------------------------------------------------------------------- */

void trine_hebbian_observe_text(trine_hebbian_state_t *state,
                                 const char *text_a, size_t len_a,
                                 const char *text_b, size_t len_b)
{
    if (!state || !text_a || !text_b) return;

    /* Encode both texts */
    uint8_t emb_a[240], emb_b[240];
    if (trine_encode_shingle(text_a, len_a, emb_a) != 0) return;
    if (trine_encode_shingle(text_b, len_b, emb_b) != 0) return;

    /* Compute Stage-1 cosine similarity (uniform lens) */
    trine_s1_lens_t uniform = TRINE_S1_LENS_UNIFORM;
    float sim = trine_s1_compare(emb_a, emb_b, &uniform);

    /* Feed to accumulator */
    trine_hebbian_observe(state, emb_a, emb_b, sim);
}

/* --------------------------------------------------------------------- */
/* Train from JSONL file                                                  */
/* --------------------------------------------------------------------- */

/* Initial line buffer size for JSONL parsing.  Buffer grows dynamically
 * via getline() so lines longer than this are handled correctly. */
#define HEBBIAN_INIT_LINE  8192
#define HEBBIAN_MAX_TEXT   4096

/* Simple LCG for source-weight downsampling (deterministic per-pair).
 * State is passed by pointer to avoid shared static state (thread-safe). */
static float train_random_float(uint32_t *lcg_state)
{
    *lcg_state = (*lcg_state) * 1664525u + 1013904223u;
    return (float)(*lcg_state >> 8) / 16777216.0f;
}

/* Look up source weight from config.  Returns 1.0 if not configured. */
static float lookup_source_weight(const trine_hebbian_config_t *cfg,
                                   const char *source)
{
    if (!cfg || cfg->n_source_weights == 0 || !source) return 1.0f;
    for (int i = 0; i < cfg->n_source_weights; i++) {
        if (strncmp(cfg->source_weights[i].name, source,
                    TRINE_SOURCE_NAME_LEN) == 0) {
            return cfg->source_weights[i].weight;
        }
    }
    return 1.0f;  /* unrecognized source: default weight */
}

int64_t trine_hebbian_train_file(trine_hebbian_state_t *state,
                                  const char *path, uint32_t epochs)
{
    if (!state || !path || epochs == 0) return -1;

    int64_t total_pairs = 0;
    int has_source_weights = (state->config.n_source_weights > 0);

    /* Initialize local LCG state for deterministic downsampling.
     * Uses config rng_seed if set, else falls back to pairs_observed. */
    uint32_t lcg_state = (state->config.rng_seed != 0)
        ? (uint32_t)(state->config.rng_seed & 0xFFFFFFFF)
        : (uint32_t)(state->pairs_observed & 0xFFFFFFFF);

    /* Dynamically-growing line buffer, reused across epochs */
    char *line = NULL;
    size_t line_cap = 0;

    for (uint32_t epoch = 0; epoch < epochs; epoch++) {
        FILE *fp = fopen(path, "r");
        if (!fp) { free(line); return -1; }

        char text_a[HEBBIAN_MAX_TEXT];
        char text_b[HEBBIAN_MAX_TEXT];
        char source[TRINE_SOURCE_NAME_LEN];

        while (getline(&line, &line_cap, fp) != -1) {
            /* Skip empty lines */
            if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
                continue;

            /* Extract text_a and text_b */
            int len_a = trine_jsonl_extract_string(line, 0, "text_a",
                                                    text_a, sizeof(text_a));
            int len_b = trine_jsonl_extract_string(line, 0, "text_b",
                                                    text_b, sizeof(text_b));

            if (len_a < 0 || len_b < 0) continue;
            if (len_a == 0 || len_b == 0) continue;

            /* Source-based weighting (Phase A2) */
            float src_weight = 1.0f;
            if (has_source_weights) {
                int src_len = trine_jsonl_extract_source(line, 0, source,
                                                          sizeof(source));
                if (src_len > 0) {
                    src_weight = lookup_source_weight(&state->config, source);
                }
                /* Probabilistic downsampling for weight < 1.0 */
                if (src_weight < 1.0f) {
                    if (train_random_float(&lcg_state) > src_weight) continue;
                    src_weight = 1.0f;  /* accepted: use weight 1 */
                }
            }

            /* Encode both texts to trit vectors */
            uint8_t emb_a[240], emb_b[240];
            if (trine_encode_shingle(text_a, (size_t)len_a, emb_a) != 0) continue;
            if (trine_encode_shingle(text_b, (size_t)len_b, emb_b) != 0) continue;

            /* Use labeled score from data if available, else fall back
             * to Stage-1 cosine. */
            float sim;
            if (trine_jsonl_extract_float(line, 0, "score", &sim)) {
                /* Use labeled score directly */
            } else {
                trine_s1_lens_t uniform = TRINE_S1_LENS_UNIFORM;
                sim = trine_s1_compare(emb_a, emb_b, &uniform);
            }

            /* Repeat observation for upweighted sources */
            int reps = (int)src_weight;
            if (reps < 1) reps = 1;
            for (int r = 0; r < reps; r++) {
                trine_hebbian_observe(state, emb_a, emb_b, sim);
            }
            total_pairs++;
        }

        fclose(fp);
    }

    free(line);

    return total_pairs;
}

/* --------------------------------------------------------------------- */
/* Threshold and accumulator access                                       */
/* --------------------------------------------------------------------- */

void trine_hebbian_set_threshold(trine_hebbian_state_t *state, float threshold)
{
    if (state) state->config.similarity_threshold = threshold;
}

struct trine_accumulator *trine_hebbian_get_accumulator(trine_hebbian_state_t *state)
{
    if (!state) return NULL;
    return state->accumulator;
}

/* --------------------------------------------------------------------- */
/* Freeze to Stage-2 model                                                */
/* --------------------------------------------------------------------- */

/* FNV-1a hash over raw bytes for deterministic topology seed */
static uint64_t fnv1a_bytes(const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

struct trine_s2_model *trine_hebbian_freeze(const trine_hebbian_state_t *state)
{
    if (!state) return NULL;

    /* Block-diagonal freeze path (v1.0.3) */
    if (state->block_accumulator) {
        /* 1. Determine threshold */
        int32_t threshold = state->config.freeze_threshold;
        if (threshold <= 0) {
            threshold = trine_freeze_block_auto_threshold(
                state->block_accumulator,
                state->config.freeze_target_density);
        }

        /* 2. Freeze block accumulators to ternary weights */
        size_t block_size = (size_t)TRINE_ACC_K * TRINE_BLOCK_CHAINS
                            * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;
        uint8_t *W_blocks = calloc(block_size, 1);
        if (!W_blocks) return NULL;

        trine_freeze_block(state->block_accumulator, threshold, W_blocks);

        /* 3. Derive topology seed from block weights (reproducible) */
        uint64_t topo_seed = fnv1a_bytes(W_blocks, block_size);

        /* 4. Assemble into a Stage-2 model with block-diagonal projection */
        struct trine_s2_model *model = trine_s2_create_block_diagonal(
            W_blocks, TRINE_ACC_K,
            state->config.cascade_cells, topo_seed);

        free(W_blocks);

        /* Set block-diagonal projection mode */
        if (model) {
            trine_s2_set_projection_mode(model, TRINE_S2_PROJ_BLOCK_DIAG);
        }

        return model;
    }

    /* Standard (full matrix) freeze path */
    if (!state->accumulator) return NULL;

    /* 2. Freeze accumulators to ternary projection */
    trine_projection_t proj;

    if (state->config.sparse_k > 0) {
        /* Sparse freeze: keep top-K per output row */
        trine_freeze_sparse(state->accumulator, state->config.sparse_k, &proj);
    } else {
        /* Standard freeze with global threshold */
        int32_t threshold = state->config.freeze_threshold;
        if (threshold <= 0) {
            threshold = trine_freeze_auto_threshold(state->accumulator,
                                                     state->config.freeze_target_density);
        }
        trine_freeze_projection(state->accumulator, threshold, &proj);
    }

    /* 3. Derive topology seed from projection weights (reproducible) */
    uint64_t topo_seed = fnv1a_bytes(proj.W, sizeof(proj.W));

    /* 4. Assemble into a Stage-2 model via trine_s2_create_from_parts */
    struct trine_s2_model *model = trine_s2_create_from_parts(
        &proj, state->config.cascade_cells, topo_seed);

    /* Set sparse projection mode if applicable */
    if (model && state->config.sparse_k > 0) {
        trine_s2_set_projection_mode(model, TRINE_S2_PROJ_SPARSE);
    }

    return model;
}

/* --------------------------------------------------------------------- */
/* Metrics                                                                */
/* --------------------------------------------------------------------- */

int trine_hebbian_metrics(const trine_hebbian_state_t *state,
                           trine_hebbian_metrics_t *out)
{
    if (!state || !out) return -1;

    memset(out, 0, sizeof(*out));

    out->pairs_observed = state->pairs_observed;

    if (state->block_accumulator) {
        /* Block-diagonal metrics path */
        int32_t max_val = 0, min_val = 0;
        uint64_t nonzero = 0;
        trine_block_accumulator_stats(state->block_accumulator,
                                       &max_val, &min_val, &nonzero);

        int32_t abs_min = (min_val == INT32_MIN) ? INT32_MAX : -min_val;
        out->max_abs_counter = (max_val > abs_min) ? max_val : abs_min;

        /* Total block-diagonal entries: K * 4 * 60 * 60 */
        uint32_t total = (uint32_t)TRINE_ACC_K * TRINE_BLOCK_CHAINS
                         * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;
        out->n_zero_weights = total - (uint32_t)nonzero;
        /* Block accumulator stats don't split pos/neg; report nonzero as positive
         * (callers inspect pairs_observed + max_abs for real diagnostics) */
        out->n_positive_weights = (uint32_t)nonzero;
        out->n_negative_weights = 0;

        /* Compute effective threshold */
        int32_t threshold = state->config.freeze_threshold;
        if (threshold <= 0) {
            threshold = trine_freeze_block_auto_threshold(
                state->block_accumulator,
                state->config.freeze_target_density);
        }
        out->effective_threshold = threshold;

        /* Trial freeze to compute actual density */
        size_t block_size = (size_t)TRINE_ACC_K * TRINE_BLOCK_CHAINS
                            * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM;
        uint8_t *W_blocks = calloc(block_size, 1);
        if (W_blocks) {
            trine_freeze_block(state->block_accumulator, threshold, W_blocks);
            uint32_t nz = 0;
            for (size_t i = 0; i < block_size; i++) {
                if (W_blocks[i] != 0) nz++;
            }
            out->weight_density = (float)nz / (float)block_size;
            free(W_blocks);
        }

        return 0;
    }

    /* Standard full accumulator metrics path */
    if (!state->accumulator) return -1;

    /* Get accumulator statistics */
    trine_accumulator_stats_t acc_stats;
    trine_accumulator_stats(state->accumulator, &acc_stats);

    out->max_abs_counter   = acc_stats.max_abs;
    out->n_positive_weights = acc_stats.n_positive;
    out->n_negative_weights = acc_stats.n_negative;
    out->n_zero_weights     = acc_stats.n_zero;

    /* Compute density and effective threshold by doing a trial freeze */
    int32_t threshold = state->config.freeze_threshold;
    if (threshold <= 0) {
        threshold = trine_freeze_auto_threshold(state->accumulator,
                                                 state->config.freeze_target_density);
    }
    out->effective_threshold = threshold;

    /* Freeze to compute actual density */
    trine_projection_t proj;
    trine_freeze_projection(state->accumulator, threshold, &proj);

    trine_freeze_stats_t fstats;
    trine_freeze_stats(&proj, &fstats);
    out->weight_density = fstats.density;

    return 0;
}
