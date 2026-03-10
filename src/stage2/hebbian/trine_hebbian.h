/* =====================================================================
 * TRINE Stage-2 — Hebbian Training Orchestrator
 * =====================================================================
 *
 * High-level API for training a Stage-2 semantic projection from
 * text pairs.  Wraps the accumulator, freeze, projection, and cascade
 * subsystems into a single train-then-freeze workflow.
 *
 * Usage:
 *   1. Create state with trine_hebbian_create()
 *   2. Feed pairs: observe(), observe_text(), or train_file()
 *   3. Freeze to a Stage-2 model: trine_hebbian_freeze()
 *   4. Free state with trine_hebbian_free()
 *
 * The training signal comes from Stage-1 cosine similarity: pairs
 * with s1 > threshold are "similar" (positive Hebbian update) and
 * pairs with s1 <= threshold are "dissimilar" (negative update).
 *
 * ===================================================================== */

#ifndef TRINE_HEBBIAN_H
#define TRINE_HEBBIAN_H

#include <stdint.h>
#include <stddef.h>

/* Forward declaration */
struct trine_s2_model;

typedef struct trine_hebbian_state trine_hebbian_state_t;

/* Per-source training weight entry */
#define TRINE_MAX_SOURCES  8
#define TRINE_SOURCE_NAME_LEN 16

typedef struct {
    char  name[TRINE_SOURCE_NAME_LEN];
    float weight;
} trine_source_weight_t;

typedef struct {
    float    similarity_threshold; /* s1 > thresh = similar. Default: 0.5 */
    int32_t  freeze_threshold;     /* T for quantization. 0 = auto         */
    float    freeze_target_density;/* Target density for auto-T. Def: 0.33 */
    uint32_t cascade_cells;        /* Cascade mixing cells. Default: 512   */
    uint32_t cascade_depth;        /* Default inference depth. Default: 4  */
    int      projection_mode;      /* 0=sign, 1=diagonal. Default: 0      */

    /* Weighted Hebbian (Phase A1) */
    int      weighted_mode;        /* 0=binary sign, 1=weighted magnitude   */
    float    pos_scale;            /* positive magnitude scale (default 10) */
    float    neg_scale;            /* negative magnitude scale (default 3)  */

    /* Dataset rebalancing (Phase A2) */
    trine_source_weight_t source_weights[TRINE_MAX_SOURCES];
    int      n_source_weights;     /* Number of configured source weights   */

    /* Sparse cross-channel projection (Phase D1) */
    uint32_t sparse_k;             /* 0=disabled, >0=top-K per output row   */

    /* Block-diagonal projection (v1.0.3) */
    int      block_diagonal;       /* 0 = use existing mode, 1 = block-diagonal */

    /* RNG seed for deterministic downsampling (thread-safe)                 */
    uint64_t rng_seed;             /* 0 = derive from pairs_observed         */
} trine_hebbian_config_t;

#define TRINE_HEBBIAN_CONFIG_DEFAULT { \
    .similarity_threshold = 0.5f, \
    .freeze_threshold = 0, \
    .freeze_target_density = 0.33f, \
    .cascade_cells = 512, \
    .cascade_depth = 4, \
    .projection_mode = 0, \
    .weighted_mode = 0, \
    .pos_scale = 10.0f, \
    .neg_scale = 3.0f, \
    .source_weights = {{{0}, 0.0f}}, \
    .n_source_weights = 0, \
    .sparse_k = 0, \
    .block_diagonal = 0, \
    .rng_seed = 0 \
}

/* Create training state. Returns NULL on failure. */
trine_hebbian_state_t *trine_hebbian_create(const trine_hebbian_config_t *config);

/* Free training state. Safe to call with NULL. */
void trine_hebbian_free(trine_hebbian_state_t *state);

/* Observe a single training pair (raw trits).
 * a, b are 240-trit Stage-1 embeddings.
 * similarity is the Stage-1 cosine similarity (used to determine sign). */
void trine_hebbian_observe(trine_hebbian_state_t *state,
                            const uint8_t a[240], const uint8_t b[240],
                            float similarity);

/* Observe a pair from text (encode + compare + accumulate). */
void trine_hebbian_observe_text(trine_hebbian_state_t *state,
                                 const char *text_a, size_t len_a,
                                 const char *text_b, size_t len_b);

/* Train from a JSONL file. Each line has text_a, text_b, score, label.
 * Returns number of pairs processed, or -1 on error.
 * If epochs > 1, re-reads the file that many times. */
int64_t trine_hebbian_train_file(trine_hebbian_state_t *state,
                                  const char *path, uint32_t epochs);

/* Freeze current state to a Stage-2 model.
 * Caller owns the returned model and must free it with trine_s2_free().
 * Returns NULL on failure. */
struct trine_s2_model *trine_hebbian_freeze(const trine_hebbian_state_t *state);

/* Training metrics. */
typedef struct {
    int64_t  pairs_observed;       /* Total pairs fed to observe() */
    int32_t  max_abs_counter;      /* Max |counter| across all weights */
    uint32_t n_positive_weights;   /* Counters > 0 */
    uint32_t n_negative_weights;   /* Counters < 0 */
    uint32_t n_zero_weights;       /* Counters == 0 */
    float    weight_density;       /* Fraction non-zero after freeze */
    int32_t  effective_threshold;  /* Threshold used (explicit or auto) */
} trine_hebbian_metrics_t;

int trine_hebbian_metrics(const trine_hebbian_state_t *state,
                           trine_hebbian_metrics_t *out);

/* Reset accumulators (start fresh, keep config). */
void trine_hebbian_reset(trine_hebbian_state_t *state);

/* Get the current config (read-only copy). */
trine_hebbian_config_t trine_hebbian_get_config(const trine_hebbian_state_t *state);

/* Update the similarity threshold between epochs (for threshold schedule).
 * This takes effect on subsequent observe() calls. */
void trine_hebbian_set_threshold(trine_hebbian_state_t *state, float threshold);

/* Get a pointer to the internal accumulator (for persistence).
 * Caller must NOT free the returned pointer. */
struct trine_accumulator *trine_hebbian_get_accumulator(trine_hebbian_state_t *state);

/* Self-supervised deepening (implemented in trine_self_deepen.c).
 * Recursively bootstraps: freeze -> re-encode -> re-accumulate.
 * n_rounds: number of deepening rounds.
 * data_path: JSONL training file.
 * Returns the final deepened model, or NULL on error.
 * Caller owns the returned model. */
struct trine_s2_model *trine_self_deepen(trine_hebbian_state_t *state,
                                          const char *data_path,
                                          uint32_t n_rounds);

#endif /* TRINE_HEBBIAN_H */
