/* =====================================================================
 * TRINE Stage-2 — Forward Pass Implementation
 * =====================================================================
 *
 * struct trine_s2_model {
 *     trine_projection_t       projection;   ~169 KB
 *     trine_learned_cascade_t *cascade;      ~2.5 KB (512 cells)
 *     uint32_t                 default_depth;
 *     int                      is_identity;
 * };
 *
 * Thread-safe: no mutable state in model after construction.
 *
 * ===================================================================== */

#include "trine_stage2.h"
#include "trine_project.h"
#include "trine_learned_cascade.h"
#include "trine_encode.h"
#include "trine_stage1.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward declaration for topology generator */
extern void trine_topology_random(trine_learned_cascade_t *lc, uint64_t seed);

struct trine_s2_model {
    trine_projection_t       projection;    /* ~169 KB */
    trine_learned_cascade_t *cascade;       /* heap-allocated */
    uint32_t                 default_depth;
    int                      is_identity;
    int                      projection_mode; /* 0=sign, 1=diagonal, 2=sparse, 3=block-diag */
    int                      stacked_depth;   /* re-project instead of cascade */
    uint8_t                 *block_weights;   /* K * 4 * 60 * 60 bytes (NULL if not block mode) */
    int                      block_K;         /* number of projection copies for block mode */
    float                   *adaptive_alpha;  /* 10-element bucket array (NULL if disabled) */
};

/* ── Lifecycle ──────────────────────────────────────────────────────── */

trine_s2_model_t *trine_s2_create_identity(void)
{
    trine_s2_model_t *m = calloc(1, sizeof(*m));
    if (!m) return NULL;

    trine_projection_identity(&m->projection);

    /* Zero-cell cascade: tick is a pure memcpy pass-through */
    trine_cascade_config_t cfg = { .n_cells = 0, .max_depth = 0 };
    m->cascade = trine_learned_cascade_create(&cfg);
    if (!m->cascade) {
        free(m);
        return NULL;
    }

    m->default_depth = 0;
    m->is_identity = 1;
    return m;
}

trine_s2_model_t *trine_s2_create_random(uint32_t n_cells, uint64_t seed)
{
    trine_s2_model_t *m = calloc(1, sizeof(*m));
    if (!m) return NULL;

    trine_projection_random(&m->projection, seed);

    trine_cascade_config_t cfg = {
        .n_cells   = n_cells,
        .max_depth = TRINE_CASCADE_MAX_DEPTH
    };
    m->cascade = trine_learned_cascade_create(&cfg);
    if (!m->cascade) {
        free(m);
        return NULL;
    }

    /* Use a different seed for topology to avoid correlation */
    trine_topology_random(m->cascade, seed ^ 0xDEADBEEFCAFEBABEULL);

    m->default_depth = TRINE_CASCADE_DEFAULT_DEPTH;
    m->is_identity = 0;
    return m;
}

trine_s2_model_t *trine_s2_create_from_parts(const void *proj,
                                               uint32_t n_cells,
                                               uint64_t topo_seed)
{
    if (!proj) return NULL;

    trine_s2_model_t *m = calloc(1, sizeof(*m));
    if (!m) return NULL;

    /* Copy the frozen projection weights */
    memcpy(&m->projection, proj, sizeof(trine_projection_t));

    /* Create cascade with the given cell count */
    trine_cascade_config_t cfg = {
        .n_cells   = n_cells,
        .max_depth = TRINE_CASCADE_MAX_DEPTH
    };
    m->cascade = trine_learned_cascade_create(&cfg);
    if (!m->cascade) {
        free(m);
        return NULL;
    }

    /* Initialize topology deterministically from seed */
    if (n_cells > 0) {
        trine_topology_random(m->cascade, topo_seed);
    }

    m->default_depth = (n_cells > 0) ? TRINE_CASCADE_DEFAULT_DEPTH : 0;
    m->is_identity = 0;
    return m;
}

trine_s2_model_t *trine_s2_create_block_diagonal(
    const uint8_t *block_weights,
    int K,
    uint32_t n_cells,
    uint64_t topo_seed)
{
    if (!block_weights || K <= 0) return NULL;

    trine_s2_model_t *m = calloc(1, sizeof(*m));
    if (!m) return NULL;

    /* Copy block-diagonal weights: K * 4 * 60 * 60 bytes */
    size_t bw_size = (size_t)K * 4 * 60 * 60;
    m->block_weights = malloc(bw_size);
    if (!m->block_weights) {
        free(m);
        return NULL;
    }
    memcpy(m->block_weights, block_weights, bw_size);
    m->block_K = K;
    m->projection_mode = TRINE_S2_PROJ_BLOCK_DIAG;

    /* Initialize projection to identity (unused for block mode, but safe) */
    trine_projection_identity(&m->projection);

    /* Create cascade */
    trine_cascade_config_t cfg = {
        .n_cells   = n_cells,
        .max_depth = TRINE_CASCADE_MAX_DEPTH
    };
    m->cascade = trine_learned_cascade_create(&cfg);
    if (!m->cascade) {
        free(m->block_weights);
        free(m);
        return NULL;
    }

    if (n_cells > 0) {
        trine_topology_random(m->cascade, topo_seed);
    }

    m->default_depth = (n_cells > 0) ? TRINE_CASCADE_DEFAULT_DEPTH : 0;
    m->is_identity = 0;
    return m;
}

void trine_s2_free(trine_s2_model_t *model)
{
    if (!model) return;
    trine_learned_cascade_free(model->cascade);
    free(model->block_weights);
    free(model->adaptive_alpha);
    free(model);
}

/* ── Stacked Depth ─────────────────────────────────────────────────── */

void trine_s2_set_stacked_depth(trine_s2_model_t *model, int enable)
{
    if (model) model->stacked_depth = enable ? 1 : 0;
}

int trine_s2_get_stacked_depth(const trine_s2_model_t *model)
{
    if (!model) return 0;
    return model->stacked_depth;
}

/* Apply one projection step based on model's configured mode. */
static void apply_projection(const trine_s2_model_t *model,
                               const uint8_t in[TRINE_S2_DIM],
                               uint8_t out[TRINE_S2_DIM])
{
    if (model->is_identity) {
        trine_project_majority(&model->projection, in, out);
    } else if (model->projection_mode == TRINE_S2_PROJ_BLOCK_DIAG) {
        trine_projection_majority_block(model->block_weights, model->block_K,
                                        in, out);
    } else if (model->projection_mode == TRINE_S2_PROJ_DIAGONAL) {
        trine_project_diagonal_gate(&model->projection, in, out);
    } else if (model->projection_mode == TRINE_S2_PROJ_SPARSE) {
        trine_project_majority_sparse_sign(&model->projection, in, out);
    } else {
        trine_project_majority_sign(&model->projection, in, out);
    }
}

/* ── Core Forward Pass ──────────────────────────────────────────────── */

int trine_s2_encode_from_trits(const trine_s2_model_t *model,
                                const uint8_t stage1[TRINE_S2_DIM],
                                uint32_t depth, uint8_t out[TRINE_S2_DIM])
{
    if (!model || !stage1 || !out) return -1;
    if (depth > TRINE_CASCADE_MAX_DEPTH) return -1;

    /* Step 1: Projection */
    uint8_t projected[TRINE_S2_DIM];
    apply_projection(model, stage1, projected);

    /* Step 2: Depth processing */
    if (depth == 0) {
        memcpy(out, projected, TRINE_S2_DIM);
        return 0;
    }

    uint8_t cur[TRINE_S2_DIM], nxt[TRINE_S2_DIM];
    memcpy(cur, projected, TRINE_S2_DIM);

    if (model->stacked_depth) {
        /* Stacked: re-apply learned projection at each depth */
        for (uint32_t d = 0; d < depth; d++) {
            apply_projection(model, cur, nxt);
            memcpy(cur, nxt, TRINE_S2_DIM);
        }
    } else if (model->cascade &&
               trine_learned_cascade_n_cells(model->cascade) > 0) {
        /* Cascade: random mixing network */
        for (uint32_t d = 0; d < depth; d++) {
            trine_learned_cascade_tick(model->cascade, cur, nxt);
            memcpy(cur, nxt, TRINE_S2_DIM);
        }
    }

    memcpy(out, cur, TRINE_S2_DIM);
    return 0;
}

int trine_s2_encode(const trine_s2_model_t *model,
                     const char *text, size_t len,
                     uint32_t depth, uint8_t out[TRINE_S2_DIM])
{
    if (!model || !text || !out) return -1;
    if (depth > TRINE_CASCADE_MAX_DEPTH) return -1;

    /* Stage-1 shingle encode */
    uint8_t stage1[TRINE_S2_DIM];
    if (trine_encode_shingle(text, len, stage1) != 0) return -1;

    return trine_s2_encode_from_trits(model, stage1, depth, out);
}

int trine_s2_encode_depths(const trine_s2_model_t *model,
                            const char *text, size_t len,
                            uint32_t max_depth, uint8_t *out,
                            size_t out_size)
{
    if (!model || !text || !out || max_depth == 0) return -1;
    if (max_depth > TRINE_CASCADE_MAX_DEPTH) return -1;
    if (out_size < (size_t)max_depth * TRINE_S2_DIM) return -1;

    /* Stage-1 shingle encode */
    uint8_t stage1[TRINE_S2_DIM];
    if (trine_encode_shingle(text, len, stage1) != 0) return -1;

    /* Projection */
    uint8_t projected[TRINE_S2_DIM];
    apply_projection(model, stage1, projected);

    /* Depth 0 = projection only */
    memcpy(out, projected, TRINE_S2_DIM);

    if (max_depth == 1) return 0;

    /* Depth ticks */
    uint8_t cur[TRINE_S2_DIM], nxt[TRINE_S2_DIM];
    memcpy(cur, projected, TRINE_S2_DIM);

    for (uint32_t d = 1; d < max_depth; d++) {
        if (model->stacked_depth) {
            apply_projection(model, cur, nxt);
        } else {
            trine_learned_cascade_tick(model->cascade, cur, nxt);
        }
        memcpy(out + d * TRINE_S2_DIM, nxt, TRINE_S2_DIM);
        memcpy(cur, nxt, TRINE_S2_DIM);
    }

    return 0;
}

/* ── Batch Encoding ─────────────────────────────────────────────────── */

int trine_s2_encode_batch(
    const trine_s2_model_t *model,
    const char *const *texts,
    const size_t *lens,
    size_t n,
    int depth,
    uint8_t *out)
{
    if (!model || !out) return -1;
    if (n == 0) return 0;
    if (!texts || !lens) return -1;
    if (depth < 0 || (uint32_t)depth > TRINE_CASCADE_MAX_DEPTH) return -1;

    /* Stage-1 batch encode into a temporary buffer */
    uint8_t *s1_buf = (uint8_t *)malloc(n * TRINE_S2_DIM);
    if (!s1_buf) return -1;

    if (trine_encode_shingle_batch(texts, lens, n, s1_buf) != 0) {
        free(s1_buf);
        return -1;
    }

    /* Project + cascade each Stage-1 vector into the output */
    for (size_t i = 0; i < n; i++) {
        if (trine_s2_encode_from_trits(model,
                                        s1_buf + i * TRINE_S2_DIM,
                                        (uint32_t)depth,
                                        out + i * TRINE_S2_DIM) != 0) {
            free(s1_buf);
            return -1;
        }
    }

    free(s1_buf);
    return 0;
}

/* ── Comparison ─────────────────────────────────────────────────────── */

float trine_s2_compare(const uint8_t a[TRINE_S2_DIM],
                        const uint8_t b[TRINE_S2_DIM],
                        const void *lens)
{
    if (!a || !b) return -1.0f;

    if (lens) {
        return trine_s1_compare(a, b, (const trine_s1_lens_t *)lens);
    }

    /* Uniform cosine (no lens) */
    double dot = 0.0, ma = 0.0, mb = 0.0;
    for (int i = 0; i < TRINE_S2_DIM; i++) {
        double va = (double)a[i];
        double vb = (double)b[i];
        dot += va * vb;
        ma  += va * va;
        mb  += vb * vb;
    }
    if (ma < 1e-12 || mb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(ma) * sqrt(mb)));
}

/* ── Gate-Aware Comparison (Phase B1) ──────────────────────────────── */

float trine_s2_compare_gated(const trine_s2_model_t *model,
                              const uint8_t a[TRINE_S2_DIM],
                              const uint8_t b[TRINE_S2_DIM])
{
    if (!model || !a || !b) return 0.0f;

    /* Determine which channels are active via majority-voted diagonal gates */
    const trine_projection_t *proj = &model->projection;
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;

    for (int i = 0; i < TRINE_S2_DIM; i++) {
        /* Count non-zero gates across K=3 copies */
        int active = 0;
        for (int k = 0; k < TRINE_PROJECT_K; k++) {
            if (proj->W[k][i][i] != 0) active++;
        }
        /* Skip if majority of gates are zero (uninformative channel) */
        if (active < 2) continue;

        double va = (double)a[i] - 1.0;
        double vb = (double)b[i] - 1.0;
        dot    += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    if (norm_a < 1e-12 || norm_b < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(norm_a) * sqrt(norm_b)));
}

/* ── Per-Chain Blend Comparison (Phase B2) ─────────────────────────── */

static float chain_cosine_centered(const uint8_t *a, const uint8_t *b, int n)
{
    double dot = 0.0, ma = 0.0, mb = 0.0;
    for (int i = 0; i < n; i++) {
        double va = (double)a[i] - 1.0;
        double vb = (double)b[i] - 1.0;
        dot += va * vb;
        ma  += va * va;
        mb  += vb * vb;
    }
    if (ma < 1e-12 || mb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(ma) * sqrt(mb)));
}

float trine_s2_compare_chain_blend(const uint8_t s1_a[TRINE_S2_DIM],
                                    const uint8_t s1_b[TRINE_S2_DIM],
                                    const uint8_t s2_a[TRINE_S2_DIM],
                                    const uint8_t s2_b[TRINE_S2_DIM],
                                    const float alpha[4])
{
    if (!s1_a || !s1_b || !s2_a || !s2_b || !alpha) return 0.0f;

    /* Default lens: uniform weighting across chains */
    static const float lens[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float total = 0.0f;
    float w_sum = 0.0f;

    for (int c = 0; c < 4; c++) {
        int off = c * 60;
        float s1_c = chain_cosine_centered(&s1_a[off], &s1_b[off], 60);
        float s2_c = chain_cosine_centered(&s2_a[off], &s2_b[off], 60);
        float blended_c = alpha[c] * s1_c + (1.0f - alpha[c]) * s2_c;
        total += lens[c] * blended_c;
        w_sum += lens[c];
    }

    return (w_sum > 0.0f) ? total / w_sum : 0.0f;
}

/* ── Adaptive Blend ────────────────────────────────────────────────── */

void trine_s2_set_adaptive_alpha(trine_s2_model_t *model, const float buckets[10])
{
    if (!model) return;

    if (!buckets) {
        /* Disable adaptive blending */
        free(model->adaptive_alpha);
        model->adaptive_alpha = NULL;
        return;
    }

    if (!model->adaptive_alpha) {
        model->adaptive_alpha = malloc(10 * sizeof(float));
        if (!model->adaptive_alpha) return;
    }
    memcpy(model->adaptive_alpha, buckets, 10 * sizeof(float));
}

static float full_centered_cosine(const uint8_t *a, const uint8_t *b, int n)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        double va = (double)a[i] - 1.0;
        double vb = (double)b[i] - 1.0;
        dot += va * vb;
        na  += va * va;
        nb  += vb * vb;
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

float trine_s2_compare_adaptive_blend(
    const trine_s2_model_t *model,
    const uint8_t s1_a[TRINE_S2_DIM], const uint8_t s1_b[TRINE_S2_DIM],
    const uint8_t s2_a[TRINE_S2_DIM], const uint8_t s2_b[TRINE_S2_DIM])
{
    if (!model || !model->adaptive_alpha) return 0.0f;
    if (!s1_a || !s1_b || !s2_a || !s2_b) return 0.0f;

    /* Compute S1 similarity (uniform cosine, raw values) */
    double dot = 0.0, ma = 0.0, mb = 0.0;
    for (int i = 0; i < TRINE_S2_DIM; i++) {
        double va = (double)s1_a[i];
        double vb = (double)s1_b[i];
        dot += va * vb;
        ma  += va * va;
        mb  += vb * vb;
    }
    float s1_sim = 0.0f;
    if (ma >= 1e-12 && mb >= 1e-12) {
        s1_sim = (float)(dot / (sqrt(ma) * sqrt(mb)));
    }

    /* Look up alpha from bucket: floor(s1_sim * 10), clamped to [0, 9] */
    int bucket = (int)(s1_sim * 10.0f);
    if (bucket < 0)  bucket = 0;
    if (bucket > 9)  bucket = 9;
    float alpha = model->adaptive_alpha[bucket];

    /* Compute S2 centered cosine */
    float s2_sim = full_centered_cosine(s2_a, s2_b, TRINE_S2_DIM);

    return alpha * s1_sim + (1.0f - alpha) * s2_sim;
}

/* ── Configuration ─────────────────────────────────────────────────── */

void trine_s2_set_projection_mode(trine_s2_model_t *model, int mode)
{
    if (model) model->projection_mode = mode;
}

/* ── Introspection ──────────────────────────────────────────────────── */

int trine_s2_info(const trine_s2_model_t *model, trine_s2_info_t *info)
{
    if (!model || !info) return -1;

    info->projection_k    = TRINE_PROJECT_K;
    info->projection_dims = TRINE_PROJECT_DIM;
    info->cascade_cells   = model->cascade
                            ? trine_learned_cascade_n_cells(model->cascade) : 0;
    info->max_depth       = model->cascade
                            ? trine_learned_cascade_max_depth(model->cascade) : 0;
    info->is_identity     = model->is_identity;

    return 0;
}

int trine_s2_get_projection_mode(const trine_s2_model_t *model)
{
    if (!model) return -1;
    return model->projection_mode;
}

const uint8_t *trine_s2_get_block_projection(const trine_s2_model_t *model)
{
    if (!model) return NULL;
    if (model->projection_mode != TRINE_S2_PROJ_BLOCK_DIAG) return NULL;
    return model->block_weights;
}

const void *trine_s2_get_projection(const trine_s2_model_t *model)
{
    if (!model) return NULL;
    return &model->projection;
}

uint32_t trine_s2_get_cascade_cells(const trine_s2_model_t *model)
{
    if (!model || !model->cascade) return 0;
    return trine_learned_cascade_n_cells(model->cascade);
}

uint32_t trine_s2_get_default_depth(const trine_s2_model_t *model)
{
    if (!model) return 0;
    return model->default_depth;
}

int trine_s2_is_identity(const trine_s2_model_t *model)
{
    if (!model) return 0;
    return model->is_identity;
}
