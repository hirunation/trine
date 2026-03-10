/* =====================================================================
 * TRINE Stage-2 — Semantic Embedding Layer (Public API)
 * =====================================================================
 *
 * Forward pass:
 *   text → Stage-1 encode → x₀ ∈ Z₃²⁴⁰
 *     → Projection (K=3 majority-vote matmul) → x₁ ∈ Z₃²⁴⁰
 *     → Cascade tick 1 → x₂ ∈ Z₃²⁴⁰
 *     → Cascade tick N → x_{N+1} ∈ Z₃²⁴⁰
 *
 * Every intermediate vector is a valid 240-trit embedding.
 * The inference path (encode + project + cascade) is ZERO FLOAT.
 * Comparison (trine_s2_compare) uses float via the Stage-1 lens system.
 *
 * Thread-safe: the model struct is immutable after construction.
 *
 * ===================================================================== */

#ifndef TRINE_STAGE2_H
#define TRINE_STAGE2_H

#include <stdint.h>
#include <stddef.h>
#include "trine_error.h"

#define TRINE_S2_DIM  240

typedef struct trine_s2_model trine_s2_model_t;

/* ── Lifecycle ──────────────────────────────────────────────────────── */

/* Create an identity model: identity projection + zero-cell cascade.
 * trine_s2_encode() on this model returns the Stage-1 encoding unchanged.
 * This IS the backward-compatibility contract. */
trine_s2_model_t *trine_s2_create_identity(void);

/* Create a random model: random projection + random cascade topology.
 * n_cells = number of cascade mixing cells (512 is a good default).
 * seed = deterministic PRNG seed.
 * Returns NULL on allocation failure. */
trine_s2_model_t *trine_s2_create_random(uint32_t n_cells, uint64_t seed);

/* Create a model from pre-trained parts (projection + cascade topology).
 * proj:      frozen ternary projection weights (copied into model).
 * n_cells:   number of cascade mixing cells.
 * topo_seed: PRNG seed for deterministic cascade topology.
 * Returns NULL on allocation failure. */
trine_s2_model_t *trine_s2_create_from_parts(const void *proj,
                                               uint32_t n_cells,
                                               uint64_t topo_seed);

/* Create a model with block-diagonal projection weights.
 * block_weights: K * 4 * 60 * 60 bytes of per-chain ternary weights (copied).
 * K:            number of projection copies (typically 3).
 * n_cells:      number of cascade mixing cells.
 * topo_seed:    PRNG seed for deterministic cascade topology.
 * Returns NULL on allocation failure. */
trine_s2_model_t *trine_s2_create_block_diagonal(
    const uint8_t *block_weights,
    int K,
    uint32_t n_cells,
    uint64_t topo_seed);

/* Free a model and all internal structures. */
void trine_s2_free(trine_s2_model_t *model);

/* ── Core Forward Pass (ZERO FLOAT) ─────────────────────────────── */

/* Full pipeline: Stage-1 shingle encode → projection → cascade.
 * depth = number of cascade ticks (0 = projection only).
 * Returns 0 on success, -1 on error (null model/text). */
int trine_s2_encode(const trine_s2_model_t *model,
                     const char *text, size_t len,
                     uint32_t depth, uint8_t out[TRINE_S2_DIM]);

/* From pre-computed Stage-1 trits: projection → cascade.
 * Useful when Stage-1 encoding is already available. */
int trine_s2_encode_from_trits(const trine_s2_model_t *model,
                                const uint8_t stage1[TRINE_S2_DIM],
                                uint32_t depth, uint8_t out[TRINE_S2_DIM]);

/* Multi-depth: extract embedding at every depth 0..max_depth-1.
 * out must be pre-allocated: max_depth * 240 bytes.
 * out_size = total byte capacity of `out` buffer.
 * Returns -1 if out_size < max_depth * 240 or max_depth > 64.
 * out[d*240 .. d*240+239] = embedding at depth d.
 * Depth 0 = projection-only output. */
int trine_s2_encode_depths(const trine_s2_model_t *model,
                            const char *text, size_t len,
                            uint32_t max_depth, uint8_t *out,
                            size_t out_size);

/* ── Batch Encoding ────────────────────────────────────────────────── */

/* Batch Stage-2 encode: encodes n texts into pre-allocated output array.
 * texts[i] points to text of length lens[i].
 * depth  = number of cascade ticks (0 = projection only).
 * out must point to n * 240 bytes.
 * Returns 0 on success, -1 on any error (null model, OOM, etc.). */
int trine_s2_encode_batch(
    const trine_s2_model_t *model,
    const char *const *texts,
    const size_t *lens,
    size_t n,
    int depth,
    uint8_t *out);

/* ── Comparison (uses float, delegates to Stage-1 lens) ──────────── */

/* Compare two Stage-2 embeddings using a lens.
 * lens may be NULL (uniform weights) or a trine_s1_lens_t pointer.
 * Returns similarity in [0.0, 1.0], or -1.0 on error. */
float trine_s2_compare(const uint8_t a[TRINE_S2_DIM],
                        const uint8_t b[TRINE_S2_DIM],
                        const void *lens);

/* ── Gate-Aware Comparison (Phase B1) ──────────────────────────────── */

/* Compare two Stage-2 embeddings using only channels with active gates.
 * Channels where all K=3 diagonal gates are 0 (uninformative) are
 * excluded from the cosine similarity computation.  This eliminates
 * noise from zeroed channels and focuses on learned signal.
 * Returns similarity in [-1.0, 1.0], or 0.0 if no active channels. */
float trine_s2_compare_gated(const trine_s2_model_t *model,
                              const uint8_t a[TRINE_S2_DIM],
                              const uint8_t b[TRINE_S2_DIM]);

/* ── Per-Chain Blend Comparison (Phase B2) ─────────────────────────── */

/* Compare using per-chain alpha blend of S1 and S2 similarities.
 * s1_a, s1_b: Stage-1 embeddings (pre-projection)
 * s2_a, s2_b: Stage-2 embeddings (post-projection)
 * alpha: 4-element array, one per chain (Edit, Morph, Phrase, Vocab).
 *        alpha[c]=1.0 means pure S1 for chain c, alpha[c]=0.0 means pure S2. */
float trine_s2_compare_chain_blend(const uint8_t s1_a[TRINE_S2_DIM],
                                    const uint8_t s1_b[TRINE_S2_DIM],
                                    const uint8_t s2_a[TRINE_S2_DIM],
                                    const uint8_t s2_b[TRINE_S2_DIM],
                                    const float alpha[4]);

/* ── Introspection ─────────────────────────────────────────────────── */

typedef struct {
    uint32_t projection_k;      /* Number of projection copies (3) */
    uint32_t projection_dims;   /* Projection dimensionality (240) */
    uint32_t cascade_cells;     /* Number of cascade mixing cells */
    uint32_t max_depth;         /* Max cascade depth */
    int      is_identity;       /* 1 if model is identity (pass-through) */
} trine_s2_info_t;

/* Query model parameters.  Returns 0 on success, -1 on null model. */
int trine_s2_info(const trine_s2_model_t *model, trine_s2_info_t *info);

/* Projection modes for trained (non-identity) models:
 * 0 = sign-based (default) — full 240x240 centered dot product + sign
 * 1 = diagonal gating — per-channel keep/flip/zero using W[k][i][i]
 * 2 = sparse sign — like sign but W=0 entries are skipped (absent)
 * 3 = block-diagonal — 4 independent 60x60 chain-local projections */
#define TRINE_S2_PROJ_SIGN     0
#define TRINE_S2_PROJ_DIAGONAL 1
#define TRINE_S2_PROJ_SPARSE   2

#ifndef TRINE_S2_PROJ_BLOCK_DIAG
#define TRINE_S2_PROJ_BLOCK_DIAG 3
#endif

/* Enable stacked depth: re-apply projection instead of cascade ticks.
 * When set, each depth tick re-projects using learned weights rather
 * than running the random cascade network. */
void trine_s2_set_stacked_depth(trine_s2_model_t *model, int enable);
int  trine_s2_get_stacked_depth(const trine_s2_model_t *model);

void trine_s2_set_projection_mode(trine_s2_model_t *model, int mode);

/* Get projection mode (0=sign, 1=diagonal). Returns -1 on null model. */
int trine_s2_get_projection_mode(const trine_s2_model_t *model);

/* Get a read-only pointer to the block-diagonal projection weights.
 * Returns NULL if model is NULL or not in block-diagonal mode.
 * Layout: K * 4 * 60 * 60 bytes = 43,200 bytes total. */
const uint8_t *trine_s2_get_block_projection(const trine_s2_model_t *model);

/* Get a read-only pointer to the projection weights.
 * Returns NULL on null model. The pointer is valid for the model's lifetime. */
const void *trine_s2_get_projection(const trine_s2_model_t *model);

/* Get cascade cell count. Returns 0 on null model or no cascade. */
uint32_t trine_s2_get_cascade_cells(const trine_s2_model_t *model);

/* Get default depth. Returns 0 on null model. */
uint32_t trine_s2_get_default_depth(const trine_s2_model_t *model);

/* Check if model is an identity model. Returns 1 if identity, 0 otherwise. */
int trine_s2_is_identity(const trine_s2_model_t *model);

/* ── Adaptive Blend ────────────────────────────────────────────────── */

/* Set per-S1-bucket alpha values for adaptive blending.
 * buckets[10]: alpha for S1 similarity in [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
 * When set, trine_s2_compare_adaptive_blend() uses the bucket lookup.
 * Pass NULL to disable adaptive blending. */
void trine_s2_set_adaptive_alpha(trine_s2_model_t *model, const float buckets[10]);

/* Adaptive blend: alpha selected based on S1 similarity bucket.
 * Computes S1 similarity (uniform cosine), looks up alpha from the
 * model's adaptive_alpha buckets, then blends:
 *   result = alpha * s1_sim + (1 - alpha) * s2_centered_cosine
 * Returns 0.0 if adaptive_alpha is not set or on error. */
float trine_s2_compare_adaptive_blend(
    const trine_s2_model_t *model,
    const uint8_t s1_a[TRINE_S2_DIM], const uint8_t s1_b[TRINE_S2_DIM],
    const uint8_t s2_a[TRINE_S2_DIM], const uint8_t s2_b[TRINE_S2_DIM]);

#endif /* TRINE_STAGE2_H */
