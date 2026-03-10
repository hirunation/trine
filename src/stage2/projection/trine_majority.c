/* =====================================================================
 * TRINE Stage-2 — Majority Vote + Projection Initializers
 * =====================================================================
 *
 * majority3(a, b, c): if two or more agree, return that value.
 * If all three differ (0,1,2 in some order), tie-break to first (a).
 *
 * Initializers:
 *   identity — W[k][i][i] = 1, rest = 0.  Pass-through.
 *   random   — LCG-seeded pseudorandom trits.
 *
 * ===================================================================== */

#include "trine_project.h"
#include <string.h>

/* Per-channel majority of three trits.
 * If two or more match, return that value.
 * If all differ (one each of 0,1,2), tie-break to a (first projection). */
static inline uint8_t majority3(uint8_t a, uint8_t b, uint8_t c)
{
    if (a == b || a == c) return a;
    if (b == c) return b;
    return a;  /* all differ: tie-break to first */
}

void trine_project_majority(const trine_projection_t *proj,
                             const uint8_t x[TRINE_PROJECT_DIM],
                             uint8_t y[TRINE_PROJECT_DIM])
{
    uint8_t p0[TRINE_PROJECT_DIM];
    uint8_t p1[TRINE_PROJECT_DIM];
    uint8_t p2[TRINE_PROJECT_DIM];

    trine_project_single(proj->W[0], x, p0);
    trine_project_single(proj->W[1], x, p1);
    trine_project_single(proj->W[2], x, p2);

    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        y[i] = majority3(p0[i], p1[i], p2[i]);
}

void trine_projection_identity(trine_projection_t *proj)
{
    memset(proj, 0, sizeof(*proj));
    for (int k = 0; k < TRINE_PROJECT_K; k++)
        for (int i = 0; i < TRINE_PROJECT_DIM; i++)
            proj->W[k][i][i] = 1;
}

void trine_project_majority_sign(const trine_projection_t *proj,
                                  const uint8_t x[TRINE_PROJECT_DIM],
                                  uint8_t y[TRINE_PROJECT_DIM])
{
    uint8_t p0[TRINE_PROJECT_DIM];
    uint8_t p1[TRINE_PROJECT_DIM];
    uint8_t p2[TRINE_PROJECT_DIM];

    trine_project_single_sign(proj->W[0], x, p0);
    trine_project_single_sign(proj->W[1], x, p1);
    trine_project_single_sign(proj->W[2], x, p2);

    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        y[i] = majority3(p0[i], p1[i], p2[i]);
}

/* Per-channel diagonal gating using only W[k][i][i]. */
static inline uint8_t apply_gate(uint8_t gate, uint8_t x)
{
    if (gate == 2) return x;                    /* keep */
    if (gate == 1) return (uint8_t)((3 - x) % 3); /* flip (Z3 negation) */
    return 1;                                    /* zero → neutral */
}

void trine_project_diagonal_gate(const trine_projection_t *proj,
                                  const uint8_t x[TRINE_PROJECT_DIM],
                                  uint8_t y[TRINE_PROJECT_DIM])
{
    for (int i = 0; i < TRINE_PROJECT_DIM; i++) {
        uint8_t g0 = apply_gate(proj->W[0][i][i], x[i]);
        uint8_t g1 = apply_gate(proj->W[1][i][i], x[i]);
        uint8_t g2 = apply_gate(proj->W[2][i][i], x[i]);
        y[i] = majority3(g0, g1, g2);
    }
}

void trine_project_majority_sparse_sign(const trine_projection_t *proj,
                                         const uint8_t x[TRINE_PROJECT_DIM],
                                         uint8_t y[TRINE_PROJECT_DIM])
{
    uint8_t p0[TRINE_PROJECT_DIM];
    uint8_t p1[TRINE_PROJECT_DIM];
    uint8_t p2[TRINE_PROJECT_DIM];

    trine_project_single_sparse_sign(proj->W[0], x, p0);
    trine_project_single_sparse_sign(proj->W[1], x, p1);
    trine_project_single_sparse_sign(proj->W[2], x, p2);

    for (int i = 0; i < TRINE_PROJECT_DIM; i++)
        y[i] = majority3(p0[i], p1[i], p2[i]);
}

void trine_projection_random(trine_projection_t *proj, uint64_t seed)
{
    /* LCG: x_{n+1} = (a * x_n + c) mod 2^64
     * Knuth MMIX constants. */
    uint64_t state = seed;
    for (int k = 0; k < TRINE_PROJECT_K; k++)
        for (int i = 0; i < TRINE_PROJECT_DIM; i++)
            for (int j = 0; j < TRINE_PROJECT_DIM; j++) {
                state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                proj->W[k][i][j] = (uint8_t)((state >> 32) % 3);
            }
}

/* ── Block-Diagonal Majority Vote + Initializers ───────────────────── */

void trine_projection_majority_block(
    const uint8_t *W_blocks,  /* K * 4 * 60 * 60 bytes */
    int K,
    const uint8_t x[TRINE_S2_DIM],
    uint8_t y[TRINE_S2_DIM])
{
    /* Block size for one projection: N_CHAINS * CHAIN_DIM * CHAIN_DIM */
    const int block_size = TRINE_S2_N_CHAINS * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM;

    /* Apply K block-diagonal projections into temp buffers.
     * For K=3 (the common case), we use static arrays to avoid malloc. */
    uint8_t p0[TRINE_S2_DIM], p1[TRINE_S2_DIM], p2[TRINE_S2_DIM];
    uint8_t *projs[3] = { p0, p1, p2 };

    /* Clamp K to 3 for the majority3 combiner */
    int k_use = (K > 3) ? 3 : K;
    if (k_use < 1) {
        memset(y, 1, TRINE_S2_DIM);
        return;
    }

    for (int k = 0; k < k_use; k++) {
        const uint8_t (*block)[TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM] =
            (const uint8_t (*)[TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM])
            (W_blocks + k * block_size);
        trine_project_block_diagonal(block, x, projs[k]);
    }

    if (k_use == 1) {
        memcpy(y, p0, TRINE_S2_DIM);
    } else if (k_use == 2) {
        /* Two projections: take p0 on tie (effectively p0 wins on disagreement) */
        for (int i = 0; i < TRINE_S2_DIM; i++)
            y[i] = majority3(p0[i], p1[i], p0[i]);
    } else {
        for (int i = 0; i < TRINE_S2_DIM; i++)
            y[i] = majority3(p0[i], p1[i], p2[i]);
    }
}

void trine_projection_block_identity(uint8_t *W_blocks, int K)
{
    const int block_size = TRINE_S2_N_CHAINS * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM;

    /* Zero all weights first */
    memset(W_blocks, 0, (size_t)K * block_size);

    /* Set diagonal entries to 1 within each 60x60 block */
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < TRINE_S2_N_CHAINS; c++) {
            for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++) {
                int idx = k * block_size
                        + c * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM
                        + i * TRINE_S2_CHAIN_DIM + i;
                W_blocks[idx] = 1;
            }
        }
    }
}

void trine_projection_block_random(uint8_t *W_blocks, int K, uint64_t seed)
{
    const int block_size = TRINE_S2_N_CHAINS * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM;
    int total = K * block_size;

    /* LCG: Knuth MMIX constants (same as trine_projection_random) */
    uint64_t state = seed;
    for (int n = 0; n < total; n++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        W_blocks[n] = (uint8_t)((state >> 32) % 3);
    }
}
