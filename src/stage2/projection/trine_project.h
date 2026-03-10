/* =====================================================================
 * TRINE Stage-2 — Ternary Projection Layer
 * =====================================================================
 *
 * K=3 majority-vote ternary matmul over Z_3^240.
 *
 * The projection layer rotates a 240-trit surface fingerprint (Stage-1)
 * into a 240-trit semantic embedding space. Each projection matrix
 * W[240][240] operates over Z_3: accumulator = sum(W[i][j] * x[j]),
 * output[i] = acc % 3.  All arithmetic is integer; zero floats.
 *
 * K=3 independent projections are combined via per-channel majority
 * vote, providing noise resistance: any single projection can be
 * wrong on a channel and the majority still wins.
 *
 * Max accumulator value per row: 240 * 2 * 2 = 960, fits in int32.
 *
 * ===================================================================== */

#ifndef TRINE_PROJECT_H
#define TRINE_PROJECT_H

#include <stdint.h>
#include <stddef.h>

#define TRINE_PROJECT_K    3     /* Number of projection copies      */
#define TRINE_PROJECT_DIM  240   /* Input/output dimensionality      */

/* K=3 copies of 240x240 ternary weight matrices (~169 KB total) */
typedef struct {
    uint8_t W[TRINE_PROJECT_K][TRINE_PROJECT_DIM][TRINE_PROJECT_DIM];
} trine_projection_t;

/* Single-matrix ternary matmul: y[i] = (sum_j W[i][j] * x[j]) % 3.
 * Pure integer, zero float.  W and x must contain values in {0,1,2}. */
void trine_project_single(const uint8_t W[TRINE_PROJECT_DIM][TRINE_PROJECT_DIM],
                           const uint8_t x[TRINE_PROJECT_DIM],
                           uint8_t y[TRINE_PROJECT_DIM]);

/* K=3 majority-vote projection: apply 3 independent projections,
 * take per-channel majority.  Tie-break: first projection wins. */
void trine_project_majority(const trine_projection_t *proj,
                             const uint8_t x[TRINE_PROJECT_DIM],
                             uint8_t y[TRINE_PROJECT_DIM]);

/* Initialize all K matrices to identity: W[k][i][j] = (i==j ? 1 : 0).
 * Identity matmul: y[i] = (1 * x[i]) % 3 = x[i].
 * Majority of (x[i], x[i], x[i]) = x[i].  So identity projection
 * is a perfect pass-through. */
void trine_projection_identity(trine_projection_t *proj);

/* Fill all K matrices with pseudorandom trits (0/1/2) using LCG.
 * Deterministic: same seed always produces same matrices. */
void trine_projection_random(trine_projection_t *proj, uint64_t seed);

/* Sign-based ternary projection: centers W and x to {-1,0,+1},
 * computes dot product, quantizes by sign:
 *   acc = sum((W[i][j]-1) * (x[j]-1))
 *   y[i] = (acc > 0) ? 2 : (acc < 0) ? 0 : 1
 * Preserves distance structure much better than mod-3 reduction.
 * Pure integer, zero float. */
void trine_project_single_sign(const uint8_t W[TRINE_PROJECT_DIM][TRINE_PROJECT_DIM],
                                const uint8_t x[TRINE_PROJECT_DIM],
                                uint8_t y[TRINE_PROJECT_DIM]);

/* K=3 majority-vote with sign-based projection. */
void trine_project_majority_sign(const trine_projection_t *proj,
                                  const uint8_t x[TRINE_PROJECT_DIM],
                                  uint8_t y[TRINE_PROJECT_DIM]);

/* Diagonal gating: uses only the diagonal of W for per-channel ops.
 * W[i][i]=2 (positive correlation) → keep: y[i] = x[i]
 * W[i][i]=1 (negative correlation) → flip: y[i] = (3-x[i])%3
 * W[i][i]=0 (uninformative)        → zero: y[i] = 1 (neutral)
 * K=3 majority vote across the K projections' diagonals.
 * Preserves channel independence. */
void trine_project_diagonal_gate(const trine_projection_t *proj,
                                  const uint8_t x[TRINE_PROJECT_DIM],
                                  uint8_t y[TRINE_PROJECT_DIM]);

/* Sparse sign-based projection (single matrix):
 * W[i][j]=0 entries are SKIPPED (absent), not treated as -1.
 * W[i][j]=2 → +1 (positive correlation), W[i][j]=1 → -1 (negative).
 * acc = sum over non-zero W: w_c * (x[j]-1)
 * y[i] = (acc > 0) ? 2 : (acc < 0) ? 0 : 1 */
void trine_project_single_sparse_sign(
    const uint8_t W[TRINE_PROJECT_DIM][TRINE_PROJECT_DIM],
    const uint8_t x[TRINE_PROJECT_DIM],
    uint8_t y[TRINE_PROJECT_DIM]);

/* K=3 majority-vote with sparse sign-based projection. */
void trine_project_majority_sparse_sign(const trine_projection_t *proj,
                                         const uint8_t x[TRINE_PROJECT_DIM],
                                         uint8_t y[TRINE_PROJECT_DIM]);

/* ── Block-Diagonal Projection ─────────────────────────────────────── */

#ifndef TRINE_S2_PROJ_BLOCK_DIAG
#define TRINE_S2_PROJ_BLOCK_DIAG 3  /* Block-diagonal: 4 x 60x60 blocks */
#endif
#define TRINE_S2_N_CHAINS   4
#define TRINE_S2_CHAIN_DIM  60

#ifndef TRINE_S2_DIM
#define TRINE_S2_DIM  240
#endif

/* Block-diagonal projection: 4 independent 60x60 ternary matmuls.
 * W_block[chain][60][60] contains ternary weights for each chain.
 * x[240] input, y[240] output. Each chain's 60 dims are projected
 * independently using its own 60x60 block. Z3 reduction per output. */
void trine_project_block_diagonal(
    const uint8_t W_block[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM],
    const uint8_t x[TRINE_S2_DIM],
    uint8_t y[TRINE_S2_DIM]);

/* K=3 majority vote over block-diagonal projections.
 * W_blocks[K][4][60][60], input x[240], output y[240].
 * Each of K projections is applied independently, then per-channel majority vote. */
void trine_projection_majority_block(
    const uint8_t *W_blocks,  /* K * 4 * 60 * 60 bytes */
    int K,
    const uint8_t x[TRINE_S2_DIM],
    uint8_t y[TRINE_S2_DIM]);

/* Initialize K block-diagonal projections as identity (within each chain) */
void trine_projection_block_identity(uint8_t *W_blocks, int K);

/* Initialize K block-diagonal projections randomly */
void trine_projection_block_random(uint8_t *W_blocks, int K, uint64_t seed);

#endif /* TRINE_PROJECT_H */
