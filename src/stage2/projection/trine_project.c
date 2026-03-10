/* =====================================================================
 * TRINE Stage-2 — Ternary Matmul (single projection)
 * =====================================================================
 *
 * y[i] = (sum_j W[i][j] * x[j]) % 3
 *
 * Pure integer arithmetic.  Max accumulator = 240 * 2 * 2 = 960.
 * Fits comfortably in int32.
 *
 * ===================================================================== */

#include "trine_project.h"

void trine_project_single(const uint8_t W[TRINE_PROJECT_DIM][TRINE_PROJECT_DIM],
                           const uint8_t x[TRINE_PROJECT_DIM],
                           uint8_t y[TRINE_PROJECT_DIM])
{
    for (int i = 0; i < TRINE_PROJECT_DIM; i++) {
        int acc = 0;
        for (int j = 0; j < TRINE_PROJECT_DIM; j++)
            acc += (int)W[i][j] * (int)x[j];
        y[i] = (uint8_t)(((acc % 3) + 3) % 3);
    }
}

void trine_project_single_sign(const uint8_t W[TRINE_PROJECT_DIM][TRINE_PROJECT_DIM],
                                const uint8_t x[TRINE_PROJECT_DIM],
                                uint8_t y[TRINE_PROJECT_DIM])
{
    for (int i = 0; i < TRINE_PROJECT_DIM; i++) {
        int acc = 0;
        for (int j = 0; j < TRINE_PROJECT_DIM; j++) {
            int w_c = (int)W[i][j] - 1;   /* center: {0,1,2} -> {-1,0,+1} */
            int x_c = (int)x[j] - 1;
            acc += w_c * x_c;
        }
        /* Quantize by sign: positive → 2, negative → 0, zero → 1 */
        y[i] = (acc > 0) ? 2 : (acc < 0) ? 0 : 1;
    }
}

void trine_project_single_sparse_sign(
    const uint8_t W[TRINE_PROJECT_DIM][TRINE_PROJECT_DIM],
    const uint8_t x[TRINE_PROJECT_DIM],
    uint8_t y[TRINE_PROJECT_DIM])
{
    for (int i = 0; i < TRINE_PROJECT_DIM; i++) {
        int acc = 0;
        for (int j = 0; j < TRINE_PROJECT_DIM; j++) {
            uint8_t w = W[i][j];
            if (w == 0) continue;  /* absent entry — skip */
            int w_c = (w == 2) ? 1 : -1;  /* 2 → +1, 1 → -1 */
            int x_c = (int)x[j] - 1;
            acc += w_c * x_c;
        }
        y[i] = (acc > 0) ? 2 : (acc < 0) ? 0 : 1;
    }
}

void trine_project_block_diagonal(
    const uint8_t W_block[TRINE_S2_N_CHAINS][TRINE_S2_CHAIN_DIM][TRINE_S2_CHAIN_DIM],
    const uint8_t x[TRINE_S2_DIM],
    uint8_t y[TRINE_S2_DIM])
{
    for (int c = 0; c < TRINE_S2_N_CHAINS; c++) {
        int offset = c * TRINE_S2_CHAIN_DIM;
        for (int i = 0; i < TRINE_S2_CHAIN_DIM; i++) {
            int acc = 0;
            for (int j = 0; j < TRINE_S2_CHAIN_DIM; j++) {
                acc += (int)W_block[c][i][j] * (int)x[offset + j];
            }
            y[offset + i] = (uint8_t)(((acc % 3) + 3) % 3);
        }
    }
}
