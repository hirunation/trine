/* =====================================================================
 * TRINE Stage-2 — Topology Generators for Learned Cascade
 * =====================================================================
 *
 * Three topology initializers for cascade cell arrays:
 *
 *   random      — uniform random src/dst/endo (LCG seeded)
 *   layered     — structured layers with residual connections
 *   chain_local — chain-aware local mixing (within/between chains)
 *
 * All generators fill the cell_endos, cell_src, cell_dst arrays
 * of an already-created cascade.  The cascade must have n_cells > 0.
 *
 * ===================================================================== */

#include "trine_learned_cascade.h"
#include <string.h>

#define DIM  TRINE_CASCADE_DIM
#define CHAIN_WIDTH 60
#define N_CHAINS    4

/* LCG step: Knuth MMIX */
static inline uint64_t lcg_next(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return *state;
}

void trine_topology_random(trine_learned_cascade_t *lc, uint64_t seed)
{
    uint32_t n = trine_learned_cascade_n_cells(lc);
    if (n == 0) return;

    uint8_t  *endos = trine_learned_cascade_endos(lc);
    uint16_t *srcs  = trine_learned_cascade_srcs(lc);
    uint16_t *dsts  = trine_learned_cascade_dsts(lc);

    uint64_t state = seed;
    for (uint32_t k = 0; k < n; k++) {
        endos[k] = (uint8_t)((lcg_next(&state) >> 32) % 27);
        srcs[k]  = (uint16_t)((lcg_next(&state) >> 32) % DIM);
        dsts[k]  = (uint16_t)((lcg_next(&state) >> 32) % DIM);
    }
}

void trine_topology_layered(trine_learned_cascade_t *lc, uint64_t seed)
{
    uint32_t n = trine_learned_cascade_n_cells(lc);
    if (n == 0) return;

    uint8_t  *endos = trine_learned_cascade_endos(lc);
    uint16_t *srcs  = trine_learned_cascade_srcs(lc);
    uint16_t *dsts  = trine_learned_cascade_dsts(lc);

    /* Divide cells into layers of ~DIM cells each.
     * Within each layer, cell k connects channel k%DIM to a
     * pseudorandom destination, with residual bias (50% chance
     * of dst == src, creating identity-like passthrough). */
    uint64_t state = seed;
    for (uint32_t k = 0; k < n; k++) {
        uint16_t src = (uint16_t)(k % DIM);
        srcs[k] = src;

        uint64_t r = lcg_next(&state);
        /* 50% residual: dst = src */
        if ((r >> 33) & 1) {
            dsts[k] = src;
        } else {
            dsts[k] = (uint16_t)((r >> 32) % DIM);
        }

        endos[k] = (uint8_t)((lcg_next(&state) >> 32) % 27);
    }
}

void trine_topology_chain_local(trine_learned_cascade_t *lc, uint64_t seed)
{
    uint32_t n = trine_learned_cascade_n_cells(lc);
    if (n == 0) return;

    uint8_t  *endos = trine_learned_cascade_endos(lc);
    uint16_t *srcs  = trine_learned_cascade_srcs(lc);
    uint16_t *dsts  = trine_learned_cascade_dsts(lc);

    /* 75% of cells: intra-chain (src and dst in same chain).
     * 25% of cells: inter-chain (src and dst in different chains). */
    uint64_t state = seed;
    for (uint32_t k = 0; k < n; k++) {
        uint64_t r = lcg_next(&state);
        int chain_src = (int)((r >> 32) % N_CHAINS);
        int offset_src = (int)((lcg_next(&state) >> 32) % CHAIN_WIDTH);
        srcs[k] = (uint16_t)(chain_src * CHAIN_WIDTH + offset_src);

        r = lcg_next(&state);
        if ((r >> 33) % 4 < 3) {
            /* Intra-chain: dst in same chain */
            int offset_dst = (int)((r >> 32) % CHAIN_WIDTH);
            dsts[k] = (uint16_t)(chain_src * CHAIN_WIDTH + offset_dst);
        } else {
            /* Inter-chain: dst in different chain */
            int chain_dst = (int)((lcg_next(&state) >> 32) % N_CHAINS);
            int offset_dst = (int)((lcg_next(&state) >> 32) % CHAIN_WIDTH);
            dsts[k] = (uint16_t)(chain_dst * CHAIN_WIDTH + offset_dst);
        }

        endos[k] = (uint8_t)((lcg_next(&state) >> 32) % 27);
    }
}
