/* =====================================================================
 * TRINE Stage-2 — Learned Cascade Engine (implementation)
 * =====================================================================
 *
 * Each tick: residual copy + n_cells endomorphism lookups + Z_3 adds.
 * At n_cells=512, depth=4: ~4 * (240 + 512*3) ~= 7K ops.
 *
 * ===================================================================== */

#include "trine_learned_cascade.h"
#include "trine_algebra.h"
#include <stdlib.h>
#include <string.h>

struct trine_learned_cascade {
    uint32_t n_cells;          /* number of mixing cells */
    uint8_t  *cell_endos;      /* endomorphism index per cell [n_cells] (0-26) */
    uint16_t *cell_src;        /* source channel per cell [n_cells] (0-239) */
    uint16_t *cell_dst;        /* destination channel per cell [n_cells] (0-239) */
    uint32_t max_depth;
};

trine_learned_cascade_t *trine_learned_cascade_create(const trine_cascade_config_t *config)
{
    trine_learned_cascade_t *lc = calloc(1, sizeof(*lc));
    if (!lc) return NULL;

    lc->n_cells   = config->n_cells;
    lc->max_depth = config->max_depth;

    if (lc->n_cells > 0) {
        lc->cell_endos = calloc(lc->n_cells, sizeof(uint8_t));
        lc->cell_src   = calloc(lc->n_cells, sizeof(uint16_t));
        lc->cell_dst   = calloc(lc->n_cells, sizeof(uint16_t));
        if (!lc->cell_endos || !lc->cell_src || !lc->cell_dst) {
            trine_learned_cascade_free(lc);
            return NULL;
        }
    }

    return lc;
}

void trine_learned_cascade_free(trine_learned_cascade_t *lc)
{
    if (!lc) return;
    free(lc->cell_endos);
    free(lc->cell_src);
    free(lc->cell_dst);
    free(lc);
}

void trine_learned_cascade_tick(const trine_learned_cascade_t *lc,
                                 const uint8_t in[TRINE_CASCADE_DIM],
                                 uint8_t out[TRINE_CASCADE_DIM])
{
    memcpy(out, in, TRINE_CASCADE_DIM);   /* residual connection */

    for (uint32_t k = 0; k < lc->n_cells; k++) {
        uint8_t val = TRINE_ENDO[lc->cell_endos[k]][in[lc->cell_src[k]]];
        out[lc->cell_dst[k]] = (out[lc->cell_dst[k]] + val) % 3;
    }
}

uint32_t trine_learned_cascade_n_cells(const trine_learned_cascade_t *lc)
{
    return lc->n_cells;
}

uint32_t trine_learned_cascade_max_depth(const trine_learned_cascade_t *lc)
{
    return lc->max_depth;
}

uint8_t *trine_learned_cascade_endos(trine_learned_cascade_t *lc)
{
    return lc->cell_endos;
}

uint16_t *trine_learned_cascade_srcs(trine_learned_cascade_t *lc)
{
    return lc->cell_src;
}

uint16_t *trine_learned_cascade_dsts(trine_learned_cascade_t *lc)
{
    return lc->cell_dst;
}
