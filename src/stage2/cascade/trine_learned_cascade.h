/* =====================================================================
 * TRINE Stage-2 — Learned Cascade Engine
 * =====================================================================
 *
 * A lightweight ternary mixing network using the ENDO[27][3] table
 * from trine_algebra.h.  Each "cell" reads from one input channel,
 * applies an endomorphism, and accumulates (Z_3 addition) into one
 * output channel.
 *
 * The cascade is iterated for N "ticks" (depth).  Each tick reads
 * from the previous output and writes a new 240-trit vector.  A
 * residual connection (memcpy) ensures identity passthrough when
 * n_cells = 0.
 *
 * Thread-safe: the cascade struct is immutable after construction.
 * No float operations in the tick path.
 *
 * ===================================================================== */

#ifndef TRINE_LEARNED_CASCADE_H
#define TRINE_LEARNED_CASCADE_H

#include <stdint.h>
#include <stddef.h>

#define TRINE_CASCADE_DIM          240
#define TRINE_CASCADE_MAX_DEPTH    64
#define TRINE_CASCADE_DEFAULT_DEPTH 4

typedef struct trine_learned_cascade trine_learned_cascade_t;

typedef struct {
    uint32_t n_cells;       /* Number of mixing cells (0 = identity passthrough) */
    uint32_t max_depth;     /* Max cascade ticks (default 32) */
} trine_cascade_config_t;

#define TRINE_CASCADE_CONFIG_DEFAULT { .n_cells = 512, .max_depth = 32 }

/* Create a cascade with the given config.  Allocates cell arrays.
 * Cell weights are zeroed (identity behavior until initialized).
 * Returns NULL on allocation failure. */
trine_learned_cascade_t *trine_learned_cascade_create(const trine_cascade_config_t *config);

/* Free a cascade and all its cell arrays. */
void trine_learned_cascade_free(trine_learned_cascade_t *lc);

/* Execute one cascade tick.  Reads from in[240], writes to out[240].
 * in and out may NOT alias.
 *
 * Algorithm:
 *   1. memcpy(out, in, 240)          -- residual connection
 *   2. for each cell k:
 *        val = ENDO[cell_endos[k]][in[cell_src[k]]]
 *        out[cell_dst[k]] = (out[cell_dst[k]] + val) % 3
 *
 * Zero float.  At n_cells=0, output = input via residual. */
void trine_learned_cascade_tick(const trine_learned_cascade_t *lc,
                                 const uint8_t in[TRINE_CASCADE_DIM],
                                 uint8_t out[TRINE_CASCADE_DIM]);

/* Get number of cells. */
uint32_t trine_learned_cascade_n_cells(const trine_learned_cascade_t *lc);

/* Get max depth. */
uint32_t trine_learned_cascade_max_depth(const trine_learned_cascade_t *lc);

/* Direct access to cell arrays for initialization (topology generators).
 * Returns pointer to internal arrays.  Caller must not free. */
uint8_t  *trine_learned_cascade_endos(trine_learned_cascade_t *lc);
uint16_t *trine_learned_cascade_srcs(trine_learned_cascade_t *lc);
uint16_t *trine_learned_cascade_dsts(trine_learned_cascade_t *lc);

#endif /* TRINE_LEARNED_CASCADE_H */
