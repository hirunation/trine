/* =====================================================================
 * TRINE Stage-2 — Model Persistence
 * =====================================================================
 *
 * Binary format for saving/loading trained Stage-2 models.
 *
 * .trine2 file layout:
 *   Offset  Field                  Size (bytes)
 *   ------  -----                  -----------
 *   0       magic                  4    ("TR2\0" = 0x00325254)
 *   4       version                4    (1)
 *   8       flags                  4    (bit 0 = diagonal mode,
 *                                        bit 1 = identity,
 *                                        bit 2 = block-diagonal)
 *   12      projection_k           4    (3)
 *   16      projection_dim         4    (240)
 *   20      cascade_cells          4    (0-512)
 *   24      cascade_depth          4    (default inference depth)
 *   28      topo_seed              8    (topology PRNG seed)
 *   36      similarity_threshold   4    (float, training param)
 *   40      density                4    (float, freeze density)
 *   44      reserved               20   (zero-filled)
 *   64      checksum               8    (FNV-1a of bytes 0..63)
 *   72      projection weights     W bytes (see below)
 *   72+W    payload checksum       8    (FNV-1a of weights)
 *
 * Payload sizes by mode:
 *   Full-matrix:     K * DIM * DIM         = 172,800 bytes (default)
 *   Block-diagonal:  K * 4 * 60 * 60       =  43,200 bytes (flag bit 2)
 *
 * Total: 80 + W bytes
 *   Full:  172,880 for K=3, DIM=240
 *   Block:  43,280 for K=3, 4 chains x 60
 *
 * For diagonal-only models, the full projection is still stored
 * (only the diagonal is used at inference time, but storing the
 * full matrix enables mode switching and analysis).
 *
 * Thread-safe: all functions are stateless.
 *
 * ===================================================================== */

#ifndef TRINE_S2_PERSIST_H
#define TRINE_S2_PERSIST_H

#include <stdint.h>
#include <stddef.h>

/* Forward declarations */
struct trine_s2_model;

/* Magic: "TR2\0" in little-endian */
#define TRINE_S2_MAGIC    0x00325254u
#define TRINE_S2_FORMAT_VERSION  1u

/* Flags */
#define TRINE_S2_FLAG_DIAGONAL    (1u << 0)
#define TRINE_S2_FLAG_IDENTITY    (1u << 1)
#define TRINE_S2_FLAG_BLOCK_DIAG  (1u << 2)  /* Block-diagonal weights */

/* Header (72 bytes: 64 metadata + 8 checksum) */
typedef struct {
    uint32_t magic;                 /* TRINE_S2_MAGIC                  */
    uint32_t version;               /* TRINE_S2_FORMAT_VERSION         */
    uint32_t flags;                 /* TRINE_S2_FLAG_*                 */
    uint32_t projection_k;          /* Number of projection copies (3) */
    uint32_t projection_dim;        /* Dimensionality (240)            */
    uint32_t cascade_cells;         /* Cascade mixing cells            */
    uint32_t cascade_depth;         /* Default inference depth          */
    uint64_t topo_seed;             /* Cascade topology PRNG seed      */
    float    similarity_threshold;  /* Training similarity threshold   */
    float    density;               /* Freeze target density           */
    uint8_t  reserved[20];          /* Reserved, zero-filled           */
    uint64_t header_checksum;       /* FNV-1a of bytes 0..63           */
} __attribute__((packed)) trine_s2_file_header_t;

_Static_assert(sizeof(trine_s2_file_header_t) == 72,
               "trine_s2_file_header_t must be exactly 72 bytes");

/* ── Save/Load API ───────────────────────────────────────────────── */

/* Save a trained Stage-2 model to a .trine2 file.
 * model:    the model to save (must not be NULL).
 * path:     output file path.
 * config:   training parameters to embed in the header (may be NULL
 *           for defaults: thresh=0.5, density=0.33).
 * Returns 0 on success, -1 on error (message to stderr). */
typedef struct {
    float    similarity_threshold;
    float    density;
    uint64_t topo_seed;
} trine_s2_save_config_t;

int trine_s2_save(const struct trine_s2_model *model,
                   const char *path,
                   const trine_s2_save_config_t *config);

/* Load a Stage-2 model from a .trine2 file.
 * path:     input file path.
 * Returns the loaded model, or NULL on error (message to stderr).
 * Caller owns the returned model and must free with trine_s2_free(). */
struct trine_s2_model *trine_s2_load(const char *path);

/* Validate a .trine2 file without loading it.
 * Returns 0 if valid, -1 on error (message to stderr). */
int trine_s2_validate(const char *path);

#endif /* TRINE_S2_PERSIST_H */
