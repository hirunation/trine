/* =====================================================================
 * TRINE Stage-2 — Accumulator Persistence (.trine2a format)
 * =====================================================================
 *
 * Binary format for saving/loading Hebbian accumulator state.
 * Enables curriculum learning, warm-start, and incremental training
 * without retraining from scratch.
 *
 * File layout:
 *   Bytes  0-3:   Magic "TR2A" (0x41325254)
 *   Bytes  4-7:   Version (1)
 *   Bytes  8-11:  Flags (bit 0: diagonal mode, bit 1: block-diagonal)
 *   Bytes 12-15:  projection_k (3)
 *   Bytes 16-19:  projection_dim (240)
 *   Bytes 20-23:  total_pairs_observed (uint32_t)
 *   Bytes 24-27:  similarity_threshold (float32)
 *   Bytes 28-31:  freeze_target_density (float32)
 *   Bytes 32-55:  reserved (24 bytes, zero)
 *   Bytes 56-63:  header_checksum (FNV-1a of bytes 0-55)
 *   Bytes 64-...: accumulators (int32_t payload, size depends on mode)
 *                 Full:  K * DIM * DIM * 4   = 691,200 bytes
 *                 Block: K * 4 * 60 * 60 * 4 = 172,800 bytes
 *   ...+8:        payload_checksum (FNV-1a of accumulator data)
 *
 * Total: Full  = 691,272 bytes (~675 KB)
 *        Block = 172,872 bytes (~169 KB)
 *
 * ===================================================================== */

#ifndef TRINE_ACCUMULATOR_PERSIST_H
#define TRINE_ACCUMULATOR_PERSIST_H

#include <stdint.h>
#include "trine_accumulator.h"
#include "trine_hebbian.h"

#define TRINE_ACC_MAGIC           0x41325254u  /* "TR2A" */
#define TRINE_ACC_FORMAT_VERSION  1u
#define TRINE_ACC_FLAG_DIAGONAL   (1u << 0)
#define TRINE_ACC_FLAG_BLOCK_DIAG (1u << 1)   /* Block-diagonal accumulators */

/* Save accumulator state to a .trine2a file.
 * Returns 0 on success, -1 on error. */
int trine_accumulator_save(const trine_accumulator_t *acc,
                            const trine_hebbian_config_t *config,
                            int64_t pairs_observed,
                            const char *path);

/* Load accumulator state from a .trine2a file.
 * Returns newly allocated accumulator, or NULL on error.
 * If config_out is non-NULL, training config is restored.
 * If pairs_out is non-NULL, pairs_observed count is restored. */
trine_accumulator_t *trine_accumulator_load(const char *path,
                                             trine_hebbian_config_t *config_out,
                                             int64_t *pairs_out);

/* Validate a .trine2a file without loading.
 * Returns 0 if valid, -1 on error. */
int trine_accumulator_validate(const char *path);

/* Reconstruct approximate accumulators from a frozen .trine2 model.
 * gate=2 -> counter=+scale, gate=1 -> counter=-scale, gate=0 -> counter=0.
 * Allows continued training from a saved model without accumulator state. */
trine_accumulator_t *trine_accumulator_from_frozen(
    const void *projection_weights,
    int32_t reconstruction_scale);

/* ── Block-Diagonal Accumulator Persistence ────────────────────────── */

/* Save block-diagonal accumulator to a .trine2a file.
 * Sets TRINE_ACC_FLAG_BLOCK_DIAG in header flags.
 * Payload: K * 4 * 60 * 60 * sizeof(int32_t) = 172,800 bytes (K=3).
 * Returns 0 on success, -1 on error. */
int trine_block_accumulator_save(const trine_block_accumulator_t *acc,
                                  float similarity_threshold,
                                  float freeze_target_density,
                                  int projection_mode,
                                  const char *path);

/* Load block-diagonal accumulator from a .trine2a file.
 * Only succeeds if the file has TRINE_ACC_FLAG_BLOCK_DIAG set.
 * Returns newly allocated block accumulator, or NULL on error. */
trine_block_accumulator_t *trine_block_accumulator_load(
    const char *path,
    float *threshold_out,
    float *density_out,
    uint32_t *pairs_out);

#endif /* TRINE_ACCUMULATOR_PERSIST_H */
