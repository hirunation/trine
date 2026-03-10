/* =====================================================================
 * TRINE Stage-2 — 2-bit Trit Packing
 * =====================================================================
 *
 * Packs ternary weights (0, 1, 2) into 2 bits each, yielding 4 trits
 * per byte for a 4x storage reduction.
 *
 * Byte layout (LSB-first):
 *   bits 0-1 : trit[0]
 *   bits 2-3 : trit[1]
 *   bits 4-5 : trit[2]
 *   bits 6-7 : trit[3]
 *
 * Packed sizes by model type:
 *   Diagonal:       720 trits  ->    180 packed bytes (vs    720)
 *   Block-diagonal: 43,200     -> 10,800 packed bytes (vs 43,200)
 *   Full-matrix:    172,800    -> 43,200 packed bytes (vs 172,800)
 *
 * Thread-safe: all functions are stateless and pure.
 *
 * ===================================================================== */

#ifndef TRINE_PACK_H
#define TRINE_PACK_H

#include <stdint.h>
#include <stddef.h>

/* Return the number of packed bytes needed for n_trits trits.
 * Always returns ceil(n_trits / 4). */
size_t trine_pack_size(size_t n_trits);

/* Pack n trits (each 0, 1, or 2) from trits[] into packed[].
 * The caller must ensure packed[] has at least trine_pack_size(n) bytes.
 * Returns the number of packed bytes written (== trine_pack_size(n)).
 * Does NOT validate input values; call trine_pack_validate() on the
 * result if validation is needed. */
size_t trine_pack_trits(const uint8_t *trits, size_t n, uint8_t *packed);

/* Unpack ceil(n_trits/4) packed bytes back into n_trits individual
 * trit bytes (each 0, 1, or 2).
 * The caller must ensure trits[] has at least n_trits bytes. */
void trine_unpack_trits(const uint8_t *packed, size_t n_trits, uint8_t *trits);

/* Validate that all trits in a packed buffer are in {0, 1, 2}.
 * Unpacks internally and checks each trit.
 * Returns 0 if all trits are valid, -1 if any trit > 2. */
int trine_pack_validate(const uint8_t *packed, size_t n_trits);

#endif /* TRINE_PACK_H */
