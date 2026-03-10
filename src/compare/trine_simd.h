/* ======================================================================
 * TRINE -- Ternary Resonance Interference Network Embedding
 * SSE2-Accelerated Ternary Comparison
 * ======================================================================
 *
 * OVERVIEW
 *   SIMD-accelerated dot product, norm, and cosine similarity for
 *   TRINE 240-dimensional ternary vectors (values in {0, 1, 2}).
 *
 *   The centered representation maps {0, 1, 2} -> {-1, 0, +1}, so
 *   the dot product of centered vectors is:
 *     sum_i (a[i]-1) * (b[i]-1)
 *
 *   SSE2 processes 16 elements per iteration by widening uint8 to
 *   int16, subtracting 1, multiplying, and accumulating via
 *   horizontal add into 32-bit accumulators.
 *
 * RUNTIME DETECTION
 *   trine_simd_available() returns 1 if SSE2 was compiled in.
 *   If __SSE2__ is not defined at compile time, all functions use
 *   scalar fallback implementations.
 *
 * BUILD
 *   cc -O2 -Wall -Wextra -Werror -msse2 -Isrc/encode -Isrc/compare \
 *      -c -o build/trine_simd.o src/compare/trine_simd.c
 *
 * ====================================================================== */

#ifndef TRINE_SIMD_H
#define TRINE_SIMD_H

#include <stdint.h>

/* --------------------------------------------------------------------
 * Runtime feature detection
 * -------------------------------------------------------------------- */

/* Returns 1 if SSE2 is available (compiled in), 0 otherwise. */
int trine_simd_available(void);

/* --------------------------------------------------------------------
 * Centered ternary dot product
 * -------------------------------------------------------------------- */

/* Compute sum_i (a[i]-1) * (b[i]-1) for ternary vectors of length len.
 * Values in a[] and b[] must be in {0, 1, 2}.
 * Uses SSE2 intrinsics when available, scalar fallback otherwise. */
int trine_simd_dot_sse2(const uint8_t *a, const uint8_t *b, int len);

/* --------------------------------------------------------------------
 * Centered ternary norm squared
 * -------------------------------------------------------------------- */

/* Compute sum_i (a[i]-1)^2 for a ternary vector of length len.
 * Values in a[] must be in {0, 1, 2}.
 * Uses SSE2 intrinsics when available, scalar fallback otherwise. */
int trine_simd_norm2_sse2(const uint8_t *a, int len);

/* --------------------------------------------------------------------
 * Centered ternary cosine similarity
 * -------------------------------------------------------------------- */

/* Compute cosine similarity of centered ternary vectors:
 *   dot(a-1, b-1) / sqrt(norm2(a-1) * norm2(b-1))
 * Returns 0.0f if either vector has zero norm.
 * Values in a[] and b[] must be in {0, 1, 2}. */
float trine_simd_cosine_sse2(const uint8_t *a, const uint8_t *b, int len);

/* --------------------------------------------------------------------
 * Self-test
 * -------------------------------------------------------------------- */

/* Run a self-test comparing SSE2 results against scalar reference.
 * Returns 0 if all checks pass, -1 on failure (message to stderr). */
int trine_simd_selftest(void);

#endif /* TRINE_SIMD_H */
