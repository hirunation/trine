/* ======================================================================
 * TRINE -- Ternary Resonance Interference Network Embedding
 * SSE2-Accelerated Ternary Comparison -- Implementation
 * ======================================================================
 *
 * SSE2 strategy for centered ternary vectors:
 *
 *   Each trit value is in {0, 1, 2}. Centering subtracts 1 to get
 *   {-1, 0, +1}. The dot product of two centered vectors is:
 *     sum_i (a[i]-1) * (b[i]-1)
 *
 *   SSE2 implementation processes 16 elements per iteration:
 *   1. Load 16 uint8 values from a[] and b[]
 *   2. Unpack low/high 8 bytes to int16 lanes
 *   3. Subtract the constant 1 from each int16 lane
 *   4. Multiply corresponding int16 lanes (result fits in int16
 *      since values are in {-1,0,+1} and products in {-1,0,+1})
 *   5. Use _mm_madd_epi16 to multiply-and-add adjacent pairs
 *      into int32, accumulating into a running sum register
 *   6. After the loop, horizontal-sum the 4 int32 lanes
 *
 *   Since products are in {0, 1} (squares) or {-1, 0, 1} (cross),
 *   and 240 elements max, the int32 accumulator never overflows.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -Werror -msse2 -Isrc/encode -Isrc/compare \
 *      -c -o build/trine_simd.o src/compare/trine_simd.c
 *
 * ====================================================================== */

#include "trine_simd.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

/* ======================================================================
 * I. SCALAR REFERENCE IMPLEMENTATIONS
 *
 * These are always compiled. They serve as both the fallback path
 * (when SSE2 is unavailable) and the reference for the self-test.
 * ====================================================================== */

static int scalar_dot(const uint8_t *a, const uint8_t *b, int len)
{
    int sum = 0;
    for (int i = 0; i < len; i++) {
        int ca = (int)a[i] - 1;
        int cb = (int)b[i] - 1;
        sum += ca * cb;
    }
    return sum;
}

static int scalar_norm2(const uint8_t *a, int len)
{
    int sum = 0;
    for (int i = 0; i < len; i++) {
        int ca = (int)a[i] - 1;
        sum += ca * ca;
    }
    return sum;
}

static float scalar_cosine(const uint8_t *a, const uint8_t *b, int len)
{
    int dot = scalar_dot(a, b, len);
    int na  = scalar_norm2(a, len);
    int nb  = scalar_norm2(b, len);

    if (na == 0 || nb == 0) return 0.0f;

    double denom = sqrt((double)na) * sqrt((double)nb);
    double sim   = (double)dot / denom;

    /* Clamp to [-1, 1] for floating-point safety */
    if (sim >  1.0) sim =  1.0;
    if (sim < -1.0) sim = -1.0;

    return (float)sim;
}

/* ======================================================================
 * II. SSE2 IMPLEMENTATIONS
 * ====================================================================== */

#ifdef __SSE2__

/*
 * SSE2 dot product: sum_i (a[i]-1) * (b[i]-1)
 *
 * Processes 16 uint8 elements per iteration:
 *   - Load 16 bytes from a and b
 *   - Unpack to two groups of 8 int16 values (low half, high half)
 *   - Subtract constant 1 from each int16
 *   - Use _mm_madd_epi16 to multiply pairs and sum adjacent results
 *     into int32 accumulators
 *   - Horizontal sum the 4 int32 lanes at the end
 */
static int sse2_dot(const uint8_t *a, const uint8_t *b, int len)
{
    __m128i acc = _mm_setzero_si128();
    __m128i ones_16 = _mm_set1_epi16(1);
    __m128i zero = _mm_setzero_si128();

    int i = 0;
    int n16 = len & ~15;  /* round down to multiple of 16 */

    for (; i < n16; i += 16) {
        /* Load 16 uint8 values */
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(b + i));

        /* Unpack low 8 bytes to int16 */
        __m128i a_lo = _mm_unpacklo_epi8(va, zero);
        __m128i b_lo = _mm_unpacklo_epi8(vb, zero);

        /* Unpack high 8 bytes to int16 */
        __m128i a_hi = _mm_unpackhi_epi8(va, zero);
        __m128i b_hi = _mm_unpackhi_epi8(vb, zero);

        /* Center: subtract 1 */
        a_lo = _mm_sub_epi16(a_lo, ones_16);
        b_lo = _mm_sub_epi16(b_lo, ones_16);
        a_hi = _mm_sub_epi16(a_hi, ones_16);
        b_hi = _mm_sub_epi16(b_hi, ones_16);

        /* Multiply-and-add adjacent pairs: (a*b)[0]+(a*b)[1] into int32 */
        __m128i prod_lo = _mm_madd_epi16(a_lo, b_lo);
        __m128i prod_hi = _mm_madd_epi16(a_hi, b_hi);

        /* Accumulate into int32 lanes */
        acc = _mm_add_epi32(acc, prod_lo);
        acc = _mm_add_epi32(acc, prod_hi);
    }

    /* Horizontal sum of 4 int32 lanes */
    /* acc = [s0, s1, s2, s3] */
    __m128i shuf = _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2));
    acc = _mm_add_epi32(acc, shuf);
    shuf = _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1));
    acc = _mm_add_epi32(acc, shuf);

    int result;
    /* Extract lowest int32 */
    result = _mm_cvtsi128_si32(acc);

    /* Scalar tail for remaining elements */
    for (; i < len; i++) {
        int ca = (int)a[i] - 1;
        int cb = (int)b[i] - 1;
        result += ca * cb;
    }

    return result;
}

/*
 * SSE2 norm squared: sum_i (a[i]-1)^2
 *
 * Same strategy as dot, but squares the centered value.
 */
static int sse2_norm2(const uint8_t *a, int len)
{
    __m128i acc = _mm_setzero_si128();
    __m128i ones_16 = _mm_set1_epi16(1);
    __m128i zero = _mm_setzero_si128();

    int i = 0;
    int n16 = len & ~15;

    for (; i < n16; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + i));

        __m128i a_lo = _mm_unpacklo_epi8(va, zero);
        __m128i a_hi = _mm_unpackhi_epi8(va, zero);

        a_lo = _mm_sub_epi16(a_lo, ones_16);
        a_hi = _mm_sub_epi16(a_hi, ones_16);

        /* madd with self = sum of squares */
        __m128i sq_lo = _mm_madd_epi16(a_lo, a_lo);
        __m128i sq_hi = _mm_madd_epi16(a_hi, a_hi);

        acc = _mm_add_epi32(acc, sq_lo);
        acc = _mm_add_epi32(acc, sq_hi);
    }

    /* Horizontal sum */
    __m128i shuf = _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2));
    acc = _mm_add_epi32(acc, shuf);
    shuf = _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1));
    acc = _mm_add_epi32(acc, shuf);

    int result = _mm_cvtsi128_si32(acc);

    /* Scalar tail */
    for (; i < len; i++) {
        int ca = (int)a[i] - 1;
        result += ca * ca;
    }

    return result;
}

#endif /* __SSE2__ */

/* ======================================================================
 * III. PUBLIC API
 * ====================================================================== */

int trine_simd_available(void)
{
#ifdef __SSE2__
    return 1;
#else
    return 0;
#endif
}

int trine_simd_dot_sse2(const uint8_t *a, const uint8_t *b, int len)
{
    if (!a || !b || len <= 0) return 0;

#ifdef __SSE2__
    return sse2_dot(a, b, len);
#else
    return scalar_dot(a, b, len);
#endif
}

int trine_simd_norm2_sse2(const uint8_t *a, int len)
{
    if (!a || len <= 0) return 0;

#ifdef __SSE2__
    return sse2_norm2(a, len);
#else
    return scalar_norm2(a, len);
#endif
}

float trine_simd_cosine_sse2(const uint8_t *a, const uint8_t *b, int len)
{
    if (!a || !b || len <= 0) return 0.0f;

#ifdef __SSE2__
    int dot = sse2_dot(a, b, len);
    int na  = sse2_norm2(a, len);
    int nb  = sse2_norm2(b, len);
#else
    int dot = scalar_dot(a, b, len);
    int na  = scalar_norm2(a, len);
    int nb  = scalar_norm2(b, len);
#endif

    if (na == 0 || nb == 0) return 0.0f;

    double denom = sqrt((double)na) * sqrt((double)nb);
    double sim   = (double)dot / denom;

    /* Clamp to [-1, 1] for floating-point safety */
    if (sim >  1.0) sim =  1.0;
    if (sim < -1.0) sim = -1.0;

    return (float)sim;
}

/* ======================================================================
 * IV. SELF-TEST
 *
 * Verifies SSE2 results match scalar reference for several test cases:
 *   1. Known hand-crafted vectors
 *   2. All-ones vector (centered = all zeros)
 *   3. Alternating 0/2 pattern
 *   4. TRINE-dimensioned (240) pseudo-random vectors
 *   5. Edge cases: length 1, length not multiple of 16
 * ====================================================================== */

/* Simple deterministic PRNG for test vectors (xorshift32) */
static uint32_t selftest_rng(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

int trine_simd_selftest(void)
{
    int failures = 0;

    /* ----------------------------------------------------------
     * Test 1: Hand-crafted 8-element vectors
     * a = [0, 1, 2, 0, 1, 2, 0, 1]  centered: [-1,  0, 1, -1, 0, 1, -1, 0]
     * b = [2, 1, 0, 2, 1, 0, 2, 1]  centered: [ 1,  0,-1,  1, 0,-1,  1, 0]
     * dot = (-1*1)+(0*0)+(1*-1)+(-1*1)+(0*0)+(1*-1)+(-1*1)+(0*0)
     *     = -1 + 0 + (-1) + (-1) + 0 + (-1) + (-1) + 0 = -5
     * ---------------------------------------------------------- */
    {
        uint8_t a[] = {0, 1, 2, 0, 1, 2, 0, 1};
        uint8_t b[] = {2, 1, 0, 2, 1, 0, 2, 1};
        int len = 8;

        int ref_dot = scalar_dot(a, b, len);
        int simd_dot = trine_simd_dot_sse2(a, b, len);

        if (ref_dot != -5) {
            fprintf(stderr, "SIMD selftest FAIL: test1 scalar_dot expected -5, got %d\n",
                    ref_dot);
            failures++;
        }
        if (simd_dot != ref_dot) {
            fprintf(stderr, "SIMD selftest FAIL: test1 dot mismatch (scalar=%d, simd=%d)\n",
                    ref_dot, simd_dot);
            failures++;
        }

        int ref_na = scalar_norm2(a, len);
        int simd_na = trine_simd_norm2_sse2(a, len);
        if (simd_na != ref_na) {
            fprintf(stderr, "SIMD selftest FAIL: test1 norm2(a) mismatch (scalar=%d, simd=%d)\n",
                    ref_na, simd_na);
            failures++;
        }
    }

    /* ----------------------------------------------------------
     * Test 2: All-ones (centered = all zeros)
     * dot = 0, norm = 0, cosine = 0.0
     * ---------------------------------------------------------- */
    {
        uint8_t a[240];
        memset(a, 1, 240);
        uint8_t b[240];
        memset(b, 1, 240);

        int dot = trine_simd_dot_sse2(a, b, 240);
        int norm = trine_simd_norm2_sse2(a, 240);
        float cos_val = trine_simd_cosine_sse2(a, b, 240);

        if (dot != 0) {
            fprintf(stderr, "SIMD selftest FAIL: test2 all-ones dot expected 0, got %d\n", dot);
            failures++;
        }
        if (norm != 0) {
            fprintf(stderr, "SIMD selftest FAIL: test2 all-ones norm2 expected 0, got %d\n", norm);
            failures++;
        }
        if (cos_val != 0.0f) {
            fprintf(stderr, "SIMD selftest FAIL: test2 all-ones cosine expected 0.0, got %f\n",
                    (double)cos_val);
            failures++;
        }
    }

    /* ----------------------------------------------------------
     * Test 3: Alternating 0/2 pattern (240 elements)
     * centered: [-1, 1, -1, 1, ...]
     * dot(a,a) = norm2(a) = 240  (all squares = 1)
     * cosine(a,a) = 1.0
     * ---------------------------------------------------------- */
    {
        uint8_t a[240];
        for (int i = 0; i < 240; i++) {
            a[i] = (i % 2 == 0) ? 0 : 2;
        }

        int ref_norm = scalar_norm2(a, 240);
        int simd_norm = trine_simd_norm2_sse2(a, 240);

        if (ref_norm != 240) {
            fprintf(stderr, "SIMD selftest FAIL: test3 scalar norm2 expected 240, got %d\n",
                    ref_norm);
            failures++;
        }
        if (simd_norm != ref_norm) {
            fprintf(stderr, "SIMD selftest FAIL: test3 norm2 mismatch (scalar=%d, simd=%d)\n",
                    ref_norm, simd_norm);
            failures++;
        }

        float cos_self = trine_simd_cosine_sse2(a, a, 240);
        if (fabsf(cos_self - 1.0f) > 1e-6f) {
            fprintf(stderr, "SIMD selftest FAIL: test3 self-cosine expected 1.0, got %f\n",
                    (double)cos_self);
            failures++;
        }
    }

    /* ----------------------------------------------------------
     * Test 4: Pseudo-random 240-element vectors
     * Compare SSE2 against scalar reference.
     * ---------------------------------------------------------- */
    {
        uint8_t a[240], b[240];
        uint32_t rng_state = 0xDEADBEEF;

        for (int i = 0; i < 240; i++) {
            a[i] = (uint8_t)(selftest_rng(&rng_state) % 3);
            b[i] = (uint8_t)(selftest_rng(&rng_state) % 3);
        }

        int ref_dot  = scalar_dot(a, b, 240);
        int simd_dot = trine_simd_dot_sse2(a, b, 240);
        if (simd_dot != ref_dot) {
            fprintf(stderr, "SIMD selftest FAIL: test4 dot mismatch (scalar=%d, simd=%d)\n",
                    ref_dot, simd_dot);
            failures++;
        }

        int ref_na  = scalar_norm2(a, 240);
        int simd_na = trine_simd_norm2_sse2(a, 240);
        if (simd_na != ref_na) {
            fprintf(stderr, "SIMD selftest FAIL: test4 norm2(a) mismatch (scalar=%d, simd=%d)\n",
                    ref_na, simd_na);
            failures++;
        }

        int ref_nb  = scalar_norm2(b, 240);
        int simd_nb = trine_simd_norm2_sse2(b, 240);
        if (simd_nb != ref_nb) {
            fprintf(stderr, "SIMD selftest FAIL: test4 norm2(b) mismatch (scalar=%d, simd=%d)\n",
                    ref_nb, simd_nb);
            failures++;
        }

        float ref_cos  = scalar_cosine(a, b, 240);
        float simd_cos = trine_simd_cosine_sse2(a, b, 240);
        if (fabsf(simd_cos - ref_cos) > 1e-6f) {
            fprintf(stderr,
                    "SIMD selftest FAIL: test4 cosine mismatch (scalar=%f, simd=%f)\n",
                    (double)ref_cos, (double)simd_cos);
            failures++;
        }
    }

    /* ----------------------------------------------------------
     * Test 5: Non-multiple-of-16 lengths (scalar tail coverage)
     * Test lengths: 1, 7, 15, 17, 31, 33, 100
     * ---------------------------------------------------------- */
    {
        int test_lens[] = {1, 7, 15, 17, 31, 33, 100};
        int n_tests = (int)(sizeof(test_lens) / sizeof(test_lens[0]));

        uint8_t a[240], b[240];
        uint32_t rng_state = 0xCAFEBABE;

        for (int i = 0; i < 240; i++) {
            a[i] = (uint8_t)(selftest_rng(&rng_state) % 3);
            b[i] = (uint8_t)(selftest_rng(&rng_state) % 3);
        }

        for (int t = 0; t < n_tests; t++) {
            int len = test_lens[t];

            int ref_dot  = scalar_dot(a, b, len);
            int simd_dot = trine_simd_dot_sse2(a, b, len);
            if (simd_dot != ref_dot) {
                fprintf(stderr,
                        "SIMD selftest FAIL: test5 len=%d dot mismatch "
                        "(scalar=%d, simd=%d)\n",
                        len, ref_dot, simd_dot);
                failures++;
            }

            int ref_na  = scalar_norm2(a, len);
            int simd_na = trine_simd_norm2_sse2(a, len);
            if (simd_na != ref_na) {
                fprintf(stderr,
                        "SIMD selftest FAIL: test5 len=%d norm2 mismatch "
                        "(scalar=%d, simd=%d)\n",
                        len, ref_na, simd_na);
                failures++;
            }

            float ref_cos  = scalar_cosine(a, b, len);
            float simd_cos = trine_simd_cosine_sse2(a, b, len);
            if (fabsf(simd_cos - ref_cos) > 1e-6f) {
                fprintf(stderr,
                        "SIMD selftest FAIL: test5 len=%d cosine mismatch "
                        "(scalar=%f, simd=%f)\n",
                        len, (double)ref_cos, (double)simd_cos);
                failures++;
            }
        }
    }

    /* ----------------------------------------------------------
     * Test 6: Orthogonal vectors
     * a = all 0s (centered: all -1), b = all 1s (centered: all 0s)
     * dot = 0, cosine = 0.0
     * ---------------------------------------------------------- */
    {
        uint8_t a[240], b[240];
        memset(a, 0, 240);
        memset(b, 1, 240);

        int dot = trine_simd_dot_sse2(a, b, 240);
        if (dot != 0) {
            fprintf(stderr, "SIMD selftest FAIL: test6 orthogonal dot expected 0, got %d\n", dot);
            failures++;
        }

        float cos_val = trine_simd_cosine_sse2(a, b, 240);
        if (cos_val != 0.0f) {
            fprintf(stderr, "SIMD selftest FAIL: test6 orthogonal cosine expected 0.0, got %f\n",
                    (double)cos_val);
            failures++;
        }
    }

    /* ----------------------------------------------------------
     * Test 7: Anti-parallel vectors
     * a = all 0 (centered: -1), b = all 2 (centered: +1)
     * dot = -240, cosine = -1.0
     * ---------------------------------------------------------- */
    {
        uint8_t a[240], b[240];
        memset(a, 0, 240);
        memset(b, 2, 240);

        int ref_dot = scalar_dot(a, b, 240);
        int simd_dot = trine_simd_dot_sse2(a, b, 240);

        if (ref_dot != -240) {
            fprintf(stderr, "SIMD selftest FAIL: test7 scalar_dot expected -240, got %d\n",
                    ref_dot);
            failures++;
        }
        if (simd_dot != ref_dot) {
            fprintf(stderr, "SIMD selftest FAIL: test7 dot mismatch (scalar=%d, simd=%d)\n",
                    ref_dot, simd_dot);
            failures++;
        }

        float cos_val = trine_simd_cosine_sse2(a, b, 240);
        if (fabsf(cos_val - (-1.0f)) > 1e-6f) {
            fprintf(stderr, "SIMD selftest FAIL: test7 anti-parallel cosine expected -1.0, got %f\n",
                    (double)cos_val);
            failures++;
        }
    }

    /* ----------------------------------------------------------
     * Test 8: NULL/invalid input safety
     * ---------------------------------------------------------- */
    {
        uint8_t a[16];
        memset(a, 1, 16);

        if (trine_simd_dot_sse2(NULL, a, 16) != 0) {
            fprintf(stderr, "SIMD selftest FAIL: test8 NULL a dot should return 0\n");
            failures++;
        }
        if (trine_simd_dot_sse2(a, NULL, 16) != 0) {
            fprintf(stderr, "SIMD selftest FAIL: test8 NULL b dot should return 0\n");
            failures++;
        }
        if (trine_simd_dot_sse2(a, a, 0) != 0) {
            fprintf(stderr, "SIMD selftest FAIL: test8 len=0 dot should return 0\n");
            failures++;
        }
        if (trine_simd_dot_sse2(a, a, -1) != 0) {
            fprintf(stderr, "SIMD selftest FAIL: test8 len=-1 dot should return 0\n");
            failures++;
        }
        if (trine_simd_norm2_sse2(NULL, 16) != 0) {
            fprintf(stderr, "SIMD selftest FAIL: test8 NULL norm2 should return 0\n");
            failures++;
        }
        if (trine_simd_cosine_sse2(NULL, a, 16) != 0.0f) {
            fprintf(stderr, "SIMD selftest FAIL: test8 NULL cosine should return 0.0\n");
            failures++;
        }
    }

    if (failures > 0) {
        fprintf(stderr, "SIMD selftest: %d failure(s)\n", failures);
        return -1;
    }

    return 0;
}
