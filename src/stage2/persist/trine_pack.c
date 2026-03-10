/* =====================================================================
 * TRINE Stage-2 — 2-bit Trit Packing (implementation)
 * =====================================================================
 *
 * Packs ternary weights (0, 1, 2) at 4 trits per byte using 2-bit
 * encoding.  Provides pack, unpack, validate, and size functions.
 *
 * Build standalone test:
 *   cc -O2 -Wall -Wextra -Werror -DTRINE_PACK_TEST \
 *      -o test_pack src/stage2/persist/trine_pack.c && ./test_pack
 *
 * ===================================================================== */

#include "trine_pack.h"

#include <stdlib.h>
#include <string.h>

/* ── Size calculation ────────────────────────────────────────────── */

size_t trine_pack_size(size_t n_trits)
{
    return (n_trits + 3) / 4;
}

/* ── Pack ────────────────────────────────────────────────────────── */

size_t trine_pack_trits(const uint8_t *trits, size_t n, uint8_t *packed)
{
    size_t packed_len = trine_pack_size(n);
    size_t full_quads = n / 4;
    size_t remainder  = n % 4;
    size_t i;

    /* Pack groups of 4 trits into one byte each */
    for (i = 0; i < full_quads; i++) {
        const uint8_t *t = trits + i * 4;
        packed[i] = (uint8_t)(
            ((t[0] & 0x03)      ) |
            ((t[1] & 0x03) <<  2) |
            ((t[2] & 0x03) <<  4) |
            ((t[3] & 0x03) <<  6)
        );
    }

    /* Pack remaining 1-3 trits into the final byte (zero-padded) */
    if (remainder > 0) {
        uint8_t byte = 0;
        for (size_t r = 0; r < remainder; r++) {
            byte |= (uint8_t)((trits[full_quads * 4 + r] & 0x03) << (r * 2));
        }
        packed[full_quads] = byte;
    }

    return packed_len;
}

/* ── Unpack ──────────────────────────────────────────────────────── */

void trine_unpack_trits(const uint8_t *packed, size_t n_trits, uint8_t *trits)
{
    size_t full_quads = n_trits / 4;
    size_t remainder  = n_trits % 4;
    size_t i;

    /* Unpack groups of 4 trits from each byte */
    for (i = 0; i < full_quads; i++) {
        uint8_t byte = packed[i];
        trits[i * 4    ] = (byte      ) & 0x03;
        trits[i * 4 + 1] = (byte >>  2) & 0x03;
        trits[i * 4 + 2] = (byte >>  4) & 0x03;
        trits[i * 4 + 3] = (byte >>  6) & 0x03;
    }

    /* Unpack remaining 1-3 trits from the final byte */
    if (remainder > 0) {
        uint8_t byte = packed[full_quads];
        for (size_t r = 0; r < remainder; r++) {
            trits[full_quads * 4 + r] = (byte >> (r * 2)) & 0x03;
        }
    }
}

/* ── Validate ────────────────────────────────────────────────────── */

int trine_pack_validate(const uint8_t *packed, size_t n_trits)
{
    size_t packed_len = trine_pack_size(n_trits);
    size_t full_quads = n_trits / 4;
    size_t remainder  = n_trits % 4;

    /* Check full quads: every 2-bit field must be <= 2, so value 3
     * (binary 11) in any slot means invalid.  We check all 4 slots. */
    for (size_t i = 0; i < full_quads; i++) {
        uint8_t byte = packed[i];
        if (((byte      ) & 0x03) > 2) return -1;
        if (((byte >>  2) & 0x03) > 2) return -1;
        if (((byte >>  4) & 0x03) > 2) return -1;
        if (((byte >>  6) & 0x03) > 2) return -1;
    }

    /* Check the remainder slots in the final byte (if any) */
    if (remainder > 0) {
        uint8_t byte = packed[full_quads];
        for (size_t r = 0; r < remainder; r++) {
            if (((byte >> (r * 2)) & 0x03) > 2) return -1;
        }
    }

    (void)packed_len; /* suppress unused warning */
    return 0;
}

/* ── Standalone test ─────────────────────────────────────────────── */

#ifdef TRINE_PACK_TEST

#include <stdio.h>
#include <assert.h>

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) \
    do { printf("  %-50s ", name); } while (0)
#define PASS() \
    do { printf("PASS\n"); g_tests_passed++; } while (0)
#define FAIL(msg) \
    do { printf("FAIL: %s\n", msg); g_tests_failed++; } while (0)

/* Verify pack/unpack round-trip for a given trit array */
static void test_roundtrip(const char *label, const uint8_t *trits, size_t n)
{
    TEST(label);

    size_t packed_sz = trine_pack_size(n);
    uint8_t *packed = (uint8_t *)calloc(packed_sz, 1);
    uint8_t *unpacked = (uint8_t *)calloc(n > 0 ? n : 1, 1);

    if (!packed || !unpacked) {
        FAIL("allocation failed");
        free(packed);
        free(unpacked);
        return;
    }

    size_t written = trine_pack_trits(trits, n, packed);
    if (written != packed_sz) {
        FAIL("pack returned wrong size");
        free(packed);
        free(unpacked);
        return;
    }

    trine_unpack_trits(packed, n, unpacked);

    for (size_t i = 0; i < n; i++) {
        if (unpacked[i] != trits[i]) {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "mismatch at index %zu: got %u, expected %u",
                     i, (unsigned)unpacked[i], (unsigned)trits[i]);
            FAIL(buf);
            free(packed);
            free(unpacked);
            return;
        }
    }

    /* Validate should pass for valid trits */
    if (trine_pack_validate(packed, n) != 0) {
        FAIL("validate rejected valid packed data");
        free(packed);
        free(unpacked);
        return;
    }

    PASS();
    free(packed);
    free(unpacked);
}

/* Test pack_size calculation */
static void test_pack_size(void)
{
    TEST("trine_pack_size");

    int ok = 1;
    if (trine_pack_size(0) != 0) ok = 0;
    if (trine_pack_size(1) != 1) ok = 0;
    if (trine_pack_size(2) != 1) ok = 0;
    if (trine_pack_size(3) != 1) ok = 0;
    if (trine_pack_size(4) != 1) ok = 0;
    if (trine_pack_size(5) != 2) ok = 0;
    if (trine_pack_size(7) != 2) ok = 0;
    if (trine_pack_size(8) != 2) ok = 0;
    if (trine_pack_size(9) != 3) ok = 0;
    if (trine_pack_size(240) != 60) ok = 0;
    if (trine_pack_size(720) != 180) ok = 0;
    if (trine_pack_size(43200) != 10800) ok = 0;
    if (trine_pack_size(172800) != 43200) ok = 0;

    if (ok) PASS(); else FAIL("size mismatch");
}

/* Test empty array (n=0) */
static void test_empty(void)
{
    TEST("empty (n=0)");

    uint8_t packed[1] = {0xFF};
    size_t written = trine_pack_trits(NULL, 0, packed);
    if (written != 0) {
        FAIL("expected 0 bytes written");
        return;
    }
    PASS();
}

/* Test single trit */
static void test_single_trits(void)
{
    uint8_t t0[] = {0};
    uint8_t t1[] = {1};
    uint8_t t2[] = {2};
    test_roundtrip("single trit 0", t0, 1);
    test_roundtrip("single trit 1", t1, 1);
    test_roundtrip("single trit 2", t2, 1);
}

/* Test exact quad boundary (n=4) */
static void test_quad(void)
{
    uint8_t trits[] = {0, 1, 2, 0};
    test_roundtrip("quad boundary (n=4)", trits, 4);
}

/* Test non-quad boundaries */
static void test_remainders(void)
{
    uint8_t t5[] = {2, 1, 0, 2, 1};
    uint8_t t6[] = {0, 0, 1, 1, 2, 2};
    uint8_t t7[] = {1, 2, 0, 1, 2, 0, 1};
    test_roundtrip("remainder 1 (n=5)", t5, 5);
    test_roundtrip("remainder 2 (n=6)", t6, 6);
    test_roundtrip("remainder 3 (n=7)", t7, 7);
}

/* Test all-zeros, all-ones, all-twos */
static void test_uniform(void)
{
    uint8_t zeros[16];
    uint8_t ones[16];
    uint8_t twos[16];
    memset(zeros, 0, 16);
    memset(ones,  1, 16);
    memset(twos,  2, 16);
    test_roundtrip("all zeros (n=16)", zeros, 16);
    test_roundtrip("all ones  (n=16)", ones,  16);
    test_roundtrip("all twos  (n=16)", twos,  16);
}

/* Test bit layout explicitly */
static void test_bit_layout(void)
{
    TEST("bit layout verification");

    /* trits = {0, 1, 2, 0} should pack to:
     *   0b00_10_01_00 = 0x24 */
    uint8_t trits[] = {0, 1, 2, 0};
    uint8_t packed[1];
    trine_pack_trits(trits, 4, packed);

    /* trit[0]=0 in bits 0-1 -> 00
     * trit[1]=1 in bits 2-3 -> 01
     * trit[2]=2 in bits 4-5 -> 10
     * trit[3]=0 in bits 6-7 -> 00
     * byte = 00_10_01_00 = 0x24 */
    if (packed[0] != 0x24) {
        char buf[64];
        snprintf(buf, sizeof(buf), "expected 0x24, got 0x%02x", packed[0]);
        FAIL(buf);
        return;
    }

    /* Another: {2, 2, 2, 2} -> 0b10_10_10_10 = 0xAA */
    uint8_t all2[] = {2, 2, 2, 2};
    trine_pack_trits(all2, 4, packed);
    if (packed[0] != 0xAA) {
        char buf[64];
        snprintf(buf, sizeof(buf), "expected 0xAA, got 0x%02x", packed[0]);
        FAIL(buf);
        return;
    }

    /* {1, 1, 1, 1} -> 0b01_01_01_01 = 0x55 */
    uint8_t all1[] = {1, 1, 1, 1};
    trine_pack_trits(all1, 4, packed);
    if (packed[0] != 0x55) {
        char buf[64];
        snprintf(buf, sizeof(buf), "expected 0x55, got 0x%02x", packed[0]);
        FAIL(buf);
        return;
    }

    PASS();
}

/* Test validation: invalid trit value 3 */
static void test_validate_invalid(void)
{
    TEST("validate rejects trit value 3");

    /* Manually create a packed byte containing value 3 in slot 2:
     * byte = 0b00_11_00_00 = 0x0C (slot 2 has value 3) */
    uint8_t packed[] = {0x0C};
    if (trine_pack_validate(packed, 4) != -1) {
        FAIL("should have rejected value 3");
        return;
    }
    PASS();
}

/* Test validation: invalid value in all 4 slots */
static void test_validate_each_slot(void)
{
    TEST("validate rejects value 3 in each slot");

    /* slot 0: value 3 -> 0b00_00_00_11 = 0x03 */
    uint8_t p0[] = {0x03};
    if (trine_pack_validate(p0, 4) != -1) { FAIL("slot 0"); return; }

    /* slot 1: value 3 -> 0b00_00_11_00 = 0x0C */
    uint8_t p1[] = {0x0C};
    if (trine_pack_validate(p1, 4) != -1) { FAIL("slot 1"); return; }

    /* slot 2: value 3 -> 0b00_11_00_00 = 0x30 */
    uint8_t p2[] = {0x30};
    if (trine_pack_validate(p2, 4) != -1) { FAIL("slot 2"); return; }

    /* slot 3: value 3 -> 0b11_00_00_00 = 0xC0 */
    uint8_t p3[] = {0xC0};
    if (trine_pack_validate(p3, 4) != -1) { FAIL("slot 3"); return; }

    PASS();
}

/* Test validation: invalid value 3 in remainder slot */
static void test_validate_remainder_invalid(void)
{
    TEST("validate rejects value 3 in remainder");

    /* 5 trits: first 4 valid, 5th = value 3
     * byte 0: {0,0,0,0} -> 0x00
     * byte 1: trit[4]=3 in bits 0-1 -> 0x03 */
    uint8_t packed[] = {0x00, 0x03};
    if (trine_pack_validate(packed, 5) != -1) {
        FAIL("should have rejected remainder value 3");
        return;
    }
    PASS();
}

/* Test validation: passes for valid packed data */
static void test_validate_valid(void)
{
    TEST("validate accepts valid packed data");

    /* Pack some valid trits and verify validation passes */
    uint8_t trits[] = {0, 1, 2, 0, 1, 2, 2, 1, 0};
    uint8_t packed[3];
    trine_pack_trits(trits, 9, packed);

    if (trine_pack_validate(packed, 9) != 0) {
        FAIL("rejected valid data");
        return;
    }
    PASS();
}

/* Large round-trip test (simulating diagonal model: 720 trits) */
static void test_large_diagonal(void)
{
    size_t n = 720;
    uint8_t *trits = (uint8_t *)malloc(n);
    if (!trits) { TEST("large diagonal (720)"); FAIL("alloc"); return; }

    /* Fill with a repeating pattern */
    for (size_t i = 0; i < n; i++)
        trits[i] = (uint8_t)(i % 3);

    test_roundtrip("large diagonal (720 trits)", trits, n);

    /* Verify packed size */
    TEST("diagonal packed size == 180");
    if (trine_pack_size(n) == 180) PASS(); else FAIL("wrong size");

    free(trits);
}

/* Large round-trip test (simulating block-diagonal: 43200 trits) */
static void test_large_block_diagonal(void)
{
    size_t n = 43200;
    uint8_t *trits = (uint8_t *)malloc(n);
    if (!trits) { TEST("large block-diag (43200)"); FAIL("alloc"); return; }

    for (size_t i = 0; i < n; i++)
        trits[i] = (uint8_t)(i % 3);

    test_roundtrip("large block-diagonal (43200 trits)", trits, n);

    TEST("block-diagonal packed size == 10800");
    if (trine_pack_size(n) == 10800) PASS(); else FAIL("wrong size");

    free(trits);
}

/* Large round-trip test (simulating full matrix: 172800 trits) */
static void test_large_full_matrix(void)
{
    size_t n = 172800;
    uint8_t *trits = (uint8_t *)malloc(n);
    if (!trits) { TEST("large full-matrix (172800)"); FAIL("alloc"); return; }

    for (size_t i = 0; i < n; i++)
        trits[i] = (uint8_t)(i % 3);

    test_roundtrip("large full-matrix (172800 trits)", trits, n);

    TEST("full-matrix packed size == 43200");
    if (trine_pack_size(n) == 43200) PASS(); else FAIL("wrong size");

    free(trits);
}

/* Compression ratio verification */
static void test_compression_ratios(void)
{
    TEST("4x compression ratio");

    int ok = 1;
    /* Diagonal: 720 -> 180 = 4x */
    if (720 / trine_pack_size(720) != 4) ok = 0;
    /* Block-diagonal: 43200 -> 10800 = 4x */
    if (43200 / trine_pack_size(43200) != 4) ok = 0;
    /* Full: 172800 -> 43200 = 4x */
    if (172800 / trine_pack_size(172800) != 4) ok = 0;

    if (ok) PASS(); else FAIL("ratio not 4x");
}

int main(void)
{
    printf("trine_pack: 2-bit trit packing tests\n");
    printf("====================================\n");

    test_pack_size();
    test_empty();
    test_single_trits();
    test_quad();
    test_remainders();
    test_uniform();
    test_bit_layout();
    test_validate_invalid();
    test_validate_each_slot();
    test_validate_remainder_invalid();
    test_validate_valid();
    test_large_diagonal();
    test_large_block_diagonal();
    test_large_full_matrix();
    test_compression_ratios();

    printf("====================================\n");
    printf("Results: %d passed, %d failed\n", g_tests_passed, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}

#endif /* TRINE_PACK_TEST */
