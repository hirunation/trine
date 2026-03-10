/* ═══════════════════════════════════════════════════════════════════════
 * TRINE — Ternary Resonance Interference Network Embedding
 * Text Encoding Layer — Implementation
 * ═══════════════════════════════════════════════════════════════════════
 *
 * CHARACTER-TO-TRIT MAPPING DESIGN
 *
 *   Code 0 = (0,0,0,0,0) is RESERVED for PAD (no character).
 *   Each ASCII character (0-127) is assigned a unique code in {1..242},
 *   which is then expanded to 5 trits (base-3 little-endian).
 *
 *   The mapping was constructed by greedy maximal-minimum-distance
 *   placement in the 5-dimensional ternary Hamming space:
 *
 *   1. The 12 most common English letters (e,t,a,o,i,n,s,h,r,d,l,u)
 *      are placed first with minimum pairwise Hamming distance >= 3.
 *      Several pairs achieve distance 4 or 5.
 *
 *   2. Remaining lowercase letters are placed to maximize minimum
 *      distance to all previously placed codes.
 *
 *   3. Uppercase letters are placed at Hamming distance exactly 1
 *      from their lowercase counterpart (single-trit case flip).
 *
 *   4. Digits, punctuation, and control characters fill remaining
 *      positions, each maximizing distance to already-used codes.
 *
 *   The 128 codes are injective (no collisions), guaranteeing lossless
 *   encoding. Code 0 is reserved for padding. 114 of 243 codes unused.
 *
 * LENGTH DETERMINATION
 *   Length is determined by the PAD sentinel: scan backward from position
 *   11 to find the last non-PAD (non-zero code) position. All characters
 *   have non-zero codes, so padding is unambiguous.
 *
 *   For truncated texts (> 12 chars), the last position holds a non-zero
 *   overflow hash marker (hash % 242 + 1), and channels 58-59 encode a
 *   truncation flag (both set to 2,2 = impossible for normal length data).
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#include "trine_encode.h"
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * I. LOOKUP TABLES
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * TRINE_CHAR_TO_CODE: ASCII (0-127) -> 5-trit code (1-242).
 *
 * Code 0 is NEVER used for any character — it is reserved for PAD.
 * This guarantees that the all-zeros pattern (0,0,0,0,0) unambiguously
 * signals an empty/padding position, enabling length recovery.
 *
 * Layout: 16 rows of 8 values. Row i covers ASCII i*8 .. i*8+7.
 */
static const uint8_t TRINE_CHAR_TO_CODE[128] = {
    /* 0x00-0x07 */  54,  55,  56,  57,  58,  61,  62,  63,
    /* 0x08-0x0F */  65,  66,  67,  68,  69,  70,  71,  73,
    /* 0x10-0x17 */  74,  75,  77,  78,  80,  83,  85,  86,
    /* 0x18-0x1F */  87,  88,  89,  92,  93,  94,  97,  98,
    /*  !"#$%&'  */ 154, 203, 216,  30,  31,  32,  37, 209,
    /* ()*+,-./ */ 227, 228,  38,  41, 159, 223, 158,  47,
    /* 01234567 */ 113, 119, 120, 124, 126, 130, 134, 137,
    /* 89:;<=>? */ 139, 152, 192, 185,  43,  42,  45, 204,
    /* @ABCDEFG */  27,  76,   3,   7,  25,  17,   6,   2,
    /* HIJKLMNO */  10,  11,  28,   9,   8,  13,  21,   1,
    /* PQRSTUVW */  19,   5,  14,  33,  82, 109,  18,  52,
    /* XYZ[\]^_ */  90,  15,  84, 242,  49,  22,  35,  39,
    /* `abcdefg */  51, 238,  12,  16, 106,  44,  24,  29,
    /* hijklmno */  64,  20,  34,  36, 170,  40,  48,   4,
    /* pqrstuvw */  46,  59,  95,  60,  81, 190,  72,  79,
    /* xyz{|}~  */  91,  96, 102,  23,  50,  26,  53,  99
};

/*
 * TRINE_CODE_TO_CHAR: 5-trit code (0-242) -> ASCII character.
 *
 *   Code 0 (PAD):  mapped to 0xFE (sentinel, not a valid ASCII char)
 *   Valid codes:    mapped to their ASCII character (0x00-0x7F)
 *   Unused codes:   mapped to 0xFF (invalid marker)
 *
 * The forward chain uses this table for decoding.
 */
static const uint8_t TRINE_CODE_TO_CHAR[243] = {
    /* 000-007 */ 0xFE, 0x4F, 0x47, 0x42, 0x6F, 0x51, 0x46, 0x43,
    /* 008-015 */ 0x4C, 0x4B, 0x48, 0x49, 0x62, 0x4D, 0x52, 0x59,
    /* 016-023 */ 0x63, 0x45, 0x56, 0x50, 0x69, 0x4E, 0x5D, 0x7B,
    /* 024-031 */ 0x66, 0x44, 0x7D, 0x40, 0x4A, 0x67, 0x23, 0x24,
    /* 032-039 */ 0x25, 0x53, 0x6A, 0x5E, 0x6B, 0x26, 0x2A, 0x5F,
    /* 040-047 */ 0x6D, 0x2B, 0x3D, 0x3C, 0x65, 0x3E, 0x70, 0x2F,
    /* 048-055 */ 0x6E, 0x5C, 0x7C, 0x60, 0x57, 0x7E, 0x00, 0x01,
    /* 056-063 */ 0x02, 0x03, 0x04, 0x71, 0x73, 0x05, 0x06, 0x07,
    /* 064-071 */ 0x68, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
    /* 072-079 */ 0x76, 0x0F, 0x10, 0x11, 0x41, 0x12, 0x13, 0x77,
    /* 080-087 */ 0x14, 0x74, 0x54, 0x15, 0x5A, 0x16, 0x17, 0x18,
    /* 088-095 */ 0x19, 0x1A, 0x58, 0x78, 0x1B, 0x1C, 0x1D, 0x72,
    /* 096-103 */ 0x79, 0x1E, 0x1F, 0x7F, 0xFF, 0xFF, 0x7A, 0xFF,
    /* 104-111 */ 0xFF, 0xFF, 0x64, 0xFF, 0xFF, 0x55, 0xFF, 0xFF,
    /* 112-119 */ 0xFF, 0x30, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x31,
    /* 120-127 */ 0x32, 0xFF, 0xFF, 0xFF, 0x33, 0xFF, 0x34, 0xFF,
    /* 128-135 */ 0xFF, 0xFF, 0x35, 0xFF, 0xFF, 0xFF, 0x36, 0xFF,
    /* 136-143 */ 0xFF, 0x37, 0xFF, 0x38, 0xFF, 0xFF, 0xFF, 0xFF,
    /* 144-151 */ 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* 152-159 */ 0x39, 0xFF, 0x20, 0xFF, 0xFF, 0xFF, 0x2E, 0x2C,
    /* 160-167 */ 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* 168-175 */ 0xFF, 0xFF, 0x6C, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* 176-183 */ 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* 184-191 */ 0xFF, 0x3B, 0xFF, 0xFF, 0xFF, 0xFF, 0x75, 0xFF,
    /* 192-199 */ 0x3A, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* 200-207 */ 0xFF, 0xFF, 0xFF, 0x21, 0x3F, 0xFF, 0xFF, 0xFF,
    /* 208-215 */ 0xFF, 0x27, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    /* 216-223 */ 0x22, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x2D,
    /* 224-231 */ 0xFF, 0xFF, 0xFF, 0x28, 0x29, 0xFF, 0xFF, 0xFF,
    /* 232-239 */ 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x61, 0xFF,
    /* 240-242 */ 0xFF, 0xFF, 0x5B
};

/*
 * Character classification tables for the structural view (Chain 4).
 *
 * Each character is classified along 3 static axes:
 *   - Category:  0=letter, 1=digit, 2=other
 *   - Subclass:  context-dependent (see below)
 *   - Case:      0=lower, 1=upper, 2=non-letter
 *
 * The remaining 2 trits (position-in-word, frequency class) are computed
 * dynamically during encoding.
 */

/* Trit 0: character category */
static const uint8_t TRINE_CATEGORY[128] = {
    /* 0x00-0x0F: control */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x10-0x1F: control */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x20-0x2F:  !"#../ */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x30-0x3F: 0-9:..? */ 1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,
    /* 0x40-0x4F: @A-O    */ 2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0x50-0x5F: P-Z[\]^ */ 0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,
    /* 0x60-0x6F: `a-o    */ 2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0x70-0x7F: p-z{|}~ */ 0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2
};

/*
 * Trit 1: subclass.
 *   Letters: 0=vowel (a,e,i,o,u), 1=consonant, 2=ambiguous (y)
 *   Digits:  value mod 3
 *   Other:   0=space, 1=punctuation, 2=control
 */
static const uint8_t TRINE_SUBCLASS[128] = {
    /* 0x00-0x0F */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x10-0x1F */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x20-0x2F */ 0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    /* 0x30-0x3F */ 0,1,2,0,1,2,0,1,2,0,1,1,1,1,1,1,
    /* 0x40-0x4F */ 1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,
    /* 0x50-0x5F */ 1,1,1,1,1,0,1,1,1,2,1,1,1,1,1,1,
    /* 0x60-0x6F */ 1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,
    /* 0x70-0x7F */ 1,1,1,1,1,0,1,1,1,2,1,1,1,1,1,2
};

/* Trit 2: case */
static const uint8_t TRINE_CASE[128] = {
    /* 0x00-0x0F */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x10-0x1F */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x20-0x2F */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x30-0x3F */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x40-0x4F */ 2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    /* 0x50-0x5F */ 1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,
    /* 0x60-0x6F */ 2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    /* 0x70-0x7F */ 0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2
};

/*
 * Frequency class for letters.
 *   0 = common  (e,t,a,o,i,n,s,h,r,d,l,u -- the top 12)
 *   1 = medium  (c,m,f,g,p,w,y,b)
 *   2 = rare    (v,k,j,x,q,z)
 *   Non-letters get class 2.
 */
static const uint8_t TRINE_FREQ_CLASS[128] = {
    /* 0x00-0x0F */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x10-0x1F */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x20-0x2F */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* 0x30-0x3F */ 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    /* @ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_ */
    2,
    0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 1, 0, 0,
    1, 2, 0, 0, 0, 0, 2, 1, 2, 1, 2,
    2, 2, 2, 2, 2,
    /* `abcdefghijklmnopqrstuvwxyz{|}~DEL */
    2,
    0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 1, 0, 0,
    1, 2, 0, 0, 0, 0, 2, 1, 2, 1, 2,
    2, 2, 2, 2, 2
};

/* ═══════════════════════════════════════════════════════════════════════
 * II. INTERNAL HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Expand a 5-trit code (0-242) into 5 individual trit values.
 * Base-3 little-endian: trit[0] is the least significant.
 */
static void code_to_trits(uint8_t code, uint8_t trits[5])
{
    trits[0] = code % 3;  code /= 3;
    trits[1] = code % 3;  code /= 3;
    trits[2] = code % 3;  code /= 3;
    trits[3] = code % 3;  code /= 3;
    trits[4] = code % 3;
}

/*
 * Compact 5 individual trit values into a code (0-242).
 */
static uint8_t trits_to_code(const uint8_t trits[5])
{
    return (uint8_t)(trits[0]
                   + trits[1] * 3
                   + trits[2] * 9
                   + trits[3] * 27
                   + trits[4] * 81);
}

/*
 * FNV-1a hash (32-bit) for overflow detection.
 * Matches the OICOS convention (oicos_fnv1a in oicos.h, 32-bit variant).
 */
static uint32_t trine_fnv1a_32(const char *data, size_t len)
{
    uint32_t h = 0x811c9dc5u;
    for (size_t i = 0; i < len; i++)
        h = (h ^ (uint8_t)data[i]) * 0x01000193u;
    return h;
}

/*
 * Sanitize a character to the ASCII range 0-127.
 * Characters with bit 7 set are masked to 7 bits.
 */
static uint8_t sanitize_char(char c)
{
    return (uint8_t)c & 0x7Fu;
}

/*
 * Determine if a character is a word boundary (space, punctuation, control).
 */
static int is_word_boundary(uint8_t c)
{
    if (c <= 0x20) return 1;                   /* control + space       */
    if (c >= 0x7F) return 1;                   /* DEL                   */
    if (c >= '!' && c <= '/') return 1;        /* punctuation block 1   */
    if (c >= ':' && c <= '@') return 1;        /* punctuation block 2   */
    if (c >= '[' && c <= '`') return 1;        /* punctuation block 3   */
    if (c >= '{' && c <= '~') return 1;        /* punctuation block 4   */
    return 0;
}

/*
 * Write a 5-trit code into a chain at position pos (channels pos*5..pos*5+4).
 * Writes all 5 trits into the channel array ch (which has 60 entries).
 */
static void write_code_at(uint8_t *ch, int pos, uint8_t code)
{
    uint8_t trits[5];
    code_to_trits(code, trits);
    int base = pos * TRINE_TRITS_PER_CHAR;
    for (int t = 0; t < TRINE_TRITS_PER_CHAR; t++)
        ch[base + t] = trits[t];
}

/*
 * Read a 5-trit code from a chain at position pos.
 */
static uint8_t read_code_at(const uint8_t *ch, int pos)
{
    int base = pos * TRINE_TRITS_PER_CHAR;
    uint8_t trits[5];
    for (int t = 0; t < 5; t++)
        trits[t] = ch[base + t];
    return trits_to_code(trits);
}

/* ═══════════════════════════════════════════════════════════════════════
 * III. CHAIN ENCODERS
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Encode the forward view (Chain 1): left-to-right character encoding.
 *
 * Characters are encoded sequentially into 12 positions of 5 trits each.
 * Position i occupies channels [i*5 .. i*5+4], covering channels 0-59.
 * Unused positions are zero-padded (code 0 = PAD).
 *
 * Since all character codes are non-zero and PAD is zero, the text length
 * is recoverable by scanning backward from position 11 for the last
 * non-zero code.
 *
 * For truncated texts, the last position holds a non-zero overflow hash
 * marker (hash % 242 + 1), ensuring it is distinguishable from PAD.
 */
static void encode_forward(const uint8_t *chars, int n_chars,
                           int is_truncated, uint32_t overflow_hash,
                           uint8_t *ch)
{
    /* Zero the entire chain (PAD fill) */
    memset(ch, 0, TRINE_CHAIN_WIDTH);

    int encode_count = n_chars;
    if (encode_count > TRINE_MAX_CHARS)
        encode_count = TRINE_MAX_CHARS;

    /* If truncated, encode first 11 chars normally, position 11 gets hash */
    int normal_count = encode_count;
    if (is_truncated && encode_count == TRINE_MAX_CHARS)
        normal_count = TRINE_MAX_CHARS - 1;

    /* Encode characters into positions 0..normal_count-1 */
    for (int i = 0; i < normal_count; i++)
        write_code_at(ch, i, TRINE_CHAR_TO_CODE[chars[i]]);

    /* If truncated, write overflow hash into the last position */
    if (is_truncated && encode_count == TRINE_MAX_CHARS) {
        /* Map hash to non-zero code: (hash % 242) + 1 gives range [1..242] */
        uint8_t hash_code = (uint8_t)((overflow_hash % 242) + 1);
        write_code_at(ch, TRINE_MAX_CHARS - 1, hash_code);
    }
}

/*
 * Encode the reverse view (Chain 2): right-to-left character encoding.
 *
 * The same characters but in reverse order. Position 0 gets the LAST
 * character, position N-1 gets the FIRST character. Remaining positions
 * are PAD-filled (zero).
 */
static void encode_reverse(const uint8_t *chars, int n_chars,
                           int is_truncated, uint32_t overflow_hash,
                           uint8_t *ch)
{
    memset(ch, 0, TRINE_CHAIN_WIDTH);

    int encode_count = n_chars;
    if (encode_count > TRINE_MAX_CHARS)
        encode_count = TRINE_MAX_CHARS;

    int normal_count = encode_count;
    if (is_truncated && encode_count == TRINE_MAX_CHARS)
        normal_count = TRINE_MAX_CHARS - 1;

    /* Write characters in reverse: slot i gets char[normal_count-1-i] */
    for (int i = 0; i < normal_count; i++) {
        int src_idx = normal_count - 1 - i;
        write_code_at(ch, i, TRINE_CHAR_TO_CODE[chars[src_idx]]);
    }

    /* Overflow hash in the position right after the reversed chars */
    if (is_truncated && encode_count == TRINE_MAX_CHARS) {
        uint8_t hash_code = (uint8_t)((overflow_hash % 242) + 1);
        write_code_at(ch, TRINE_MAX_CHARS - 1, hash_code);
    }
}

/*
 * Encode the differential view (Chain 3): consecutive character deltas.
 *
 * Position 0: raw encoding of char[0] (its code).
 * Position i (i > 0): (code[i] - code[i-1] + 243) mod 243, expanded to 5 trits.
 *
 * This captures the local "texture" of the text -- transitions between
 * characters. Similar word patterns produce similar differential encodings
 * regardless of absolute character values.
 *
 * The differential encoding is invertible: given position 0 (raw code)
 * and all deltas, the original codes can be reconstructed exactly.
 */
static void encode_differential(const uint8_t *chars, int n_chars,
                                int is_truncated, uint32_t overflow_hash,
                                uint8_t *ch)
{
    memset(ch, 0, TRINE_CHAIN_WIDTH);

    int encode_count = n_chars;
    if (encode_count > TRINE_MAX_CHARS)
        encode_count = TRINE_MAX_CHARS;

    int normal_count = encode_count;
    if (is_truncated && encode_count == TRINE_MAX_CHARS)
        normal_count = TRINE_MAX_CHARS - 1;

    uint8_t prev_code = 0;

    for (int i = 0; i < normal_count; i++) {
        uint8_t code = TRINE_CHAR_TO_CODE[chars[i]];
        uint8_t diff_code;

        if (i == 0) {
            diff_code = code;  /* First character: encode raw */
        } else {
            diff_code = (uint8_t)((code - prev_code + 243) % 243);
        }

        prev_code = code;
        write_code_at(ch, i, diff_code);
    }

    /* Overflow hash */
    if (is_truncated && encode_count == TRINE_MAX_CHARS) {
        uint8_t hash_code = (uint8_t)((overflow_hash % 242) + 1);
        write_code_at(ch, TRINE_MAX_CHARS - 1, hash_code);
    }
}

/*
 * Encode the structural view (Chain 4): character class features.
 *
 * Each character position encodes 5 classification trits:
 *   Trit 0: category       (0=letter, 1=digit, 2=other)
 *   Trit 1: subclass        (letters: 0=vowel, 1=consonant, 2=y/ambiguous;
 *                            digits: value mod 3;
 *                            other: 0=space, 1=punct, 2=control)
 *   Trit 2: case            (0=lower, 1=upper, 2=non-letter)
 *   Trit 3: position-in-word (0=start, 1=middle, 2=end/standalone)
 *   Trit 4: frequency class  (0=common, 1=medium, 2=rare)
 *
 * Position-in-word is computed dynamically from local context.
 */
static void encode_structural(const uint8_t *chars, int n_chars,
                              int is_truncated, uint32_t overflow_hash,
                              uint8_t *ch)
{
    memset(ch, 0, TRINE_CHAIN_WIDTH);

    int encode_count = n_chars;
    if (encode_count > TRINE_MAX_CHARS)
        encode_count = TRINE_MAX_CHARS;

    int normal_count = encode_count;
    if (is_truncated && encode_count == TRINE_MAX_CHARS)
        normal_count = TRINE_MAX_CHARS - 1;

    for (int i = 0; i < normal_count; i++) {
        uint8_t c = chars[i];
        int base = i * TRINE_TRITS_PER_CHAR;

        /* Trit 0: category */
        ch[base + 0] = TRINE_CATEGORY[c];

        /* Trit 1: subclass */
        ch[base + 1] = TRINE_SUBCLASS[c];

        /* Trit 2: case */
        ch[base + 2] = TRINE_CASE[c];

        /* Trit 3: position-in-word */
        uint8_t pos;
        int cur_is_boundary = is_word_boundary(c);

        if (cur_is_boundary) {
            pos = 2;  /* Non-word characters: standalone */
        } else {
            int prev_boundary = (i == 0) ? 1 : is_word_boundary(chars[i - 1]);
            int next_boundary = (i >= normal_count - 1) ? 1 : is_word_boundary(chars[i + 1]);

            if (prev_boundary)
                pos = 0;  /* Word start (or single-char word) */
            else if (next_boundary)
                pos = 2;  /* Word end */
            else
                pos = 1;  /* Word middle */
        }
        ch[base + 3] = pos;

        /* Trit 4: frequency class */
        ch[base + 4] = TRINE_FREQ_CLASS[c];
    }

    /* Overflow hash in last position if truncated */
    if (is_truncated && encode_count == TRINE_MAX_CHARS) {
        uint8_t hash_code = (uint8_t)((overflow_hash % 242) + 1);
        write_code_at(ch, TRINE_MAX_CHARS - 1, hash_code);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * IV. PUBLIC API
 * ═══════════════════════════════════════════════════════════════════════ */

void trine_encode(const char *text, size_t len, uint8_t channels[240])
{
    /* Zero all channels */
    memset(channels, 0, TRINE_CHANNELS);

    if (len == 0 || text == NULL)
        return;

    /* Sanitize input: mask to 7-bit ASCII */
    uint8_t chars[TRINE_MAX_CHARS];
    int n_chars = (int)len;
    int encode_count = n_chars;
    if (encode_count > TRINE_MAX_CHARS)
        encode_count = TRINE_MAX_CHARS;

    for (int i = 0; i < encode_count; i++)
        chars[i] = sanitize_char(text[i]);

    /* Truncation detection */
    int is_truncated = (n_chars > TRINE_MAX_CHARS) ? 1 : 0;
    uint32_t overflow_hash = 0;

    if (is_truncated) {
        /* Hash the overflow portion (characters from position 11 onward) */
        overflow_hash = trine_fnv1a_32(text + (TRINE_MAX_CHARS - 1),
                                       len - (TRINE_MAX_CHARS - 1));
    }

    /* Encode all 4 chains */
    encode_forward(chars, n_chars, is_truncated, overflow_hash,
                   channels + TRINE_CHAIN_FORWARD);

    encode_reverse(chars, encode_count, is_truncated, overflow_hash,
                   channels + TRINE_CHAIN_REVERSE);

    encode_differential(chars, n_chars, is_truncated, overflow_hash,
                        channels + TRINE_CHAIN_DIFF);

    encode_structural(chars, n_chars, is_truncated, overflow_hash,
                      channels + TRINE_CHAIN_STRUCT);
}

int trine_decode(const uint8_t channels[240], char *text, size_t max_len)
{
    if (text == NULL || max_len == 0)
        return 0;

    const uint8_t *ch = channels + TRINE_CHAIN_FORWARD;

    /* Determine text length by scanning backward for last non-PAD position.
     * PAD = code 0 = all-zeros in a 5-trit slot. All character codes are
     * non-zero, so the first non-zero code from the end marks the last char.
     */
    int last_pos = -1;
    for (int i = TRINE_MAX_CHARS - 1; i >= 0; i--) {
        uint8_t code = read_code_at(ch, i);
        if (code != 0) {
            last_pos = i;
            break;
        }
    }

    if (last_pos < 0) {
        /* All positions are PAD: empty input */
        text[0] = '\0';
        return 0;
    }

    int n_chars = last_pos + 1;

    /* Check for truncation: a truncated encoding has a code in the last
     * position that does NOT map to any valid ASCII character (or maps
     * to a character inconsistent with what the differential chain says).
     *
     * Simpler heuristic: check if the code at the last occupied position
     * maps to a valid character in the reverse table. If it maps to 0xFF
     * (unused code), it's a hash marker and the text was truncated.
     *
     * However, the hash marker CAN collide with a valid character code
     * (since hash % 242 + 1 can produce any code 1-242, including valid
     * character codes). So we need another signal.
     *
     * Solution: use the STRUCTURAL chain as a consistency check. For
     * truncated texts, the structural chain's last position also contains
     * a hash marker, which will NOT match the expected structural features
     * of the character at that position in the forward chain.
     *
     * But for a simpler, reliable approach: check if ALL 4 chains agree
     * on the last position. For non-truncated texts, the forward and
     * reverse chains will be consistent. For truncated texts with < 12
     * characters, there's no truncation issue. For truncated texts with
     * >= 13 original characters, all 12 positions are occupied.
     *
     * PRACTICAL APPROACH: We use the reverse chain for cross-validation.
     * In a non-truncated 12-char encoding:
     *   forward[11] == reverse[0]
     * In a truncated encoding:
     *   forward[11] = hash_code, reverse[11] = hash_code
     *   forward[10] == reverse[0] (the 11th char is at both positions)
     *
     * Actually, the cleanest approach: if n_chars == 12, check if
     * forward[11] == reverse[0]. If they match, it's a valid 12-char
     * text. If they don't match, it's truncated.
     */
    int is_truncated = 0;
    if (n_chars == TRINE_MAX_CHARS) {
        const uint8_t *rev = channels + TRINE_CHAIN_REVERSE;
        uint8_t fwd_last = read_code_at(ch, TRINE_MAX_CHARS - 1);
        uint8_t rev_first = read_code_at(rev, 0);
        if (fwd_last != rev_first) {
            /* Inconsistency: the last forward position doesn't match
             * what the reverse chain says should be the last character.
             * This means the last position is a hash marker. */
            is_truncated = 1;
        }
    }

    int decode_count = n_chars;
    if (is_truncated)
        decode_count = n_chars - 1;  /* Last position is hash, not a char */

    if ((size_t)decode_count >= max_len)
        decode_count = (int)(max_len - 1);

    /* Decode each character from the forward chain */
    for (int i = 0; i < decode_count; i++) {
        uint8_t code = read_code_at(ch, i);

        if (code > 0 && code < 243 && TRINE_CODE_TO_CHAR[code] != 0xFF
                                   && TRINE_CODE_TO_CHAR[code] != 0xFE) {
            text[i] = (char)TRINE_CODE_TO_CHAR[code];
        } else {
            /* Invalid or PAD code -- should not happen in valid encoding */
            text[i] = '?';
        }
    }

    text[decode_count] = '\0';

    return is_truncated ? -1 : decode_count;
}

void trine_encode_info(const char *text, size_t len, trine_encode_info_t *info)
{
    if (info == NULL)
        return;

    info->char_count = (int)len;
    if (info->char_count > TRINE_MAX_CHARS)
        info->char_count = TRINE_MAX_CHARS;

    info->is_truncated = (len > (size_t)TRINE_MAX_CHARS) ? 1 : 0;

    if (info->is_truncated) {
        info->overflow_hash = trine_fnv1a_32(
            text + (TRINE_MAX_CHARS - 1),
            len - (TRINE_MAX_CHARS - 1));
    } else {
        info->overflow_hash = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * V. MULTI-SCALE N-GRAM SHINGLE ENCODER (v2.0)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Maps arbitrary-length text to 240 I/O channels using overlapping
 * n-gram hashing across 4 independent scales:
 *
 *   Chain 1 [  0.. 59]: Char unigrams + bigrams — character-level presence & sequence
 *   Chain 2 [ 60..119]: Character trigrams      — morphological structure
 *   Chain 3 [120..179]: Character 5-grams       — word-length fragments
 *   Chain 4 [180..239]: Word unigrams           — vocabulary fingerprint
 *
 * Each n-gram hashes (seeded FNV-1a) to a slot and a value.
 * Values accumulate via Z₃ addition: channel = (channel + value) % 3.
 * Z₃ addition is commutative and associative, so the encoding is
 * independent of n-gram processing order within each slot.
 *
 * Locality guarantee: texts sharing k of n total n-grams will share
 * approximately k/n of their channel values (modulo hash collisions),
 * producing cosine similarity proportional to textual overlap.
 *
 * ═══════════════════════════════════════════════════════════════════════ */

/* Chain-specific seeds for FNV-1a (independent slot distributions).
 * Each chain (and sub-feature type) uses a distinct seed to ensure
 * independent hash function families. The actual values are arbitrary
 * constants; only distinctness matters. */
#define SHINGLE_SEED_CH1_UNI   0x54523035u  /* Chain 1: character unigrams       */
#define SHINGLE_SEED_CH1_BI    0x54523031u  /* Chain 1: character bigrams        */
#define SHINGLE_SEED_CH2_TRI   0x54523032u  /* Chain 2: character trigrams       */
#define SHINGLE_SEED_CH3_5G    0x54523033u  /* Chain 3: character 5-grams        */
#define SHINGLE_SEED_CH4_WORD  0x54523034u  /* Chain 4: word unigrams            */

/*
 * Seeded FNV-1a hash (32-bit).
 * The seed is XOR-folded into the standard FNV offset basis,
 * producing independent hash families for each chain.
 * Input bytes are masked to 7-bit ASCII (consistent with TRINE convention).
 */
static uint32_t fnv1a_seeded(const char *data, size_t len, uint32_t seed)
{
    uint32_t h = 0x811c9dc5u ^ seed;
    for (size_t i = 0; i < len; i++)
        h = (h ^ ((uint8_t)data[i] & 0x7Fu)) * 0x01000193u;
    return h;
}

/*
 * Hash an n-gram into a chain's channel band using K independent
 * hash functions.
 *
 * For each of K sub-hashes, extracts a slot (0-59) and a value
 * (1 or 2), then accumulates via Z₃ addition. Multiple hash
 * functions per n-gram increase channel fill rate, which is critical
 * for short texts where few n-grams would otherwise leave the
 * encoding too sparse for meaningful cosine similarity.
 *
 * The sub-seed progression uses the golden ratio constant (0x9E3779B9)
 * to ensure each hash function produces an independent slot distribution.
 *
 * K=3 targets ~25% fill for single-word inputs, ~60% for sentences.
 */
#define SHINGLE_K  3   /* Independent hashes per n-gram */

static void hash_shingle(const char *data, size_t ngram_len,
                          uint32_t seed, uint8_t *chain)
{
    for (int k = 0; k < SHINGLE_K; k++) {
        uint32_t h     = fnv1a_seeded(data, ngram_len,
                                       seed + (uint32_t)k * 0x9E3779B9u);
        uint32_t slot  = h % 60;
        uint32_t value = ((h >> 16) % 2) + 1;   /* 1 or 2, never 0 */
        chain[slot] = (uint8_t)((chain[slot] + value) % 3);
    }
}

/*
 * Light hash: K=1 variant for character unigrams.
 * A single hash per feature reduces false positive rate for
 * individual characters while still providing presence signal.
 * This balances the unigram contribution relative to the K=3
 * bigram/trigram features.
 */
static void hash_shingle_lite(const char *data, size_t ngram_len,
                               uint32_t seed, uint8_t *chain)
{
    uint32_t h     = fnv1a_seeded(data, ngram_len, seed);
    uint32_t slot  = h % 60;
    uint32_t value = ((h >> 16) % 2) + 1;
    chain[slot] = (uint8_t)((chain[slot] + value) % 3);
}

/*
 * Test if a byte is whitespace (word boundary for Chain 4).
 */
static int shingle_is_ws(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/*
 * Internal: encode shingle into channels using a caller-supplied casefold
 * buffer.  If ext_buf is non-NULL and ext_buf_cap >= len, it is used for
 * the case-folded copy (no allocation).  Otherwise, the function falls
 * back to a stack buffer or malloc, exactly as the public API does.
 */
static int encode_shingle_inner(const char *text, size_t len,
                                 uint8_t channels[240],
                                 char *ext_buf, size_t ext_buf_cap)
{
    memset(channels, 0, TRINE_CHANNELS);

    if (!text || len == 0)
        return 0;

    /* Case-fold to lowercase for case-insensitive similarity.
     * Without this, "Cat" and "cat" share only 2/3 character features
     * and "CAT" vs "cat" shares zero. Case-folding ensures "The" and
     * "the" hash to identical slots, matching user expectation.
     * Stack-allocated buffer; falls back to heap for very long text. */
    char stack_buf[512];
    char *folded;
    int need_free = 0;

    if (ext_buf && ext_buf_cap >= len) {
        folded = ext_buf;
    } else if (len <= sizeof(stack_buf)) {
        folded = stack_buf;
    } else {
        folded = (char *)malloc(len);
        if (!folded) return -1;  /* OOM: channels already zeroed */
        need_free = 1;
    }
    for (size_t i = 0; i < len; i++) {
        char c = text[i] & 0x7F;  /* Mask to 7-bit ASCII */
        folded[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
    }
    text = folded;  /* Use folded text for all hashing below */

    uint8_t *ch1 = channels + TRINE_CHAIN_FORWARD;  /*   0.. 59: bigrams  */
    uint8_t *ch2 = channels + TRINE_CHAIN_REVERSE;   /*  60..119: trigrams */
    uint8_t *ch3 = channels + TRINE_CHAIN_DIFF;      /* 120..179: 5-grams */
    uint8_t *ch4 = channels + TRINE_CHAIN_STRUCT;    /* 180..239: words    */

    /* ─── Chain 1: Character unigrams + bigrams (1+2-grams) ───
     * Two complementary feature types in one chain:
     *
     * Unigrams: Each character hashes independently. Ensures that
     *   words sharing ANY character produce non-zero similarity.
     *   Fixes the "middle-char edit" blind spot where changing the
     *   middle of a 3-letter word ("run"→"ran") destroys both bigrams
     *   but preserves 2/3 of the character set.
     *
     * Bigrams: Consecutive character pairs capture sequential structure.
     *   "cat"→{ca,at}, "bat"→{ba,at}. A single-char substitution
     *   changes at most 2 bigrams out of (len-1).
     *
     * Using different seeds ensures unigrams and bigrams hash to
     * independent slot distributions within the same 60-slot band.
     * Z₃ accumulation handles any slot collisions gracefully. */

    /* Character unigrams (K=1 lite hash: presence only, low noise) */
    for (size_t i = 0; i < len; i++)
        hash_shingle_lite(text + i, 1, SHINGLE_SEED_CH1_UNI, ch1);

    /* Character bigrams */
    if (len >= 2) {
        for (size_t i = 0; i + 1 < len; i++)
            hash_shingle(text + i, 2, SHINGLE_SEED_CH1_BI, ch1);
    }

    /* ─── Chain 2: Character trigrams (3-grams) ───
     * Captures morphological structure: prefixes, suffixes, stems.
     * Standard n-gram size for text similarity in NLP.
     * "hello" → {hel, ell, llo}. */
    if (len >= 3) {
        for (size_t i = 0; i + 2 < len; i++)
            hash_shingle(text + i, 3, SHINGLE_SEED_CH2_TRI, ch2);
    } else {
        hash_shingle(text, len, SHINGLE_SEED_CH2_TRI, ch2);
    }

    /* ─── Chain 3: Character 5-grams ───
     * Captures word-length fragments. English mean word length ~4.7,
     * so 5-grams often span complete short words.
     * "hello world" → {hello, ello_, llo_w, lo_wo, o_wor, _worl, world} */
    if (len >= 5) {
        for (size_t i = 0; i + 4 < len; i++)
            hash_shingle(text + i, 5, SHINGLE_SEED_CH3_5G, ch3);
    } else {
        hash_shingle(text, len, SHINGLE_SEED_CH3_5G, ch3);
    }

    /* ─── Chain 4: Word unigrams ───
     * Whitespace-delimited words, order-independent within each slot.
     * "the quick fox" and "the lazy fox" share 2/3 word features. */
    {
        size_t word_start = 0;
        int in_word = 0;
        int word_count = 0;

        for (size_t i = 0; i <= len; i++) {
            int boundary = (i == len) || shingle_is_ws(text[i]);
            if (boundary && in_word) {
                size_t wlen = i - word_start;
                hash_shingle(text + word_start, wlen, SHINGLE_SEED_CH4_WORD, ch4);
                in_word = 0;
                word_count++;
            } else if (!boundary && !in_word) {
                word_start = i;
                in_word = 1;
            }
        }

        /* If no words found (all whitespace), hash raw text */
        if (word_count == 0)
            hash_shingle(text, len, SHINGLE_SEED_CH4_WORD, ch4);
    }

    /* Free heap buffer if allocated internally for long text */
    if (need_free)
        free(folded);

    return 0;
}

int trine_encode_shingle(const char *text, size_t len, uint8_t channels[240])
{
    return encode_shingle_inner(text, len, channels, NULL, 0);
}

int trine_encode_shingle_batch(
    const char *const *texts,
    const size_t *lens,
    size_t n,
    uint8_t *out)
{
    if (!out) return -1;
    if (n == 0) return 0;
    if (!texts || !lens) return -1;

    /* Find the maximum text length so we can allocate one reusable
     * casefold buffer that covers every input. */
    size_t max_len = 0;
    for (size_t i = 0; i < n; i++) {
        if (lens[i] > max_len) max_len = lens[i];
    }

    /* Allocate a single shared casefold buffer.  The inner function's
     * 512-byte stack buffer handles short texts, but for batches with
     * any text > 512 bytes this avoids per-call malloc/free. */
    char *shared_buf = NULL;
    size_t shared_cap = 0;
    if (max_len > 512) {
        shared_buf = (char *)malloc(max_len);
        if (!shared_buf) return -1;
        shared_cap = max_len;
    }

    for (size_t i = 0; i < n; i++) {
        int rc = encode_shingle_inner(texts[i], lens[i],
                                       out + i * TRINE_CHANNELS,
                                       shared_buf, shared_cap);
        if (rc != 0) {
            free(shared_buf);
            return -1;
        }
    }

    free(shared_buf);
    return 0;
}
