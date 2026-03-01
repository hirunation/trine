/* ═══════════════════════════════════════════════════════════════════════
 * TRINE — Ternary Resonance Interference Network Embedding
 * Text Encoding Layer v1.0.1
 * ═══════════════════════════════════════════════════════════════════════
 *
 * OVERVIEW
 *   Lossless text-to-trit encoder for the TRINE embedding model.
 *   Maps ASCII text into 240 I/O channels (trits 0-2) across 4 parallel
 *   chains of 60 channels each. Each chain provides a different algebraic
 *   view of the same input text.
 *
 * ENCODING
 *   Code 0 = (0,0,0,0,0) is RESERVED for PAD (no character).
 *   Each ASCII character (0-127) maps to exactly 5 trits via a carefully
 *   designed lookup table using codes 1-242. 3^5 - 1 = 242 > 128, so the
 *   mapping is injective. The 12 most common English letters
 *   (e,t,a,o,i,n,s,h,r,d,l,u) are placed with minimum pairwise Hamming
 *   distance >= 3 in trit space. Upper/lowercase pairs differ by exactly
 *   1 trit (Hamming distance 1).
 *
 * CHANNEL LAYOUT (240 channels total)
 *   Chain 1 [  0.. 59]: Forward -- left-to-right character encoding
 *   Chain 2 [ 60..119]: Reverse -- right-to-left character encoding
 *   Chain 3 [120..179]: Differential -- consecutive character deltas
 *   Chain 4 [180..239]: Structural -- character class features
 *
 * CAPACITY
 *   12 characters per chain (5 trits x 12 = 60 channels).
 *   Unused positions are PAD-filled (all-zero). Text length is recovered
 *   by scanning backward for the last non-PAD position.
 *   Texts longer than 12 characters are truncated: the first 11 characters
 *   are encoded faithfully and position 11 holds an overflow hash marker.
 *
 * LOSSLESSNESS
 *   For texts of 1-12 ASCII characters (0-127), trine_decode(trine_encode(t))
 *   returns the original text exactly. This is guaranteed by:
 *     1. Injective character-to-code mapping (128 chars -> 128 unique codes)
 *     2. Code 0 reserved for PAD (unambiguous length recovery)
 *     3. Forward/reverse chain cross-validation (truncation detection)
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef TRINE_ENCODE_H
#define TRINE_ENCODE_H

#include <stdint.h>
#include <stddef.h>

#define TRINE_VERSION     "1.0.1"
#define TRINE_CHANNELS    240       /* Total I/O channels                    */
#define TRINE_CHAINS      4         /* Number of parallel encoding chains    */
#define TRINE_CHAIN_WIDTH 60        /* Channels per chain                    */
#define TRINE_TRITS_PER_CHAR 5      /* Trits per encoded character           */
#define TRINE_MAX_CHARS   12        /* Characters per chain (60/5)           */
#define TRINE_PAD_VALUE   0         /* Padding trit value (all zeros)        */

/* Chain offsets within the 240-channel array */
#define TRINE_CHAIN_FORWARD    0    /* Channels   0.. 59                     */
#define TRINE_CHAIN_REVERSE   60    /* Channels  60..119                     */
#define TRINE_CHAIN_DIFF     120    /* Channels 120..179                     */
#define TRINE_CHAIN_STRUCT   180    /* Channels 180..239                     */

/* ═══════════════════════════════════════════════════════════════════════
 * Encoding Metadata
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int      char_count;       /* Actual characters encoded (1-12)          */
    int      is_truncated;     /* 1 if text was longer than 12 chars        */
    uint32_t overflow_hash;    /* FNV-1a hash of truncated portion (or 0)   */
} trine_encode_info_t;

/* ═══════════════════════════════════════════════════════════════════════
 * API
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * trine_encode — Encode text into 240 I/O channel trit values.
 *
 * @param text     Input text (ASCII 0-127). Characters >= 128 are masked.
 * @param len      Length of input text in bytes.
 * @param channels Output array of 240 trit values (each 0, 1, or 2).
 *
 * All 240 channels are written. Unused positions are zero-padded.
 * For texts longer than 12 characters, the first 11 characters are
 * encoded normally and the 12th position contains an overflow hash.
 */
void trine_encode(const char *text, size_t len, uint8_t channels[240]);

/*
 * trine_decode — Decode 240 channels back to text.
 *
 * @param channels Input array of 240 trit values.
 * @param text     Output text buffer.
 * @param max_len  Size of output buffer (including null terminator).
 * @return         Number of characters decoded (0 on error, -1 if truncated).
 *
 * Decodes from the forward chain (channels 0-59). The encoding is
 * lossless for texts of 1-12 characters. If the text was truncated
 * during encoding, returns -1 (partial decode is still written).
 */
int trine_decode(const uint8_t channels[240], char *text, size_t max_len);

/*
 * trine_encode_info — Get metadata about an encoding operation.
 *
 * @param text Input text.
 * @param len  Length of input text.
 * @param info Output metadata structure.
 *
 * Does not perform the actual encoding; only computes metadata.
 */
void trine_encode_info(const char *text, size_t len, trine_encode_info_t *info);

/* ═══════════════════════════════════════════════════════════════════════
 * Shingle Encoding (v2.3) — Locality-Preserving, Arbitrary-Length
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Multi-scale n-gram shingling encoder. Replaces the original 12-char
 * position-based encoding with overlapping n-gram hashing at 4 scales:
 *
 *   Chain 1 [  0.. 59]: Character unigrams (K=1) + bigrams (K=3)
 *   Chain 2 [ 60..119]: Character trigrams (K=3)
 *   Chain 3 [120..179]: Character 5-grams (K=3)
 *   Chain 4 [180..239]: Word unigrams (K=3, whitespace-delimited)
 *
 * Each n-gram is hashed (seeded FNV-1a) to K independent slots in its
 * chain's 60-slot band. Values (1 or 2) accumulate via Z₃ addition
 * (mod 3), producing a deterministic fingerprint where similar texts
 * share n-grams → share channel values → produce cosine similarity
 * proportional to textual overlap.
 *
 * Input is case-folded to lowercase before hashing, so "Cat", "cat",
 * and "CAT" produce identical encodings.
 *
 * Key properties:
 *   - Arbitrary length input (no truncation, no cap)
 *   - Case-insensitive: "CAT" == "cat" == "Cat"
 *   - Locality-preserving: edit distance correlates with cosine distance
 *   - Deterministic: same input always produces same encoding
 *   - Multi-scale: character, morpheme, word-fragment, and vocabulary
 *   - ~4M embeddings/sec (0.25 µs per embedding)
 *
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * trine_encode_shingle — Multi-scale n-gram shingling encoder.
 *
 * @param text     Input text (ASCII, case-insensitive). >= 128 masked.
 * @param len      Length of input text in bytes. No upper limit.
 * @param channels Output array of 240 trit values (each 0, 1, or 2).
 *
 * All 240 channels are written. Channels not targeted by any n-gram
 * are zero-valued. Texts shorter than the minimum n-gram size for a
 * chain use a whole-text fallback hash, ensuring all non-empty inputs
 * produce at least one feature per chain.
 */
void trine_encode_shingle(const char *text, size_t len, uint8_t channels[240]);

#endif /* TRINE_ENCODE_H */
