/* ═══════════════════════════════════════════════════════════════════════
 * TRINE — Ternary Resonance Interference Network Embedding
 * Canonicalization Presets — Implementation
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Pure C99, zero dependencies, zero heap allocation.
 * All transforms operate in-place on a caller-provided buffer.
 *
 * Pattern matching uses simple state machines — no regex engine.
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#include "trine_canon.h"
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Internal Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static int is_digit(char c) { return c >= '0' && c <= '9'; }
static int is_hex(char c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
static int is_upper(char c) { return c >= 'A' && c <= 'Z'; }
static int is_lower(char c) { return c >= 'a' && c <= 'z'; }
static int is_alpha(char c) { return is_upper(c) || is_lower(c); }
static int is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}
static char to_lower(char c) { return is_upper(c) ? (char)(c + 32) : c; }

/* Check if n consecutive characters starting at buf[pos] are hex digits.
 * Returns 1 if so, 0 otherwise. Caller must ensure pos + n <= len. */
static int hex_run(const char *buf, size_t pos, size_t len, size_t n) {
    if (pos + n > len) return 0;
    for (size_t i = 0; i < n; i++)
        if (!is_hex(buf[pos + i])) return 0;
    return 1;
}

/* Check if n consecutive characters starting at buf[pos] are digits.
 * Returns 1 if so, 0 otherwise. Caller must ensure pos + n <= len. */
static int digit_run(const char *buf, size_t pos, size_t len, size_t n) {
    if (pos + n > len) return 0;
    for (size_t i = 0; i < n; i++)
        if (!is_digit(buf[pos + i])) return 0;
    return 1;
}

/* Compact: remove a span [start, start+count) from buf, shift remainder left.
 * Updates *len accordingly. */
static void compact(char *buf, size_t *len, size_t start, size_t count) {
    if (start + count > *len) count = *len - start;
    memmove(buf + start, buf + start + count, *len - start - count);
    *len -= count;
    buf[*len] = '\0';
}

/* ═══════════════════════════════════════════════════════════════════════
 * trine_canon_normalize_whitespace
 * ═══════════════════════════════════════════════════════════════════════ */

void trine_canon_normalize_whitespace(char *buf, size_t *len) {
    size_t r = 0, w = 0;
    size_t n = *len;
    int prev_ws = 1;  /* treat start as whitespace to trim leading */

    while (r < n) {
        if (is_ws(buf[r])) {
            if (!prev_ws) {
                buf[w++] = ' ';
                prev_ws = 1;
            }
            r++;
        } else {
            buf[w++] = buf[r++];
            prev_ws = 0;
        }
    }

    /* trim trailing space */
    if (w > 0 && buf[w - 1] == ' ') w--;

    buf[w] = '\0';
    *len = w;
}

/* ═══════════════════════════════════════════════════════════════════════
 * trine_canon_strip_timestamps
 *
 * Removes:
 *   - ISO-8601 dates:    YYYY-MM-DD  (10 chars: \d{4}-\d{2}-\d{2})
 *   - Times:             HH:MM:SS    (8 chars: \d{2}:\d{2}:\d{2})
 *   - Unix timestamps:   exactly 10 consecutive digits at word boundary
 *
 * Scans left-to-right, removes longest match first.
 * ═══════════════════════════════════════════════════════════════════════ */

void trine_canon_strip_timestamps(char *buf, size_t *len) {
    size_t i = 0;

    while (i < *len) {
        /* ISO-8601 date: YYYY-MM-DD (exactly \d{4}-\d{2}-\d{2}) */
        if (digit_run(buf, i, *len, 4) &&
            i + 10 <= *len &&
            buf[i + 4] == '-' &&
            digit_run(buf, i + 5, *len, 2) &&
            buf[i + 7] == '-' &&
            digit_run(buf, i + 8, *len, 2)) {

            /* Check for trailing 'T' + time (ISO-8601 combined) */
            size_t span = 10;
            if (i + 19 <= *len &&
                buf[i + 10] == 'T' &&
                digit_run(buf, i + 11, *len, 2) &&
                buf[i + 13] == ':' &&
                digit_run(buf, i + 14, *len, 2) &&
                buf[i + 16] == ':' &&
                digit_run(buf, i + 17, *len, 2)) {
                span = 19;
                /* Optional timezone Z or +HH:MM */
                if (i + span < *len && buf[i + span] == 'Z') {
                    span++;
                } else if (i + span + 5 < *len &&
                           (buf[i + span] == '+' || buf[i + span] == '-') &&
                           digit_run(buf, i + span + 1, *len, 2) &&
                           buf[i + span + 3] == ':' &&
                           digit_run(buf, i + span + 4, *len, 2)) {
                    span += 6;
                }
            }
            compact(buf, len, i, span);
            continue;
        }

        /* Time: HH:MM:SS (exactly \d{2}:\d{2}:\d{2}) */
        if (digit_run(buf, i, *len, 2) &&
            i + 8 <= *len &&
            buf[i + 2] == ':' &&
            digit_run(buf, i + 3, *len, 2) &&
            buf[i + 5] == ':' &&
            digit_run(buf, i + 6, *len, 2)) {
            /* Ensure it's not part of a larger number */
            int left_ok  = (i == 0 || !is_digit(buf[i - 1]));
            int right_ok = (i + 8 >= *len || !is_digit(buf[i + 8]));
            if (left_ok && right_ok) {
                compact(buf, len, i, 8);
                continue;
            }
        }

        /* Unix epoch: exactly 10 digits at a word boundary */
        if (is_digit(buf[i])) {
            size_t start = i;
            while (i < *len && is_digit(buf[i])) i++;
            size_t dlen = i - start;
            if (dlen == 10) {
                int left_ok  = (start == 0 || !is_alpha(buf[start - 1]));
                int right_ok = (i >= *len || !is_alpha(buf[i]));
                if (left_ok && right_ok) {
                    compact(buf, len, start, 10);
                    i = start;
                    continue;
                }
            }
            /* don't re-advance i, it's already past the digit run */
            continue;
        }

        i++;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * trine_canon_strip_uuids
 *
 * Removes UUID patterns: 8-4-4-4-12 hex digits with dashes.
 * Pattern: [0-9a-fA-F]{8}-[0-9a-fA-F]{4}-...-[0-9a-fA-F]{12}
 * Total: 36 characters (32 hex + 4 dashes).
 * ═══════════════════════════════════════════════════════════════════════ */

void trine_canon_strip_uuids(char *buf, size_t *len) {
    size_t i = 0;

    while (i + 36 <= *len) {
        if (hex_run(buf, i, *len, 8)      && buf[i +  8] == '-' &&
            hex_run(buf, i +  9, *len, 4) && buf[i + 13] == '-' &&
            hex_run(buf, i + 14, *len, 4) && buf[i + 18] == '-' &&
            hex_run(buf, i + 19, *len, 4) && buf[i + 23] == '-' &&
            hex_run(buf, i + 24, *len, 12)) {
            compact(buf, len, i, 36);
            /* don't advance i — new content shifted into position */
        } else {
            i++;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * trine_canon_normalize_identifiers
 *
 * Converts camelCase and snake_case identifiers to lowercase words.
 *
 * camelCase:  Insert space at lowercase-to-uppercase boundary.
 *             "processData" -> "process data"
 *             "HTMLParser"  -> "h t m l parser" (each uppercase before
 *              next lowercase gets split — acceptable for embedding)
 *
 * snake_case: Replace underscores between alphanumeric chars with space.
 *             "process_data" -> "process data"
 *
 * All output is lowercased.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Internal: identifier normalization with explicit buffer capacity.
 * If cap is 0 the caller did not provide a capacity — fall back to
 * the legacy 2 * len + 1 assumption.  Otherwise, the expansion is
 * hard-capped at cap - 1 characters (leaving room for the null
 * terminator).  Any camelCase insertions that would exceed the cap
 * are silently dropped so we never write past the buffer. */
static void normalize_identifiers_capped(char *buf, size_t *len,
                                          size_t cap) {
    /* Pass 1: camelCase splitting.
     * Strategy: two-pass — count insertions, then shift right-to-left.
     */
    size_t n = *len;

    /* Count camelCase boundaries */
    size_t inserts = 0;
    for (size_t i = 1; i < n; i++) {
        if (is_lower(buf[i - 1]) && is_upper(buf[i]))
            inserts++;
    }

    /* Clamp insertions so new_len never exceeds cap - 1 */
    if (inserts > 0) {
        size_t max_len = cap > 0 ? cap - 1 : 2 * n;
        if (n + inserts > max_len) {
            inserts = (max_len > n) ? max_len - n : 0;
        }
    }

    /* Expand: shift from right to left, inserting spaces.
     * We insert at most `inserts` spaces, skipping later boundaries
     * once the budget is exhausted. */
    if (inserts > 0) {
        size_t new_len = n + inserts;
        buf[new_len] = '\0';
        size_t w = new_len;
        size_t remaining = inserts;
        for (size_t i = n; i > 0; i--) {
            w--;
            buf[w] = buf[i - 1];
            if (remaining > 0 && i > 1 &&
                is_lower(buf[i - 2]) && is_upper(buf[i - 1])) {
                w--;
                buf[w] = ' ';
                remaining--;
            }
        }
        n = new_len;
    }

    /* Pass 2: snake_case — replace underscores between alnum with space */
    for (size_t i = 1; i + 1 < n; i++) {
        if (buf[i] == '_' &&
            (is_alpha(buf[i - 1]) || is_digit(buf[i - 1])) &&
            (is_alpha(buf[i + 1]) || is_digit(buf[i + 1]))) {
            buf[i] = ' ';
        }
    }

    /* Pass 3: lowercase everything */
    for (size_t i = 0; i < n; i++)
        buf[i] = to_lower(buf[i]);

    buf[n] = '\0';
    *len = n;
}

void trine_canon_normalize_identifiers(char *buf, size_t *len) {
    /* Public API: no capacity known — pass 0 to use legacy fallback */
    normalize_identifiers_capped(buf, len, 0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * trine_canon_bucket_numbers
 *
 * Replace runs of digits with the placeholder "<N>".
 * "error 404 on port 8080" -> "error <N> on port <N>"
 * ═══════════════════════════════════════════════════════════════════════ */

void trine_canon_bucket_numbers(char *buf, size_t *len) {
    size_t r = 0, w = 0;
    size_t n = *len;

    while (r < n) {
        if (is_digit(buf[r])) {
            /* skip all digits */
            while (r < n && is_digit(buf[r])) r++;
            /* write placeholder <N> */
            buf[w++] = '<';
            buf[w++] = 'N';
            buf[w++] = '>';
        } else {
            buf[w++] = buf[r++];
        }
    }

    buf[w] = '\0';
    *len = w;
}

/* ═══════════════════════════════════════════════════════════════════════
 * trine_canon_apply
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_canon_apply(const char *text, size_t len, int preset,
                      char *out, size_t out_cap, size_t *out_len) {
    /* We need at least len + 1 bytes for the copy + null terminator.
     * For CODE preset with identifier normalization, camelCase splitting
     * can expand the text. We need extra room. A safe upper bound is
     * 3 * len + 1 to handle pathological cases where many consecutive
     * uppercase letters each produce an underscore insertion. */
    /* CODE: identifier expansion can triple length.
     * SUPPORT/POLICY: bucket_numbers can expand single digits to "<N>" (3x).
     * All presets that expand text need 3x headroom. */
    size_t need;
    if (preset == TRINE_CANON_CODE || preset == TRINE_CANON_SUPPORT ||
        preset == TRINE_CANON_POLICY) {
        need = 3 * len + 1;
    } else {
        need = len + 1;
    }
    if (out_cap < need) {
        if (out_len) *out_len = 0;
        return -1;
    }

    /* Copy input to output buffer */
    memcpy(out, text, len);
    out[len] = '\0';
    size_t n = len;

    /* Apply transforms based on preset */
    switch (preset) {
    case TRINE_CANON_NONE:
        break;

    case TRINE_CANON_SUPPORT:
        trine_canon_normalize_whitespace(out, &n);
        trine_canon_strip_timestamps(out, &n);
        trine_canon_strip_uuids(out, &n);
        trine_canon_bucket_numbers(out, &n);
        /* Clean up whitespace artifacts left by removals */
        trine_canon_normalize_whitespace(out, &n);
        break;

    case TRINE_CANON_CODE:
        trine_canon_normalize_whitespace(out, &n);
        normalize_identifiers_capped(out, &n, out_cap);
        break;

    case TRINE_CANON_POLICY:
        trine_canon_normalize_whitespace(out, &n);
        trine_canon_bucket_numbers(out, &n);
        break;

    case TRINE_CANON_GENERAL:
        trine_canon_normalize_whitespace(out, &n);
        break;

    default:
        /* Unknown preset — passthrough */
        break;
    }

    if (out_len) *out_len = n;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * trine_canon_preset_name
 * ═══════════════════════════════════════════════════════════════════════ */

const char *trine_canon_preset_name(int preset) {
    switch (preset) {
    case TRINE_CANON_NONE:    return "NONE";
    case TRINE_CANON_SUPPORT: return "SUPPORT";
    case TRINE_CANON_CODE:    return "CODE";
    case TRINE_CANON_POLICY:  return "POLICY";
    case TRINE_CANON_GENERAL: return "GENERAL";
    default:                  return "UNKNOWN";
    }
}
