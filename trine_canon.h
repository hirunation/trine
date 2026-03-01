/* ═══════════════════════════════════════════════════════════════════════
 * TRINE — Ternary Resonance Interference Network Embedding
 * Canonicalization Presets v1.0.1
 * ═══════════════════════════════════════════════════════════════════════
 *
 * OVERVIEW
 *   Deterministic text transforms applied BEFORE encoding to improve
 *   near-duplicate detection on real-world corpora (logs, tickets,
 *   legal docs, code). All transforms are regex-free, heap-free, and
 *   operate in a caller-provided output buffer.
 *
 * PRESETS
 *   NONE    — no transforms (passthrough)
 *   SUPPORT — whitespace + timestamps + uuids + bucket_numbers
 *   CODE    — whitespace + identifiers
 *   POLICY  — whitespace + bucket_numbers
 *   GENERAL — whitespace only
 *
 * GUARANTEES
 *   - Pure C99, no external dependencies, no malloc
 *   - Deterministic: same input always produces same output
 *   - Safe: bounded writes, null-terminated output
 *   - Output length <= input length (transforms only shrink or preserve)
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef TRINE_CANON_H
#define TRINE_CANON_H

#include <stddef.h>

#define TRINE_CANON_VERSION "1.0.1"

/* ═══════════════════════════════════════════════════════════════════════
 * Preset Enum
 * ═══════════════════════════════════════════════════════════════════════ */

enum {
    TRINE_CANON_NONE    = 0,   /* No transforms (passthrough)             */
    TRINE_CANON_SUPPORT = 1,   /* whitespace + timestamps + uuids + nums  */
    TRINE_CANON_CODE    = 2,   /* whitespace + identifiers                */
    TRINE_CANON_POLICY  = 3,   /* whitespace + bucket_numbers             */
    TRINE_CANON_GENERAL = 4    /* whitespace only                         */
};

/* ═══════════════════════════════════════════════════════════════════════
 * Main API
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * trine_canon_apply — Apply a canonicalization preset to text.
 *
 * Copies input to out, then applies the preset's transforms in sequence.
 * Output is always null-terminated. out_len receives actual output length
 * (excluding null terminator).
 *
 * Returns 0 on success, -1 if out_cap is insufficient (need len + 1).
 */
int trine_canon_apply(const char *text, size_t len, int preset,
                      char *out, size_t out_cap, size_t *out_len);

/*
 * trine_canon_preset_name — Return human-readable name for a preset.
 *
 * Returns a pointer to a static string, e.g. "SUPPORT".
 * Returns "UNKNOWN" for invalid preset values.
 */
const char *trine_canon_preset_name(int preset);

/* ═══════════════════════════════════════════════════════════════════════
 * Individual Transforms (composable, in-place)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Collapse runs of whitespace to single space, trim leading/trailing. */
void trine_canon_normalize_whitespace(char *buf, size_t *len);

/* Remove ISO-8601 dates, HH:MM:SS times, 10-digit Unix timestamps. */
void trine_canon_strip_timestamps(char *buf, size_t *len);

/* Remove UUID patterns (8-4-4-4-12 hex format). */
void trine_canon_strip_uuids(char *buf, size_t *len);

/* Normalize identifiers: camelCase and snake_case to lowercase words. */
void trine_canon_normalize_identifiers(char *buf, size_t *len);

/* Replace digit runs with <N> placeholder. */
void trine_canon_bucket_numbers(char *buf, size_t *len);

#endif /* TRINE_CANON_H */
