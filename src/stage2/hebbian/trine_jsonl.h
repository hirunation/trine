/* =====================================================================
 * TRINE Stage-2 — Shared JSONL Parsing Utilities
 * =====================================================================
 *
 * Lightweight, zero-dependency JSON string/number extraction for JSONL
 * training data.  Used by trine_hebbian.c and trine_self_deepen.c.
 *
 * Only extracts top-level string and numeric values by key name.
 * Handles basic JSON escape sequences: \", \\, \n, \t, \r, \/.
 *
 * ===================================================================== */

#ifndef TRINE_JSONL_H
#define TRINE_JSONL_H

#include <stddef.h>

/* Extract a JSON string value by key from a single JSON line.
 *
 * Searches for "key" in `json`, then extracts the string value after
 * the colon and opening quote.  Writes to `out` (up to out_size-1
 * chars, always NUL-terminated on success).
 *
 * Handles escape sequences: \", \\, \n, \t, \r, \/.
 *
 * Parameters:
 *   json      - The JSON line to search (NUL-terminated).
 *   json_len  - Length of json (unused; reserved for future use).
 *               Pass 0 if unknown; the function uses NUL termination.
 *   key       - The key name to search for (without quotes).
 *   out       - Output buffer for the extracted string.
 *   out_size  - Size of the output buffer.
 *
 * Returns:
 *   Length of extracted string on success (>= 0), or -1 if not found
 *   or on error. */
int trine_jsonl_extract_string(const char *json, size_t json_len,
                               const char *key,
                               char *out, size_t out_size);

/* Extract a floating-point number value by key from a single JSON line.
 *
 * Searches for "key" in `json`, then parses the numeric value after
 * the colon using strtof().
 *
 * Parameters:
 *   json      - The JSON line to search (NUL-terminated).
 *   json_len  - Length of json (unused; reserved for future use).
 *               Pass 0 if unknown; the function uses NUL termination.
 *   key       - The key name to search for (without quotes).
 *   out       - Pointer to store the extracted float value.
 *
 * Returns:
 *   1 on success (value stored in *out), 0 if not found or on error. */
int trine_jsonl_extract_float(const char *json, size_t json_len,
                              const char *key, float *out);

/* Extract the "source" field from a JSON line.
 *
 * Convenience wrapper around trine_jsonl_extract_string() with
 * key = "source".
 *
 * Parameters:
 *   json      - The JSON line to search (NUL-terminated).
 *   json_len  - Length of json (unused; reserved for future use).
 *   out       - Output buffer for the extracted source string.
 *   out_size  - Size of the output buffer.
 *
 * Returns:
 *   Length of extracted string on success (>= 0), or -1 if not found. */
int trine_jsonl_extract_source(const char *json, size_t json_len,
                               char *out, size_t out_size);

#endif /* TRINE_JSONL_H */
