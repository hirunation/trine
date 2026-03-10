/* =====================================================================
 * TRINE Stage-2 — Shared JSONL Parsing Utilities (implementation)
 * =====================================================================
 *
 * Simple strstr-based JSON extraction with no external library.
 * Extracts top-level string and numeric values by key name.
 *
 * Handles escape sequences: \", \\, \n, \t, \r, \/.
 * Properly skips escaped quotes inside string values.
 *
 * ===================================================================== */

#include "trine_jsonl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------- */
/* Internal: advance past "key" pattern and colon to the value position   */
/* --------------------------------------------------------------------- */

/* Searches for "key" in json, then advances past the colon and any
 * surrounding whitespace.  Returns pointer to the start of the value,
 * or NULL if not found. */
static const char *find_value_start(const char *json, const char *key)
{
    if (!json || !key) return NULL;

    /* Build search pattern: "key" */
    char pattern[128];
    int plen = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (plen < 0 || (size_t)plen >= sizeof(pattern)) return NULL;

    const char *pos = strstr(json, pattern);
    if (!pos) return NULL;

    /* Advance past the key pattern */
    pos += (size_t)plen;

    /* Skip whitespace before colon */
    while (*pos == ' ' || *pos == '\t') pos++;
    if (*pos != ':') return NULL;
    pos++;

    /* Skip whitespace after colon */
    while (*pos == ' ' || *pos == '\t') pos++;

    return pos;
}

/* --------------------------------------------------------------------- */
/* trine_jsonl_extract_string                                             */
/* --------------------------------------------------------------------- */

int trine_jsonl_extract_string(const char *json, size_t json_len,
                               const char *key,
                               char *out, size_t out_size)
{
    (void)json_len;  /* Reserved for future use */

    if (!json || !key || !out || out_size == 0) return -1;

    const char *pos = find_value_start(json, key);
    if (!pos) return -1;

    /* Expect opening quote */
    if (*pos != '"') return -1;
    pos++;

    /* Extract characters until unescaped closing quote */
    size_t out_len = 0;
    while (*pos && *pos != '"' && out_len < out_size - 1) {
        if (*pos == '\\' && pos[1]) {
            pos++;
            switch (*pos) {
                case '"':  out[out_len++] = '"';  break;
                case '\\': out[out_len++] = '\\'; break;
                case 'n':  out[out_len++] = '\n'; break;
                case 't':  out[out_len++] = '\t'; break;
                case 'r':  out[out_len++] = '\r'; break;
                case '/':  out[out_len++] = '/';  break;
                default:   out[out_len++] = *pos; break;
            }
        } else {
            out[out_len++] = *pos;
        }
        pos++;
    }

    out[out_len] = '\0';
    return (int)out_len;
}

/* --------------------------------------------------------------------- */
/* trine_jsonl_extract_float                                              */
/* --------------------------------------------------------------------- */

int trine_jsonl_extract_float(const char *json, size_t json_len,
                              const char *key, float *out)
{
    (void)json_len;  /* Reserved for future use */

    if (!json || !key || !out) return 0;

    const char *pos = find_value_start(json, key);
    if (!pos) return 0;

    /* Parse number with strtof */
    char *end = NULL;
    float val = strtof(pos, &end);
    if (end == pos) return 0;  /* No number parsed */

    *out = val;
    return 1;
}

/* --------------------------------------------------------------------- */
/* trine_jsonl_extract_source                                             */
/* --------------------------------------------------------------------- */

int trine_jsonl_extract_source(const char *json, size_t json_len,
                               char *out, size_t out_size)
{
    return trine_jsonl_extract_string(json, json_len, "source", out, out_size);
}
