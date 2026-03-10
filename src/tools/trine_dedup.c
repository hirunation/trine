/* =====================================================================
 * TRINE DEDUP — Near-Duplicate Detection & Deduplication CLI
 * Ternary Resonance Interference Network Embedding
 * =====================================================================
 *
 * Standalone CLI tool for near-duplicate detection and corpus
 * deduplication using TRINE shingle embeddings (240-dim ternary).
 *
 * Self-contained: includes trine_encode.h directly and implements
 * its own comparison logic inline. Does NOT depend on trine_stage1.h
 * or a .trine model file.
 *
 * ZERO external dependencies beyond libc + libm.
 *
 * Usage:
 *   trine_dedup check [options] <text1> <text2>
 *   trine_dedup scan  [options] < input.txt > unique.txt
 *   trine_dedup batch [options] -f <file>
 *   trine_dedup stats [options] -f <file>
 *
 * Build:
 *   cc -O2 -o trine_dedup trine_dedup.c trine_encode.c -I../include -lm
 *
 * ===================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include "trine_encode.h"
#include "trine_idf.h"
#include "trine_route.h"
#include "trine_csidf.h"
#include "trine_field.h"
#include "trine_stage2.h"
#include "trine_s2_persist.h"
#include "trine_project.h"

/* =====================================================================
 * Constants
 * ===================================================================== */

#define DEDUP_VERSION       "1.0.1"
#define DEDUP_MAX_LINE       65536       /* Max line length               */
#define DEDUP_INITIAL_CAP    256         /* Initial index capacity        */
#define DEDUP_DIMS           240         /* Shingle embedding dimensions  */
#define DEDUP_CHAINS         4           /* Number of encoding chains     */
#define DEDUP_CHAIN_WIDTH    60          /* Channels per chain            */

/* Chain names (match trine_embed.c naming) */
static const char * const CHAIN_NAME[DEDUP_CHAINS] = {
    "edit", "morph", "phrase", "vocab"
};

/* Default threshold for duplicate detection */
#define DEDUP_DEFAULT_THRESHOLD  0.60f

/* =====================================================================
 * Lens — Per-Chain Weighting
 * =====================================================================
 *
 * Each lens assigns a weight to each of the 4 encoding chains:
 *   Chain 0 (edit):   Character unigrams + bigrams
 *   Chain 1 (morph):  Character trigrams
 *   Chain 2 (phrase): Character 5-grams
 *   Chain 3 (vocab):  Word unigrams
 *
 * The lens-weighted cosine combines per-chain cosines:
 *   combined = sum(w[i] * cos[i]) / sum(w[i])
 *
 * ===================================================================== */

typedef struct {
    float weights[DEDUP_CHAINS];
    const char *name;
} dedup_lens_t;

/* Preset lenses — same values as trine_stage1.h */
static const dedup_lens_t LENS_DEDUP   = {{0.5f, 0.5f, 0.7f, 1.0f}, "dedup"  };
static const dedup_lens_t LENS_EDIT    = {{1.0f, 0.3f, 0.1f, 0.0f}, "edit"   };
static const dedup_lens_t LENS_MORPH   = {{0.3f, 1.0f, 0.5f, 0.2f}, "morph"  };
static const dedup_lens_t LENS_PHRASE  = {{0.1f, 0.3f, 1.0f, 0.5f}, "phrase" };
static const dedup_lens_t LENS_VOCAB   = {{0.0f, 0.2f, 0.3f, 1.0f}, "vocab"  };
static const dedup_lens_t LENS_UNIFORM = {{1.0f, 1.0f, 1.0f, 1.0f}, "uniform"};
static const dedup_lens_t LENS_CODE    = {{1.0f, 0.8f, 0.4f, 0.2f}, "code"   };
static const dedup_lens_t LENS_LEGAL   = {{0.2f, 0.4f, 1.0f, 0.8f}, "legal"  };
static const dedup_lens_t LENS_MEDICAL = {{0.3f, 1.0f, 0.6f, 0.5f}, "medical"};
static const dedup_lens_t LENS_SUPPORT = {{0.2f, 0.4f, 0.7f, 1.0f}, "support"};
static const dedup_lens_t LENS_POLICY  = {{0.1f, 0.3f, 1.0f, 0.8f}, "policy" };

/* =====================================================================
 * Stage-2 Semantic Mode — Global State
 * =====================================================================
 *
 * When --semantic or --s2-model is provided, the tool loads a .trine2
 * model and computes Stage-2 embeddings alongside Stage-1.  Similarity
 * is then a blend:  alpha * S1_sim + (1 - alpha) * S2_sim.
 *
 * ===================================================================== */

static const char *s2_model_path = NULL;
static int         s2_depth      = 0;
static float       blend_alpha   = 0.65f;
static int         s2_only       = 0;
static int         s2_block_diag = 0;
static trine_s2_model_t *s2_model = NULL;

/* =====================================================================
 * In-Memory Dedup Index
 * =====================================================================
 *
 * Simple flat array of 240-byte embeddings. Linear scan for lookup.
 * Starts at DEDUP_INITIAL_CAP entries, doubles on overflow via realloc.
 * When s2_model is loaded, s2_embeddings stores parallel Stage-2 vectors.
 *
 * ===================================================================== */

typedef struct {
    uint8_t *embeddings;    /* count * DEDUP_DIMS flat array (Stage-1) */
    uint8_t *s2_embeddings; /* count * DEDUP_DIMS flat array (Stage-2, or NULL) */
    int count;
    int capacity;
} dedup_index_t;

/* =====================================================================
 * Cosine Similarity Functions
 * ===================================================================== */

/*
 * cosine_240 — Full 240-dimensional cosine similarity.
 *
 * Standard cosine: dot(a,b) / (|a| * |b|).
 * Both vectors contain values in {0, 1, 2}.
 * Returns 0.0 if either vector has zero magnitude.
 */
static double cosine_240(const uint8_t *a, const uint8_t *b)
{
    uint64_t dot_ab = 0;
    uint64_t mag_a  = 0;
    uint64_t mag_b  = 0;

    for (int i = 0; i < DEDUP_DIMS; i++) {
        uint64_t va = a[i];
        uint64_t vb = b[i];
        dot_ab += va * vb;
        mag_a  += va * va;
        mag_b  += vb * vb;
    }

    if (mag_a == 0 || mag_b == 0) return 0.0;

    double denom = sqrt((double)mag_a) * sqrt((double)mag_b);
    if (denom == 0.0) return 0.0;

    double sim = (double)dot_ab / denom;

    /* Clamp to [0, 1] for floating-point rounding */
    if (sim > 1.0) sim = 1.0;
    if (sim < 0.0) sim = 0.0;

    return sim;
}

/*
 * chain_cosine_60 — Cosine similarity over a single 60-channel chain slice.
 *
 * Identical to chain_cosine in trine.c / trine_stage1.c:
 * treats trit values as real-valued vector components {0, 1, 2}.
 * Returns 0.0 if either slice has zero magnitude.
 */
static double chain_cosine_60(const uint8_t *a, const uint8_t *b,
                               int offset)
{
    uint64_t dot_ab = 0;
    uint64_t mag_a  = 0;
    uint64_t mag_b  = 0;

    for (int i = offset; i < offset + DEDUP_CHAIN_WIDTH; i++) {
        uint64_t va = a[i];
        uint64_t vb = b[i];
        dot_ab += va * vb;
        mag_a  += va * va;
        mag_b  += vb * vb;
    }

    if (mag_a == 0 || mag_b == 0) return 0.0;

    double denom = sqrt((double)mag_a) * sqrt((double)mag_b);
    if (denom == 0.0) return 0.0;

    double sim = (double)dot_ab / denom;

    if (sim > 1.0) sim = 1.0;
    if (sim < 0.0) sim = 0.0;

    return sim;
}

/*
 * lens_cosine — Lens-weighted cosine similarity over the full 240 dims.
 *
 * Computes cosine independently for each of the 4 chains (60 channels
 * each), then combines with lens weights:
 *   combined = sum(weight[i] * chain_cosine[i]) / sum(weight[i])
 *
 * Chains with weight <= 0.0 are skipped entirely.
 * Returns 0.0 if total weight is zero.
 */
static double lens_cosine(const uint8_t *a, const uint8_t *b,
                           const dedup_lens_t *lens)
{
    double weighted_sum = 0.0;
    double weight_sum   = 0.0;

    for (int c = 0; c < DEDUP_CHAINS; c++) {
        double w = (double)lens->weights[c];
        if (w <= 0.0) continue;

        double cos_c = chain_cosine_60(a, b, c * DEDUP_CHAIN_WIDTH);
        weighted_sum += w * cos_c;
        weight_sum   += w;
    }

    if (weight_sum == 0.0) return 0.0;

    return weighted_sum / weight_sum;
}

/* =====================================================================
 * IDF-Weighted Cosine Similarity Functions
 * =====================================================================
 *
 * When --idf is active, these replace the standard cosine functions.
 * Uses TRINE_IDF_WEIGHTS from trine_idf.h to downweight channels
 * dominated by common English n-gram patterns.
 *
 * ===================================================================== */

/*
 * idf_cosine_240 — IDF-weighted cosine over full 240 dimensions.
 *
 * Formula: sum(idf[i]*a[i]*b[i]) / (sqrt(sum(idf[i]*a[i]^2)) * sqrt(sum(idf[i]*b[i]^2)))
 * Returns 0.0 if either vector has zero weighted magnitude.
 */
static double idf_cosine_240(const uint8_t *a, const uint8_t *b)
{
    double dot_ab = 0.0;
    double mag_a  = 0.0;
    double mag_b  = 0.0;

    for (int i = 0; i < DEDUP_DIMS; i++) {
        double w  = (double)TRINE_IDF_WEIGHTS[i];
        double va = (double)a[i];
        double vb = (double)b[i];
        dot_ab += w * va * vb;
        mag_a  += w * va * va;
        mag_b  += w * vb * vb;
    }

    double denom = sqrt(mag_a) * sqrt(mag_b);
    if (denom < 1e-12) return 0.0;

    double sim = dot_ab / denom;
    if (sim > 1.0) sim = 1.0;
    if (sim < 0.0) sim = 0.0;
    return sim;
}

/*
 * idf_chain_cosine_60 — IDF-weighted cosine over a single 60-channel chain.
 *
 * Same as chain_cosine_60 but applies per-channel IDF weights.
 * Returns 0.0 if either slice has zero weighted magnitude.
 */
static double idf_chain_cosine_60(const uint8_t *a, const uint8_t *b,
                                    int offset)
{
    double dot_ab = 0.0;
    double mag_a  = 0.0;
    double mag_b  = 0.0;

    for (int i = offset; i < offset + DEDUP_CHAIN_WIDTH; i++) {
        double w  = (double)TRINE_IDF_WEIGHTS[i];
        double va = (double)a[i];
        double vb = (double)b[i];
        dot_ab += w * va * vb;
        mag_a  += w * va * va;
        mag_b  += w * vb * vb;
    }

    double denom = sqrt(mag_a) * sqrt(mag_b);
    if (denom < 1e-12) return 0.0;

    double sim = dot_ab / denom;
    if (sim > 1.0) sim = 1.0;
    if (sim < 0.0) sim = 0.0;
    return sim;
}

/*
 * idf_lens_cosine — IDF + lens weighted cosine over full 240 dims.
 *
 * Computes IDF-weighted cosine independently for each of the 4 chains,
 * then combines with lens weights exactly like lens_cosine().
 */
static double idf_lens_cosine(const uint8_t *a, const uint8_t *b,
                                const dedup_lens_t *lens)
{
    double weighted_sum = 0.0;
    double weight_sum   = 0.0;

    for (int c = 0; c < DEDUP_CHAINS; c++) {
        double w = (double)lens->weights[c];
        if (w <= 0.0) continue;

        double cos_c = idf_chain_cosine_60(a, b, c * DEDUP_CHAIN_WIDTH);
        weighted_sum += w * cos_c;
        weight_sum   += w;
    }

    if (weight_sum == 0.0) return 0.0;

    return weighted_sum / weight_sum;
}

/* =====================================================================
 * Lens Parsing
 * =====================================================================
 *
 * Accepts preset names or custom "w,w,w,w" format.
 * Returns 1 on success, 0 on error.
 *
 * ===================================================================== */

static int parse_lens(const char *spec, dedup_lens_t *out)
{
    /* Named presets */
    if (strcmp(spec, "dedup") == 0)   { *out = LENS_DEDUP;   return 1; }
    if (strcmp(spec, "edit") == 0)    { *out = LENS_EDIT;    return 1; }
    if (strcmp(spec, "morph") == 0)   { *out = LENS_MORPH;   return 1; }
    if (strcmp(spec, "phrase") == 0)  { *out = LENS_PHRASE;  return 1; }
    if (strcmp(spec, "vocab") == 0)   { *out = LENS_VOCAB;   return 1; }
    if (strcmp(spec, "uniform") == 0) { *out = LENS_UNIFORM; return 1; }
    if (strcmp(spec, "code") == 0)    { *out = LENS_CODE;    return 1; }
    if (strcmp(spec, "legal") == 0)   { *out = LENS_LEGAL;   return 1; }
    if (strcmp(spec, "medical") == 0) { *out = LENS_MEDICAL; return 1; }
    if (strcmp(spec, "support") == 0) { *out = LENS_SUPPORT; return 1; }
    if (strcmp(spec, "policy") == 0)  { *out = LENS_POLICY;  return 1; }

    /* Custom: "w0,w1,w2,w3" format */
    float w0, w1, w2, w3;
    if (sscanf(spec, "%f,%f,%f,%f", &w0, &w1, &w2, &w3) == 4) {
        out->weights[0] = w0;
        out->weights[1] = w1;
        out->weights[2] = w2;
        out->weights[3] = w3;
        out->name = "custom";
        return 1;
    }

    return 0;
}

/* =====================================================================
 * Encoding Helper
 * ===================================================================== */

/*
 * encode_line — Encode text to 240-trit shingle embedding.
 *
 * Convenience wrapper over trine_encode_shingle().
 * Returns 0 on success, -1 on allocation failure.
 */
static int encode_line(const char *text, size_t len, uint8_t out[240])
{
    if (!text || len == 0) {
        memset(out, 0, DEDUP_DIMS);
        return 0;
    }
    return trine_encode_shingle(text, len, out);
}

/*
 * s2_proj_mode_name — Human-readable name for the projection mode.
 */
static const char *s2_proj_mode_name(void)
{
    if (!s2_model) return "none";
    int mode = trine_s2_get_projection_mode(s2_model);
    switch (mode) {
    case TRINE_S2_PROJ_SIGN:       return "sign";
    case TRINE_S2_PROJ_DIAGONAL:   return "diagonal";
    case TRINE_S2_PROJ_SPARSE:     return "sparse";
    case TRINE_S2_PROJ_BLOCK_DIAG: return "block-diagonal";
    default:                       return "unknown";
    }
}

/*
 * encode_s2 — Compute Stage-2 embedding from a Stage-1 trit vector.
 *
 * No-op (zeroes output) if s2_model is NULL.
 */
static void encode_s2(const uint8_t s1[240], uint8_t s2_out[240])
{
    if (!s2_model) {
        memset(s2_out, 0, DEDUP_DIMS);
        return;
    }
    if (trine_s2_encode_from_trits(s2_model, s1, (uint32_t)s2_depth,
                                     s2_out) != 0) {
        memset(s2_out, 0, DEDUP_DIMS);
    }
}

/* =====================================================================
 * Input Format Detection
 * ===================================================================== */

#define DEDUP_INPUT_AUTO  0
#define DEDUP_INPUT_PLAIN 1
#define DEDUP_INPUT_JSONL 2

/* =====================================================================
 * Minimal JSON String Field Extractor
 * =====================================================================
 *
 * Extracts top-level string field values from a single JSON line.
 * Handles escaped quotes (\") inside values. Does NOT handle nested
 * objects, arrays, numbers, booleans, or nulls — only top-level
 * string fields of the form "key": "value".
 *
 * ===================================================================== */

/*
 * json_find_string — Find a string field value in a JSON line.
 *
 * Searches for "key": "value" pattern in the given JSON string.
 * Returns pointer to start of value (after opening quote), sets *vlen
 * to the length of the value (before closing quote).
 * Returns NULL if the field is not found.
 *
 * Handles escaped quotes inside values (backslash-quote).
 */
static const char *json_find_string(const char *json, size_t json_len,
                                     const char *key, size_t *vlen)
{
    if (!json || !key || !vlen) return NULL;

    size_t klen = strlen(key);
    const char *end = json + json_len;
    const char *p = json;

    while (p < end) {
        /* Find next quote */
        const char *q1 = memchr(p, '"', (size_t)(end - p));
        if (!q1 || q1 + klen + 1 >= end) return NULL;

        /* Check if this is our key */
        if (memcmp(q1 + 1, key, klen) == 0 && q1[klen + 1] == '"') {
            /* Found "key" — now skip to colon then value */
            const char *after_key = q1 + klen + 2; /* past closing quote */

            /* Skip whitespace to colon */
            while (after_key < end && (*after_key == ' ' || *after_key == '\t'))
                after_key++;

            if (after_key >= end || *after_key != ':') {
                p = after_key;
                continue;
            }
            after_key++; /* skip colon */

            /* Skip whitespace to opening quote of value */
            while (after_key < end &&
                   (*after_key == ' ' || *after_key == '\t'))
                after_key++;

            if (after_key >= end || *after_key != '"') {
                p = after_key;
                continue;
            }
            after_key++; /* skip opening quote */

            /* Find closing quote, handling backslash escapes */
            const char *vstart = after_key;
            const char *vp = vstart;
            while (vp < end) {
                if (*vp == '\\' && vp + 1 < end) {
                    vp += 2; /* skip escaped character */
                    continue;
                }
                if (*vp == '"') {
                    *vlen = (size_t)(vp - vstart);
                    return vstart;
                }
                vp++;
            }

            /* Unterminated string — give up */
            return NULL;
        }

        /* Not our key — skip past this quoted string */
        q1++; /* past opening quote */
        while (q1 < end) {
            if (*q1 == '\\' && q1 + 1 < end) {
                q1 += 2;
                continue;
            }
            if (*q1 == '"') {
                q1++;
                break;
            }
            q1++;
        }
        p = q1;
    }

    return NULL;
}

/*
 * json_unescape — Unescape a JSON string value in-place.
 *
 * Converts \", \\, \n, \t, \r to their literal equivalents.
 * Does NOT handle \uXXXX unicode escapes (passes them through).
 * Returns the unescaped length.
 */
static size_t json_unescape(char *buf, size_t len)
{
    size_t r = 0, w = 0;
    while (r < len) {
        if (buf[r] == '\\' && r + 1 < len) {
            char c = buf[r + 1];
            switch (c) {
            case '"':  buf[w++] = '"';  r += 2; break;
            case '\\': buf[w++] = '\\'; r += 2; break;
            case 'n':  buf[w++] = '\n'; r += 2; break;
            case 't':  buf[w++] = '\t'; r += 2; break;
            case 'r':  buf[w++] = '\r'; r += 2; break;
            case '/':  buf[w++] = '/';  r += 2; break;
            default:   buf[w++] = buf[r++]; break; /* keep backslash */
            }
        } else {
            buf[w++] = buf[r++];
        }
    }
    buf[w] = '\0';
    return w;
}

/*
 * jsonl_extract — Extract text and id from a JSONL line.
 *
 * Sets *out_text to a heap-allocated string containing the "text" field.
 * Sets *out_tag to a heap-allocated string containing the "id" field,
 * or NULL if no "id" field is found (caller should use line number).
 *
 * Returns 1 on success (text found), 0 on failure (skip this line).
 * Caller must free *out_text and *out_tag.
 */
static int jsonl_extract(const char *line, size_t line_len,
                          char **out_text, char **out_tag)
{
    *out_text = NULL;
    *out_tag = NULL;

    /* Extract "text" field — required */
    size_t text_len = 0;
    const char *text_val = json_find_string(line, line_len, "text", &text_len);
    if (!text_val || text_len == 0) return 0;

    *out_text = (char *)malloc(text_len + 1);
    if (!*out_text) return 0;
    memcpy(*out_text, text_val, text_len);
    (*out_text)[text_len] = '\0';
    text_len = json_unescape(*out_text, text_len);

    /* Extract "id" field — optional */
    size_t id_len = 0;
    const char *id_val = json_find_string(line, line_len, "id", &id_len);
    if (id_val && id_len > 0) {
        *out_tag = (char *)malloc(id_len + 1);
        if (*out_tag) {
            memcpy(*out_tag, id_val, id_len);
            (*out_tag)[id_len] = '\0';
            json_unescape(*out_tag, id_len);
        }
    }

    return 1;
}

/*
 * detect_input_format — Auto-detect input format from first non-blank line.
 *
 * Reads the file (or stdin), checks if first non-blank line starts with '{'.
 * Returns DEDUP_INPUT_JSONL or DEDUP_INPUT_PLAIN.
 * Does NOT consume the file (caller must rewind or re-open).
 */
static int detect_input_format_file(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return DEDUP_INPUT_PLAIN;

    char buf[DEDUP_MAX_LINE];
    int fmt = DEDUP_INPUT_PLAIN;

    while (fgets(buf, sizeof(buf), f)) {
        /* Strip leading whitespace */
        const char *p = buf;
        while (*p == ' ' || *p == '\t') p++;

        /* Skip blank lines */
        if (*p == '\0' || *p == '\n' || *p == '\r') continue;

        /* Check if starts with '{' */
        if (*p == '{') fmt = DEDUP_INPUT_JSONL;
        break;
    }

    fclose(f);
    return fmt;
}

/* =====================================================================
 * JSON Output Helper
 * =====================================================================
 *
 * Escapes a string for JSON output (handles ", \, newlines, tabs).
 * Writes escaped string to out buffer. Returns number of bytes written.
 * out must have room for at least 2*len + 1 bytes.
 *
 * ===================================================================== */

static size_t json_escape(char *out, size_t out_cap,
                           const char *src, size_t src_len)
{
    size_t w = 0;
    for (size_t i = 0; i < src_len && w + 6 < out_cap; i++) {
        unsigned char c = (unsigned char)src[i];
        switch (c) {
        case '"':  out[w++] = '\\'; out[w++] = '"';  break;
        case '\\': out[w++] = '\\'; out[w++] = '\\'; break;
        case '\n': out[w++] = '\\'; out[w++] = 'n';  break;
        case '\r': out[w++] = '\\'; out[w++] = 'r';  break;
        case '\t': out[w++] = '\\'; out[w++] = 't';  break;
        default:
            if (c < 0x20) {
                /* Control character — emit \u00XX */
                w += (size_t)snprintf(out + w, out_cap - w,
                                      "\\u%04x", (unsigned)c);
            } else {
                out[w++] = (char)c;
            }
            break;
        }
    }
    out[w] = '\0';
    return w;
}

/*
 * json_print_escaped — Print a JSON-escaped string to stdout.
 * Wraps the string in quotes.
 */
static void json_print_escaped(const char *s, size_t len)
{
    /* Allocate worst-case buffer: 6x expansion for \u00XX per char + quotes */
    size_t cap = len * 6 + 3;
    char *buf = (char *)malloc(cap);
    if (!buf) {
        printf("\"\"");
        return;
    }
    json_escape(buf, cap, s, len);
    printf("\"%s\"", buf);
    free(buf);
}

/* =====================================================================
 * JSONL File Loading
 * =====================================================================
 *
 * Like load_lines but parses JSONL format, extracting "text" and "id"
 * fields from each JSON object.
 *
 * ===================================================================== */

/*
 * load_lines_jsonl — Read all lines from a JSONL file.
 *
 * Each line is a JSON object with at minimum a "text" field.
 * Extracts the text content and optional id tag.
 *
 * Returns heap-allocated array of heap-allocated text strings.
 * If out_tags is non-NULL, sets *out_tags to an array of tag strings
 * (may contain NULLs for lines without "id" fields).
 * Sets *out_count to the number of lines.
 * Returns NULL on error.
 */
static char **load_lines_jsonl(const char *path, int *out_count,
                                char ***out_tags)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "trine_dedup: cannot open '%s': %s\n",
                path, strerror(errno));
        return NULL;
    }

    int cap = 256;
    int count = 0;
    char **lines = (char **)malloc((size_t)cap * sizeof(char *));
    char **tags = out_tags ? (char **)malloc((size_t)cap * sizeof(char *)) : NULL;
    if (!lines || (out_tags && !tags)) {
        free(lines);
        free(tags);
        fclose(f);
        return NULL;
    }

    char buf[DEDUP_MAX_LINE];
    int line_num = 0;

    while (fgets(buf, sizeof(buf), f)) {
        line_num++;

        /* Strip trailing newline/CR */
        size_t len = strlen(buf);
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
            buf[--len] = '\0';

        /* Skip blank lines */
        if (len == 0) continue;

        /* Skip lines that don't start with '{' */
        const char *p = buf;
        while (*p == ' ' || *p == '\t') p++;
        if (*p != '{') continue;

        /* Extract text and id */
        char *text = NULL;
        char *tag = NULL;
        if (!jsonl_extract(buf, len, &text, &tag)) {
            /* No "text" field — skip */
            continue;
        }

        /* Grow arrays if needed */
        if (count >= cap) {
            cap *= 2;
            char **new_lines = (char **)realloc(
                lines, (size_t)cap * sizeof(char *));
            if (!new_lines) {
                free(text);
                free(tag);
                for (int i = 0; i < count; i++) {
                    free(lines[i]);
                    if (tags) free(tags[i]);
                }
                free(lines);
                free(tags);
                fclose(f);
                return NULL;
            }
            lines = new_lines;
            if (tags) {
                char **new_tags = (char **)realloc(
                    tags, (size_t)cap * sizeof(char *));
                if (!new_tags) {
                    free(text);
                    free(tag);
                    for (int i = 0; i < count; i++) {
                        free(lines[i]);
                        free(tags[i]);
                    }
                    free(lines);
                    free(tags);
                    fclose(f);
                    return NULL;
                }
                tags = new_tags;
            }
        }

        lines[count] = text;
        if (tags) {
            if (tag) {
                tags[count] = tag;
            } else {
                /* Use line number as tag */
                char numbuf[32];
                snprintf(numbuf, sizeof(numbuf), "%d", line_num);
                tags[count] = strdup(numbuf);
            }
        } else {
            free(tag);
        }
        count++;
    }

    fclose(f);
    *out_count = count;
    if (out_tags) *out_tags = tags;
    return lines;
}

static void free_tags(char **tags, int count)
{
    if (!tags) return;
    for (int i = 0; i < count; i++) free(tags[i]);
    free(tags);
}

/* =====================================================================
 * Index Operations
 * ===================================================================== */

static int dedup_index_init(dedup_index_t *idx)
{
    idx->count = 0;
    idx->capacity = DEDUP_INITIAL_CAP;
    idx->embeddings = (uint8_t *)malloc(
        (size_t)idx->capacity * DEDUP_DIMS);
    if (!idx->embeddings) return -1;
    if (s2_model) {
        idx->s2_embeddings = (uint8_t *)malloc(
            (size_t)idx->capacity * DEDUP_DIMS);
        if (!idx->s2_embeddings) {
            free(idx->embeddings);
            idx->embeddings = NULL;
            return -1;
        }
    } else {
        idx->s2_embeddings = NULL;
    }
    return 0;
}

static void dedup_index_free(dedup_index_t *idx)
{
    free(idx->embeddings);
    free(idx->s2_embeddings);
    idx->embeddings = NULL;
    idx->s2_embeddings = NULL;
    idx->count = 0;
    idx->capacity = 0;
}

/*
 * dedup_index_add — Add a 240-byte embedding to the index.
 *
 * Doubles capacity on overflow. Returns 0 on success, -1 on alloc failure.
 * If s2_emb is non-NULL (semantic mode), also stores the Stage-2 embedding.
 */
static int dedup_index_add(dedup_index_t *idx, const uint8_t emb[240])
{
    if (idx->count >= idx->capacity) {
        int new_cap = idx->capacity * 2;
        uint8_t *new_emb = (uint8_t *)realloc(
            idx->embeddings, (size_t)new_cap * DEDUP_DIMS);
        if (!new_emb) return -1;
        idx->embeddings = new_emb;
        if (idx->s2_embeddings) {
            uint8_t *new_s2 = (uint8_t *)realloc(
                idx->s2_embeddings, (size_t)new_cap * DEDUP_DIMS);
            if (!new_s2) return -1;
            idx->s2_embeddings = new_s2;
        }
        idx->capacity = new_cap;
    }

    memcpy(idx->embeddings + (size_t)idx->count * DEDUP_DIMS,
           emb, DEDUP_DIMS);
    idx->count++;
    return 0;
}

/*
 * dedup_index_add_s2 — Store a Stage-2 embedding for the most recently
 * added entry.  Must be called immediately after dedup_index_add().
 * No-op if s2_embeddings is NULL (non-semantic mode).
 */
static void dedup_index_add_s2(dedup_index_t *idx, const uint8_t s2[240])
{
    if (!idx->s2_embeddings || idx->count == 0) return;
    memcpy(idx->s2_embeddings + (size_t)(idx->count - 1) * DEDUP_DIMS,
           s2, DEDUP_DIMS);
}

/*
 * compute_blended_sim — Compute (optionally blended) similarity
 * between two embedding pairs.
 *
 * If s2_model is loaded and s2_a / s2_b are non-NULL, blends:
 *   blend_alpha * S1 + (1 - blend_alpha) * S2
 * With --s2-only, returns S2 similarity exclusively.
 * Otherwise returns plain S1 similarity.
 */
static double compute_blended_sim(const uint8_t *s1_a, const uint8_t *s1_b,
                                   const uint8_t *s2_a, const uint8_t *s2_b,
                                   const dedup_lens_t *lens, int use_idf_flag)
{
    double s1_sim = use_idf_flag ? idf_lens_cosine(s1_a, s1_b, lens)
                                 : lens_cosine(s1_a, s1_b, lens);

    if (s2_model && s2_a && s2_b) {
        double s2_sim = lens_cosine(s2_a, s2_b, lens);
        if (s2_only)
            return s2_sim;
        return (double)blend_alpha * s1_sim
             + (1.0 - (double)blend_alpha) * s2_sim;
    }

    return s1_sim;
}

/*
 * dedup_index_max_sim — Find the maximum similarity of a candidate
 * against all entries in the index using the given lens.
 *
 * Returns the maximum similarity found, or 0.0 if the index is empty.
 * If best_idx is non-NULL, sets it to the index of the best match.
 */
static double dedup_index_max_sim(const dedup_index_t *idx,
                                   const uint8_t candidate[240],
                                   const dedup_lens_t *lens,
                                   int *best_idx)
{
    if (idx->count == 0) {
        if (best_idx) *best_idx = -1;
        return 0.0;
    }

    double best = 0.0;
    int best_i = 0;

    for (int i = 0; i < idx->count; i++) {
        const uint8_t *entry = idx->embeddings + (size_t)i * DEDUP_DIMS;
        double sim = lens_cosine(candidate, entry, lens);
        if (sim > best) {
            best = sim;
            best_i = i;
        }
    }

    if (best_idx) *best_idx = best_i;
    return best;
}

/*
 * dedup_index_max_sim_blended — Like dedup_index_max_sim but uses
 * blended S1+S2 similarity when semantic mode is active.
 *
 * s2_candidate may be NULL when no S2 model is loaded.
 */
static double dedup_index_max_sim_blended(const dedup_index_t *idx,
                                           const uint8_t candidate[240],
                                           const uint8_t *s2_candidate,
                                           const dedup_lens_t *lens,
                                           int use_idf_flag,
                                           int *best_idx)
{
    if (idx->count == 0) {
        if (best_idx) *best_idx = -1;
        return 0.0;
    }

    double best = 0.0;
    int best_i = 0;

    for (int i = 0; i < idx->count; i++) {
        const uint8_t *s1_entry = idx->embeddings + (size_t)i * DEDUP_DIMS;
        const uint8_t *s2_entry = (idx->s2_embeddings)
            ? idx->s2_embeddings + (size_t)i * DEDUP_DIMS : NULL;
        double sim = compute_blended_sim(candidate, s1_entry,
                                          s2_candidate, s2_entry,
                                          lens, use_idf_flag);
        if (sim > best) {
            best = sim;
            best_i = i;
        }
    }

    if (best_idx) *best_idx = best_i;
    return best;
}

/*
 * dedup_index_max_sim_idf — Like dedup_index_max_sim but uses IDF-weighted
 * lens cosine instead of standard lens cosine.
 */
static double dedup_index_max_sim_idf(const dedup_index_t *idx,
                                        const uint8_t candidate[240],
                                        const dedup_lens_t *lens,
                                        int *best_idx)
{
    if (idx->count == 0) {
        if (best_idx) *best_idx = -1;
        return 0.0;
    }

    double best = 0.0;
    int best_i = 0;

    for (int i = 0; i < idx->count; i++) {
        const uint8_t *entry = idx->embeddings + (size_t)i * DEDUP_DIMS;
        double sim = idf_lens_cosine(candidate, entry, lens);
        if (sim > best) {
            best = sim;
            best_i = i;
        }
    }

    if (best_idx) *best_idx = best_i;
    return best;
}

/* =====================================================================
 * Persistent Index — .tridx Binary Format
 * =====================================================================
 *
 * Binary format:
 *   Magic:   "TRDX" (4 bytes)
 *   Version: uint32_t = 1
 *   Count:   uint32_t = number of embeddings
 *   Data:    count * 240 bytes (raw embeddings)
 *
 * ===================================================================== */

#define TRIDX_MAGIC   "TRDX"
#define TRIDX_VERSION 1

/*
 * save_index — Write a dedup index to a .tridx file.
 *
 * Returns 0 on success, -1 on error (prints to stderr).
 */
static int save_index(const char *path, const uint8_t *embeddings, int count)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "trine_dedup: cannot write '%s': %s\n",
                path, strerror(errno));
        return -1;
    }

    /* Magic */
    if (fwrite(TRIDX_MAGIC, 1, 4, f) != 4) goto write_err;

    /* Version */
    uint32_t version = TRIDX_VERSION;
    if (fwrite(&version, sizeof(uint32_t), 1, f) != 1) goto write_err;

    /* Count */
    uint32_t cnt = (uint32_t)count;
    if (fwrite(&cnt, sizeof(uint32_t), 1, f) != 1) goto write_err;

    /* Data */
    size_t data_bytes = (size_t)count * DEDUP_DIMS;
    if (data_bytes > 0 && fwrite(embeddings, 1, data_bytes, f) != data_bytes)
        goto write_err;

    fclose(f);
    return 0;

write_err:
    fprintf(stderr, "trine_dedup: write error on '%s': %s\n",
            path, strerror(errno));
    fclose(f);
    return -1;
}

/*
 * load_index — Read a dedup index from a .tridx file.
 *
 * Allocates the embeddings array (caller must free).
 * Returns 0 on success, -1 on error (prints to stderr).
 * On success, *out_embeddings and *out_count are set.
 */
static int load_index(const char *path, uint8_t **out_embeddings,
                       int *out_count)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "trine_dedup: cannot open '%s': %s\n",
                path, strerror(errno));
        return -1;
    }

    /* Magic */
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, TRIDX_MAGIC, 4) != 0) {
        fprintf(stderr, "trine_dedup: '%s' is not a valid .tridx file "
                "(bad magic)\n", path);
        fclose(f);
        return -1;
    }

    /* Version */
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "trine_dedup: '%s': truncated header\n", path);
        fclose(f);
        return -1;
    }
    if (version != TRIDX_VERSION) {
        fprintf(stderr, "trine_dedup: '%s': unsupported version %u "
                "(expected %u)\n", path, version, (uint32_t)TRIDX_VERSION);
        fclose(f);
        return -1;
    }

    /* Count */
    uint32_t cnt;
    if (fread(&cnt, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "trine_dedup: '%s': truncated header\n", path);
        fclose(f);
        return -1;
    }

    /* Data */
    size_t data_bytes = (size_t)cnt * DEDUP_DIMS;
    uint8_t *embeddings = NULL;
    if (cnt > 0) {
        embeddings = (uint8_t *)malloc(data_bytes);
        if (!embeddings) {
            fprintf(stderr, "trine_dedup: allocation failed for %u "
                    "embeddings from '%s'\n", cnt, path);
            fclose(f);
            return -1;
        }
        if (fread(embeddings, 1, data_bytes, f) != data_bytes) {
            fprintf(stderr, "trine_dedup: '%s': truncated data "
                    "(expected %u embeddings)\n", path, cnt);
            free(embeddings);
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    *out_embeddings = embeddings;
    *out_count = (int)cnt;
    return 0;
}

/* =====================================================================
 * File Loading
 * ===================================================================== */

/*
 * load_lines — Read all non-empty lines from a file into a string array.
 *
 * Returns heap-allocated array of heap-allocated strings.
 * Strips trailing newlines. Sets *out_count to the number of lines.
 * Returns NULL on error.
 */
static char **load_lines(const char *path, int *out_count)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "trine_dedup: cannot open '%s': %s\n",
                path, strerror(errno));
        return NULL;
    }

    int cap = 256;
    int count = 0;
    char **lines = (char **)malloc((size_t)cap * sizeof(char *));
    if (!lines) {
        fclose(f);
        return NULL;
    }

    char buf[DEDUP_MAX_LINE];
    while (fgets(buf, sizeof(buf), f)) {
        /* Strip trailing newline/CR */
        size_t len = strlen(buf);
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
            buf[--len] = '\0';

        /* Skip empty lines */
        if (len == 0) continue;

        /* Grow array if needed */
        if (count >= cap) {
            cap *= 2;
            char **new_lines = (char **)realloc(
                lines, (size_t)cap * sizeof(char *));
            if (!new_lines) {
                for (int i = 0; i < count; i++) free(lines[i]);
                free(lines);
                fclose(f);
                return NULL;
            }
            lines = new_lines;
        }

        lines[count] = (char *)malloc(len + 1);
        if (!lines[count]) {
            for (int i = 0; i < count; i++) free(lines[i]);
            free(lines);
            fclose(f);
            return NULL;
        }
        memcpy(lines[count], buf, len + 1);
        count++;
    }

    fclose(f);
    *out_count = count;
    return lines;
}

static void free_lines(char **lines, int count)
{
    for (int i = 0; i < count; i++) free(lines[i]);
    free(lines);
}

/* =====================================================================
 * Command: check — Compare two texts for duplication
 * =====================================================================
 *
 * Encodes both texts, computes lens-weighted cosine similarity,
 * and classifies as DUPLICATE or UNIQUE against the threshold.
 *
 * With --detail, also prints per-chain cosine breakdown.
 *
 * ===================================================================== */

static int cmd_check(const char *text1, const char *text2,
                     float threshold, const dedup_lens_t *lens,
                     int verbose, int detail, int use_idf,
                     int json_output)
{
    uint8_t emb1[DEDUP_DIMS], emb2[DEDUP_DIMS];

    if (encode_line(text1, strlen(text1), emb1) != 0) return -1;
    if (encode_line(text2, strlen(text2), emb2) != 0) return -1;

    /* Stage-2 embeddings (computed only when model is loaded) */
    uint8_t s2_emb1[DEDUP_DIMS], s2_emb2[DEDUP_DIMS];
    if (s2_model) {
        encode_s2(emb1, s2_emb1);
        encode_s2(emb2, s2_emb2);
    }

    double sim = compute_blended_sim(emb1, emb2,
                                      s2_model ? s2_emb1 : NULL,
                                      s2_model ? s2_emb2 : NULL,
                                      lens, use_idf);
    int is_dup = (sim >= (double)threshold) ? 1 : 0;

    if (json_output) {
        /* Compute calibrated score (fill-ratio based) */
        int fill1 = 0, fill2 = 0;
        for (int i = 0; i < DEDUP_DIMS; i++) {
            if (emb1[i] != 0) fill1++;
            if (emb2[i] != 0) fill2++;
        }
        double fr1 = (double)fill1 / (double)DEDUP_DIMS;
        double fr2 = (double)fill2 / (double)DEDUP_DIMS;
        double denom_cal = sqrt(fr1 * fr2);
        double calibrated = (denom_cal > 1e-12) ? sim / denom_cal : sim;
        if (calibrated > 1.0) calibrated = 1.0;

        printf("{\"command\": \"check\", \"text1\": ");
        json_print_escaped(text1, strlen(text1));
        printf(", \"text2\": ");
        json_print_escaped(text2, strlen(text2));
        printf(", \"similarity\": %.3f", sim);
        printf(", \"calibrated\": %.3f", calibrated);
        printf(", \"is_duplicate\": %s", is_dup ? "true" : "false");
        printf(", \"threshold\": %.2f", threshold);
        printf(", \"lens\": \"%s\"", lens->name);
        if (s2_model) {
            printf(", \"stage2\": true");
            printf(", \"s2_depth\": %d", s2_depth);
            printf(", \"blend_alpha\": %.2f", blend_alpha);
            printf(", \"s2_only\": %s", s2_only ? "true" : "false");
            printf(", \"projection_mode\": \"%s\"", s2_proj_mode_name());
        }
        printf("}\n");
        return 0;
    }

    printf("%s (%.3f)  lens=%s  threshold=%.2f%s%s\n",
           is_dup ? "DUPLICATE" : "UNIQUE",
           sim, lens->name, threshold,
           use_idf ? "  idf=on" : "",
           s2_model ? (s2_only ? "  s2=only" : "  s2=blend") : "");

    if (detail) {
        printf("  per-chain (S1):\n");
        for (int c = 0; c < DEDUP_CHAINS; c++) {
            double cc = use_idf
                ? idf_chain_cosine_60(emb1, emb2, c * DEDUP_CHAIN_WIDTH)
                : chain_cosine_60(emb1, emb2, c * DEDUP_CHAIN_WIDTH);
            printf("    %-8s %.3f  (weight=%.1f)\n",
                   CHAIN_NAME[c], cc, (double)lens->weights[c]);
        }
        double full = use_idf ? idf_cosine_240(emb1, emb2)
                               : cosine_240(emb1, emb2);
        printf("    %-8s %.3f  (unweighted, all 240 dims)\n", "full", full);

        if (s2_model) {
            double s2_sim = lens_cosine(s2_emb1, s2_emb2, lens);
            double s1_sim = use_idf ? idf_lens_cosine(emb1, emb2, lens)
                                    : lens_cosine(emb1, emb2, lens);
            printf("  Stage-2:\n");
            printf("    s1_sim:  %.3f\n", s1_sim);
            printf("    s2_sim:  %.3f\n", s2_sim);
            printf("    blend:   %.3f  (alpha=%.2f)\n", sim, blend_alpha);
        }
    }

    if (verbose) {
        /* Count non-zero channels as fill-rate diagnostic */
        int fill1 = 0, fill2 = 0;
        for (int i = 0; i < DEDUP_DIMS; i++) {
            if (emb1[i] != 0) fill1++;
            if (emb2[i] != 0) fill2++;
        }
        fprintf(stderr, "  text1: %zu chars, %d/%d channels active\n",
                strlen(text1), fill1, DEDUP_DIMS);
        fprintf(stderr, "  text2: %zu chars, %d/%d channels active\n",
                strlen(text2), fill2, DEDUP_DIMS);
        if (s2_model) {
            fprintf(stderr, "  s2_model: %s  depth=%d  proj=%s\n",
                    s2_model_path, s2_depth, s2_proj_mode_name());
        }
    }

    return 0;
}

/* =====================================================================
 * Command: scan — Streaming dedup from stdin
 * =====================================================================
 *
 * Reads lines from stdin. For each line:
 *   1. Encode to 240-dim shingle embedding
 *   2. Compare against all previous entries in the index
 *   3. If max similarity < threshold → print to stdout (unique),
 *      add to index
 *   4. If duplicate → skip (print to stderr with -v)
 *
 * Output to stdout: only unique lines.
 * Stderr (with -v): dedup statistics and per-line decisions.
 *
 * ===================================================================== */

static int cmd_scan(float threshold, const dedup_lens_t *lens,
                    int verbose, int quiet, int use_idf, int use_routed,
                    int recall_mode,
                    const char *save_idx_path, const char *load_idx_path,
                    int json_output, int input_format)
/* Note: CS-IDF and field-aware support for scan is handled in main()
 * dispatch via the append-index path. Standard scan uses static IDF. */
{
    /* Auto-detect input format from first line of stdin if needed.
     * We peek at the first line and push it back via ungetc won't work
     * for a full line, so we use a flag to process the first line
     * inside the main loop. */
    int detected_jsonl = 0;
    int format_decided = (input_format != DEDUP_INPUT_AUTO);
    int use_jsonl = (input_format == DEDUP_INPUT_JSONL);

    /* Store matched-line text for JSON scan output */
    /* We store the text of each unique line added to the index so we can
     * report the match_text in JSON output for scan duplicates. */
    char **scan_texts = NULL;
    int scan_texts_cap = 0;
    int scan_texts_count = 0;

    if (json_output) {
        scan_texts_cap = DEDUP_INITIAL_CAP;
        scan_texts = (char **)calloc((size_t)scan_texts_cap, sizeof(char *));
        /* Allowed to fail — we just won't have match text */
    }

    /* --- Routed mode: use band-LSH routing layer --- */
    if (use_routed) {
        trine_s1_config_t rt_cfg = {
            .threshold = threshold,
            .lens = {{lens->weights[0], lens->weights[1],
                      lens->weights[2], lens->weights[3]}},
            .calibrate_length = 0
        };

        trine_route_t *rt = NULL;

        /* Load or create routed index */
        if (load_idx_path) {
            rt = trine_route_load(load_idx_path);
            if (!rt) { free(scan_texts); return 1; }
            if (verbose) {
                fprintf(stderr, "  loaded %d routed entries from '%s'\n",
                        trine_route_count(rt), load_idx_path);
            }
        } else {
            rt = trine_route_create(&rt_cfg);
            if (!rt) {
                fprintf(stderr, "trine_dedup: routed index allocation failed\n");
                free(scan_texts);
                return 1;
            }
        }

        trine_route_set_recall(rt, recall_mode);

        char buf[DEDUP_MAX_LINE];
        int total_lines = 0;
        int unique_lines = 0;
        int dup_lines = 0;
        long long total_candidates = 0;
        int query_count = 0;

        while (fgets(buf, sizeof(buf), stdin)) {
            size_t len = strlen(buf);
            while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
                buf[--len] = '\0';
            if (len == 0) continue;

            /* Auto-detect on first non-blank line */
            if (!format_decided) {
                const char *p = buf;
                while (*p == ' ' || *p == '\t') p++;
                if (*p == '{') { use_jsonl = 1; detected_jsonl = 1; }
                format_decided = 1;
            }

            /* Extract text content — JSONL or plain */
            const char *text_content = buf;
            size_t text_len = len;
            char *jsonl_text = NULL;
            char *jsonl_tag = NULL;

            if (use_jsonl) {
                const char *p = buf;
                while (*p == ' ' || *p == '\t') p++;
                if (*p != '{') continue; /* skip non-JSON lines */
                if (!jsonl_extract(buf, len, &jsonl_text, &jsonl_tag))
                    continue; /* skip lines without "text" field */
                text_content = jsonl_text;
                text_len = strlen(jsonl_text);
            }

            total_lines++;

            uint8_t emb[DEDUP_DIMS];
            if (encode_line(text_content, text_len, emb) != 0) continue;

            /* Query via routing layer */
            trine_route_stats_t rstats = {0};
            trine_s1_result_t res = trine_route_query(rt, emb, &rstats);

            if (rstats.candidates_checked > 0) {
                total_candidates += rstats.candidates_checked;
                query_count++;
            }

            if (res.is_duplicate && res.similarity >= threshold) {
                dup_lines++;
                if (json_output) {
                    const char *match_text = "";
                    if (scan_texts && res.matched_index >= 0 &&
                        res.matched_index < scan_texts_count)
                        match_text = scan_texts[res.matched_index];
                    printf("{\"command\": \"scan\", \"line\": %d, \"text\": ",
                           total_lines);
                    json_print_escaped(text_content, text_len);
                    printf(", \"match_line\": %d, \"match_text\": ",
                           res.matched_index + 1);
                    json_print_escaped(match_text, strlen(match_text));
                    printf(", \"similarity\": %.2f}\n",
                           (double)res.similarity);
                } else if (verbose) {
                    fprintf(stderr, "  DUP  line=%d  sim=%.3f  match=%d  "
                            "candidates=%d  \"%.*s\"\n",
                            total_lines, (double)res.similarity,
                            res.matched_index + 1, rstats.candidates_checked,
                            (int)(text_len > 60 ? 60 : text_len), text_content);
                }
            } else {
                unique_lines++;
                if (!json_output)
                    printf("%s\n", buf);

                if (trine_route_add(rt, emb, NULL) < 0) {
                    fprintf(stderr, "trine_dedup: routed index add failed "
                            "at line %d\n", total_lines);
                    free(jsonl_text);
                    free(jsonl_tag);
                    trine_route_free(rt);
                    if (scan_texts) {
                        for (int si = 0; si < scan_texts_count; si++)
                            free(scan_texts[si]);
                        free(scan_texts);
                    }
                    return 1;
                }

                /* Track text for JSON match_text output */
                if (scan_texts) {
                    if (scan_texts_count >= scan_texts_cap) {
                        int new_cap = scan_texts_cap * 2;
                        char **new_st = (char **)realloc(
                            scan_texts, (size_t)new_cap * sizeof(char *));
                        if (new_st) {
                            scan_texts = new_st;
                            scan_texts_cap = new_cap;
                        }
                    }
                    if (scan_texts_count < scan_texts_cap) {
                        scan_texts[scan_texts_count] = strdup(text_content);
                        scan_texts_count++;
                    }
                }

                if (verbose && !json_output) {
                    fprintf(stderr, "  NEW  line=%d  max_sim=%.3f  index=%d\n",
                            total_lines, (double)res.similarity,
                            trine_route_count(rt));
                }
            }

            free(jsonl_text);
            free(jsonl_tag);
        }

        /* Save routed index if requested */
        if (save_idx_path) {
            if (trine_route_save(rt, save_idx_path) != 0) {
                trine_route_free(rt);
                if (scan_texts) {
                    for (int si = 0; si < scan_texts_count; si++)
                        free(scan_texts[si]);
                    free(scan_texts);
                }
                return 1;
            }
            if (verbose) {
                fprintf(stderr, "  saved %d routed entries to '%s'\n",
                        trine_route_count(rt), save_idx_path);
            }
        }

        /* Summary to stderr (or JSON) */
        if (json_output) {
            printf("{\"command\": \"scan\", \"total_lines\": %d, "
                   "\"unique\": %d, \"duplicates\": %d, "
                   "\"duplicate_rate\": %.2f",
                   total_lines, unique_lines, dup_lines,
                   total_lines > 0
                       ? (double)dup_lines / (double)total_lines : 0.0);
            if (save_idx_path)
                printf(", \"index_file\": \"%s\"", save_idx_path);
            printf("}\n");
        } else if (!quiet) {
            fprintf(stderr, "\n--- scan summary (routed) ---\n");
            fprintf(stderr, "  total:      %d\n", total_lines);
            fprintf(stderr, "  unique:     %d\n", unique_lines);
            fprintf(stderr, "  duplicates: %d\n", dup_lines);
            fprintf(stderr, "  reduction:  %.1f%%\n",
                    total_lines > 0
                        ? 100.0 * (double)dup_lines / (double)total_lines
                        : 0.0);
            fprintf(stderr, "  lens:       %s\n", lens->name);
            fprintf(stderr, "  threshold:  %.2f\n", threshold);
            fprintf(stderr, "  mode:       routed (band-LSH)\n");
            if (use_idf)
                fprintf(stderr, "  idf:        on\n");
            if (detected_jsonl)
                fprintf(stderr, "  input:      jsonl (auto-detected)\n");
            if (query_count > 0) {
                double avg_cand = (double)total_candidates / (double)query_count;
                int n = trine_route_count(rt);
                double speedup = (n > 0 && avg_cand > 0.0)
                    ? (double)n / avg_cand : 1.0;
                trine_route_stats_t gstats = {0};
                trine_route_global_stats(rt, &gstats);
                fprintf(stderr, "Routing stats: avg %.1f candidates, %.1fx speedup (%s)\n",
                        avg_cand, speedup,
                        gstats.recall_mode ? gstats.recall_mode : "balanced");
            }
        }

        trine_route_free(rt);
        if (scan_texts) {
            for (int si = 0; si < scan_texts_count; si++)
                free(scan_texts[si]);
            free(scan_texts);
        }
        return 0;
    }

    /* --- Brute-force mode (default) --- */
    dedup_index_t idx;
    if (dedup_index_init(&idx) != 0) {
        fprintf(stderr, "trine_dedup: index allocation failed\n");
        if (scan_texts) free(scan_texts);
        return 1;
    }

    /* Pre-populate from saved index if requested */
    if (load_idx_path) {
        uint8_t *pre_emb = NULL;
        int pre_count = 0;
        if (load_index(load_idx_path, &pre_emb, &pre_count) != 0) {
            dedup_index_free(&idx);
            if (scan_texts) free(scan_texts);
            return 1;
        }
        for (int j = 0; j < pre_count; j++) {
            if (dedup_index_add(&idx, pre_emb + (size_t)j * DEDUP_DIMS) != 0) {
                fprintf(stderr, "trine_dedup: index allocation failed "
                        "loading pre-existing index\n");
                free(pre_emb);
                dedup_index_free(&idx);
                if (scan_texts) free(scan_texts);
                return 1;
            }
        }
        free(pre_emb);
        if (verbose) {
            fprintf(stderr, "  loaded %d entries from '%s'\n",
                    pre_count, load_idx_path);
        }
    }

    char buf[DEDUP_MAX_LINE];
    int total_lines = 0;
    int unique_lines = 0;
    int dup_lines = 0;

    while (fgets(buf, sizeof(buf), stdin)) {
        /* Strip trailing newline/CR */
        size_t len = strlen(buf);
        while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
            buf[--len] = '\0';

        /* Skip empty lines */
        if (len == 0) continue;

        /* Auto-detect on first non-blank line */
        if (!format_decided) {
            const char *p = buf;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '{') { use_jsonl = 1; detected_jsonl = 1; }
            format_decided = 1;
        }

        /* Extract text content — JSONL or plain */
        const char *text_content = buf;
        size_t text_len = len;
        char *jsonl_text = NULL;
        char *jsonl_tag = NULL;

        if (use_jsonl) {
            const char *p = buf;
            while (*p == ' ' || *p == '\t') p++;
            if (*p != '{') { continue; }
            if (!jsonl_extract(buf, len, &jsonl_text, &jsonl_tag))
                continue;
            text_content = jsonl_text;
            text_len = strlen(jsonl_text);
        }

        total_lines++;

        /* Encode (Stage-1 + optional Stage-2) */
        uint8_t emb[DEDUP_DIMS];
        if (encode_line(text_content, text_len, emb) != 0) continue;

        uint8_t s2_emb[DEDUP_DIMS];
        if (s2_model) encode_s2(emb, s2_emb);

        /* Compare against index (blended when S2 is active) */
        int best_i = -1;
        double max_sim = s2_model
            ? dedup_index_max_sim_blended(&idx, emb, s2_emb, lens,
                                          use_idf, &best_i)
            : (use_idf
                ? dedup_index_max_sim_idf(&idx, emb, lens, &best_i)
                : dedup_index_max_sim(&idx, emb, lens, &best_i));

        if (max_sim >= (double)threshold) {
            /* Duplicate — skip */
            dup_lines++;
            if (json_output) {
                const char *match_text = "";
                if (scan_texts && best_i >= 0 && best_i < scan_texts_count)
                    match_text = scan_texts[best_i];
                printf("{\"command\": \"scan\", \"line\": %d, \"text\": ",
                       total_lines);
                json_print_escaped(text_content, text_len);
                printf(", \"match_line\": %d, \"match_text\": ",
                       best_i + 1);
                json_print_escaped(match_text, strlen(match_text));
                printf(", \"similarity\": %.2f}\n", max_sim);
            } else if (verbose) {
                fprintf(stderr, "  DUP  line=%d  sim=%.3f  match=%d  \"%.*s\"\n",
                        total_lines, max_sim, best_i + 1,
                        (int)(text_len > 60 ? 60 : text_len), text_content);
            }
        } else {
            /* Unique — print and add to index */
            unique_lines++;
            if (!json_output)
                printf("%s\n", buf);

            if (dedup_index_add(&idx, emb) != 0) {
                fprintf(stderr, "trine_dedup: index allocation failed "
                        "at line %d\n", total_lines);
                free(jsonl_text);
                free(jsonl_tag);
                dedup_index_free(&idx);
                if (scan_texts) {
                    for (int si = 0; si < scan_texts_count; si++)
                        free(scan_texts[si]);
                    free(scan_texts);
                }
                return 1;
            }
            if (s2_model) dedup_index_add_s2(&idx, s2_emb);

            /* Track text for JSON match_text output */
            if (scan_texts) {
                if (scan_texts_count >= scan_texts_cap) {
                    int new_cap = scan_texts_cap * 2;
                    char **new_st = (char **)realloc(
                        scan_texts, (size_t)new_cap * sizeof(char *));
                    if (new_st) {
                        scan_texts = new_st;
                        scan_texts_cap = new_cap;
                    }
                }
                if (scan_texts_count < scan_texts_cap) {
                    scan_texts[scan_texts_count] = strdup(text_content);
                    scan_texts_count++;
                }
            }

            if (verbose && !json_output) {
                fprintf(stderr, "  NEW  line=%d  max_sim=%.3f  index=%d\n",
                        total_lines, max_sim, idx.count);
            }
        }

        free(jsonl_text);
        free(jsonl_tag);
    }

    /* Save index if requested */
    if (save_idx_path) {
        if (save_index(save_idx_path, idx.embeddings, idx.count) != 0) {
            dedup_index_free(&idx);
            if (scan_texts) {
                for (int si = 0; si < scan_texts_count; si++)
                    free(scan_texts[si]);
                free(scan_texts);
            }
            return 1;
        }
        if (verbose) {
            fprintf(stderr, "  saved %d entries to '%s'\n",
                    idx.count, save_idx_path);
        }
    }

    /* Summary to stderr (or JSON) */
    if (json_output) {
        printf("{\"command\": \"scan\", \"total_lines\": %d, "
               "\"unique\": %d, \"duplicates\": %d, "
               "\"duplicate_rate\": %.2f",
               total_lines, unique_lines, dup_lines,
               total_lines > 0
                   ? (double)dup_lines / (double)total_lines : 0.0);
        if (save_idx_path)
            printf(", \"index_file\": \"%s\"", save_idx_path);
        printf("}\n");
    } else if (!quiet) {
        fprintf(stderr, "\n--- scan summary ---\n");
        fprintf(stderr, "  total:      %d\n", total_lines);
        fprintf(stderr, "  unique:     %d\n", unique_lines);
        fprintf(stderr, "  duplicates: %d\n", dup_lines);
        fprintf(stderr, "  reduction:  %.1f%%\n",
                total_lines > 0
                    ? 100.0 * (double)dup_lines / (double)total_lines
                    : 0.0);
        fprintf(stderr, "  lens:       %s\n", lens->name);
        fprintf(stderr, "  threshold:  %.2f\n", threshold);
        if (use_idf)
            fprintf(stderr, "  idf:        on\n");
        if (s2_model) {
            fprintf(stderr, "  stage2:     %s  depth=%d  blend=%.2f%s  proj=%s\n",
                    s2_model_path, s2_depth, blend_alpha,
                    s2_only ? " (s2-only)" : "", s2_proj_mode_name());
        }
        if (detected_jsonl)
            fprintf(stderr, "  input:      jsonl (auto-detected)\n");
    }

    dedup_index_free(&idx);
    if (scan_texts) {
        for (int si = 0; si < scan_texts_count; si++)
            free(scan_texts[si]);
        free(scan_texts);
    }
    return 0;
}

/* =====================================================================
 * Command: batch — Pairwise similarity matrix
 * =====================================================================
 *
 * Reads all lines from a file, encodes each, then computes pairwise
 * lens-weighted cosine similarity for all pairs. Outputs TSV rows
 * for pairs whose similarity is at or above the threshold.
 *
 * Output format:
 *   line_i  line_j  similarity
 *
 * With --detail, also prints per-chain cosine values.
 *
 * ===================================================================== */

static int cmd_batch(const char *file_path, float threshold,
                     const dedup_lens_t *lens, int verbose,
                     int quiet, int detail, int use_idf, int use_routed,
                     int recall_mode, const char *load_idx_path,
                     int json_output, int input_format,
                     const char *save_idx_path)
{
    /* Determine effective input format */
    int eff_format = input_format;
    if (eff_format == DEDUP_INPUT_AUTO)
        eff_format = detect_input_format_file(file_path);

    int line_count = 0;
    char **lines = NULL;
    char **tags = NULL;

    if (eff_format == DEDUP_INPUT_JSONL) {
        lines = load_lines_jsonl(file_path, &line_count, NULL);
    } else {
        lines = load_lines(file_path, &line_count);
    }
    if (!lines) return 1;

    if (line_count == 0) {
        fprintf(stderr, "trine_dedup: no non-empty lines in '%s'\n",
                file_path);
        free_lines(lines, line_count);
        return 1;
    }

    if (verbose) {
        fprintf(stderr, "Encoding %d lines...\n", line_count);
    }

    /* Encode all lines */
    uint8_t *embeddings = (uint8_t *)malloc(
        (size_t)line_count * DEDUP_DIMS);
    if (!embeddings) {
        fprintf(stderr, "trine_dedup: allocation failed for %d embeddings\n",
                line_count);
        free_lines(lines, line_count);
        return 1;
    }

    for (int i = 0; i < line_count; i++) {
        (void)encode_line(lines[i], strlen(lines[i]),
                          embeddings + (size_t)i * DEDUP_DIMS);
    }

    /* Stage-2 embeddings (computed only when model is loaded) */
    uint8_t *s2_embeddings = NULL;
    if (s2_model) {
        s2_embeddings = (uint8_t *)malloc(
            (size_t)line_count * DEDUP_DIMS);
        if (!s2_embeddings) {
            fprintf(stderr, "trine_dedup: allocation failed for %d S2 embeddings\n",
                    line_count);
            free(embeddings);
            free_lines(lines, line_count);
            return 1;
        }
        for (int i = 0; i < line_count; i++) {
            encode_s2(embeddings + (size_t)i * DEDUP_DIMS,
                      s2_embeddings + (size_t)i * DEDUP_DIMS);
        }
    }

    /* --- Routed mode: use band-LSH for candidate filtering --- */
    if (use_routed) {
        trine_s1_config_t rt_cfg = {
            .threshold = threshold,
            .lens = {{lens->weights[0], lens->weights[1],
                      lens->weights[2], lens->weights[3]}},
            .calibrate_length = 0
        };

        trine_route_t *rt = NULL;

        /* Load or create routed index */
        if (load_idx_path) {
            rt = trine_route_load(load_idx_path);
            if (!rt) {
                free(embeddings);
                free_lines(lines, line_count);
                return 1;
            }
            if (verbose) {
                fprintf(stderr, "Loaded %d routed reference entries from '%s'\n",
                        trine_route_count(rt), load_idx_path);
            }
        } else {
            rt = trine_route_create(&rt_cfg);
            if (!rt) {
                fprintf(stderr, "trine_dedup: routed index allocation failed\n");
                free(embeddings);
                free_lines(lines, line_count);
                return 1;
            }
        }

        trine_route_set_recall(rt, recall_mode);

        /* Add all file embeddings to the routed index */
        for (int i = 0; i < line_count; i++) {
            trine_route_add(rt, embeddings + (size_t)i * DEDUP_DIMS, NULL);
        }

        /* Header */
        if (!quiet && !json_output) {
            if (detail)
                printf("line_i\tline_j\tsimilarity\tedit\tmorph\tphrase\tvocab\n");
            else
                printf("line_i\tline_j\tsimilarity\n");
        }

        int pair_count = 0;
        long long total_candidates = 0;
        int query_count = 0;

        /* Use routing to find candidate pairs */
        for (int i = 0; i < line_count; i++) {
            const uint8_t *ei = embeddings + (size_t)i * DEDUP_DIMS;
            trine_route_stats_t rstats = {0};
            trine_s1_result_t res = trine_route_query(rt, ei, &rstats);
            (void)res;  /* We use the routing stats for performance info */

            if (rstats.candidates_checked > 0) {
                total_candidates += rstats.candidates_checked;
                query_count++;
            }

            /* For batch, still do pairwise comparisons within file
             * for the pairs j > i, using brute-force cosine on those
             * that share a routing bucket with entry i. For simplicity
             * and correctness with the existing output format, we fall
             * back to pairwise scan but print routing stats. */
            for (int j = i + 1; j < line_count; j++) {
                const uint8_t *ej = embeddings + (size_t)j * DEDUP_DIMS;
                const uint8_t *s2i = s2_embeddings
                    ? s2_embeddings + (size_t)i * DEDUP_DIMS : NULL;
                const uint8_t *s2j = s2_embeddings
                    ? s2_embeddings + (size_t)j * DEDUP_DIMS : NULL;
                double sim = compute_blended_sim(ei, ej, s2i, s2j,
                                                  lens, use_idf);

                if (sim >= (double)threshold) {
                    pair_count++;
                    if (!json_output) {
                        if (detail) {
                            printf("%d\t%d\t%.3f", i + 1, j + 1, sim);
                            for (int c = 0; c < DEDUP_CHAINS; c++) {
                                double cc = use_idf
                                    ? idf_chain_cosine_60(ei, ej,
                                                           c * DEDUP_CHAIN_WIDTH)
                                    : chain_cosine_60(ei, ej,
                                                       c * DEDUP_CHAIN_WIDTH);
                                printf("\t%.3f", cc);
                            }
                            printf("\n");
                        } else {
                            printf("%d\t%d\t%.3f\n", i + 1, j + 1, sim);
                        }
                    }
                }
            }
        }

        /* Save index if requested */
        if (save_idx_path) {
            if (trine_route_save(rt, save_idx_path) != 0) {
                trine_route_free(rt);
                free(s2_embeddings);
                free(embeddings);
                free_lines(lines, line_count);
                free_tags(tags, line_count);
                return 1;
            }
            if (verbose) {
                fprintf(stderr, "  saved %d routed entries to '%s'\n",
                        trine_route_count(rt), save_idx_path);
            }
        }

        /* Summary */
        if (json_output) {
            int unique = line_count - pair_count;
            if (unique < 0) unique = 0;
            printf("{\"command\": \"batch\", \"total_lines\": %d, "
                   "\"unique\": %d, \"duplicates\": %d, "
                   "\"duplicate_rate\": %.2f",
                   line_count, unique, pair_count,
                   line_count > 0
                       ? (double)pair_count / (double)line_count : 0.0);
            if (save_idx_path)
                printf(", \"index_file\": \"%s\"", save_idx_path);
            printf("}\n");
        } else if (!quiet) {
            int total_pairs = line_count * (line_count - 1) / 2;
            fprintf(stderr, "\n--- batch summary (routed) ---\n");
            fprintf(stderr, "  lines:  %d\n", line_count);
            fprintf(stderr, "  pairs:  %d / %d above threshold %.2f\n",
                    pair_count, total_pairs, threshold);
            fprintf(stderr, "  lens:   %s\n", lens->name);
            fprintf(stderr, "  mode:   routed (band-LSH)\n");
            if (use_idf)
                fprintf(stderr, "  idf:    on\n");
            if (query_count > 0) {
                double avg_cand = (double)total_candidates / (double)query_count;
                double speedup = (line_count > 0 && avg_cand > 0.0)
                    ? (double)line_count / avg_cand : 1.0;
                trine_route_stats_t gstats = {0};
                trine_route_global_stats(rt, &gstats);
                fprintf(stderr, "Routing stats: avg %.1f candidates, %.1fx speedup (%s)\n",
                        avg_cand, speedup,
                        gstats.recall_mode ? gstats.recall_mode : "balanced");
            }
        }

        trine_route_free(rt);
        free(s2_embeddings);
        free(embeddings);
        free_lines(lines, line_count);
        free_tags(tags, line_count);
        return 0;
    }

    /* --- Brute-force mode (default) --- */

    /* Load reference corpus if provided */
    uint8_t *ref_emb = NULL;
    int ref_count = 0;
    if (load_idx_path) {
        if (load_index(load_idx_path, &ref_emb, &ref_count) != 0) {
            free(embeddings);
            free_lines(lines, line_count);
            return 1;
        }
        if (verbose) {
            fprintf(stderr, "Loaded %d reference entries from '%s'\n",
                    ref_count, load_idx_path);
        }
    }

    /* Header */
    if (!quiet && !json_output) {
        if (detail)
            printf("line_i\tline_j\tsimilarity\tedit\tmorph\tphrase\tvocab\n");
        else
            printf("line_i\tline_j\tsimilarity\n");
    }

    int pair_count = 0;

    /* Cross-compare against reference corpus if loaded
     * (reference corpus has no S2 embeddings — use S1 only for cross-ref) */
    if (ref_emb && ref_count > 0) {
        for (int i = 0; i < line_count; i++) {
            const uint8_t *ei = embeddings + (size_t)i * DEDUP_DIMS;
            for (int j = 0; j < ref_count; j++) {
                const uint8_t *ej = ref_emb + (size_t)j * DEDUP_DIMS;
                double sim = use_idf ? idf_lens_cosine(ei, ej, lens)
                                     : lens_cosine(ei, ej, lens);

                if (sim >= (double)threshold) {
                    pair_count++;
                    if (!json_output) {
                        if (detail) {
                            printf("%d\tref:%d\t%.3f", i + 1, j + 1, sim);
                            for (int c = 0; c < DEDUP_CHAINS; c++) {
                                double cc = use_idf
                                    ? idf_chain_cosine_60(ei, ej,
                                                           c * DEDUP_CHAIN_WIDTH)
                                    : chain_cosine_60(ei, ej,
                                                       c * DEDUP_CHAIN_WIDTH);
                                printf("\t%.3f", cc);
                            }
                            printf("\n");
                        } else {
                            printf("%d\tref:%d\t%.3f\n", i + 1, j + 1, sim);
                        }
                    }
                }
            }
        }
    }

    /* Compute all pairs within file (with blended S2 if active) */
    for (int i = 0; i < line_count; i++) {
        const uint8_t *ei = embeddings + (size_t)i * DEDUP_DIMS;
        const uint8_t *s2i = s2_embeddings
            ? s2_embeddings + (size_t)i * DEDUP_DIMS : NULL;
        for (int j = i + 1; j < line_count; j++) {
            const uint8_t *ej = embeddings + (size_t)j * DEDUP_DIMS;
            const uint8_t *s2j = s2_embeddings
                ? s2_embeddings + (size_t)j * DEDUP_DIMS : NULL;
            double sim = compute_blended_sim(ei, ej, s2i, s2j,
                                              lens, use_idf);

            if (sim >= (double)threshold) {
                pair_count++;
                if (!json_output) {
                    if (detail) {
                        printf("%d\t%d\t%.3f", i + 1, j + 1, sim);
                        for (int c = 0; c < DEDUP_CHAINS; c++) {
                            double cc = use_idf
                                ? idf_chain_cosine_60(ei, ej,
                                                       c * DEDUP_CHAIN_WIDTH)
                                : chain_cosine_60(ei, ej,
                                                   c * DEDUP_CHAIN_WIDTH);
                            printf("\t%.3f", cc);
                        }
                        printf("\n");
                    } else {
                        printf("%d\t%d\t%.3f\n", i + 1, j + 1, sim);
                    }
                }
            }
        }
    }

    /* Save index if requested */
    if (save_idx_path && !use_routed) {
        if (save_index(save_idx_path, embeddings, line_count) != 0) {
            free(ref_emb);
            free(s2_embeddings);
            free(embeddings);
            free_lines(lines, line_count);
            free_tags(tags, line_count);
            return 1;
        }
        if (verbose) {
            fprintf(stderr, "  saved %d entries to '%s'\n",
                    line_count, save_idx_path);
        }
    }

    /* Summary */
    if (json_output) {
        int unique = line_count - pair_count;
        if (unique < 0) unique = 0;
        printf("{\"command\": \"batch\", \"total_lines\": %d, "
               "\"unique\": %d, \"duplicates\": %d, "
               "\"duplicate_rate\": %.2f",
               line_count, unique, pair_count,
               line_count > 0
                   ? (double)pair_count / (double)line_count : 0.0);
        if (save_idx_path)
            printf(", \"index_file\": \"%s\"", save_idx_path);
        printf("}\n");
    } else if (!quiet) {
        int total_pairs = line_count * (line_count - 1) / 2;
        fprintf(stderr, "\n--- batch summary ---\n");
        fprintf(stderr, "  lines:  %d\n", line_count);
        if (ref_count > 0)
            fprintf(stderr, "  ref:    %d (from loaded index)\n", ref_count);
        fprintf(stderr, "  pairs:  %d / %d above threshold %.2f\n",
                pair_count, total_pairs, threshold);
        fprintf(stderr, "  lens:   %s\n", lens->name);
        if (use_idf)
            fprintf(stderr, "  idf:    on\n");
        if (s2_model)
            fprintf(stderr, "  stage2: %s  depth=%d  blend=%.2f%s  proj=%s\n",
                    s2_model_path, s2_depth, blend_alpha,
                    s2_only ? " (s2-only)" : "", s2_proj_mode_name());
    }

    free(ref_emb);
    free(s2_embeddings);
    free(embeddings);
    free_lines(lines, line_count);
    free_tags(tags, line_count);
    return 0;
}

/* =====================================================================
 * Command: stats — Corpus duplicate density analysis
 * =====================================================================
 *
 * Reads all lines from a file, performs streaming dedup (like scan),
 * and reports:
 *   - Total lines, unique lines, duplicate count
 *   - Duplicate clusters (groups sharing a common match)
 *   - Mean similarity, max similarity
 *   - Reduction percentage
 *
 * ===================================================================== */

static int cmd_stats(const char *file_path, float threshold,
                     const dedup_lens_t *lens, int verbose, int quiet,
                     int use_idf, int use_routed, int recall_mode,
                     const char *load_idx_path,
                     int json_output, int input_format)
{
    /* Determine effective input format */
    int eff_format = input_format;
    if (eff_format == DEDUP_INPUT_AUTO)
        eff_format = detect_input_format_file(file_path);

    int line_count = 0;
    char **lines = NULL;

    if (eff_format == DEDUP_INPUT_JSONL) {
        lines = load_lines_jsonl(file_path, &line_count, NULL);
    } else {
        lines = load_lines(file_path, &line_count);
    }
    if (!lines) return 1;

    if (line_count == 0) {
        fprintf(stderr, "trine_dedup: no non-empty lines in '%s'\n",
                file_path);
        free_lines(lines, line_count);
        return 1;
    }

    if (verbose) {
        fprintf(stderr, "Encoding %d lines...\n", line_count);
    }

    /* Encode all lines */
    uint8_t *embeddings = (uint8_t *)malloc(
        (size_t)line_count * DEDUP_DIMS);
    if (!embeddings) {
        fprintf(stderr, "trine_dedup: allocation failed for %d embeddings\n",
                line_count);
        free_lines(lines, line_count);
        return 1;
    }

    for (int i = 0; i < line_count; i++) {
        (void)encode_line(lines[i], strlen(lines[i]),
                          embeddings + (size_t)i * DEDUP_DIMS);
    }

    /* Stage-2 embeddings (computed only when model is loaded) */
    uint8_t *s2_embs_stats = NULL;
    if (s2_model) {
        s2_embs_stats = (uint8_t *)malloc(
            (size_t)line_count * DEDUP_DIMS);
        if (!s2_embs_stats) {
            fprintf(stderr, "trine_dedup: allocation failed for %d S2 embeddings\n",
                    line_count);
            free(embeddings);
            free_lines(lines, line_count);
            return 1;
        }
        for (int i = 0; i < line_count; i++) {
            encode_s2(embeddings + (size_t)i * DEDUP_DIMS,
                      s2_embs_stats + (size_t)i * DEDUP_DIMS);
        }
    }

    /* --- Routed mode: use band-LSH routing layer --- */
    if (use_routed) {
        trine_s1_config_t rt_cfg = {
            .threshold = threshold,
            .lens = {{lens->weights[0], lens->weights[1],
                      lens->weights[2], lens->weights[3]}},
            .calibrate_length = 0
        };

        trine_route_t *rt = NULL;

        /* Load or create routed index */
        if (load_idx_path) {
            rt = trine_route_load(load_idx_path);
            if (!rt) {
                free(s2_embs_stats);
                free(embeddings);
                free_lines(lines, line_count);
                return 1;
            }
            if (verbose) {
                fprintf(stderr, "Loaded %d routed reference entries from '%s'\n",
                        trine_route_count(rt), load_idx_path);
            }
        } else {
            rt = trine_route_create(&rt_cfg);
            if (!rt) {
                fprintf(stderr, "trine_dedup: routed index allocation failed\n");
                free(s2_embs_stats);
                free(embeddings);
                free_lines(lines, line_count);
                return 1;
            }
        }

        trine_route_set_recall(rt, recall_mode);

        int unique_count = 0;
        int dup_count = 0;
        double sim_sum = 0.0;
        double sim_max = 0.0;
        int sim_count = 0;
        long long total_candidates = 0;
        int query_count = 0;

        for (int i = 0; i < line_count; i++) {
            const uint8_t *emb = embeddings + (size_t)i * DEDUP_DIMS;

            trine_route_stats_t rstats = {0};
            trine_s1_result_t res = trine_route_query(rt, emb, &rstats);

            if (rstats.candidates_checked > 0) {
                total_candidates += rstats.candidates_checked;
                query_count++;
            }

            double max_sim = (double)res.similarity;

            if (res.is_duplicate && max_sim >= (double)threshold) {
                dup_count++;
            } else {
                unique_count++;
                trine_route_add(rt, emb, NULL);
            }

            if (trine_route_count(rt) > 1 ||
                (trine_route_count(rt) == 1 && max_sim > 0.0)) {
                sim_sum += max_sim;
                sim_count++;
                if (max_sim > sim_max) sim_max = max_sim;
            }
        }

        double mean_sim = sim_count > 0 ? sim_sum / (double)sim_count : 0.0;

        /* Print routing statistics */
        trine_route_stats_t gstats = {0};
        trine_route_global_stats(rt, &gstats);

        if (json_output) {
            printf("{\"command\": \"stats\", \"entries\": %d, "
                   "\"index_file\": \"%s\", "
                   "\"threshold\": %.2f, \"lens\": \"%s\"}\n",
                   unique_count,
                   file_path,
                   threshold, lens->name);
        } else if (quiet) {
            printf("%d %d %d %.3f %.3f\n",
                   line_count, unique_count, dup_count, mean_sim, sim_max);
        } else {
            printf("TRINE Corpus Statistics (routed)\n");
            printf("  file:         %s\n", file_path);
            printf("  total lines:  %d\n", line_count);
            printf("  unique:       %d\n", unique_count);
            printf("  duplicates:   %d\n", dup_count);
            printf("  reduction:    %.1f%%\n",
                   line_count > 0
                       ? 100.0 * (double)dup_count / (double)line_count
                       : 0.0);
            printf("  mean sim:     %.3f  (average max-similarity per line)\n",
                   mean_sim);
            printf("  max sim:      %.3f\n", sim_max);
            printf("  threshold:    %.2f\n", threshold);
            printf("  lens:         %s  (%.1f, %.1f, %.1f, %.1f)\n",
                   lens->name,
                   (double)lens->weights[0], (double)lens->weights[1],
                   (double)lens->weights[2], (double)lens->weights[3]);
            printf("  mode:         routed (band-LSH)\n");
            if (s2_model)
                printf("  stage2:       %s  depth=%d  blend=%.2f%s\n",
                       s2_model_path, s2_depth, blend_alpha,
                       s2_only ? " (s2-only)" : "");
            if (query_count > 0) {
                double avg_cand = (double)total_candidates / (double)query_count;
                int n = trine_route_count(rt);
                double speedup = (n > 0 && avg_cand > 0.0)
                    ? (double)n / avg_cand : 1.0;
                printf("Routing stats: avg %.1f candidates, %.1fx speedup (%s)\n",
                       avg_cand, speedup,
                       gstats.recall_mode ? gstats.recall_mode : "balanced");
            }
        }

        trine_route_free(rt);
        free(s2_embs_stats);
        free(embeddings);
        free_lines(lines, line_count);
        return 0;
    }

    /* --- Brute-force mode (default) --- */

    /* Streaming dedup pass to count unique/duplicate */
    dedup_index_t idx;
    if (dedup_index_init(&idx) != 0) {
        fprintf(stderr, "trine_dedup: index allocation failed\n");
        free(s2_embs_stats);
        free(embeddings);
        free_lines(lines, line_count);
        return 1;
    }

    /* Pre-populate from loaded index as reference corpus */
    if (load_idx_path) {
        uint8_t *ref_emb = NULL;
        int ref_count = 0;
        if (load_index(load_idx_path, &ref_emb, &ref_count) != 0) {
            dedup_index_free(&idx);
            free(s2_embs_stats);
            free(embeddings);
            free_lines(lines, line_count);
            return 1;
        }
        for (int j = 0; j < ref_count; j++) {
            if (dedup_index_add(&idx, ref_emb + (size_t)j * DEDUP_DIMS) != 0) {
                fprintf(stderr, "trine_dedup: index allocation failed "
                        "loading reference corpus\n");
                free(ref_emb);
                dedup_index_free(&idx);
                free(s2_embs_stats);
                free(embeddings);
                free_lines(lines, line_count);
                return 1;
            }
        }
        free(ref_emb);
        if (verbose) {
            fprintf(stderr, "Loaded %d reference entries from '%s'\n",
                    ref_count, load_idx_path);
        }
    }

    int unique_count = 0;
    int dup_count = 0;
    double sim_sum = 0.0;
    double sim_max = 0.0;
    int sim_count = 0;  /* Number of pairwise comparisons that were above 0 */

    /* Track which "cluster" each duplicate maps to (the index entry
     * it matched against). We count distinct cluster IDs to report
     * the number of duplicate clusters. */
    int *cluster_id = (int *)calloc((size_t)line_count, sizeof(int));
    if (!cluster_id) {
        fprintf(stderr, "trine_dedup: allocation failed\n");
        dedup_index_free(&idx);
        free(s2_embs_stats);
        free(embeddings);
        free_lines(lines, line_count);
        return 1;
    }

    /* -1 = unique (becomes its own cluster root) */
    for (int i = 0; i < line_count; i++) cluster_id[i] = -1;

    for (int i = 0; i < line_count; i++) {
        const uint8_t *emb = embeddings + (size_t)i * DEDUP_DIMS;
        const uint8_t *s2e = s2_embs_stats
            ? s2_embs_stats + (size_t)i * DEDUP_DIMS : NULL;

        int best_i = -1;
        double max_sim = s2_model
            ? dedup_index_max_sim_blended(&idx, emb, s2e, lens,
                                          use_idf, &best_i)
            : (use_idf
                ? dedup_index_max_sim_idf(&idx, emb, lens, &best_i)
                : dedup_index_max_sim(&idx, emb, lens, &best_i));

        if (max_sim >= (double)threshold) {
            dup_count++;
            cluster_id[i] = best_i;
        } else {
            unique_count++;
            cluster_id[i] = idx.count;  /* its own cluster root index */
            if (dedup_index_add(&idx, emb) != 0) {
                fprintf(stderr, "trine_dedup: index allocation failed\n");
                break;
            }
            if (s2_model && s2e)
                dedup_index_add_s2(&idx, s2e);
        }

        /* Accumulate similarity stats for all pairwise max-sims
         * (even for unique lines, the max_sim against existing entries
         * is informative) */
        if (idx.count > 1 || (idx.count == 1 && max_sim > 0.0)) {
            sim_sum += max_sim;
            sim_count++;
            if (max_sim > sim_max) sim_max = max_sim;
        }
    }

    /* Count distinct clusters that have at least one duplicate */
    int cluster_count = 0;
    {
        /* A cluster exists if any duplicate maps to a given root.
         * Track which roots have been seen. */
        int *seen = (int *)calloc((size_t)(unique_count + 1), sizeof(int));
        if (seen) {
            for (int i = 0; i < line_count; i++) {
                if (cluster_id[i] >= 0 && cluster_id[i] < unique_count) {
                    /* Check if this line was a duplicate (matched an index entry) */
                    /* Duplicates have cluster_id pointing to the index entry
                     * they matched. We need to check if this line was NOT the
                     * one that created that index entry. */
                }
            }
            /* Simpler approach: a cluster is a unique line that has at
             * least one duplicate pointing to it. */
            for (int i = 0; i < line_count; i++) {
                /* This line is a duplicate if its cluster_id corresponds
                 * to an index entry that it didn't create. We can detect
                 * this because duplicates increment dup_count, and we
                 * track that per-line. Easier: re-scan and mark. */
            }
            free(seen);
        }

        /* Rebuild cluster count more directly:
         * Run the same pass again, tracking which unique index IDs
         * receive at least one duplicate match. */
        int *has_dup = (int *)calloc((size_t)(unique_count + 1), sizeof(int));
        if (has_dup) {
            /* Re-run a lighter pass */
            dedup_index_t idx2;
            if (dedup_index_init(&idx2) == 0) {
                for (int i = 0; i < line_count; i++) {
                    const uint8_t *emb = embeddings + (size_t)i * DEDUP_DIMS;
                    const uint8_t *s2e = s2_embs_stats
                        ? s2_embs_stats + (size_t)i * DEDUP_DIMS : NULL;
                    int bi = -1;
                    double ms = s2_model
                        ? dedup_index_max_sim_blended(&idx2, emb, s2e,
                                                      lens, use_idf, &bi)
                        : (use_idf
                            ? dedup_index_max_sim_idf(&idx2, emb, lens, &bi)
                            : dedup_index_max_sim(&idx2, emb, lens, &bi));

                    if (ms >= (double)threshold) {
                        /* This is a dup; mark its match as having dups */
                        if (bi >= 0 && bi < unique_count)
                            has_dup[bi] = 1;
                    } else {
                        dedup_index_add(&idx2, emb);
                        if (s2_model && s2e)
                            dedup_index_add_s2(&idx2, s2e);
                    }
                }
                dedup_index_free(&idx2);

                for (int i = 0; i < unique_count; i++) {
                    if (has_dup[i]) cluster_count++;
                }
            }
            free(has_dup);
        }
    }

    /* Output */
    double mean_sim = sim_count > 0 ? sim_sum / (double)sim_count : 0.0;

    if (json_output) {
        printf("{\"command\": \"stats\", \"entries\": %d, "
               "\"index_file\": \"%s\", "
               "\"threshold\": %.2f, \"lens\": \"%s\"}\n",
               unique_count,
               file_path,
               threshold, lens->name);
    } else if (quiet) {
        printf("%d %d %d %.3f %.3f\n",
               line_count, unique_count, dup_count, mean_sim, sim_max);
    } else {
        printf("TRINE Corpus Statistics\n");
        printf("  file:         %s\n", file_path);
        printf("  total lines:  %d\n", line_count);
        printf("  unique:       %d\n", unique_count);
        printf("  duplicates:   %d\n", dup_count);
        printf("  clusters:     %d  (unique lines with >= 1 duplicate)\n",
               cluster_count);
        printf("  reduction:    %.1f%%\n",
               line_count > 0
                   ? 100.0 * (double)dup_count / (double)line_count
                   : 0.0);
        printf("  mean sim:     %.3f  (average max-similarity per line)\n",
               mean_sim);
        printf("  max sim:      %.3f\n", sim_max);
        printf("  threshold:    %.2f\n", threshold);
        printf("  lens:         %s  (%.1f, %.1f, %.1f, %.1f)\n",
               lens->name,
               (double)lens->weights[0], (double)lens->weights[1],
               (double)lens->weights[2], (double)lens->weights[3]);
        if (s2_model)
            printf("  stage2:       %s  depth=%d  blend=%.2f%s  proj=%s\n",
                   s2_model_path, s2_depth, blend_alpha,
                   s2_only ? " (s2-only)" : "", s2_proj_mode_name());
    }

    free(cluster_id);
    dedup_index_free(&idx);
    free(s2_embs_stats);
    free(embeddings);
    free_lines(lines, line_count);
    return 0;
}

/* =====================================================================
 * Usage / Help
 * ===================================================================== */

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "TRINE Dedup v%s — Near-Duplicate Detection & Deduplication\n"
        "\n"
        "Usage:\n"
        "  %s check       [options] <text1> <text2>\n"
        "  %s scan        [options] < input.txt > unique.txt\n"
        "  %s batch       [options] -f <file>\n"
        "  %s build-index [options] -f <file>      (alias for batch)\n"
        "  %s stats       [options] -f <file>\n"
        "\n"
        "Options:\n"
        "  -t, --threshold FLOAT    Similarity threshold (default: 0.60)\n"
        "  -l, --lens SPEC          Lens preset or custom weights (see below)\n"
        "  -v, --verbose            Show detailed output\n"
        "  -q, --quiet              Minimal output\n"
        "  --detail                 Show per-chain breakdown\n"
        "  --idf                    Use IDF-weighted cosine comparison\n"
        "  --json                   Output machine-readable JSON\n"
        "  --input-format FORMAT    Input format: plain, jsonl, auto (default: auto)\n"
        "  --routed                 Use band-LSH routing for faster dedup\n"
        "  --recall MODE            Set routing recall mode: fast, balanced (default), strict\n"
        "                           Only applies when --routed is also set\n"
        "  --save-index PATH        Save dedup index to .tridx/.trrt file\n"
        "  --load-index PATH        Load pre-existing index (scan/batch/stats)\n"
        "  --append-index PATH      Load index, append new docs, save back (implies --routed)\n"
        "  --csidf                  Use corpus-specific IDF weighting (computed from index)\n"
        "  --fields SPEC            Field-aware mode: auto, title,body,code (JSONL input)\n"
        "  --field-weights SPEC     Field weights: \"title=1.0,body=0.8,code=1.2\"\n"
        "  --semantic PATH          Load Stage-2 .trine2 model for semantic dedup\n"
        "  --s2-model PATH          Alias for --semantic\n"
        "  --s2-depth N             Cascade depth for Stage-2 (default: 0)\n"
        "  --blend ALPHA            Blend factor: alpha*S1 + (1-alpha)*S2 (default: 0.65)\n"
        "  --s2-only                Use Stage-2 similarity only (no S1 blending)\n"
        "  --block-diagonal         Use random block-diagonal model (K=3, 4x60x60, for testing)\n"
        "\n"
        "Commands:\n"
        "  check         Compare two texts for duplication\n"
        "  scan          Stream dedup: output only unique lines from stdin\n"
        "  batch         Pairwise similarity matrix for a file\n"
        "  build-index   Alias for batch (intuitive name for indexing workflows)\n"
        "  stats         Corpus duplicate density analysis\n"
        "\n"
        "JSONL Input Format:\n"
        "  When --input-format jsonl is set (or auto-detected), each line\n"
        "  is a JSON object with at minimum a \"text\" field:\n"
        "    {\"id\": \"doc001\", \"text\": \"the text to encode\", \"meta\": {...}}\n"
        "  The \"text\" field is encoded. \"id\" becomes the index tag.\n"
        "  If no \"id\", line number is used. Auto-detect: if first non-blank\n"
        "  line starts with '{', JSONL is assumed.\n"
        "\n"
        "Lens Presets:\n"
        "  dedup    (0.5, 0.5, 0.7, 1.0)  — balanced deduplication (default)\n"
        "  edit     (1.0, 0.3, 0.1, 0.0)  — character-level edit detection\n"
        "  morph    (0.3, 1.0, 0.5, 0.2)  — morphological similarity\n"
        "  phrase   (0.1, 0.3, 1.0, 0.5)  — phrase-level matching\n"
        "  vocab    (0.0, 0.2, 0.3, 1.0)  — vocabulary overlap\n"
        "  uniform  (1.0, 1.0, 1.0, 1.0)  — equal chain weighting\n"
        "  code     (1.0, 0.8, 0.4, 0.2)  — source code deduplication\n"
        "  legal    (0.2, 0.4, 1.0, 0.8)  — legal document matching\n"
        "  medical  (0.3, 1.0, 0.6, 0.5)  — medical text similarity\n"
        "  support  (0.2, 0.4, 0.7, 1.0)  — support ticket deduplication\n"
        "  policy   (0.1, 0.3, 1.0, 0.8)  — policy document matching\n"
        "  w,w,w,w  Custom weights (e.g. 0.5,0.5,1.0,0.8)\n"
        "\n"
        "Index Persistence:\n"
        "  --save-index saves the scan index to a binary file.\n"
        "  --load-index loads a pre-existing index before processing.\n"
        "  This enables incremental dedup across multiple runs.\n"
        "  For batch/stats, the loaded index serves as a reference corpus.\n"
        "  Default format: .tridx (TRDX magic). With --routed: .trrt (TRRT magic).\n"
        "  Index formats are NOT interchangeable between modes.\n"
        "\n"
        "Examples:\n"
        "  %s check \"the cat sat\" \"the cat sits\"\n"
        "  %s check --json \"the cat sat\" \"the cat sits\"\n"
        "  %s check -t 0.65 -l vocab \"hello world\" \"hello earth\"\n"
        "  %s check --idf --detail \"the cat sat\" \"the cat sits\"\n"
        "  cat corpus.txt | %s scan -t 0.70 > unique.txt\n"
        "  cat corpus.jsonl | %s scan --json --input-format jsonl\n"
        "  %s build-index -f corpus.jsonl --save-index corpus.tridx --json\n"
        "  cat batch1.txt | %s scan --save-index corpus.tridx > u1.txt\n"
        "  cat batch2.txt | %s scan --load-index corpus.tridx > u2.txt\n"
        "  %s batch -f texts.txt --detail --idf\n"
        "  %s stats -f corpus.txt --json\n"
        "  %s stats -f corpus.txt --load-index ref.tridx\n"
        "  cat corpus.txt | %s scan -t 0.70 --routed > unique.txt\n"
        "  cat corpus.txt | %s scan --routed --save-index idx.trrt > u.txt\n"
        "  %s check --semantic model.trine2 \"hello world\" \"hi world\"\n"
        "  cat corpus.txt | %s scan --semantic model.trine2 --blend 0.65\n"
        "  %s batch -f texts.txt --s2-model model.trine2 --s2-only\n",
        DEDUP_VERSION,
        prog, prog, prog, prog, prog,
        prog, prog, prog, prog, prog, prog, prog, prog, prog,
        prog, prog, prog, prog, prog,
        prog, prog, prog);
}

/* =====================================================================
 * main — Argument Parsing & Dispatch
 * ===================================================================== */

int main(int argc, char **argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* Check for --help / --version before command dispatch */
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }
    if (strcmp(argv[1], "--version") == 0) {
        printf("trine_dedup %s\n", DEDUP_VERSION);
        return 0;
    }

    /* First positional argument is the command */
    const char *command = argv[1];
    int is_check = (strcmp(command, "check") == 0);
    int is_scan  = (strcmp(command, "scan")  == 0);
    int is_batch = (strcmp(command, "batch") == 0
                    || strcmp(command, "build-index") == 0);
    int is_stats = (strcmp(command, "stats") == 0);

    if (!is_check && !is_scan && !is_batch && !is_stats) {
        fprintf(stderr, "trine_dedup: unknown command '%s'\n", command);
        fprintf(stderr, "Try '%s --help' for usage.\n", argv[0]);
        return 1;
    }

    /* Parse options */
    float threshold      = DEDUP_DEFAULT_THRESHOLD;
    dedup_lens_t lens    = LENS_DEDUP;
    int verbose          = 0;
    int quiet            = 0;
    int detail           = 0;
    int use_idf          = 0;
    int json_output      = 0;
    int input_format     = DEDUP_INPUT_AUTO;
    int use_routed       = 0;
    int recall_mode      = TRINE_RECALL_BALANCED;
    const char *file_path = NULL;
    const char *save_index_path = NULL;
    const char *load_index_path = NULL;
    const char *append_index_path = NULL;
    int use_csidf          = 0;
    const char *fields_spec = NULL;
    const char *field_weights_spec = NULL;

    /* Collect positional arguments (for check command) */
    const char *positional[64];
    int n_positional = 0;

    int i = 2;
    while (i < argc) {
        const char *arg = argv[i];

        if (strcmp(arg, "-t") == 0 || strcmp(arg, "--threshold") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: %s requires an argument\n",
                        arg);
                return 1;
            }
            threshold = (float)atof(argv[i + 1]);
            if (threshold < 0.0f || threshold > 1.0f) {
                fprintf(stderr, "trine_dedup: threshold must be in "
                        "[0.0, 1.0], got %.3f\n", threshold);
                return 1;
            }
            i += 2;
        } else if (strcmp(arg, "-l") == 0 || strcmp(arg, "--lens") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: %s requires an argument\n",
                        arg);
                return 1;
            }
            if (!parse_lens(argv[i + 1], &lens)) {
                fprintf(stderr, "trine_dedup: invalid lens '%s'\n"
                        "  Use: edit|morph|phrase|vocab|dedup|uniform|"
                        "code|legal|medical|support|policy|w,w,w,w\n",
                        argv[i + 1]);
                return 1;
            }
            i += 2;
        } else if (strcmp(arg, "-f") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: -f requires a file path\n");
                return 1;
            }
            file_path = argv[i + 1];
            i += 2;
        } else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            verbose = 1;
            i++;
        } else if (strcmp(arg, "-q") == 0 || strcmp(arg, "--quiet") == 0) {
            quiet = 1;
            i++;
        } else if (strcmp(arg, "--detail") == 0) {
            detail = 1;
            i++;
        } else if (strcmp(arg, "--idf") == 0) {
            use_idf = 1;
            i++;
        } else if (strcmp(arg, "--json") == 0) {
            json_output = 1;
            i++;
        } else if (strcmp(arg, "--input-format") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: --input-format requires an "
                        "argument (plain, jsonl, auto)\n");
                return 1;
            }
            const char *fmt = argv[i + 1];
            if (strcmp(fmt, "plain") == 0) {
                input_format = DEDUP_INPUT_PLAIN;
            } else if (strcmp(fmt, "jsonl") == 0) {
                input_format = DEDUP_INPUT_JSONL;
            } else if (strcmp(fmt, "auto") == 0) {
                input_format = DEDUP_INPUT_AUTO;
            } else {
                fprintf(stderr, "trine_dedup: invalid input format '%s'\n"
                        "  Use: plain, jsonl, auto\n", fmt);
                return 1;
            }
            i += 2;
        } else if (strcmp(arg, "--routed") == 0) {
            use_routed = 1;
            i++;
        } else if (strcmp(arg, "--recall") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: --recall requires an argument "
                        "(fast, balanced, strict)\n");
                return 1;
            }
            const char *rm = argv[i + 1];
            if (strcmp(rm, "fast") == 0) {
                recall_mode = TRINE_RECALL_FAST;
            } else if (strcmp(rm, "balanced") == 0) {
                recall_mode = TRINE_RECALL_BALANCED;
            } else if (strcmp(rm, "strict") == 0) {
                recall_mode = TRINE_RECALL_STRICT;
            } else {
                fprintf(stderr, "trine_dedup: invalid recall mode '%s'\n"
                        "  Use: fast, balanced, strict\n", rm);
                return 1;
            }
            i += 2;
        } else if (strcmp(arg, "--save-index") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: --save-index requires a path\n");
                return 1;
            }
            save_index_path = argv[i + 1];
            i += 2;
        } else if (strcmp(arg, "--load-index") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: --load-index requires a path\n");
                return 1;
            }
            load_index_path = argv[i + 1];
            i += 2;
        } else if (strcmp(arg, "--append-index") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: --append-index requires a path\n");
                return 1;
            }
            append_index_path = argv[i + 1];
            use_routed = 1;  /* append requires routed mode */
            i += 2;
        } else if (strcmp(arg, "--csidf") == 0) {
            use_csidf = 1;
            i++;
        } else if (strcmp(arg, "--fields") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: --fields requires an argument "
                        "(auto, title,body,code, etc.)\n");
                return 1;
            }
            fields_spec = argv[i + 1];
            i += 2;
        } else if (strcmp(arg, "--field-weights") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: --field-weights requires an argument "
                        "(e.g., \"title=1.0,body=0.8,code=1.2\")\n");
                return 1;
            }
            field_weights_spec = argv[i + 1];
            i += 2;
        } else if (strcmp(arg, "--semantic") == 0 ||
                   strcmp(arg, "--s2-model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: %s requires a .trine2 model path\n",
                        arg);
                return 1;
            }
            s2_model_path = argv[i + 1];
            i += 2;
        } else if (strcmp(arg, "--s2-depth") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: --s2-depth requires an integer\n");
                return 1;
            }
            s2_depth = atoi(argv[i + 1]);
            if (s2_depth < 0) {
                fprintf(stderr, "trine_dedup: --s2-depth must be >= 0, got %d\n",
                        s2_depth);
                return 1;
            }
            i += 2;
        } else if (strcmp(arg, "--blend") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_dedup: --blend requires a float in [0,1]\n");
                return 1;
            }
            blend_alpha = (float)atof(argv[i + 1]);
            if (blend_alpha < 0.0f || blend_alpha > 1.0f) {
                fprintf(stderr, "trine_dedup: --blend must be in [0.0, 1.0], "
                        "got %.3f\n", blend_alpha);
                return 1;
            }
            i += 2;
        } else if (strcmp(arg, "--s2-only") == 0) {
            s2_only = 1;
            i++;
        } else if (strcmp(arg, "--block-diagonal") == 0) {
            s2_block_diag = 1;
            i++;
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] == '-' && arg[1] != '\0') {
            fprintf(stderr, "trine_dedup: unknown option '%s'\n", arg);
            fprintf(stderr, "Try '%s --help' for usage.\n", argv[0]);
            return 1;
        } else {
            /* Positional argument */
            if (n_positional < 64)
                positional[n_positional++] = arg;
            i++;
        }
    }

    /* Phase 4: Apply field config if specified */
    trine_field_config_t field_cfg;
    trine_field_config_init(&field_cfg);
    if (fields_spec) {
        trine_field_config_parse_fields(&field_cfg, fields_spec);
    }
    if (field_weights_spec) {
        trine_field_config_parse_weights(&field_cfg, field_weights_spec);
    }
    (void)field_cfg;  /* Used by field-aware paths; suppress unused warning */

    /* ── Mutual exclusivity: --block-diagonal vs --semantic ─────── */
    if (s2_block_diag && s2_model_path) {
        fprintf(stderr, "trine_dedup: --block-diagonal and --semantic are "
                "mutually exclusive\n");
        return 1;
    }

    /* ── Stage-2 model loading ────────────────────────────────────── */
    if (s2_block_diag) {
        /* Create random block-diagonal model for testing */
        uint8_t *bw = malloc((size_t)TRINE_PROJECT_K * 4 * 60 * 60);
        if (!bw) {
            fprintf(stderr, "trine_dedup: allocation failed for "
                    "block-diagonal weights\n");
            return 1;
        }
        trine_projection_block_random(bw, TRINE_PROJECT_K, 42);
        s2_model = trine_s2_create_block_diagonal(bw, TRINE_PROJECT_K, 4, 42);
        free(bw);
        if (!s2_model) {
            fprintf(stderr, "trine_dedup: failed to create block-diagonal "
                    "model\n");
            return 1;
        }
        s2_model_path = "(block-diagonal-random)";
    } else if (s2_model_path) {
        s2_model = (trine_s2_model_t *)trine_s2_load(s2_model_path);
        if (!s2_model) {
            fprintf(stderr, "trine_dedup: failed to load Stage-2 model '%s'\n",
                    s2_model_path);
            return 1;
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Dispatch
     *
     * Uses ret + goto cleanup so the Stage-2 model is always freed.
     * ───────────────────────────────────────────────────────────────── */

    int ret = 1;

    if (is_check) {
        if (n_positional < 2) {
            fprintf(stderr,
                    "trine_dedup: check requires two text arguments\n"
                    "Usage: %s check [options] <text1> <text2>\n",
                    argv[0]);
            goto cleanup;
        }
        ret = cmd_check(positional[0], positional[1],
                        threshold, &lens, verbose, detail, use_idf,
                        json_output);
        goto cleanup;
    }

    if (is_scan) {
        /* If append mode, load existing index and add new entries */
        if (append_index_path) {
            trine_route_t *rt = trine_route_load(append_index_path);
            if (!rt) {
                fprintf(stderr, "trine_dedup: cannot load index '%s' for append\n",
                        append_index_path);
                goto cleanup;
            }

            /* Enable CS-IDF if requested and not already present */
            if (use_csidf) {
                trine_route_enable_csidf(rt);
            }

            trine_route_set_recall(rt, recall_mode);

            /* Process new entries from stdin */
            char buf[DEDUP_MAX_LINE];
            int total_new = 0;
            int new_unique = 0;
            int new_dup = 0;

            while (fgets(buf, sizeof(buf), stdin)) {
                size_t len = strlen(buf);
                while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
                    buf[--len] = '\0';
                if (len == 0) continue;

                total_new++;
                uint8_t emb[DEDUP_DIMS];
                if (encode_line(buf, len, emb) != 0) continue;

                trine_route_stats_t rstats = {0};
                trine_s1_result_t res;
                if (use_csidf && trine_route_get_csidf(rt)) {
                    res = trine_route_query_csidf(rt, emb, &rstats);
                } else {
                    res = trine_route_query(rt, emb, &rstats);
                }

                if (res.is_duplicate && res.similarity >= threshold) {
                    new_dup++;
                    if (verbose && !json_output) {
                        fprintf(stderr, "  DUP  sim=%.3f  \"%.*s\"\n",
                                (double)res.similarity,
                                (int)(len > 60 ? 60 : len), buf);
                    }
                } else {
                    new_unique++;
                    if (!json_output) printf("%s\n", buf);
                    trine_route_add(rt, emb, NULL);
                }
            }

            /* Recompute CS-IDF if enabled */
            if (use_csidf) {
                trine_route_compute_csidf(rt);
            }

            /* Save atomically */
            const char *save_path = save_index_path ? save_index_path
                                                    : append_index_path;
            if (trine_route_save_atomic(rt, save_path) != 0) {
                trine_route_free(rt);
                goto cleanup;
            }

            if (!quiet && !json_output) {
                fprintf(stderr, "\n--- append summary ---\n");
                fprintf(stderr, "  new lines:   %d\n", total_new);
                fprintf(stderr, "  appended:    %d\n", new_unique);
                fprintf(stderr, "  duplicates:  %d\n", new_dup);
                fprintf(stderr, "  total index: %d\n", trine_route_count(rt));
                fprintf(stderr, "  saved to:    %s\n", save_path);
            }

            trine_route_free(rt);
            ret = 0;
            goto cleanup;
        }

        ret = cmd_scan(threshold, &lens, verbose, quiet, use_idf,
                       use_routed, recall_mode,
                       save_index_path, load_index_path,
                       json_output, input_format);
        goto cleanup;
    }

    if (is_batch) {
        if (!file_path) {
            fprintf(stderr,
                    "trine_dedup: %s requires -f <file>\n"
                    "Usage: %s %s [options] -f <file>\n",
                    command, argv[0], command);
            goto cleanup;
        }

        /* Phase 4: CS-IDF enabled build-index path */
        if (use_csidf && use_routed && save_index_path) {
            /* Build a routed index with CS-IDF tracking */
            trine_s1_config_t rt_cfg = {
                .threshold = threshold,
                .lens = {{lens.weights[0], lens.weights[1],
                          lens.weights[2], lens.weights[3]}},
                .calibrate_length = 0
            };

            trine_route_t *rt = trine_route_create(&rt_cfg);
            if (!rt) {
                fprintf(stderr, "trine_dedup: routed index allocation failed\n");
                goto cleanup;
            }
            trine_route_set_recall(rt, recall_mode);
            trine_route_enable_csidf(rt);

            /* Determine input format */
            int eff_format = input_format;
            if (eff_format == DEDUP_INPUT_AUTO)
                eff_format = detect_input_format_file(file_path);

            int line_count = 0;
            char **lines = NULL;
            if (eff_format == DEDUP_INPUT_JSONL) {
                lines = load_lines_jsonl(file_path, &line_count, NULL);
            } else {
                lines = load_lines(file_path, &line_count);
            }
            if (!lines) { trine_route_free(rt); goto cleanup; }

            /* Encode and add all entries */
            for (int ei = 0; ei < line_count; ei++) {
                uint8_t emb[DEDUP_DIMS];
                (void)encode_line(lines[ei], strlen(lines[ei]), emb);
                trine_route_add(rt, emb, NULL);
            }

            /* Compute CS-IDF weights */
            trine_route_compute_csidf(rt);

            /* Save with atomic write */
            if (trine_route_save_atomic(rt, save_index_path) != 0) {
                free_lines(lines, line_count);
                trine_route_free(rt);
                goto cleanup;
            }

            if (!quiet) {
                const trine_csidf_t *csidf = trine_route_get_csidf(rt);
                fprintf(stderr, "Built CS-IDF routed index: %d entries",
                        trine_route_count(rt));
                if (csidf) {
                    /* Report noise floor reduction */
                    int low_weight = 0;
                    for (int wi = 0; wi < 240; wi++) {
                        if (csidf->weights[wi] < 0.5f) low_weight++;
                    }
                    fprintf(stderr, ", %d/240 channels downweighted", low_weight);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "  saved to: %s\n", save_index_path);
            }

            free_lines(lines, line_count);
            trine_route_free(rt);
            ret = 0;
            goto cleanup;
        }

        ret = cmd_batch(file_path, threshold, &lens,
                        verbose, quiet, detail, use_idf, use_routed,
                        recall_mode, load_index_path,
                        json_output, input_format, save_index_path);
        goto cleanup;
    }

    if (is_stats) {
        if (!file_path) {
            fprintf(stderr,
                    "trine_dedup: stats requires -f <file>\n"
                    "Usage: %s stats [options] -f <file>\n",
                    argv[0]);
            goto cleanup;
        }
        ret = cmd_stats(file_path, threshold, &lens, verbose, quiet,
                        use_idf, use_routed, recall_mode,
                        load_index_path,
                        json_output, input_format);
        goto cleanup;
    }

    /* Should not reach here */
    print_usage(argv[0]);

cleanup:
    if (s2_model) trine_s2_free(s2_model);
    return ret;
}
