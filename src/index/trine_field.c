/* =====================================================================
 * TRINE FIELD — Field-Aware Indexing — Implementation
 * =====================================================================
 *
 * Multi-field document encoding and scoring. Each field (title, body,
 * code) is independently encoded into its own 240-dim shingle embedding.
 * Scoring combines per-field cosine similarities with configurable
 * field weights.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -c trine_field.c -o trine_field.o
 *
 * ===================================================================== */

#include "trine_field.h"
#include "trine_encode.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <ctype.h>

/* =====================================================================
 * I. DOMAIN PRESET TABLES
 * ===================================================================== */

/* Preset field weights: [title, body, code, meta] */
static const float PRESET_WEIGHTS[3][TRINE_FIELD_MAX] = {
    /* CODE:    emphasize code/title, reduce body boilerplate */
    { 1.0f, 0.3f, 1.2f, 0.0f },
    /* SUPPORT: emphasize title/body, suppress signatures */
    { 1.0f, 0.8f, 0.3f, 0.0f },
    /* POLICY:  emphasize title/body, suppress headers/footers */
    { 1.0f, 0.9f, 0.0f, 0.0f },
};

/* =====================================================================
 * II. CONFIGURATION
 * ===================================================================== */

void trine_field_config_init(trine_field_config_t *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(trine_field_config_t));

    cfg->field_count = 3;
    strncpy(cfg->field_names[0], "title", TRINE_FIELD_NAME_LEN - 1);
    strncpy(cfg->field_names[1], "body",  TRINE_FIELD_NAME_LEN - 1);
    strncpy(cfg->field_names[2], "code",  TRINE_FIELD_NAME_LEN - 1);

    cfg->field_weights[0] = 1.0f;
    cfg->field_weights[1] = 1.0f;
    cfg->field_weights[2] = 1.0f;
    cfg->field_weights[3] = 0.0f;

    cfg->route_field = TRINE_FIELD_BODY;
}

int trine_field_config_preset(trine_field_config_t *cfg, int preset)
{
    if (!cfg) return -1;
    if (preset < 0 || preset > 2) return -1;

    trine_field_config_init(cfg);
    memcpy(cfg->field_weights, PRESET_WEIGHTS[preset],
           sizeof(float) * TRINE_FIELD_MAX);

    return 0;
}

int trine_field_config_parse_fields(trine_field_config_t *cfg,
                                     const char *spec)
{
    if (!cfg || !spec) return -1;

    if (strcmp(spec, "auto") == 0) {
        trine_field_config_init(cfg);
        return 0;
    }

    /* Parse comma-separated field names: "title,body,code" */
    memset(cfg->field_names, 0, sizeof(cfg->field_names));
    cfg->field_count = 0;

    const char *p = spec;
    while (*p && cfg->field_count < TRINE_FIELD_MAX) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0') break;

        /* Find end of field name */
        const char *start = p;
        while (*p && *p != ',') p++;
        size_t len = (size_t)(p - start);

        /* Trim trailing whitespace */
        while (len > 0 && (start[len - 1] == ' ' || start[len - 1] == '\t'))
            len--;

        if (len == 0) {
            if (*p == ',') p++;
            continue;
        }

        if (len >= TRINE_FIELD_NAME_LEN) len = TRINE_FIELD_NAME_LEN - 1;
        memcpy(cfg->field_names[cfg->field_count], start, len);
        cfg->field_names[cfg->field_count][len] = '\0';

        /* Set default weight */
        cfg->field_weights[cfg->field_count] = 1.0f;

        /* Detect route field */
        if (strncmp(cfg->field_names[cfg->field_count], "body", 4) == 0) {
            cfg->route_field = cfg->field_count;
        }

        cfg->field_count++;
        if (*p == ',') p++;
    }

    /* If no body field found, route on first field */
    if (cfg->field_count > 0) {
        int found_body = 0;
        for (int i = 0; i < cfg->field_count; i++) {
            if (strcmp(cfg->field_names[i], "body") == 0) {
                found_body = 1;
                break;
            }
        }
        if (!found_body) cfg->route_field = 0;
    }

    return cfg->field_count > 0 ? 0 : -1;
}

int trine_field_config_parse_weights(trine_field_config_t *cfg,
                                      const char *spec)
{
    if (!cfg || !spec) return -1;

    /* Parse "name=weight,name=weight,..." format */
    const char *p = spec;
    while (*p) {
        /* Skip whitespace */
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0') break;

        /* Find '=' separator */
        const char *name_start = p;
        while (*p && *p != '=') p++;
        if (*p != '=') return -1;

        size_t name_len = (size_t)(p - name_start);
        /* Trim trailing whitespace from name */
        while (name_len > 0 &&
               (name_start[name_len - 1] == ' ' ||
                name_start[name_len - 1] == '\t'))
            name_len--;

        p++; /* skip '=' */

        /* Parse weight value */
        char *endp;
        float weight = strtof(p, &endp);
        if (endp == p) return -1;
        p = endp;

        /* Skip comma */
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ',') p++;

        /* Find matching field and set weight */
        for (int i = 0; i < cfg->field_count; i++) {
            if (strlen(cfg->field_names[i]) == name_len &&
                strncmp(cfg->field_names[i], name_start, name_len) == 0) {
                cfg->field_weights[i] = weight;
                break;
            }
        }
    }

    return 0;
}

/* =====================================================================
 * III. ENCODING
 * ===================================================================== */

int trine_field_encode(const trine_field_config_t *cfg,
                       const char **texts, const size_t *lens,
                       trine_field_entry_t *out)
{
    if (!cfg || !out) return -1;

    memset(out, 0, sizeof(trine_field_entry_t));
    out->field_count = cfg->field_count;

    for (int f = 0; f < cfg->field_count; f++) {
        if (texts && texts[f] && lens && lens[f] > 0) {
            if (trine_encode_shingle(texts[f], lens[f], out->embeddings[f]) != 0)
                return -1;
        }
        /* else: embeddings[f] stays zeroed */
    }

    return 0;
}

const uint8_t *trine_field_route_embedding(const trine_field_config_t *cfg,
                                            const trine_field_entry_t *entry)
{
    if (!cfg || !entry) return NULL;

    int rf = cfg->route_field;
    if (rf < 0 || rf >= entry->field_count) rf = 0;

    return entry->embeddings[rf];
}

/* =====================================================================
 * IV. SIMILARITY
 * ===================================================================== */

/* Per-chain cosine over a 60-channel slice */
static float field_chain_cosine(const uint8_t *a, const uint8_t *b,
                                 int offset, int width)
{
    uint64_t dot_ab = 0;
    uint64_t mag_a  = 0;
    uint64_t mag_b  = 0;

    for (int i = offset; i < offset + width; i++) {
        uint64_t va = a[i];
        uint64_t vb = b[i];
        dot_ab += va * vb;
        mag_a  += va * va;
        mag_b  += vb * vb;
    }

    if (mag_a == 0 || mag_b == 0) return 0.0f;

    double denom = sqrt((double)mag_a) * sqrt((double)mag_b);
    if (denom == 0.0) return 0.0f;

    double sim = (double)dot_ab / denom;
    if (sim > 1.0) sim = 1.0;
    if (sim < 0.0) sim = 0.0;
    return (float)sim;
}

/* Full 240-dim cosine for a single field */
static float field_cosine_240(const uint8_t *a, const uint8_t *b)
{
    /* Use 4-chain weighted average with uniform weights */
    double sum = 0.0;
    int active = 0;

    for (int c = 0; c < 4; c++) {
        float cos_c = field_chain_cosine(a, b, c * 60, 60);
        sum += (double)cos_c;
        active++;
    }

    if (active == 0) return 0.0f;
    float result = (float)(sum / (double)active);
    if (result > 1.0f) result = 1.0f;
    if (result < 0.0f) result = 0.0f;
    return result;
}

float trine_field_cosine(const trine_field_entry_t *a,
                          const trine_field_entry_t *b,
                          const trine_field_config_t *cfg)
{
    if (!a || !b || !cfg) return 0.0f;

    int fc = cfg->field_count;
    if (fc <= 0) return 0.0f;
    if (a->field_count < fc) fc = a->field_count;
    if (b->field_count < fc) fc = b->field_count;

    double weighted_sum = 0.0;
    double weight_sum   = 0.0;

    for (int f = 0; f < fc; f++) {
        double w = (double)cfg->field_weights[f];
        if (w <= 0.0) continue;

        float cos_f = field_cosine_240(a->embeddings[f], b->embeddings[f]);
        weighted_sum += w * (double)cos_f;
        weight_sum   += w;
    }

    if (weight_sum == 0.0) return 0.0f;
    return (float)(weighted_sum / weight_sum);
}

float trine_field_cosine_idf(const trine_field_entry_t *a,
                              const trine_field_entry_t *b,
                              const trine_field_config_t *cfg,
                              const float idf_weights[240])
{
    if (!a || !b || !cfg || !idf_weights) return 0.0f;

    int fc = cfg->field_count;
    if (fc <= 0) return 0.0f;
    if (a->field_count < fc) fc = a->field_count;
    if (b->field_count < fc) fc = b->field_count;

    double weighted_sum = 0.0;
    double weight_sum   = 0.0;

    for (int f = 0; f < fc; f++) {
        double w = (double)cfg->field_weights[f];
        if (w <= 0.0) continue;

        /* IDF-weighted cosine for this field */
        float dot = 0.0f;
        float mag_a_f = 0.0f;
        float mag_b_f = 0.0f;

        for (int i = 0; i < TRINE_FIELD_DIMS; i++) {
            float iw = idf_weights[i];
            float ai = (float)a->embeddings[f][i];
            float bi = (float)b->embeddings[f][i];
            dot     += iw * ai * bi;
            mag_a_f += iw * ai * ai;
            mag_b_f += iw * bi * bi;
        }

        float denom = sqrtf(mag_a_f) * sqrtf(mag_b_f);
        float cos_f = 0.0f;
        if (denom >= 1e-12f) {
            cos_f = dot / denom;
            if (cos_f > 1.0f) cos_f = 1.0f;
            if (cos_f < 0.0f) cos_f = 0.0f;
        }

        weighted_sum += w * (double)cos_f;
        weight_sum   += w;
    }

    if (weight_sum == 0.0) return 0.0f;
    return (float)(weighted_sum / weight_sum);
}

/* =====================================================================
 * V. JSONL FIELD EXTRACTION
 * ===================================================================== */

/*
 * Minimal JSON string field finder. Searches for "key": "value" pattern.
 * Returns pointer to value start (after opening quote), sets *vlen.
 * Handles escaped quotes inside values.
 */
static const char *field_json_find(const char *json, size_t json_len,
                                    const char *key, size_t *vlen)
{
    if (!json || !key || !vlen) return NULL;

    size_t klen = strlen(key);
    const char *end = json + json_len;
    const char *p = json;

    while (p < end) {
        const char *q1 = (const char *)memchr(p, '"', (size_t)(end - p));
        if (!q1) return NULL;
        q1++;

        /* Check if this is our key */
        if ((size_t)(end - q1) >= klen + 1 &&
            memcmp(q1, key, klen) == 0 && q1[klen] == '"') {
            /* Found key — skip to value */
            const char *after_key = q1 + klen + 1;
            while (after_key < end && (*after_key == ' ' || *after_key == '\t'
                   || *after_key == ':'))
                after_key++;

            if (after_key >= end || *after_key != '"') {
                p = after_key;
                continue;
            }

            /* Parse string value — skip escaped quotes, respect bounds */
            const char *val_start = after_key + 1;
            const char *vp = val_start;
            while (vp < end) {
                if (*vp == '\\') {
                    vp++;               /* skip the backslash */
                    if (vp < end) vp++; /* skip the escaped char */
                    continue;
                }
                if (*vp == '"') break;
                vp++;
            }

            *vlen = (size_t)(vp - val_start);
            return val_start;
        }

        /* Skip past this string — handle escapes with bounds check */
        const char *q2 = q1;
        while (q2 < end) {
            if (*q2 == '\\') { q2++; if (q2 < end) q2++; continue; }
            if (*q2 == '"') break;
            q2++;
        }
        p = (q2 < end) ? q2 + 1 : end;
    }

    return NULL;
}

int trine_field_extract_jsonl(const char *jsonl_buf, size_t jsonl_len,
                               const trine_field_config_t *cfg,
                               const char **out_texts, size_t *out_lens,
                               char **out_tag)
{
    if (!jsonl_buf || !cfg || !out_texts || !out_lens) return 0;

    int found = 0;

    /* Zero outputs */
    for (int f = 0; f < cfg->field_count; f++) {
        out_texts[f] = NULL;
        out_lens[f]  = 0;
    }
    if (out_tag) *out_tag = NULL;

    /* Extract each configured field */
    for (int f = 0; f < cfg->field_count; f++) {
        size_t vlen = 0;
        const char *val = field_json_find(jsonl_buf, jsonl_len,
                                           cfg->field_names[f], &vlen);
        if (val && vlen > 0) {
            out_texts[f] = val;
            out_lens[f]  = vlen;
            found++;
        }
    }

    /* Extract "id" as tag */
    if (out_tag) {
        size_t id_len = 0;
        const char *id_val = field_json_find(jsonl_buf, jsonl_len,
                                              "id", &id_len);
        if (id_val && id_len > 0) {
            char *tag = (char *)malloc(id_len + 1);
            if (tag) {
                memcpy(tag, id_val, id_len);
                tag[id_len] = '\0';
                *out_tag = tag;
            }
        }
    }

    /* Fallback: if no configured fields found, try "text" field as body */
    if (found == 0) {
        size_t text_len = 0;
        const char *text_val = field_json_find(jsonl_buf, jsonl_len,
                                                "text", &text_len);
        if (text_val && text_len > 0) {
            /* Put it in the body field (or first field if no body) */
            int body_idx = -1;
            for (int f = 0; f < cfg->field_count; f++) {
                if (strcmp(cfg->field_names[f], "body") == 0) {
                    body_idx = f;
                    break;
                }
            }
            if (body_idx < 0) body_idx = 0;

            if (body_idx < cfg->field_count) {
                out_texts[body_idx] = text_val;
                out_lens[body_idx]  = text_len;
                found = 1;
            }
        }
    }

    return found;
}

/* =====================================================================
 * VI. SERIALIZATION
 * ===================================================================== */

int trine_field_config_write(const trine_field_config_t *cfg, void *fp_void)
{
    FILE *fp = (FILE *)fp_void;
    if (!cfg || !fp) return -1;

    int32_t fc = (int32_t)cfg->field_count;
    if (fwrite(&fc, sizeof(int32_t), 1, fp) != 1) return -1;

    for (int i = 0; i < TRINE_FIELD_MAX; i++) {
        if (fwrite(cfg->field_names[i], 1, TRINE_FIELD_NAME_LEN, fp)
            != TRINE_FIELD_NAME_LEN)
            return -1;
    }

    if (fwrite(cfg->field_weights, sizeof(float), TRINE_FIELD_MAX, fp)
        != TRINE_FIELD_MAX)
        return -1;

    int32_t rf = (int32_t)cfg->route_field;
    if (fwrite(&rf, sizeof(int32_t), 1, fp) != 1) return -1;

    return 0;
}

int trine_field_config_read(trine_field_config_t *cfg, void *fp_void)
{
    FILE *fp = (FILE *)fp_void;
    if (!cfg || !fp) return -1;

    int32_t fc;
    if (fread(&fc, sizeof(int32_t), 1, fp) != 1) return -1;
    if (fc < 0 || fc > TRINE_FIELD_MAX) return -1;
    cfg->field_count = (int)fc;

    for (int i = 0; i < TRINE_FIELD_MAX; i++) {
        if (fread(cfg->field_names[i], 1, TRINE_FIELD_NAME_LEN, fp)
            != TRINE_FIELD_NAME_LEN)
            return -1;
    }

    if (fread(cfg->field_weights, sizeof(float), TRINE_FIELD_MAX, fp)
        != TRINE_FIELD_MAX)
        return -1;

    int32_t rf;
    if (fread(&rf, sizeof(int32_t), 1, fp) != 1) return -1;
    cfg->route_field = (int)rf;

    return 0;
}
