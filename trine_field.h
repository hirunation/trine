/* =====================================================================
 * TRINE FIELD — Field-Aware Indexing
 * Ternary Resonance Interference Network Embedding
 * =====================================================================
 *
 * OVERVIEW
 *   Supports structured indexing where documents have multiple fields
 *   (title, body, code, etc.). Each field is independently encoded
 *   into its own 240-dim shingle embedding, and scoring combines
 *   per-field similarities via configurable weights.
 *
 * SCORING
 *   score = sum(w_f * cosine(q_f, d_f)) / sum(w_f)
 *   where f iterates over active fields and w_f is the field weight.
 *
 * ROUTING
 *   Routing uses the primary field (default: body, or first non-empty)
 *   for bucket hashing. This keeps routing simple while field-weighted
 *   scoring improves precision on the candidate set.
 *
 * DOMAIN PRESETS
 *   code:    title=1.0, body=0.3, code=1.2 — emphasize code/title
 *   support: title=1.0, body=0.8, code=0.3 — emphasize title/body
 *   policy:  title=1.0, body=0.9, code=0.0 — emphasize title/body
 *
 * ===================================================================== */

#ifndef TRINE_FIELD_H
#define TRINE_FIELD_H

#include <stdint.h>
#include <stddef.h>

#define TRINE_FIELD_MAX       4        /* Maximum number of fields */
#define TRINE_FIELD_NAME_LEN  32       /* Max field name length (including NUL) */
#define TRINE_FIELD_DIMS      240      /* Embedding dims per field */

/* Well-known field indices */
#define TRINE_FIELD_TITLE     0
#define TRINE_FIELD_BODY      1
#define TRINE_FIELD_CODE      2
#define TRINE_FIELD_META      3        /* Passthrough — not scored by default */

/* Domain presets */
#define TRINE_FIELD_PRESET_CODE    0
#define TRINE_FIELD_PRESET_SUPPORT 1
#define TRINE_FIELD_PRESET_POLICY  2

/* Field configuration */
typedef struct {
    int   field_count;                                /* Active field count (1-4) */
    char  field_names[TRINE_FIELD_MAX][TRINE_FIELD_NAME_LEN]; /* Field names */
    float field_weights[TRINE_FIELD_MAX];             /* Per-field scoring weights */
    int   route_field;                                /* Which field to use for routing (index into fields) */
} trine_field_config_t;

/* A single document's field embeddings */
typedef struct {
    uint8_t embeddings[TRINE_FIELD_MAX][TRINE_FIELD_DIMS]; /* Per-field embeddings */
    int     field_count;                                    /* Number of active fields */
} trine_field_entry_t;

/* Initialize a field config with default settings.
 * Default: 3 fields (title, body, code) with uniform weights (1.0 each).
 * Route field: body (index 1). */
void trine_field_config_init(trine_field_config_t *cfg);

/* Apply a domain preset to a field config.
 * Preset values: TRINE_FIELD_PRESET_CODE, _SUPPORT, _POLICY.
 * Returns 0 on success, -1 on invalid preset. */
int trine_field_config_preset(trine_field_config_t *cfg, int preset);

/* Parse a field spec string (e.g., "auto", "title,body,code", "title,body").
 * Sets field_count and field_names accordingly.
 * "auto" = default (title, body, code).
 * Returns 0 on success, -1 on error. */
int trine_field_config_parse_fields(trine_field_config_t *cfg,
                                     const char *spec);

/* Parse a field weight spec string (e.g., "title=1.0,body=0.8,code=1.2").
 * Updates field_weights for matching field names.
 * Returns 0 on success, -1 on error. */
int trine_field_config_parse_weights(trine_field_config_t *cfg,
                                      const char *spec);

/* Encode a document's fields into a field entry.
 * texts[i] and lens[i] correspond to field i in cfg->field_names.
 * Missing fields (NULL text or 0 length) produce zero embeddings.
 * field_count in the entry is set to cfg->field_count.
 * Returns 0 on success. */
int trine_field_encode(const trine_field_config_t *cfg,
                       const char **texts, const size_t *lens,
                       trine_field_entry_t *out);

/* Get the routing embedding for a field entry (the primary/route field).
 * Returns pointer to 240-byte embedding for the route field. */
const uint8_t *trine_field_route_embedding(const trine_field_config_t *cfg,
                                            const trine_field_entry_t *entry);

/* Field-weighted cosine similarity between two field entries.
 * score = sum(w_f * cosine(a_f, b_f)) / sum(w_f)
 * Only active fields with weight > 0 contribute. */
float trine_field_cosine(const trine_field_entry_t *a,
                          const trine_field_entry_t *b,
                          const trine_field_config_t *cfg);

/* Field-weighted cosine with CS-IDF weighting per channel.
 * Uses idf_weights[240] for per-channel weighting within each field. */
float trine_field_cosine_idf(const trine_field_entry_t *a,
                              const trine_field_entry_t *b,
                              const trine_field_config_t *cfg,
                              const float idf_weights[240]);

/* Extract fields from a JSONL line.
 * Looks for "title", "body", "code", "id" top-level string fields.
 * out_texts[i] and out_lens[i] are set for each field found.
 * out_tag is set to the "id" field value (caller must free).
 * out_texts entries are pointers into jsonl_buf (no allocation).
 * Returns the number of fields found (0 if none). */
int trine_field_extract_jsonl(const char *jsonl_buf, size_t jsonl_len,
                               const trine_field_config_t *cfg,
                               const char **out_texts, size_t *out_lens,
                               char **out_tag);

/* Serialize field config to a file stream.
 * Writes: field_count(4) + names(4*32) + weights(4*4) + route_field(4) = 152 bytes.
 * Returns 0 on success, -1 on error. */
int trine_field_config_write(const trine_field_config_t *cfg, void *fp);

/* Deserialize field config from a file stream.
 * Returns 0 on success, -1 on error. */
int trine_field_config_read(trine_field_config_t *cfg, void *fp);

/* Serialized size of field config */
#define TRINE_FIELD_CONFIG_SERIAL_SIZE \
    (4 + TRINE_FIELD_MAX * TRINE_FIELD_NAME_LEN + TRINE_FIELD_MAX * 4 + 4)

#endif /* TRINE_FIELD_H */
