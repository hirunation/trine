/* ═══════════════════════════════════════════════════════════════════════
 * TRINE — Ternary Resonance Interference Network Embedding
 * Stage-1 Retriever/Deduper API — Implementation
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Fast near-duplicate detection and candidate retrieval using
 * TRINE shingle (240-dim) embeddings.
 *
 * COMPARISON ALGORITHM
 *   Per-chain cosine similarity with lens weighting, identical to
 *   the algorithm in trine.c (trine_compare_lens / chain_cosine).
 *   Each of the 4 chains (60 channels) gets an independent cosine,
 *   then combined = sum(weight[i] * cosine[i]) / sum(weight[i]).
 *
 * LENGTH CALIBRATION
 *   Short texts activate fewer n-gram slots, producing sparse
 *   embeddings with lower baseline cosine. The calibrator divides
 *   raw cosine by sqrt(fill_a * fill_b), compensating for sparsity.
 *   Fill values below 0.05 are clamped to avoid division explosion.
 *
 * INDEX
 *   Simple dynamic array with linear scan. Realloc-based growth
 *   (start at 64, double on overflow). Each entry stores 240 bytes
 *   of embedding data plus an optional tag string.
 *
 * Build:
 *   cc -O2 -c trine_stage1.c -o trine_stage1.o
 *   cc -O2 -c trine_encode.c -o trine_encode.o
 *   ar rcs libtrine_s1.a trine_stage1.o trine_encode.o
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#include "trine_stage1.h"
#include "trine_encode.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════
 * I. INDEX STRUCTURE
 * ═══════════════════════════════════════════════════════════════════════ */

#define S1_INITIAL_CAPACITY 64
#define S1_FILL_CLAMP       0.05f

struct trine_s1_index {
    trine_s1_config_t config;
    uint8_t *embeddings;    /* Flat array: count * 240 bytes */
    char **tags;            /* Array of tag strings (may contain NULLs) */
    int count;
    int capacity;
};

/* ═══════════════════════════════════════════════════════════════════════
 * II. INTERNAL HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Per-chain cosine similarity over a 60-channel slice.
 *
 * Identical to chain_cosine() in trine.c: treats trit values as
 * real-valued vector components {0, 1, 2} and computes standard
 * cosine = dot(a,b) / (|a| * |b|) over the slice.
 *
 * Returns 0.0 if either vector has zero magnitude in the slice.
 */
static float s1_chain_cosine(const uint8_t *a, const uint8_t *b,
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

    /* Clamp to [0, 1] to handle floating-point rounding */
    if (sim > 1.0) sim = 1.0;
    if (sim < 0.0) sim = 0.0;

    return (float)sim;
}

/*
 * Lens-weighted cosine similarity over the full 240-channel embedding.
 *
 * Computes cosine independently for each of the 4 chains (60 channels
 * each), then combines with lens weights:
 *   combined = sum(weight[i] * chain_cosine[i]) / sum(weight[i])
 *
 * Chains with weight <= 0.0 are skipped entirely.
 * Returns 0.0 if total weight is zero.
 */
static float s1_lens_cosine(const uint8_t *a, const uint8_t *b,
                             const trine_s1_lens_t *lens)
{
    double weighted_sum = 0.0;
    double weight_sum   = 0.0;

    for (int c = 0; c < TRINE_S1_CHAINS; c++) {
        double w = (double)lens->weights[c];
        if (w <= 0.0) continue;

        float cos_c = s1_chain_cosine(a, b,
                                       c * TRINE_S1_CHAIN_WIDTH,
                                       TRINE_S1_CHAIN_WIDTH);
        weighted_sum += w * (double)cos_c;
        weight_sum   += w;
    }

    if (weight_sum == 0.0) return 0.0f;

    return (float)(weighted_sum / weight_sum);
}

/* ═══════════════════════════════════════════════════════════════════════
 * III. ENCODING API
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_s1_encode(const char *text, size_t len, uint8_t out[240])
{
    if (!out) return -1;

    if (!text || len == 0) {
        memset(out, 0, TRINE_S1_DIMS);
        return 0;
    }

    if (trine_encode_shingle(text, len, out) != 0)
        return -1;
    return 0;
}

int trine_s1_encode_batch(const char **texts, const size_t *lens,
                           int count, uint8_t *out)
{
    if (!texts || !lens || !out || count <= 0) return -1;

    for (int i = 0; i < count; i++) {
        int rc = trine_s1_encode(texts[i], lens[i],
                                  out + (size_t)i * TRINE_S1_DIMS);
        if (rc != 0) return rc;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * IV. COMPARISON API
 * ═══════════════════════════════════════════════════════════════════════ */

float trine_s1_compare(const uint8_t a[240], const uint8_t b[240],
                        const trine_s1_lens_t *lens)
{
    if (!a || !b || !lens) return -1.0f;

    return s1_lens_cosine(a, b, lens);
}

trine_s1_result_t trine_s1_check(const uint8_t candidate[240],
                                  const uint8_t reference[240],
                                  const trine_s1_config_t *config)
{
    trine_s1_result_t result;
    memset(&result, 0, sizeof(result));
    result.matched_index = -1;

    if (!candidate || !reference || !config) return result;

    /* Compute raw lens-weighted cosine */
    float raw = s1_lens_cosine(candidate, reference, &config->lens);
    result.similarity = raw;

    /* Length calibration */
    if (config->calibrate_length) {
        float fill_a = trine_s1_fill_ratio(candidate);
        float fill_b = trine_s1_fill_ratio(reference);
        result.calibrated = trine_s1_calibrate(raw, fill_a, fill_b);
    } else {
        result.calibrated = raw;
    }

    /* Threshold check against calibrated score (if calibration enabled)
     * or raw score (if calibration disabled) */
    float score = config->calibrate_length ? result.calibrated : result.similarity;
    result.is_duplicate = (score >= config->threshold) ? 1 : 0;

    return result;
}

int trine_s1_compare_batch(const uint8_t candidate[240],
                            const uint8_t *refs, int ref_count,
                            const trine_s1_config_t *config,
                            float *best_sim)
{
    if (!candidate || !refs || ref_count <= 0 || !config) return -1;

    float fill_cand = 0.0f;
    if (config->calibrate_length) {
        fill_cand = trine_s1_fill_ratio(candidate);
    }

    float best_score = -1.0f;
    int   best_idx   = -1;

    for (int i = 0; i < ref_count; i++) {
        const uint8_t *ref = refs + (size_t)i * TRINE_S1_DIMS;

        float raw = s1_lens_cosine(candidate, ref, &config->lens);

        float score;
        if (config->calibrate_length) {
            float fill_ref = trine_s1_fill_ratio(ref);
            score = trine_s1_calibrate(raw, fill_cand, fill_ref);
        } else {
            score = raw;
        }

        if (score > best_score) {
            best_score = score;
            best_idx   = i;
        }
    }

    /* Apply threshold */
    if (best_score < config->threshold) {
        if (best_sim) *best_sim = best_score;
        return -1;
    }

    if (best_sim) *best_sim = best_score;
    return best_idx;
}

/* ═══════════════════════════════════════════════════════════════════════
 * V. INDEX API
 * ═══════════════════════════════════════════════════════════════════════ */

trine_s1_index_t *trine_s1_index_create(const trine_s1_config_t *config)
{
    trine_s1_index_t *idx = (trine_s1_index_t *)calloc(1, sizeof(trine_s1_index_t));
    if (!idx) return NULL;

    if (config) {
        idx->config = *config;
    } else {
        trine_s1_config_t default_config = TRINE_S1_CONFIG_DEFAULT;
        idx->config = default_config;
    }

    idx->capacity = S1_INITIAL_CAPACITY;
    idx->count = 0;

    idx->embeddings = (uint8_t *)malloc(
        (size_t)idx->capacity * TRINE_S1_DIMS);
    if (!idx->embeddings) {
        free(idx);
        return NULL;
    }

    idx->tags = (char **)calloc((size_t)idx->capacity, sizeof(char *));
    if (!idx->tags) {
        free(idx->embeddings);
        free(idx);
        return NULL;
    }

    return idx;
}

int trine_s1_index_add(trine_s1_index_t *idx, const uint8_t emb[240],
                        const char *tag)
{
    if (!idx || !emb) return -1;

    /* Grow if at capacity: double the allocation.
     * Both reallocs must succeed before updating the struct to avoid
     * leaving the index in an inconsistent state (embeddings and tags
     * arrays with different capacities). */
    if (idx->count >= idx->capacity) {
        int new_cap = idx->capacity * 2;

        uint8_t *new_emb = (uint8_t *)realloc(
            idx->embeddings, (size_t)new_cap * TRINE_S1_DIMS);
        if (!new_emb) return -1;

        char **new_tags = (char **)realloc(
            idx->tags, (size_t)new_cap * sizeof(char *));
        if (!new_tags) {
            /* Tags realloc failed. The embeddings realloc succeeded,
             * but we must NOT update idx->embeddings yet — however,
             * realloc may have moved the block. Assign the new pointer
             * back so the existing data remains accessible, but do NOT
             * update capacity (the index stays at its old size). */
            idx->embeddings = new_emb;
            return -1;
        }

        /* Both succeeded — commit the new pointers and capacity */
        idx->embeddings = new_emb;
        idx->tags = new_tags;

        /* Zero the new tag slots */
        for (int i = idx->capacity; i < new_cap; i++)
            idx->tags[i] = NULL;

        idx->capacity = new_cap;
    }

    /* Copy embedding into the flat array */
    memcpy(idx->embeddings + (size_t)idx->count * TRINE_S1_DIMS,
           emb, TRINE_S1_DIMS);

    /* Copy tag string (or set NULL) */
    if (tag) {
        size_t tag_len = strlen(tag);
        idx->tags[idx->count] = (char *)malloc(tag_len + 1);
        if (idx->tags[idx->count]) {
            memcpy(idx->tags[idx->count], tag, tag_len + 1);
        }
        /* If malloc fails for tag, we still add the entry with NULL tag */
    } else {
        idx->tags[idx->count] = NULL;
    }

    return idx->count++;
}

trine_s1_result_t trine_s1_index_query(const trine_s1_index_t *idx,
                                        const uint8_t candidate[240])
{
    trine_s1_result_t result;
    memset(&result, 0, sizeof(result));
    result.matched_index = -1;

    if (!idx || !candidate || idx->count == 0) return result;

    float fill_cand = 0.0f;
    if (idx->config.calibrate_length) {
        fill_cand = trine_s1_fill_ratio(candidate);
    }

    float best_score = -1.0f;
    float best_raw   = 0.0f;
    float best_cal   = 0.0f;
    int   best_idx   = -1;

    for (int i = 0; i < idx->count; i++) {
        const uint8_t *entry = idx->embeddings + (size_t)i * TRINE_S1_DIMS;

        /* Raw lens-weighted cosine */
        float raw = s1_lens_cosine(candidate, entry, &idx->config.lens);

        /* Calibrate if enabled */
        float cal;
        if (idx->config.calibrate_length) {
            float fill_entry = trine_s1_fill_ratio(entry);
            cal = trine_s1_calibrate(raw, fill_cand, fill_entry);
        } else {
            cal = raw;
        }

        /* Use calibrated score for ranking when calibration is on,
         * raw score otherwise */
        float score = idx->config.calibrate_length ? cal : raw;

        if (score > best_score) {
            best_score = score;
            best_raw   = raw;
            best_cal   = cal;
            best_idx   = i;
        }
    }

    result.similarity = best_raw;
    result.calibrated = best_cal;

    /* Apply threshold */
    if (best_score >= idx->config.threshold) {
        result.is_duplicate = 1;
        result.matched_index = best_idx;
    }

    return result;
}

int trine_s1_index_count(const trine_s1_index_t *idx)
{
    if (!idx) return 0;
    return idx->count;
}

const char *trine_s1_index_tag(const trine_s1_index_t *idx, int index)
{
    if (!idx || index < 0 || index >= idx->count) return NULL;
    return idx->tags[index];
}

void trine_s1_index_free(trine_s1_index_t *idx)
{
    if (!idx) return;

    /* Free all tag strings */
    for (int i = 0; i < idx->count; i++) {
        free(idx->tags[i]);
    }

    free(idx->tags);
    free(idx->embeddings);
    free(idx);
}

/* ═══════════════════════════════════════════════════════════════════════
 * VI. INDEX PERSISTENCE (save/load)
 * ═══════════════════════════════════════════════════════════════════════ */

#define S1_FILE_VERSION   2       /* Current version (v2: endian + flags + checksum) */
#define S1_FILE_VERSION_V1 1     /* Can still load v1 files (no endian/flags/checksum) */

/* ═══════════════════════════════════════════════════════════════════════
 * FNV-1a 64-bit hash (for payload checksum)
 * ═══════════════════════════════════════════════════════════════════════ */

static uint64_t trine_fnv1a_64(const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(0x00000100000001b1);
    }
    return h;
}

/*
 * v2 header layout (written after magic):
 *   uint32_t  version       = 2
 *   uint32_t  endian_check  = 0x01020304
 *   uint32_t  flags         = 0 (reserved)
 *   uint64_t  checksum      = FNV-1a over payload bytes after header
 *
 * Total header: magic(4) + version(4) + endian(4) + flags(4) + checksum(8) = 24 bytes
 */
#define S1_ENDIAN_MARKER  0x01020304u
#define S1_V2_HEADER_SIZE 24     /* 4 + 4 + 4 + 4 + 8 */
#define S1_V2_CHECKSUM_OFFSET 16 /* offset of checksum field from file start */

int trine_s1_index_save(const trine_s1_index_t *idx, const char *path)
{
    if (!idx || !path) return -1;

    FILE *fp = fopen(path, "w+b");  /* w+b: read+write for checksum patching */
    if (!fp) return -1;

    /* Magic */
    if (fwrite(TRINE_S1_INDEX_MAGIC, 1, 4, fp) != 4) goto fail;

    /* Version */
    uint32_t version = S1_FILE_VERSION;
    if (fwrite(&version, sizeof(uint32_t), 1, fp) != 1) goto fail;

    /* Endianness marker (v2) */
    uint32_t endian_check = S1_ENDIAN_MARKER;
    if (fwrite(&endian_check, sizeof(uint32_t), 1, fp) != 1) goto fail;

    /* Feature flags (v2, reserved) */
    uint32_t flags = 0;
    if (fwrite(&flags, sizeof(uint32_t), 1, fp) != 1) goto fail;

    /* Checksum placeholder (v2) — patched after payload is written */
    uint64_t checksum = 0;
    if (fwrite(&checksum, sizeof(uint64_t), 1, fp) != 1) goto fail;

    /* --- payload starts here (byte offset S1_V2_HEADER_SIZE) --- */

    /* Count */
    uint32_t count = (uint32_t)idx->count;
    if (fwrite(&count, sizeof(uint32_t), 1, fp) != 1) goto fail;

    /* Config: threshold, 4 lens weights, calibrate_length */
    if (fwrite(&idx->config.threshold, sizeof(float), 1, fp) != 1) goto fail;
    if (fwrite(idx->config.lens.weights, sizeof(float), TRINE_S1_CHAINS, fp)
        != TRINE_S1_CHAINS) goto fail;
    int32_t cal = (int32_t)idx->config.calibrate_length;
    if (fwrite(&cal, sizeof(int32_t), 1, fp) != 1) goto fail;

    /* Entries */
    for (uint32_t i = 0; i < count; i++) {
        /* Embedding: 240 bytes */
        const uint8_t *emb = idx->embeddings + (size_t)i * TRINE_S1_DIMS;
        if (fwrite(emb, 1, TRINE_S1_DIMS, fp) != TRINE_S1_DIMS) goto fail;

        /* Tag length + tag data */
        if (idx->tags[i]) {
            uint32_t tag_len = (uint32_t)strlen(idx->tags[i]);
            if (fwrite(&tag_len, sizeof(uint32_t), 1, fp) != 1) goto fail;
            if (tag_len > 0) {
                if (fwrite(idx->tags[i], 1, tag_len, fp) != tag_len) goto fail;
            }
        } else {
            uint32_t tag_len = 0;
            if (fwrite(&tag_len, sizeof(uint32_t), 1, fp) != 1) goto fail;
        }
    }

    /* Compute checksum over the payload (everything after the header) */
    {
        long file_end = ftell(fp);
        if (file_end < 0) goto fail;

        size_t payload_size = (size_t)file_end - S1_V2_HEADER_SIZE;
        if (payload_size > 0) {
            uint8_t *payload_buf = (uint8_t *)malloc(payload_size);
            if (!payload_buf) goto fail;

            if (fseek(fp, S1_V2_HEADER_SIZE, SEEK_SET) != 0) {
                free(payload_buf);
                goto fail;
            }
            if (fread(payload_buf, 1, payload_size, fp) != payload_size) {
                free(payload_buf);
                goto fail;
            }

            checksum = trine_fnv1a_64(payload_buf, payload_size);
            free(payload_buf);
        }

        /* Seek back and patch the checksum field */
        if (fseek(fp, S1_V2_CHECKSUM_OFFSET, SEEK_SET) != 0) goto fail;
        if (fwrite(&checksum, sizeof(uint64_t), 1, fp) != 1) goto fail;
    }

    fclose(fp);
    return 0;

fail:
    fclose(fp);
    return -1;
}

trine_s1_index_t *trine_s1_index_load(const char *path)
{
    if (!path) return NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "trine_s1_index_load: cannot open '%s'\n", path);
        return NULL;
    }

    /* Magic */
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, TRINE_S1_INDEX_MAGIC, 4) != 0) {
        fprintf(stderr, "trine_s1_index_load: bad magic in '%s'\n", path);
        fclose(fp);
        return NULL;
    }

    /* Version */
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "trine_s1_index_load: truncated version in '%s'\n", path);
        fclose(fp);
        return NULL;
    }

    if (version > S1_FILE_VERSION) {
        fprintf(stderr,
                "trine_s1_index_load: unsupported version %u in '%s' "
                "(this build supports up to v%u)\n",
                version, path, (uint32_t)S1_FILE_VERSION);
        fclose(fp);
        return NULL;
    }

    /* v2 header fields: endianness marker, feature flags, checksum */
    long payload_offset = 4 + 4;  /* magic + version (v1 payload starts here) */
    uint64_t stored_checksum = 0;
    int have_checksum = 0;

    if (version >= 2) {
        /* Endianness marker */
        uint32_t endian_check;
        if (fread(&endian_check, sizeof(uint32_t), 1, fp) != 1) goto truncated;
        if (endian_check != S1_ENDIAN_MARKER) {
            fprintf(stderr,
                    "trine_s1_index_load: endianness mismatch in '%s' "
                    "(expected 0x%08X, got 0x%08X) — file may be from a "
                    "different architecture\n",
                    path, S1_ENDIAN_MARKER, endian_check);
            fclose(fp);
            return NULL;
        }

        /* Feature flags (reserved, ignored for now) */
        uint32_t flags;
        if (fread(&flags, sizeof(uint32_t), 1, fp) != 1) goto truncated;
        (void)flags;

        /* Payload checksum */
        if (fread(&stored_checksum, sizeof(uint64_t), 1, fp) != 1) goto truncated;
        have_checksum = 1;

        payload_offset = S1_V2_HEADER_SIZE;
    }

    /* If v2, verify checksum over the payload before parsing it */
    if (have_checksum) {
        long cur = ftell(fp);
        if (cur < 0) goto truncated;

        /* Determine file size */
        if (fseek(fp, 0, SEEK_END) != 0) goto truncated;
        long file_end = ftell(fp);
        if (file_end < 0) goto truncated;

        size_t payload_size = (size_t)(file_end - payload_offset);
        if (payload_size > 0) {
            uint8_t *payload_buf = (uint8_t *)malloc(payload_size);
            if (!payload_buf) {
                fprintf(stderr, "trine_s1_index_load: allocation failed for checksum\n");
                fclose(fp);
                return NULL;
            }

            if (fseek(fp, payload_offset, SEEK_SET) != 0) {
                free(payload_buf);
                goto truncated;
            }
            if (fread(payload_buf, 1, payload_size, fp) != payload_size) {
                free(payload_buf);
                goto truncated;
            }

            uint64_t computed = trine_fnv1a_64(payload_buf, payload_size);
            free(payload_buf);

            if (computed != stored_checksum) {
                fprintf(stderr,
                        "trine_s1_index_load: WARNING — checksum mismatch in '%s' "
                        "(stored 0x%016llx, computed 0x%016llx). "
                        "File may be corrupted. Loading anyway.\n",
                        path,
                        (unsigned long long)stored_checksum,
                        (unsigned long long)computed);
            }
        }

        /* Seek back to start of payload for normal parsing */
        if (fseek(fp, payload_offset, SEEK_SET) != 0) goto truncated;
    }

    /* Count */
    uint32_t count;
    if (fread(&count, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "trine_s1_index_load: truncated header in '%s'\n", path);
        fclose(fp);
        return NULL;
    }

    /* Config */
    trine_s1_config_t config;
    if (fread(&config.threshold, sizeof(float), 1, fp) != 1) goto truncated;
    if (fread(config.lens.weights, sizeof(float), TRINE_S1_CHAINS, fp)
        != TRINE_S1_CHAINS) goto truncated;
    int32_t cal;
    if (fread(&cal, sizeof(int32_t), 1, fp) != 1) goto truncated;
    config.calibrate_length = (int)cal;

    /* Create index with loaded config */
    trine_s1_index_t *idx = trine_s1_index_create(&config);
    if (!idx) {
        fprintf(stderr, "trine_s1_index_load: allocation failed\n");
        fclose(fp);
        return NULL;
    }

    /* Read entries */
    for (uint32_t i = 0; i < count; i++) {
        uint8_t emb[TRINE_S1_DIMS];
        if (fread(emb, 1, TRINE_S1_DIMS, fp) != TRINE_S1_DIMS) {
            fprintf(stderr, "trine_s1_index_load: truncated entry %u in '%s'\n",
                    i, path);
            trine_s1_index_free(idx);
            fclose(fp);
            return NULL;
        }

        uint32_t tag_len;
        if (fread(&tag_len, sizeof(uint32_t), 1, fp) != 1) {
            fprintf(stderr, "trine_s1_index_load: truncated tag length at entry %u in '%s'\n",
                    i, path);
            trine_s1_index_free(idx);
            fclose(fp);
            return NULL;
        }

        char *tag = NULL;
        if (tag_len > 0) {
            tag = (char *)malloc(tag_len + 1);
            if (!tag) {
                fprintf(stderr, "trine_s1_index_load: allocation failed for tag\n");
                trine_s1_index_free(idx);
                fclose(fp);
                return NULL;
            }
            if (fread(tag, 1, tag_len, fp) != tag_len) {
                fprintf(stderr, "trine_s1_index_load: truncated tag data at entry %u in '%s'\n",
                        i, path);
                free(tag);
                trine_s1_index_free(idx);
                fclose(fp);
                return NULL;
            }
            tag[tag_len] = '\0';
        }

        if (trine_s1_index_add(idx, emb, tag) < 0) {
            fprintf(stderr, "trine_s1_index_load: failed to add entry %u\n", i);
            free(tag);
            trine_s1_index_free(idx);
            fclose(fp);
            return NULL;
        }

        free(tag);  /* index_add copies the tag, so free our temporary */
    }

    fclose(fp);
    return idx;

truncated:
    fprintf(stderr, "trine_s1_index_load: truncated config in '%s'\n", path);
    fclose(fp);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
 * VII. PACKED TRIT STORAGE
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Packing scheme: 5 trits per byte.
 * byte = t[0] + t[1]*3 + t[2]*9 + t[3]*27 + t[4]*81
 * Each trit is in {0, 1, 2}, so max value = 2+6+18+54+162 = 242.
 * 240 trits / 5 = 48 packed bytes.
 */

int trine_s1_pack(const uint8_t trits[240], uint8_t packed[48])
{
    if (!trits || !packed) return -1;

    for (int i = 0; i < TRINE_S1_PACKED_SIZE; i++) {
        int base = i * 5;
        packed[i] = (uint8_t)(
            trits[base + 0]       +
            trits[base + 1] * 3   +
            trits[base + 2] * 9   +
            trits[base + 3] * 27  +
            trits[base + 4] * 81
        );
    }

    return 0;
}

int trine_s1_unpack(const uint8_t packed[48], uint8_t trits[240])
{
    if (!packed || !trits) return -1;

    for (int i = 0; i < TRINE_S1_PACKED_SIZE; i++) {
        int base = i * 5;
        uint8_t val = packed[i];
        trits[base + 0] = val % 3;  val /= 3;
        trits[base + 1] = val % 3;  val /= 3;
        trits[base + 2] = val % 3;  val /= 3;
        trits[base + 3] = val % 3;  val /= 3;
        trits[base + 4] = val % 3;
    }

    return 0;
}

float trine_s1_compare_packed(const uint8_t a[48], const uint8_t b[48],
                               const trine_s1_lens_t *lens)
{
    if (!a || !b || !lens) return -1.0f;

    uint8_t ua[TRINE_S1_DIMS], ub[TRINE_S1_DIMS];
    trine_s1_unpack(a, ua);
    trine_s1_unpack(b, ub);

    return s1_lens_cosine(ua, ub, lens);
}

/* ═══════════════════════════════════════════════════════════════════════
 * VIII. LENGTH CALIBRATION
 * ═══════════════════════════════════════════════════════════════════════ */

float trine_s1_fill_ratio(const uint8_t emb[240])
{
    if (!emb) return 0.0f;

    int nonzero = 0;
    for (int i = 0; i < TRINE_S1_DIMS; i++) {
        if (emb[i] != 0) nonzero++;
    }

    return (float)nonzero / (float)TRINE_S1_DIMS;
}

float trine_s1_calibrate(float raw_cosine, float fill_a, float fill_b)
{
    /* Clamp fill values to avoid division explosion */
    if (fill_a < S1_FILL_CLAMP) fill_a = S1_FILL_CLAMP;
    if (fill_b < S1_FILL_CLAMP) fill_b = S1_FILL_CLAMP;

    float denom = sqrtf(fill_a * fill_b);

    /* Defensive: denom should never be zero after clamping,
     * but guard anyway */
    if (denom <= 0.0f) return 0.0f;

    float calibrated = raw_cosine / denom;

    /* Clamp to [0.0, 1.0] */
    if (calibrated > 1.0f) calibrated = 1.0f;
    if (calibrated < 0.0f) calibrated = 0.0f;

    return calibrated;
}
