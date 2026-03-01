/* ═══════════════════════════════════════════════════════════════════════
 * TRINE — Ternary Resonance Interference Network Embedding
 * Band-LSH Routing Overlay — Implementation v1.0.1
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Band-LSH routing for sub-linear candidate selection. Each of the 4
 * TRINE chains (60 trits) is hashed to a bucket via FNV-1a. At query
 * time, only entries sharing at least one bucket with the query undergo
 * full lens-weighted cosine comparison.
 *
 * Multi-probe improves recall by also checking buckets obtained from
 * trit-flipped variants of the query chains, catching near-misses.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -c trine_route.c -o trine_route.o
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#include "trine_route.h"
#include "trine_csidf.h"
#include "trine_field.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <errno.h>
#ifndef _WIN32
#include <unistd.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * I. CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

#define RT_INITIAL_CAPACITY   64
#define RT_FILL_CLAMP         0.05f
#define RT_FILE_VERSION       4       /* Current version (v4: CS-IDF + fields) */
#define RT_FILE_VERSION_V3    3       /* Can still load v3 files (endian + flags + checksum) */
#define RT_FILE_VERSION_V2    2       /* Can still load v2 files (recall mode, no checksum) */
#define RT_FILE_VERSION_V1    1       /* Can still load v1 files (no recall mode) */
#define RT_MAGIC              "TRRT"
#define RT_BUCKET_INIT_CAP    4

/*
 * v3 header layout (written after magic):
 *   uint32_t  version       = 3
 *   uint32_t  endian_check  = 0x01020304
 *   uint32_t  flags         = 0 (reserved)
 *   uint64_t  checksum      = FNV-1a over payload bytes after header
 *
 * Total header: magic(4) + version(4) + endian(4) + flags(4) + checksum(8) = 24 bytes
 */
#define RT_ENDIAN_MARKER      0x01020304u
#define RT_V3_HEADER_SIZE     24     /* 4 + 4 + 4 + 4 + 8 */
#define RT_V3_CHECKSUM_OFFSET 16     /* offset of checksum field from file start */

/* Maximum probe positions across all recall modes (STRICT uses 5) */
#define RT_MAX_PROBES         5

/* Probe position tables indexed by recall mode.
 * FAST:     1 probe  at position 0
 * BALANCED: 3 probes at positions 0, 15, 30
 * STRICT:   5 probes at positions 0, 10, 20, 30, 45 */
static const int PROBE_TABLE[3][RT_MAX_PROBES] = {
    { 0,  0,  0,  0,  0 },     /* FAST — only [0] used */
    { 0, 15, 30,  0,  0 },     /* BALANCED — [0..2] used */
    { 0, 10, 20, 30, 45 },     /* STRICT — [0..4] used */
};

/* Recall mode names for stats reporting */
static const char *RECALL_MODE_NAMES[3] = { "fast", "balanced", "strict" };

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

/* ═══════════════════════════════════════════════════════════════════════
 * II. INTERNAL STRUCTURES
 * ═══════════════════════════════════════════════════════════════════════ */

/* A single bucket slot: dynamic array of entry indices */
typedef struct {
    int *indices;
    int count;
    int capacity;
} rt_bucket_t;

/* One band: a hash table of TRINE_ROUTE_BUCKETS bucket slots */
typedef struct {
    rt_bucket_t slots[TRINE_ROUTE_BUCKETS];
} rt_band_t;

/* The routed index */
struct trine_route {
    trine_s1_config_t config;

    /* Entry storage (flat arrays, same layout as trine_s1_index) */
    uint8_t *embeddings;     /* count * 240 bytes */
    char   **tags;           /* count pointers, each may be NULL */
    int      count;
    int      capacity;

    /* Band-LSH tables: 4 bands */
    rt_band_t bands[TRINE_ROUTE_BANDS];

    /* Recall preset state (v1.1.0) */
    int      recall_mode;    /* TRINE_RECALL_FAST / BALANCED / STRICT */
    int      probes;         /* Active probe count (1 / 3 / 5) */
    int      candidate_cap;  /* Max unique candidates before early stop */

    /* Phase 4: CS-IDF (v2.0.0) */
    trine_csidf_t *csidf;   /* Corpus-specific IDF tracker (NULL if disabled) */

    /* Phase 4: Field-aware (v2.0.0) */
    trine_field_config_t *field_cfg;    /* Field config (NULL if disabled) */
    uint8_t *field_embeddings;          /* count * field_count * 240 bytes */
    int      field_emb_capacity;        /* Allocated field entries */
};

/* ═══════════════════════════════════════════════════════════════════════
 * III. COMPARISON HELPERS (copied from trine_stage1.c)
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Per-chain cosine similarity over a 60-channel slice.
 * Treats trit values as real-valued vector components {0, 1, 2}.
 * Returns 0.0 if either vector has zero magnitude.
 */
static float rt_chain_cosine(const uint8_t *a, const uint8_t *b,
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

/*
 * Lens-weighted cosine over the full 240-channel embedding.
 * combined = sum(weight[i] * chain_cosine[i]) / sum(weight[i])
 */
static float rt_lens_cosine(const uint8_t *a, const uint8_t *b,
                             const trine_s1_lens_t *lens)
{
    double weighted_sum = 0.0;
    double weight_sum   = 0.0;

    for (int c = 0; c < TRINE_S1_CHAINS; c++) {
        double w = (double)lens->weights[c];
        if (w <= 0.0) continue;

        float cos_c = rt_chain_cosine(a, b,
                                       c * TRINE_S1_CHAIN_WIDTH,
                                       TRINE_S1_CHAIN_WIDTH);
        weighted_sum += w * (double)cos_c;
        weight_sum   += w;
    }

    if (weight_sum == 0.0) return 0.0f;

    return (float)(weighted_sum / weight_sum);
}

/* ═══════════════════════════════════════════════════════════════════════
 * IV. BAND HASHING
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Hash a chain's 60 trit values to a bucket key using FNV-1a.
 * chain_idx seeds the hash to give each band an independent hash space.
 */
static uint32_t hash_band(const uint8_t *emb, int chain_idx)
{
    uint32_t h = 0x811c9dc5u ^ (uint32_t)(chain_idx * 0x9E3779B9u);
    int offset = chain_idx * TRINE_S1_CHAIN_WIDTH;
    for (int i = 0; i < TRINE_S1_CHAIN_WIDTH; i++) {
        h ^= (uint32_t)emb[offset + i];
        h *= 0x01000193u;
    }
    return h;
}

/*
 * Multi-probe hash: flip a single trit position within the chain
 * before hashing. The trit at probe_pos is changed to (v+1)%3.
 * This produces a nearby hash for catching near-miss embeddings.
 */
static uint32_t hash_band_probe(const uint8_t *emb, int chain_idx, int probe_pos)
{
    uint32_t h = 0x811c9dc5u ^ (uint32_t)(chain_idx * 0x9E3779B9u);
    int offset = chain_idx * TRINE_S1_CHAIN_WIDTH;
    for (int i = 0; i < TRINE_S1_CHAIN_WIDTH; i++) {
        uint32_t v = (uint32_t)emb[offset + i];
        if (i == probe_pos) {
            v = (v + 1) % 3;
        }
        h ^= v;
        h *= 0x01000193u;
    }
    return h;
}

/* ═══════════════════════════════════════════════════════════════════════
 * V. BUCKET OPERATIONS
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Insert entry_index into bucket slot, growing if needed.
 * Returns 0 on success, -1 on allocation failure.
 */
static int bucket_insert(rt_bucket_t *bkt, int entry_index)
{
    if (bkt->count >= bkt->capacity) {
        int new_cap = (bkt->capacity == 0) ? RT_BUCKET_INIT_CAP
                                           : bkt->capacity * 2;
        int *new_idx = (int *)realloc(bkt->indices,
                                       (size_t)new_cap * sizeof(int));
        if (!new_idx) return -1;
        bkt->indices  = new_idx;
        bkt->capacity = new_cap;
    }
    bkt->indices[bkt->count++] = entry_index;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * VI. PUBLIC API — CREATE / FREE
 * ═══════════════════════════════════════════════════════════════════════ */

trine_route_t *trine_route_create(const trine_s1_config_t *config)
{
    trine_route_t *rt = (trine_route_t *)calloc(1, sizeof(trine_route_t));
    if (!rt) return NULL;

    if (config) {
        rt->config = *config;
    } else {
        trine_s1_config_t default_config = TRINE_S1_CONFIG_DEFAULT;
        rt->config = default_config;
    }

    rt->capacity = RT_INITIAL_CAPACITY;
    rt->count    = 0;

    rt->embeddings = (uint8_t *)malloc(
        (size_t)rt->capacity * TRINE_S1_DIMS);
    if (!rt->embeddings) {
        free(rt);
        return NULL;
    }

    rt->tags = (char **)calloc((size_t)rt->capacity, sizeof(char *));
    if (!rt->tags) {
        free(rt->embeddings);
        free(rt);
        return NULL;
    }

    /* Band tables are zero-initialized by calloc (all slots empty) */

    /* Default recall preset: BALANCED (3 probes, cap 500) */
    rt->recall_mode  = TRINE_RECALL_BALANCED;
    rt->probes       = 3;
    rt->candidate_cap = 500;

    return rt;
}

void trine_route_free(trine_route_t *rt)
{
    if (!rt) return;

    /* Free tag strings */
    for (int i = 0; i < rt->count; i++) {
        free(rt->tags[i]);
    }
    free(rt->tags);
    free(rt->embeddings);

    /* Free bucket index arrays */
    for (int b = 0; b < TRINE_ROUTE_BANDS; b++) {
        for (int s = 0; s < TRINE_ROUTE_BUCKETS; s++) {
            free(rt->bands[b].slots[s].indices);
        }
    }

    /* Free Phase 4 extensions */
    free(rt->csidf);
    free(rt->field_cfg);
    free(rt->field_embeddings);

    free(rt);
}

/* ═══════════════════════════════════════════════════════════════════════
 * VII. PUBLIC API — ADD
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_route_add(trine_route_t *rt, const uint8_t emb[240], const char *tag)
{
    if (!rt || !emb) return -1;

    /* Grow storage if at capacity */
    if (rt->count >= rt->capacity) {
        int new_cap = rt->capacity * 2;

        uint8_t *new_emb = (uint8_t *)realloc(
            rt->embeddings, (size_t)new_cap * TRINE_S1_DIMS);
        if (!new_emb) return -1;
        rt->embeddings = new_emb;

        char **new_tags = (char **)realloc(
            rt->tags, (size_t)new_cap * sizeof(char *));
        if (!new_tags) return -1;
        rt->tags = new_tags;

        for (int i = rt->capacity; i < new_cap; i++)
            rt->tags[i] = NULL;

        rt->capacity = new_cap;
    }

    int idx = rt->count;

    /* Copy embedding */
    memcpy(rt->embeddings + (size_t)idx * TRINE_S1_DIMS,
           emb, TRINE_S1_DIMS);

    /* Copy tag */
    if (tag) {
        size_t tag_len = strlen(tag);
        rt->tags[idx] = (char *)malloc(tag_len + 1);
        if (rt->tags[idx]) {
            memcpy(rt->tags[idx], tag, tag_len + 1);
        }
    } else {
        rt->tags[idx] = NULL;
    }

    /* Insert into band hash tables */
    for (int b = 0; b < TRINE_ROUTE_BANDS; b++) {
        uint32_t key = hash_band(emb, b);
        int slot = (int)(key % TRINE_ROUTE_BUCKETS);
        if (bucket_insert(&rt->bands[b].slots[slot], idx) < 0) {
            /* Allocation failure in bucket — entry is still stored,
             * but may not be routed via this band. Non-fatal. */
        }
    }

    /* Phase 4: Track CS-IDF document frequencies */
    if (rt->csidf) {
        trine_csidf_observe(rt->csidf, emb);
    }

    rt->count++;
    return idx;
}

/* ═══════════════════════════════════════════════════════════════════════
 * VIII. PUBLIC API — QUERY
 * ═══════════════════════════════════════════════════════════════════════ */

trine_s1_result_t trine_route_query(const trine_route_t *rt,
                                     const uint8_t candidate[240],
                                     trine_route_stats_t *stats)
{
    trine_s1_result_t result;
    memset(&result, 0, sizeof(result));
    result.matched_index = -1;

    if (!rt || !candidate || rt->count == 0) {
        if (stats) {
            memset(stats, 0, sizeof(*stats));
            stats->recall_mode = RECALL_MODE_NAMES[rt ? rt->recall_mode : TRINE_RECALL_BALANCED];
        }
        return result;
    }

    /* ---------------------------------------------------------------
     * Small-N fallback: when the index has fewer entries than the
     * threshold, LSH buckets are too sparse for reliable recall.
     * Fall back to brute-force linear scan (invisible to callers).
     * --------------------------------------------------------------- */
    if (rt->count < TRINE_ROUTE_FALLBACK_THRESHOLD) {
        float fill_cand = 0.0f;
        if (rt->config.calibrate_length) {
            fill_cand = trine_s1_fill_ratio(candidate);
        }

        float best_score = -1.0f;
        float best_raw   = 0.0f;
        float best_cal   = 0.0f;
        int   best_idx   = -1;

        for (int i = 0; i < rt->count; i++) {
            const uint8_t *entry = rt->embeddings + (size_t)i * TRINE_S1_DIMS;

            float raw = rt_lens_cosine(candidate, entry, &rt->config.lens);

            float cal;
            if (rt->config.calibrate_length) {
                float fill_entry = trine_s1_fill_ratio(entry);
                cal = trine_s1_calibrate(raw, fill_cand, fill_entry);
            } else {
                cal = raw;
            }

            float score = rt->config.calibrate_length ? cal : raw;

            if (score > best_score) {
                best_score = score;
                best_raw   = raw;
                best_cal   = cal;
                best_idx   = i;
            }
        }

        result.similarity = best_raw;
        result.calibrated = best_cal;

        if (best_score >= rt->config.threshold) {
            result.is_duplicate = 1;
            result.matched_index = best_idx;
        }

        if (stats) {
            stats->candidates_checked = rt->count;
            stats->total_entries      = rt->count;
            stats->candidate_ratio    = 1.0f;
            stats->speedup            = 1.0f;
            stats->recall_mode        = RECALL_MODE_NAMES[rt->recall_mode];
        }

        return result;
    }

    /* ---------------------------------------------------------------
     * Standard LSH-routed query path (count >= threshold).
     * Uses rt->probes and rt->candidate_cap from the recall preset.
     * --------------------------------------------------------------- */

    /* Allocate seen-bitset for deduplication: 1 bit per entry */
    size_t bitset_bytes = ((size_t)rt->count + 7) / 8;
    uint8_t *seen = (uint8_t *)calloc(bitset_bytes, 1);
    if (!seen) {
        /* Fallback: if we cannot allocate the bitset, return empty */
        if (stats) {
            memset(stats, 0, sizeof(*stats));
            stats->total_entries = rt->count;
            stats->recall_mode  = RECALL_MODE_NAMES[rt->recall_mode];
        }
        return result;
    }

    /* Collect candidate indices from all bands + probes */
    int *candidates = NULL;
    int  cand_count = 0;
    int  cand_alloc = 0;
    int  cap_hit    = 0;

    const int active_probes = rt->probes;
    const int active_cap    = rt->candidate_cap;
    const int *probe_pos    = PROBE_TABLE[rt->recall_mode];

    for (int b = 0; b < TRINE_ROUTE_BANDS && !cap_hit; b++) {
        /* Primary bucket */
        uint32_t key0 = hash_band(candidate, b);
        int slot0 = (int)(key0 % TRINE_ROUTE_BUCKETS);
        const rt_bucket_t *bkt = &rt->bands[b].slots[slot0];

        for (int j = 0; j < bkt->count; j++) {
            int ei = bkt->indices[j];
            int byte_idx = ei / 8;
            int bit_idx  = ei % 8;
            if (seen[byte_idx] & (1 << bit_idx)) continue;
            seen[byte_idx] |= (uint8_t)(1 << bit_idx);

            /* Append to candidates array */
            if (cand_count >= cand_alloc) {
                int new_alloc = (cand_alloc == 0) ? 64 : cand_alloc * 2;
                int *nc = (int *)realloc(candidates,
                                          (size_t)new_alloc * sizeof(int));
                if (!nc) goto done_collect;
                candidates = nc;
                cand_alloc = new_alloc;
            }
            candidates[cand_count++] = ei;

            if (cand_count >= active_cap) { cap_hit = 1; break; }
        }
        if (cap_hit) break;

        /* Multi-probe buckets */
        for (int p = 0; p < active_probes && !cap_hit; p++) {
            uint32_t key_p = hash_band_probe(candidate, b, probe_pos[p]);
            int slot_p = (int)(key_p % TRINE_ROUTE_BUCKETS);
            if (slot_p == slot0) continue;  /* Same bucket, skip */

            const rt_bucket_t *pbkt = &rt->bands[b].slots[slot_p];
            for (int j = 0; j < pbkt->count; j++) {
                int ei = pbkt->indices[j];
                int byte_idx = ei / 8;
                int bit_idx  = ei % 8;
                if (seen[byte_idx] & (1 << bit_idx)) continue;
                seen[byte_idx] |= (uint8_t)(1 << bit_idx);

                if (cand_count >= cand_alloc) {
                    int new_alloc = (cand_alloc == 0) ? 64 : cand_alloc * 2;
                    int *nc = (int *)realloc(candidates,
                                              (size_t)new_alloc * sizeof(int));
                    if (!nc) goto done_collect;
                    candidates = nc;
                    cand_alloc = new_alloc;
                }
                candidates[cand_count++] = ei;

                if (cand_count >= active_cap) { cap_hit = 1; break; }
            }
        }
    }

done_collect:
    free(seen);

    /* Compare candidate embeddings using full lens-weighted cosine */
    float fill_cand = 0.0f;
    if (rt->config.calibrate_length) {
        fill_cand = trine_s1_fill_ratio(candidate);
    }

    float best_score = -1.0f;
    float best_raw   = 0.0f;
    float best_cal   = 0.0f;
    int   best_idx   = -1;

    for (int i = 0; i < cand_count; i++) {
        int ei = candidates[i];
        const uint8_t *entry = rt->embeddings + (size_t)ei * TRINE_S1_DIMS;

        float raw = rt_lens_cosine(candidate, entry, &rt->config.lens);

        float cal;
        if (rt->config.calibrate_length) {
            float fill_entry = trine_s1_fill_ratio(entry);
            cal = trine_s1_calibrate(raw, fill_cand, fill_entry);
        } else {
            cal = raw;
        }

        float score = rt->config.calibrate_length ? cal : raw;

        if (score > best_score) {
            best_score = score;
            best_raw   = raw;
            best_cal   = cal;
            best_idx   = ei;
        }
    }

    free(candidates);

    result.similarity = best_raw;
    result.calibrated = best_cal;

    if (best_score >= rt->config.threshold) {
        result.is_duplicate = 1;
        result.matched_index = best_idx;
    }

    /* Fill stats */
    if (stats) {
        stats->candidates_checked = cand_count;
        stats->total_entries      = rt->count;
        stats->recall_mode        = RECALL_MODE_NAMES[rt->recall_mode];
        if (rt->count > 0) {
            stats->candidate_ratio = (float)cand_count / (float)rt->count;
            stats->speedup = (cand_count > 0)
                ? (float)rt->count / (float)cand_count
                : (float)rt->count;
        } else {
            stats->candidate_ratio = 0.0f;
            stats->speedup         = 0.0f;
        }
    }

    return result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * IX. PUBLIC API — ACCESSORS
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_route_count(const trine_route_t *rt)
{
    if (!rt) return 0;
    return rt->count;
}

const char *trine_route_tag(const trine_route_t *rt, int index)
{
    if (!rt || index < 0 || index >= rt->count) return NULL;
    return rt->tags[index];
}

const uint8_t *trine_route_embedding(const trine_route_t *rt, int index)
{
    if (!rt || index < 0 || index >= rt->count) return NULL;
    return rt->embeddings + (size_t)index * TRINE_S1_DIMS;
}

/* ═══════════════════════════════════════════════════════════════════════
 * X. GLOBAL STATISTICS
 * ═══════════════════════════════════════════════════════════════════════ */

void trine_route_global_stats(const trine_route_t *rt, trine_route_stats_t *stats)
{
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));

    if (!rt) return;

    stats->total_entries = rt->count;
    stats->recall_mode   = RECALL_MODE_NAMES[rt->recall_mode];

    if (rt->count == 0) return;

    /* Compute average bucket occupancy across all bands */
    int total_occupied = 0;
    long total_entries_in_buckets = 0;

    for (int b = 0; b < TRINE_ROUTE_BANDS; b++) {
        for (int s = 0; s < TRINE_ROUTE_BUCKETS; s++) {
            if (rt->bands[b].slots[s].count > 0) {
                total_occupied++;
                total_entries_in_buckets += rt->bands[b].slots[s].count;
            }
        }
    }

    /* Average candidates per query ~ average bucket size * (bands + probes) */
    if (total_occupied > 0) {
        float avg_bucket = (float)total_entries_in_buckets / (float)total_occupied;
        int probes_per_query = TRINE_ROUTE_BANDS * (1 + rt->probes);
        float est_candidates = avg_bucket * (float)probes_per_query;

        /* Clamp to candidate cap and entry count */
        if (est_candidates > (float)rt->candidate_cap) {
            est_candidates = (float)rt->candidate_cap;
        }
        if (est_candidates > (float)rt->count) {
            est_candidates = (float)rt->count;
        }

        stats->candidates_checked = (int)est_candidates;
        stats->candidate_ratio = est_candidates / (float)rt->count;
        stats->speedup = (est_candidates > 0.0f)
            ? (float)rt->count / est_candidates
            : (float)rt->count;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * XI. RECALL PRESETS
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_route_set_recall(trine_route_t *rt, int mode)
{
    if (!rt) return -1;
    if (mode < TRINE_RECALL_FAST || mode > TRINE_RECALL_STRICT) return -1;

    rt->recall_mode = mode;

    switch (mode) {
        case TRINE_RECALL_FAST:
            rt->probes        = 1;
            rt->candidate_cap = 200;
            break;
        case TRINE_RECALL_BALANCED:
            rt->probes        = 3;
            rt->candidate_cap = 500;
            break;
        case TRINE_RECALL_STRICT:
            rt->probes        = 5;
            rt->candidate_cap = 2000;
            break;
    }

    return 0;
}

int trine_route_get_recall(const trine_route_t *rt)
{
    if (!rt) return TRINE_RECALL_BALANCED;
    return rt->recall_mode;
}

/* ═══════════════════════════════════════════════════════════════════════
 * XII. PERSISTENCE — SAVE
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_route_save(const trine_route_t *rt, const char *path)
{
    if (!rt || !path) return -1;

    FILE *fp = fopen(path, "w+b");  /* w+b: read+write for checksum patching */
    if (!fp) return -1;

    /* Magic */
    if (fwrite(RT_MAGIC, 1, 4, fp) != 4) goto fail;

    /* Version */
    uint32_t version = RT_FILE_VERSION;
    if (fwrite(&version, sizeof(uint32_t), 1, fp) != 1) goto fail;

    /* Endianness marker (v3+) */
    uint32_t endian_check = RT_ENDIAN_MARKER;
    if (fwrite(&endian_check, sizeof(uint32_t), 1, fp) != 1) goto fail;

    /* Feature flags (v4: CS-IDF and/or field sections) */
    uint32_t flags = 0;
    if (rt->csidf && rt->csidf->computed) flags |= TRINE_ROUTE_FLAG_CSIDF;
    if (rt->field_cfg) flags |= TRINE_ROUTE_FLAG_FIELDS;
    if (fwrite(&flags, sizeof(uint32_t), 1, fp) != 1) goto fail;

    /* Checksum placeholder — patched after payload is written */
    uint64_t checksum = 0;
    if (fwrite(&checksum, sizeof(uint64_t), 1, fp) != 1) goto fail;

    /* --- payload starts here (byte offset RT_V3_HEADER_SIZE) --- */

    /* Count */
    uint32_t count = (uint32_t)rt->count;
    if (fwrite(&count, sizeof(uint32_t), 1, fp) != 1) goto fail;

    /* Config: threshold, 4 lens weights, calibrate_length */
    if (fwrite(&rt->config.threshold, sizeof(float), 1, fp) != 1) goto fail;
    if (fwrite(rt->config.lens.weights, sizeof(float), TRINE_S1_CHAINS, fp)
        != TRINE_S1_CHAINS) goto fail;
    int32_t cal = (int32_t)rt->config.calibrate_length;
    if (fwrite(&cal, sizeof(int32_t), 1, fp) != 1) goto fail;

    /* Recall mode (present since v2) */
    int32_t rm = (int32_t)rt->recall_mode;
    if (fwrite(&rm, sizeof(int32_t), 1, fp) != 1) goto fail;

    /* Embeddings: all entries contiguous */
    if (count > 0) {
        size_t emb_bytes = (size_t)count * TRINE_S1_DIMS;
        if (fwrite(rt->embeddings, 1, emb_bytes, fp) != emb_bytes) goto fail;
    }

    /* Tags */
    for (uint32_t i = 0; i < count; i++) {
        if (rt->tags[i]) {
            uint32_t tag_len = (uint32_t)strlen(rt->tags[i]);
            if (fwrite(&tag_len, sizeof(uint32_t), 1, fp) != 1) goto fail;
            if (tag_len > 0) {
                if (fwrite(rt->tags[i], 1, tag_len, fp) != tag_len) goto fail;
            }
        } else {
            uint32_t tag_len = 0;
            if (fwrite(&tag_len, sizeof(uint32_t), 1, fp) != 1) goto fail;
        }
    }

    /* Bucket tables: for each band, for each slot, write count + indices */
    for (int b = 0; b < TRINE_ROUTE_BANDS; b++) {
        for (int s = 0; s < TRINE_ROUTE_BUCKETS; s++) {
            uint32_t bc = (uint32_t)rt->bands[b].slots[s].count;
            if (fwrite(&bc, sizeof(uint32_t), 1, fp) != 1) goto fail;
            if (bc > 0) {
                if (fwrite(rt->bands[b].slots[s].indices,
                           sizeof(int), bc, fp) != bc) goto fail;
            }
        }
    }

    /* Phase 4: CS-IDF section (if enabled and computed) */
    if (flags & TRINE_ROUTE_FLAG_CSIDF) {
        if (trine_csidf_write(rt->csidf, fp) != 0) goto fail;
    }

    /* Phase 4: Field section (if enabled) */
    if (flags & TRINE_ROUTE_FLAG_FIELDS) {
        /* Write field config */
        if (trine_field_config_write(rt->field_cfg, fp) != 0) goto fail;

        /* Write field embeddings: count * field_count * 240 bytes */
        if (count > 0 && rt->field_embeddings) {
            size_t field_bytes = (size_t)count * (size_t)rt->field_cfg->field_count
                                 * TRINE_S1_DIMS;
            if (fwrite(rt->field_embeddings, 1, field_bytes, fp) != field_bytes)
                goto fail;
        }
    }

    /* Compute checksum over the payload (everything after the header) */
    {
        long file_end = ftell(fp);
        if (file_end < 0) goto fail;

        size_t payload_size = (size_t)file_end - RT_V3_HEADER_SIZE;
        if (payload_size > 0) {
            uint8_t *payload_buf = (uint8_t *)malloc(payload_size);
            if (!payload_buf) goto fail;

            if (fseek(fp, RT_V3_HEADER_SIZE, SEEK_SET) != 0) {
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
        if (fseek(fp, RT_V3_CHECKSUM_OFFSET, SEEK_SET) != 0) goto fail;
        if (fwrite(&checksum, sizeof(uint64_t), 1, fp) != 1) goto fail;
    }

    fclose(fp);
    return 0;

fail:
    fclose(fp);
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * XIII. PERSISTENCE — LOAD
 * ═══════════════════════════════════════════════════════════════════════ */

trine_route_t *trine_route_load(const char *path)
{
    if (!path) return NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "trine_route_load: cannot open '%s'\n", path);
        return NULL;
    }

    /* Magic */
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, RT_MAGIC, 4) != 0) {
        fprintf(stderr, "trine_route_load: bad magic in '%s'\n", path);
        fclose(fp);
        return NULL;
    }

    /* Version */
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "trine_route_load: truncated version in '%s'\n", path);
        fclose(fp);
        return NULL;
    }

    if (version > (uint32_t)RT_FILE_VERSION) {
        fprintf(stderr,
                "trine_route_load: unsupported version %u in '%s' "
                "(this build supports up to v%u)\n",
                version, path, (uint32_t)RT_FILE_VERSION);
        fclose(fp);
        return NULL;
    }

    /* v3 header fields: endianness marker, feature flags, checksum */
    long payload_offset = 4 + 4;  /* magic + version (v1/v2 payload starts here) */
    uint64_t stored_checksum = 0;
    int have_checksum = 0;

    uint32_t loaded_flags = 0;

    if (version >= 3) {
        /* Endianness marker */
        uint32_t endian_check;
        if (fread(&endian_check, sizeof(uint32_t), 1, fp) != 1) goto truncated;
        if (endian_check != RT_ENDIAN_MARKER) {
            fprintf(stderr,
                    "trine_route_load: endianness mismatch in '%s' "
                    "(expected 0x%08X, got 0x%08X) — file may be from a "
                    "different architecture\n",
                    path, RT_ENDIAN_MARKER, endian_check);
            fclose(fp);
            return NULL;
        }

        /* Feature flags */
        if (fread(&loaded_flags, sizeof(uint32_t), 1, fp) != 1) goto truncated;

        /* Payload checksum */
        if (fread(&stored_checksum, sizeof(uint64_t), 1, fp) != 1) goto truncated;
        have_checksum = 1;

        payload_offset = RT_V3_HEADER_SIZE;
    }

    /* If v3, verify checksum over the payload before parsing it */
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
                fprintf(stderr, "trine_route_load: allocation failed for checksum\n");
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
                        "trine_route_load: WARNING — checksum mismatch in '%s' "
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
        fprintf(stderr, "trine_route_load: truncated header in '%s'\n", path);
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

    /* v2+: recall mode (v1 files default to BALANCED) */
    int recall_mode = TRINE_RECALL_BALANCED;
    if (version >= (uint32_t)RT_FILE_VERSION_V2) {
        int32_t rm;
        if (fread(&rm, sizeof(int32_t), 1, fp) != 1) goto truncated;
        if (rm >= TRINE_RECALL_FAST && rm <= TRINE_RECALL_STRICT) {
            recall_mode = (int)rm;
        }
    }

    /* Allocate the route structure */
    trine_route_t *rt = (trine_route_t *)calloc(1, sizeof(trine_route_t));
    if (!rt) {
        fprintf(stderr, "trine_route_load: allocation failed\n");
        fclose(fp);
        return NULL;
    }
    rt->config = config;

    /* Apply recall mode (sets probes and candidate_cap) */
    rt->recall_mode = TRINE_RECALL_BALANCED;
    rt->probes      = 3;
    rt->candidate_cap = 500;
    trine_route_set_recall(rt, recall_mode);

    /* Set capacity to max(count, RT_INITIAL_CAPACITY) */
    rt->capacity = ((int)count > RT_INITIAL_CAPACITY)
                 ? (int)count : RT_INITIAL_CAPACITY;
    rt->count = (int)count;

    rt->embeddings = (uint8_t *)malloc((size_t)rt->capacity * TRINE_S1_DIMS);
    if (!rt->embeddings) {
        fprintf(stderr, "trine_route_load: allocation failed for embeddings\n");
        free(rt);
        fclose(fp);
        return NULL;
    }

    rt->tags = (char **)calloc((size_t)rt->capacity, sizeof(char *));
    if (!rt->tags) {
        fprintf(stderr, "trine_route_load: allocation failed for tags\n");
        free(rt->embeddings);
        free(rt);
        fclose(fp);
        return NULL;
    }

    /* Read embeddings */
    if (count > 0) {
        size_t emb_bytes = (size_t)count * TRINE_S1_DIMS;
        if (fread(rt->embeddings, 1, emb_bytes, fp) != emb_bytes) {
            fprintf(stderr, "trine_route_load: truncated embeddings in '%s'\n",
                    path);
            trine_route_free(rt);
            fclose(fp);
            return NULL;
        }
    }

    /* Read tags */
    for (uint32_t i = 0; i < count; i++) {
        uint32_t tag_len;
        if (fread(&tag_len, sizeof(uint32_t), 1, fp) != 1) {
            fprintf(stderr, "trine_route_load: truncated tag at entry %u in '%s'\n",
                    i, path);
            trine_route_free(rt);
            fclose(fp);
            return NULL;
        }

        if (tag_len > 0) {
            char *tag = (char *)malloc(tag_len + 1);
            if (!tag) {
                fprintf(stderr, "trine_route_load: allocation failed for tag\n");
                trine_route_free(rt);
                fclose(fp);
                return NULL;
            }
            if (fread(tag, 1, tag_len, fp) != tag_len) {
                fprintf(stderr,
                        "trine_route_load: truncated tag data at entry %u in '%s'\n",
                        i, path);
                free(tag);
                trine_route_free(rt);
                fclose(fp);
                return NULL;
            }
            tag[tag_len] = '\0';
            rt->tags[i] = tag;
        } else {
            rt->tags[i] = NULL;
        }
    }

    /* Read bucket tables */
    for (int b = 0; b < TRINE_ROUTE_BANDS; b++) {
        for (int s = 0; s < TRINE_ROUTE_BUCKETS; s++) {
            uint32_t bc;
            if (fread(&bc, sizeof(uint32_t), 1, fp) != 1) {
                fprintf(stderr,
                        "trine_route_load: truncated bucket table at band %d slot %d in '%s'\n",
                        b, s, path);
                trine_route_free(rt);
                fclose(fp);
                return NULL;
            }

            rt->bands[b].slots[s].count    = (int)bc;
            rt->bands[b].slots[s].capacity = (int)bc;

            if (bc > 0) {
                rt->bands[b].slots[s].indices = (int *)malloc(
                    (size_t)bc * sizeof(int));
                if (!rt->bands[b].slots[s].indices) {
                    fprintf(stderr,
                            "trine_route_load: allocation failed for bucket\n");
                    rt->bands[b].slots[s].count    = 0;
                    rt->bands[b].slots[s].capacity = 0;
                    trine_route_free(rt);
                    fclose(fp);
                    return NULL;
                }
                if (fread(rt->bands[b].slots[s].indices,
                          sizeof(int), bc, fp) != bc) {
                    fprintf(stderr,
                            "trine_route_load: truncated bucket indices at band %d slot %d in '%s'\n",
                            b, s, path);
                    trine_route_free(rt);
                    fclose(fp);
                    return NULL;
                }
            } else {
                rt->bands[b].slots[s].indices = NULL;
            }
        }
    }

    /* Phase 4: Load CS-IDF section if flag is set */
    if (loaded_flags & TRINE_ROUTE_FLAG_CSIDF) {
        rt->csidf = (trine_csidf_t *)calloc(1, sizeof(trine_csidf_t));
        if (!rt->csidf) {
            fprintf(stderr, "trine_route_load: allocation failed for CS-IDF\n");
            trine_route_free(rt);
            fclose(fp);
            return NULL;
        }
        if (trine_csidf_read(rt->csidf, fp) != 0) {
            fprintf(stderr, "trine_route_load: truncated CS-IDF section in '%s'\n",
                    path);
            trine_route_free(rt);
            fclose(fp);
            return NULL;
        }
    }

    /* Phase 4: Load field section if flag is set */
    if (loaded_flags & TRINE_ROUTE_FLAG_FIELDS) {
        rt->field_cfg = (trine_field_config_t *)calloc(1,
                            sizeof(trine_field_config_t));
        if (!rt->field_cfg) {
            fprintf(stderr, "trine_route_load: allocation failed for field config\n");
            trine_route_free(rt);
            fclose(fp);
            return NULL;
        }
        if (trine_field_config_read(rt->field_cfg, fp) != 0) {
            fprintf(stderr, "trine_route_load: truncated field config in '%s'\n",
                    path);
            trine_route_free(rt);
            fclose(fp);
            return NULL;
        }

        /* Read field embeddings */
        if (count > 0 && rt->field_cfg->field_count > 0) {
            size_t field_bytes = (size_t)count
                                 * (size_t)rt->field_cfg->field_count
                                 * TRINE_S1_DIMS;
            rt->field_embeddings = (uint8_t *)malloc(field_bytes);
            if (!rt->field_embeddings) {
                fprintf(stderr, "trine_route_load: allocation failed for field embeddings\n");
                trine_route_free(rt);
                fclose(fp);
                return NULL;
            }
            if (fread(rt->field_embeddings, 1, field_bytes, fp) != field_bytes) {
                fprintf(stderr, "trine_route_load: truncated field embeddings in '%s'\n",
                        path);
                trine_route_free(rt);
                fclose(fp);
                return NULL;
            }
            rt->field_emb_capacity = (int)count;
        }
    }

    fclose(fp);
    return rt;

truncated:
    fprintf(stderr, "trine_route_load: truncated config in '%s'\n", path);
    fclose(fp);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
 * XIV. BUCKET DIAGNOSTIC — SIZES
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_route_bucket_sizes(const trine_route_t *rt, int band, int *sizes)
{
    if (!rt || !sizes || band < 0 || band >= TRINE_ROUTE_BANDS)
        return -1;

    for (int s = 0; s < TRINE_ROUTE_BUCKETS; s++) {
        sizes[s] = rt->bands[band].slots[s].count;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * XV. PHASE 4: CS-IDF INTEGRATION
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_route_enable_csidf(trine_route_t *rt)
{
    if (!rt) return -1;
    if (rt->csidf) return 0;  /* Already enabled */

    rt->csidf = (trine_csidf_t *)calloc(1, sizeof(trine_csidf_t));
    if (!rt->csidf) return -1;

    trine_csidf_init(rt->csidf);

    /* If index already has entries, retroactively observe them */
    for (int i = 0; i < rt->count; i++) {
        const uint8_t *emb = rt->embeddings + (size_t)i * TRINE_S1_DIMS;
        trine_csidf_observe(rt->csidf, emb);
    }

    return 0;
}

int trine_route_compute_csidf(trine_route_t *rt)
{
    if (!rt || !rt->csidf) return -1;
    return trine_csidf_compute(rt->csidf);
}

const trine_csidf_t *trine_route_get_csidf(const trine_route_t *rt)
{
    if (!rt || !rt->csidf || !rt->csidf->computed) return NULL;
    return rt->csidf;
}

/*
 * CS-IDF weighted cosine: per-chain IDF-weighted cosine combined via lens.
 */
static float rt_csidf_lens_cosine(const uint8_t *a, const uint8_t *b,
                                    const trine_s1_lens_t *lens,
                                    const trine_csidf_t *csidf)
{
    return trine_csidf_cosine_lens(a, b, csidf, lens->weights);
}

trine_s1_result_t trine_route_query_csidf(const trine_route_t *rt,
                                           const uint8_t candidate[240],
                                           trine_route_stats_t *stats)
{
    trine_s1_result_t result;
    memset(&result, 0, sizeof(result));
    result.matched_index = -1;

    if (!rt || !candidate || rt->count == 0 ||
        !rt->csidf || !rt->csidf->computed) {
        if (stats) {
            memset(stats, 0, sizeof(*stats));
            stats->recall_mode = RECALL_MODE_NAMES[rt ? rt->recall_mode : TRINE_RECALL_BALANCED];
        }
        return result;
    }

    /* Small-N fallback: brute-force scan */
    if (rt->count < TRINE_ROUTE_FALLBACK_THRESHOLD) {
        float best_score = -1.0f;
        int   best_idx   = -1;

        for (int i = 0; i < rt->count; i++) {
            const uint8_t *entry = rt->embeddings + (size_t)i * TRINE_S1_DIMS;
            float score = rt_csidf_lens_cosine(candidate, entry,
                                                &rt->config.lens, rt->csidf);
            if (score > best_score) {
                best_score = score;
                best_idx   = i;
            }
        }

        result.similarity = best_score;
        result.calibrated = best_score;
        if (best_score >= rt->config.threshold) {
            result.is_duplicate = 1;
            result.matched_index = best_idx;
        }

        if (stats) {
            stats->candidates_checked = rt->count;
            stats->total_entries      = rt->count;
            stats->candidate_ratio    = 1.0f;
            stats->speedup            = 1.0f;
            stats->recall_mode        = RECALL_MODE_NAMES[rt->recall_mode];
        }
        return result;
    }

    /* Standard LSH-routed query with CS-IDF scoring */
    size_t bitset_bytes = ((size_t)rt->count + 7) / 8;
    uint8_t *seen = (uint8_t *)calloc(bitset_bytes, 1);
    if (!seen) {
        if (stats) {
            memset(stats, 0, sizeof(*stats));
            stats->total_entries = rt->count;
            stats->recall_mode  = RECALL_MODE_NAMES[rt->recall_mode];
        }
        return result;
    }

    int *candidates = NULL;
    int  cand_count = 0;
    int  cand_alloc = 0;
    int  cap_hit    = 0;

    const int active_probes = rt->probes;
    const int active_cap    = rt->candidate_cap;
    const int *probe_pos    = PROBE_TABLE[rt->recall_mode];

    for (int b = 0; b < TRINE_ROUTE_BANDS && !cap_hit; b++) {
        uint32_t key0 = hash_band(candidate, b);
        int slot0 = (int)(key0 % TRINE_ROUTE_BUCKETS);
        const rt_bucket_t *bkt = &rt->bands[b].slots[slot0];

        for (int j = 0; j < bkt->count; j++) {
            int ei = bkt->indices[j];
            int byte_idx = ei / 8;
            int bit_idx  = ei % 8;
            if (seen[byte_idx] & (1 << bit_idx)) continue;
            seen[byte_idx] |= (uint8_t)(1 << bit_idx);

            if (cand_count >= cand_alloc) {
                int new_alloc = (cand_alloc == 0) ? 64 : cand_alloc * 2;
                int *nc = (int *)realloc(candidates,
                                          (size_t)new_alloc * sizeof(int));
                if (!nc) goto done_csidf_collect;
                candidates = nc;
                cand_alloc = new_alloc;
            }
            candidates[cand_count++] = ei;
            if (cand_count >= active_cap) { cap_hit = 1; break; }
        }
        if (cap_hit) break;

        for (int p = 0; p < active_probes && !cap_hit; p++) {
            uint32_t key_p = hash_band_probe(candidate, b, probe_pos[p]);
            int slot_p = (int)(key_p % TRINE_ROUTE_BUCKETS);
            if (slot_p == slot0) continue;

            const rt_bucket_t *pbkt = &rt->bands[b].slots[slot_p];
            for (int j = 0; j < pbkt->count; j++) {
                int ei = pbkt->indices[j];
                int byte_idx = ei / 8;
                int bit_idx  = ei % 8;
                if (seen[byte_idx] & (1 << bit_idx)) continue;
                seen[byte_idx] |= (uint8_t)(1 << bit_idx);

                if (cand_count >= cand_alloc) {
                    int new_alloc = (cand_alloc == 0) ? 64 : cand_alloc * 2;
                    int *nc = (int *)realloc(candidates,
                                              (size_t)new_alloc * sizeof(int));
                    if (!nc) goto done_csidf_collect;
                    candidates = nc;
                    cand_alloc = new_alloc;
                }
                candidates[cand_count++] = ei;
                if (cand_count >= active_cap) { cap_hit = 1; break; }
            }
        }
    }

done_csidf_collect:
    free(seen);

    float best_score = -1.0f;
    int   best_idx   = -1;

    for (int i = 0; i < cand_count; i++) {
        int ei = candidates[i];
        const uint8_t *entry = rt->embeddings + (size_t)ei * TRINE_S1_DIMS;
        float score = rt_csidf_lens_cosine(candidate, entry,
                                            &rt->config.lens, rt->csidf);
        if (score > best_score) {
            best_score = score;
            best_idx   = ei;
        }
    }

    free(candidates);

    result.similarity = best_score;
    result.calibrated = best_score;
    if (best_score >= rt->config.threshold) {
        result.is_duplicate = 1;
        result.matched_index = best_idx;
    }

    if (stats) {
        stats->candidates_checked = cand_count;
        stats->total_entries      = rt->count;
        stats->recall_mode        = RECALL_MODE_NAMES[rt->recall_mode];
        if (rt->count > 0) {
            stats->candidate_ratio = (float)cand_count / (float)rt->count;
            stats->speedup = (cand_count > 0)
                ? (float)rt->count / (float)cand_count
                : (float)rt->count;
        }
    }

    return result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * XVI. PHASE 4: FIELD-AWARE INTEGRATION
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_route_enable_fields(trine_route_t *rt,
                               const trine_field_config_t *fcfg)
{
    if (!rt || !fcfg) return -1;
    if (rt->field_cfg) return 0;  /* Already enabled */

    rt->field_cfg = (trine_field_config_t *)calloc(1,
                        sizeof(trine_field_config_t));
    if (!rt->field_cfg) return -1;
    *rt->field_cfg = *fcfg;

    /* Allocate field embedding storage */
    int cap = rt->capacity > 0 ? rt->capacity : RT_INITIAL_CAPACITY;
    size_t field_bytes = (size_t)cap * (size_t)fcfg->field_count * TRINE_S1_DIMS;
    rt->field_embeddings = (uint8_t *)calloc(field_bytes, 1);
    if (!rt->field_embeddings) {
        free(rt->field_cfg);
        rt->field_cfg = NULL;
        return -1;
    }
    rt->field_emb_capacity = cap;

    return 0;
}

int trine_route_add_fields(trine_route_t *rt,
                            const trine_field_entry_t *entry,
                            const char *tag)
{
    if (!rt || !entry || !rt->field_cfg) return -1;

    /* Route on the primary field embedding */
    const uint8_t *route_emb = trine_field_route_embedding(rt->field_cfg, entry);
    if (!route_emb) return -1;

    /* Add routing embedding via standard path */
    int idx = trine_route_add(rt, route_emb, tag);
    if (idx < 0) return -1;

    /* Grow field embeddings if needed */
    if (idx >= rt->field_emb_capacity) {
        int new_cap = rt->field_emb_capacity * 2;
        if (new_cap <= idx) new_cap = idx + 1;
        size_t new_bytes = (size_t)new_cap * (size_t)rt->field_cfg->field_count
                           * TRINE_S1_DIMS;
        uint8_t *new_femb = (uint8_t *)realloc(rt->field_embeddings, new_bytes);
        if (!new_femb) return idx;  /* Non-fatal: routing still works */
        /* Zero new area */
        size_t old_bytes = (size_t)rt->field_emb_capacity
                           * (size_t)rt->field_cfg->field_count * TRINE_S1_DIMS;
        memset(new_femb + old_bytes, 0, new_bytes - old_bytes);
        rt->field_embeddings = new_femb;
        rt->field_emb_capacity = new_cap;
    }

    /* Copy field embeddings */
    int fc = rt->field_cfg->field_count;
    size_t offset = (size_t)idx * (size_t)fc * TRINE_S1_DIMS;
    for (int f = 0; f < fc && f < entry->field_count; f++) {
        memcpy(rt->field_embeddings + offset + (size_t)f * TRINE_S1_DIMS,
               entry->embeddings[f], TRINE_S1_DIMS);
    }

    return idx;
}

trine_s1_result_t trine_route_query_fields(const trine_route_t *rt,
                                            const trine_field_entry_t *query,
                                            trine_route_stats_t *stats)
{
    trine_s1_result_t result;
    memset(&result, 0, sizeof(result));
    result.matched_index = -1;

    if (!rt || !query || !rt->field_cfg || rt->count == 0) {
        if (stats) {
            memset(stats, 0, sizeof(*stats));
            stats->recall_mode = RECALL_MODE_NAMES[rt ? rt->recall_mode : TRINE_RECALL_BALANCED];
        }
        return result;
    }

    /* Get routing embedding from query */
    const uint8_t *route_emb = trine_field_route_embedding(rt->field_cfg, query);
    if (!route_emb) return result;

    /* Use standard routing to get candidates */
    trine_route_stats_t rstats = {0};
    (void)trine_route_query(rt, route_emb, &rstats);

    /* Re-score candidates with field-weighted cosine.
     * For simplicity, do a brute-force scan of all entries with
     * field-weighted scoring (since routing already filtered). */
    /* TODO: In a future optimization, extract the candidate list
     * from the routing layer. For now, fall back to brute force. */

    int fc = rt->field_cfg->field_count;
    float best_score = -1.0f;
    int   best_idx   = -1;

    for (int i = 0; i < rt->count; i++) {
        /* Build a temporary field entry for this index entry */
        trine_field_entry_t doc_entry;
        memset(&doc_entry, 0, sizeof(doc_entry));
        doc_entry.field_count = fc;

        size_t offset = (size_t)i * (size_t)fc * TRINE_S1_DIMS;
        for (int f = 0; f < fc; f++) {
            memcpy(doc_entry.embeddings[f],
                   rt->field_embeddings + offset + (size_t)f * TRINE_S1_DIMS,
                   TRINE_S1_DIMS);
        }

        float score = trine_field_cosine(query, &doc_entry, rt->field_cfg);
        if (score > best_score) {
            best_score = score;
            best_idx   = i;
        }
    }

    result.similarity = best_score;
    result.calibrated = best_score;
    if (best_score >= rt->config.threshold) {
        result.is_duplicate = 1;
        result.matched_index = best_idx;
    }

    if (stats) {
        *stats = rstats;
    }

    return result;
}

const trine_field_entry_t *trine_route_field_entry(const trine_route_t *rt,
                                                     int index)
{
    /* Cannot return a pointer to a stack-allocated struct.
     * This function is not ideal — callers should use the raw
     * field_embeddings pointer instead. Return NULL. */
    (void)rt;
    (void)index;
    return NULL;
}

const trine_field_config_t *trine_route_field_config(const trine_route_t *rt)
{
    if (!rt) return NULL;
    return rt->field_cfg;
}

/* ═══════════════════════════════════════════════════════════════════════
 * XVII. PHASE 4: ATOMIC SAVE
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_route_save_atomic(const trine_route_t *rt, const char *path)
{
    if (!rt || !path) return -1;

    /* Generate temp path */
    size_t path_len = strlen(path);
    char *tmp_path = (char *)malloc(path_len + 8);
    if (!tmp_path) return -1;
    snprintf(tmp_path, path_len + 8, "%s.tmp", path);

    /* Save to temp file */
    if (trine_route_save(rt, tmp_path) != 0) {
        free(tmp_path);
        return -1;
    }

    /* fsync the temp file */
    FILE *fp = fopen(tmp_path, "rb");
    if (fp) {
#ifdef _WIN32
        fflush(fp);
#else
        int fd = fileno(fp);
        if (fd >= 0) fsync(fd);
#endif
        fclose(fp);
    }

    /* Rename temp to final path (atomic on POSIX) */
    if (rename(tmp_path, path) != 0) {
        fprintf(stderr, "trine_route_save_atomic: rename '%s' -> '%s' failed: %s\n",
                tmp_path, path, strerror(errno));
        /* Try to clean up temp file */
        remove(tmp_path);
        free(tmp_path);
        return -1;
    }

    free(tmp_path);
    return 0;
}
