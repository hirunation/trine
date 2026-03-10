/* ═══════════════════════════════════════════════════════════════════════
 * TRINE — Ternary Resonance Interference Network Embedding
 * Band-LSH Routing Overlay v1.0.1
 * ═══════════════════════════════════════════════════════════════════════
 *
 * OVERVIEW
 *   Locality-Sensitive Hashing (LSH) routing layer that sits on top of
 *   the Stage-1 index. Reduces per-query comparisons by 10-50x without
 *   hurting recall, enabling sub-linear scaling for large corpora.
 *
 * DESIGN
 *   Each of the 4 TRINE chains (60 trits) is hashed to a bucket key
 *   using FNV-1a. Embeddings sharing at least one bucket key with a
 *   query become candidates for full lens-weighted cosine comparison.
 *
 *   Multi-probe: at query time, each band is probed not only at its
 *   primary key but also at TRINE_ROUTE_PROBES additional keys formed
 *   by flipping specific trit positions before hashing. This catches
 *   near-miss embeddings that land in adjacent buckets.
 *
 * PRODUCTION HARDENING (v1.1.0)
 *   - Small-N fallback: when count < TRINE_ROUTE_FALLBACK_THRESHOLD,
 *     queries fall back to brute-force scan for perfect recall.
 *   - Candidate cap: bounds worst-case query time regardless of bucket
 *     distribution, protecting tail latency.
 *   - Recall presets: FAST / BALANCED / STRICT trade speed for recall.
 *
 * THREAD SAFETY
 *   Not thread-safe. Callers must synchronize access to a shared index.
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#ifndef TRINE_ROUTE_H
#define TRINE_ROUTE_H

#include "trine_stage1.h"
#include "trine_csidf.h"
#include "trine_field.h"

#define TRINE_ROUTE_VERSION "1.0.1"
#define TRINE_ROUTE_BANDS     4
#define TRINE_ROUTE_BUCKETS   4096   /* Hash table size per band */
#define TRINE_ROUTE_PROBES    3      /* Multi-probe count per band (default) */

/* Below this entry count, queries use brute-force scan for perfect recall */
#define TRINE_ROUTE_FALLBACK_THRESHOLD 500

/* Feature flags (stored in .trrt v4 header) */
#define TRINE_ROUTE_FLAG_CSIDF   0x01   /* Corpus-specific IDF section present */
#define TRINE_ROUTE_FLAG_FIELDS  0x02   /* Field-aware section present */

/* Recall presets — trade speed for recall completeness */
#define TRINE_RECALL_FAST      0   /* 1 probe/band, cap 200 — maximum speed */
#define TRINE_RECALL_BALANCED  1   /* 3 probes/band, cap 500 — default */
#define TRINE_RECALL_STRICT    2   /* 5 probes/band, cap 2000 — maximum recall */

typedef struct trine_route trine_route_t;

typedef struct {
    int candidates_checked;    /* How many full comparisons were done */
    int total_entries;         /* Total entries in the index */
    float candidate_ratio;    /* candidates_checked / total_entries */
    float speedup;            /* total_entries / candidates_checked */
    const char *recall_mode;  /* Name of active recall preset (or NULL) */
} trine_route_stats_t;

/* Create a routed index. Config controls comparison (lens, threshold, calibration). */
trine_route_t *trine_route_create(const trine_s1_config_t *config);

/* Add an embedding to the routed index. Returns assigned index (0-based), -1 on error.
 * Optional tag string (can be NULL). */
int trine_route_add(trine_route_t *rt, const uint8_t emb[240], const char *tag);

/* Query: find best match using routing for candidate selection.
 * Only entries sharing at least one bucket with the query are compared.
 * stats (if non-NULL) receives query statistics. */
trine_s1_result_t trine_route_query(const trine_route_t *rt,
                                     const uint8_t candidate[240],
                                     trine_route_stats_t *stats);

/* Get entry count. */
int trine_route_count(const trine_route_t *rt);

/* Get tag for an entry. */
const char *trine_route_tag(const trine_route_t *rt, int index);

/* Get the raw embedding for an entry. */
const uint8_t *trine_route_embedding(const trine_route_t *rt, int index);

/* Free the routed index. */
void trine_route_free(trine_route_t *rt);

/* Save/load routed index to/from file. Format (v3):
 *   Magic:        "TRRT" (4 bytes)
 *   Version:      uint32_t = 3
 *   Endian check: uint32_t = 0x01020304 (byte order marker)
 *   Flags:        uint32_t = 0 (reserved for future use)
 *   Checksum:     uint64_t = FNV-1a over payload after header
 *   --- payload ---
 *   Count + config + recall_mode + entries + bucket tables
 *
 * v1 files (no recall_mode, no checksum) and v2 files (recall_mode, no
 * checksum) are still loadable.
 * Returns 0 on success, -1 on error. */
int trine_route_save(const trine_route_t *rt, const char *path);
trine_route_t *trine_route_load(const char *path);

/* Global statistics for the index (averages over all buckets). */
void trine_route_global_stats(const trine_route_t *rt, trine_route_stats_t *stats);

/* Set the recall mode for a routed index. Adjusts probes and candidate cap.
 * Can be called at any time (takes effect on next query).
 * Returns 0 on success, -1 on invalid mode. */
int trine_route_set_recall(trine_route_t *rt, int mode);

/* Get current recall mode. */
int trine_route_get_recall(const trine_route_t *rt);

/* Get bucket occupancy for a specific band.
 * sizes must point to an array of at least TRINE_ROUTE_BUCKETS ints.
 * Each element receives the number of entries in that bucket slot.
 * Returns 0 on success, -1 on invalid arguments. */
int trine_route_bucket_sizes(const trine_route_t *rt, int band, int *sizes);

/* ═══════════════════════════════════════════════════════════════════════
 * Phase 4: CS-IDF Integration
 * ═══════════════════════════════════════════════════════════════════════ */

/* Enable corpus-specific IDF tracking for this index.
 * Must be called before adding entries. Enables DF counting per add().
 * Returns 0 on success, -1 on error. */
int trine_route_enable_csidf(trine_route_t *rt);

/* Compute CS-IDF weights from accumulated DF counters.
 * Call after all entries have been added (or after append).
 * Returns 0 on success, -1 on error (no CS-IDF enabled or no docs). */
int trine_route_compute_csidf(trine_route_t *rt);

/* Get the computed CS-IDF weights (or NULL if not enabled/computed).
 * Returns pointer to internal trine_csidf_t (do not free). */
const trine_csidf_t *trine_route_get_csidf(const trine_route_t *rt);

/* Query using CS-IDF weighted comparison instead of standard lens cosine. */
trine_s1_result_t trine_route_query_csidf(const trine_route_t *rt,
                                           const uint8_t candidate[240],
                                           trine_route_stats_t *stats);

/* ═══════════════════════════════════════════════════════════════════════
 * Phase 4: Field-Aware Integration
 * ═══════════════════════════════════════════════════════════════════════ */

/* Enable field-aware mode for this index.
 * Must be called before adding entries.
 * Returns 0 on success, -1 on error. */
int trine_route_enable_fields(trine_route_t *rt,
                               const trine_field_config_t *fcfg);

/* Add a field-aware entry. Routes on the primary field embedding.
 * Returns assigned index, -1 on error. */
int trine_route_add_fields(trine_route_t *rt,
                            const trine_field_entry_t *entry,
                            const char *tag);

/* Query with field-aware scoring.
 * Uses field-weighted cosine on candidates from routing. */
trine_s1_result_t trine_route_query_fields(const trine_route_t *rt,
                                            const trine_field_entry_t *query,
                                            trine_route_stats_t *stats);

/* Get field entry for an index entry. Returns NULL if fields not enabled. */
const trine_field_entry_t *trine_route_field_entry(const trine_route_t *rt,
                                                     int index);

/* Get field config. Returns NULL if fields not enabled. */
const trine_field_config_t *trine_route_field_config(const trine_route_t *rt);

/* ═══════════════════════════════════════════════════════════════════════
 * Phase 4: Append Mode
 * ═══════════════════════════════════════════════════════════════════════ */

/* Save index atomically (write to temp file, fsync, rename).
 * If path is NULL, uses the path from which the index was loaded.
 * Returns 0 on success, -1 on error. */
int trine_route_save_atomic(const trine_route_t *rt, const char *path);

#endif /* TRINE_ROUTE_H */
