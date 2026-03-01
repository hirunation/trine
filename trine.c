/* =====================================================================
 * TRINE — Ternary Resonance Interference Network Embedding
 * Runtime Library Implementation v1.0.1
 * =====================================================================
 *
 * This file implements the complete TRINE runtime: model loading,
 * text embedding via ternary algebraic cascade, and comparison.
 *
 * ZERO external dependencies. The algebra comes from trine_algebra.h,
 * text encoding from trine_encode.h, and file format from trine_format.h.
 *
 * Build:
 *   cc -O2 -c trine.c -I../include -o trine.o
 *   cc -O2 -c trine_format.c -I../include -o trine_format.o
 *   cc -O2 -c trine_encode.c -o trine_encode.o
 *   ar rcs libtrine.a trine.o trine_format.o trine_encode.o
 *
 * ===================================================================== */

#include "trine.h"
#include "trine_algebra.h"
#include "trine_encode.h"
/* trine_encode.h defines TRINE_VERSION as "1.0.1" (string) while
 * trine_format.h defines it as 1u (integer). Undefine to avoid collision. */
#undef TRINE_VERSION
#include "trine_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* =====================================================================
 * I. VERSION
 * ===================================================================== */

static const char TRINE_LIB_VERSION[] = "1.0.1";

const char *trine_version(void) {
    return TRINE_LIB_VERSION;
}

/* =====================================================================
 * II. MODEL STRUCTURE (opaque to callers)
 * =====================================================================
 *
 * The model struct holds the loaded .trine file data plus precomputed
 * lookup tables for fast embedding extraction. After trine_load()
 * completes, the model is immutable and safe for concurrent use.
 *
 * The snap arena is stored as trine_snap_t (from trine_algebra.h),
 * which is binary-identical to snap_t (from oicos.h). The file loader
 * returns snap_t; we cast to trine_snap_t for the cascade engine.
 */

struct trine_model {
    /* The loaded .trine file contents */
    trine_file_t file;

    /* Precomputed: input channel snap indices (up to 240 channels).
     * input_snap[ch] = snap index that services input channel ch.
     * Set to TRINE_SNAP_NIL if the channel has no assigned snap. */
    uint32_t input_snaps[TRINE_CHANNELS];

    /* Precomputed: verify output channel snap index.
     * The snap whose I/O output channel == 98 (VERIFY score). */
    uint32_t verify_snap_index;

    /* Precomputed: resolution extraction indices.
     * For each resolution tier, a flat array of snap indices to read.
     * Allocated contiguously for all three tiers. */
    uint32_t *res_indices[TRINE_NUM_RESOLUTIONS];
    uint32_t  res_dims[TRINE_NUM_RESOLUTIONS];
};

/* =====================================================================
 * III. MODEL LOADING
 * =====================================================================
 *
 * trine_load() reads the .trine file, validates it, and precomputes
 * the input channel map and resolution extraction indices.
 */

/* Build the input_snaps[] lookup from the I/O channel map.
 * For each input channel (direction==0), records the snap index. */
static void build_input_map(trine_model_t *model) {
    /* Initialize all channels to NIL */
    for (int i = 0; i < TRINE_CHANNELS; i++)
        model->input_snaps[i] = TRINE_SNAP_NIL;

    for (int i = 0; i < (int)model->file.header.io_channel_count; i++) {
        const trine_io_channel_t *ch = &model->file.io_channels[i];
        if (ch->direction == 0 && ch->channel_id < TRINE_CHANNELS) {
            model->input_snaps[ch->channel_id] = ch->snap_index;
        }
    }
}

/* Find the snap index assigned to output channel 98 (VERIFY score).
 * Falls back to TRINE_SNAP_NIL if not found. */
static void find_verify_snap(trine_model_t *model) {
    model->verify_snap_index = TRINE_SNAP_NIL;

    for (int i = 0; i < (int)model->file.header.io_channel_count; i++) {
        const trine_io_channel_t *ch = &model->file.io_channels[i];
        if (ch->direction == 1 && ch->channel_id == 98) {
            model->verify_snap_index = ch->snap_index;
            return;
        }
    }
}

/* Build the resolution extraction index arrays.
 * For each resolution tier, expand the ranges into a flat array
 * of snap indices (capped at dim_count). */
static int build_resolution_indices(trine_model_t *model) {
    for (int r = 0; r < TRINE_NUM_RESOLUTIONS; r++) {
        const trine_resolution_t *res = &model->file.resolutions[r];
        uint32_t dims = res->dim_count;
        model->res_dims[r] = dims;

        if (dims == 0) {
            model->res_indices[r] = NULL;
            continue;
        }

        model->res_indices[r] = (uint32_t *)malloc(
            (size_t)dims * sizeof(uint32_t));
        if (!model->res_indices[r]) {
            fprintf(stderr, "trine: allocation failed for resolution %d "
                    "indices (%u dims)\n", r, dims);
            return -1;
        }

        /* Expand ranges into the flat array, stopping at dim_count */
        uint32_t pos = 0;
        for (uint32_t ri = 0; ri < res->range_count && pos < dims; ri++) {
            uint32_t start = res->ranges[ri][0];
            uint32_t end   = res->ranges[ri][1];
            for (uint32_t idx = start; idx < end && pos < dims; idx++) {
                model->res_indices[r][pos++] = idx;
            }
        }

        /* If ranges didn't fill all dims (shouldn't happen in valid
         * models), pad remaining with the last valid index */
        if (pos > 0) {
            uint32_t last = model->res_indices[r][pos - 1];
            while (pos < dims) {
                model->res_indices[r][pos++] = last;
            }
        }
    }

    return 0;
}

trine_model_t *trine_load(const char *path) {
    if (!path) {
        fprintf(stderr, "trine_load: null path\n");
        return NULL;
    }

    /* Allocate model struct */
    trine_model_t *model = (trine_model_t *)calloc(1, sizeof(trine_model_t));
    if (!model) {
        fprintf(stderr, "trine_load: allocation failed\n");
        return NULL;
    }

    /* Read and validate the .trine file */
    if (trine_file_read(path, &model->file) != 0) {
        free(model);
        return NULL;
    }

    /* Validate model integrity (ROM tables, layer boundaries, etc.) */
    if (trine_file_validate(&model->file) != 0) {
        trine_file_free(&model->file);
        free(model);
        return NULL;
    }

    /* Verify the standalone algebra ROM matches the file's ROM.
     * This ensures the trine_algebra.h tables we'll use for cascade
     * computation are consistent with the model's embedded ROM. */
    if (!trine_verify()) {
        fprintf(stderr, "trine_load: algebra self-check failed\n");
        trine_file_free(&model->file);
        free(model);
        return NULL;
    }

    /* Build precomputed lookup structures */
    build_input_map(model);
    find_verify_snap(model);

    if (build_resolution_indices(model) != 0) {
        /* Cleanup partially built indices */
        for (int r = 0; r < TRINE_NUM_RESOLUTIONS; r++) {
            free(model->res_indices[r]);
            model->res_indices[r] = NULL;
        }
        trine_file_free(&model->file);
        free(model);
        return NULL;
    }

    return model;
}

void trine_free(trine_model_t *model) {
    if (!model) return;

    for (int r = 0; r < TRINE_NUM_RESOLUTIONS; r++) {
        free(model->res_indices[r]);
        model->res_indices[r] = NULL;
    }

    trine_file_free(&model->file);
    free(model);
}

/* =====================================================================
 * IV. CORE EMBEDDING
 * =====================================================================
 *
 * trine_embed() is the heart of the library. It:
 *   1. Encodes text into 240 I/O channel trits (4 chains x 60 channels)
 *   2. Copies the model's snap arena into a private workspace
 *   3. Pre-loads input channels into the I/O snap data fields
 *   4. Seeds the cascade wave from the root snap
 *   5. Runs the cascade for the appropriate number of ticks
 *   6. Extracts the embedding from resolution-appropriate snaps
 *   7. Reads the VERIFY consistency score from I/O channel 98
 *
 * Thread safety: The model is read-only. Each call allocates its own
 * workspace (arena copy + cascade buffers), so concurrent calls are safe.
 */

/* Tick counts per resolution tier.
 * Screening needs fewer ticks because it only reads early-stage snaps.
 * Deep needs more ticks to ensure all downstream layers converge. */
#define TICKS_SCREENING   50
#define TICKS_STANDARD   100
#define TICKS_DEEP       200

static const int TICKS_PER_RESOLUTION[3] = {
    TICKS_SCREENING,
    TICKS_STANDARD,
    TICKS_DEEP
};

/* Pre-load the encoded text trits into the I/O channel array.
 *
 * The encoding layer produces 240 trits across 4 chains:
 *   Chain 1 [  0.. 59]: Forward encoding
 *   Chain 2 [ 60..119]: Reverse encoding
 *   Chain 3 [120..179]: Differential encoding
 *   Chain 4 [180..239]: Structural encoding
 *
 * The model's I/O channel map tells us which snap services each
 * input channel. For channels 0-59, the forward chain trits are
 * loaded directly. Channels 60-119, 120-179, 180-239 correspond
 * to the other chains.
 *
 * The I/O channel array (256 entries) is used by the cascade engine:
 * DOM_IO snaps read their input trit from io_channels[TRINE_IO_IN(s)]
 * and write their output to io_channels[TRINE_IO_OUT(s)].
 *
 * We load the first 240 channels with the encoded trits. The remaining
 * channels (240-255) are zeroed and available for output. */
static void preload_io_channels(uint8_t io_channels[256],
                                const uint8_t encoded[TRINE_CHANNELS]) {
    memset(io_channels, 0, 256);

    /* Load all 240 encoded trits into I/O channel positions 0-239.
     * The cascade engine reads io_channels[TRINE_IO_IN(snap)] for
     * DOM_IO snaps. The INGRESS layer snaps have their data field
     * set so that TRINE_IO_IN(snap) == their assigned channel. */
    for (int i = 0; i < TRINE_CHANNELS; i++) {
        io_channels[i] = encoded[i];
    }
}

/* Seed the cascade wave with a single root snap.
 *
 * The composed topology has a single root (the kernel entry point).
 * We mark it STAT_LIVE and add it to the initial wave. All other
 * snaps that need to be in the initial wave should already be marked
 * STAT_LIVE in the snap arena from the .trine file. The cascade init
 * function seeds from all STAT_LIVE snaps. */
static void seed_cascade(trine_snap_t *arena, uint32_t snap_count,
                          uint32_t root_index) {
    /* Ensure root is marked LIVE so the cascade picks it up */
    if (root_index < snap_count) {
        trine_snap_set_stat(&arena[root_index], TRINE_STAT_LIVE);
    }
}

/* Extract embedding trits from the cascade-processed arena.
 *
 * For each snap index in the resolution's extraction list, read the
 * snap's current FSM state (2 bits, value in {0, 1, 2}). This state
 * is the result of the snap having been driven by the cascade wave
 * carrying the encoded text signal through the topology.
 *
 * The FSM state captures the snap's response to its input history,
 * which is determined by its position in the topology, its cell type,
 * and the propagated signal from the text encoding. */
static void extract_embedding(const trine_snap_t *arena,
                               const uint32_t *indices,
                               uint32_t dim_count,
                               uint8_t *out_trits) {
    for (uint32_t i = 0; i < dim_count; i++) {
        uint32_t idx = indices[i];
        out_trits[i] = trine_snap_st(&arena[idx]);
    }
}

/* Read the VERIFY layer's consistency score from the cascade output.
 *
 * The VERIFY layer compares original INGRESS signals against
 * reconstructed signals (from the RECONSTRUCT layer). The final
 * summary snap outputs a trit to I/O channel 98:
 *   0 = no agreement (inconsistent)
 *   1 = partial agreement
 *   2 = full agreement (consistent)
 *
 * If the verify snap is not found or not reachable, returns 0. */
static uint32_t read_verify_score(const trine_snap_t *arena,
                                   uint32_t snap_count,
                                   uint32_t verify_snap_index,
                                   const uint8_t io_channels[256]) {
    /* Primary method: read from I/O channel 98 (set by cascade) */
    uint8_t score = io_channels[98];
    if (score <= 2) return score;

    /* Fallback: read the verify snap's FSM state directly */
    if (verify_snap_index != TRINE_SNAP_NIL &&
        verify_snap_index < snap_count) {
        return trine_snap_st(&arena[verify_snap_index]);
    }

    return 0;
}

int trine_embed(trine_model_t *model,
                const char *text, size_t len,
                int resolution,
                trine_embedding_t *out) {
    /* -----------------------------------------------------------
     * 1. Validate arguments
     * ----------------------------------------------------------- */
    if (!model || !out) {
        fprintf(stderr, "trine_embed: null argument\n");
        return -1;
    }

    if (resolution < TRINE_SCREENING || resolution > TRINE_SHINGLE) {
        fprintf(stderr, "trine_embed: invalid resolution %d "
                "(must be 0-3)\n", resolution);
        return -1;
    }

    /* Zero the output struct */
    memset(out, 0, sizeof(*out));

    /* -----------------------------------------------------------
     * 2. Encode text to 240 I/O channel trits
     * ----------------------------------------------------------- */
    uint8_t encoded[TRINE_CHANNELS];
    if (text && len > 0) {
        trine_encode_shingle(text, len, encoded);
    } else {
        /* Empty text: all-zero encoding (PAD-filled) */
        memset(encoded, 0, sizeof(encoded));
    }

    /* -----------------------------------------------------------
     * 2b. SHINGLE tier: return encoded channels directly
     *
     * The shingle tier bypasses the cascade entirely, returning
     * the 240-dimensional n-gram channel encoding as the embedding.
     * This preserves locality: texts sharing n-grams share channel
     * values, producing meaningful cosine similarity.
     *
     * Use for: similarity search, nearest-neighbor, clustering.
     * The cascade tiers (0-2) remain for fingerprinting/dedup.
     * ----------------------------------------------------------- */
    if (resolution == TRINE_SHINGLE) {
        uint8_t *trits = (uint8_t *)malloc(TRINE_CHANNELS);
        if (!trits) {
            fprintf(stderr, "trine_embed: allocation failed (240 dims)\n");
            return -2;
        }

        memcpy(trits, encoded, TRINE_CHANNELS);

        out->trits       = trits;
        out->dims        = TRINE_CHANNELS;
        out->resolution  = TRINE_SHINGLE;
        out->consistency = 2;    /* N/A for shingle tier; report full */
        out->ticks       = 0;    /* No cascade ticks */
        return 0;
    }

    /* -----------------------------------------------------------
     * 3. Cascade tiers (screening/standard/deep): full pipeline
     * ----------------------------------------------------------- */

    /* Validate resolution tier has indices */
    uint32_t dim_count = model->res_dims[resolution];
    if (dim_count == 0 || !model->res_indices[resolution]) {
        fprintf(stderr, "trine_embed: resolution %d has no extraction "
                "indices\n", resolution);
        return -1;
    }

    /* Copy model's snap arena into private workspace.
     * The cascade modifies snap states in place (FSM state, gen
     * counters, rank degradation). We need a private copy so the
     * model remains immutable and concurrent calls are safe. */
    uint32_t snap_count = model->file.snap_count;
    size_t arena_bytes = (size_t)snap_count * sizeof(trine_snap_t);

    trine_snap_t *arena = (trine_snap_t *)aligned_alloc(32, arena_bytes);
    if (!arena) {
        fprintf(stderr, "trine_embed: arena allocation failed "
                "(%u snaps, %zu bytes)\n", snap_count, arena_bytes);
        return -2;
    }

    memcpy(arena, model->file.snaps, arena_bytes);

    /* Prepare I/O channels and seed the cascade */
    uint8_t io_channels[256];
    preload_io_channels(io_channels, encoded);
    seed_cascade(arena, snap_count, model->file.header.root_index);

    /* Initialize and run the cascade */
    trine_cascade_t cas;
    if (trine_cascade_init(&cas, arena, snap_count) != 0) {
        fprintf(stderr, "trine_embed: cascade init failed\n");
        free(arena);
        return -3;
    }

    int max_ticks = TICKS_PER_RESOLUTION[resolution];
    uint32_t total_emit = 0;

    uint32_t final_tick = trine_cascade_run_full(
        &cas, (uint32_t)max_ticks, io_channels, &total_emit);

    /* Extract the embedding */
    uint8_t *trits = (uint8_t *)malloc((size_t)dim_count);
    if (!trits) {
        fprintf(stderr, "trine_embed: embedding allocation failed "
                "(%u dims)\n", dim_count);
        trine_cascade_free(&cas);
        free(arena);
        return -2;
    }

    extract_embedding(arena, model->res_indices[resolution],
                      dim_count, trits);

    /* Read consistency score from VERIFY layer */
    uint32_t consistency = read_verify_score(
        arena, snap_count, model->verify_snap_index, io_channels);

    /* Populate output struct */
    out->trits       = trits;
    out->dims        = (int)dim_count;
    out->resolution  = resolution;
    out->consistency = consistency;
    out->ticks       = (int)final_tick;

    /* Cleanup workspace (model remains intact) */
    trine_cascade_free(&cas);
    free(arena);

    return 0;
}

void trine_embedding_free(trine_embedding_t *emb) {
    if (!emb) return;
    free(emb->trits);
    memset(emb, 0, sizeof(*emb));
}

/* =====================================================================
 * V. COMPARISON FUNCTIONS
 * =====================================================================
 *
 * Ternary cosine similarity and Hamming distance. These operate on
 * the raw trit vectors, treating each trit as a value in {0, 1, 2}.
 */

double trine_compare(const trine_embedding_t *a,
                     const trine_embedding_t *b) {
    if (!a || !b || !a->trits || !b->trits) return -1.0;
    if (a->dims != b->dims) return -1.0;
    if (a->dims == 0) return 0.0;

    /* Cosine similarity: dot(a,b) / (|a| * |b|)
     *
     * Treating trit values as real-valued vector components:
     *   dot(a,b) = sum(a[i] * b[i])
     *   |a|      = sqrt(sum(a[i]^2))
     *   |b|      = sqrt(sum(b[i]^2))
     *
     * All values are non-negative (0, 1, 2), so the result is
     * always in [0, 1] (no negative cosine possible). */
    uint64_t dot_ab = 0;
    uint64_t mag_a  = 0;
    uint64_t mag_b  = 0;

    int n = a->dims;
    const uint8_t *ta = a->trits;
    const uint8_t *tb = b->trits;

    for (int i = 0; i < n; i++) {
        uint64_t va = ta[i];
        uint64_t vb = tb[i];
        dot_ab += va * vb;
        mag_a  += va * va;
        mag_b  += vb * vb;
    }

    if (mag_a == 0 || mag_b == 0) return 0.0;

    double denom = sqrt((double)mag_a) * sqrt((double)mag_b);
    if (denom == 0.0) return 0.0;

    double sim = (double)dot_ab / denom;

    /* Clamp to [0, 1] to handle floating-point rounding */
    if (sim > 1.0) sim = 1.0;
    if (sim < 0.0) sim = 0.0;

    return sim;
}

int trine_hamming(const trine_embedding_t *a,
                  const trine_embedding_t *b) {
    if (!a || !b || !a->trits || !b->trits) return -1;
    if (a->dims != b->dims) return -1;

    int dist = 0;
    int n = a->dims;
    const uint8_t *ta = a->trits;
    const uint8_t *tb = b->trits;

    for (int i = 0; i < n; i++) {
        if (ta[i] != tb[i]) dist++;
    }

    return dist;
}

/* =====================================================================
 * VI. LENS COMPARISON
 * =====================================================================
 *
 * Weighted per-chain cosine similarity for TRINE_SHINGLE embeddings.
 * The 240 channels are divided into 4 chains of 60 channels each.
 * A lens assigns importance weights to each chain, enabling
 * application-specific similarity tuning (edit distance, morphology,
 * phrase matching, vocabulary overlap, deduplication, etc.).
 */

/* Cosine similarity over a single chain slice [offset, offset+width).
 * Same math as trine_compare but restricted to a contiguous subrange.
 * Returns 0.0 if either vector has zero magnitude in the slice. */
static double chain_cosine(const uint8_t *a, const uint8_t *b,
                           int offset, int width) {
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

    if (mag_a == 0 || mag_b == 0) return 0.0;

    double denom = sqrt((double)mag_a) * sqrt((double)mag_b);
    if (denom == 0.0) return 0.0;

    double sim = (double)dot_ab / denom;

    /* Clamp to [0, 1] to handle floating-point rounding */
    if (sim > 1.0) sim = 1.0;
    if (sim < 0.0) sim = 0.0;

    return sim;
}

double trine_compare_lens(const trine_embedding_t *a,
                           const trine_embedding_t *b,
                           const trine_lens_t *lens) {
    if (!a || !b || !lens || !a->trits || !b->trits) return -1.0;
    if (a->dims != TRINE_CHANNELS || b->dims != TRINE_CHANNELS) return -1.0;

    /* Compute per-chain cosine similarities */
    double weighted_sum = 0.0;
    double weight_sum   = 0.0;

    for (int c = 0; c < TRINE_NUM_CHAINS; c++) {
        double w = (double)lens->weights[c];
        if (w <= 0.0) continue;

        double cos_c = chain_cosine(a->trits, b->trits,
                                    c * TRINE_CHAIN_WIDTH,
                                    TRINE_CHAIN_WIDTH);
        weighted_sum += w * cos_c;
        weight_sum   += w;
    }

    if (weight_sum == 0.0) return 0.0;

    return weighted_sum / weight_sum;
}

int trine_compare_detail(const trine_embedding_t *a,
                          const trine_embedding_t *b,
                          const trine_lens_t *lens,
                          trine_similarity_t *out) {
    if (!a || !b || !lens || !out || !a->trits || !b->trits) return -1;
    if (a->dims != TRINE_CHANNELS || b->dims != TRINE_CHANNELS) return -1;

    /* Per-chain cosines */
    for (int c = 0; c < TRINE_NUM_CHAINS; c++) {
        out->chain[c] = (float)chain_cosine(a->trits, b->trits,
                                            c * TRINE_CHAIN_WIDTH,
                                            TRINE_CHAIN_WIDTH);
    }

    /* Weighted combination */
    double weighted_sum = 0.0;
    double weight_sum   = 0.0;

    for (int c = 0; c < TRINE_NUM_CHAINS; c++) {
        double w = (double)lens->weights[c];
        if (w <= 0.0) continue;

        weighted_sum += w * (double)out->chain[c];
        weight_sum   += w;
    }

    out->combined = (weight_sum > 0.0)
                  ? (float)(weighted_sum / weight_sum)
                  : 0.0f;

    /* Flat (uniform) cosine — delegate to existing function */
    double flat = trine_compare(a, b);
    out->uniform = (flat >= 0.0) ? (float)flat : 0.0f;

    return 0;
}

/* =====================================================================
 * VII. MODEL INTROSPECTION
 * ===================================================================== */

int trine_info(const trine_model_t *model, trine_info_t *info) {
    if (!model || !info) return -1;

    info->snap_count  = model->file.snap_count;
    info->version     = model->file.header.version;
    info->layer_count = (int)model->file.header.layer_count;
    info->model_name  = "TRINE v1";

    return 0;
}
