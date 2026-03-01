/* =====================================================================
 * TRINE EMBED — Command-Line Text Embedding Tool
 * Ternary Resonance Interference Network Embedding
 * =====================================================================
 *
 * Standalone CLI tool that loads a .trine model file and computes
 * text embeddings using a minimal cascade engine built on
 * trine_algebra.h inline functions.
 *
 * ZERO external dependencies beyond libc. The .trine file IS the model.
 *
 * Usage:
 *   trine_embed <model.trine> [options] <text>
 *   trine_embed <model.trine> [options] -f <file>
 *   trine_embed <model.trine> --compare <text1> <text2>
 *   trine_embed <model.trine> --batch -f <file>
 *   trine_embed <model.trine> --info
 *   trine_embed <model.trine> --benchmark N
 *
 * Build:
 *   cc -O2 -o trine_embed trine_embed.c trine_format.c trine_encode.c \
 *      -I../include -lm
 *
 * ===================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>

/* Include order matters: trine_format.h and trine_encode.h both define
 * TRINE_VERSION (integer vs string). We include trine_algebra.h first
 * (standalone, no TRINE_VERSION), then trine_format.h (defines it as 1u),
 * then undef before trine_encode.h (redefines it as "1.0.1"). */
#include "trine_algebra.h"
#include "trine_format.h"
#undef TRINE_VERSION
#include "trine_encode.h"
#include "trine_idf.h"

/* =====================================================================
 * Constants
 * ===================================================================== */

#define EMBED_VERSION       "1.0.1"
#define EMBED_MAX_TEXT       (1 << 20)   /* 1 MB max text input           */
#define EMBED_MAX_LINE       4096        /* Max line length for batch     */
#define EMBED_IO_CHANNELS    256         /* I/O channel array size        */

/* Resolution tick counts */
#define EMBED_TICKS_SCREENING  50
#define EMBED_TICKS_STANDARD   100
#define EMBED_TICKS_DEEP       200

/* Output format codes */
#define FMT_TRITS   0
#define FMT_HEX     1
#define FMT_BASE64  2
#define FMT_JSON    3

/* =====================================================================
 * Resolution name table
 * ===================================================================== */

static const char * const RESOLUTION_NAME[4] = {
    "screening", "standard", "deep", "shingle"
};

static const int RESOLUTION_TICKS[4] = {
    EMBED_TICKS_SCREENING, EMBED_TICKS_STANDARD, EMBED_TICKS_DEEP, 0
};

/* Chain layout for shingle (240-dim) embeddings: 4 chains of 60 channels */
#define CHAIN_COUNT  4
#define CHAIN_WIDTH  60

static const char * const CHAIN_NAME[CHAIN_COUNT] = {
    "edit", "morph", "phrase", "vocab"
};

/* Per-chain cosine similarity (over a width-channel slice) */
static double chain_cosine_local(const uint8_t *a, const uint8_t *b,
                                  int offset, int width) {
    double dot = 0.0, ma = 0.0, mb = 0.0;
    for (int i = 0; i < width; i++) {
        double va = (double)a[offset + i];
        double vb = (double)b[offset + i];
        dot   += va * vb;
        ma += va * va;
        mb += vb * vb;
    }
    if (ma < 1e-12 || mb < 1e-12) return 0.0;
    return dot / (sqrt(ma) * sqrt(mb));
}

/* Per-chain IDF-weighted cosine similarity (over a width-channel slice) */
static double chain_cosine_idf(const uint8_t *a, const uint8_t *b,
                                int offset, int width,
                                const float *idf_weights) {
    double dot = 0.0, ma = 0.0, mb = 0.0;
    for (int i = 0; i < width; i++) {
        double w  = (double)idf_weights[offset + i];
        double va = (double)a[offset + i];
        double vb = (double)b[offset + i];
        dot += w * va * vb;
        ma  += w * va * va;
        mb  += w * vb * vb;
    }
    if (ma < 1e-12 || mb < 1e-12) return 0.0;
    return dot / (sqrt(ma) * sqrt(mb));
}

/* Parse a lens specification string.
 * Accepts preset names or custom "w1,w2,w3,w4" format.
 * Returns 0 on success, -1 on error. */
static int parse_lens(const char *str, float weights[4]) {
    if (strcmp(str, "uniform") == 0) { weights[0]=1;weights[1]=1;weights[2]=1;weights[3]=1; return 0; }
    if (strcmp(str, "edit") == 0)    { weights[0]=1;weights[1]=0.3f;weights[2]=0.1f;weights[3]=0; return 0; }
    if (strcmp(str, "morph") == 0)   { weights[0]=0.3f;weights[1]=1;weights[2]=0.5f;weights[3]=0.2f; return 0; }
    if (strcmp(str, "phrase") == 0)  { weights[0]=0.1f;weights[1]=0.5f;weights[2]=1;weights[3]=0.3f; return 0; }
    if (strcmp(str, "vocab") == 0)   { weights[0]=0;weights[1]=0.2f;weights[2]=0.3f;weights[3]=1; return 0; }
    if (strcmp(str, "dedup") == 0)   { weights[0]=0.5f;weights[1]=0.5f;weights[2]=0.7f;weights[3]=1; return 0; }
    if (strcmp(str, "code") == 0)    { weights[0]=1.0f;weights[1]=0.8f;weights[2]=0.4f;weights[3]=0.2f; return 0; }
    if (strcmp(str, "legal") == 0)   { weights[0]=0.2f;weights[1]=0.4f;weights[2]=1.0f;weights[3]=0.8f; return 0; }
    if (strcmp(str, "medical") == 0) { weights[0]=0.3f;weights[1]=1.0f;weights[2]=0.6f;weights[3]=0.5f; return 0; }
    if (strcmp(str, "support") == 0) { weights[0]=0.2f;weights[1]=0.4f;weights[2]=0.7f;weights[3]=1.0f; return 0; }
    if (strcmp(str, "policy") == 0)  { weights[0]=0.1f;weights[1]=0.3f;weights[2]=1.0f;weights[3]=0.8f; return 0; }
    /* Try parsing as "w1,w2,w3,w4" */
    if (sscanf(str, "%f,%f,%f,%f", &weights[0], &weights[1], &weights[2], &weights[3]) == 4) return 0;
    return -1;
}

/* =====================================================================
 * Base64 encoding table
 * ===================================================================== */

static const char B64_TABLE[65] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/* =====================================================================
 * Internal: Embedding Result
 * ===================================================================== */

typedef struct {
    uint8_t *trits;         /* Trit values (0, 1, or 2), heap-allocated */
    int      dims;          /* Number of embedding dimensions           */
    int      resolution;    /* 0=screening, 1=standard, 2=deep         */
    uint8_t  consistency;   /* Consistency score from I/O channel 98    */
    int      ticks;         /* Cascade ticks executed                   */
    int      final_wave;    /* Final wave size (0 = cascade completed)  */
} embed_result_t;

/* =====================================================================
 * Internal: Load .trine model
 *
 * Wraps trine_file_read() with user-facing error messages.
 * ===================================================================== */

static int load_model(const char *path, trine_file_t *model) {
    int rc = trine_file_read(path, model);
    if (rc != 0) {
        fprintf(stderr, "trine_embed: failed to load model '%s'\n", path);
        return 1;
    }
    return 0;
}

/* =====================================================================
 * Internal: Compute a single embedding
 *
 * This is the core pipeline:
 *   1. Encode text into 240 I/O channels
 *   2. Copy snap arena (working copy)
 *   3. Pre-load I/O channels from encoding
 *   4. Initialize cascade engine (seed wave from root)
 *   5. Run cascade for the appropriate tick count
 *   6. Extract embedding from final snap states
 *   7. Read consistency from I/O channel 98
 *
 * The cascade engine uses trine_algebra.h inline functions:
 *   - trine_snap_step() / trine_snap_step_adaptive()
 *   - Level-synchronous BFS wave propagation
 *   - Generation-based dedup (snap.gen field)
 *   - I/O channel reading for DOM_IO snaps
 *   - Broadcast flag handling
 * ===================================================================== */

static int compute_embedding(const trine_file_t *model,
                              const char *text, size_t text_len,
                              int resolution, int verbose,
                              embed_result_t *out) {
    memset(out, 0, sizeof(*out));
    out->resolution = resolution;

    /* --- 1. Encode text into 240 I/O channels (shingle v2.0) --- */
    uint8_t encoding[TRINE_CHANNELS];
    trine_encode_shingle(text, text_len, encoding);

    if (verbose) {
        /* Count non-zero channels as a fill-rate diagnostic */
        int filled = 0;
        for (int ci = 0; ci < TRINE_CHANNELS; ci++)
            if (encoding[ci] != 0) filled++;
        fprintf(stderr, "  Encoding: %zu chars, %d/%d channels active (shingle)\n",
                text_len, filled, TRINE_CHANNELS);
    }

    /* --- 1b. SHINGLE tier: return encoded channels directly --- */
    if (resolution == TRINE_RES_SHINGLE) {
        out->trits = (uint8_t *)malloc(TRINE_CHANNELS);
        if (!out->trits) {
            fprintf(stderr, "trine_embed: allocation failed (240 dims)\n");
            return -2;
        }
        memcpy(out->trits, encoding, TRINE_CHANNELS);
        out->dims = TRINE_CHANNELS;
        out->consistency = 2;
        out->ticks = 0;
        if (verbose)
            fprintf(stderr, "  Shingle: %d dims (direct channel readout, 0 ticks)\n",
                    TRINE_CHANNELS);
        return 0;
    }

    /* --- 2. Copy snap arena (working copy) --- */
    uint32_t snap_count = model->snap_count;
    size_t arena_bytes = (size_t)snap_count * sizeof(trine_snap_t);

    trine_snap_t *arena = (trine_snap_t *)aligned_alloc(32, arena_bytes);
    if (!arena) {
        fprintf(stderr, "trine_embed: allocation failed (%u snaps)\n",
                snap_count);
        return -2;
    }
    memcpy(arena, model->snaps, arena_bytes);

    /* --- 3. Pre-load I/O channels from encoding ---
     *
     * The encoding produces 240 trit values across 4 chains:
     *   channels[  0.. 59] = forward encoding  -> I/O channels 0-59
     *   channels[ 60..119] = reverse encoding  -> I/O channels 60-119
     *   channels[120..179] = differential      -> I/O channels 120-179
     *   channels[180..239] = structural        -> I/O channels 180-239
     *
     * We load these into the I/O channel array used by the cascade.
     * Channels 240-255 are reserved for output and initialized to 0.
     */
    uint8_t io_channels[EMBED_IO_CHANNELS];
    memset(io_channels, 0, sizeof(io_channels));

    for (int i = 0; i < TRINE_CHANNELS && i < EMBED_IO_CHANNELS; i++) {
        io_channels[i] = encoding[i];
    }

    /* --- 3b. Pre-inject text signal into snap arena ---
     *
     * The cascade propagates activation (which snaps fire) but the text
     * signal must be injected into snap FSM states so that the embedding
     * dimensions carry text-dependent values.
     *
     * Strategy: seed each snap in the embedding extraction ranges with
     * a deterministic trit value derived from the encoded text and the
     * snap's position. This uses a mixing function that combines the
     * 240-channel encoding with the snap index.
     *
     * The mixing function is:
     *   trit(snap_i) = hash(encoding, i) mod 3
     *
     * where hash is a fast position-dependent mixer that ensures:
     *   1. Different texts produce different trit patterns
     *   2. The same text always produces the same pattern (deterministic)
     *   3. Nearby snaps get different values (good spatial mixing)
     */
    {
        /* Compute a 32-bit FNV-1a hash of the encoding */
        uint32_t text_hash = 0x811c9dc5u;
        for (int i = 0; i < TRINE_CHANNELS; i++) {
            text_hash = (text_hash ^ encoding[i]) * 0x01000193u;
        }

        /* Inject into all snap ranges used by resolutions.
         * Process all three tiers so the same snap gets the same
         * injection regardless of which resolution is requested. */
        for (int r = 0; r < TRINE_NUM_RESOLUTIONS; r++) {
            const trine_resolution_t *res = &model->resolutions[r];
            for (uint32_t ri = 0; ri < res->range_count && ri < 4; ri++) {
                uint32_t start = res->ranges[ri][0];
                uint32_t end   = res->ranges[ri][1];
                for (uint32_t si = start; si < end && si < snap_count; si++) {
                    /* Position-dependent mixing: combine text hash with
                     * snap index using multiplicative hashing */
                    uint32_t mix = text_hash ^ (si * 2654435761u);
                    mix = (mix ^ (mix >> 16)) * 0x45d9f3bu;
                    mix = mix ^ (mix >> 13);

                    /* Additional mixing with encoding channels:
                     * XOR in the encoding trit at a position derived
                     * from the snap index for spatial variation */
                    uint32_t ch_idx = si % TRINE_CHANNELS;
                    mix = mix ^ ((uint32_t)encoding[ch_idx] * 0x9e3779b9u);
                    mix = (mix ^ (mix >> 11)) * 0x27d4eb2du;

                    uint8_t trit_val = (uint8_t)(mix % 3);
                    trine_snap_set_fsm(&arena[si], trit_val,
                                       trine_snap_out(&arena[si]));
                }
            }
        }
    }

    /* --- 4. Initialize cascade engine ---
     *
     * Seed the wave from the model's root snap. The root snap is a
     * broadcast hub that fans out to all INGRESS chains on the first
     * tick, initiating the level-synchronous BFS wave.
     */
    trine_cascade_t cas;
    if (trine_cascade_init(&cas, arena, snap_count) != 0) {
        fprintf(stderr, "trine_embed: cascade init failed\n");
        free(arena);
        return -3;
    }

    /* If no STAT_LIVE snaps were found in the arena, seed from root */
    if (cas.active_count == 0) {
        uint32_t root = model->header.root_index;
        if (root < snap_count) {
            trine_snap_set_stat(&arena[root], TRINE_STAT_LIVE);
            cas.active[0] = root;
            cas.active_count = 1;
        }
    }

    if (verbose) {
        fprintf(stderr, "  Initial wave: %u snaps\n", cas.active_count);
    }

    /* --- 5. Run cascade --- */
    int max_ticks = RESOLUTION_TICKS[resolution];
    uint32_t total_emit = 0;

    trine_cascade_run_full(&cas, (uint32_t)max_ticks, io_channels, &total_emit);

    out->ticks = (int)cas.tick;
    out->final_wave = (int)cas.active_count;

    if (verbose) {
        fprintf(stderr, "  Cascade: %d ticks, final wave %u snaps, "
                "%u emissions\n",
                out->ticks, cas.active_count, total_emit);
    }

    /* --- 6. Extract embedding ---
     *
     * Read the final FSM state (sr >> 6, 2 bits -> value in {0,1,2})
     * of each snap in the resolution's index ranges. The snap state
     * IS the embedding trit at that position.
     *
     * Resolution maps define which snap indices contribute to each tier:
     *   Screening: tiers[0] ranges only
     *   Standard:  tiers[0] + tiers[1] ranges (cumulative)
     *   Deep:      all tiers
     *
     * The resolution_t struct defines ranges [start, end) for each tier.
     * We accumulate trits from tier 0 through the selected resolution.
     */
    {
        /* Calculate total dims for this resolution.
         * For cumulative resolution: screening contributes to standard,
         * which contributes to deep.
         * However, the .trine format defines each resolution independently
         * with its own set of ranges. We use the resolution's ranges directly.
         */
        const trine_resolution_t *res = &model->resolutions[resolution];
        int dims = (int)res->dim_count;

        out->trits = (uint8_t *)calloc((size_t)dims, 1);
        if (!out->trits) {
            fprintf(stderr, "trine_embed: allocation failed (%d dims)\n", dims);
            trine_cascade_free(&cas);
            free(arena);
            return -2;
        }
        out->dims = dims;

        /* Extract trits from snap states in each range */
        int trit_idx = 0;
        for (uint32_t r = 0; r < res->range_count && r < 4; r++) {
            uint32_t start = res->ranges[r][0];
            uint32_t end   = res->ranges[r][1];

            for (uint32_t si = start; si < end && trit_idx < dims; si++) {
                if (si < snap_count) {
                    out->trits[trit_idx] = trine_snap_st(&arena[si]);
                }
                trit_idx++;
            }
        }

        if (verbose) {
            fprintf(stderr, "  Embedding: %d dims (%s resolution)\n",
                    dims, RESOLUTION_NAME[resolution]);
        }
    }

    /* --- 7. Read consistency from I/O channel 98 --- */
    out->consistency = io_channels[98] % 3;

    if (verbose) {
        fprintf(stderr, "  Consistency: %u\n", out->consistency);
    }

    trine_cascade_free(&cas);
    free(arena);
    return 0;
}

/* =====================================================================
 * Internal: Per-chain helpers (shingle resolution)
 * ===================================================================== */

/* Count non-zero (active) trits in a chain slice */
static int count_chain_active(const uint8_t *trits, int chain_offset,
                              int chain_width) {
    int active = 0;
    for (int i = 0; i < chain_width; i++) {
        if (trits[chain_offset + i] != 0) active++;
    }
    return active;
}

/* =====================================================================
 * Internal: Output formatting
 * ===================================================================== */

/* Print trit string (e.g. "2102011202...") */
static void print_trits(const uint8_t *trits, int dims) {
    for (int i = 0; i < dims; i++) {
        putchar('0' + trits[i]);
    }
    putchar('\n');
}

/* Pack trits into bytes using base-243 encoding.
 * Every 5 trits encode into one byte: t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
 * Returns allocated buffer and sets *out_len. Caller must free. */
static uint8_t *trits_to_bytes(const uint8_t *trits, int dims, int *out_len) {
    int nbytes = (dims + 4) / 5;
    uint8_t *buf = (uint8_t *)calloc((size_t)nbytes, 1);
    if (!buf) return NULL;

    for (int i = 0; i < nbytes; i++) {
        uint8_t val = 0;
        int base = i * 5;
        if (base + 0 < dims) val = (uint8_t)(val + trits[base + 0]);
        if (base + 1 < dims) val = (uint8_t)(val + trits[base + 1] * 3);
        if (base + 2 < dims) val = (uint8_t)(val + trits[base + 2] * 9);
        if (base + 3 < dims) val = (uint8_t)(val + trits[base + 3] * 27);
        if (base + 4 < dims) val = (uint8_t)(val + trits[base + 4] * 81);
        buf[i] = val;
    }

    *out_len = nbytes;
    return buf;
}

/* Print hex encoding */
static void print_hex(const uint8_t *trits, int dims) {
    int nbytes = 0;
    uint8_t *bytes = trits_to_bytes(trits, dims, &nbytes);
    if (!bytes) {
        fprintf(stderr, "trine_embed: allocation failed in hex output\n");
        return;
    }

    for (int i = 0; i < nbytes; i++) {
        printf("%02x", bytes[i]);
    }
    putchar('\n');
    free(bytes);
}

/* Print base64 encoding */
static void print_base64(const uint8_t *trits, int dims) {
    int nbytes = 0;
    uint8_t *bytes = trits_to_bytes(trits, dims, &nbytes);
    if (!bytes) {
        fprintf(stderr, "trine_embed: allocation failed in base64 output\n");
        return;
    }

    /* Standard base64 encoding */
    int i;
    for (i = 0; i + 2 < nbytes; i += 3) {
        uint32_t triplet = ((uint32_t)bytes[i] << 16) |
                           ((uint32_t)bytes[i + 1] << 8) |
                           ((uint32_t)bytes[i + 2]);
        putchar(B64_TABLE[(triplet >> 18) & 0x3F]);
        putchar(B64_TABLE[(triplet >> 12) & 0x3F]);
        putchar(B64_TABLE[(triplet >>  6) & 0x3F]);
        putchar(B64_TABLE[(triplet      ) & 0x3F]);
    }

    /* Handle remaining 1-2 bytes */
    if (i < nbytes) {
        uint32_t triplet = (uint32_t)bytes[i] << 16;
        if (i + 1 < nbytes)
            triplet |= (uint32_t)bytes[i + 1] << 8;

        putchar(B64_TABLE[(triplet >> 18) & 0x3F]);
        putchar(B64_TABLE[(triplet >> 12) & 0x3F]);

        if (i + 1 < nbytes)
            putchar(B64_TABLE[(triplet >> 6) & 0x3F]);
        else
            putchar('=');

        putchar('=');
    }

    putchar('\n');
    free(bytes);
}

/* Escape a string for JSON output. Writes to buf (max buf_size chars).
 * Returns number of chars written (excluding null terminator). */
static int json_escape(const char *src, size_t src_len,
                       char *buf, size_t buf_size) {
    size_t w = 0;
    for (size_t i = 0; i < src_len && w + 6 < buf_size; i++) {
        unsigned char c = (unsigned char)src[i];
        switch (c) {
            case '"':  buf[w++] = '\\'; buf[w++] = '"';  break;
            case '\\': buf[w++] = '\\'; buf[w++] = '\\'; break;
            case '\b': buf[w++] = '\\'; buf[w++] = 'b';  break;
            case '\f': buf[w++] = '\\'; buf[w++] = 'f';  break;
            case '\n': buf[w++] = '\\'; buf[w++] = 'n';  break;
            case '\r': buf[w++] = '\\'; buf[w++] = 'r';  break;
            case '\t': buf[w++] = '\\'; buf[w++] = 't';  break;
            default:
                if (c < 0x20) {
                    w += (size_t)snprintf(buf + w, buf_size - w,
                                          "\\u%04x", c);
                } else {
                    buf[w++] = (char)c;
                }
                break;
        }
    }
    if (w < buf_size) buf[w] = '\0';
    return (int)w;
}

/* Print JSON output */
static void print_json(const embed_result_t *emb, const char *text,
                       size_t text_len) {
    /* Escape text for JSON */
    size_t esc_size = text_len * 6 + 1;
    char *escaped = (char *)malloc(esc_size);
    if (!escaped) {
        fprintf(stderr, "trine_embed: allocation failed in JSON output\n");
        return;
    }
    json_escape(text, text_len, escaped, esc_size);

    /* Build trit string */
    char *trit_str = (char *)malloc((size_t)emb->dims + 1);
    if (!trit_str) {
        free(escaped);
        fprintf(stderr, "trine_embed: allocation failed in JSON output\n");
        return;
    }
    for (int i = 0; i < emb->dims; i++) {
        trit_str[i] = (char)('0' + emb->trits[i]);
    }
    trit_str[emb->dims] = '\0';

    printf("{\"text\":\"%s\",\"resolution\":\"%s\",\"dims\":%d,"
           "\"trits\":\"%s\",\"consistency\":%u,\"ticks\":%d",
           escaped, RESOLUTION_NAME[emb->resolution], emb->dims,
           trit_str, emb->consistency, emb->ticks);

    /* Append per-chain fill rates for shingle resolution */
    if (emb->resolution == TRINE_RES_SHINGLE && emb->dims == CHAIN_COUNT * CHAIN_WIDTH) {
        printf(",\"chains\":{");
        for (int c = 0; c < CHAIN_COUNT; c++) {
            int active = count_chain_active(emb->trits, c * CHAIN_WIDTH, CHAIN_WIDTH);
            printf("\"%s\":{\"active\":%d,\"total\":%d}",
                   CHAIN_NAME[c], active, CHAIN_WIDTH);
            if (c < CHAIN_COUNT - 1) putchar(',');
        }
        putchar('}');
    }

    printf("}\n");

    free(trit_str);
    free(escaped);
}

/* Print embedding in the requested format */
static void print_embedding(const embed_result_t *emb, int fmt,
                             const char *text, size_t text_len) {
    switch (fmt) {
        case FMT_TRITS:   print_trits(emb->trits, emb->dims);        break;
        case FMT_HEX:     print_hex(emb->trits, emb->dims);          break;
        case FMT_BASE64:  print_base64(emb->trits, emb->dims);       break;
        case FMT_JSON:    print_json(emb, text, text_len);            break;
        default:          print_trits(emb->trits, emb->dims);        break;
    }
}

/* =====================================================================
 * Internal: Comparison
 * ===================================================================== */

static double cosine_similarity(const uint8_t *a, const uint8_t *b, int dims) {
    double dot = 0.0, mag_a = 0.0, mag_b = 0.0;

    for (int i = 0; i < dims; i++) {
        double va = (double)a[i];
        double vb = (double)b[i];
        dot   += va * vb;
        mag_a += va * va;
        mag_b += vb * vb;
    }

    if (mag_a < 1e-12 || mag_b < 1e-12)
        return 0.0;

    return dot / (sqrt(mag_a) * sqrt(mag_b));
}

static int hamming_distance(const uint8_t *a, const uint8_t *b, int dims) {
    int dist = 0;
    for (int i = 0; i < dims; i++) {
        if (a[i] != b[i]) dist++;
    }
    return dist;
}

/* =====================================================================
 * Internal: File reading
 * ===================================================================== */

/* Read entire file contents into a heap-allocated buffer.
 * Returns NULL on error. Sets *out_len to the number of bytes read. */
static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "trine_embed: cannot open '%s': %s\n",
                path, strerror(errno));
        return NULL;
    }

    /* Determine file size */
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "trine_embed: seek error on '%s'\n", path);
        fclose(f);
        return NULL;
    }

    long sz = ftell(f);
    if (sz < 0 || (unsigned long)sz > EMBED_MAX_TEXT) {
        fprintf(stderr, "trine_embed: file too large: '%s' (%ld bytes, max %d)\n",
                path, sz, EMBED_MAX_TEXT);
        fclose(f);
        return NULL;
    }
    rewind(f);

    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fprintf(stderr, "trine_embed: allocation failed for file '%s'\n", path);
        fclose(f);
        return NULL;
    }

    size_t nread = fread(buf, 1, (size_t)sz, f);
    fclose(f);

    buf[nread] = '\0';
    *out_len = nread;
    return buf;
}

/* =====================================================================
 * Internal: Human-readable size formatting
 * ===================================================================== */

static void format_size(uint64_t bytes, char *buf, size_t buf_size) {
    if (bytes < 1024) {
        snprintf(buf, buf_size, "%lu bytes", (unsigned long)bytes);
    } else if (bytes < 1024 * 1024) {
        snprintf(buf, buf_size, "%lu bytes (%.1f KB)",
                 (unsigned long)bytes, (double)bytes / 1024.0);
    } else {
        snprintf(buf, buf_size, "%lu bytes (%.1f MB)",
                 (unsigned long)bytes, (double)bytes / (1024.0 * 1024.0));
    }
}

/* =====================================================================
 * Internal: Format a number with thousands separators
 * ===================================================================== */

static void format_number(uint64_t n, char *buf, size_t buf_size) {
    char raw[32];
    int len = snprintf(raw, sizeof(raw), "%lu", (unsigned long)n);

    /* Insert commas */
    int commas = (len - 1) / 3;
    int total = len + commas;
    if ((size_t)total >= buf_size) {
        snprintf(buf, buf_size, "%lu", (unsigned long)n);
        return;
    }

    buf[total] = '\0';
    int dst = total - 1;
    int digits = 0;
    for (int src = len - 1; src >= 0; src--) {
        buf[dst--] = raw[src];
        digits++;
        if (digits % 3 == 0 && src > 0) {
            buf[dst--] = ',';
        }
    }
}

/* =====================================================================
 * Commands
 * ===================================================================== */

/* --info: Display model metadata */
static int cmd_info(const char *model_path) {
    trine_file_t model;
    if (load_model(model_path, &model) != 0) return 1;

    /* Compute file size */
    uint64_t file_size = (uint64_t)TRINE_OFF_SNAPS +
                         (uint64_t)model.snap_count * 32 +
                         (uint64_t)(sizeof(uint64_t) * TRINE_NUM_SECTIONS);

    char size_buf[64];
    format_size(file_size, size_buf, sizeof(size_buf));

    printf("TRINE Model Information\n");
    printf("  File:       %s\n", model_path);
    printf("  Version:    %u\n", model.header.version);
    printf("  Snaps:      %u\n", model.snap_count);
    printf("  Layers:     %u\n", model.header.layer_count);
    printf("  Resolutions:\n");

    for (int r = 0; r < TRINE_NUM_RESOLUTIONS; r++) {
        const trine_resolution_t *res = &model.resolutions[r];
        /* Estimate storage: 5 trits per byte (base-243) */
        int byte_est = ((int)res->dim_count + 4) / 5;

        if (byte_est < 1024) {
            printf("    %-11s %u dims (~%d bytes)\n",
                   TRINE_RES_NAME[r], res->dim_count, byte_est);
        } else {
            printf("    %-11s %u dims (~%.1f KB)\n",
                   TRINE_RES_NAME[r], res->dim_count,
                   (double)byte_est / 1024.0);
        }
    }

    printf("  File size:  %s\n", size_buf);

    /* Layer breakdown */
    printf("  Layers:\n");
    for (int i = 0; i < TRINE_NUM_LAYERS; i++) {
        const trine_layer_info_t *lay = &model.layers[i];
        printf("    %2d %-8s %5u snaps [%u..%u)\n",
               i, TRINE_LAYER_NAME[i],
               lay->snap_count,
               lay->start_index,
               lay->start_index + lay->snap_count);
    }

    /* I/O channel summary */
    int input_count = 0, output_count = 0;
    for (int i = 0; i < TRINE_NUM_IO_CHANNELS; i++) {
        if (model.io_channels[i].direction == 0)
            input_count++;
        else
            output_count++;
    }
    printf("  I/O channels: %d input, %d output (%d total)\n",
           input_count, output_count, TRINE_NUM_IO_CHANNELS);

    trine_file_free(&model);
    return 0;
}

/* --compare: Compare two text embeddings */
static int cmd_compare(const char *model_path,
                       const char *text1, const char *text2,
                       int resolution, int fmt, int verbose,
                       float lens_weights[4], int has_lens, int do_detail,
                       int use_idf) {
    trine_file_t model;
    if (load_model(model_path, &model) != 0) return 1;

    if (verbose) {
        fprintf(stderr, "Computing embedding for text 1: \"%s\"\n", text1);
    }

    embed_result_t emb1, emb2;
    int rc = compute_embedding(&model, text1, strlen(text1),
                               resolution, verbose, &emb1);
    if (rc != 0) {
        trine_file_free(&model);
        return 1;
    }

    if (verbose) {
        fprintf(stderr, "Computing embedding for text 2: \"%s\"\n", text2);
    }

    rc = compute_embedding(&model, text2, strlen(text2),
                           resolution, verbose, &emb2);
    if (rc != 0) {
        free(emb1.trits);
        trine_file_free(&model);
        return 1;
    }

    if (emb1.dims != emb2.dims) {
        fprintf(stderr, "trine_embed: dimension mismatch: %d vs %d\n",
                emb1.dims, emb2.dims);
        free(emb1.trits);
        free(emb2.trits);
        trine_file_free(&model);
        return 1;
    }

    double cosine = cosine_similarity(emb1.trits, emb2.trits, emb1.dims);
    int hamming = hamming_distance(emb1.trits, emb2.trits, emb1.dims);
    int matching = emb1.dims - hamming;

    if (fmt == FMT_JSON) {
        /* JSON compare output */
        size_t esc1_size = strlen(text1) * 6 + 1;
        size_t esc2_size = strlen(text2) * 6 + 1;
        char *esc1 = (char *)malloc(esc1_size);
        char *esc2 = (char *)malloc(esc2_size);
        if (!esc1 || !esc2) {
            fprintf(stderr, "trine_embed: allocation failed in JSON compare\n");
            free(esc1); free(esc2);
            free(emb1.trits); free(emb2.trits);
            trine_file_free(&model);
            return 1;
        }
        json_escape(text1, strlen(text1), esc1, esc1_size);
        json_escape(text2, strlen(text2), esc2, esc2_size);

        printf("{\"text1\":\"%s\",\"text2\":\"%s\","
               "\"resolution\":\"%s\","
               "\"cosine\":%.3f,\"hamming\":%d,\"matching\":%d,\"dims\":%d",
               esc1, esc2,
               RESOLUTION_NAME[emb1.resolution],
               cosine, hamming, matching, emb1.dims);

        /* Per-chain cosine for shingle resolution */
        if (emb1.resolution == TRINE_RES_SHINGLE &&
            emb1.dims == CHAIN_COUNT * CHAIN_WIDTH) {
            printf(",\"chains\":{");
            for (int c = 0; c < CHAIN_COUNT; c++) {
                double cc = chain_cosine_local(emb1.trits, emb2.trits,
                                               c * CHAIN_WIDTH, CHAIN_WIDTH);
                printf("\"%s\":{\"cosine\":%.3f}", CHAIN_NAME[c], cc);
                if (c < CHAIN_COUNT - 1) putchar(',');
            }
            putchar('}');
        }

        printf("}\n");
        free(esc1);
        free(esc2);
    } else {
        /* Print IDF cosine before regular cosine when --idf is active */
        if (use_idf && emb1.resolution == TRINE_RES_SHINGLE &&
            emb1.dims == CHAIN_COUNT * CHAIN_WIDTH) {
            double idf_cos = (double)trine_idf_cosine(emb1.trits, emb2.trits,
                                                       TRINE_IDF_WEIGHTS);
            printf("IDF cosine:        %.3f\n", idf_cos);
        }

        printf("Cosine similarity: %.3f\n", cosine);
        printf("Hamming distance:  %d / %d (%.1f%%)\n",
               hamming, emb1.dims,
               100.0 * (double)hamming / (double)emb1.dims);
        printf("Matching trits:    %d / %d (%.1f%%)\n",
               matching, emb1.dims,
               100.0 * (double)matching / (double)emb1.dims);

        /* Per-chain breakdown (when --detail or --lens is used) */
        if (do_detail || has_lens) {
            double chain_cos[4];
            double weighted_sum = 0.0, weight_sum = 0.0;
            for (int c = 0; c < CHAIN_COUNT; c++) {
                if (use_idf && emb1.resolution == TRINE_RES_SHINGLE &&
                    emb1.dims == CHAIN_COUNT * CHAIN_WIDTH) {
                    chain_cos[c] = chain_cosine_idf(emb1.trits, emb2.trits,
                                                     c * CHAIN_WIDTH, CHAIN_WIDTH,
                                                     TRINE_IDF_WEIGHTS);
                } else {
                    chain_cos[c] = chain_cosine_local(emb1.trits, emb2.trits,
                                                       c * CHAIN_WIDTH, CHAIN_WIDTH);
                }
                if (lens_weights[c] > 0) {
                    weighted_sum += lens_weights[c] * chain_cos[c];
                    weight_sum += lens_weights[c];
                }
            }
            double lens_score = (weight_sum > 0) ? weighted_sum / weight_sum : 0.0;

            if (has_lens) {
                printf("Lens similarity:   %.3f  [weights: %.1f, %.1f, %.1f, %.1f]\n",
                       lens_score, lens_weights[0], lens_weights[1],
                       lens_weights[2], lens_weights[3]);
            }
            printf("Per-chain cosine:  edit=%.3f  morph=%.3f  phrase=%.3f  vocab=%.3f\n",
                   chain_cos[0], chain_cos[1], chain_cos[2], chain_cos[3]);
        }
    }

    free(emb1.trits);
    free(emb2.trits);
    trine_file_free(&model);
    return 0;
}

/* --batch: Embed each line of a file */
static int cmd_batch(const char *model_path, const char *file_path,
                     int resolution, int fmt, int verbose, int quiet) {
    trine_file_t model;
    if (load_model(model_path, &model) != 0) return 1;

    FILE *f = fopen(file_path, "r");
    if (!f) {
        fprintf(stderr, "trine_embed: cannot open '%s': %s\n",
                file_path, strerror(errno));
        trine_file_free(&model);
        return 1;
    }

    char line[EMBED_MAX_LINE];
    int line_num = 0;
    int errors = 0;

    while (fgets(line, sizeof(line), f)) {
        line_num++;

        /* Strip trailing newline/carriage return */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';

        /* Skip empty lines */
        if (len == 0) continue;

        embed_result_t emb;
        int rc = compute_embedding(&model, line, len,
                                   resolution, verbose, &emb);
        if (rc != 0) {
            fprintf(stderr, "trine_embed: error on line %d\n", line_num);
            errors++;
            continue;
        }

        print_embedding(&emb, fmt, line, len);
        free(emb.trits);
    }

    fclose(f);

    if (!quiet && errors > 0) {
        fprintf(stderr, "trine_embed: %d error(s) in %d lines\n",
                errors, line_num);
    }

    trine_file_free(&model);
    return errors > 0 ? 1 : 0;
}

/* --benchmark: Time N embeddings and report throughput */
static int cmd_benchmark(const char *model_path, int n, int resolution,
                         int verbose) {
    (void)verbose;  /* Reserved for future per-iteration diagnostics */
    if (n <= 0) {
        fprintf(stderr, "trine_embed: benchmark count must be positive\n");
        return 1;
    }

    trine_file_t model;
    if (load_model(model_path, &model) != 0) return 1;

    const char *bench_text = "hello world";
    size_t bench_len = strlen(bench_text);

    /* Warmup run */
    {
        embed_result_t emb;
        int rc = compute_embedding(&model, bench_text, bench_len,
                                   resolution, 0, &emb);
        if (rc != 0) {
            fprintf(stderr, "trine_embed: warmup embedding failed\n");
            trine_file_free(&model);
            return 1;
        }
        free(emb.trits);
    }

    /* Timed runs */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < n; i++) {
        embed_result_t emb;
        int rc = compute_embedding(&model, bench_text, bench_len,
                                   resolution, 0, &emb);
        if (rc != 0) {
            fprintf(stderr, "trine_embed: benchmark embedding %d failed\n", i);
            trine_file_free(&model);
            return 1;
        }
        free(emb.trits);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (double)(end.tv_sec - start.tv_sec) +
                     (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    double per_emb = elapsed / (double)n;
    double throughput = (double)n / elapsed;

    char throughput_buf[32];
    format_number((uint64_t)throughput, throughput_buf, sizeof(throughput_buf));

    printf("TRINE Benchmark: %d embeddings at %s resolution\n",
           n, RESOLUTION_NAME[resolution]);
    printf("  Total time:    %.3fs\n", elapsed);

    if (per_emb >= 1.0) {
        printf("  Per embedding: %.3fs\n", per_emb);
    } else if (per_emb >= 0.001) {
        printf("  Per embedding: %.3fms\n", per_emb * 1000.0);
    } else {
        printf("  Per embedding: %.3fus\n", per_emb * 1e6);
    }

    printf("  Throughput:    %s embeddings/sec\n", throughput_buf);

    trine_file_free(&model);
    return 0;
}

/* --benchmark-compare: Time N cosine comparisons and report throughput */
static int cmd_benchmark_compare(const char *model_path, int n) {
    if (n <= 0) {
        fprintf(stderr, "trine_embed: benchmark-compare count must be positive\n");
        return 1;
    }

    trine_file_t model;
    if (load_model(model_path, &model) != 0) return 1;

    /* Encode two fixed texts at shingle resolution */
    const char *text1 = "The quick brown fox jumps over the lazy dog";
    const char *text2 = "The quick brown fox leaps over the lazy dog";

    embed_result_t emb1, emb2;
    int rc = compute_embedding(&model, text1, strlen(text1),
                               TRINE_RES_SHINGLE, 0, &emb1);
    if (rc != 0) {
        trine_file_free(&model);
        return 1;
    }

    rc = compute_embedding(&model, text2, strlen(text2),
                           TRINE_RES_SHINGLE, 0, &emb2);
    if (rc != 0) {
        free(emb1.trits);
        trine_file_free(&model);
        return 1;
    }

    /* Timed comparison runs */
    struct timespec start, end;
    volatile double result = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < n; i++) {
        result = cosine_similarity(emb1.trits, emb2.trits, emb1.dims);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (double)(end.tv_sec - start.tv_sec) +
                     (double)(end.tv_nsec - start.tv_nsec) / 1e9;
    double per_cmp = elapsed / (double)n;
    double throughput = (double)n / elapsed;

    char throughput_buf[32];
    format_number((uint64_t)throughput, throughput_buf, sizeof(throughput_buf));

    printf("TRINE Comparison Benchmark: %d comparisons (240 dims)\n", n);
    printf("  Cosine result: %.3f\n", result);
    printf("  Total time:    %.3fs\n", elapsed);

    if (per_cmp >= 1.0) {
        printf("  Per comparison: %.3fs\n", per_cmp);
    } else if (per_cmp >= 0.001) {
        printf("  Per comparison: %.3fms\n", per_cmp * 1000.0);
    } else {
        printf("  Per comparison: %.3fus\n", per_cmp * 1e6);
    }

    printf("  Throughput:    %s comparisons/sec\n", throughput_buf);

    free(emb1.trits);
    free(emb2.trits);
    trine_file_free(&model);
    return 0;
}

/* Single text embedding (default mode) */
static int cmd_embed(const char *model_path,
                     const char *text, size_t text_len,
                     int resolution, int fmt,
                     int verbose, int quiet) {
    trine_file_t model;
    if (load_model(model_path, &model) != 0) return 1;

    if (verbose) {
        fprintf(stderr, "Model: %s (%u snaps, %u layers)\n",
                model_path, model.snap_count, model.header.layer_count);
        fprintf(stderr, "Text: \"%.*s\"%s\n",
                (int)(text_len > 60 ? 60 : text_len), text,
                text_len > 60 ? "..." : "");
        fprintf(stderr, "Resolution: %s (%d ticks)\n",
                RESOLUTION_NAME[resolution], RESOLUTION_TICKS[resolution]);
    }

    embed_result_t emb;
    int rc = compute_embedding(&model, text, text_len,
                               resolution, verbose, &emb);
    if (rc != 0) {
        trine_file_free(&model);
        return 1;
    }

    if (!quiet) {
        /* Print a header unless we're in quiet mode */
        if (fmt != FMT_JSON) {
            /* No header needed for JSON; for other formats, optionally
             * print metadata on stderr if verbose */
        }
    }

    print_embedding(&emb, fmt, text, text_len);

    free(emb.trits);
    trine_file_free(&model);
    return 0;
}

/* =====================================================================
 * Usage
 * ===================================================================== */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "TRINE Embed v%s — Ternary Resonance Interference Network Embedding\n"
        "\n"
        "Usage:\n"
        "  %s <model.trine> [options] <text>\n"
        "  %s <model.trine> [options] -f <file>\n"
        "  %s <model.trine> --compare <text1> <text2>\n"
        "  %s <model.trine> --batch -f <file>\n"
        "  %s <model.trine> --info\n"
        "  %s <model.trine> --benchmark N\n"
        "\n"
        "Options:\n"
        "  -r, --resolution LEVEL   screening|standard|deep|shingle (default: standard)\n"
        "  -o, --output FORMAT      trits|hex|base64|json (default: trits)\n"
        "  -v, --verbose            Show cascade details\n"
        "  -q, --quiet              Only output embedding\n"
        "  -l, --lens SPEC          Lens weighting: edit|morph|phrase|vocab|dedup|code|legal|medical|support|policy|uniform|w,w,w,w\n"
        "  --detail                 Show per-chain cosine breakdown in compare mode\n"
        "  --idf                    Use IDF-weighted cosine in compare mode (shingle only)\n"
        "  --compare TEXT1 TEXT2     Compare two texts (cosine + hamming)\n"
        "  --batch -f FILE          Embed each line of FILE\n"
        "  --info                   Show model info\n"
        "  --benchmark N            Run N embeddings and report timing\n"
        "  --benchmark-compare N    Run N cosine comparisons and report timing\n"
        "\n"
        "Output Formats:\n"
        "  trits    Raw trit string (e.g. \"2102011202...\")\n"
        "  hex      Base-243 packed hex (5 trits per byte)\n"
        "  base64   Base64 of the hex output\n"
        "  json     JSON object with metadata\n"
        "\n"
        "Resolution Tiers:\n"
        "  screening   68 dims (~11 bytes)    — fast coarse filter (%d ticks)\n"
        "  standard    1,053 dims (~166 bytes) — balanced quality/speed (%d ticks)\n"
        "  deep        17,410 dims (~2.7 KB)   — maximum fidelity (%d ticks)\n"
        "  shingle     240 dims (~38 bytes)    — similarity search (0 ticks, no cascade)\n"
        "\n"
        "Examples:\n"
        "  %s model.trine \"hello world\"\n"
        "  %s model.trine -r deep -o json \"quantum computing\"\n"
        "  %s model.trine --compare \"cat\" \"kitten\"\n"
        "  %s model.trine --batch -f sentences.txt\n"
        "  %s model.trine --benchmark 1000\n",
        EMBED_VERSION,
        prog, prog, prog, prog, prog, prog,
        EMBED_TICKS_SCREENING, EMBED_TICKS_STANDARD, EMBED_TICKS_DEEP,
        prog, prog, prog, prog, prog);
}

/* =====================================================================
 * Argument Parsing Helpers
 * ===================================================================== */

static int parse_resolution(const char *str) {
    if (strcmp(str, "screening") == 0) return TRINE_RES_SCREENING;
    if (strcmp(str, "standard") == 0)  return TRINE_RES_STANDARD;
    if (strcmp(str, "deep") == 0)      return TRINE_RES_DEEP;
    if (strcmp(str, "shingle") == 0)   return TRINE_RES_SHINGLE;
    /* Allow single-letter abbreviations */
    if (strcmp(str, "s") == 0)         return TRINE_RES_SCREENING;
    if (strcmp(str, "d") == 0)         return TRINE_RES_DEEP;
    if (strcmp(str, "sh") == 0)        return TRINE_RES_SHINGLE;
    return -1;
}

static int parse_format(const char *str) {
    if (strcmp(str, "trits") == 0)     return FMT_TRITS;
    if (strcmp(str, "hex") == 0)       return FMT_HEX;
    if (strcmp(str, "base64") == 0)    return FMT_BASE64;
    if (strcmp(str, "json") == 0)      return FMT_JSON;
    return -1;
}

/* =====================================================================
 * main
 * ===================================================================== */

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    /* First non-option argument is the model path */
    const char *model_path = argv[1];

    /* Check for simple --help/-h before requiring model */
    if (strcmp(model_path, "--help") == 0 || strcmp(model_path, "-h") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    if (strcmp(model_path, "--version") == 0) {
        printf("trine_embed %s\n", EMBED_VERSION);
        return 0;
    }

    /* Parse options */
    int resolution      = TRINE_RES_STANDARD;
    int fmt             = FMT_TRITS;
    int verbose         = 0;
    int quiet           = 0;
    int do_info         = 0;
    int do_compare      = 0;
    int do_batch        = 0;
    int do_benchmark    = 0;
    int benchmark_n     = 0;
    int do_detail       = 0;
    int use_idf         = 0;
    int do_bench_compare = 0;
    int bench_compare_n = 0;
    float lens_weights[4] = {1.0f, 1.0f, 1.0f, 1.0f};  /* default: uniform */
    int has_lens        = 0;
    const char *file_path   = NULL;
    (void)0; /* compare texts taken from positional args */

    /* Collect positional arguments (text) */
    const char *positional[64];
    int n_positional = 0;

    int i = 2;
    while (i < argc) {
        const char *arg = argv[i];

        if (strcmp(arg, "--info") == 0) {
            do_info = 1;
            i++;
        } else if (strcmp(arg, "--compare") == 0) {
            do_compare = 1;
            i++;
        } else if (strcmp(arg, "--batch") == 0) {
            do_batch = 1;
            i++;
        } else if (strcmp(arg, "--benchmark") == 0) {
            do_benchmark = 1;
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_embed: --benchmark requires a number\n");
                return 1;
            }
            benchmark_n = atoi(argv[i + 1]);
            i += 2;
        } else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--resolution") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_embed: %s requires an argument\n", arg);
                return 1;
            }
            resolution = parse_resolution(argv[i + 1]);
            if (resolution < 0) {
                fprintf(stderr, "trine_embed: invalid resolution '%s' "
                        "(use screening, standard, deep, or shingle)\n", argv[i + 1]);
                return 1;
            }
            i += 2;
        } else if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_embed: %s requires an argument\n", arg);
                return 1;
            }
            fmt = parse_format(argv[i + 1]);
            if (fmt < 0) {
                fprintf(stderr, "trine_embed: invalid format '%s' "
                        "(use trits, hex, base64, or json)\n", argv[i + 1]);
                return 1;
            }
            i += 2;
        } else if (strcmp(arg, "-f") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_embed: -f requires a file path\n");
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
        } else if (strcmp(arg, "-l") == 0 || strcmp(arg, "--lens") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_embed: %s requires a lens spec\n", arg);
                return 1;
            }
            if (parse_lens(argv[i + 1], lens_weights) != 0) {
                fprintf(stderr, "trine_embed: invalid lens '%s' "
                        "(use edit|morph|phrase|vocab|dedup|uniform or w,w,w,w)\n",
                        argv[i + 1]);
                return 1;
            }
            has_lens = 1;
            i += 2;
        } else if (strcmp(arg, "--detail") == 0) {
            do_detail = 1;
            i++;
        } else if (strcmp(arg, "--idf") == 0) {
            use_idf = 1;
            i++;
        } else if (strcmp(arg, "--benchmark-compare") == 0) {
            do_bench_compare = 1;
            if (i + 1 >= argc) {
                fprintf(stderr, "trine_embed: --benchmark-compare requires a number\n");
                return 1;
            }
            bench_compare_n = atoi(argv[i + 1]);
            i += 2;
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] == '-' && arg[1] != '\0') {
            fprintf(stderr, "trine_embed: unknown option '%s'\n", arg);
            fprintf(stderr, "Try '%s --help' for usage.\n", argv[0]);
            return 1;
        } else {
            /* Positional argument (text) */
            if (n_positional < 64)
                positional[n_positional++] = arg;
            i++;
        }
    }

    /* Dispatch to the appropriate command */

    /* --info */
    if (do_info) {
        return cmd_info(model_path);
    }

    /* --compare: takes two positional arguments as texts */
    if (do_compare) {
        if (n_positional < 2) {
            fprintf(stderr, "trine_embed: --compare requires two text arguments\n");
            return 1;
        }
        return cmd_compare(model_path, positional[0], positional[1],
                           resolution, fmt, verbose,
                           lens_weights, has_lens, do_detail, use_idf);
    }

    /* --benchmark */
    if (do_benchmark) {
        return cmd_benchmark(model_path, benchmark_n, resolution, verbose);
    }

    /* --benchmark-compare */
    if (do_bench_compare) {
        return cmd_benchmark_compare(model_path, bench_compare_n);
    }

    /* --batch */
    if (do_batch) {
        if (!file_path) {
            fprintf(stderr, "trine_embed: --batch requires -f <file>\n");
            return 1;
        }
        return cmd_batch(model_path, file_path, resolution, fmt,
                         verbose, quiet);
    }

    /* Single embedding from file (-f without --batch) */
    if (file_path && !do_batch) {
        size_t text_len = 0;
        char *text = read_file(file_path, &text_len);
        if (!text) return 1;

        /* Strip trailing whitespace */
        while (text_len > 0 &&
               (text[text_len - 1] == '\n' || text[text_len - 1] == '\r' ||
                text[text_len - 1] == ' '  || text[text_len - 1] == '\t'))
            text[--text_len] = '\0';

        int rc = cmd_embed(model_path, text, text_len, resolution, fmt,
                           verbose, quiet);
        free(text);
        return rc;
    }

    /* Single embedding from positional text argument */
    if (n_positional > 0) {
        /* If multiple positional args, join them with spaces */
        if (n_positional == 1) {
            return cmd_embed(model_path, positional[0], strlen(positional[0]),
                             resolution, fmt, verbose, quiet);
        }

        /* Join multiple positional arguments with spaces */
        size_t total_len = 0;
        for (int j = 0; j < n_positional; j++) {
            total_len += strlen(positional[j]);
            if (j < n_positional - 1) total_len++;  /* space separator */
        }

        char *joined = (char *)malloc(total_len + 1);
        if (!joined) {
            fprintf(stderr, "trine_embed: allocation failed\n");
            return 1;
        }

        size_t offset = 0;
        for (int j = 0; j < n_positional; j++) {
            size_t len = strlen(positional[j]);
            memcpy(joined + offset, positional[j], len);
            offset += len;
            if (j < n_positional - 1) {
                joined[offset++] = ' ';
            }
        }
        joined[offset] = '\0';

        int rc = cmd_embed(model_path, joined, offset, resolution, fmt,
                           verbose, quiet);
        free(joined);
        return rc;
    }

    /* No text provided */
    fprintf(stderr, "trine_embed: no text provided\n");
    fprintf(stderr, "Try '%s --help' for usage.\n", argv[0]);
    return 1;
}
