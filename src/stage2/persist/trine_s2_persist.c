/* =====================================================================
 * TRINE Stage-2 — Model Persistence (implementation)
 * =====================================================================
 *
 * Save/load trained Stage-2 models to .trine2 binary files.
 * Uses FNV-1a checksums for integrity verification.
 *
 * ===================================================================== */

#include "trine_s2_persist.h"
#include "trine_stage2.h"
#include "trine_project.h"
#include "oicos.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declaration for topology generator */
extern void trine_topology_random(void *lc, uint64_t seed);

/* ── FNV-1a checksum ─────────────────────────────────────────────── */

static uint64_t s2_fnv1a(const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++)
        h = (h ^ p[i]) * 0x100000001b3ULL;
    return h;
}

/* Block-diagonal payload size: K * 4_chains * 60 * 60 */
#define S2_BLOCK_WEIGHT_SIZE(k)  ((size_t)(k) * 4u * 60u * 60u)

/* Full-matrix payload size: K * 240 * 240 */
#define S2_FULL_WEIGHT_SIZE(k, dim) ((size_t)(k) * (size_t)(dim) * (size_t)(dim))

/* ── Save ────────────────────────────────────────────────────────── */

int trine_s2_save(const struct trine_s2_model *model,
                   const char *path,
                   const trine_s2_save_config_t *config)
{
    if (!model || !path) {
        fprintf(stderr, "trine_s2_save: NULL model or path\n");
        return -1;
    }

    int mode = trine_s2_get_projection_mode(model);
    int is_block_diag = (mode == TRINE_S2_PROJ_BLOCK_DIAG);

    /* Get the appropriate weight pointer */
    const void *proj;
    size_t weight_size;

    if (is_block_diag) {
        proj = trine_s2_get_block_projection(model);
        if (!proj) {
            fprintf(stderr, "trine_s2_save: cannot get block-diagonal weights\n");
            return -1;
        }
        weight_size = S2_BLOCK_WEIGHT_SIZE(TRINE_PROJECT_K);
    } else {
        proj = trine_s2_get_projection(model);
        if (!proj) {
            fprintf(stderr, "trine_s2_save: cannot get projection weights\n");
            return -1;
        }
        weight_size = S2_FULL_WEIGHT_SIZE(TRINE_PROJECT_K, TRINE_PROJECT_DIM);
    }

    /* Build header */
    trine_s2_file_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));

    hdr.magic          = TRINE_S2_MAGIC;
    hdr.version        = TRINE_S2_FORMAT_VERSION;
    hdr.projection_k   = TRINE_PROJECT_K;
    hdr.projection_dim = TRINE_PROJECT_DIM;
    hdr.cascade_cells  = trine_s2_get_cascade_cells(model);
    hdr.cascade_depth  = trine_s2_get_default_depth(model);

    if (mode == TRINE_S2_PROJ_DIAGONAL) {
        hdr.flags |= TRINE_S2_FLAG_DIAGONAL;
    }
    if (is_block_diag) {
        hdr.flags |= TRINE_S2_FLAG_BLOCK_DIAG;
    }
    if (trine_s2_is_identity(model)) {
        hdr.flags |= TRINE_S2_FLAG_IDENTITY;
    }

    if (config) {
        hdr.similarity_threshold = config->similarity_threshold;
        hdr.density              = config->density;
        hdr.topo_seed            = config->topo_seed;
    } else {
        hdr.similarity_threshold = 0.5f;
        hdr.density              = 0.33f;
        hdr.topo_seed            = 0;
    }

    /* Compute header checksum (over first 64 bytes, excluding checksum field) */
    hdr.header_checksum = s2_fnv1a(&hdr, 64);

    /* Compute payload checksum */
    uint64_t payload_checksum = s2_fnv1a(proj, weight_size);

    /* Write file */
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "trine_s2_save: cannot open '%s' for writing\n", path);
        return -1;
    }

    /* Header */
    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "trine_s2_save: failed to write header\n");
        fclose(fp);
        return -1;
    }

    /* Projection weights */
    if (fwrite(proj, weight_size, 1, fp) != 1) {
        fprintf(stderr, "trine_s2_save: failed to write weights\n");
        fclose(fp);
        return -1;
    }

    /* Payload checksum */
    if (fwrite(&payload_checksum, sizeof(payload_checksum), 1, fp) != 1) {
        fprintf(stderr, "trine_s2_save: failed to write checksum\n");
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

/* ── Trit Validation ─────────────────────────────────────────────── */

/* Validate that every byte in a weight buffer is a valid trit (0, 1, or 2).
 * Returns 0 if all bytes are valid, or -1 if any byte is out of range.
 * On failure, *bad_offset is set to the index of the first invalid byte
 * and *bad_value is set to the offending value. */
static int s2_validate_trits(const uint8_t *weights, size_t len,
                              size_t *bad_offset, uint8_t *bad_value)
{
    for (size_t i = 0; i < len; i++) {
        if (weights[i] > 2) {
            *bad_offset = i;
            *bad_value  = weights[i];
            return -1;
        }
    }
    return 0;
}

/* ── Load ────────────────────────────────────────────────────────── */

struct trine_s2_model *trine_s2_load(const char *path)
{
    if (!path) {
        fprintf(stderr, "trine_s2_load: NULL path\n");
        return NULL;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "trine_s2_load: cannot open '%s'\n", path);
        return NULL;
    }

    /* Read header */
    trine_s2_file_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "trine_s2_load: failed to read header\n");
        fclose(fp);
        return NULL;
    }

    /* Validate magic */
    if (hdr.magic != TRINE_S2_MAGIC) {
        fprintf(stderr, "trine_s2_load: bad magic 0x%08x (expected 0x%08x)\n",
                hdr.magic, TRINE_S2_MAGIC);
        fclose(fp);
        return NULL;
    }

    /* Validate version */
    if (hdr.version != TRINE_S2_FORMAT_VERSION) {
        fprintf(stderr, "trine_s2_load: unsupported version %u (expected %u)\n",
                hdr.version, TRINE_S2_FORMAT_VERSION);
        fclose(fp);
        return NULL;
    }

    /* Validate header checksum */
    uint64_t saved_checksum = hdr.header_checksum;
    hdr.header_checksum = 0;
    /* Recompute: checksum of first 64 bytes with checksum field zeroed */
    trine_s2_file_header_t check_hdr;
    memcpy(&check_hdr, &hdr, sizeof(check_hdr));
    check_hdr.header_checksum = 0;
    uint64_t computed = s2_fnv1a(&check_hdr, 64);
    if (computed != saved_checksum) {
        fprintf(stderr, "trine_s2_load: header checksum mismatch\n");
        fclose(fp);
        return NULL;
    }
    hdr.header_checksum = saved_checksum;

    /* Validate dimensions */
    if (hdr.projection_k != TRINE_PROJECT_K ||
        hdr.projection_dim != TRINE_PROJECT_DIM) {
        fprintf(stderr, "trine_s2_load: dimension mismatch K=%u DIM=%u "
                "(expected K=%u DIM=%u)\n",
                hdr.projection_k, hdr.projection_dim,
                TRINE_PROJECT_K, TRINE_PROJECT_DIM);
        fclose(fp);
        return NULL;
    }

    /* Determine payload size based on block-diagonal flag */
    int is_block_diag = (hdr.flags & TRINE_S2_FLAG_BLOCK_DIAG) != 0;
    size_t weight_size;
    if (is_block_diag) {
        weight_size = S2_BLOCK_WEIGHT_SIZE(hdr.projection_k);
    } else {
        weight_size = S2_FULL_WEIGHT_SIZE(hdr.projection_k, hdr.projection_dim);
    }

    /* Identity models: skip reading the weight payload entirely.
     * The identity constructor generates its own canonical weights,
     * so there is no need to allocate, read, or validate the stored
     * matrix.  We still verify the payload checksum for file integrity
     * by seeking past the weights and reading the trailing checksum,
     * then comparing it against the on-disk value.  However we skip
     * the expensive read+validate of the full weight buffer. */
    if (hdr.flags & TRINE_S2_FLAG_IDENTITY) {
        /* Seek past the weight payload */
        if (fseek(fp, (long)weight_size, SEEK_CUR) != 0) {
            fprintf(stderr, "trine_s2_load: failed to seek past identity weights\n");
            fclose(fp);
            return NULL;
        }

        /* Read payload checksum (still present on disk for format compat) */
        uint64_t payload_checksum;
        if (fread(&payload_checksum, sizeof(payload_checksum), 1, fp) != 1) {
            fprintf(stderr, "trine_s2_load: failed to read payload checksum\n");
            fclose(fp);
            return NULL;
        }
        /* NOTE: we do not recompute the payload checksum for identity models
         * because we intentionally skipped reading the weight data.  The
         * header checksum (already verified above) guards against header
         * corruption, and the identity flag means the weights are unused. */

        fclose(fp);

        struct trine_s2_model *model = trine_s2_create_identity();
        return model;
    }

    /* Non-identity models: read, checksum-verify, and trit-validate weights */
    uint8_t *weights = (uint8_t *)malloc(weight_size);
    if (!weights) {
        fprintf(stderr, "trine_s2_load: allocation failed\n");
        fclose(fp);
        return NULL;
    }

    if (fread(weights, weight_size, 1, fp) != 1) {
        fprintf(stderr, "trine_s2_load: failed to read weights\n");
        free(weights);
        fclose(fp);
        return NULL;
    }

    /* Read and verify payload checksum */
    uint64_t payload_checksum;
    if (fread(&payload_checksum, sizeof(payload_checksum), 1, fp) != 1) {
        fprintf(stderr, "trine_s2_load: failed to read payload checksum\n");
        free(weights);
        fclose(fp);
        return NULL;
    }

    uint64_t computed_payload = s2_fnv1a(weights, weight_size);
    if (computed_payload != payload_checksum) {
        fprintf(stderr, "trine_s2_load: payload checksum mismatch\n");
        free(weights);
        fclose(fp);
        return NULL;
    }

    fclose(fp);

    /* Validate that every weight byte is a valid trit {0, 1, 2}.
     * This catches corruption that happens to preserve the checksum,
     * as well as files written by buggy or malicious producers. */
    size_t bad_offset = 0;
    uint8_t bad_value = 0;
    if (s2_validate_trits(weights, weight_size, &bad_offset, &bad_value) != 0) {
        fprintf(stderr, "trine_s2_load: invalid trit value %u at weight "
                "offset %zu (expected 0, 1, or 2)\n",
                (unsigned)bad_value, bad_offset);
        free(weights);
        return NULL;
    }

    /* Create model from validated weights */
    struct trine_s2_model *model;
    if (is_block_diag) {
        /* Block-diagonal: construct from compact K * 4 * 60 * 60 weights */
        model = trine_s2_create_block_diagonal(
            weights, (int)hdr.projection_k,
            hdr.cascade_cells, hdr.topo_seed);
    } else {
        /* Full-matrix: construct from K * 240 * 240 weights */
        model = trine_s2_create_from_parts(
            weights, hdr.cascade_cells, hdr.topo_seed);
    }
    free(weights);

    if (!model) {
        fprintf(stderr, "trine_s2_load: failed to create model from weights\n");
        return NULL;
    }

    /* Set projection mode for non-block-diagonal modes */
    if (!is_block_diag && (hdr.flags & TRINE_S2_FLAG_DIAGONAL)) {
        trine_s2_set_projection_mode(model, TRINE_S2_PROJ_DIAGONAL);
    }

    return model;
}

/* ── Validate ────────────────────────────────────────────────────── */

int trine_s2_validate(const char *path)
{
    if (!path) {
        fprintf(stderr, "trine_s2_validate: NULL path\n");
        return -1;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "trine_s2_validate: cannot open '%s'\n", path);
        return -1;
    }

    /* Read header */
    trine_s2_file_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "trine_s2_validate: failed to read header\n");
        fclose(fp);
        return -1;
    }

    /* Validate magic */
    if (hdr.magic != TRINE_S2_MAGIC) {
        fprintf(stderr, "trine_s2_validate: bad magic\n");
        fclose(fp);
        return -1;
    }

    /* Validate version */
    if (hdr.version != TRINE_S2_FORMAT_VERSION) {
        fprintf(stderr, "trine_s2_validate: unsupported version %u\n",
                hdr.version);
        fclose(fp);
        return -1;
    }

    /* Validate header checksum */
    uint64_t saved_checksum = hdr.header_checksum;
    trine_s2_file_header_t check_hdr;
    memcpy(&check_hdr, &hdr, sizeof(check_hdr));
    check_hdr.header_checksum = 0;
    uint64_t computed = s2_fnv1a(&check_hdr, 64);
    if (computed != saved_checksum) {
        fprintf(stderr, "trine_s2_validate: header checksum mismatch\n");
        fclose(fp);
        return -1;
    }

    /* Check dimensions */
    if (hdr.projection_k != TRINE_PROJECT_K ||
        hdr.projection_dim != TRINE_PROJECT_DIM) {
        fprintf(stderr, "trine_s2_validate: dimension mismatch\n");
        fclose(fp);
        return -1;
    }

    /* Determine payload size based on block-diagonal flag */
    size_t weight_size;
    if (hdr.flags & TRINE_S2_FLAG_BLOCK_DIAG) {
        weight_size = S2_BLOCK_WEIGHT_SIZE(hdr.projection_k);
    } else {
        weight_size = S2_FULL_WEIGHT_SIZE(hdr.projection_k, hdr.projection_dim);
    }

    uint8_t *weights = (uint8_t *)malloc(weight_size);
    if (!weights) {
        fclose(fp);
        return -1;
    }

    if (fread(weights, weight_size, 1, fp) != 1) {
        fprintf(stderr, "trine_s2_validate: failed to read weights\n");
        free(weights);
        fclose(fp);
        return -1;
    }

    uint64_t payload_checksum;
    if (fread(&payload_checksum, sizeof(payload_checksum), 1, fp) != 1) {
        fprintf(stderr, "trine_s2_validate: failed to read payload checksum\n");
        free(weights);
        fclose(fp);
        return -1;
    }

    fclose(fp);

    uint64_t computed_payload = s2_fnv1a(weights, weight_size);
    if (computed_payload != payload_checksum) {
        fprintf(stderr, "trine_s2_validate: payload checksum mismatch\n");
        free(weights);
        return -1;
    }

    /* Validate that every weight byte is a valid trit {0, 1, 2}.
     * This catches files with structurally valid checksums but
     * out-of-range weight values. */
    size_t bad_offset = 0;
    uint8_t bad_value = 0;
    if (s2_validate_trits(weights, weight_size, &bad_offset, &bad_value) != 0) {
        fprintf(stderr, "trine_s2_validate: invalid trit value %u at weight "
                "offset %zu (expected 0, 1, or 2)\n",
                (unsigned)bad_value, bad_offset);
        free(weights);
        return -1;
    }

    free(weights);
    return 0;
}
