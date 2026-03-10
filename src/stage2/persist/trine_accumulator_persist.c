/* =====================================================================
 * TRINE Stage-2 — Accumulator Persistence (implementation)
 * =====================================================================
 *
 * Save/load Hebbian accumulator state for curriculum learning and
 * incremental training.  See trine_accumulator_persist.h for format.
 *
 * ===================================================================== */

#include "trine_accumulator_persist.h"
#include "trine_project.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

/* --------------------------------------------------------------------- */
/* FNV-1a checksum                                                        */
/* --------------------------------------------------------------------- */

static uint64_t acc_fnv1a(const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; i++)
        h = (h ^ p[i]) * 0x100000001b3ULL;
    return h;
}

/* --------------------------------------------------------------------- */
/* File header (packed, 64 bytes)                                         */
/* --------------------------------------------------------------------- */

typedef struct {
    uint32_t magic;                 /* TRINE_ACC_MAGIC */
    uint32_t version;               /* TRINE_ACC_FORMAT_VERSION */
    uint32_t flags;                 /* bit 0: diagonal, bit 1: block-diag */
    uint32_t projection_k;          /* 3 */
    uint32_t projection_dim;        /* 240 */
    uint32_t total_pairs;           /* pairs observed (capped at UINT32_MAX) */
    float    similarity_threshold;  /* training threshold */
    float    freeze_target_density; /* freeze density */
    uint8_t  reserved[24];          /* zero-filled */
    uint64_t header_checksum;       /* FNV-1a of bytes 0..55 */
} __attribute__((packed)) trine_acc_file_header_t;

_Static_assert(sizeof(trine_acc_file_header_t) == 64,
               "accumulator file header must be 64 bytes");

/* Full-matrix accumulator payload: K * 240 * 240 * sizeof(int32_t) */
#define ACC_FULL_PAYLOAD(k, dim) \
    ((size_t)(k) * (size_t)(dim) * (size_t)(dim) * sizeof(int32_t))

/* Block-diagonal accumulator payload: K * 4 * 60 * 60 * sizeof(int32_t) */
#define ACC_BLOCK_PAYLOAD(k) \
    ((size_t)(k) * 4u * 60u * 60u * sizeof(int32_t))

/* --------------------------------------------------------------------- */
/* Save                                                                    */
/* --------------------------------------------------------------------- */

int trine_accumulator_save(const trine_accumulator_t *acc,
                            const trine_hebbian_config_t *config,
                            int64_t pairs_observed,
                            const char *path)
{
    if (!acc || !path) {
        fprintf(stderr, "trine_accumulator_save: NULL accumulator or path\n");
        return -1;
    }

    /* Build header */
    trine_acc_file_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));

    hdr.magic          = TRINE_ACC_MAGIC;
    hdr.version        = TRINE_ACC_FORMAT_VERSION;
    hdr.projection_k   = TRINE_ACC_K;
    hdr.projection_dim = TRINE_ACC_DIM;
    hdr.total_pairs    = (pairs_observed > (int64_t)UINT32_MAX)
                          ? UINT32_MAX : (uint32_t)pairs_observed;

    if (config) {
        hdr.similarity_threshold  = config->similarity_threshold;
        hdr.freeze_target_density = config->freeze_target_density;
        if (config->projection_mode == 1) {
            hdr.flags |= TRINE_ACC_FLAG_DIAGONAL;
        }
    } else {
        hdr.similarity_threshold  = 0.5f;
        hdr.freeze_target_density = 0.33f;
    }

    /* Compute header checksum (over first 56 bytes) */
    hdr.header_checksum = acc_fnv1a(&hdr, 56);

    /* Gather counter data from all K matrices */
    size_t payload_size = (size_t)TRINE_ACC_K * TRINE_ACC_DIM * TRINE_ACC_DIM
                          * sizeof(int32_t);
    int32_t *payload = (int32_t *)malloc(payload_size);
    if (!payload) {
        fprintf(stderr, "trine_accumulator_save: allocation failed\n");
        return -1;
    }

    for (uint32_t k = 0; k < TRINE_ACC_K; k++) {
        const int32_t (*mat)[TRINE_ACC_DIM] =
            trine_accumulator_counters_const(acc, k);
        if (!mat) {
            free(payload);
            return -1;
        }
        memcpy(payload + k * TRINE_ACC_DIM * TRINE_ACC_DIM,
               mat, TRINE_ACC_DIM * TRINE_ACC_DIM * sizeof(int32_t));
    }

    uint64_t payload_checksum = acc_fnv1a(payload, payload_size);

    /* Write file */
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "trine_accumulator_save: cannot open '%s'\n", path);
        free(payload);
        return -1;
    }

    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1 ||
        fwrite(payload, payload_size, 1, fp) != 1 ||
        fwrite(&payload_checksum, sizeof(payload_checksum), 1, fp) != 1) {
        fprintf(stderr, "trine_accumulator_save: write failed\n");
        fclose(fp);
        free(payload);
        return -1;
    }

    fclose(fp);
    free(payload);
    return 0;
}

/* --------------------------------------------------------------------- */
/* Load                                                                    */
/* --------------------------------------------------------------------- */

trine_accumulator_t *trine_accumulator_load(const char *path,
                                             trine_hebbian_config_t *config_out,
                                             int64_t *pairs_out)
{
    if (!path) {
        fprintf(stderr, "trine_accumulator_load: NULL path\n");
        return NULL;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "trine_accumulator_load: cannot open '%s'\n", path);
        return NULL;
    }

    /* Read header */
    trine_acc_file_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "trine_accumulator_load: failed to read header\n");
        fclose(fp);
        return NULL;
    }

    /* Validate magic */
    if (hdr.magic != TRINE_ACC_MAGIC) {
        fprintf(stderr, "trine_accumulator_load: bad magic 0x%08x\n", hdr.magic);
        fclose(fp);
        return NULL;
    }

    /* Validate version */
    if (hdr.version != TRINE_ACC_FORMAT_VERSION) {
        fprintf(stderr, "trine_accumulator_load: unsupported version %u\n",
                hdr.version);
        fclose(fp);
        return NULL;
    }

    /* Validate header checksum */
    uint64_t saved_cksum = hdr.header_checksum;
    hdr.header_checksum = 0;
    uint64_t computed = acc_fnv1a(&hdr, 56);
    if (computed != saved_cksum) {
        fprintf(stderr, "trine_accumulator_load: header checksum mismatch\n");
        fclose(fp);
        return NULL;
    }
    hdr.header_checksum = saved_cksum;

    /* Validate dimensions */
    if (hdr.projection_k != TRINE_ACC_K ||
        hdr.projection_dim != TRINE_ACC_DIM) {
        fprintf(stderr, "trine_accumulator_load: dimension mismatch\n");
        fclose(fp);
        return NULL;
    }

    /* Block-diagonal files must use trine_block_accumulator_load() */
    if (hdr.flags & TRINE_ACC_FLAG_BLOCK_DIAG) {
        fprintf(stderr, "trine_accumulator_load: file is block-diagonal, "
                "use trine_block_accumulator_load()\n");
        fclose(fp);
        return NULL;
    }

    /* Read payload */
    size_t payload_size = ACC_FULL_PAYLOAD(TRINE_ACC_K, TRINE_ACC_DIM);
    int32_t *payload = (int32_t *)malloc(payload_size);
    if (!payload) {
        fclose(fp);
        return NULL;
    }

    if (fread(payload, payload_size, 1, fp) != 1) {
        fprintf(stderr, "trine_accumulator_load: failed to read counters\n");
        free(payload);
        fclose(fp);
        return NULL;
    }

    /* Read and verify payload checksum */
    uint64_t payload_checksum;
    if (fread(&payload_checksum, sizeof(payload_checksum), 1, fp) != 1) {
        fprintf(stderr, "trine_accumulator_load: failed to read checksum\n");
        free(payload);
        fclose(fp);
        return NULL;
    }

    uint64_t computed_payload = acc_fnv1a(payload, payload_size);
    if (computed_payload != payload_checksum) {
        fprintf(stderr, "trine_accumulator_load: payload checksum mismatch\n");
        free(payload);
        fclose(fp);
        return NULL;
    }

    fclose(fp);

    /* Create accumulator and copy data */
    trine_accumulator_t *acc = trine_accumulator_create();
    if (!acc) {
        free(payload);
        return NULL;
    }

    for (uint32_t k = 0; k < TRINE_ACC_K; k++) {
        int32_t (*mat)[TRINE_ACC_DIM] = trine_accumulator_counters(acc, k);
        if (!mat) {
            trine_accumulator_free(acc);
            free(payload);
            return NULL;
        }
        memcpy(mat, payload + k * TRINE_ACC_DIM * TRINE_ACC_DIM,
               TRINE_ACC_DIM * TRINE_ACC_DIM * sizeof(int32_t));
    }

    free(payload);

    /* Restore config if requested */
    if (config_out) {
        trine_hebbian_config_t defaults = TRINE_HEBBIAN_CONFIG_DEFAULT;
        *config_out = defaults;
        config_out->similarity_threshold  = hdr.similarity_threshold;
        config_out->freeze_target_density = hdr.freeze_target_density;
        config_out->projection_mode = (hdr.flags & TRINE_ACC_FLAG_DIAGONAL) ? 1 : 0;
    }

    if (pairs_out) {
        *pairs_out = (int64_t)hdr.total_pairs;
    }

    return acc;
}

/* --------------------------------------------------------------------- */
/* Validate                                                                */
/* --------------------------------------------------------------------- */

int trine_accumulator_validate(const char *path)
{
    if (!path) return -1;

    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    trine_acc_file_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    if (hdr.magic != TRINE_ACC_MAGIC || hdr.version != TRINE_ACC_FORMAT_VERSION) {
        fclose(fp);
        return -1;
    }

    uint64_t saved_cksum = hdr.header_checksum;
    hdr.header_checksum = 0;
    if (acc_fnv1a(&hdr, 56) != saved_cksum) {
        fclose(fp);
        return -1;
    }

    if (hdr.projection_k != TRINE_ACC_K || hdr.projection_dim != TRINE_ACC_DIM) {
        fclose(fp);
        return -1;
    }

    /* Determine payload size based on block-diagonal flag */
    size_t payload_size;
    if (hdr.flags & TRINE_ACC_FLAG_BLOCK_DIAG) {
        payload_size = ACC_BLOCK_PAYLOAD(hdr.projection_k);
    } else {
        payload_size = ACC_FULL_PAYLOAD(hdr.projection_k, hdr.projection_dim);
    }
    int32_t *payload = (int32_t *)malloc(payload_size);
    if (!payload) {
        fclose(fp);
        return -1;
    }

    if (fread(payload, payload_size, 1, fp) != 1) {
        free(payload);
        fclose(fp);
        return -1;
    }

    uint64_t payload_checksum;
    if (fread(&payload_checksum, sizeof(payload_checksum), 1, fp) != 1) {
        free(payload);
        fclose(fp);
        return -1;
    }

    uint64_t computed = acc_fnv1a(payload, payload_size);
    free(payload);
    fclose(fp);

    return (computed == payload_checksum) ? 0 : -1;
}

/* --------------------------------------------------------------------- */
/* Reconstruct from frozen model                                           */
/* --------------------------------------------------------------------- */

trine_accumulator_t *trine_accumulator_from_frozen(
    const void *projection_weights,
    int32_t reconstruction_scale)
{
    if (!projection_weights) return NULL;
    if (reconstruction_scale < 1) reconstruction_scale = 100;

    const trine_projection_t *proj = (const trine_projection_t *)projection_weights;

    trine_accumulator_t *acc = trine_accumulator_create();
    if (!acc) return NULL;

    for (uint32_t k = 0; k < TRINE_ACC_K; k++) {
        int32_t (*mat)[TRINE_ACC_DIM] = trine_accumulator_counters(acc, k);
        if (!mat) {
            trine_accumulator_free(acc);
            return NULL;
        }

        for (uint32_t i = 0; i < TRINE_ACC_DIM; i++) {
            for (uint32_t j = 0; j < TRINE_ACC_DIM; j++) {
                uint8_t w = proj->W[k][i][j];
                if (w == 2) {
                    mat[i][j] = reconstruction_scale;   /* positive gate */
                } else if (w == 1) {
                    mat[i][j] = -reconstruction_scale;  /* negated gate */
                } else {
                    mat[i][j] = 0;                      /* zero / uninformative */
                }
            }
        }
    }

    return acc;
}

/* --------------------------------------------------------------------- */
/* Block-Diagonal Accumulator: Save                                        */
/* --------------------------------------------------------------------- */

int trine_block_accumulator_save(const trine_block_accumulator_t *acc,
                                  float similarity_threshold,
                                  float freeze_target_density,
                                  int projection_mode,
                                  const char *path)
{
    if (!acc || !path) {
        fprintf(stderr, "trine_block_accumulator_save: NULL accumulator or path\n");
        return -1;
    }
    if (!acc->counters || acc->K <= 0) {
        fprintf(stderr, "trine_block_accumulator_save: invalid accumulator state\n");
        return -1;
    }

    /* Build header */
    trine_acc_file_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));

    hdr.magic          = TRINE_ACC_MAGIC;
    hdr.version        = TRINE_ACC_FORMAT_VERSION;
    hdr.flags          = TRINE_ACC_FLAG_BLOCK_DIAG;
    hdr.projection_k   = (uint32_t)acc->K;
    hdr.projection_dim = TRINE_ACC_DIM;
    hdr.total_pairs    = acc->pairs_observed;
    hdr.similarity_threshold  = similarity_threshold;
    hdr.freeze_target_density = freeze_target_density;

    if (projection_mode == 1) {
        hdr.flags |= TRINE_ACC_FLAG_DIAGONAL;
    }

    /* Compute header checksum (over first 56 bytes) */
    hdr.header_checksum = acc_fnv1a(&hdr, 56);

    /* Payload: K * 4 * 60 * 60 int32_t counters */
    size_t payload_size = ACC_BLOCK_PAYLOAD((uint32_t)acc->K);
    uint64_t payload_checksum = acc_fnv1a(acc->counters, payload_size);

    /* Write file */
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "trine_block_accumulator_save: cannot open '%s'\n", path);
        return -1;
    }

    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1 ||
        fwrite(acc->counters, payload_size, 1, fp) != 1 ||
        fwrite(&payload_checksum, sizeof(payload_checksum), 1, fp) != 1) {
        fprintf(stderr, "trine_block_accumulator_save: write failed\n");
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

/* --------------------------------------------------------------------- */
/* Block-Diagonal Accumulator: Load                                        */
/* --------------------------------------------------------------------- */

trine_block_accumulator_t *trine_block_accumulator_load(
    const char *path,
    float *threshold_out,
    float *density_out,
    uint32_t *pairs_out)
{
    if (!path) {
        fprintf(stderr, "trine_block_accumulator_load: NULL path\n");
        return NULL;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "trine_block_accumulator_load: cannot open '%s'\n", path);
        return NULL;
    }

    /* Read header */
    trine_acc_file_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "trine_block_accumulator_load: failed to read header\n");
        fclose(fp);
        return NULL;
    }

    /* Validate magic and version */
    if (hdr.magic != TRINE_ACC_MAGIC) {
        fprintf(stderr, "trine_block_accumulator_load: bad magic 0x%08x\n",
                hdr.magic);
        fclose(fp);
        return NULL;
    }
    if (hdr.version != TRINE_ACC_FORMAT_VERSION) {
        fprintf(stderr, "trine_block_accumulator_load: unsupported version %u\n",
                hdr.version);
        fclose(fp);
        return NULL;
    }

    /* Validate header checksum */
    uint64_t saved_cksum = hdr.header_checksum;
    hdr.header_checksum = 0;
    uint64_t computed = acc_fnv1a(&hdr, 56);
    if (computed != saved_cksum) {
        fprintf(stderr, "trine_block_accumulator_load: header checksum mismatch\n");
        fclose(fp);
        return NULL;
    }
    hdr.header_checksum = saved_cksum;

    /* Must be a block-diagonal file */
    if (!(hdr.flags & TRINE_ACC_FLAG_BLOCK_DIAG)) {
        fprintf(stderr, "trine_block_accumulator_load: file is not block-diagonal, "
                "use trine_accumulator_load()\n");
        fclose(fp);
        return NULL;
    }

    /* Validate dimensions */
    if (hdr.projection_k != TRINE_ACC_K ||
        hdr.projection_dim != TRINE_ACC_DIM) {
        fprintf(stderr, "trine_block_accumulator_load: dimension mismatch\n");
        fclose(fp);
        return NULL;
    }

    /* Read payload */
    size_t payload_size = ACC_BLOCK_PAYLOAD(hdr.projection_k);
    int32_t *payload = (int32_t *)malloc(payload_size);
    if (!payload) {
        fclose(fp);
        return NULL;
    }

    if (fread(payload, payload_size, 1, fp) != 1) {
        fprintf(stderr, "trine_block_accumulator_load: failed to read counters\n");
        free(payload);
        fclose(fp);
        return NULL;
    }

    /* Read and verify payload checksum */
    uint64_t payload_checksum;
    if (fread(&payload_checksum, sizeof(payload_checksum), 1, fp) != 1) {
        fprintf(stderr, "trine_block_accumulator_load: failed to read checksum\n");
        free(payload);
        fclose(fp);
        return NULL;
    }

    uint64_t computed_payload = acc_fnv1a(payload, payload_size);
    if (computed_payload != payload_checksum) {
        fprintf(stderr, "trine_block_accumulator_load: payload checksum mismatch\n");
        free(payload);
        fclose(fp);
        return NULL;
    }

    fclose(fp);

    /* Create block accumulator and copy counter data */
    trine_block_accumulator_t *bacc = trine_block_accumulator_create(
        (int)hdr.projection_k);
    if (!bacc) {
        free(payload);
        return NULL;
    }

    memcpy(bacc->counters, payload, payload_size);
    bacc->pairs_observed = hdr.total_pairs;
    free(payload);

    /* Return optional metadata */
    if (threshold_out) *threshold_out = hdr.similarity_threshold;
    if (density_out)   *density_out   = hdr.freeze_target_density;
    if (pairs_out)     *pairs_out     = hdr.total_pairs;

    return bacc;
}
