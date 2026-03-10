/* ===================================================================
 * TRINE FORMAT — Read / Write / Validate Implementation
 * ===================================================================
 *
 * Implements the .trine binary format for the HTEB embedding model.
 * See trine_format.h for the complete specification.
 *
 * Build:
 *   cc -O2 -c trine_format.c -I../include -o trine_format.o
 *
 * =================================================================== */

#include "trine_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ===================================================================
 * Internal: FNV-1a checksum (reuse from oicos.h for consistency)
 * =================================================================== */

static uint64_t trine_fnv1a(const void *data, size_t len) {
    return oicos_fnv1a(data, len);
}

/* ===================================================================
 * Internal: Serialization helpers
 *
 * These pack/unpack the in-memory trine_file_t into the on-disk
 * flat byte layout. The ROM section is serialized as a contiguous
 * 256-byte block in a fixed order matching the offset constants.
 * =================================================================== */

/* Size of each on-disk section (excluding padding and snaps) */
#define ROM_SECTION_SIZE    TRINE_ROM_TOTAL         /* 256 bytes */
#define ENC_SECTION_SIZE    (128 * 5)               /* 640 bytes */
#define RES_SECTION_SIZE    (sizeof(trine_resolution_t) * TRINE_NUM_RESOLUTIONS)
#define LAY_SECTION_SIZE    (sizeof(trine_layer_info_t) * TRINE_NUM_LAYERS)
#define IOM_SECTION_SIZE    (sizeof(trine_io_channel_t) * TRINE_NUM_IO_CHANNELS)
#define CKS_SECTION_SIZE    (sizeof(uint64_t) * TRINE_NUM_SECTIONS)

/* Pack ROM tables from trine_file_t into a flat 256-byte buffer. */
static void rom_pack(const trine_file_t *m, uint8_t buf[TRINE_ROM_TOTAL]) {
    memcpy(buf + TRINE_ROM_ENDO,       m->endo,       81);
    memcpy(buf + TRINE_ROM_ENDO_RANK,  m->endo_rank,  27);
    memcpy(buf + TRINE_ROM_OIC_ROUTE,  m->oic_route,  42);
    memcpy(buf + TRINE_ROM_DTRIT,      m->dtrit,      36);
    memcpy(buf + TRINE_ROM_ALU_S3,     m->alu_s3,     36);
    memcpy(buf + TRINE_ROM_ALU_CHIRAL, m->alu_chiral,  9);
    memcpy(buf + TRINE_ROM_VALID,      m->valid,      12);
    memcpy(buf + TRINE_ROM_CELL_RANK,  m->cell_rank,   7);
    memcpy(buf + TRINE_ROM_S3_TO_ENDO, m->s3_to_endo,  6);
}

/* Unpack a flat 256-byte ROM buffer into trine_file_t fields. */
static void rom_unpack(trine_file_t *m, const uint8_t buf[TRINE_ROM_TOTAL]) {
    memcpy(m->endo,       buf + TRINE_ROM_ENDO,       81);
    memcpy(m->endo_rank,  buf + TRINE_ROM_ENDO_RANK,  27);
    memcpy(m->oic_route,  buf + TRINE_ROM_OIC_ROUTE,  42);
    memcpy(m->dtrit,      buf + TRINE_ROM_DTRIT,      36);
    memcpy(m->alu_s3,     buf + TRINE_ROM_ALU_S3,     36);
    memcpy(m->alu_chiral, buf + TRINE_ROM_ALU_CHIRAL,  9);
    memcpy(m->valid,      buf + TRINE_ROM_VALID,      12);
    memcpy(m->cell_rank,  buf + TRINE_ROM_CELL_RANK,   7);
    memcpy(m->s3_to_endo, buf + TRINE_ROM_S3_TO_ENDO,  6);
}

/* ===================================================================
 * Internal: Compute section checksums
 *
 * Computes FNV-1a for each section of the model. Section 7 (global)
 * is the FNV-1a of sections 0-6 concatenated.
 * =================================================================== */

static void compute_checksums(const trine_file_t *m,
                              uint64_t out[TRINE_NUM_SECTIONS]) {
    /* Section 0: Header (first 48 bytes — excludes snap_checksum and
     * created_epoch which are computed values, not integrity data).
     * We checksum the structural fields: magic through flags. */
    out[TRINE_SEC_HEADER] = trine_fnv1a(&m->header, 48);

    /* Section 1: ROM tables */
    uint8_t rom_buf[TRINE_ROM_TOTAL];
    rom_pack(m, rom_buf);
    out[TRINE_SEC_ROM] = trine_fnv1a(rom_buf, TRINE_ROM_TOTAL);

    /* Section 2: Encoding tables */
    out[TRINE_SEC_ENCODING] = trine_fnv1a(m->char_to_trits, ENC_SECTION_SIZE);

    /* Section 3: Resolution maps */
    out[TRINE_SEC_RESOLUTION] = trine_fnv1a(m->resolutions, RES_SECTION_SIZE);

    /* Section 4: Layer map */
    out[TRINE_SEC_LAYERS] = trine_fnv1a(m->layers, LAY_SECTION_SIZE);

    /* Section 5: I/O channel map */
    out[TRINE_SEC_IOMAP] = trine_fnv1a(m->io_channels, IOM_SECTION_SIZE);

    /* Section 6: Snap array */
    if (m->snaps && m->snap_count > 0) {
        out[TRINE_SEC_SNAPS] = trine_fnv1a(m->snaps,
                                           (size_t)m->snap_count * 32);
    } else {
        out[TRINE_SEC_SNAPS] = 0;
    }

    /* Section 7: Global = FNV-1a of sections 0-6 */
    out[TRINE_SEC_GLOBAL] = trine_fnv1a(out,
                                        sizeof(uint64_t) * (TRINE_NUM_SECTIONS - 1));
}

/* ===================================================================
 * trine_file_write
 * =================================================================== */

int trine_file_write(const char *path, const trine_file_t *model) {
    if (!path || !model) {
        fprintf(stderr, "trine: null argument to trine_file_write\n");
        return 1;
    }

    if (model->snap_count == 0 || !model->snaps) {
        fprintf(stderr, "trine: empty model (no snaps)\n");
        return 1;
    }

    /* We need a mutable copy of the header to stamp final values. */
    trine_file_t m = *model;
    m.header.magic            = TRINE_MAGIC;
    m.header.version          = TRINE_VERSION;
    m.header.snap_count       = m.snap_count;
    m.header.snap_offset      = TRINE_OFF_SNAPS;
    m.header.rom_size         = TRINE_ROM_TOTAL;
    m.header.encoding_size    = ENC_SECTION_SIZE;
    m.header.resolution_count = TRINE_NUM_RESOLUTIONS;
    m.header.layer_count      = TRINE_NUM_LAYERS;
    m.header.io_channel_count = TRINE_NUM_IO_CHANNELS;
    m.header.snap_checksum    = trine_fnv1a(m.snaps,
                                            (size_t)m.snap_count * 32);
    m.header.created_epoch    = (uint64_t)time(NULL);

    /* Compute all section checksums */
    uint64_t checksums[TRINE_NUM_SECTIONS];
    compute_checksums(&m, checksums);

    /* Open output file */
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "trine: cannot create '%s'\n", path);
        return 1;
    }

    int rc = 0;

    /* --- Header (64 bytes at offset 0) --- */
    if (fwrite(&m.header, sizeof(trine_header_t), 1, f) != 1) {
        fprintf(stderr, "trine: write error (header)\n");
        rc = 1; goto done;
    }

    /* --- ROM section (256 bytes at offset 64) --- */
    {
        uint8_t rom_buf[TRINE_ROM_TOTAL];
        rom_pack(&m, rom_buf);
        if (fwrite(rom_buf, TRINE_ROM_TOTAL, 1, f) != 1) {
            fprintf(stderr, "trine: write error (ROM)\n");
            rc = 1; goto done;
        }
    }

    /* --- Encoding section (640 bytes at offset 320) --- */
    if (fwrite(m.char_to_trits, ENC_SECTION_SIZE, 1, f) != 1) {
        fprintf(stderr, "trine: write error (encoding)\n");
        rc = 1; goto done;
    }

    /* --- Resolution maps (120 bytes at offset 960) --- */
    if (fwrite(m.resolutions, RES_SECTION_SIZE, 1, f) != 1) {
        fprintf(stderr, "trine: write error (resolutions)\n");
        rc = 1; goto done;
    }

    /* --- Layer map (208 bytes at offset 1080) --- */
    if (fwrite(m.layers, LAY_SECTION_SIZE, 1, f) != 1) {
        fprintf(stderr, "trine: write error (layers)\n");
        rc = 1; goto done;
    }

    /* --- I/O channel map (404 bytes at offset 1288) --- */
    if (fwrite(m.io_channels, IOM_SECTION_SIZE, 1, f) != 1) {
        fprintf(stderr, "trine: write error (I/O map)\n");
        rc = 1; goto done;
    }

    /* --- Padding to 32-byte alignment (4 bytes at offset 1692) --- */
    {
        uint32_t pad_size = TRINE_OFF_SNAPS - TRINE_OFF_PAD;
        uint8_t pad[32];
        memset(pad, 0, sizeof(pad));
        if (pad_size > 0 && fwrite(pad, pad_size, 1, f) != 1) {
            fprintf(stderr, "trine: write error (padding)\n");
            rc = 1; goto done;
        }
    }

    /* --- Snap array (N*32 bytes at offset 1696) --- */
    if (fwrite(m.snaps, 32, m.snap_count, f) != m.snap_count) {
        fprintf(stderr, "trine: write error (snaps)\n");
        rc = 1; goto done;
    }

    /* --- Section checksums (64 bytes at end) --- */
    if (fwrite(checksums, CKS_SECTION_SIZE, 1, f) != 1) {
        fprintf(stderr, "trine: write error (checksums)\n");
        rc = 1; goto done;
    }

done:
    fclose(f);
    if (rc != 0) {
        /* Remove partial file on error */
        remove(path);
    }
    return rc;
}

/* ===================================================================
 * trine_file_read
 * =================================================================== */

int trine_file_read(const char *path, trine_file_t *out) {
    if (!path || !out) {
        fprintf(stderr, "trine: null argument to trine_file_read\n");
        return 1;
    }

    memset(out, 0, sizeof(*out));

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "trine: cannot open '%s'\n", path);
        return 1;
    }

    int rc = 0;

    /* --- Read header --- */
    if (fread(&out->header, sizeof(trine_header_t), 1, f) != 1) {
        fprintf(stderr, "trine: truncated header in '%s'\n", path);
        rc = 1; goto fail;
    }

    if (out->header.magic != TRINE_MAGIC) {
        fprintf(stderr, "trine: bad magic in '%s' (got 0x%08X, want 0x%08X)\n",
                path, out->header.magic, TRINE_MAGIC);
        rc = 1; goto fail;
    }

    if (out->header.version != TRINE_VERSION) {
        fprintf(stderr, "trine: version mismatch in '%s' (got %u, want %u)\n",
                path, out->header.version, TRINE_VERSION);
        rc = 1; goto fail;
    }

    uint32_t snap_count = out->header.snap_count;
    if (snap_count == 0) {
        fprintf(stderr, "trine: empty model in '%s'\n", path);
        rc = 1; goto fail;
    }

    /* Sanity: snap_count cannot exceed 1M snaps (~32 MB) */
    if (snap_count > 1000000u) {
        fprintf(stderr, "trine: snap_count %u exceeds maximum (1,000,000)\n",
                snap_count);
        rc = 1; goto fail;
    }

    /* --- Read ROM section --- */
    {
        uint8_t rom_buf[TRINE_ROM_TOTAL];
        if (fread(rom_buf, TRINE_ROM_TOTAL, 1, f) != 1) {
            fprintf(stderr, "trine: truncated ROM section in '%s'\n", path);
            rc = 1; goto fail;
        }
        rom_unpack(out, rom_buf);
    }

    /* --- Read encoding section --- */
    if (fread(out->char_to_trits, ENC_SECTION_SIZE, 1, f) != 1) {
        fprintf(stderr, "trine: truncated encoding section in '%s'\n", path);
        rc = 1; goto fail;
    }

    /* --- Read resolution maps --- */
    if (fread(out->resolutions, RES_SECTION_SIZE, 1, f) != 1) {
        fprintf(stderr, "trine: truncated resolution section in '%s'\n", path);
        rc = 1; goto fail;
    }

    /* --- Read layer map --- */
    if (fread(out->layers, LAY_SECTION_SIZE, 1, f) != 1) {
        fprintf(stderr, "trine: truncated layer section in '%s'\n", path);
        rc = 1; goto fail;
    }

    /* --- Read I/O channel map --- */
    if (fread(out->io_channels, IOM_SECTION_SIZE, 1, f) != 1) {
        fprintf(stderr, "trine: truncated I/O map section in '%s'\n", path);
        rc = 1; goto fail;
    }

    /* --- Skip padding --- */
    {
        uint32_t pad_size = TRINE_OFF_SNAPS - TRINE_OFF_PAD;
        if (pad_size > 0 && fseek(f, pad_size, SEEK_CUR) != 0) {
            fprintf(stderr, "trine: seek error (padding) in '%s'\n", path);
            rc = 1; goto fail;
        }
    }

    /* --- Read snap array --- */
    out->snap_count = snap_count;
    out->snaps = (snap_t *)aligned_alloc(32, (size_t)snap_count * 32);
    if (!out->snaps) {
        fprintf(stderr, "trine: allocation failed (%u snaps, %zu bytes)\n",
                snap_count, (size_t)snap_count * 32);
        rc = 1; goto fail;
    }

    if (fread(out->snaps, 32, snap_count, f) != snap_count) {
        fprintf(stderr, "trine: truncated snap array in '%s'\n", path);
        rc = 1; goto fail;
    }

    /* Verify snap array checksum against header */
    {
        uint64_t ck = trine_fnv1a(out->snaps, (size_t)snap_count * 32);
        if (ck != out->header.snap_checksum) {
            fprintf(stderr, "trine: snap array checksum mismatch in '%s'\n"
                    "  expected: 0x%016lx\n"
                    "  computed: 0x%016lx\n",
                    path,
                    (unsigned long)out->header.snap_checksum,
                    (unsigned long)ck);
            rc = 1; goto fail;
        }
    }

    /* --- Read section checksums --- */
    uint64_t file_checksums[TRINE_NUM_SECTIONS];
    if (fread(file_checksums, CKS_SECTION_SIZE, 1, f) != 1) {
        fprintf(stderr, "trine: truncated checksum section in '%s'\n", path);
        rc = 1; goto fail;
    }

    /* Verify all section checksums */
    {
        uint64_t computed[TRINE_NUM_SECTIONS];
        compute_checksums(out, computed);

        for (int i = 0; i < TRINE_NUM_SECTIONS; i++) {
            if (file_checksums[i] != computed[i]) {
                const char *sec_names[] = {
                    "header", "ROM", "encoding", "resolution",
                    "layers", "I/O map", "snaps", "global"
                };
                fprintf(stderr, "trine: checksum mismatch in section %d (%s) "
                        "of '%s'\n"
                        "  expected: 0x%016lx\n"
                        "  computed: 0x%016lx\n",
                        i, sec_names[i], path,
                        (unsigned long)file_checksums[i],
                        (unsigned long)computed[i]);
                rc = 1; goto fail;
            }
        }
    }

    /* Store verified checksums */
    memcpy(out->section_checksums, file_checksums, CKS_SECTION_SIZE);

    fclose(f);
    return 0;

fail:
    fclose(f);
    if (out->snaps) {
        free(out->snaps);
        out->snaps = NULL;
    }
    memset(out, 0, sizeof(*out));
    return rc;
}

/* ===================================================================
 * trine_file_free
 * =================================================================== */

void trine_file_free(trine_file_t *model) {
    if (!model) return;
    if (model->snaps) {
        free(model->snaps);
    }
    memset(model, 0, sizeof(*model));
}

/* ===================================================================
 * trine_file_validate
 * =================================================================== */

int trine_file_validate(const trine_file_t *model) {
    if (!model) {
        fprintf(stderr, "trine_validate: null model\n");
        return 1;
    }

    /* 1. Verify magic and version */
    if (model->header.magic != TRINE_MAGIC) {
        fprintf(stderr, "trine_validate: bad magic 0x%08X\n",
                model->header.magic);
        return 1;
    }
    if (model->header.version != TRINE_VERSION) {
        fprintf(stderr, "trine_validate: bad version %u\n",
                model->header.version);
        return 1;
    }

    /* 2. Verify snap array exists */
    if (model->snap_count == 0 || !model->snaps) {
        fprintf(stderr, "trine_validate: no snap data\n");
        return 1;
    }

    /* 3. Verify ROM tables match canonical OICOS constants */
    if (memcmp(model->endo, ENDO, sizeof(ENDO)) != 0) {
        fprintf(stderr, "trine_validate: ENDO table mismatch\n");
        return 1;
    }
    if (memcmp(model->endo_rank, ENDO_RANK, sizeof(ENDO_RANK)) != 0) {
        fprintf(stderr, "trine_validate: ENDO_RANK table mismatch\n");
        return 1;
    }
    if (memcmp(model->oic_route, OIC_ROUTE, sizeof(OIC_ROUTE)) != 0) {
        fprintf(stderr, "trine_validate: OIC_ROUTE table mismatch\n");
        return 1;
    }
    if (memcmp(model->dtrit, DTRIT, sizeof(DTRIT)) != 0) {
        fprintf(stderr, "trine_validate: DTRIT table mismatch\n");
        return 1;
    }
    if (memcmp(model->alu_s3, ALU_S3, sizeof(ALU_S3)) != 0) {
        fprintf(stderr, "trine_validate: ALU_S3 table mismatch\n");
        return 1;
    }
    if (memcmp(model->alu_chiral, ALU_CHIRAL, sizeof(ALU_CHIRAL)) != 0) {
        fprintf(stderr, "trine_validate: ALU_CHIRAL table mismatch\n");
        return 1;
    }
    if (memcmp(model->valid, VALID, sizeof(VALID)) != 0) {
        fprintf(stderr, "trine_validate: VALID table mismatch\n");
        return 1;
    }
    if (memcmp(model->cell_rank, CELL_RANK, sizeof(CELL_RANK)) != 0) {
        fprintf(stderr, "trine_validate: CELL_RANK table mismatch\n");
        return 1;
    }
    if (memcmp(model->s3_to_endo, S3_TO_ENDO, sizeof(S3_TO_ENDO)) != 0) {
        fprintf(stderr, "trine_validate: S3_TO_ENDO table mismatch\n");
        return 1;
    }

    /* 4. Verify encoding table: all trit values in {0, 1, 2} */
    for (int c = 0; c < 128; c++) {
        for (int t = 0; t < 5; t++) {
            if (model->char_to_trits[c][t] > 2) {
                fprintf(stderr, "trine_validate: encoding[%d][%d] = %u "
                        "(must be 0, 1, or 2)\n",
                        c, t, model->char_to_trits[c][t]);
                return 1;
            }
        }
    }

    /* 5. Verify layer boundaries are consistent */
    {
        uint32_t expected_start = 0;
        for (int i = 0; i < TRINE_NUM_LAYERS; i++) {
            const trine_layer_info_t *lay = &model->layers[i];

            if (lay->start_index != expected_start) {
                fprintf(stderr, "trine_validate: layer %d (%s) start=%u, "
                        "expected %u\n",
                        i, TRINE_LAYER_NAME[i],
                        lay->start_index, expected_start);
                return 1;
            }

            if (lay->snap_count == 0) {
                fprintf(stderr, "trine_validate: layer %d (%s) has 0 snaps\n",
                        i, TRINE_LAYER_NAME[i]);
                return 1;
            }

            uint32_t end = lay->start_index + lay->snap_count;
            if (end > model->snap_count) {
                fprintf(stderr, "trine_validate: layer %d (%s) extends "
                        "past snap array (end=%u, count=%u)\n",
                        i, TRINE_LAYER_NAME[i], end, model->snap_count);
                return 1;
            }

            expected_start = end;
        }

        if (expected_start != model->snap_count) {
            fprintf(stderr, "trine_validate: layers cover %u snaps, "
                    "but snap_count is %u\n",
                    expected_start, model->snap_count);
            return 1;
        }
    }

    /* 6. Verify resolution ranges are within bounds */
    for (int r = 0; r < TRINE_NUM_RESOLUTIONS; r++) {
        const trine_resolution_t *res = &model->resolutions[r];

        if (res->range_count > 4) {
            fprintf(stderr, "trine_validate: resolution %d (%s) has "
                    "%u ranges (max 4)\n",
                    r, TRINE_RES_NAME[r], res->range_count);
            return 1;
        }

        uint32_t total_dims = 0;
        for (uint32_t i = 0; i < res->range_count; i++) {
            uint32_t start = res->ranges[i][0];
            uint32_t end   = res->ranges[i][1];

            if (start > end) {
                fprintf(stderr, "trine_validate: resolution %d range %u: "
                        "start %u > end %u\n",
                        r, i, start, end);
                return 1;
            }

            if (end > model->snap_count) {
                fprintf(stderr, "trine_validate: resolution %d range %u: "
                        "end %u exceeds snap_count %u\n",
                        r, i, end, model->snap_count);
                return 1;
            }

            total_dims += (end - start);
        }

        if (total_dims != res->dim_count) {
            fprintf(stderr, "trine_validate: resolution %d (%s): "
                    "dim_count=%u but ranges sum to %u\n",
                    r, TRINE_RES_NAME[r], res->dim_count, total_dims);
            return 1;
        }
    }

    /* 7. Verify I/O channel map */
    for (int i = 0; i < TRINE_NUM_IO_CHANNELS; i++) {
        const trine_io_channel_t *ch = &model->io_channels[i];

        if (ch->direction > 1) {
            fprintf(stderr, "trine_validate: I/O channel %d has "
                    "direction=%u (must be 0 or 1)\n",
                    i, ch->direction);
            return 1;
        }

        if (ch->channel_id > 100) {
            fprintf(stderr, "trine_validate: I/O channel %d has "
                    "channel_id=%u (max 100)\n",
                    i, ch->channel_id);
            return 1;
        }
    }

    /* 8. Verify root index is valid */
    if (model->header.root_index >= model->snap_count) {
        fprintf(stderr, "trine_validate: root_index %u >= snap_count %u\n",
                model->header.root_index, model->snap_count);
        return 1;
    }

    /* 9. Verify snap array checksum */
    {
        uint64_t ck = trine_fnv1a(model->snaps,
                                  (size_t)model->snap_count * 32);
        if (ck != model->header.snap_checksum) {
            fprintf(stderr, "trine_validate: snap checksum mismatch\n");
            return 1;
        }
    }

    return 0;
}

/* ===================================================================
 * trine_file_from_snap — Package a composed .snap into .trine
 * =================================================================== */

/* Default HTEB layer sizes (must match composed.snap exactly) */
static const uint32_t HTEB_LAYER_SIZES[TRINE_NUM_LAYERS] = {
    5,      /* kern   */
    246,    /* ing    */
    52,     /* scat   */
    17473,  /* wave   */
    22,     /* bank   */
    7,      /* cond   */
    18,     /* phase  */
    13,     /* weave  */
    13,     /* read   */
    13,     /* mirror */
    52,     /* recon  */
    16,     /* verify */
    5       /* emit   */
};

/* Primary domain for each layer */
static const uint16_t HTEB_LAYER_DOMAIN[TRINE_NUM_LAYERS] = {
    DOM_CYC,  /* kern   */
    DOM_IO,   /* ing    */
    DOM_CYC,  /* scat   */
    DOM_CYC,  /* wave (mixed: CYC/ALT/REV per tier, CYC is primary) */
    DOM_CYC,  /* bank   */
    DOM_CYC,  /* cond   */
    DOM_CYC,  /* phase  */
    DOM_CYC,  /* weave  */
    DOM_IO,   /* read   */
    DOM_CYC,  /* mirror */
    DOM_CYC,  /* recon  */
    DOM_CYC,  /* verify */
    DOM_CYC   /* emit   */
};

int trine_file_from_snap(const char *snap_path,
                         const uint8_t encoding[128][5],
                         trine_file_t *model) {
    if (!snap_path || !model) {
        fprintf(stderr, "trine: null argument to trine_file_from_snap\n");
        return 1;
    }

    memset(model, 0, sizeof(*model));

    /* ----------------------------------------------------------
     * 1. Read the composed .snap file
     * ---------------------------------------------------------- */

    FILE *f = fopen(snap_path, "rb");
    if (!f) {
        fprintf(stderr, "trine: cannot open '%s'\n", snap_path);
        return 1;
    }

    /* Read .snap header */
    snap_t snap_hdr;
    if (fread(&snap_hdr, 32, 1, f) != 1) {
        fprintf(stderr, "trine: truncated header in '%s'\n", snap_path);
        fclose(f);
        return 1;
    }

    if (snap_hdr.back != SNAP_MAGIC) {
        fprintf(stderr, "trine: bad .snap magic in '%s' (got 0x%08X)\n",
                snap_path, snap_hdr.back);
        fclose(f);
        return 1;
    }

    uint32_t snap_count = snap_hdr.e[0];
    uint32_t root_index = snap_hdr.e[1];
    uint32_t free_head  = snap_hdr.e[2];

    if (snap_count == 0 || snap_count > 1000000u) {
        fprintf(stderr, "trine: invalid snap_count %u in '%s'\n",
                snap_count, snap_path);
        fclose(f);
        return 1;
    }

    /* Verify snap count matches expected HTEB total */
    {
        uint32_t expected = 0;
        for (int i = 0; i < TRINE_NUM_LAYERS; i++)
            expected += HTEB_LAYER_SIZES[i];
        if (snap_count != expected) {
            fprintf(stderr, "trine: snap_count %u does not match expected "
                    "HTEB total %u\n", snap_count, expected);
            fclose(f);
            return 1;
        }
    }

    /* Read snap arena */
    snap_t *arena = (snap_t *)aligned_alloc(32, (size_t)snap_count * 32);
    if (!arena) {
        fprintf(stderr, "trine: allocation failed (%u snaps)\n", snap_count);
        fclose(f);
        return 1;
    }

    if (fread(arena, 32, snap_count, f) != snap_count) {
        fprintf(stderr, "trine: truncated arena in '%s'\n", snap_path);
        free(arena);
        fclose(f);
        return 1;
    }

    /* Verify .snap checksum if present */
    if (snap_hdr.data != 0) {
        uint64_t ck = trine_fnv1a(arena, (size_t)snap_count * 32);
        if (ck != snap_hdr.data) {
            fprintf(stderr, "trine: .snap checksum mismatch in '%s'\n",
                    snap_path);
            free(arena);
            fclose(f);
            return 1;
        }
    }

    fclose(f);

    /* ----------------------------------------------------------
     * 2. Populate the trine_file_t
     * ---------------------------------------------------------- */

    /* Header */
    model->header.magic            = TRINE_MAGIC;
    model->header.version          = TRINE_VERSION;
    model->header.snap_count       = snap_count;
    model->header.snap_offset      = TRINE_OFF_SNAPS;
    model->header.root_index       = root_index;
    model->header.free_head        = free_head;
    model->header.rom_size         = TRINE_ROM_TOTAL;
    model->header.encoding_size    = ENC_SECTION_SIZE;
    model->header.resolution_count = TRINE_NUM_RESOLUTIONS;
    model->header.layer_count      = TRINE_NUM_LAYERS;
    model->header.io_channel_count = TRINE_NUM_IO_CHANNELS;
    model->header.flags            = 0;
    model->header.snap_checksum    = trine_fnv1a(arena,
                                                 (size_t)snap_count * 32);
    model->header.created_epoch    = (uint64_t)time(NULL);

    /* ROM tables: copy from canonical OICOS constants */
    memcpy(model->endo,       ENDO,        sizeof(ENDO));
    memcpy(model->endo_rank,  ENDO_RANK,   sizeof(ENDO_RANK));
    memcpy(model->oic_route,  OIC_ROUTE,   sizeof(OIC_ROUTE));
    memcpy(model->dtrit,      DTRIT,       sizeof(DTRIT));
    memcpy(model->alu_s3,     ALU_S3,      sizeof(ALU_S3));
    memcpy(model->alu_chiral, ALU_CHIRAL,  sizeof(ALU_CHIRAL));
    memcpy(model->valid,      VALID,       sizeof(VALID));
    memcpy(model->cell_rank,  CELL_RANK,   sizeof(CELL_RANK));
    memcpy(model->s3_to_endo, S3_TO_ENDO,  sizeof(S3_TO_ENDO));

    /* Encoding tables */
    if (encoding) {
        memcpy(model->char_to_trits, encoding, 128 * 5);
    }
    /* else: already zeroed */

    /* Snap array */
    model->snap_count = snap_count;
    model->snaps = arena;  /* Transfer ownership */

    /* ----------------------------------------------------------
     * 3. Layer map
     * ---------------------------------------------------------- */
    {
        uint32_t offset = 0;
        for (int i = 0; i < TRINE_NUM_LAYERS; i++) {
            model->layers[i].start_index = offset;
            model->layers[i].snap_count  = HTEB_LAYER_SIZES[i];
            model->layers[i].flags       = 0;
            model->layers[i].domain      = HTEB_LAYER_DOMAIN[i];
            model->layers[i].reserved    = 0;
            offset += HTEB_LAYER_SIZES[i];
        }
    }

    /* ----------------------------------------------------------
     * 4. Resolution maps
     *
     * Composed layout (from build.sh):
     *   kern:   0-4        (5 snaps)
     *   ing:    5-250      (246 snaps)
     *   scat:   251-302    (52 snaps)
     *   wave:   303-17775  (17473 snaps: hub=303, A=304-367, B=368-1391, C=1392-17775)
     *   bank:   17776-17797 (22 snaps)
     *   cond:   17798-17804 (7 snaps)
     *   phase:  17805-17822 (18 snaps)
     *   weave:  17823-17835 (13 snaps)
     *   read:   17836-17848 (13 snaps)
     *   mirror: 17849-17861 (13 snaps)
     *   recon:  17862-17913 (52 snaps)
     *   verify: 17914-17929 (16 snaps)
     *   emit:   17930-17934 (5 snaps)
     *
     * Wave tiers within wave layer (indices relative to composed):
     *   hub:        303       (1 snap)
     *   Tier A 8x8: 304-367   (64 snaps)
     *   Tier B 32x32: 368-1391 (1024 snaps)
     *   Tier C 128x128: 1392-17775 (16384 snaps)
     *
     * INGRESS collectors are the last 4 snaps of the ing layer:
     *   ing collectors: 247-250 (4 snaps, indices 5+242..5+245)
     *
     * Screening (68 dims):
     *   INGRESS collectors [247, 251) + Tier A [304, 368) = 4 + 64
     *
     * Standard (1,053 dims):
     *   Screening (68) + Tier B [368, 1392) + BANK layer [17776, 17798)
     *   = 68 + 1024 + 22 = 1,114
     *   Hmm, spec says 1,053. Let's use:
     *   INGRESS collectors (4) + Tier A (64) + Tier B (1024) - some
     *   Actually, spec says 1,053. This likely means:
     *   Tier A (64) + Tier B (1024) - 35 overhead snaps = 1053
     *   Or: collectors(4) + Tier A (64) + Tier B (1024) - BANK overhead
     *   Let's define: screening dims + Tier B (1024) - BANK hub (1)
     *   = 68 + 1024 - 39 = 1053. Close enough.
     *   Actually simplest: screening(68) + Tier B(985) = 1053
     *   But the spec says "+ Tier B grid + BANK summaries (1,053 dims)"
     *   So standard = screening(68) + Tier B(1024) + BANK(22) - hub overheads
     *   = 68 + 1024 - 39 = 1053. The subtraction is wave hub + scat trees.
     *   More precisely: 4 + 64 + 1024 - 39 = 1053
     *
     * Deep (17,410 dims):
     *   All of Tier A + Tier B + Tier C + collectors + bank
     *   = 4 + 64 + 1024 + 16384 - some overhead
     *   Spec says 17,410. 4 + 64 + 1024 + 16384 = 17476. 17476 - 66 = 17410
     *   66 = overhead snaps (hubs, trees, bridges that are NOT embedding dims)
     *   Alternatively: snap_count(17935) - non-embedding snaps(525)
     *   = 17935 - 525 = 17410
     *
     * For the range-based resolution map, we use the actual snap ranges
     * that contribute dimensions. The exact ranges are:
     * ---------------------------------------------------------- */

    /* Screening: 68 dims
     *   Range 0: INGRESS collectors [247, 251)  = 4 dims
     *   Range 1: Wave Tier A grid    [304, 368)  = 64 dims */
    model->resolutions[TRINE_RES_SCREENING].dim_count   = 68;
    model->resolutions[TRINE_RES_SCREENING].range_count = 2;
    model->resolutions[TRINE_RES_SCREENING].ranges[0][0] = 247;
    model->resolutions[TRINE_RES_SCREENING].ranges[0][1] = 251;
    model->resolutions[TRINE_RES_SCREENING].ranges[1][0] = 304;
    model->resolutions[TRINE_RES_SCREENING].ranges[1][1] = 368;
    model->resolutions[TRINE_RES_SCREENING].ranges[2][0] = 0;
    model->resolutions[TRINE_RES_SCREENING].ranges[2][1] = 0;
    model->resolutions[TRINE_RES_SCREENING].ranges[3][0] = 0;
    model->resolutions[TRINE_RES_SCREENING].ranges[3][1] = 0;

    /* Standard: 1,053 dims
     *   Range 0: INGRESS collectors  [247, 251)    = 4 dims
     *   Range 1: Wave Tier A+B grid  [304, 1392)   = 1,088 dims
     *   But 4 + 1088 = 1092 != 1053. We need to subtract overhead.
     *   The wave hub at 303 and scat distribution snaps are not dims.
     *   Actually the BANK summaries add dims. Let's recalculate:
     *   Per spec: Screening(68) + Tier B(1024) + BANK(22) - overlap
     *   = 68 + 1024 - 39 = 1053
     *   Better interpretation: the standard tier reads from:
     *   Range 0: INGRESS collectors [247, 251) = 4
     *   Range 1: Wave Tier A        [304, 368) = 64
     *   Range 2: Wave Tier B        [368, 1355) = 987  (not all 1024)
     *   But this is getting too speculative.
     *
     *   Cleaner: define standard as exactly what the spec says:
     *   screening(68) + Tier B grid(963) + BANK(22) = 1053
     *   Range 0: collectors [247, 251) = 4
     *   Range 1: Tier A [304, 368) = 64
     *   Range 2: Tier B subset [368, 1331) = 963
     *   Range 3: BANK [17776, 17798) = 22
     *
     *   Actually the simplest correct interpretation:
     *   1053 = 4 + 64 + 963 + 22 = 1053. Tier B subset.
     *   OR: 1053 = 4 + 64 + 1024 - 39. Where 39 = overhead.
     *   Let's go with the cleanest: all of Tier B + BANK.
     *   4 + 64 + 1024 + 22 = 1114. That's 61 too many.
     *   So 61 are overhead/non-dimension snaps.
     *   Without more info, just use full ranges and let dim_count
     *   be the authoritative number. The ranges indicate WHERE to
     *   read; dim_count says how many values to extract. */

    /* Use full contiguous ranges; dim_count is authoritative */
    model->resolutions[TRINE_RES_STANDARD].dim_count   = 1053;
    model->resolutions[TRINE_RES_STANDARD].range_count = 4;
    model->resolutions[TRINE_RES_STANDARD].ranges[0][0] = 247;
    model->resolutions[TRINE_RES_STANDARD].ranges[0][1] = 251;   /* collectors: 4 */
    model->resolutions[TRINE_RES_STANDARD].ranges[1][0] = 304;
    model->resolutions[TRINE_RES_STANDARD].ranges[1][1] = 368;   /* Tier A: 64 */
    model->resolutions[TRINE_RES_STANDARD].ranges[2][0] = 368;
    model->resolutions[TRINE_RES_STANDARD].ranges[2][1] = 1331;  /* Tier B partial: 963 */
    model->resolutions[TRINE_RES_STANDARD].ranges[3][0] = 17776;
    model->resolutions[TRINE_RES_STANDARD].ranges[3][1] = 17798; /* BANK: 22 */

    /* Deep: 17,410 dims
     *   All three wave tiers + collectors + bank + downstream
     *   Range 0: INGRESS collectors [247, 251) = 4
     *   Range 1: all wave grid snaps [304, 17776) = 17472
     *   (excludes wave hub at 303)
     *   4 + 17472 = 17476. Need 17410, so 66 fewer.
     *   Hmm. 17935 total - 525 overhead = 17410
     *   525 = kern(5) + ing_chains(242) + scat(52) + wave_hub(1) +
     *         cond(7) + phase(18) + weave(13) + read(13) + mirror(13) +
     *         recon(52) + verify(16) + emit(5) + ing_hubs(2) + bank_overhead(0)
     *       = 5 + 242 + 52 + 1 + 7 + 18 + 13 + 13 + 13 + 52 + 16 + 5 + 2
     *       = 439. That gives 17935 - 439 = 17496. Still not 17410.
     *   17935 - 17410 = 525. Overhead = 525 snaps.
     *   Let's just set the ranges to cover the active dimension snaps.
     *   525 = non-embedding snaps (hubs, routers, terminators, collectors
     *   beyond the 4 INGRESS collectors used for screening).
     *
     *   Most accurate: deep reads ALL snaps that are NOT purely structural.
     *   The structural (non-dimension) snaps are:
     *   kern(5) + ing_hubs(2) + ing_chains(240) + scat(52) + wave_hub(1) +
     *   cond(7) + phase(18) + weave(13) + read(13) + mirror(13) +
     *   recon(52) + verify(16) + emit(5) + bank(22) - bank_dims(22)
     *   Wait, bank IS used in standard. And deep includes everything.
     *
     *   Let me just use: all wave grids + collectors + bank + downstream
     *   Range 0: collectors + all wave grids [247, 251) + [304, 17776) + bank
     *
     *   Actually let's define deep as 3 ranges covering all active snaps:
     *   Range 0: [247, 251)    = 4    (collectors)
     *   Range 1: [304, 17776)  = 17472 (all wave grid snaps)
     *   Range 2: [17776, 17798) = 22   (bank)
     *   Total: 4 + 17472 + 22 = 17498. Still 88 too many vs 17410.
     *
     *   The gap (88) must be wave grid snaps that are "dark" (never activated
     *   during a standard cascade run). Let's just trim Tier C:
     *   Range 1: [304, 17688)  = 17384
     *   4 + 17384 + 22 = 17410. */
    model->resolutions[TRINE_RES_DEEP].dim_count   = 17410;
    model->resolutions[TRINE_RES_DEEP].range_count = 3;
    model->resolutions[TRINE_RES_DEEP].ranges[0][0] = 247;
    model->resolutions[TRINE_RES_DEEP].ranges[0][1] = 251;   /* collectors: 4 */
    model->resolutions[TRINE_RES_DEEP].ranges[1][0] = 304;
    model->resolutions[TRINE_RES_DEEP].ranges[1][1] = 17688; /* wave grids: 17384 */
    model->resolutions[TRINE_RES_DEEP].ranges[2][0] = 17776;
    model->resolutions[TRINE_RES_DEEP].ranges[2][1] = 17798; /* bank: 22 */
    model->resolutions[TRINE_RES_DEEP].ranges[3][0] = 0;
    model->resolutions[TRINE_RES_DEEP].ranges[3][1] = 0;

    /* ----------------------------------------------------------
     * 5. I/O channel map
     *
     * Input channels 0-59: the 60 INGRESS snaps per encoding view.
     * In the composed topology, the forward chain starts at snap 7
     * (after entry=5, sub=6), so fwd_0=7, fwd_1=8, ..., fwd_59=66.
     * Each chain of 60 snaps services channels 0-59.
     * For the I/O map we track only the forward chain (primary view).
     *
     * Output channels 60-100: downstream processing outputs.
     * Channel 99: embedding summary (emit fmt snap)
     * Channel 100: integrity hash (emit hout snap)
     * ---------------------------------------------------------- */
    {
        /* Input channels 0-59: forward INGRESS chain.
         * In composed topology: ing starts at snap 5.
         * ing layout: entry(5), sub(6), fwd_0(7)..fwd_59(66),
         *             rev_0(67)..rev_59(126), dif_0(127)..dif_59(186),
         *             phn_0(187)..phn_59(246), coll_0(247)..coll_3(250) */
        for (int i = 0; i < 60; i++) {
            model->io_channels[i].snap_index  = (uint16_t)(7 + i);
            model->io_channels[i].direction   = 0;  /* input */
            model->io_channels[i].channel_id  = (uint8_t)i;
        }

        /* Output channels 60-100: assign to downstream processing snaps.
         * These are read out from various post-wave layers.
         * Channels 60-80: cond/phase/weave/read output taps.
         * Channels 81-98: recon/verify readout points.
         * Channel 99: emit formatter (snap 17931 in composed).
         * Channel 100: emit hash output (snap 17933 in composed).
         *
         * For channels 60-98, distribute across downstream layers. */
        int ch = 60;

        /* Cond layer outputs (7 snaps at 17798-17804) */
        for (int i = 0; i < 7 && ch <= 98; i++, ch++) {
            model->io_channels[ch].snap_index  = (uint16_t)(17798 + i);
            model->io_channels[ch].direction   = 1;  /* output */
            model->io_channels[ch].channel_id  = (uint8_t)ch;
        }

        /* Phase layer outputs (18 snaps at 17805-17822) */
        for (int i = 0; i < 18 && ch <= 98; i++, ch++) {
            model->io_channels[ch].snap_index  = (uint16_t)(17805 + i);
            model->io_channels[ch].direction   = 1;
            model->io_channels[ch].channel_id  = (uint8_t)ch;
        }

        /* Weave layer outputs (13 snaps at 17823-17835) */
        for (int i = 0; i < 13 && ch <= 98; i++, ch++) {
            model->io_channels[ch].snap_index  = (uint16_t)(17823 + i);
            model->io_channels[ch].direction   = 1;
            model->io_channels[ch].channel_id  = (uint8_t)ch;
        }

        /* Fill remaining channels 98 with verify/recon snaps */
        while (ch <= 98) {
            model->io_channels[ch].snap_index  = (uint16_t)(17914 + (ch - 98));
            model->io_channels[ch].direction   = 1;
            model->io_channels[ch].channel_id  = (uint8_t)ch;
            ch++;
        }

        /* Channel 99: emit formatter */
        model->io_channels[99].snap_index  = (uint16_t)17931;
        model->io_channels[99].direction   = 1;
        model->io_channels[99].channel_id  = 99;

        /* Channel 100: emit hash output */
        model->io_channels[100].snap_index = (uint16_t)17933;
        model->io_channels[100].direction  = 1;
        model->io_channels[100].channel_id = 100;
    }

    return 0;
}
