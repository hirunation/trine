/* ===================================================================
 * TRINE PACKAGER — Compose .snap -> .trine model file
 * ===================================================================
 *
 * The bridge from the OICOS build system to the standalone TRINE
 * embedding artifact. Reads a composed .snap file, populates all
 * sections (ROM, encoding, layers, resolution, I/O), and writes
 * the self-describing .trine binary.
 *
 * Usage:
 *   ./trine_pack <composed.snap> -o <model.trine> [-v] [--name NAME]
 *
 * Build:
 *   cc -O2 -o trine_pack trine_pack.c trine_format.c trine_encode.c \
 *      -I. -I../include
 *
 * =================================================================== */

#include "trine_format.h"

/* trine_encode.h defines TRINE_VERSION as a string ("1.0.1") while
 * trine_format.h defines it as an integer (1u) for the binary header.
 * Undefine before including to suppress the redefinition warning.
 * The format header's integer version is the one used in file I/O. */
#undef TRINE_VERSION
#include "trine_encode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ===================================================================
 * Constants
 * =================================================================== */

#define PROGRAM_NAME     "trine_pack"
#define PROGRAM_VERSION  "1.0.1"

/* Expected total snap count for the HTEB composed topology */
#define HTEB_SNAP_COUNT  17935

/* ===================================================================
 * Build the char_to_trits[128][5] encoding table
 *
 * Uses trine_encode() to encode each ASCII character individually,
 * then extracts the first 5 channels from the forward chain.
 * This guarantees the table exactly matches the encoder's mapping.
 * =================================================================== */

static void build_char_to_trits(uint8_t table[128][5])
{
    memset(table, 0, 128 * 5);

    for (int c = 0; c < 128; c++) {
        char ch = (char)c;
        uint8_t channels[240];
        trine_encode(&ch, 1, channels);

        /* Forward chain occupies channels 0-59.
         * A single character encodes into positions 0-4 (5 trits). */
        for (int t = 0; t < 5; t++)
            table[c][t] = channels[t];
    }
}

/* ===================================================================
 * Usage / help
 * =================================================================== */

static void usage(const char *progname)
{
    fprintf(stderr,
        "Usage: %s <composed.snap> -o <model.trine> [-v] [--name NAME]\n"
        "\n"
        "  Packages a composed .snap file into a .trine model file.\n"
        "\n"
        "  Arguments:\n"
        "    <composed.snap>   Input composed snap arena\n"
        "    -o <model.trine>  Output .trine model file\n"
        "    -v                Verbose: print summary after packaging\n"
        "    --name NAME       Model name (informational, printed in summary)\n"
        "\n"
        "  %s v%s\n",
        progname, PROGRAM_NAME, PROGRAM_VERSION);
}

/* ===================================================================
 * Verbose summary
 * =================================================================== */

static void print_summary(const char *output_path,
                           const char *model_name,
                           const trine_file_t *model,
                           uint64_t file_size)
{
    printf("\n");
    printf("==========================================================\n");
    printf("  TRINE Model: %s\n", model_name ? model_name : "(unnamed)");
    printf("==========================================================\n");
    printf("\n");

    /* File info */
    printf("  Output:       %s\n", output_path);
    printf("  File size:    %lu bytes (%.1f KB)\n",
           (unsigned long)file_size, (double)file_size / 1024.0);
    printf("  Snap count:   %u\n", model->snap_count);
    printf("  Root index:   %u\n", model->header.root_index);
    printf("  Free head:    0x%08X", model->header.free_head);
    if (model->header.free_head == SNAP_NIL)
        printf(" (NIL)");
    printf("\n");
    printf("\n");

    /* Layer breakdown */
    printf("  Layer Map (%u layers):\n", model->header.layer_count);
    printf("  %-10s  %6s  %6s  %s\n", "Layer", "Start", "Count", "Domain");
    printf("  %-10s  %6s  %6s  %s\n", "-----", "-----", "-----", "------");

    uint32_t total_snaps = 0;
    for (int i = 0; i < TRINE_NUM_LAYERS; i++) {
        const trine_layer_info_t *lay = &model->layers[i];
        const char *dom_name = "?";
        if (lay->domain < 6)
            dom_name = OICOS_DOM_NAME[lay->domain];

        printf("  %-10s  %6u  %6u  %s\n",
               TRINE_LAYER_NAME[i],
               lay->start_index,
               lay->snap_count,
               dom_name);

        total_snaps += lay->snap_count;
    }
    printf("  %-10s  %6s  %6u\n", "TOTAL", "", total_snaps);
    printf("\n");

    /* Resolution tiers */
    printf("  Resolution Tiers (%u tiers):\n", model->header.resolution_count);
    for (int r = 0; r < TRINE_NUM_RESOLUTIONS; r++) {
        const trine_resolution_t *res = &model->resolutions[r];
        printf("    %-12s  %6u dims  (%u ranges:",
               TRINE_RES_NAME[r], res->dim_count, res->range_count);

        for (uint32_t i = 0; i < res->range_count; i++) {
            if (res->ranges[i][0] == res->ranges[i][1])
                continue;
            printf(" [%u,%u)", res->ranges[i][0], res->ranges[i][1]);
        }
        printf(")\n");
    }
    printf("\n");

    /* I/O channels */
    int input_count = 0, output_count = 0;
    for (int i = 0; i < TRINE_NUM_IO_CHANNELS; i++) {
        if (model->io_channels[i].direction == 0)
            input_count++;
        else
            output_count++;
    }
    printf("  I/O Channels: %d input, %d output (%d total)\n",
           input_count, output_count, TRINE_NUM_IO_CHANNELS);
    printf("\n");

    /* Checksums */
    printf("  Section Checksums (FNV-1a):\n");
    const char *sec_names[] = {
        "Header", "ROM", "Encoding", "Resolution",
        "Layers", "I/O Map", "Snaps", "Global"
    };
    for (int i = 0; i < TRINE_NUM_SECTIONS; i++) {
        printf("    [%d] %-12s  0x%016lx\n",
               i, sec_names[i],
               (unsigned long)model->section_checksums[i]);
    }
    printf("\n");

    /* Golden hash: FNV-1a of the complete file */
    printf("  Snap checksum:  0x%016lx\n",
           (unsigned long)model->header.snap_checksum);
    printf("\n");

    /* Encoding table summary */
    int nonzero_entries = 0;
    for (int c = 0; c < 128; c++)
        for (int t = 0; t < 5; t++)
            if (model->char_to_trits[c][t] != 0)
                nonzero_entries++;

    printf("  Encoding table: %d/%d non-zero trit entries\n",
           nonzero_entries, 128 * 5);

    /* ROM verification */
    printf("  ROM:            %u bytes (canonical OICOS v%s)\n",
           model->header.rom_size, OICOS_VERSION);

    /* Creation time */
    time_t epoch = (time_t)model->header.created_epoch;
    char timebuf[64];
    struct tm *tm = localtime(&epoch);
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S %Z", tm);
    printf("  Created:        %s\n", timebuf);

    printf("\n");
    printf("==========================================================\n");
}

/* ===================================================================
 * Compute file golden hash (FNV-1a of the complete file on disk)
 * =================================================================== */

static uint64_t compute_file_hash(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return 0;
    }

    uint8_t *buf = (uint8_t *)malloc((size_t)size);
    if (!buf) {
        fclose(f);
        return 0;
    }

    if ((long)fread(buf, 1, (size_t)size, f) != size) {
        free(buf);
        fclose(f);
        return 0;
    }

    fclose(f);

    uint64_t hash = oicos_fnv1a(buf, (size_t)size);
    free(buf);
    return hash;
}

/* ===================================================================
 * Get file size
 * =================================================================== */

static uint64_t get_file_size(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fclose(f);

    return (size > 0) ? (uint64_t)size : 0;
}

/* ===================================================================
 * Main
 * =================================================================== */

int main(int argc, char *argv[])
{
    /* ----------------------------------------------------------
     * 1. Parse command line arguments
     * ---------------------------------------------------------- */

    const char *input_path = NULL;
    const char *output_path = NULL;
    const char *model_name = NULL;
    int verbose = 0;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: -o requires an argument\n", PROGRAM_NAME);
                return 1;
            }
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--name") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: --name requires an argument\n", PROGRAM_NAME);
                return 1;
            }
            model_name = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "%s: unknown option '%s'\n", PROGRAM_NAME, argv[i]);
            usage(argv[0]);
            return 1;
        } else {
            if (input_path != NULL) {
                fprintf(stderr, "%s: multiple input files specified\n", PROGRAM_NAME);
                return 1;
            }
            input_path = argv[i];
        }
    }

    if (!input_path) {
        fprintf(stderr, "%s: no input .snap file specified\n", PROGRAM_NAME);
        usage(argv[0]);
        return 1;
    }

    if (!output_path) {
        fprintf(stderr, "%s: no output file specified (use -o)\n", PROGRAM_NAME);
        usage(argv[0]);
        return 1;
    }

    /* ----------------------------------------------------------
     * 2. Build the encoding table from trine_encode
     * ---------------------------------------------------------- */

    uint8_t char_to_trits[128][5];
    build_char_to_trits(char_to_trits);

    /* Validate: all trit values must be in {0, 1, 2} */
    for (int c = 0; c < 128; c++) {
        for (int t = 0; t < 5; t++) {
            if (char_to_trits[c][t] > 2) {
                fprintf(stderr, "%s: encoding table error: char_to_trits[%d][%d] = %u "
                        "(must be 0, 1, or 2)\n",
                        PROGRAM_NAME, c, t, char_to_trits[c][t]);
                return 1;
            }
        }
    }

    /* ----------------------------------------------------------
     * 3. Convert composed .snap to .trine using trine_file_from_snap
     *
     * trine_file_from_snap handles:
     *   - Reading and validating the .snap file
     *   - Populating the header (magic, version, snap_count, etc.)
     *   - Copying all ROM tables from canonical OICOS constants
     *   - Setting up the layer map (13 layers with correct offsets)
     *   - Setting up resolution maps (screening, standard, deep)
     *   - Setting up I/O channel map (60 input + 41 output)
     *   - Computing snap array checksum
     * ---------------------------------------------------------- */

    fprintf(stderr, "%s: reading %s ...\n", PROGRAM_NAME, input_path);

    trine_file_t model;
    int rc = trine_file_from_snap(input_path, char_to_trits, &model);
    if (rc != 0) {
        fprintf(stderr, "%s: failed to package .snap file\n", PROGRAM_NAME);
        return 1;
    }

    fprintf(stderr, "%s: loaded %u snaps from %s\n",
            PROGRAM_NAME, model.snap_count, input_path);

    /* ----------------------------------------------------------
     * 4. Pre-write validation
     *
     * Validate the model before writing to catch any issues
     * in the layer map, resolution ranges, ROM tables, etc.
     * ---------------------------------------------------------- */

    fprintf(stderr, "%s: validating model ...\n", PROGRAM_NAME);

    rc = trine_file_validate(&model);
    if (rc != 0) {
        fprintf(stderr, "%s: model validation FAILED before write\n",
                PROGRAM_NAME);
        trine_file_free(&model);
        return 1;
    }

    fprintf(stderr, "%s: model valid\n", PROGRAM_NAME);

    /* ----------------------------------------------------------
     * 5. Write the .trine file
     *
     * trine_file_write handles:
     *   - Stamping final header values (magic, version, timestamp)
     *   - Computing FNV-1a checksums for every section
     *   - Writing all sections in order (header, ROM, encoding,
     *     resolution, layers, I/O map, padding, snaps, checksums)
     *   - Removing partial files on error
     * ---------------------------------------------------------- */

    fprintf(stderr, "%s: writing %s ...\n", PROGRAM_NAME, output_path);

    rc = trine_file_write(output_path, &model);
    if (rc != 0) {
        fprintf(stderr, "%s: failed to write .trine file\n", PROGRAM_NAME);
        trine_file_free(&model);
        return 1;
    }

    uint64_t file_size = get_file_size(output_path);
    fprintf(stderr, "%s: wrote %lu bytes to %s\n",
            PROGRAM_NAME, (unsigned long)file_size, output_path);

    /* ----------------------------------------------------------
     * 6. Read-back validation
     *
     * Read the written file back and validate it to confirm
     * the on-disk representation is correct and consistent.
     * This catches serialization bugs and disk errors.
     * ---------------------------------------------------------- */

    fprintf(stderr, "%s: validating written file ...\n", PROGRAM_NAME);

    trine_file_t readback;
    rc = trine_file_read(output_path, &readback);
    if (rc != 0) {
        fprintf(stderr, "%s: read-back FAILED for %s\n",
                PROGRAM_NAME, output_path);
        trine_file_free(&model);
        return 1;
    }

    rc = trine_file_validate(&readback);
    if (rc != 0) {
        fprintf(stderr, "%s: read-back validation FAILED\n", PROGRAM_NAME);
        trine_file_free(&readback);
        trine_file_free(&model);
        return 1;
    }

    /* Verify snap arrays match between original and read-back */
    if (readback.snap_count != model.snap_count) {
        fprintf(stderr, "%s: snap count mismatch after read-back "
                "(wrote %u, read %u)\n",
                PROGRAM_NAME, model.snap_count, readback.snap_count);
        trine_file_free(&readback);
        trine_file_free(&model);
        return 1;
    }

    if (memcmp(readback.snaps, model.snaps,
               (size_t)model.snap_count * 32) != 0) {
        fprintf(stderr, "%s: snap array mismatch after read-back\n",
                PROGRAM_NAME);
        trine_file_free(&readback);
        trine_file_free(&model);
        return 1;
    }

    /* Verify encoding tables match */
    if (memcmp(readback.char_to_trits, model.char_to_trits,
               128 * 5) != 0) {
        fprintf(stderr, "%s: encoding table mismatch after read-back\n",
                PROGRAM_NAME);
        trine_file_free(&readback);
        trine_file_free(&model);
        return 1;
    }

    /* Verify ROM tables match */
    if (memcmp(readback.endo, model.endo, sizeof(model.endo)) != 0 ||
        memcmp(readback.endo_rank, model.endo_rank,
               sizeof(model.endo_rank)) != 0 ||
        memcmp(readback.oic_route, model.oic_route,
               sizeof(model.oic_route)) != 0) {
        fprintf(stderr, "%s: ROM table mismatch after read-back\n",
                PROGRAM_NAME);
        trine_file_free(&readback);
        trine_file_free(&model);
        return 1;
    }

    /* Store the verified checksums from the written file */
    memcpy(model.section_checksums, readback.section_checksums,
           sizeof(model.section_checksums));

    /* Update the header to match what was actually written
     * (trine_file_write stamps created_epoch and snap_checksum) */
    model.header = readback.header;

    fprintf(stderr, "%s: read-back validation passed\n", PROGRAM_NAME);

    trine_file_free(&readback);

    /* ----------------------------------------------------------
     * 7. Compute golden hash of the complete file
     * ---------------------------------------------------------- */

    uint64_t golden_hash = compute_file_hash(output_path);

    /* ----------------------------------------------------------
     * 8. Print summary if verbose
     * ---------------------------------------------------------- */

    if (verbose) {
        print_summary(output_path, model_name, &model, file_size);
        printf("  Golden hash:    0x%016lx\n", (unsigned long)golden_hash);
        printf("\n");
    }

    /* ----------------------------------------------------------
     * 9. Done
     * ---------------------------------------------------------- */

    fprintf(stderr, "%s: SUCCESS — %s (%lu bytes, %u snaps)\n",
            PROGRAM_NAME, output_path,
            (unsigned long)file_size, model.snap_count);

    trine_file_free(&model);
    return 0;
}
