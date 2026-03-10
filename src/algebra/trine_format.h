/* ===================================================================
 * TRINE FORMAT — Ternary Resonance Interference Network Embedding
 * Self-describing binary format for text embedding models
 * ===================================================================
 *
 * A .trine file contains EVERYTHING needed to compute text embeddings:
 *   - The complete 270-byte OICOS ROM (endomorphism tables)
 *   - The snap topology (17,935 snaps x 32 bytes)
 *   - Character encoding tables (4 views, balanced ternary)
 *   - Resolution maps (3 tiers: screening, standard, deep)
 *   - Layer map (13 layers with offsets and counts)
 *   - I/O channel map (101 channels: 60 input, 41 output)
 *   - Per-section FNV-1a checksums
 *
 * No external dependencies. No configuration files. No model weights
 * to download. The file IS the model.
 *
 * DESIGN PRINCIPLES
 *   1. Self-describing: a reader determines everything from the file
 *   2. Memory-mappable: snap array is contiguous for mmap efficiency
 *   3. Checksummed: corrupt files are detected (FNV-1a per section)
 *   4. Versioned: format version in header enables future evolution
 *   5. Compatible: snap_t records are IDENTICAL to .snap format
 *      (32 bytes each, directly memcpy-able)
 *
 * FILE LAYOUT
 *   Offset  Section                          Size (bytes)
 *   ------  -------                          -----------
 *   0       trine_header_t                   64
 *   64      ROM: ENDO[27][3]                 81
 *   145     ROM: ENDO_RANK[27]               27
 *   172     ROM: OIC_ROUTE[3][2][7]          42
 *   214     ROM: DTRIT[6][6]                 36
 *   250     ROM: ALU_S3[6][6]                36
 *   286     ROM: ALU_CHIRAL[3][3]            9
 *   295     ROM: VALID[3][4]                 12
 *   307     ROM: CELL_RANK[7]                7
 *   314     ROM: S3_TO_ENDO[6]               6
 *   320     Encoding: char_to_trits[128][5]  640
 *   960     Resolution maps: 3 x 40          120
 *   1080    Layer map: 13 x 16               208
 *   1288    I/O channel map: 101 x 4         404
 *   1692    Padding to 32-byte alignment     4
 *   1696    Snap array: N x 32               N*32
 *   1696+N*32  Section checksums: 8 x 8      64
 *
 * Total: 1760 + N*32 bytes (N=17935 -> 575,680 bytes, ~562 KB)
 *
 * The snap array begins at a 32-byte aligned offset so that mmap
 * yields properly aligned snap_t pointers without copying.
 *
 * =================================================================== */

#ifndef TRINE_FORMAT_H
#define TRINE_FORMAT_H

#include <stdint.h>
#include <stddef.h>
#include "oicos.h"

/* =================================================================
 * Constants
 * ================================================================= */

#define TRINE_MAGIC       0x54524E45u   /* "TRNE" in ASCII (LE)      */
#define TRINE_VERSION     1u

/* Section count for checksums */
#define TRINE_NUM_SECTIONS 8

/* Section indices (for checksum array) */
#define TRINE_SEC_HEADER    0   /* Header (excluding checksum fields)  */
#define TRINE_SEC_ROM       1   /* All ROM tables                      */
#define TRINE_SEC_ENCODING  2   /* Character encoding tables           */
#define TRINE_SEC_RESOLUTION 3  /* Resolution maps                     */
#define TRINE_SEC_LAYERS    4   /* Layer map                           */
#define TRINE_SEC_IOMAP     5   /* I/O channel map                     */
#define TRINE_SEC_SNAPS     6   /* Snap array                          */
#define TRINE_SEC_GLOBAL    7   /* Global checksum of sections 0-6     */

/* Layer count */
#define TRINE_NUM_LAYERS    13

/* Layer indices */
#define TRINE_LAYER_KERN    0
#define TRINE_LAYER_ING     1
#define TRINE_LAYER_SCAT    2
#define TRINE_LAYER_WAVE    3
#define TRINE_LAYER_BANK    4
#define TRINE_LAYER_COND    5
#define TRINE_LAYER_PHASE   6
#define TRINE_LAYER_WEAVE   7
#define TRINE_LAYER_READ    8
#define TRINE_LAYER_MIRROR  9
#define TRINE_LAYER_RECON   10
#define TRINE_LAYER_VERIFY  11
#define TRINE_LAYER_EMIT    12

/* Resolution tier count */
#define TRINE_NUM_RESOLUTIONS 3

/* Resolution tier indices */
#define TRINE_RES_SCREENING 0   /* 68 dims: INGRESS collectors + Tier A  */
#define TRINE_RES_STANDARD  1   /* 1,053 dims: + Tier B + BANK summaries */
#define TRINE_RES_DEEP      2   /* 17,410 dims: + Tier C + all active    */
#define TRINE_RES_SHINGLE   3   /* 240 dims: direct channel encoding     */

/* I/O channel limits */
#define TRINE_NUM_IO_CHANNELS 101
#define TRINE_NUM_INPUT_CHANNELS  60   /* Channels 0-59: text input       */
#define TRINE_NUM_OUTPUT_CHANNELS 41   /* Channels 60-100: embedding out  */

/* Encoding view count */
#define TRINE_NUM_VIEWS     4

/* Encoding view indices */
#define TRINE_VIEW_FORWARD      0
#define TRINE_VIEW_REVERSE      1
#define TRINE_VIEW_DIFFERENTIAL 2
#define TRINE_VIEW_PHONETIC     3

/* File layout offsets (byte positions) */
#define TRINE_OFF_HEADER     0
#define TRINE_OFF_ROM        64
#define TRINE_OFF_ENCODING   320
#define TRINE_OFF_RESOLUTION 960
#define TRINE_OFF_LAYERS     1080
#define TRINE_OFF_IOMAP      1288
#define TRINE_OFF_PAD        1692
#define TRINE_OFF_SNAPS      1696

/* ROM sub-offsets (relative to TRINE_OFF_ROM) */
#define TRINE_ROM_ENDO       0     /* 81 bytes  */
#define TRINE_ROM_ENDO_RANK  81    /* 27 bytes  */
#define TRINE_ROM_OIC_ROUTE  108   /* 42 bytes  */
#define TRINE_ROM_DTRIT      150   /* 36 bytes  */
#define TRINE_ROM_ALU_S3     186   /* 36 bytes  */
#define TRINE_ROM_ALU_CHIRAL 222   /* 9 bytes   */
#define TRINE_ROM_VALID      231   /* 12 bytes  */
#define TRINE_ROM_CELL_RANK  243   /* 7 bytes   */
#define TRINE_ROM_S3_TO_ENDO 250   /* 6 bytes   */
#define TRINE_ROM_TOTAL      256   /* Total ROM section size */

/* =================================================================
 * Structures
 * ================================================================= */

/*
 * File header (64 bytes, at offset 0).
 *
 * Contains magic, version, global metadata, and snap array parameters.
 * The header is designed so that all fields are naturally aligned.
 */
typedef struct {
    uint32_t magic;             /* TRINE_MAGIC (0x54524E45)            */
    uint32_t version;           /* TRINE_VERSION (1)                   */
    uint32_t snap_count;        /* Number of snaps in the snap array   */
    uint32_t snap_offset;       /* Byte offset of snap array in file   */
    uint32_t root_index;        /* Root snap index (boot wave seed)    */
    uint32_t free_head;         /* First snap in free list (SNAP_NIL)  */
    uint32_t rom_size;          /* ROM section size in bytes (256)     */
    uint32_t encoding_size;     /* Encoding section size (640)         */
    uint32_t resolution_count;  /* Number of resolution tiers (3)      */
    uint32_t layer_count;       /* Number of layers (13)               */
    uint32_t io_channel_count;  /* Number of I/O channels (101)        */
    uint32_t flags;             /* Reserved flags (0 for now)          */
    uint64_t snap_checksum;     /* FNV-1a of the entire snap array     */
    uint64_t created_epoch;     /* Unix timestamp of creation          */
} __attribute__((packed)) trine_header_t;

_Static_assert(sizeof(trine_header_t) == 64,
               "trine_header_t must be exactly 64 bytes");

/*
 * Resolution tier descriptor (40 bytes).
 *
 * Defines which snap indices contribute to each resolution tier.
 * The embedding dimension count equals the number of active snaps
 * whose final FSM state is read after cascade completion.
 */
typedef struct {
    uint32_t dim_count;         /* Number of embedding dimensions      */
    uint32_t range_count;       /* Number of index ranges (max 8)      */
    /* Up to 4 index ranges [start, end) that form this tier.
     * Ranges are inclusive-start, exclusive-end.
     * Unused ranges have start == end == 0. */
    uint32_t ranges[4][2];      /* [i][0]=start, [i][1]=end            */
} __attribute__((packed)) trine_resolution_t;

_Static_assert(sizeof(trine_resolution_t) == 40,
               "trine_resolution_t must be exactly 40 bytes");

/*
 * Layer descriptor (16 bytes).
 *
 * Maps each of the 13 processing layers to its contiguous span
 * within the composed snap array.
 */
typedef struct {
    uint32_t start_index;       /* First snap index in this layer      */
    uint32_t snap_count;        /* Number of snaps in this layer       */
    uint32_t flags;             /* Layer-specific flags (reserved)     */
    uint16_t domain;            /* Primary domain of this layer        */
    uint16_t reserved;          /* Padding for alignment               */
} __attribute__((packed)) trine_layer_info_t;

_Static_assert(sizeof(trine_layer_info_t) == 16,
               "trine_layer_info_t must be exactly 16 bytes");

/*
 * I/O channel descriptor (4 bytes).
 *
 * Maps each I/O channel to its role (input or output) and the
 * snap index that services it.
 */
typedef struct {
    uint16_t snap_index;        /* Snap index servicing this channel   */
    uint8_t  direction;         /* 0 = input, 1 = output               */
    uint8_t  channel_id;        /* Channel number (0-100)              */
} __attribute__((packed)) trine_io_channel_t;

_Static_assert(sizeof(trine_io_channel_t) == 4,
               "trine_io_channel_t must be exactly 4 bytes");

/*
 * In-memory representation of a loaded .trine file.
 *
 * This struct is populated by trine_file_read() and freed by
 * trine_file_free(). It owns all allocated memory.
 *
 * The snap array is heap-allocated (aligned to 32 bytes) and can
 * be used directly as an oicos_t arena.
 */
typedef struct {
    /* Header */
    trine_header_t header;

    /* ROM tables (embedded, not pointers — identical to oicos.h) */
    uint8_t endo[27][3];            /* Endomorphism transition matrix  */
    uint8_t endo_rank[27];          /* Endomorphism rank LUT           */
    uint8_t oic_route[3][2][7];     /* OIC routing matrix              */
    uint8_t dtrit[6][6];            /* Domain input trit pattern       */
    uint8_t alu_s3[6][6];           /* S3 Cayley composition table     */
    uint8_t alu_chiral[3][3];       /* Chirality composition           */
    uint8_t valid[3][4];            /* Quark validity matrix           */
    uint8_t cell_rank[7];           /* Cell -> natural rank            */
    uint8_t s3_to_endo[6];          /* Dense S3 index -> endo index    */

    /* Encoding tables: ASCII -> 5 balanced trits per character.
     * char_to_trits[c][t] = trit value (0, 1, or 2) for character c,
     * trit position t. Designed for maximal Hamming distance between
     * visually/semantically similar characters. */
    uint8_t char_to_trits[128][5];

    /* Snap topology (heap-allocated, 32-byte aligned) */
    uint32_t snap_count;
    snap_t  *snaps;                 /* Owned; freed by trine_file_free */

    /* Resolution tier maps */
    trine_resolution_t resolutions[TRINE_NUM_RESOLUTIONS];

    /* Layer map */
    trine_layer_info_t layers[TRINE_NUM_LAYERS];

    /* I/O channel map */
    trine_io_channel_t io_channels[TRINE_NUM_IO_CHANNELS];

    /* Per-section checksums (FNV-1a) */
    uint64_t section_checksums[TRINE_NUM_SECTIONS];
} trine_file_t;

/* =================================================================
 * Layer name table
 * ================================================================= */

static const char * const TRINE_LAYER_NAME[TRINE_NUM_LAYERS] = {
    "kern", "ing", "scat", "wave", "bank", "cond", "phase",
    "weave", "read", "mirror", "recon", "verify", "emit"
};

static const char * const TRINE_RES_NAME[TRINE_NUM_RESOLUTIONS] = {
    "screening", "standard", "deep"
};

/* =================================================================
 * API
 * ================================================================= */

/*
 * trine_file_read — Load a .trine file into memory.
 *
 * Reads the file at `path`, validates magic/version/checksums, and
 * populates `out` with the complete model. The snap array is heap-
 * allocated with 32-byte alignment.
 *
 * Returns 0 on success, nonzero on error (message printed to stderr).
 * On error, `out` is zeroed and no memory is leaked.
 */
int trine_file_read(const char *path, trine_file_t *out);

/*
 * trine_file_write — Write a .trine file to disk.
 *
 * Serializes the complete model in `model` to the file at `path`.
 * Computes and embeds FNV-1a checksums for every section.
 *
 * Returns 0 on success, nonzero on error (message printed to stderr).
 */
int trine_file_write(const char *path, const trine_file_t *model);

/*
 * trine_file_free — Release memory owned by a trine_file_t.
 *
 * Frees the heap-allocated snap array and zeroes the struct.
 * Safe to call on a zeroed or already-freed struct (no-op).
 */
void trine_file_free(trine_file_t *model);

/*
 * trine_file_validate — Validate in-memory model integrity.
 *
 * Checks that ROM tables match the canonical OICOS constants,
 * layer boundaries are consistent, resolution ranges are within
 * bounds, and all snap indices are valid.
 *
 * Returns 0 if valid, nonzero on first detected inconsistency
 * (message printed to stderr).
 */
int trine_file_validate(const trine_file_t *model);

/*
 * trine_file_from_snap — Package a composed .snap file into a .trine.
 *
 * Reads the composed snap arena from `snap_path`, populates the ROM
 * from the canonical oicos.h constants, sets up the default HTEB
 * layer map and resolution tiers, and initializes encoding tables.
 *
 * The caller must have already populated `model->char_to_trits`
 * before calling this, OR pass NULL for `encoding` to use zeroed
 * encoding tables (to be filled later).
 *
 * Returns 0 on success, nonzero on error.
 */
int trine_file_from_snap(const char *snap_path,
                         const uint8_t encoding[128][5],
                         trine_file_t *model);

#endif /* TRINE_FORMAT_H */
