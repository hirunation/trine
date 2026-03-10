/* =====================================================================
 * TRINE Stage-2 -- Block-Diagonal Persistence Tests (8 categories, 30 checks)
 * =====================================================================
 *
 * Tests for block-diagonal model (.trine2) and accumulator (.trine2a)
 * save/load round-tripping, file size validation, format rejection,
 * corruption detection, and stats preservation.
 *
 * Category 1: Block model save/load round-trip (4 checks)
 * Category 2: Block model comparison consistency (2 checks)
 * Category 3: Block model file size (2 checks)
 * Category 4: Block accumulator save/load (4 checks)
 * Category 5: Block accumulator stats preservation (4 checks)
 * Category 6: Format rejection / cross-load (4 checks)
 * Category 7: Validate block model file (3 checks)
 * Category 8: Corrupt file detection (7 checks)
 *
 * ===================================================================== */

#include "trine_s2_persist.h"
#include "trine_accumulator_persist.h"
#include "trine_stage2.h"
#include "trine_project.h"
#include "trine_encode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Test framework ─────────────────────────────────────────────────── */

static int g_passed = 0;
static int g_failed = 0;
static int g_total  = 0;

static void check(const char *name, int cond)
{
    g_total++;
    if (cond) {
        g_passed++;
    } else {
        g_failed++;
        printf("  FAIL  block_persist: %s\n", name);
    }
}

/* Temp file paths */
static const char *TMP_MODEL     = "/tmp/trine_test_block_model.trine2";
static const char *TMP_MODEL2    = "/tmp/trine_test_block_model2.trine2";
static const char *TMP_ACC       = "/tmp/trine_test_block_acc.trine2a";
static const char *TMP_ACC2      = "/tmp/trine_test_block_acc2.trine2a";
static const char *TMP_FULL_ACC  = "/tmp/trine_test_full_acc.trine2a";

static void cleanup(void)
{
    (void)remove(TMP_MODEL);
    (void)remove(TMP_MODEL2);
    (void)remove(TMP_ACC);
    (void)remove(TMP_ACC2);
    (void)remove(TMP_FULL_ACC);
}

/* ── Helpers ────────────────────────────────────────────────────────── */

/* Block weight size for K copies: K * 4 * 60 * 60 bytes */
#define BLOCK_WEIGHT_BYTES(K) ((size_t)(K) * TRINE_S2_N_CHAINS * TRINE_S2_CHAIN_DIM * TRINE_S2_CHAIN_DIM)

/* Expected block model file size: header(72) + weights(43200) + checksum(8) = 43280 */
#define EXPECTED_BLOCK_FILE_SIZE  (72 + (3 * 4 * 60 * 60) + 8)

/* Expected full model file size: header(72) + weights(172800) + checksum(8) = 172880 */
#define EXPECTED_FULL_FILE_SIZE   (72 + (3 * 240 * 240) + 8)

/* Create a block-diagonal model from random block weights */
static trine_s2_model_t *make_block_model(uint64_t seed)
{
    size_t bsz = BLOCK_WEIGHT_BYTES(TRINE_PROJECT_K);
    uint8_t *W = (uint8_t *)malloc(bsz);
    if (!W) return NULL;

    trine_projection_block_random(W, TRINE_PROJECT_K, seed);
    trine_s2_model_t *m = trine_s2_create_block_diagonal(W, TRINE_PROJECT_K, 0, 0);
    free(W);
    return m;
}

/* Get file size in bytes */
static long file_size(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fclose(fp);
    return sz;
}

/* =====================================================================
 * Category 1: Save/Load Block Model Round-Trip (4 checks)
 * ===================================================================== */

static void test_block_model_roundtrip(void)
{
    /* Create a block-diagonal model */
    trine_s2_model_t *orig = make_block_model(42);
    check("create block model", orig != NULL);
    if (!orig) return;

    /* Verify mode */
    int mode = trine_s2_get_projection_mode(orig);
    check("block model mode is BLOCK_DIAG", mode == TRINE_S2_PROJ_BLOCK_DIAG);

    /* Save */
    int rc = trine_s2_save(orig, TMP_MODEL, NULL);
    check("save block model", rc == 0);

    /* Load */
    trine_s2_model_t *loaded = trine_s2_load(TMP_MODEL);
    check("load block model", loaded != NULL);

    if (loaded) {
        /* Verify the loaded model also is block-diagonal */
        int loaded_mode = trine_s2_get_projection_mode(loaded);
        check("loaded mode is BLOCK_DIAG", loaded_mode == TRINE_S2_PROJ_BLOCK_DIAG);

        /* Get block weights from both and compare */
        const uint8_t *w_orig   = trine_s2_get_block_projection(orig);
        const uint8_t *w_loaded = trine_s2_get_block_projection(loaded);
        check("block weights not NULL (orig)", w_orig != NULL);
        check("block weights not NULL (loaded)", w_loaded != NULL);

        if (w_orig && w_loaded) {
            size_t bsz = BLOCK_WEIGHT_BYTES(TRINE_PROJECT_K);
            int weights_match = (memcmp(w_orig, w_loaded, bsz) == 0);
            check("block weights match after roundtrip", weights_match);
        }

        trine_s2_free(loaded);
    }

    trine_s2_free(orig);
}

/* =====================================================================
 * Category 2: Block Model Comparison Consistency (2 checks)
 * ===================================================================== */

static void test_block_model_comparison_consistency(void)
{
    /* Create, save, load */
    trine_s2_model_t *orig = make_block_model(123);
    if (!orig) { check("comparison_orig_alloc", 0); return; }

    trine_s2_save(orig, TMP_MODEL, NULL);
    trine_s2_model_t *loaded = trine_s2_load(TMP_MODEL);
    if (!loaded) { check("comparison_loaded_alloc", 0); trine_s2_free(orig); return; }

    /* Encode same text with both models */
    const char *texts[] = {
        "Hello, world!",
        "The quick brown fox",
        "TRINE encoding test",
        "block diagonal persistence check"
    };

    int encode_match = 1;
    for (int i = 0; i < 4; i++) {
        uint8_t out_a[240], out_b[240];
        size_t len = strlen(texts[i]);

        int ra = trine_s2_encode(orig, texts[i], len, 0, out_a);
        int rb = trine_s2_encode(loaded, texts[i], len, 0, out_b);

        if (ra != rb || (ra == 0 && memcmp(out_a, out_b, 240) != 0)) {
            encode_match = 0;
            break;
        }
    }
    check("encode outputs match after load", encode_match);

    /* Compare pairs: results should match between orig and loaded */
    uint8_t e1_orig[240], e2_orig[240], e1_loaded[240], e2_loaded[240];
    trine_s2_encode(orig,   "alpha bravo charlie", 19, 0, e1_orig);
    trine_s2_encode(orig,   "alpha bravo delta",   17, 0, e2_orig);
    trine_s2_encode(loaded, "alpha bravo charlie", 19, 0, e1_loaded);
    trine_s2_encode(loaded, "alpha bravo delta",   17, 0, e2_loaded);

    float sim_orig   = trine_s2_compare(e1_orig, e2_orig, NULL);
    float sim_loaded = trine_s2_compare(e1_loaded, e2_loaded, NULL);

    check("compare results match", fabsf(sim_orig - sim_loaded) < 1e-6f);

    trine_s2_free(orig);
    trine_s2_free(loaded);
}

/* =====================================================================
 * Category 3: Block Model File Size (2 checks)
 * ===================================================================== */

static void test_block_model_file_size(void)
{
    /* Block model should produce 43,280 byte file */
    trine_s2_model_t *block = make_block_model(77);
    if (!block) { check("file_size_block_alloc", 0); return; }
    trine_s2_save(block, TMP_MODEL, NULL);
    trine_s2_free(block);

    long bsz = file_size(TMP_MODEL);
    check("block model file is 43280 bytes",
          bsz == (long)EXPECTED_BLOCK_FILE_SIZE);

    /* Full model should produce 172,880 byte file */
    trine_s2_model_t *full = trine_s2_create_random(0, 99);
    if (!full) { check("file_size_full_alloc", 0); return; }
    trine_s2_save(full, TMP_MODEL2, NULL);
    trine_s2_free(full);

    long fsz = file_size(TMP_MODEL2);
    check("full model file is 172880 bytes",
          fsz == (long)EXPECTED_FULL_FILE_SIZE);
}

/* =====================================================================
 * Category 4: Block Accumulator Save/Load (4 checks)
 * ===================================================================== */

static void test_block_accumulator_roundtrip(void)
{
    /* Create accumulator and populate it */
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    check("accum create", acc != NULL);
    if (!acc) return;

    /* Feed some training pairs */
    uint8_t a[240], b[240];
    trine_encode_shingle("block accumulator save test", 27, a);
    trine_encode_shingle("block accumulator load test", 27, b);

    for (int i = 0; i < 50; i++)
        trine_block_accumulator_update(acc, a, b, (i % 3 != 0));

    /* Save */
    int rc = trine_block_accumulator_save(
        acc, 0.5f, 0.33f, 0, TMP_ACC);
    check("accum save", rc == 0);

    /* Load */
    float thresh_out = 0.0f, density_out = 0.0f;
    uint32_t pairs_out = 0;
    trine_block_accumulator_t *loaded = trine_block_accumulator_load(
        TMP_ACC, &thresh_out, &density_out, &pairs_out);
    check("accum load", loaded != NULL);

    if (loaded) {
        /* Verify counter data matches */
        size_t counter_bytes = (size_t)TRINE_PROJECT_K * TRINE_BLOCK_CHAINS
                               * TRINE_BLOCK_DIM * TRINE_BLOCK_DIM
                               * sizeof(int32_t);
        int counters_match = (memcmp(acc->counters, loaded->counters,
                                      counter_bytes) == 0);
        check("accum counters match after roundtrip", counters_match);

        trine_block_accumulator_free(loaded);
    }

    trine_block_accumulator_free(acc);
}

/* =====================================================================
 * Category 5: Block Accumulator Stats Preservation (4 checks)
 * ===================================================================== */

static void test_block_accumulator_stats_preserved(void)
{
    /* Create and populate */
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("stats_alloc", 0); return; }

    uint8_t a[240], b[240], c_enc[240];
    trine_encode_shingle("stats preservation alpha", 24, a);
    trine_encode_shingle("stats preservation beta",  23, b);
    trine_encode_shingle("stats preservation gamma", 25, c_enc);

    for (int i = 0; i < 30; i++) {
        trine_block_accumulator_update(acc, a, b, 1);
        trine_block_accumulator_update(acc, a, c_enc, 0);
    }

    /* Get original stats */
    int32_t orig_max = 0, orig_min = 0;
    uint64_t orig_nonzero = 0;
    trine_block_accumulator_stats(acc, &orig_max, &orig_min, &orig_nonzero);

    /* Save and reload */
    trine_block_accumulator_save(acc, 0.7f, 0.25f, 0, TMP_ACC);

    float thresh_out = 0.0f, density_out = 0.0f;
    uint32_t pairs_out = 0;
    trine_block_accumulator_t *loaded = trine_block_accumulator_load(
        TMP_ACC, &thresh_out, &density_out, &pairs_out);
    if (!loaded) {
        check("stats_load_fail", 0);
        trine_block_accumulator_free(acc);
        return;
    }

    /* Get loaded stats */
    int32_t loaded_max = 0, loaded_min = 0;
    uint64_t loaded_nonzero = 0;
    trine_block_accumulator_stats(loaded, &loaded_max, &loaded_min, &loaded_nonzero);

    check("stats max preserved", orig_max == loaded_max);
    check("stats min preserved", orig_min == loaded_min);
    check("stats nonzero preserved", orig_nonzero == loaded_nonzero);

    /* Verify metadata restored */
    check("threshold preserved", fabsf(thresh_out - 0.7f) < 1e-6f);

    trine_block_accumulator_free(acc);
    trine_block_accumulator_free(loaded);
}

/* =====================================================================
 * Category 6: Format Rejection (4 checks)
 * ===================================================================== */

static void test_format_rejection(void)
{
    /* 1. Create a block-diagonal accumulator file, try loading with
     *    trine_accumulator_load() -- should fail */
    trine_block_accumulator_t *bacc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!bacc) { check("reject_alloc", 0); return; }

    uint8_t v[240];
    trine_encode_shingle("format rejection test", 21, v);
    trine_block_accumulator_update(bacc, v, v, 1);

    trine_block_accumulator_save(bacc, 0.5f, 0.33f, 0, TMP_ACC);
    trine_block_accumulator_free(bacc);

    /* Try loading block-diag file with full accumulator loader */
    trine_accumulator_t *full_from_block = trine_accumulator_load(
        TMP_ACC, NULL, NULL);
    check("accumulator_load rejects block-diag file", full_from_block == NULL);
    if (full_from_block) trine_accumulator_free(full_from_block);

    /* 2. Create a full (non-block) accumulator file, try loading with
     *    trine_block_accumulator_load() -- should fail */
    trine_accumulator_t *facc = trine_accumulator_create();
    if (!facc) { check("reject_full_alloc", 0); return; }

    trine_accumulator_update(facc, v, v, 1);

    trine_hebbian_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.similarity_threshold  = 0.5f;
    cfg.freeze_target_density = 0.33f;
    cfg.projection_mode = 0;

    int rc = trine_accumulator_save(facc, &cfg, 10, TMP_FULL_ACC);
    check("save full accumulator for rejection test", rc == 0);
    trine_accumulator_free(facc);

    float th = 0.0f, dens = 0.0f;
    uint32_t pairs = 0;
    trine_block_accumulator_t *block_from_full = trine_block_accumulator_load(
        TMP_FULL_ACC, &th, &dens, &pairs);
    check("block_accumulator_load rejects full file", block_from_full == NULL);
    if (block_from_full) trine_block_accumulator_free(block_from_full);

    /* 3. Model loader rejects accumulator file (wrong magic) */
    trine_s2_model_t *model_from_acc = trine_s2_load(TMP_ACC);
    check("s2_load rejects accumulator file", model_from_acc == NULL);
    if (model_from_acc) trine_s2_free(model_from_acc);
}

/* =====================================================================
 * Category 7: Validate Block Model File (3 checks)
 * ===================================================================== */

static void test_validate_block_model(void)
{
    /* Create and save a block model */
    trine_s2_model_t *m = make_block_model(555);
    if (!m) { check("validate_alloc", 0); return; }

    trine_s2_save(m, TMP_MODEL, NULL);
    trine_s2_free(m);

    /* Validate should succeed */
    int rc = trine_s2_validate(TMP_MODEL);
    check("validate accepts valid block model file", rc == 0);

    /* Also validate the accumulator file format */
    trine_block_accumulator_t *acc = trine_block_accumulator_create(TRINE_PROJECT_K);
    if (!acc) { check("validate_acc_alloc", 0); return; }

    uint8_t v[240];
    trine_encode_shingle("validate block accum", 20, v);
    trine_block_accumulator_update(acc, v, v, 1);
    trine_block_accumulator_save(acc, 0.5f, 0.33f, 0, TMP_ACC);
    trine_block_accumulator_free(acc);

    int rc2 = trine_accumulator_validate(TMP_ACC);
    check("validate accepts valid block accum file", rc2 == 0);

    /* Validate non-existent file should fail */
    int rc3 = trine_s2_validate("/tmp/trine_nonexistent_block.trine2");
    check("validate rejects nonexistent file", rc3 != 0);
}

/* =====================================================================
 * Category 8: Corrupt File Detection (7 checks)
 * ===================================================================== */

/* Helper: read an entire file into a malloc'd buffer.
 * Returns buffer and sets *out_size.  Returns NULL on error. */
static uint8_t *read_file(const char *path, long *out_size)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    rewind(fp);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf) { fclose(fp); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if ((long)n != sz) { free(buf); return NULL; }
    *out_size = sz;
    return buf;
}

/* Helper: write buffer to file */
static int write_file(const char *path, const uint8_t *buf, size_t len)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    size_t n = fwrite(buf, 1, len, fp);
    fclose(fp);
    return (n == len) ? 0 : -1;
}

static void test_corrupt_block_model(void)
{
    /* Save a valid block model */
    trine_s2_model_t *m = make_block_model(999);
    if (!m) { check("corrupt_alloc", 0); return; }
    trine_s2_save(m, TMP_MODEL, NULL);
    trine_s2_free(m);

    /* Verify it's valid first */
    check("valid before corruption", trine_s2_validate(TMP_MODEL) == 0);

    /* Corrupt magic byte (offset 0) */
    {
        long sz = 0;
        uint8_t *buf = read_file(TMP_MODEL, &sz);
        if (!buf) { check("corrupt_magic_read", 0); return; }

        buf[0] ^= 0xFF;  /* flip magic */
        write_file(TMP_MODEL2, buf, (size_t)sz);
        free(buf);

        int rc = trine_s2_validate(TMP_MODEL2);
        check("corrupt magic detected (validate)", rc != 0);

        trine_s2_model_t *bad = trine_s2_load(TMP_MODEL2);
        check("corrupt magic detected (load)", bad == NULL);
        if (bad) trine_s2_free(bad);
    }

    /* Corrupt a weight byte in the payload (offset 72 + 500) */
    {
        long sz = 0;
        uint8_t *buf = read_file(TMP_MODEL, &sz);
        if (!buf) { check("corrupt_payload_read", 0); return; }

        buf[72 + 500] ^= 0x3F;
        write_file(TMP_MODEL2, buf, (size_t)sz);
        free(buf);

        int rc = trine_s2_validate(TMP_MODEL2);
        check("corrupt payload detected (validate)", rc != 0);

        trine_s2_model_t *bad = trine_s2_load(TMP_MODEL2);
        check("corrupt payload detected (load)", bad == NULL);
        if (bad) trine_s2_free(bad);
    }

    /* Corrupt the header flags field (offset 8) */
    {
        long sz = 0;
        uint8_t *buf = read_file(TMP_MODEL, &sz);
        if (!buf) { check("corrupt_flags_read", 0); return; }

        buf[8] ^= 0xFF;  /* flip flags */
        write_file(TMP_MODEL2, buf, (size_t)sz);
        free(buf);

        int rc = trine_s2_validate(TMP_MODEL2);
        check("corrupt header flags detected", rc != 0);
    }

    /* Truncated file */
    {
        long sz = 0;
        uint8_t *buf = read_file(TMP_MODEL, &sz);
        if (!buf) { check("corrupt_trunc_read", 0); return; }

        write_file(TMP_MODEL2, buf, (size_t)(sz / 2));
        free(buf);

        trine_s2_model_t *bad = trine_s2_load(TMP_MODEL2);
        check("truncated file load fails", bad == NULL);
        if (bad) trine_s2_free(bad);
    }
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Block-Diagonal Persistence Tests ===\n");

    cleanup();

    /* Category 1: Save/load block model round-trip */
    printf("\n--- Block Model Round-Trip ---\n");
    test_block_model_roundtrip();

    /* Category 2: Block model comparison consistency */
    printf("\n--- Block Model Comparison Consistency ---\n");
    test_block_model_comparison_consistency();

    /* Category 3: Block model file size */
    printf("\n--- Block Model File Size ---\n");
    test_block_model_file_size();

    /* Category 4: Block accumulator save/load */
    printf("\n--- Block Accumulator Round-Trip ---\n");
    test_block_accumulator_roundtrip();

    /* Category 5: Block accumulator stats preservation */
    printf("\n--- Block Accumulator Stats Preservation ---\n");
    test_block_accumulator_stats_preserved();

    /* Category 6: Format rejection */
    printf("\n--- Format Rejection ---\n");
    test_format_rejection();

    /* Category 7: Validate block model file */
    printf("\n--- Validate Block Model ---\n");
    test_validate_block_model();

    /* Category 8: Corrupt file detection */
    printf("\n--- Corrupt File Detection ---\n");
    test_corrupt_block_model();

    cleanup();

    printf("\nBlock-diagonal persistence: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);

    return g_failed > 0 ? 1 : 0;
}
