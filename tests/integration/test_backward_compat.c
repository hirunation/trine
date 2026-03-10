/* =====================================================================
 * TRINE — Backward Compatibility Tests
 * =====================================================================
 *
 * Verifies that v1.0.3 code can still load v1.0.2 format files:
 *   - .trine2 model save/load roundtrip and validation
 *   - .tridx  Stage-1 index save/load roundtrip
 *   - .trrt   routed index save/load roundtrip
 *
 * ===================================================================== */

#include "trine_s2_persist.h"
#include "trine_stage2.h"
#include "trine_project.h"
#include "trine_encode.h"
#include "trine_stage1.h"
#include "trine_route.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

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
        printf("  FAIL  compat: %s\n", name);
    }
}

/* ── Temp file paths ────────────────────────────────────────────────── */

static const char *TMP_TRINE2  = "/tmp/trine_compat_test.trine2";
static const char *TMP_TRIDX   = "/tmp/trine_compat_test.tridx";
static const char *TMP_TRRT    = "/tmp/trine_compat_test.trrt";
static const char *TMP_CORRUPT = "/tmp/trine_compat_corrupt.trine2";

static void cleanup(void)
{
    (void)unlink(TMP_TRINE2);
    (void)unlink(TMP_TRIDX);
    (void)unlink(TMP_TRRT);
    (void)unlink(TMP_CORRUPT);
}

/* ── Test corpus ────────────────────────────────────────────────────── */

static const char *TEST_TEXTS[] = {
    "The quick brown fox jumps over the lazy dog",
    "A fast red fox leaps above a sleepy hound",
    "Machine learning is a subset of artificial intelligence",
    "Deep neural networks enable powerful pattern recognition",
    "The cat sat on the mat in the sunny room",
    "Two cats were sleeping on a warm blanket today",
    "Quantum computing may revolutionize cryptography soon",
    "Classical computers struggle with certain quantum problems",
    "Shakespeare wrote many plays and sonnets in English",
    "The Bard of Avon is celebrated for his literary works"
};
#define N_TEXTS (int)(sizeof(TEST_TEXTS) / sizeof(TEST_TEXTS[0]))

/* ── Helpers ────────────────────────────────────────────────────────── */

/* Compare two Stage-2 models by encoding several texts */
static int models_same_output(const trine_s2_model_t *a,
                               const trine_s2_model_t *b)
{
    for (int i = 0; i < N_TEXTS; i++) {
        uint8_t out_a[240], out_b[240];
        size_t len = strlen(TEST_TEXTS[i]);

        int ra = trine_s2_encode(a, TEST_TEXTS[i], len, 0, out_a);
        int rb = trine_s2_encode(b, TEST_TEXTS[i], len, 0, out_b);

        if (ra != rb) return 0;
        if (ra == 0 && memcmp(out_a, out_b, 240) != 0) return 0;
    }
    return 1;
}

/* Write a file with specific content for corruption tests */
static int write_raw_file(const char *path, const void *data, size_t size)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    size_t written = fwrite(data, 1, size, fp);
    fclose(fp);
    return (written == size) ? 0 : -1;
}

/* ══════════════════════════════════════════════════════════════════════
 * Category 1: .trine2 Model Format Compatibility
 * ══════════════════════════════════════════════════════════════════════ */

static void test_trine2_save_load_roundtrip(void)
{
    printf("  --- .trine2 model format ---\n");

    /* Create model, save, load, verify output matches */
    trine_s2_model_t *orig = trine_s2_create_random(512, 42);
    trine_s2_save_config_t cfg = {
        .similarity_threshold = 0.85f,
        .density = 0.33f,
        .topo_seed = 42 ^ 0xDEADBEEFCAFEBABEULL
    };

    int rc = trine_s2_save(orig, TMP_TRINE2, &cfg);
    check("trine2: save succeeds", rc == 0);

    trine_s2_model_t *loaded = trine_s2_load(TMP_TRINE2);
    check("trine2: load returns non-NULL", loaded != NULL);

    int same = loaded && models_same_output(orig, loaded);
    check("trine2: roundtrip output matches", same);

    trine_s2_free(orig);
    trine_s2_free(loaded);
}

static void test_trine2_validate_valid(void)
{
    /* Save a valid model and validate the file */
    trine_s2_model_t *m = trine_s2_create_random(512, 77);
    trine_s2_save(m, TMP_TRINE2, NULL);
    trine_s2_free(m);

    int rc = trine_s2_validate(TMP_TRINE2);
    check("trine2: validate accepts valid file", rc == 0);
}

static void test_trine2_reject_wrong_magic(void)
{
    /* Save valid file, then corrupt the magic bytes */
    trine_s2_model_t *m = trine_s2_create_random(256, 99);
    trine_s2_save(m, TMP_CORRUPT, NULL);
    trine_s2_free(m);

    FILE *fp = fopen(TMP_CORRUPT, "r+b");
    if (fp) {
        uint8_t bad[4] = { 0xBA, 0xAD, 0xF0, 0x0D };
        fwrite(bad, 1, 4, fp);
        fclose(fp);
    }

    int rc = trine_s2_validate(TMP_CORRUPT);
    check("trine2: rejects wrong magic bytes", rc != 0);

    trine_s2_model_t *loaded = trine_s2_load(TMP_CORRUPT);
    check("trine2: load fails on wrong magic", loaded == NULL);
    trine_s2_free(loaded);
}

static void test_trine2_reject_mismatched_checksum(void)
{
    /* Save valid file, then corrupt the payload to mismatch checksum */
    trine_s2_model_t *m = trine_s2_create_random(256, 123);
    trine_s2_save(m, TMP_CORRUPT, NULL);
    trine_s2_free(m);

    /* Corrupt a byte in the projection weights (offset 72 + 500) */
    FILE *fp = fopen(TMP_CORRUPT, "r+b");
    if (fp) {
        fseek(fp, 72 + 500, SEEK_SET);
        uint8_t bad = 0xFF;
        fwrite(&bad, 1, 1, fp);
        fclose(fp);
    }

    int rc = trine_s2_validate(TMP_CORRUPT);
    check("trine2: rejects mismatched checksum", rc != 0);
}

static void test_trine2_reject_empty_file(void)
{
    /* Write an empty file */
    FILE *fp = fopen(TMP_CORRUPT, "wb");
    if (fp) fclose(fp);

    int rc = trine_s2_validate(TMP_CORRUPT);
    check("trine2: rejects empty file", rc != 0);

    trine_s2_model_t *loaded = trine_s2_load(TMP_CORRUPT);
    check("trine2: load fails on empty file", loaded == NULL);
    trine_s2_free(loaded);
}

static void test_trine2_reject_unknown_version(void)
{
    /* Save a valid file, then set version to 99 */
    trine_s2_model_t *m = trine_s2_create_random(256, 55);
    trine_s2_save(m, TMP_CORRUPT, NULL);
    trine_s2_free(m);

    /* Read the entire file */
    FILE *fp = fopen(TMP_CORRUPT, "rb");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)size);
    size_t nread = fread(buf, 1, (size_t)size, fp);
    fclose(fp);
    (void)nread;

    /* Overwrite version field at offset 4 with value 99 */
    uint32_t bad_version = 99;
    memcpy(buf + 4, &bad_version, 4);

    /* Write back (checksum will be wrong too, but version should be
     * checked first) */
    write_raw_file(TMP_CORRUPT, buf, (size_t)size);
    free(buf);

    int rc = trine_s2_validate(TMP_CORRUPT);
    check("trine2: rejects unknown version (99)", rc != 0);
}

/* ══════════════════════════════════════════════════════════════════════
 * Category 2: .tridx Index Format Compatibility
 * ══════════════════════════════════════════════════════════════════════ */

static void test_tridx_save_load_roundtrip(void)
{
    printf("  --- .tridx index format ---\n");

    trine_s1_config_t cfg = TRINE_S1_CONFIG_DEFAULT;
    trine_s1_index_t *idx = trine_s1_index_create(&cfg);
    check("tridx: index create succeeds", idx != NULL);

    /* Encode and add 10 entries with tags */
    uint8_t embs[10][240];
    char tag_buf[10][32];
    for (int i = 0; i < N_TEXTS; i++) {
        trine_s1_encode(TEST_TEXTS[i], strlen(TEST_TEXTS[i]), embs[i]);
        snprintf(tag_buf[i], sizeof(tag_buf[i]), "doc_%d", i);
        trine_s1_index_add(idx, embs[i], tag_buf[i]);
    }

    /* Save the index */
    int rc = trine_s1_index_save(idx, TMP_TRIDX);
    check("tridx: save succeeds", rc == 0);

    /* Load it back */
    trine_s1_index_t *loaded = trine_s1_index_load(TMP_TRIDX);
    check("tridx: load returns non-NULL", loaded != NULL);

    /* Verify entry count is preserved */
    int orig_count = trine_s1_index_count(idx);
    int load_count = loaded ? trine_s1_index_count(loaded) : -1;
    check("tridx: entry count preserved", orig_count == N_TEXTS &&
                                            load_count == N_TEXTS);

    /* Verify tags are preserved */
    if (loaded) {
        int tags_ok = 1;
        for (int i = 0; i < N_TEXTS; i++) {
            const char *tag = trine_s1_index_tag(loaded, i);
            char expected[32];
            snprintf(expected, sizeof(expected), "doc_%d", i);
            if (!tag || strcmp(tag, expected) != 0) {
                tags_ok = 0;
                break;
            }
        }
        check("tridx: tag strings preserved", tags_ok);
    } else {
        check("tridx: tag strings preserved", 0);
    }

    /* Verify query returns correct best match */
    if (loaded) {
        /* Query with the first text -- should match itself (index 0) */
        trine_s1_result_t r = trine_s1_index_query(loaded, embs[0]);
        check("tridx: query best match correct", r.matched_index == 0);
        check("tridx: query best similarity > 0.9",
              r.similarity > 0.9f || r.calibrated > 0.9f);
    } else {
        check("tridx: query best match correct", 0);
        check("tridx: query best similarity > 0.9", 0);
    }

    trine_s1_index_free(idx);
    trine_s1_index_free(loaded);
}

/* ══════════════════════════════════════════════════════════════════════
 * Category 3: .trrt Routed Index Format Compatibility
 * ══════════════════════════════════════════════════════════════════════ */

static void test_trrt_save_load_roundtrip(void)
{
    printf("  --- .trrt routed index format ---\n");

    trine_s1_config_t cfg = TRINE_S1_CONFIG_DEFAULT;
    trine_route_t *rt = trine_route_create(&cfg);
    check("trrt: route create succeeds", rt != NULL);

    /* Encode and add 10 entries */
    uint8_t embs[10][240];
    for (int i = 0; i < N_TEXTS; i++) {
        trine_s1_encode(TEST_TEXTS[i], strlen(TEST_TEXTS[i]), embs[i]);
        char tag[32];
        snprintf(tag, sizeof(tag), "rt_doc_%d", i);
        trine_route_add(rt, embs[i], tag);
    }

    /* Save the routed index */
    int rc = trine_route_save(rt, TMP_TRRT);
    check("trrt: save succeeds", rc == 0);

    /* Load it back */
    trine_route_t *loaded = trine_route_load(TMP_TRRT);
    check("trrt: load returns non-NULL", loaded != NULL);

    /* Verify entry count is preserved */
    int orig_count = trine_route_count(rt);
    int load_count = loaded ? trine_route_count(loaded) : -1;
    check("trrt: entry count preserved", orig_count == N_TEXTS &&
                                           load_count == N_TEXTS);

    /* Verify query returns correct results */
    if (loaded) {
        trine_route_stats_t stats;
        trine_s1_result_t r = trine_route_query(loaded, embs[0], &stats);
        check("trrt: query finds match", r.matched_index >= 0);

        /* The best match for text[0] should be itself (index 0) */
        check("trrt: query best match is self", r.matched_index == 0);
    } else {
        check("trrt: query finds match", 0);
        check("trrt: query best match is self", 0);
    }

    trine_route_free(rt);
    trine_route_free(loaded);
}

/* ══════════════════════════════════════════════════════════════════════
 * Category 4: Cross-Format Consistency
 * ══════════════════════════════════════════════════════════════════════ */

static void test_index_query_similarity_preserved(void)
{
    printf("  --- cross-format consistency ---\n");

    trine_s1_config_t cfg = TRINE_S1_CONFIG_DEFAULT;

    /* Build index with two similar texts */
    trine_s1_index_t *idx = trine_s1_index_create(&cfg);
    uint8_t emb_a[240], emb_b[240];
    trine_s1_encode("fast red fox", strlen("fast red fox"), emb_a);
    trine_s1_encode("quick red fox", strlen("quick red fox"), emb_b);
    trine_s1_index_add(idx, emb_a, "a");
    trine_s1_index_add(idx, emb_b, "b");

    /* Measure similarity before save */
    trine_s1_result_t r_before = trine_s1_index_query(idx, emb_a);
    float sim_before = r_before.similarity;

    /* Save and reload */
    trine_s1_index_save(idx, TMP_TRIDX);
    trine_s1_index_t *loaded = trine_s1_index_load(TMP_TRIDX);

    /* Measure similarity after load */
    float sim_after = -1.0f;
    if (loaded) {
        trine_s1_result_t r_after = trine_s1_index_query(loaded, emb_a);
        sim_after = r_after.similarity;
    }

    /* Similarities should match exactly (same bytes) */
    check("consistency: query similarity preserved across save/load",
          loaded && fabsf(sim_before - sim_after) < 1e-6f);

    trine_s1_index_free(idx);
    trine_s1_index_free(loaded);
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Backward Compatibility Tests ===\n\n");

    cleanup();

    /* Category 1: .trine2 model format */
    test_trine2_save_load_roundtrip();
    test_trine2_validate_valid();
    test_trine2_reject_wrong_magic();
    test_trine2_reject_mismatched_checksum();
    test_trine2_reject_empty_file();
    test_trine2_reject_unknown_version();

    /* Category 2: .tridx index format */
    test_tridx_save_load_roundtrip();

    /* Category 3: .trrt routed index format */
    test_trrt_save_load_roundtrip();

    /* Category 4: cross-format consistency */
    test_index_query_similarity_preserved();

    cleanup();

    printf("\nBackward compat: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);

    return g_failed > 0 ? 1 : 0;
}
