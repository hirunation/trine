/* =====================================================================
 * TRINE Stage-2 — Persistence Tests
 * =====================================================================
 *
 * Tests for .trine2 file format save/load/validate.
 *
 * ===================================================================== */

#include "trine_s2_persist.h"
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
        printf("  FAIL  persist: %s\n", name);
    }
}

static const char *TMPFILE  = "/tmp/trine_test_persist.trine2";
static const char *TMPFILE2 = "/tmp/trine_test_persist2.trine2";

/* ── Helpers ────────────────────────────────────────────────────────── */

static void cleanup(void)
{
    (void)remove(TMPFILE);
    (void)remove(TMPFILE2);
}

/* Compare two models by encoding texts and comparing outputs */
static int models_same_output(const trine_s2_model_t *a,
                               const trine_s2_model_t *b)
{
    const char *texts[] = {
        "Hello, world!",
        "The quick brown fox",
        "TRINE encoding test",
        "12345"
    };

    for (int i = 0; i < 4; i++) {
        uint8_t out_a[240], out_b[240];
        size_t len = strlen(texts[i]);

        int ra = trine_s2_encode(a, texts[i], len, 0, out_a);
        int rb = trine_s2_encode(b, texts[i], len, 0, out_b);

        if (ra != rb) return 0;
        if (ra == 0 && memcmp(out_a, out_b, 240) != 0) return 0;
    }
    return 1;
}

/* ── Tests ──────────────────────────────────────────────────────────── */

static void test_save_identity(void)
{
    trine_s2_model_t *m = trine_s2_create_identity();
    int rc = trine_s2_save(m, TMPFILE, NULL);
    check("save identity model", m != NULL && rc == 0);
    trine_s2_free(m);
}

static void test_load_identity(void)
{
    trine_s2_model_t *orig = trine_s2_create_identity();
    trine_s2_save(orig, TMPFILE, NULL);

    trine_s2_model_t *loaded = trine_s2_load(TMPFILE);
    int same = loaded && models_same_output(orig, loaded);
    check("load identity model produces same output", same);

    trine_s2_free(orig);
    trine_s2_free(loaded);
}

static void test_save_random(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);
    trine_s2_save_config_t cfg = {
        .similarity_threshold = 0.85f,
        .density = 0.33f,
        .topo_seed = 42 ^ 0xDEADBEEFCAFEBABEULL
    };
    int rc = trine_s2_save(m, TMPFILE, &cfg);
    check("save random model", m != NULL && rc == 0);
    trine_s2_free(m);
}

static void test_roundtrip_random(void)
{
    trine_s2_model_t *orig = trine_s2_create_random(512, 42);
    trine_s2_save_config_t cfg = {
        .similarity_threshold = 0.85f,
        .density = 0.33f,
        .topo_seed = 42 ^ 0xDEADBEEFCAFEBABEULL
    };

    trine_s2_save(orig, TMPFILE, &cfg);
    trine_s2_model_t *loaded = trine_s2_load(TMPFILE);

    int same = loaded && models_same_output(orig, loaded);
    check("roundtrip random model", same);

    trine_s2_free(orig);
    trine_s2_free(loaded);
}

static void test_roundtrip_diagonal(void)
{
    trine_s2_model_t *orig = trine_s2_create_random(512, 123);
    trine_s2_set_projection_mode(orig, TRINE_S2_PROJ_DIAGONAL);

    trine_s2_save_config_t cfg = {
        .similarity_threshold = 0.90f,
        .density = 0.15f,
        .topo_seed = 123 ^ 0xDEADBEEFCAFEBABEULL
    };

    trine_s2_save(orig, TMPFILE, &cfg);
    trine_s2_model_t *loaded = trine_s2_load(TMPFILE);

    int mode_ok = loaded &&
                  trine_s2_get_projection_mode(loaded) == TRINE_S2_PROJ_DIAGONAL;
    check("diagonal mode preserved after roundtrip", mode_ok);

    int same = loaded && models_same_output(orig, loaded);
    check("diagonal model output matches after roundtrip", same);

    trine_s2_free(orig);
    trine_s2_free(loaded);
}

static void test_validate_valid(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);
    trine_s2_save(m, TMPFILE, NULL);
    trine_s2_free(m);

    int rc = trine_s2_validate(TMPFILE);
    check("validate accepts valid file", rc == 0);
}

static void test_validate_corrupt_magic(void)
{
    trine_s2_model_t *m = trine_s2_create_random(256, 99);
    trine_s2_save(m, TMPFILE, NULL);
    trine_s2_free(m);

    /* Corrupt first byte */
    FILE *fp = fopen(TMPFILE, "r+b");
    uint8_t bad = 0xFF;
    fwrite(&bad, 1, 1, fp);
    fclose(fp);

    int rc = trine_s2_validate(TMPFILE);
    check("detect corrupt magic", rc != 0);
}

static void test_validate_corrupt_header(void)
{
    trine_s2_model_t *m = trine_s2_create_random(256, 99);
    trine_s2_save(m, TMPFILE, NULL);
    trine_s2_free(m);

    /* Corrupt flags field at offset 8 */
    FILE *fp = fopen(TMPFILE, "r+b");
    fseek(fp, 8, SEEK_SET);
    uint8_t bad = 0xFF;
    fwrite(&bad, 1, 1, fp);
    fclose(fp);

    int rc = trine_s2_validate(TMPFILE);
    check("detect corrupt header checksum", rc != 0);
}

static void test_validate_corrupt_payload(void)
{
    trine_s2_model_t *m = trine_s2_create_random(256, 99);
    trine_s2_save(m, TMPFILE, NULL);
    trine_s2_free(m);

    /* Corrupt a byte in weights */
    FILE *fp = fopen(TMPFILE, "r+b");
    fseek(fp, 72 + 1000, SEEK_SET);
    uint8_t bad = 0xFF;
    fwrite(&bad, 1, 1, fp);
    fclose(fp);

    int rc = trine_s2_validate(TMPFILE);
    check("detect corrupt payload", rc != 0);
}

static void test_load_nonexistent(void)
{
    trine_s2_model_t *m = trine_s2_load("/tmp/nonexistent_trine2.trine2");
    check("load nonexistent returns NULL", m == NULL);
}

static void test_save_null_args(void)
{
    int rc1 = trine_s2_save(NULL, TMPFILE, NULL);
    trine_s2_model_t *m = trine_s2_create_identity();
    int rc2 = trine_s2_save(m, NULL, NULL);
    trine_s2_free(m);
    check("save NULL model returns -1", rc1 == -1);
    check("save NULL path returns -1", rc2 == -1);
}

static void test_info_preserved(void)
{
    trine_s2_model_t *orig = trine_s2_create_random(512, 42);
    trine_s2_save_config_t cfg = {
        .topo_seed = 42 ^ 0xDEADBEEFCAFEBABEULL
    };
    trine_s2_save(orig, TMPFILE, &cfg);

    trine_s2_model_t *loaded = trine_s2_load(TMPFILE);

    trine_s2_info_t oi, li;
    trine_s2_info(orig, &oi);
    trine_s2_info(loaded, &li);

    int ok = (oi.projection_k == li.projection_k) &&
             (oi.projection_dims == li.projection_dims) &&
             (oi.cascade_cells == li.cascade_cells);
    check("model info preserved after save/load", ok && loaded != NULL);

    trine_s2_free(orig);
    trine_s2_free(loaded);
}

static void test_double_roundtrip(void)
{
    trine_s2_model_t *orig = trine_s2_create_random(256, 77);
    trine_s2_set_projection_mode(orig, TRINE_S2_PROJ_DIAGONAL);

    trine_s2_save_config_t cfg = { .topo_seed = 77 ^ 0xDEADBEEFCAFEBABEULL };

    trine_s2_save(orig, TMPFILE, &cfg);
    trine_s2_model_t *m1 = trine_s2_load(TMPFILE);

    trine_s2_save(m1, TMPFILE2, &cfg);
    trine_s2_model_t *m2 = trine_s2_load(TMPFILE2);

    int same = m2 && models_same_output(orig, m2);
    check("double save/load produces same output", same);

    trine_s2_free(orig);
    trine_s2_free(m1);
    trine_s2_free(m2);
}

static void test_file_size(void)
{
    trine_s2_model_t *m = trine_s2_create_random(512, 42);
    trine_s2_save(m, TMPFILE, NULL);
    trine_s2_free(m);

    FILE *fp = fopen(TMPFILE, "rb");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fclose(fp);

    /* 72 header + 3*240*240 weights + 8 checksum = 172,880 */
    long expected = 72 + 3 * 240 * 240 + 8;
    check("file size matches format spec", size == expected);
}

static void test_zero_cascade_roundtrip(void)
{
    trine_s2_model_t *identity = trine_s2_create_identity();
    const void *proj = trine_s2_get_projection(identity);

    trine_s2_model_t *orig = trine_s2_create_from_parts(proj, 0, 0);
    trine_s2_free(identity);

    trine_s2_save(orig, TMPFILE, NULL);
    trine_s2_model_t *loaded = trine_s2_load(TMPFILE);

    int same = loaded && models_same_output(orig, loaded);
    check("zero-cascade model roundtrip", same);

    trine_s2_free(orig);
    trine_s2_free(loaded);
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
    printf("=== Stage-2 Persistence Tests ===\n\n");

    cleanup();

    test_save_identity();
    test_load_identity();
    test_save_random();
    test_roundtrip_random();
    test_roundtrip_diagonal();
    test_validate_valid();
    test_validate_corrupt_magic();
    test_validate_corrupt_header();
    test_validate_corrupt_payload();
    test_load_nonexistent();
    test_save_null_args();
    test_info_preserved();
    test_double_roundtrip();
    test_file_size();
    test_zero_cascade_roundtrip();

    cleanup();

    printf("\nPersistence: %d passed, %d failed, %d total\n",
           g_passed, g_failed, g_total);

    return g_failed > 0 ? 1 : 0;
}
