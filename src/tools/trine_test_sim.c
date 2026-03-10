/* ═══════════════════════════════════════════════════════════════════════
 * TRINE Similarity Test Harness
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Comprehensive test suite for the TRINE shingle encoder's cosine
 * similarity properties. Tests across 31 categories:
 *
 *   1. Identity           — identical texts must produce cosine = 1.0
 *   2. Case insensitivity — case variants must produce cosine = 1.0
 *   3. Edit dist 1, first — single first-char substitution
 *   4. Edit dist 1, mid   — single middle-char substitution
 *   5. Edit dist 1, last  — single last-char substitution
 *   6. Short dissimilar   — no character overlap
 *   7. Near-duplicate long — one-word substitution in sentence
 *   8. Disjoint long      — unrelated sentences
 *   9. Prefix/substring   — shorter text is prefix of longer
 *  10. Separation inv.    — near-duplicate MUST outscore unrelated
 *  11. Symmetry           — cosine(A,B) == cosine(B,A)
 *  12. Monotonicity       — more edits => lower similarity
 *  13. Long text stability — long near-duplicate paragraphs
 *  14. Single character   — single-char identity and dissimilarity
 *  15. Word reordering    — same words, different order
 *  16. Repetition/padding — repeated text, whitespace, doubled words
 *  17. Empty/whitespace   — empty strings, spaces, tabs, newlines
 *  18. Non-ASCII/binary   — high-bit bytes, control chars, mixed
 *  19. Numbers/digits     — numeric strings, format variants
 *  20. Very long text     — 10K-100K char procedural texts
 *  21. Repeated chars     — single-char and pattern repetition
 *  22. Punctuation        — punctuation sensitivity and tolerance
 *  23. IDF relevance      — IDF-weighted cosine gap widening
 *  24. Canonicalization   — deterministic transforms, preset consistency
 *  25. Format round-trip  — index save/load with v2 checksums
 *  26. CS-IDF unit        — init, observe, compute, merge, serialize
 *  27. CS-IDF scoring     — weighted cosine, IDF vs uniform divergence
 *  28. Field-aware unit   — config, presets, parse, JSONL extraction
 *  29. Field-aware scoring — weighted field cosine, route embedding
 *  30. Route v4 round-trip — CS-IDF + field save/load through route
 *  31. Append mode        — build-half + append-half equivalence
 *
 * Build:
 *   cc -O2 -o trine_test_sim trine_test_sim.c trine_encode.c -lm
 *
 * Usage:
 *   ./trine_test_sim            # normal output
 *   ./trine_test_sim --verbose  # print cosine values for every test
 *
 * Exit code: number of failures (0 = all pass).
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#include "trine_encode.h"
#include "trine_idf.h"
#include "trine_canon.h"
#include "trine_stage1.h"
#include "trine_csidf.h"
#include "trine_field.h"
#include "trine_route.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Cosine Similarity for 240-dimensional ternary vectors
 * ═══════════════════════════════════════════════════════════════════════
 *
 * Standard cosine: dot(a,b) / (|a| * |b|).
 * Both vectors contain values in {0, 1, 2}.
 * If either vector is all-zero, returns 1.0 when both are zero
 * (identical empty encodings), 0.0 otherwise.
 */
static double cosine_similarity(const uint8_t a[240], const uint8_t b[240])
{
    double dot = 0.0, mag_a = 0.0, mag_b = 0.0;

    for (int i = 0; i < 240; i++) {
        double va = (double)a[i];
        double vb = (double)b[i];
        dot   += va * vb;
        mag_a += va * va;
        mag_b += vb * vb;
    }

    /* Both zero vectors: identical empty input */
    if (mag_a == 0.0 && mag_b == 0.0)
        return 1.0;

    /* One zero, one non-zero: maximally dissimilar */
    if (mag_a == 0.0 || mag_b == 0.0)
        return 0.0;

    return dot / (sqrt(mag_a) * sqrt(mag_b));
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test Infrastructure
 * ═══════════════════════════════════════════════════════════════════════ */

static int g_verbose  = 0;
static int g_passed   = 0;
static int g_failed   = 0;
static int g_total    = 0;

/* Range test: encode two texts, check cosine is within [lo, hi]. */
static void test_range(const char *category, int case_num,
                       const char *text_a, const char *text_b,
                       double lo, double hi)
{
    uint8_t enc_a[240], enc_b[240];
    trine_encode_shingle(text_a, strlen(text_a), enc_a);
    trine_encode_shingle(text_b, strlen(text_b), enc_b);

    double sim = cosine_similarity(enc_a, enc_b);
    int pass = (sim >= lo - 1e-9) && (sim <= hi + 1e-9);

    g_total++;
    if (pass) {
        g_passed++;
        if (g_verbose)
            printf("  PASS  %s #%02d  cos=%.6f  [%.2f, %.2f]  \"%s\" vs \"%s\"\n",
                   category, case_num, sim, lo, hi, text_a, text_b);
    } else {
        g_failed++;
        printf("  FAIL  %s #%02d  cos=%.6f  expected [%.2f, %.2f]  \"%s\" vs \"%s\"\n",
               category, case_num, sim, lo, hi, text_a, text_b);
    }
}

/* Special empty-string test: both inputs are length 0. */
static void test_empty(const char *category, int case_num)
{
    uint8_t enc_a[240], enc_b[240];
    trine_encode_shingle("", 0, enc_a);
    trine_encode_shingle("", 0, enc_b);

    double sim = cosine_similarity(enc_a, enc_b);
    int pass = (sim >= 1.0 - 1e-9);

    g_total++;
    if (pass) {
        g_passed++;
        if (g_verbose)
            printf("  PASS  %s #%02d  cos=%.6f  [1.00, 1.00]  \"\" vs \"\"\n",
                   category, case_num, sim);
    } else {
        g_failed++;
        printf("  FAIL  %s #%02d  cos=%.6f  expected [1.00, 1.00]  \"\" vs \"\"\n",
               category, case_num, sim);
    }
}

/* Separation invariant test: sim(A, near) > sim(A, far).
 * Verifies that a near-duplicate of A scores strictly higher than
 * an unrelated text, which is the single most important property
 * for a similarity-preserving embedding. */
static void test_separation(const char *category, int case_num,
                            const char *text_a,
                            const char *text_near,
                            const char *text_far)
{
    uint8_t enc_a[240], enc_near[240], enc_far[240];
    trine_encode_shingle(text_a,    strlen(text_a),    enc_a);
    trine_encode_shingle(text_near, strlen(text_near), enc_near);
    trine_encode_shingle(text_far,  strlen(text_far),  enc_far);

    double sim_near = cosine_similarity(enc_a, enc_near);
    double sim_far  = cosine_similarity(enc_a, enc_far);
    int pass = (sim_near > sim_far);

    g_total++;
    if (pass) {
        g_passed++;
        if (g_verbose)
            printf("  PASS  %s #%02d  near=%.6f > far=%.6f  "
                   "\"%s\" ~ \"%s\" vs \"%s\"\n",
                   category, case_num, sim_near, sim_far,
                   text_a, text_near, text_far);
    } else {
        g_failed++;
        printf("  FAIL  %s #%02d  near=%.6f <= far=%.6f  "
               "\"%s\" ~ \"%s\" vs \"%s\"\n",
               category, case_num, sim_near, sim_far,
               text_a, text_near, text_far);
    }
}

/* Range test with explicit lengths (for binary-safe tests with embedded
 * NUL or high-bit bytes where strlen() would be wrong). */
static void test_range_len(const char *category, int case_num,
                           const char *text_a, size_t len_a,
                           const char *text_b, size_t len_b,
                           double lo, double hi,
                           const char *desc)
{
    uint8_t enc_a[240], enc_b[240];
    trine_encode_shingle(text_a, len_a, enc_a);
    trine_encode_shingle(text_b, len_b, enc_b);

    double sim = cosine_similarity(enc_a, enc_b);
    int pass = (sim >= lo - 1e-9) && (sim <= hi + 1e-9);

    g_total++;
    if (pass) {
        g_passed++;
        if (g_verbose)
            printf("  PASS  %s #%02d  cos=%.6f  [%.2f, %.2f]  %s\n",
                   category, case_num, sim, lo, hi, desc);
    } else {
        g_failed++;
        printf("  FAIL  %s #%02d  cos=%.6f  expected [%.2f, %.2f]  %s\n",
               category, case_num, sim, lo, hi, desc);
    }
}

/* Bool predicate test: pass if `condition` is true. */
static void test_bool(const char *category, int case_num,
                      int condition, double val, const char *desc)
{
    g_total++;
    if (condition) {
        g_passed++;
        if (g_verbose)
            printf("  PASS  %s #%02d  val=%.6f  %s\n",
                   category, case_num, val, desc);
    } else {
        g_failed++;
        printf("  FAIL  %s #%02d  val=%.6f  %s\n",
               category, case_num, val, desc);
    }
}

static void print_header(const char *name)
{
    printf("\n--- %s ---\n", name);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test Cases
 * ═══════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    /* Parse --verbose flag */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0)
            g_verbose = 1;
    }

    printf("TRINE Similarity Test Harness\n");
    printf("Encoder: trine_encode_shingle (240-dim ternary, 4 chains)\n");
    printf("Mode: %s\n", g_verbose ? "verbose" : "normal (use --verbose for details)");

    /* ─────────────────────────────────────────────────────────────────
     * Category 1: Identity (5 tests) — identical texts must be 1.000
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 1: Identity");
    test_range("Identity", 1,
        "cat", "cat", 1.000, 1.000);
    test_range("Identity", 2,
        "hello world", "hello world", 1.000, 1.000);
    test_empty("Identity", 3);
    test_range("Identity", 4,
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog", 1.000, 1.000);
    test_range("Identity", 5,
        "a", "a", 1.000, 1.000);

    /* ─────────────────────────────────────────────────────────────────
     * Category 2: Case Insensitivity (5 tests) — must be 1.000
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 2: Case Insensitivity");
    test_range("CaseInsens", 1,
        "Cat", "cat", 1.000, 1.000);
    test_range("CaseInsens", 2,
        "CAT", "cat", 1.000, 1.000);
    test_range("CaseInsens", 3,
        "Hello World", "hello world", 1.000, 1.000);
    test_range("CaseInsens", 4,
        "MACHINE LEARNING", "machine learning", 1.000, 1.000);
    test_range("CaseInsens", 5,
        "The Quick Brown Fox", "the quick brown fox", 1.000, 1.000);

    /* ─────────────────────────────────────────────────────────────────
     * Category 3: Edit Distance 1 — First Char (5 tests)
     * Single character substitution at position 0.
     * Expect 0.15-0.55: first-char changes destroy 1 unigram,
     * 1 bigram, and the first trigram. Remaining features overlap.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 3: Edit Dist 1 - First Char");
    test_range("EditFirst", 1, "cat",  "bat",  0.15, 0.55);
    test_range("EditFirst", 2, "dog",  "fog",  0.15, 0.55);
    test_range("EditFirst", 3, "run",  "fun",  0.15, 0.55);
    test_range("EditFirst", 4, "hat",  "bat",  0.15, 0.55);
    test_range("EditFirst", 5, "pen",  "ten",  0.15, 0.55);

    /* ─────────────────────────────────────────────────────────────────
     * Category 4: Edit Distance 1 — Middle Char (5 tests)
     * Single character substitution at the middle position.
     * Expect 0.05-0.45: middle-char changes destroy both bigrams
     * for 3-letter words, but preserve 2/3 of unigrams.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 4: Edit Dist 1 - Middle Char");
    test_range("EditMid", 1, "cat",  "cot",  0.05, 0.45);
    test_range("EditMid", 2, "dog",  "dig",  0.05, 0.45);
    test_range("EditMid", 3, "run",  "ran",  0.05, 0.45);
    test_range("EditMid", 4, "hat",  "hot",  0.05, 0.45);
    test_range("EditMid", 5, "pen",  "pin",  0.05, 0.45);

    /* ─────────────────────────────────────────────────────────────────
     * Category 5: Edit Distance 1 — Last Char (5 tests)
     * Single character substitution at the final position.
     * Expect 0.10-0.55: last-char changes destroy 1 unigram,
     * 1 bigram, and the last trigram.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 5: Edit Dist 1 - Last Char");
    test_range("EditLast", 1, "cat",  "car",  0.10, 0.55);
    test_range("EditLast", 2, "dog",  "dot",  0.10, 0.55);
    test_range("EditLast", 3, "run",  "rum",  0.10, 0.55);
    test_range("EditLast", 4, "hat",  "had",  0.10, 0.55);
    test_range("EditLast", 5, "pen",  "peg",  0.10, 0.55);

    /* ─────────────────────────────────────────────────────────────────
     * Category 6: Short Dissimilar — no character overlap (5 tests)
     * Words sharing zero characters. Expect 0.00-0.20: any non-zero
     * similarity comes from hash collisions in the 60-slot bands.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 6: Short Dissimilar");
    test_range("ShortDis", 1, "cat",  "gym",  0.00, 0.20);
    test_range("ShortDis", 2, "dog",  "few",  0.00, 0.20);
    test_range("ShortDis", 3, "hat",  "fly",  0.00, 0.20);
    test_range("ShortDis", 4, "pen",  "gym",  0.00, 0.20);
    test_range("ShortDis", 5, "bad",  "joy",  0.00, 0.20);

    /* ─────────────────────────────────────────────────────────────────
     * Category 7: Near-Duplicate Long (5 tests)
     * Sentences differing by a single word substitution.
     * Expect 0.55-0.95: most n-grams and word features are shared.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 7: Near-Duplicate Long");
    test_range("NearDup", 1,
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox leaps over the lazy dog", 0.55, 0.95);
    test_range("NearDup", 2,
        "machine learning is great",
        "machine learning is useful", 0.55, 0.95);
    test_range("NearDup", 3,
        "programming is fun",
        "programming is boring", 0.55, 0.95);
    test_range("NearDup", 4,
        "climate change affects everything",
        "climate change impacts everything", 0.55, 0.95);
    test_range("NearDup", 5,
        "the server crashed yesterday",
        "the server failed yesterday", 0.55, 0.95);

    /* ─────────────────────────────────────────────────────────────────
     * Category 8: Disjoint Long (5 tests)
     * Sentences about completely unrelated topics.
     * Expect 0.20-0.60: partial overlap from common English n-grams
     * (e.g., "the", "ing", " th") and hash collisions.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 8: Disjoint Long");
    test_range("DisjLong", 1,
        "the quick brown fox",
        "quantum physics theory", 0.20, 0.60);
    test_range("DisjLong", 2,
        "machine learning algorithms",
        "cooking italian pasta", 0.20, 0.60);
    test_range("DisjLong", 3,
        "programming in python",
        "gardening with flowers", 0.20, 0.60);
    test_range("DisjLong", 4,
        "hello beautiful world",
        "goodbye dark universe", 0.20, 0.60);
    test_range("DisjLong", 5,
        "server infrastructure design",
        "chocolate cake recipe", 0.20, 0.60);

    /* ─────────────────────────────────────────────────────────────────
     * Category 9: Structural — Prefix / Substring (5 tests)
     * Shorter text is a prefix of the longer text.
     * Expect 0.25-0.75: all n-grams from the shorter text appear
     * in the longer text, but the longer text has additional features
     * that dilute the cosine.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 9: Prefix / Substring");
    test_range("Prefix", 1, "cat",    "cats",       0.25, 0.75);
    test_range("Prefix", 2, "cat",    "catch",      0.25, 0.75);
    test_range("Prefix", 3, "run",    "running",    0.25, 0.75);
    test_range("Prefix", 4, "hello",  "helloworld", 0.25, 0.75);
    test_range("Prefix", 5, "test",   "testing",    0.25, 0.75);

    /* ─────────────────────────────────────────────────────────────────
     * Category 10: Separation Invariants (5 tests)
     *
     * THE CRITICAL TEST: for every triple (A, near_A, far_B):
     *   cosine(A, near_A) > cosine(A, far_B)
     *
     * This validates that the embedding space preserves semantic
     * neighborhoods — near-duplicates must always outscore unrelated
     * text. Failure here means the encoder is broken for retrieval.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 10: Separation Invariants");

    /* 10.1: Sentence with one-word swap vs completely different topic */
    test_separation("SepInv", 1,
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox leaps over the lazy dog",
        "quantum physics explores subatomic particles");

    /* 10.2: Same domain, near-synonym vs different domain */
    test_separation("SepInv", 2,
        "machine learning is transforming technology",
        "machine learning is revolutionizing technology",
        "baking chocolate chip cookies at home");

    /* 10.3: Short word variant vs unrelated short word */
    test_separation("SepInv", 3,
        "programming in python is efficient",
        "programming in python is effective",
        "gardening with exotic tropical flowers");

    /* 10.4: Near-duplicate with verb swap vs different sentence */
    test_separation("SepInv", 4,
        "the server crashed under heavy load",
        "the server failed under heavy load",
        "beautiful sunset over the mountain lake");

    /* 10.5: Adjective swap vs unrelated */
    test_separation("SepInv", 5,
        "deep neural networks learn complex patterns",
        "deep neural networks learn intricate patterns",
        "fresh strawberry jam on warm toast");

    /* ─────────────────────────────────────────────────────────────────
     * Category 11: Symmetry (5 tests)
     * cosine(A, B) must equal cosine(B, A) for all pairs.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 11: Symmetry");
    {
        const char *sym_pairs[][2] = {
            {"cat",          "dog"},
            {"hello world",  "goodbye world"},
            {"machine learning algorithms", "cooking italian pasta"},
            {"the quick brown fox", "the slow red hen"},
            {"a",            "z"},
        };
        for (int i = 0; i < 5; i++) {
            uint8_t ea[240], eb[240];
            trine_encode_shingle(sym_pairs[i][0], strlen(sym_pairs[i][0]), ea);
            trine_encode_shingle(sym_pairs[i][1], strlen(sym_pairs[i][1]), eb);
            double ab = cosine_similarity(ea, eb);
            double ba = cosine_similarity(eb, ea);
            int pass = fabs(ab - ba) < 1e-12;
            g_total++;
            if (pass) {
                g_passed++;
                if (g_verbose)
                    printf("  PASS  Symmetry #%02d  cos(A,B)=%.6f == cos(B,A)=%.6f"
                           "  \"%s\" <-> \"%s\"\n",
                           i + 1, ab, ba, sym_pairs[i][0], sym_pairs[i][1]);
            } else {
                g_failed++;
                printf("  FAIL  Symmetry #%02d  cos(A,B)=%.6f != cos(B,A)=%.6f"
                       "  \"%s\" <-> \"%s\"\n",
                       i + 1, ab, ba, sym_pairs[i][0], sym_pairs[i][1]);
            }
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 12: Monotonicity (5 tests)
     * More edits should produce lower (or equal) similarity.
     * sim(A, edit_1_of_A) >= sim(A, edit_2_of_A)
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 12: Monotonicity (more edits => lower sim)");
    {
        struct mono_triple {
            const char *base;
            const char *edit1;  /* 1 edit away */
            const char *edit2;  /* 2+ edits away */
        };
        struct mono_triple monos[] = {
            {"hello world", "hallo world",  "hallo werld"},
            {"programming", "programminh",  "progremmink"},
            {"testing code", "testing coda", "tasting coda"},
            {"data science", "data scienco", "dota scienco"},
            {"network layer", "network layor", "natwork layor"},
        };
        for (int i = 0; i < 5; i++) {
            uint8_t eb[240], e1[240], e2[240];
            trine_encode_shingle(monos[i].base,  strlen(monos[i].base),  eb);
            trine_encode_shingle(monos[i].edit1, strlen(monos[i].edit1), e1);
            trine_encode_shingle(monos[i].edit2, strlen(monos[i].edit2), e2);
            double sim1 = cosine_similarity(eb, e1);
            double sim2 = cosine_similarity(eb, e2);
            int pass = (sim1 >= sim2 - 1e-9);
            g_total++;
            if (pass) {
                g_passed++;
                if (g_verbose)
                    printf("  PASS  Monotone #%02d  1-edit=%.6f >= 2-edit=%.6f"
                           "  base=\"%s\"\n",
                           i + 1, sim1, sim2, monos[i].base);
            } else {
                g_failed++;
                printf("  FAIL  Monotone #%02d  1-edit=%.6f < 2-edit=%.6f"
                       "  base=\"%s\"\n",
                       i + 1, sim1, sim2, monos[i].base);
            }
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 13: Long Text Stability (5 tests)
     * Long near-duplicate paragraphs should have high similarity.
     * Expect 0.65-0.99: lower bound accounts for short-word swaps
     * (e.g., "silence"->"quiet") that share zero character n-grams.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 13: Long Text Stability");
    test_range("LongStab", 1,
        "the quick brown fox jumps over the lazy dog and runs through the meadow",
        "the quick brown fox jumps over the lazy dog and runs across the meadow",
        0.65, 0.99);
    test_range("LongStab", 2,
        "in the beginning there was nothing but darkness and silence everywhere",
        "in the beginning there was nothing but darkness and quiet everywhere",
        0.65, 0.99);
    test_range("LongStab", 3,
        "machine learning and artificial intelligence are reshaping every industry",
        "machine learning and artificial intelligence are transforming every industry",
        0.65, 0.99);
    test_range("LongStab", 4,
        "the server processes thousands of requests per second without any errors",
        "the server handles thousands of requests per second without any errors",
        0.65, 0.99);
    test_range("LongStab", 5,
        "students learn best when they practice regularly and receive timely feedback",
        "students learn best when they practice regularly and receive prompt feedback",
        0.65, 0.99);

    /* ─────────────────────────────────────────────────────────────────
     * Category 14: Single Character Tests (5 tests)
     * Individual characters — different chars should be dissimilar,
     * same char should be identical.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 14: Single Character");
    test_range("SingleCh", 1, "a", "a", 1.000, 1.000);
    test_range("SingleCh", 2, "a", "b", 0.00,  0.35);
    test_range("SingleCh", 3, "x", "y", 0.00,  0.35);
    test_range("SingleCh", 4, "A", "a", 1.000, 1.000);  /* case insensitive */
    test_range("SingleCh", 5, "z", "a", 0.00,  0.35);

    /* ─────────────────────────────────────────────────────────────────
     * Category 15: Word Reordering (5 tests)
     * Same words, different order. Word unigrams (Chain 4) are
     * order-independent, but character n-grams change at boundaries.
     * Expect moderate similarity: 0.30-0.85.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 15: Word Reordering");
    test_range("WordReord", 1,
        "cat dog", "dog cat", 0.30, 0.85);
    test_range("WordReord", 2,
        "hello world", "world hello", 0.30, 0.85);
    test_range("WordReord", 3,
        "red blue green", "green red blue", 0.30, 0.85);
    test_range("WordReord", 4,
        "machine learning deep", "deep machine learning", 0.30, 0.85);
    test_range("WordReord", 5,
        "alpha beta gamma", "gamma alpha beta", 0.30, 0.85);

    /* ─────────────────────────────────────────────────────────────────
     * Category 16: Repetition and Padding (5 tests)
     * Repeated text, whitespace padding, and doubled words.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 16: Repetition and Padding");

    /* Doubled word vs single: high overlap but different magnitude */
    test_range("Repeat", 1,
        "cat", "cat cat", 0.30, 0.85);

    /* Triple repetition vs single: Z3 accumulation can cause
     * destructive interference (mod-3 wraparound), reducing similarity */
    test_range("Repeat", 2,
        "hello", "hello hello hello", 0.15, 0.80);

    /* Leading/trailing spaces should not affect (case-fold only, not trim)
     * -- actually spaces DO affect n-grams, so expect high but not 1.0 */
    test_range("Repeat", 3,
        "test", " test ", 0.25, 0.80);

    /* Doubled sentence vs single: high overlap from shared n-grams */
    test_range("Repeat", 4,
        "the cat sat", "the cat sat the cat sat", 0.45, 0.90);

    /* Same characters, different grouping */
    test_range("Repeat", 5,
        "ab cd ef", "abc def", 0.15, 0.65);

    /* ─────────────────────────────────────────────────────────────────
     * Category 17: Empty and Whitespace (5 tests)
     *
     * Edge cases around empty input, whitespace-only input, and
     * mixed whitespace characters. Primary concern: no crashes,
     * sane return values.
     *
     * NOTE: cosine_similarity() returns 1.0 for two zero vectors
     * (both-empty convention), so "" vs "" = 1.0 here.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 17: Empty and Whitespace");

    /* 17.1: Empty vs empty — both zero vectors, convention = 1.0 */
    test_range_len("EmptyWS", 1, "", 0, "", 0, 1.000, 1.000,
                   "\"\" vs \"\"");

    /* 17.2: Empty vs non-empty — zero vector vs non-zero = 0.0 */
    test_range_len("EmptyWS", 2, "", 0, "hello", 5, 0.00, 0.00,
                   "\"\" vs \"hello\"");

    /* 17.3: Whitespace only — identical spaces */
    test_range("EmptyWS", 3, "   ", "   ", 1.000, 1.000);

    /* 17.4: Tab + newline — identical control whitespace */
    test_range("EmptyWS", 4, "\t\n", "\t\n", 1.000, 1.000);

    /* 17.5: Spaces vs text — valid result, no crash */
    test_range("EmptyWS", 5, "     ", "hello", 0.00, 0.50);

    /* ─────────────────────────────────────────────────────────────────
     * Category 18: Non-ASCII and Binary Safety (5 tests)
     *
     * Characters >= 128 are masked to 0-127 by the encoder.
     * These tests verify no crashes and sane similarity for inputs
     * containing high-bit bytes and control characters.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 18: Non-ASCII and Binary Safety");

    /* 18.1: High-bit bytes — should not crash, cosine >= 0.0 */
    test_range_len("BinSafe", 1,
                   "\x80\x81\x82", 3, "\x00\x01\x02", 3,
                   0.00, 1.00,
                   "high-bit \\x80\\x81\\x82 vs \\x00\\x01\\x02");

    /* 18.2: Mixed ASCII/high-byte — "hello\xff" vs "hello\x7f" */
    test_range_len("BinSafe", 2,
                   "hello\xff", 6, "hello\x7f", 6,
                   0.50, 1.00,
                   "\"hello\\xff\" vs \"hello\\x7f\" (high cosine)");

    /* 18.3: All-same high byte — identical encoding after mask */
    test_range_len("BinSafe", 3,
                   "\xff\xff\xff\xff", 4, "\xff\xff\xff\xff", 4,
                   1.000, 1.000,
                   "\\xff\\xff\\xff\\xff identity");

    /* 18.4: Empty vs high-byte — zero vs non-zero = 0.0 */
    test_range_len("BinSafe", 4,
                   "", 0, "\xff", 1,
                   0.00, 0.00,
                   "\"\" vs \"\\xff\"");

    /* 18.5: Control characters — identical control sequences */
    test_range_len("BinSafe", 5,
                   "\x01\x02\x03", 3, "\x01\x02\x03", 3,
                   1.000, 1.000,
                   "\\x01\\x02\\x03 identity");

    /* ─────────────────────────────────────────────────────────────────
     * Category 19: Numbers and Digits (5 tests)
     *
     * Numeric strings: identity, reordering, edit distance,
     * format variations, and numeric-vs-alpha discrimination.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 19: Numbers and Digits");

    /* 19.1: Numeric identity */
    test_range("NumDigit", 1, "12345", "12345", 1.000, 1.000);

    /* 19.2: Reversed digits — reordered, expect moderate-low */
    test_range("NumDigit", 2, "12345", "54321", 0.00, 0.80);

    /* 19.3: One-digit difference (edit-dist-1) */
    test_range("NumDigit", 3, "12345", "12346", 0.30, 0.95);

    /* 19.4: Phone format — "555-1234" vs "5551234", shared digits */
    test_range("NumDigit", 4, "555-1234", "5551234", 0.30, 0.95);

    /* 19.5: Numeric vs alpha — completely different character sets */
    test_range("NumDigit", 5, "12345", "abcde", 0.00, 0.30);

    /* ─────────────────────────────────────────────────────────────────
     * Category 20: Very Long Text (5 tests)
     *
     * Procedurally generated texts of 10K-100K characters.
     * Tests encoder stability, no crashes, and similarity properties
     * at scale. All buffers are malloc'd and freed.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 20: Very Long Text");
    {
        /* Helper pattern: "the quick brown fox " repeated */
        const char *pattern_a = "the quick brown fox jumps over the lazy dog ";
        size_t plen_a = strlen(pattern_a);
        const char *pattern_b = "pack my box with five dozen liquor jugs now ";
        size_t plen_b = strlen(pattern_b);

        /* 20.1: 10K near-duplicate — differ by 10 chars in the middle.
         * With repeating patterns and Z3 modular accumulation,
         * destructive interference limits similarity. The key
         * invariant: near-dup must outscore disjoint (test 20.2). */
        {
            size_t sz = 10000;
            char *buf_a = (char *)malloc(sz + 1);
            char *buf_b = (char *)malloc(sz + 1);
            for (size_t i = 0; i < sz; i++) {
                buf_a[i] = pattern_a[i % plen_a];
                buf_b[i] = pattern_a[i % plen_a];
            }
            buf_a[sz] = '\0';
            buf_b[sz] = '\0';
            /* Introduce 10 differences in the middle */
            for (int d = 0; d < 10; d++)
                buf_b[5000 + d * 10] = 'Z';

            uint8_t ea[240], eb[240];
            trine_encode_shingle(buf_a, sz, ea);
            trine_encode_shingle(buf_b, sz, eb);
            double sim = cosine_similarity(ea, eb);
            test_bool("VeryLong", 1, sim > 0.45 - 1e-9, sim,
                      "10K near-dup (10 diffs) > 0.45");
            free(buf_a);
            free(buf_b);
        }

        /* 20.2: 10K completely different texts */
        {
            size_t sz = 10000;
            char *buf_a = (char *)malloc(sz + 1);
            char *buf_b = (char *)malloc(sz + 1);
            for (size_t i = 0; i < sz; i++) {
                buf_a[i] = pattern_a[i % plen_a];
                buf_b[i] = pattern_b[i % plen_b];
            }
            buf_a[sz] = '\0';
            buf_b[sz] = '\0';

            uint8_t ea[240], eb[240];
            trine_encode_shingle(buf_a, sz, ea);
            trine_encode_shingle(buf_b, sz, eb);
            double sim = cosine_similarity(ea, eb);
            test_bool("VeryLong", 2, sim < 0.65 + 1e-9, sim,
                      "10K disjoint < 0.65");
            free(buf_a);
            free(buf_b);
        }

        /* 20.3: 100K identical repeated pattern — identity = 1.000 */
        {
            size_t sz = 100000;
            char *buf = (char *)malloc(sz + 1);
            for (size_t i = 0; i < sz; i++)
                buf[i] = pattern_a[i % plen_a];
            buf[sz] = '\0';

            uint8_t ea[240], eb[240];
            trine_encode_shingle(buf, sz, ea);
            trine_encode_shingle(buf, sz, eb);
            double sim = cosine_similarity(ea, eb);
            test_bool("VeryLong", 3, sim >= 1.0 - 1e-9, sim,
                      "100K identity = 1.000");
            free(buf);
        }

        /* 20.4: Long vs short — 10K chars vs 10 chars, length mismatch */
        {
            size_t sz_long = 10000;
            char *buf_long = (char *)malloc(sz_long + 1);
            for (size_t i = 0; i < sz_long; i++)
                buf_long[i] = pattern_a[i % plen_a];
            buf_long[sz_long] = '\0';

            const char *short_text = "the quick ";

            uint8_t ea[240], eb[240];
            trine_encode_shingle(buf_long, sz_long, ea);
            trine_encode_shingle(short_text, strlen(short_text), eb);
            double sim = cosine_similarity(ea, eb);
            test_bool("VeryLong", 4, sim < 0.80 + 1e-9, sim,
                      "10K vs 10-char < 0.80");
            free(buf_long);
        }

        /* 20.5: Two long texts — mostly different content.
         * Uses truly disjoint character ranges to avoid shared
         * n-grams: lowercase letters vs digit sequences.
         * The separation invariant: this must score lower than
         * the near-duplicate (test 20.1). */
        {
            size_t sz = 10000;
            char *buf_a = (char *)malloc(sz + 1);
            char *buf_b = (char *)malloc(sz + 1);
            /* buf_a: repeating "abcdefghijklmnopqrstuvwxyz " */
            const char *alpha = "abcdefghijklmnopqrstuvwxyz ";
            size_t alen = 27;
            /* buf_b: repeating "0123456789 " */
            const char *digit = "0123456789 ";
            size_t dlen = 11;
            for (size_t i = 0; i < sz; i++) {
                buf_a[i] = alpha[i % alen];
                buf_b[i] = digit[i % dlen];
            }
            buf_a[sz] = '\0';
            buf_b[sz] = '\0';

            uint8_t ea[240], eb[240];
            trine_encode_shingle(buf_a, sz, ea);
            trine_encode_shingle(buf_b, sz, eb);
            double sim = cosine_similarity(ea, eb);
            test_bool("VeryLong", 5, sim < 0.55 + 1e-9, sim,
                      "10K alpha vs 10K digits < 0.55");
            free(buf_a);
            free(buf_b);
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 21: Repeated Characters (5 tests)
     *
     * Tests for single-character repetition and alternating patterns.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 21: Repeated Characters");

    /* 21.1: "aaaaaa" vs "aaaaaa" — identity */
    test_range("RepChar", 1, "aaaaaa", "aaaaaa", 1.000, 1.000);

    /* 21.2: "aaaaaa" vs "bbbbbb" — different repeated chars */
    test_range("RepChar", 2, "aaaaaa", "bbbbbb", 0.00, 0.50);

    /* 21.3: "ababab" vs "ababab" — alternating pattern identity */
    test_range("RepChar", 3, "ababab", "ababab", 1.000, 1.000);

    /* 21.4: "ababab" vs "bababa" — shifted alternation, same bigrams.
     * Z3 accumulation means phase-shifted patterns don't perfectly
     * align; expect moderate-high similarity. */
    test_range("RepChar", 4, "ababab", "bababa", 0.60, 1.00);

    /* 21.5: "aaa" vs "aaaa" — superset of unigrams.
     * Z3 modular arithmetic: 3 repetitions wrap to 0 mod 3 on
     * some channels while 4 repetitions land on 1 mod 3,
     * creating partial destructive interference. */
    test_range("RepChar", 5, "aaa", "aaaa", 0.15, 1.00);

    /* ─────────────────────────────────────────────────────────────────
     * Category 22: Punctuation Sensitivity (5 tests)
     *
     * Tests how punctuation affects similarity. Punctuation characters
     * generate their own n-grams but surrounding text should still
     * produce high overlap.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 22: Punctuation Sensitivity");

    /* 22.1: Comma insertion */
    test_range("Punct", 1, "hello world", "hello, world", 0.60, 1.00);

    /* 22.2: Exclamation vs question mark */
    test_range("Punct", 2, "hello!", "hello?", 0.30, 1.00);

    /* 22.3: Trailing ellipsis vs bare */
    test_range("Punct", 3, "hello...", "hello", 0.50, 1.00);

    /* 22.4: Period in abbreviation — the period creates different
     * bigrams at the boundary ("r." vs "r " and ". " vs " S"),
     * so cosine is moderate-high but not perfect. */
    test_range("Punct", 4, "Mr. Smith", "Mr Smith", 0.60, 1.00);

    /* 22.5: Apostrophe contraction */
    test_range("Punct", 5, "it's", "its", 0.30, 1.00);

    /* ─────────────────────────────────────────────────────────────────
     * Category 23: IDF Relevance (5 tests)
     *
     * Validates that IDF-weighted cosine (trine_idf.h) produces
     * meaningfully different scores than unweighted cosine, and
     * that IDF weighting WIDENS the gap between near-duplicate
     * and disjoint text pairs.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 23: IDF Relevance");
    {
        /* Near-duplicate pair */
        const char *near_a = "the quick brown fox jumps over the lazy dog and runs through the meadow";
        const char *near_b = "the quick brown fox jumps over the lazy dog and runs across the meadow";
        /* Disjoint pair */
        const char *disj_a = "quantum physics explores the nature of subatomic particles in deep space";
        const char *disj_b = "chocolate cake recipe requires butter sugar flour eggs and vanilla beans";

        uint8_t enc_na[240], enc_nb[240], enc_da[240], enc_db[240];
        trine_encode_shingle(near_a, strlen(near_a), enc_na);
        trine_encode_shingle(near_b, strlen(near_b), enc_nb);
        trine_encode_shingle(disj_a, strlen(disj_a), enc_da);
        trine_encode_shingle(disj_b, strlen(disj_b), enc_db);

        double reg_near = cosine_similarity(enc_na, enc_nb);
        double reg_disj = cosine_similarity(enc_da, enc_db);
        float  idf_near = trine_idf_cosine(enc_na, enc_nb, TRINE_IDF_WEIGHTS);
        float  idf_disj = trine_idf_cosine(enc_da, enc_db, TRINE_IDF_WEIGHTS);

        double reg_gap = reg_near - reg_disj;
        double idf_gap = (double)idf_near - (double)idf_disj;

        /* 23.1: IDF cosine for near-dup should be high (>= 0.5) */
        test_bool("IDF", 1,
                  idf_near >= 0.50f, (double)idf_near,
                  "IDF near-dup cosine >= 0.50");

        /* 23.2: IDF cosine for disjoint should be lower than near-dup */
        test_bool("IDF", 2,
                  idf_near > idf_disj, (double)idf_disj,
                  "IDF near > IDF disj");

        /* 23.3: IDF gap should be wider than regular gap
         * (IDF suppresses common n-grams, amplifying discrimination) */
        test_bool("IDF", 3,
                  idf_gap > reg_gap - 1e-6, idf_gap,
                  "IDF gap >= regular gap (gap widening)");

        /* 23.4: IDF and regular cosine should differ for near-dup
         * (IDF reweights channels, so scores should not be identical) */
        test_bool("IDF", 4,
                  fabs((double)idf_near - reg_near) > 1e-6, fabs((double)idf_near - reg_near),
                  "IDF != regular for near-dup (different weighting)");

        /* 23.5: IDF and regular cosine should differ for disjoint */
        test_bool("IDF", 5,
                  fabs((double)idf_disj - reg_disj) > 1e-6, fabs((double)idf_disj - reg_disj),
                  "IDF != regular for disjoint (different weighting)");
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 24: Canonicalization (10 tests)
     *
     * Tests for trine_canon.h/c deterministic text transforms:
     * whitespace normalization, timestamp stripping, UUID removal,
     * identifier normalization, number bucketing, and preset
     * compositions.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 24: Canonicalization");
    {
        char out[1024];
        size_t out_len;
        int rc;

        /* 24.1: Determinism — same input always produces same output */
        {
            const char *input = "ERROR 2024-01-15T10:30:00Z: db timeout after 30 seconds";
            char out1[256], out2[256];
            size_t len1, len2;
            trine_canon_apply(input, strlen(input), TRINE_CANON_SUPPORT, out1, sizeof(out1), &len1);
            trine_canon_apply(input, strlen(input), TRINE_CANON_SUPPORT, out2, sizeof(out2), &len2);
            int pass = (len1 == len2) && (memcmp(out1, out2, len1) == 0);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Canon #01  determinism\n"); }
            else { g_failed++; printf("  FAIL  Canon #01  determinism — outputs differ\n"); }
        }

        /* 24.2: Whitespace normalization */
        rc = trine_canon_apply("  hello   world  ", 17, TRINE_CANON_GENERAL, out, sizeof(out), &out_len);
        {
            int pass = (rc == 0) && (strcmp(out, "hello world") == 0);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Canon #02  whitespace: \"%s\"\n", out); }
            else { g_failed++; printf("  FAIL  Canon #02  whitespace: got \"%s\"\n", out); }
        }

        /* 24.3: Timestamp stripping (ISO-8601 date) */
        rc = trine_canon_apply("error on 2024-01-15 at server", 29, TRINE_CANON_SUPPORT, out, sizeof(out), &out_len);
        {
            int pass = (rc == 0) && (strstr(out, "2024") == NULL);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Canon #03  timestamp strip: \"%s\"\n", out); }
            else { g_failed++; printf("  FAIL  Canon #03  timestamp strip: got \"%s\"\n", out); }
        }

        /* 24.4: UUID stripping */
        rc = trine_canon_apply("req 550e8400-e29b-41d4-a716-446655440000 failed", 46, TRINE_CANON_SUPPORT, out, sizeof(out), &out_len);
        {
            int pass = (rc == 0) && (strstr(out, "550e8400") == NULL);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Canon #04  UUID strip: \"%s\"\n", out); }
            else { g_failed++; printf("  FAIL  Canon #04  UUID strip: got \"%s\"\n", out); }
        }

        /* 24.5: Number bucketing */
        rc = trine_canon_apply("error code 12345 on port 8080", 29, TRINE_CANON_SUPPORT, out, sizeof(out), &out_len);
        {
            int pass = (rc == 0) && (strstr(out, "12345") == NULL) && (strstr(out, "<N>") != NULL);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Canon #05  number bucket: \"%s\"\n", out); }
            else { g_failed++; printf("  FAIL  Canon #05  number bucket: got \"%s\"\n", out); }
        }

        /* 24.6: Identifier normalization — camelCase */
        rc = trine_canon_apply("calculateTotalAmount", 20, TRINE_CANON_CODE, out, sizeof(out), &out_len);
        {
            int pass = (rc == 0) && (strstr(out, "calculate") != NULL);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Canon #06  camelCase: \"%s\"\n", out); }
            else { g_failed++; printf("  FAIL  Canon #06  camelCase: got \"%s\"\n", out); }
        }

        /* 24.7: Identifier normalization — snake_case */
        rc = trine_canon_apply("calculate_total_amount", 22, TRINE_CANON_CODE, out, sizeof(out), &out_len);
        {
            int pass = (rc == 0) && (strchr(out, '_') == NULL);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Canon #07  snake_case: \"%s\"\n", out); }
            else { g_failed++; printf("  FAIL  Canon #07  snake_case: got \"%s\"\n", out); }
        }

        /* 24.8: NONE preset is passthrough */
        {
            const char *input = "  hello   world  ";
            rc = trine_canon_apply(input, strlen(input), TRINE_CANON_NONE, out, sizeof(out), &out_len);
            int pass = (rc == 0) && (strcmp(out, input) == 0);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Canon #08  NONE passthrough\n"); }
            else { g_failed++; printf("  FAIL  Canon #08  NONE passthrough: got \"%s\"\n", out); }
        }

        /* 24.9: Preset name lookup */
        {
            int pass = (strcmp(trine_canon_preset_name(TRINE_CANON_SUPPORT), "SUPPORT") == 0) &&
                       (strcmp(trine_canon_preset_name(TRINE_CANON_CODE), "CODE") == 0) &&
                       (strcmp(trine_canon_preset_name(TRINE_CANON_POLICY), "POLICY") == 0) &&
                       (strcmp(trine_canon_preset_name(TRINE_CANON_GENERAL), "GENERAL") == 0) &&
                       (strcmp(trine_canon_preset_name(TRINE_CANON_NONE), "NONE") == 0);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Canon #09  preset names\n"); }
            else { g_failed++; printf("  FAIL  Canon #09  preset names\n"); }
        }

        /* 24.10: Canon improves near-dup detection —
         * Two log lines with different timestamps should be MORE similar
         * after SUPPORT canonicalization than before. */
        {
            const char *log_a = "2024-01-15T10:30:00Z ERROR: db timeout after 30 seconds";
            const char *log_b = "2024-03-22T14:15:45Z ERROR: db timeout after 30 seconds";
            char ca[256], cb[256];
            size_t ca_len, cb_len;
            trine_canon_apply(log_a, strlen(log_a), TRINE_CANON_SUPPORT, ca, sizeof(ca), &ca_len);
            trine_canon_apply(log_b, strlen(log_b), TRINE_CANON_SUPPORT, cb, sizeof(cb), &cb_len);

            uint8_t enc_ra[240], enc_rb[240], enc_ca[240], enc_cb[240];
            trine_encode_shingle(log_a, strlen(log_a), enc_ra);
            trine_encode_shingle(log_b, strlen(log_b), enc_rb);
            trine_encode_shingle(ca, ca_len, enc_ca);
            trine_encode_shingle(cb, cb_len, enc_cb);

            double raw_sim = cosine_similarity(enc_ra, enc_rb);
            double canon_sim = cosine_similarity(enc_ca, enc_cb);
            int pass = (canon_sim >= raw_sim - 1e-6);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Canon #10  improvement: raw=%.3f canon=%.3f\n", raw_sim, canon_sim); }
            else { g_failed++; printf("  FAIL  Canon #10  improvement: raw=%.3f canon=%.3f (expected canon >= raw)\n", raw_sim, canon_sim); }
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 25: Format Round-Trip (5 tests)
     *
     * Tests for index save/load with v2 format hardening:
     * checksums, endianness markers, backward compatibility.
     * ───────────────────────────────────────────────────────────────── */
    print_header("Category 25: Format Round-Trip");
    {
        /* 25.1: Stage-1 index save/load round-trip */
        {
            trine_s1_config_t cfg = TRINE_S1_CONFIG_DEFAULT;
            trine_s1_index_t *idx = trine_s1_index_create(&cfg);
            uint8_t emb1[240], emb2[240], emb3[240];
            trine_s1_encode("hello world", 11, emb1);
            trine_s1_encode("goodbye world", 13, emb2);
            trine_s1_encode("test string", 11, emb3);
            trine_s1_index_add(idx, emb1, "tag1");
            trine_s1_index_add(idx, emb2, "tag2");
            trine_s1_index_add(idx, emb3, "tag3");
            int save_rc = trine_s1_index_save(idx, "/tmp/trine_test_v2.tridx");
            trine_s1_index_free(idx);

            trine_s1_index_t *loaded = trine_s1_index_load("/tmp/trine_test_v2.tridx");
            int pass = (save_rc == 0) && (loaded != NULL) && (trine_s1_index_count(loaded) == 3);
            if (loaded) {
                /* Verify tags round-tripped */
                const char *t1 = trine_s1_index_tag(loaded, 0);
                const char *t2 = trine_s1_index_tag(loaded, 1);
                if (!t1 || strcmp(t1, "tag1") != 0) pass = 0;
                if (!t2 || strcmp(t2, "tag2") != 0) pass = 0;
                trine_s1_index_free(loaded);
            }
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Format #01  .tridx v2 round-trip\n"); }
            else { g_failed++; printf("  FAIL  Format #01  .tridx v2 round-trip\n"); }
            remove("/tmp/trine_test_v2.tridx");
        }

        /* 25.2: Query produces same results after reload */
        {
            trine_s1_config_t cfg = TRINE_S1_CONFIG_DEFAULT;
            trine_s1_index_t *idx = trine_s1_index_create(&cfg);
            uint8_t emb[240];
            trine_s1_encode("specific unique text", 20, emb);
            trine_s1_index_add(idx, emb, "unique");
            trine_s1_encode("other text entirely", 19, emb);
            trine_s1_index_add(idx, emb, "other");
            trine_s1_index_save(idx, "/tmp/trine_test_query.tridx");
            trine_s1_index_free(idx);

            trine_s1_index_t *loaded = trine_s1_index_load("/tmp/trine_test_query.tridx");
            uint8_t query[240];
            trine_s1_encode("specific unique text", 20, query);
            trine_s1_result_t res = trine_s1_index_query(loaded, query);
            int pass = (loaded != NULL) && (res.is_duplicate == 1) && (res.matched_index == 0);
            if (loaded) trine_s1_index_free(loaded);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Format #02  query after reload\n"); }
            else { g_failed++; printf("  FAIL  Format #02  query after reload (match_idx=%d)\n", res.matched_index); }
            remove("/tmp/trine_test_query.tridx");
        }

        /* 25.3: Empty index save/load */
        {
            trine_s1_config_t cfg = TRINE_S1_CONFIG_DEFAULT;
            trine_s1_index_t *idx = trine_s1_index_create(&cfg);
            trine_s1_index_save(idx, "/tmp/trine_test_empty.tridx");
            trine_s1_index_free(idx);

            trine_s1_index_t *loaded = trine_s1_index_load("/tmp/trine_test_empty.tridx");
            int pass = (loaded != NULL) && (trine_s1_index_count(loaded) == 0);
            if (loaded) trine_s1_index_free(loaded);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Format #03  empty index round-trip\n"); }
            else { g_failed++; printf("  FAIL  Format #03  empty index round-trip\n"); }
            remove("/tmp/trine_test_empty.tridx");
        }

        /* 25.4: Encoding determinism through canon+encode pipeline */
        {
            const char *input = "calculateTotalAmount on 2024-01-15";
            char buf1[256], buf2[256];
            size_t len1, len2;
            trine_canon_apply(input, strlen(input), TRINE_CANON_CODE, buf1, sizeof(buf1), &len1);
            trine_canon_apply(input, strlen(input), TRINE_CANON_CODE, buf2, sizeof(buf2), &len2);
            uint8_t enc1[240], enc2[240];
            trine_s1_encode(buf1, len1, enc1);
            trine_s1_encode(buf2, len2, enc2);
            int pass = (memcmp(enc1, enc2, 240) == 0);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Format #04  canon+encode determinism\n"); }
            else { g_failed++; printf("  FAIL  Format #04  canon+encode determinism\n"); }
        }

        /* 25.5: Buffer-too-small rejection */
        {
            char tiny[4];
            size_t out_len;
            int rc = trine_canon_apply("hello world", 11, TRINE_CANON_GENERAL, tiny, sizeof(tiny), &out_len);
            int pass = (rc == -1);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Format #05  buffer overflow rejection\n"); }
            else { g_failed++; printf("  FAIL  Format #05  buffer overflow rejection (rc=%d)\n", rc); }
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 26: CS-IDF — Unit Tests
     * ───────────────────────────────────────────────────────────────── */
    printf("\n--- Category 26: CS-IDF Unit ---\n");
    {
        /* 26.1: Init zeroes all state */
        {
            trine_csidf_t csidf;
            trine_csidf_init(&csidf);
            int pass = (csidf.doc_count == 0) && (csidf.computed == 0);
            for (int i = 0; i < TRINE_CSIDF_DIMS && pass; i++) {
                if (csidf.channel_df[i] != 0 || csidf.weights[i] != 0.0f)
                    pass = 0;
            }
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF #01  init zeroes state\n"); }
            else { g_failed++; printf("  FAIL  CSIDF #01  init zeroes state\n"); }
        }

        /* 26.2: Observe increments doc_count and channel_df */
        {
            trine_csidf_t csidf;
            trine_csidf_init(&csidf);
            uint8_t emb[240];
            trine_encode_shingle("hello world", 11, emb);
            trine_csidf_observe(&csidf, emb);
            int pass = (csidf.doc_count == 1) && (csidf.computed == 0);
            /* At least some channels should be non-zero */
            int nonzero_df = 0;
            for (int i = 0; i < TRINE_CSIDF_DIMS; i++) {
                if (csidf.channel_df[i] > 0) nonzero_df++;
            }
            if (nonzero_df == 0) pass = 0;
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF #02  observe increments (df_nonzero=%d)\n", nonzero_df); }
            else { g_failed++; printf("  FAIL  CSIDF #02  observe increments (doc=%u, df_nz=%d)\n", csidf.doc_count, nonzero_df); }
        }

        /* 26.3: Compute produces valid weights in [0.01, 1.0] */
        {
            trine_csidf_t csidf;
            trine_csidf_init(&csidf);
            const char *docs[] = {"hello world", "foo bar baz", "quick brown fox",
                                  "lazy dog jumped", "test document"};
            uint8_t emb[240];
            for (int d = 0; d < 5; d++) {
                trine_encode_shingle(docs[d], strlen(docs[d]), emb);
                trine_csidf_observe(&csidf, emb);
            }
            int rc = trine_csidf_compute(&csidf);
            int pass = (rc == 0) && (csidf.computed == 1);
            float min_w = 2.0f, max_w = -1.0f;
            for (int i = 0; i < TRINE_CSIDF_DIMS; i++) {
                if (csidf.weights[i] < min_w) min_w = csidf.weights[i];
                if (csidf.weights[i] > max_w) max_w = csidf.weights[i];
            }
            if (min_w < TRINE_CSIDF_MIN_WEIGHT - 1e-6f || max_w > 1.0f + 1e-6f)
                pass = 0;
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF #03  compute weights in [%.2f, %.2f]\n", min_w, max_w); }
            else { g_failed++; printf("  FAIL  CSIDF #03  compute weights range [%.4f, %.4f]\n", min_w, max_w); }
        }

        /* 26.4: Compute returns -1 on zero doc_count */
        {
            trine_csidf_t csidf;
            trine_csidf_init(&csidf);
            int rc = trine_csidf_compute(&csidf);
            int pass = (rc == -1);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF #04  compute rejects zero docs\n"); }
            else { g_failed++; printf("  FAIL  CSIDF #04  compute rejects zero docs (rc=%d)\n", rc); }
        }

        /* 26.5: Merge adds doc_count and channel_df */
        {
            trine_csidf_t a, b;
            trine_csidf_init(&a);
            trine_csidf_init(&b);
            uint8_t emb[240];
            trine_encode_shingle("alpha beta", 10, emb);
            trine_csidf_observe(&a, emb);
            trine_encode_shingle("gamma delta", 11, emb);
            trine_csidf_observe(&a, emb);
            trine_encode_shingle("epsilon zeta", 12, emb);
            trine_csidf_observe(&b, emb);
            int rc = trine_csidf_merge(&a, &b);
            int pass = (rc == 0) && (a.doc_count == 3) && (a.computed == 0);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF #05  merge adds counts\n"); }
            else { g_failed++; printf("  FAIL  CSIDF #05  merge (doc=%u, computed=%d)\n", a.doc_count, a.computed); }
        }

        /* 26.6: Serialize/deserialize round-trip */
        {
            trine_csidf_t orig, loaded;
            trine_csidf_init(&orig);
            uint8_t emb[240];
            const char *docs[] = {"round trip test one", "round trip test two",
                                  "completely different"};
            for (int d = 0; d < 3; d++) {
                trine_encode_shingle(docs[d], strlen(docs[d]), emb);
                trine_csidf_observe(&orig, emb);
            }
            trine_csidf_compute(&orig);

            FILE *fp = fopen("/tmp/trine_test_csidf.bin", "wb");
            int pass = (fp != NULL);
            if (fp) {
                pass = pass && (trine_csidf_write(&orig, fp) == 0);
                fclose(fp);
            }
            if (pass) {
                fp = fopen("/tmp/trine_test_csidf.bin", "rb");
                pass = (fp != NULL);
                if (fp) {
                    trine_csidf_init(&loaded);
                    pass = pass && (trine_csidf_read(&loaded, fp) == 0);
                    fclose(fp);
                }
            }
            if (pass) {
                pass = (loaded.doc_count == orig.doc_count) && (loaded.computed == 1);
                for (int i = 0; i < TRINE_CSIDF_DIMS && pass; i++) {
                    if (loaded.channel_df[i] != orig.channel_df[i]) pass = 0;
                    if (fabsf(loaded.weights[i] - orig.weights[i]) > 1e-6f) pass = 0;
                }
            }
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF #06  serialize round-trip\n"); }
            else { g_failed++; printf("  FAIL  CSIDF #06  serialize round-trip\n"); }
            remove("/tmp/trine_test_csidf.bin");
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 27: CS-IDF — Scoring Tests
     * ───────────────────────────────────────────────────────────────── */
    printf("\n--- Category 27: CS-IDF Scoring ---\n");
    {
        /* Build a CS-IDF from a small corpus where "the" channels are common */
        trine_csidf_t csidf;
        trine_csidf_init(&csidf);
        const char *corpus[] = {
            "the quick brown fox jumped over the lazy dog",
            "the cat sat on the mat in the house",
            "the rain in spain falls mainly on the plain",
            "a completely unique and special phrase here",
            "mathematical algorithms for sorting data",
            "quantum computing research paper abstract",
            "the old man and the sea by hemingway",
            "the great gatsby by fitzgerald chapter one"
        };
        uint8_t corpus_emb[8][240];
        for (int d = 0; d < 8; d++) {
            trine_encode_shingle(corpus[d], strlen(corpus[d]), corpus_emb[d]);
            trine_csidf_observe(&csidf, corpus_emb[d]);
        }
        trine_csidf_compute(&csidf);

        /* 27.1: CS-IDF cosine of identical vectors = 1.0 */
        {
            float sim = trine_csidf_cosine(corpus_emb[0], corpus_emb[0], &csidf);
            int pass = (fabsf(sim - 1.0f) < 1e-5f);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF-Score #01  identity = %.6f\n", sim); }
            else { g_failed++; printf("  FAIL  CSIDF-Score #01  identity = %.6f (expected 1.0)\n", sim); }
        }

        /* 27.2: CS-IDF cosine between similar texts is positive */
        {
            uint8_t a[240], b[240];
            trine_encode_shingle("the quick brown fox", 19, a);
            trine_encode_shingle("the quick brown dog", 19, b);
            float sim = trine_csidf_cosine(a, b, &csidf);
            int pass = (sim > 0.3f && sim < 1.0f);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF-Score #02  similar = %.6f\n", sim); }
            else { g_failed++; printf("  FAIL  CSIDF-Score #02  similar = %.6f\n", sim); }
        }

        /* 27.3: CS-IDF + lens cosine produces valid output */
        {
            float lens[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            float sim = trine_csidf_cosine_lens(corpus_emb[0], corpus_emb[0],
                                                  &csidf, lens);
            int pass = (fabsf(sim - 1.0f) < 1e-5f);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF-Score #03  lens identity = %.6f\n", sim); }
            else { g_failed++; printf("  FAIL  CSIDF-Score #03  lens identity = %.6f\n", sim); }
        }

        /* 27.4: CS-IDF returns 0 when not computed */
        {
            trine_csidf_t uncomputed;
            trine_csidf_init(&uncomputed);
            float sim = trine_csidf_cosine(corpus_emb[0], corpus_emb[1], &uncomputed);
            int pass = (sim == 0.0f);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF-Score #04  uncomputed returns 0\n"); }
            else { g_failed++; printf("  FAIL  CSIDF-Score #04  uncomputed returns %.6f\n", sim); }
        }

        /* 27.5: CS-IDF downweights common channels — sim of common text pair
         * should differ from uniform-weight sim */
        {
            float idf_sim = trine_csidf_cosine(corpus_emb[0], corpus_emb[1], &csidf);
            /* Compute uniform-weight cosine for comparison */
            float uni_sim = (float)cosine_similarity(corpus_emb[0], corpus_emb[1]);
            int pass = (fabsf(idf_sim - uni_sim) > 0.001f); /* Should differ */
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  CSIDF-Score #05  IDF differs from uniform (idf=%.4f, uni=%.4f)\n", idf_sim, uni_sim); }
            else { g_failed++; printf("  FAIL  CSIDF-Score #05  IDF same as uniform (idf=%.4f, uni=%.4f)\n", idf_sim, uni_sim); }
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 28: Field-Aware — Unit Tests
     * ───────────────────────────────────────────────────────────────── */
    printf("\n--- Category 28: Field-Aware Unit ---\n");
    {
        /* 28.1: Config init sets 3 fields with uniform weights */
        {
            trine_field_config_t cfg;
            trine_field_config_init(&cfg);
            int pass = (cfg.field_count == 3) &&
                       (strcmp(cfg.field_names[0], "title") == 0) &&
                       (strcmp(cfg.field_names[1], "body") == 0) &&
                       (strcmp(cfg.field_names[2], "code") == 0) &&
                       (cfg.field_weights[0] == 1.0f) &&
                       (cfg.field_weights[1] == 1.0f) &&
                       (cfg.field_weights[2] == 1.0f) &&
                       (cfg.route_field == TRINE_FIELD_BODY);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Field #01  config init defaults\n"); }
            else { g_failed++; printf("  FAIL  Field #01  config init defaults\n"); }
        }

        /* 28.2: Preset CODE sets expected weights */
        {
            trine_field_config_t cfg;
            int rc = trine_field_config_preset(&cfg, TRINE_FIELD_PRESET_CODE);
            int pass = (rc == 0) &&
                       (cfg.field_weights[0] == 1.0f) &&
                       (fabsf(cfg.field_weights[1] - 0.3f) < 1e-6f) &&
                       (fabsf(cfg.field_weights[2] - 1.2f) < 1e-6f);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Field #02  CODE preset\n"); }
            else { g_failed++; printf("  FAIL  Field #02  CODE preset\n"); }
        }

        /* 28.3: Preset SUPPORT sets expected weights */
        {
            trine_field_config_t cfg;
            int rc = trine_field_config_preset(&cfg, TRINE_FIELD_PRESET_SUPPORT);
            int pass = (rc == 0) &&
                       (cfg.field_weights[0] == 1.0f) &&
                       (fabsf(cfg.field_weights[1] - 0.8f) < 1e-6f) &&
                       (fabsf(cfg.field_weights[2] - 0.3f) < 1e-6f);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Field #03  SUPPORT preset\n"); }
            else { g_failed++; printf("  FAIL  Field #03  SUPPORT preset\n"); }
        }

        /* 28.4: Parse fields spec "title,body" */
        {
            trine_field_config_t cfg;
            memset(&cfg, 0, sizeof(cfg));
            int rc = trine_field_config_parse_fields(&cfg, "title,body");
            int pass = (rc == 0) && (cfg.field_count == 2) &&
                       (strcmp(cfg.field_names[0], "title") == 0) &&
                       (strcmp(cfg.field_names[1], "body") == 0);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Field #04  parse fields spec\n"); }
            else { g_failed++; printf("  FAIL  Field #04  parse fields spec (fc=%d)\n", cfg.field_count); }
        }

        /* 28.5: Parse weights spec "title=1.5,body=0.5" */
        {
            trine_field_config_t cfg;
            trine_field_config_init(&cfg);
            int rc = trine_field_config_parse_weights(&cfg, "title=1.5,body=0.5");
            int pass = (rc == 0) &&
                       (fabsf(cfg.field_weights[0] - 1.5f) < 1e-6f) &&
                       (fabsf(cfg.field_weights[1] - 0.5f) < 1e-6f);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Field #05  parse weight spec\n"); }
            else { g_failed++; printf("  FAIL  Field #05  parse weight spec\n"); }
        }

        /* 28.6: Parse "auto" resets to default */
        {
            trine_field_config_t cfg;
            memset(&cfg, 0xFF, sizeof(cfg));
            int rc = trine_field_config_parse_fields(&cfg, "auto");
            int pass = (rc == 0) && (cfg.field_count == 3);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Field #06  parse auto\n"); }
            else { g_failed++; printf("  FAIL  Field #06  parse auto (fc=%d)\n", cfg.field_count); }
        }

        /* 28.7: Field config serialize/deserialize round-trip */
        {
            trine_field_config_t orig, loaded;
            trine_field_config_init(&orig);
            orig.field_weights[0] = 1.5f;
            orig.field_weights[2] = 0.7f;

            FILE *fp = fopen("/tmp/trine_test_field_cfg.bin", "wb");
            int pass = (fp != NULL);
            if (fp) {
                pass = pass && (trine_field_config_write(&orig, fp) == 0);
                fclose(fp);
            }
            if (pass) {
                fp = fopen("/tmp/trine_test_field_cfg.bin", "rb");
                pass = (fp != NULL);
                if (fp) {
                    memset(&loaded, 0, sizeof(loaded));
                    pass = pass && (trine_field_config_read(&loaded, fp) == 0);
                    fclose(fp);
                }
            }
            if (pass) {
                pass = (loaded.field_count == orig.field_count) &&
                       (loaded.route_field == orig.route_field);
                for (int i = 0; i < TRINE_FIELD_MAX && pass; i++) {
                    if (strcmp(loaded.field_names[i], orig.field_names[i]) != 0) pass = 0;
                    if (fabsf(loaded.field_weights[i] - orig.field_weights[i]) > 1e-6f) pass = 0;
                }
            }
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Field #07  config round-trip\n"); }
            else { g_failed++; printf("  FAIL  Field #07  config round-trip\n"); }
            remove("/tmp/trine_test_field_cfg.bin");
        }

        /* 28.8: JSONL field extraction */
        {
            trine_field_config_t cfg;
            trine_field_config_init(&cfg);
            const char *jsonl = "{\"title\": \"Bug Report\", \"body\": \"App crashes on login\", \"id\": \"issue-42\"}";
            const char *texts[TRINE_FIELD_MAX];
            size_t lens[TRINE_FIELD_MAX];
            char *tag = NULL;
            int found = trine_field_extract_jsonl(jsonl, strlen(jsonl), &cfg,
                                                    texts, lens, &tag);
            int pass = (found >= 2) &&
                       (texts[0] != NULL) && (lens[0] == 10) && /* "Bug Report" */
                       (texts[1] != NULL) && (lens[1] == 20) && /* "App crashes on login" */
                       (tag != NULL) && (strcmp(tag, "issue-42") == 0);
            if (tag) free(tag);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Field #08  JSONL extraction (found=%d)\n", found); }
            else { g_failed++; printf("  FAIL  Field #08  JSONL extraction (found=%d)\n", found); }
        }

        /* 28.9: JSONL fallback to "text" field */
        {
            trine_field_config_t cfg;
            trine_field_config_init(&cfg);
            const char *jsonl = "{\"text\": \"fallback content\", \"id\": \"doc1\"}";
            const char *texts[TRINE_FIELD_MAX];
            size_t lens[TRINE_FIELD_MAX];
            char *tag = NULL;
            int found = trine_field_extract_jsonl(jsonl, strlen(jsonl), &cfg,
                                                    texts, lens, &tag);
            int pass = (found == 1) && (texts[1] != NULL) && /* body slot */
                       (lens[1] == 16); /* "fallback content" */
            if (tag) free(tag);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Field #09  JSONL text fallback\n"); }
            else { g_failed++; printf("  FAIL  Field #09  JSONL text fallback (found=%d)\n", found); }
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 29: Field-Aware — Scoring Tests
     * ───────────────────────────────────────────────────────────────── */
    printf("\n--- Category 29: Field-Aware Scoring ---\n");
    {
        /* Build two field entries with known content.
         * Use 2-field config (title+body) so zeroed code doesn't affect scoring. */
        trine_field_config_t cfg;
        trine_field_config_init(&cfg);
        cfg.field_count = 2;  /* title + body only */

        trine_field_entry_t a, b;
        const char *a_texts[] = {"Bug Report", "App crashes on login"};
        size_t a_lens[] = {10, 20};
        const char *b_texts[] = {"Bug Report", "App crashes on startup"};
        size_t b_lens[] = {10, 22};
        trine_field_encode(&cfg, a_texts, a_lens, &a);
        trine_field_encode(&cfg, b_texts, b_lens, &b);

        /* 29.1: Identical field entries = 1.0 */
        {
            float sim = trine_field_cosine(&a, &a, &cfg);
            int pass = (fabsf(sim - 1.0f) < 1e-4f);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  FieldScore #01  identity = %.6f\n", sim); }
            else { g_failed++; printf("  FAIL  FieldScore #01  identity = %.6f\n", sim); }
        }

        /* 29.2: Similar field entries have high similarity */
        {
            float sim = trine_field_cosine(&a, &b, &cfg);
            int pass = (sim > 0.5f && sim < 1.0f);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  FieldScore #02  similar = %.6f\n", sim); }
            else { g_failed++; printf("  FAIL  FieldScore #02  similar = %.6f\n", sim); }
        }

        /* 29.3: Field weights affect scoring — title-only vs body-only */
        {
            trine_field_config_t title_cfg, body_cfg;
            trine_field_config_init(&title_cfg);
            trine_field_config_init(&body_cfg);
            /* Title-only: weight title=1.0, body=0.0, code=0.0 */
            title_cfg.field_weights[0] = 1.0f;
            title_cfg.field_weights[1] = 0.0f;
            title_cfg.field_weights[2] = 0.0f;
            /* Body-only: weight title=0.0, body=1.0, code=0.0 */
            body_cfg.field_weights[0] = 0.0f;
            body_cfg.field_weights[1] = 1.0f;
            body_cfg.field_weights[2] = 0.0f;

            float title_sim = trine_field_cosine(&a, &b, &title_cfg);
            float body_sim = trine_field_cosine(&a, &b, &body_cfg);
            /* Title is identical ("Bug Report"), body differs — title_sim should be higher */
            int pass = (title_sim > body_sim);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  FieldScore #03  title=%.4f > body=%.4f\n", title_sim, body_sim); }
            else { g_failed++; printf("  FAIL  FieldScore #03  title=%.4f vs body=%.4f\n", title_sim, body_sim); }
        }

        /* 29.4: Field cosine with IDF weights */
        {
            /* Build dummy IDF weights (uniform 1.0) */
            float idf[240];
            for (int i = 0; i < 240; i++) idf[i] = 1.0f;
            float sim_idf = trine_field_cosine_idf(&a, &b, &cfg, idf);
            float sim_std = trine_field_cosine(&a, &b, &cfg);
            /* With uniform IDF, should be very close to standard */
            int pass = (fabsf(sim_idf - sim_std) < 0.05f);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  FieldScore #04  IDF uniform ~ standard (%.4f vs %.4f)\n", sim_idf, sim_std); }
            else { g_failed++; printf("  FAIL  FieldScore #04  IDF=%.4f vs std=%.4f\n", sim_idf, sim_std); }
        }

        /* 29.5: Route embedding selects the right field */
        {
            const uint8_t *route_emb = trine_field_route_embedding(&cfg, &a);
            int pass = (route_emb != NULL) &&
                       (memcmp(route_emb, a.embeddings[TRINE_FIELD_BODY], 240) == 0);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  FieldScore #05  route embedding = body field\n"); }
            else { g_failed++; printf("  FAIL  FieldScore #05  route embedding mismatch\n"); }
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 30: Routed Index — CS-IDF + Field Round-Trip
     * ───────────────────────────────────────────────────────────────── */
    printf("\n--- Category 30: Route v4 Round-Trip ---\n");
    {
        /* 30.1: Routed index with CS-IDF save/load round-trip */
        {
            trine_s1_config_t cfg = TRINE_S1_CONFIG_DEFAULT;
            trine_route_t *rt = trine_route_create(&cfg);
            trine_route_enable_csidf(rt);

            uint8_t emb[240];
            trine_encode_shingle("alpha document one", 18, emb);
            trine_route_add(rt, emb, "doc1");
            trine_encode_shingle("beta document two", 17, emb);
            trine_route_add(rt, emb, "doc2");
            trine_encode_shingle("gamma document three", 20, emb);
            trine_route_add(rt, emb, "doc3");
            trine_route_compute_csidf(rt);

            int save_rc = trine_route_save(rt, "/tmp/trine_test_csidf_rt.trrt");
            trine_route_free(rt);

            trine_route_t *loaded = trine_route_load("/tmp/trine_test_csidf_rt.trrt");
            int pass = (save_rc == 0) && (loaded != NULL) &&
                       (trine_route_count(loaded) == 3);

            /* Verify CS-IDF was loaded */
            if (pass) {
                const trine_csidf_t *cs = trine_route_get_csidf(loaded);
                pass = (cs != NULL) && (cs->doc_count == 3) && (cs->computed == 1);
            }

            /* Verify tags */
            if (pass) {
                const char *t1 = trine_route_tag(loaded, 0);
                const char *t2 = trine_route_tag(loaded, 1);
                if (!t1 || strcmp(t1, "doc1") != 0) pass = 0;
                if (!t2 || strcmp(t2, "doc2") != 0) pass = 0;
            }

            if (loaded) trine_route_free(loaded);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Route-v4 #01  CS-IDF round-trip\n"); }
            else { g_failed++; printf("  FAIL  Route-v4 #01  CS-IDF round-trip\n"); }
            remove("/tmp/trine_test_csidf_rt.trrt");
        }

        /* 30.2: CS-IDF query produces valid results after load */
        {
            trine_s1_config_t cfg = TRINE_S1_CONFIG_DEFAULT;
            trine_route_t *rt = trine_route_create(&cfg);
            trine_route_enable_csidf(rt);

            uint8_t emb[240];
            trine_encode_shingle("specific unique text alpha", 26, emb);
            trine_route_add(rt, emb, "unique");
            trine_encode_shingle("something else entirely different", 33, emb);
            trine_route_add(rt, emb, "other");
            trine_route_compute_csidf(rt);

            trine_route_save(rt, "/tmp/trine_test_csidf_q.trrt");
            trine_route_free(rt);

            trine_route_t *loaded = trine_route_load("/tmp/trine_test_csidf_q.trrt");
            trine_encode_shingle("specific unique text alpha", 26, emb);
            trine_route_stats_t stats;
            trine_s1_result_t res = trine_route_query_csidf(loaded, emb, &stats);
            int pass = (loaded != NULL) && (res.is_duplicate == 1) &&
                       (res.matched_index == 0);
            if (loaded) trine_route_free(loaded);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Route-v4 #02  CS-IDF query after load (sim=%.4f)\n", res.similarity); }
            else { g_failed++; printf("  FAIL  Route-v4 #02  CS-IDF query after load (dup=%d, idx=%d)\n", res.is_duplicate, res.matched_index); }
            remove("/tmp/trine_test_csidf_q.trrt");
        }

        /* 30.3: Routed index with fields save/load round-trip */
        {
            trine_s1_config_t s1cfg = TRINE_S1_CONFIG_DEFAULT;
            trine_route_t *rt = trine_route_create(&s1cfg);
            trine_field_config_t fcfg;
            trine_field_config_init(&fcfg);
            trine_route_enable_fields(rt, &fcfg);

            trine_field_entry_t entry;
            const char *t1[] = {"Login Bug", "App crashes at login screen", NULL};
            size_t l1[] = {9, 27, 0};
            trine_field_encode(&fcfg, t1, l1, &entry);
            trine_route_add_fields(rt, &entry, "bug-1");

            const char *t2[] = {"Feature Request", "Add dark mode to settings", NULL};
            size_t l2[] = {15, 25, 0};
            trine_field_encode(&fcfg, t2, l2, &entry);
            trine_route_add_fields(rt, &entry, "feat-1");

            int save_rc = trine_route_save(rt, "/tmp/trine_test_field_rt.trrt");
            trine_route_free(rt);

            trine_route_t *loaded = trine_route_load("/tmp/trine_test_field_rt.trrt");
            int pass = (save_rc == 0) && (loaded != NULL) &&
                       (trine_route_count(loaded) == 2);

            /* Verify field config was loaded */
            if (pass) {
                const trine_field_config_t *lfc = trine_route_field_config(loaded);
                pass = (lfc != NULL) && (lfc->field_count == 3);
            }

            /* Verify tags */
            if (pass) {
                const char *tag0 = trine_route_tag(loaded, 0);
                if (!tag0 || strcmp(tag0, "bug-1") != 0) pass = 0;
            }

            if (loaded) trine_route_free(loaded);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Route-v4 #03  field-aware round-trip\n"); }
            else { g_failed++; printf("  FAIL  Route-v4 #03  field-aware round-trip\n"); }
            remove("/tmp/trine_test_field_rt.trrt");
        }

        /* 30.4: Field-aware query finds best match */
        {
            trine_s1_config_t s1cfg = TRINE_S1_CONFIG_DEFAULT;
            s1cfg.threshold = 0.30f;  /* Lower threshold for field-weighted scoring */
            trine_route_t *rt = trine_route_create(&s1cfg);
            trine_field_config_t fcfg;
            trine_field_config_init(&fcfg);
            fcfg.field_count = 2;  /* title + body only (no empty code field) */
            trine_route_enable_fields(rt, &fcfg);

            trine_field_entry_t entry;
            const char *t1[] = {"Login Bug", "Application crashes at login screen"};
            size_t l1[] = {9, 35};
            trine_field_encode(&fcfg, t1, l1, &entry);
            trine_route_add_fields(rt, &entry, "bug-1");

            const char *t2[] = {"Dark Mode", "Add dark mode feature to settings page"};
            size_t l2[] = {9, 38};
            trine_field_encode(&fcfg, t2, l2, &entry);
            trine_route_add_fields(rt, &entry, "feat-1");

            /* Query with something similar to bug-1 */
            trine_field_entry_t query;
            const char *qt[] = {"Login Bug", "Application crashes during login"};
            size_t ql[] = {9, 32};
            trine_field_encode(&fcfg, qt, ql, &query);

            trine_route_stats_t stats;
            trine_s1_result_t res = trine_route_query_fields(rt, &query, &stats);
            int pass = (res.matched_index == 0); /* Should match bug-1 */
            trine_route_free(rt);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Route-v4 #04  field query match (sim=%.4f)\n", res.similarity); }
            else { g_failed++; printf("  FAIL  Route-v4 #04  field query match (idx=%d, sim=%.4f)\n", res.matched_index, res.similarity); }
        }

        /* 30.5: Atomic save works correctly */
        {
            trine_s1_config_t cfg = TRINE_S1_CONFIG_DEFAULT;
            trine_route_t *rt = trine_route_create(&cfg);
            uint8_t emb[240];
            trine_encode_shingle("atomic save test document", 25, emb);
            trine_route_add(rt, emb, "atomic");
            int rc = trine_route_save_atomic(rt, "/tmp/trine_test_atomic.trrt");
            trine_route_free(rt);

            trine_route_t *loaded = trine_route_load("/tmp/trine_test_atomic.trrt");
            int pass = (rc == 0) && (loaded != NULL) && (trine_route_count(loaded) == 1);
            if (pass) {
                const char *tag = trine_route_tag(loaded, 0);
                if (!tag || strcmp(tag, "atomic") != 0) pass = 0;
            }
            if (loaded) trine_route_free(loaded);
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Route-v4 #05  atomic save\n"); }
            else { g_failed++; printf("  FAIL  Route-v4 #05  atomic save\n"); }
            remove("/tmp/trine_test_atomic.trrt");
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Category 31: Append Mode — Build Half + Append Half
     * ───────────────────────────────────────────────────────────────── */
    printf("\n--- Category 31: Append Mode ---\n");
    {
        /* 31.1: Build half, save, load, append half, verify all present */
        {
            trine_s1_config_t cfg = TRINE_S1_CONFIG_DEFAULT;
            trine_route_t *rt = trine_route_create(&cfg);
            trine_route_enable_csidf(rt);

            /* Phase 1: add 3 docs */
            uint8_t emb[240];
            trine_encode_shingle("first document alpha", 20, emb);
            trine_route_add(rt, emb, "d1");
            trine_encode_shingle("second document beta", 20, emb);
            trine_route_add(rt, emb, "d2");
            trine_encode_shingle("third document gamma", 20, emb);
            trine_route_add(rt, emb, "d3");
            trine_route_compute_csidf(rt);
            trine_route_save(rt, "/tmp/trine_test_append.trrt");
            trine_route_free(rt);

            /* Phase 2: load and append 2 more */
            rt = trine_route_load("/tmp/trine_test_append.trrt");
            int pass = (rt != NULL) && (trine_route_count(rt) == 3);

            if (pass) {
                /* Enable CS-IDF on loaded index (retroactive observe) */
                if (!trine_route_get_csidf(rt)) {
                    trine_route_enable_csidf(rt);
                }
                trine_encode_shingle("fourth document delta", 21, emb);
                trine_route_add(rt, emb, "d4");
                trine_encode_shingle("fifth document epsilon", 22, emb);
                trine_route_add(rt, emb, "d5");

                pass = (trine_route_count(rt) == 5);
            }

            /* Recompute CS-IDF with merged data */
            if (pass) {
                trine_route_compute_csidf(rt);
                const trine_csidf_t *cs = trine_route_get_csidf(rt);
                pass = (cs != NULL) && (cs->doc_count == 5);
            }

            /* Save atomically */
            if (pass) {
                pass = (trine_route_save_atomic(rt, "/tmp/trine_test_append.trrt") == 0);
            }
            if (rt) trine_route_free(rt);

            /* Phase 3: reload and verify everything */
            rt = trine_route_load("/tmp/trine_test_append.trrt");
            if (pass) {
                pass = (rt != NULL) && (trine_route_count(rt) == 5);
            }
            if (pass) {
                const char *t1 = trine_route_tag(rt, 0);
                const char *t5 = trine_route_tag(rt, 4);
                if (!t1 || strcmp(t1, "d1") != 0) pass = 0;
                if (!t5 || strcmp(t5, "d5") != 0) pass = 0;
            }
            if (pass) {
                const trine_csidf_t *cs = trine_route_get_csidf(rt);
                pass = (cs != NULL) && (cs->doc_count == 5);
            }
            /* Verify appended embedding survives round-trip */
            if (pass) {
                trine_encode_shingle("fourth document delta", 21, emb);
                const uint8_t *stored = trine_route_embedding(rt, 3);
                pass = (stored != NULL) && (memcmp(stored, emb, 240) == 0);
            }
            if (rt) trine_route_free(rt);

            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Append #01  build-half + append-half round-trip\n"); }
            else { g_failed++; printf("  FAIL  Append #01  build-half + append-half round-trip\n"); }
            remove("/tmp/trine_test_append.trrt");
        }

        /* 31.2: Append with fields — verify field data survives */
        {
            trine_s1_config_t s1cfg = TRINE_S1_CONFIG_DEFAULT;
            trine_route_t *rt = trine_route_create(&s1cfg);
            trine_field_config_t fcfg;
            trine_field_config_init(&fcfg);
            trine_route_enable_fields(rt, &fcfg);
            trine_route_enable_csidf(rt);

            trine_field_entry_t entry;
            const char *t1[] = {"Initial Bug", "First bug report description", NULL};
            size_t l1[] = {11, 28, 0};
            trine_field_encode(&fcfg, t1, l1, &entry);
            trine_route_add_fields(rt, &entry, "init-1");
            trine_route_compute_csidf(rt);

            trine_route_save(rt, "/tmp/trine_test_append_field.trrt");
            trine_route_free(rt);

            /* Load and append */
            rt = trine_route_load("/tmp/trine_test_append_field.trrt");
            int pass = (rt != NULL) && (trine_route_count(rt) == 1);

            if (pass) {
                const trine_field_config_t *lfc = trine_route_field_config(rt);
                pass = (lfc != NULL) && (lfc->field_count == 3);
            }

            /* Append another entry using loaded field config */
            if (pass) {
                const trine_field_config_t *lfc = trine_route_field_config(rt);
                const char *t2[] = {"New Feature", "Second feature request", NULL};
                size_t l2[] = {11, 22, 0};
                trine_field_encode(lfc, t2, l2, &entry);
                trine_route_add_fields(rt, &entry, "append-1");
                pass = (trine_route_count(rt) == 2);
            }

            if (pass) {
                pass = (trine_route_save_atomic(rt, "/tmp/trine_test_append_field.trrt") == 0);
            }
            if (rt) trine_route_free(rt);

            /* Verify final state */
            rt = trine_route_load("/tmp/trine_test_append_field.trrt");
            if (pass) {
                pass = (rt != NULL) && (trine_route_count(rt) == 2);
            }
            if (pass) {
                const char *tag0 = trine_route_tag(rt, 0);
                const char *tag1 = trine_route_tag(rt, 1);
                if (!tag0 || strcmp(tag0, "init-1") != 0) pass = 0;
                if (!tag1 || strcmp(tag1, "append-1") != 0) pass = 0;
            }
            if (rt) trine_route_free(rt);

            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Append #02  field-aware append round-trip\n"); }
            else { g_failed++; printf("  FAIL  Append #02  field-aware append round-trip\n"); }
            remove("/tmp/trine_test_append_field.trrt");
        }

        /* 31.3: CS-IDF merge equivalence — build all at once vs build+append */
        {
            /* Build all 4 docs at once */
            trine_csidf_t full;
            trine_csidf_init(&full);
            uint8_t embs[4][240];
            const char *texts[] = {"merge test alpha", "merge test beta",
                                   "merge test gamma", "merge test delta"};
            for (int i = 0; i < 4; i++) {
                trine_encode_shingle(texts[i], strlen(texts[i]), embs[i]);
                trine_csidf_observe(&full, embs[i]);
            }
            trine_csidf_compute(&full);

            /* Build first 2, then merge second 2 */
            trine_csidf_t half1, half2;
            trine_csidf_init(&half1);
            trine_csidf_init(&half2);
            for (int i = 0; i < 2; i++)
                trine_csidf_observe(&half1, embs[i]);
            for (int i = 2; i < 4; i++)
                trine_csidf_observe(&half2, embs[i]);
            trine_csidf_merge(&half1, &half2);
            trine_csidf_compute(&half1);

            /* Compare: merged should equal full */
            int pass = (half1.doc_count == full.doc_count);
            for (int i = 0; i < TRINE_CSIDF_DIMS && pass; i++) {
                if (half1.channel_df[i] != full.channel_df[i]) pass = 0;
                if (fabsf(half1.weights[i] - full.weights[i]) > 1e-6f) pass = 0;
            }
            g_total++; if (pass) { g_passed++; if (g_verbose) printf("  PASS  Append #03  merge equivalence (full vs split+merge)\n"); }
            else { g_failed++; printf("  FAIL  Append #03  merge equivalence\n"); }
        }
    }

    /* ─────────────────────────────────────────────────────────────────
     * Summary
     * ───────────────────────────────────────────────────────────────── */
    printf("\n═══════════════════════════════════════════\n");
    printf("  TOTAL:  %d\n", g_total);
    printf("  PASSED: %d\n", g_passed);
    printf("  FAILED: %d\n", g_failed);
    printf("═══════════════════════════════════════════\n");

    if (g_failed == 0)
        printf("\nAll tests passed.\n");
    else
        printf("\n%d test(s) FAILED.\n", g_failed);

    return g_failed;
}
