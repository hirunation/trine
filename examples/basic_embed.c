/* =====================================================================
 * basic_embed.c -- TRINE Stage-1 Embedding & Lens Comparison Demo
 * =====================================================================
 *
 * Demonstrates:
 *   - Encoding text into 240-dimensional ternary embeddings
 *   - Comparing embeddings with different lens presets
 *   - How lens selection affects similarity scoring
 *
 * Each lens emphasizes different embedding chains:
 *   UNIFORM  {1.0, 1.0, 1.0, 1.0}  -- Equal weight to all chains
 *   DEDUP    {0.5, 0.5, 0.7, 1.0}  -- Word-chain heavy (deduplication)
 *   EDIT     {1.0, 0.3, 0.1, 0.0}  -- Character-level focus (typo detection)
 *   CODE     {1.0, 0.8, 0.4, 0.2}  -- Structural + character (identifiers)
 *
 * Build (from project root):
 *   cc -O2 -Wall -Wextra -o basic_embed examples/basic_embed.c \
 *      src/encode/trine_encode.c src/compare/trine_stage1.c \
 *      -Isrc/encode -Isrc/compare -lm
 *
 * ===================================================================== */

#include <stdio.h>
#include <string.h>

#include "trine_stage1.h"

/* ---- Text pairs to compare ------------------------------------------ */

typedef struct {
    const char *label;
    const char *text_a;
    const char *text_b;
} text_pair_t;

static const text_pair_t PAIRS[] = {
    {
        "Greeting variation",
        "hello world",
        "hello there"
    },
    {
        "Animal swap",
        "the quick brown fox",
        "the quick brown dog"
    },
    {
        "Code identifier style",
        "function calculate_total",
        "function calculateTotal"
    },
    {
        "Invoice number diff",
        "invoice #12345",
        "invoice #12346"
    },
    {
        "Semantic overlap",
        "the server crashed at midnight",
        "the server went down at midnight"
    },
    {
        "Completely unrelated",
        "banana smoothie recipe",
        "quantum physics lecture"
    },
};

#define NUM_PAIRS ((int)(sizeof(PAIRS) / sizeof(PAIRS[0])))

/* ---- Lens definitions ----------------------------------------------- */

typedef struct {
    const char *name;
    trine_s1_lens_t lens;
} named_lens_t;

static const named_lens_t LENSES[] = {
    { "UNIFORM", TRINE_S1_LENS_UNIFORM },
    { "DEDUP",   TRINE_S1_LENS_DEDUP   },
    { "EDIT",    TRINE_S1_LENS_EDIT     },
    { "CODE",    TRINE_S1_LENS_CODE     },
};

#define NUM_LENSES ((int)(sizeof(LENSES) / sizeof(LENSES[0])))

/* ---- Helpers -------------------------------------------------------- */

static void print_separator(int width)
{
    for (int i = 0; i < width; i++)
        putchar('-');
    putchar('\n');
}

/* ---- Main ----------------------------------------------------------- */

int main(void)
{
    printf("TRINE Stage-1 Basic Embedding Demo\n");
    printf("===================================\n\n");

    /* Encode all texts up front */
    uint8_t emb_a[NUM_PAIRS][240];
    uint8_t emb_b[NUM_PAIRS][240];

    for (int i = 0; i < NUM_PAIRS; i++) {
        trine_s1_encode(PAIRS[i].text_a, strlen(PAIRS[i].text_a), emb_a[i]);
        trine_s1_encode(PAIRS[i].text_b, strlen(PAIRS[i].text_b), emb_b[i]);
    }

    /* Print fill ratios to show embedding density */
    printf("Embedding fill ratios (fraction of non-zero channels):\n");
    print_separator(70);
    printf("  %-35s  %8s  %8s\n", "Text", "Fill", "Chars");
    print_separator(70);

    for (int i = 0; i < NUM_PAIRS; i++) {
        float fill_a = trine_s1_fill_ratio(emb_a[i]);
        float fill_b = trine_s1_fill_ratio(emb_b[i]);
        printf("  %-35.35s  %7.3f  %6zu\n",
               PAIRS[i].text_a, fill_a, strlen(PAIRS[i].text_a));
        printf("  %-35.35s  %7.3f  %6zu\n",
               PAIRS[i].text_b, fill_b, strlen(PAIRS[i].text_b));
    }
    printf("\n");

    /* Print comparison table: one section per pair */
    for (int p = 0; p < NUM_PAIRS; p++) {
        printf("[%d] %s\n", p + 1, PAIRS[p].label);
        printf("    A: \"%s\"\n", PAIRS[p].text_a);
        printf("    B: \"%s\"\n", PAIRS[p].text_b);
        printf("\n");
        printf("    %-10s  %10s  %10s\n", "Lens", "Raw", "Calibrated");
        print_separator(40);

        for (int l = 0; l < NUM_LENSES; l++) {
            /* Raw lens-weighted cosine */
            float raw = trine_s1_compare(emb_a[p], emb_b[p],
                                         &LENSES[l].lens);

            /* Calibrated score (adjusts for embedding sparsity) */
            float fill_a = trine_s1_fill_ratio(emb_a[p]);
            float fill_b = trine_s1_fill_ratio(emb_b[p]);
            float cal = trine_s1_calibrate(raw, fill_a, fill_b);

            printf("    %-10s  %10.4f  %10.4f\n",
                   LENSES[l].name, raw, cal);
        }
        printf("\n");
    }

    /* Summary observations */
    printf("Observations:\n");
    printf("  - EDIT lens emphasizes character-level chain 1, so single-char\n");
    printf("    differences (fox/dog, 12345/12346) show clearly.\n");
    printf("  - CODE lens captures identifier structure, making snake_case\n");
    printf("    vs camelCase differences visible.\n");
    printf("  - DEDUP lens weights the word chain heavily, grouping texts\n");
    printf("    with the same vocabulary regardless of small edits.\n");
    printf("  - Calibration adjusts raw scores based on embedding density,\n");
    printf("    giving more accurate comparisons for short texts.\n");

    return 0;
}
