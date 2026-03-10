/* =====================================================================
 * semantic_search.c -- TRINE Stage-2 Semantic Embedding Demo
 * =====================================================================
 *
 * Demonstrates Stage-2 encode/compare, depth sweep, and S1+S2 blending.
 * With an identity model (no arg), S1 == S2. With a trained .trine2, S2
 * captures learned semantic structure via Hebbian diagonal gating.
 *
 * Build (from project root):
 *   gcc -O2 -Wall -I../src/encode -I../src/compare \
 *       -I../src/stage2/inference -I../src/stage2/projection \
 *       -I../src/stage2/cascade -I../src/stage2/hebbian \
 *       -I../src/stage2/persist -I../src/algebra -I../src/model \
 *       semantic_search.c -L../build -ltrine -lm -o semantic_search
 *
 * Run:   ./semantic_search [model.trine2]
 * ===================================================================== */

#include <stdio.h>
#include <string.h>
#include "trine_stage1.h"
#include "trine_stage2.h"
#include "trine_s2_persist.h"

/* ---- Test pairs ----------------------------------------------------- */

typedef struct { const char *label, *text_a, *text_b; } text_pair_t;

/* Semantically similar but surface-different */
static const text_pair_t SEMANTIC[] = {
    { "Medical synonym",        "heart attack",
      "myocardial infarction" },
    { "Common synonym",         "car",
      "automobile" },
    { "Adjective synonym",      "happy",
      "joyful" },
    { "Paraphrase",             "The cat sat on the mat",
      "A feline rested on the rug" },
    { "Technical restatement",  "the program crashed with a segfault",
      "the application terminated due to a memory access violation" },
};

/* Surface-similar but semantically different (polysemy) */
static const text_pair_t SURFACE[] = {
    { "Polysemy: bank",  "bank of the river",           "bank account" },
    { "Polysemy: bat",   "the bat flew out of the cave",
      "he swung the bat at the ball" },
};

#define N_SEM ((int)(sizeof(SEMANTIC) / sizeof(SEMANTIC[0])))
#define N_SRF ((int)(sizeof(SURFACE)  / sizeof(SURFACE[0])))

/* ---- Helpers -------------------------------------------------------- */

static void sep(int w) { for (int i = 0; i < w; i++) putchar('-'); putchar('\n'); }

/* Blend: alpha * s1 + (1 - alpha) * s2 */
static float blend(float s1, float s2, float a) { return a*s1 + (1.0f-a)*s2; }

/* Encode, compare, and print one pair across S1 / S2(d=0) / S2(d=4) */
static void compare_pair(const trine_s2_model_t *m,
                          const text_pair_t *p, int idx, float alpha)
{
    trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;
    uint8_t s1a[240], s1b[240], d0a[240], d0b[240], d4a[240], d4b[240];

    trine_s1_encode(p->text_a, strlen(p->text_a), s1a);
    trine_s1_encode(p->text_b, strlen(p->text_b), s1b);
    trine_s2_encode(m, p->text_a, strlen(p->text_a), 0, d0a);
    trine_s2_encode(m, p->text_b, strlen(p->text_b), 0, d0b);
    trine_s2_encode(m, p->text_a, strlen(p->text_a), 4, d4a);
    trine_s2_encode(m, p->text_b, strlen(p->text_b), 4, d4b);

    float ss1 = trine_s1_compare(s1a, s1b, &lens);
    float sd0 = trine_s2_compare(d0a, d0b, &lens);
    float sd4 = trine_s2_compare(d4a, d4b, &lens);

    printf("  [%d] %s\n", idx, p->label);
    printf("      A: \"%s\"\n", p->text_a);
    printf("      B: \"%s\"\n", p->text_b);
    printf("      S1=%.4f  S2(d=0)=%.4f  S2(d=4)=%.4f  Blend=%.4f\n\n",
           ss1, sd0, sd4, blend(ss1, sd0, alpha));
}

/* ---- Main ----------------------------------------------------------- */

int main(int argc, char **argv)
{
    const float ALPHA = 0.65f;  /* 65% S1 + 35% S2 (best STS-B blend) */

    printf("============================================================\n");
    printf("TRINE Stage-2 Semantic Search Demo\n");
    printf("============================================================\n\n");

    /* ── 1. Load model ─────────────────────────────────────────────── */

    trine_s2_model_t *model = NULL;

    if (argc >= 2) {
        printf("Loading model: %s\n", argv[1]);
        if (trine_s2_validate(argv[1]) != 0) {
            fprintf(stderr, "Error: invalid .trine2 file: %s\n", argv[1]);
            return 1;
        }
        model = trine_s2_load(argv[1]);
        if (!model) {
            fprintf(stderr, "Error: failed to load model\n");
            return 1;
        }
    } else {
        printf("No .trine2 file -- using identity model (S1 == S2).\n");
        printf("Usage: %s [model.trine2]\n", argv[0]);
        model = trine_s2_create_identity();
        if (!model) {
            fprintf(stderr, "Error: failed to create identity model\n");
            return 1;
        }
    }

    /* ── 2. Model introspection ────────────────────────────────────── */

    trine_s2_info_t info;
    if (trine_s2_info(model, &info) == 0) {
        printf("\nModel: K=%u dims=%u cascade=%u cells identity=%s\n",
               info.projection_k, info.projection_dims,
               info.cascade_cells, info.is_identity ? "yes" : "no");
        printf("Blend: alpha=%.2f (%.0f%% S1 + %.0f%% S2)\n\n",
               ALPHA, ALPHA * 100, (1.0f - ALPHA) * 100);
    }

    /* ── 3. Semantic pairs (similar meaning, different surface) ───── */

    sep(60);
    printf("Semantically similar, surface-different pairs\n");
    sep(60);
    printf("\n");

    for (int i = 0; i < N_SEM; i++)
        compare_pair(model, &SEMANTIC[i], i + 1, ALPHA);

    /* ── 4. Surface pairs (similar surface, different meaning) ────── */

    sep(60);
    printf("Surface-similar, semantically different pairs (polysemy)\n");
    sep(60);
    printf("\n");

    for (int i = 0; i < N_SRF; i++)
        compare_pair(model, &SURFACE[i], i + 1, ALPHA);

    /* ── 5. Depth sweep ────────────────────────────────────────────── */

    sep(60);
    printf("Depth sweep: \"heart attack\" vs \"myocardial infarction\"\n");
    sep(60);
    printf("\n  %-8s  %10s\n", "Depth", "S2 Sim");
    sep(24);

    {
        const char *ta = "heart attack", *tb = "myocardial infarction";
        trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;
        for (uint32_t d = 0; d <= 8; d++) {
            uint8_t ea[240], eb[240];
            trine_s2_encode(model, ta, strlen(ta), d, ea);
            trine_s2_encode(model, tb, strlen(tb), d, eb);
            printf("  %-8u  %10.4f\n", d, trine_s2_compare(ea, eb, &lens));
        }
    }

    /* ── 6. Summary ────────────────────────────────────────────────── */

    printf("\nNotes:\n");
    printf("  - Identity model: S1 == S2 at all depths (pass-through).\n");
    printf("  - Trained model: per-channel gating boosts semantic pairs.\n");
    printf("  - Blend: score = alpha*S1 + (1-alpha)*S2.\n");
    printf("  - Best STS-B blend: alpha=0.65 (+28.8%% over S1 alone).\n");
    printf("  - Depth 0 = projection only; higher depths add cascade mixing.\n");

    trine_s2_free(model);
    return 0;
}
