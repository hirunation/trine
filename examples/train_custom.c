/* =====================================================================
 * train_custom.c -- End-to-End Hebbian Training Demo
 * =====================================================================
 *
 * Demonstrates the complete Stage-2 Hebbian training workflow:
 *   1. Configure and create a Hebbian trainer (diagonal mode)
 *   2. Feed hardcoded training pairs via observe_text()
 *   3. Inspect training metrics
 *   4. Freeze the trainer into a Stage-2 model
 *   5. Use the frozen model for inference (encode + compare)
 *   6. Save the model to a .trine2 file
 *   7. Load it back and verify identical encodings
 *   8. Clean up all resources
 *
 * Build (from project root):
 *   gcc -O2 -Wall -I../src/encode -I../src/compare -I../src/stage2/inference \
 *       -I../src/stage2/projection -I../src/stage2/cascade -I../src/stage2/hebbian \
 *       -I../src/stage2/persist -I../src/algebra -I../src/model \
 *       train_custom.c -L../build -ltrine -lm -o train_custom
 *
 * Run:   ./train_custom
 * ===================================================================== */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "trine_hebbian.h"
#include "trine_stage2.h"
#include "trine_s2_persist.h"
#include "trine_stage1.h"

/* ---- Training pairs: 10 similar + 10 dissimilar -------------------- */

typedef struct { const char *a, *b; const char *label; } pair_t;

static const pair_t PAIRS[] = {
    /* Similar: synonyms, paraphrases, rephrasing */
    {"the car is fast",                  "the automobile is quick",          "synonym swap"},
    {"she opened the door slowly",       "she slowly opened the door",      "word reorder"},
    {"the weather is cold today",        "it is freezing outside today",    "paraphrase"},
    {"he runs every morning",            "he jogs each morning",            "near synonym"},
    {"the food was delicious",           "the meal was tasty",              "synonym pair"},
    {"please send me the report",        "could you email the report to me","request rephrase"},
    {"the project is complete",          "the project has been finished",   "tense variation"},
    {"turn off the lights",              "switch off the lights",           "verb synonym"},
    {"the cat sat on the mat",           "a cat was sitting on the mat",    "article + tense"},
    {"prices have increased sharply",    "prices rose dramatically",        "adverb synonym"},
    /* Dissimilar: completely unrelated topics */
    {"the stock market crashed",         "banana bread recipe with walnuts", "finance vs cooking"},
    {"quantum entanglement experiment",  "how to train a puppy at home",    "physics vs pets"},
    {"annual revenue exceeded targets",  "the garden needs more sunlight",  "business vs garden"},
    {"install the software update now",  "the renaissance period in europe","tech vs history"},
    {"the patient has a fever",          "solving quadratic equations",     "medical vs math"},
    {"flight departure delayed",         "watercolor painting techniques",  "travel vs art"},
    {"the defendant pleaded not guilty", "photosynthesis in plant cells",   "legal vs biology"},
    {"configure the firewall rules",     "yoga poses for beginners",        "network vs fitness"},
    {"the bridge collapsed in the storm","chocolate cake frosting recipe",  "disaster vs baking"},
    {"compile with debug flags",         "migration patterns of terns",     "code vs ornithology"},
};

#define N_PAIRS ((int)(sizeof(PAIRS) / sizeof(PAIRS[0])))
#define MODEL_PATH "/tmp/train_custom_example.trine2"

/* ---- Main ----------------------------------------------------------- */

int main(void)
{
    printf("TRINE Stage-2 Hebbian Training Demo\n");
    printf("====================================\n\n");

    /* Step 1: Configure the Hebbian trainer.
     *
     * Diagonal mode (projection_mode=1) uses per-channel gating:
     *   gate=2 -> keep (positive),  gate=1 -> flip (Z3 negate),
     *   gate=0 -> zero (uninformative -> neutral).
     * High similarity_threshold (0.90) = only strong matches get
     * positive updates. Low density (0.15) = aggressive pruning. */
    printf("[1] Creating Hebbian trainer...\n");

    trine_hebbian_config_t config = TRINE_HEBBIAN_CONFIG_DEFAULT;
    config.projection_mode       = 1;      /* diagonal gating */
    config.freeze_target_density = 0.15f;
    config.similarity_threshold  = 0.90f;
    config.cascade_depth         = 0;      /* projection only */
    config.cascade_cells         = 0;

    trine_hebbian_state_t *trainer = trine_hebbian_create(&config);
    if (!trainer) { fprintf(stderr, "Error: create failed\n"); return 1; }

    printf("    Mode: diagonal | Threshold: %.2f | Density: %.2f | Depth: 0\n\n",
           (double)config.similarity_threshold, (double)config.freeze_target_density);

    /* Step 2: Feed training pairs.
     * observe_text() encodes both texts via Stage-1, computes S1 cosine,
     * and applies a Hebbian update (positive if s1 > threshold, else negative). */
    printf("[2] Observing %d training pairs...\n", N_PAIRS);

    for (int i = 0; i < N_PAIRS; i++) {
        trine_hebbian_observe_text(trainer,
            PAIRS[i].a, strlen(PAIRS[i].a),
            PAIRS[i].b, strlen(PAIRS[i].b));
        printf("    [%2d] %s\n", i + 1, PAIRS[i].label);
    }
    printf("\n");

    /* Step 3: Inspect training metrics.
     * The accumulator holds raw Hebbian counters after observation.
     * Positive = channels where similar pairs co-activate.
     * Negative = channels where dissimilar pairs co-activate. */
    printf("[3] Training metrics:\n");

    trine_hebbian_metrics_t m;
    if (trine_hebbian_metrics(trainer, &m) != 0) {
        fprintf(stderr, "Error: metrics failed\n");
        trine_hebbian_free(trainer); return 1;
    }

    printf("    Pairs observed:   %lld\n",  (long long)m.pairs_observed);
    printf("    Max |counter|:    %d\n",    (int)m.max_abs_counter);
    printf("    Positive/Neg/Zero: %u / %u / %u\n",
           (unsigned)m.n_positive_weights, (unsigned)m.n_negative_weights,
           (unsigned)m.n_zero_weights);
    printf("    Weight density:   %.4f  (threshold: %d)\n\n",
           (double)m.weight_density, (int)m.effective_threshold);

    /* Step 4: Freeze the trainer into a Stage-2 model.
     * Quantizes accumulators into ternary weights {0,1,2}.
     * The frozen model uses zero floating point at inference time. */
    printf("[4] Freezing trainer...\n");

    trine_s2_model_t *model = trine_hebbian_freeze(trainer);
    if (!model) {
        fprintf(stderr, "Error: freeze failed\n");
        trine_hebbian_free(trainer); return 1;
    }
    trine_s2_set_projection_mode(model, TRINE_S2_PROJ_DIAGONAL);

    trine_s2_info_t info;
    if (trine_s2_info(model, &info) == 0) {
        printf("    Projection: %ux%u (%u copies) | Cascade: %u cells\n\n",
               info.projection_dims, info.projection_dims,
               info.projection_k, info.cascade_cells);
    }

    /* Step 5: Inference -- encode and compare.
     * trine_s2_encode() = Stage-1 encode -> projection -> cascade.
     * trine_s2_compare() = lens-weighted cosine (NULL = uniform). */
    printf("[5] Inference with frozen model:\n");

    uint8_t ea[TRINE_S2_DIM], eb[TRINE_S2_DIM];
    const uint32_t depth = 0;

    const char *sim_a = "the car is very fast";
    const char *sim_b = "the automobile is really quick";
    const char *dif_a = "the car is very fast";
    const char *dif_b = "baking sourdough bread at home";

    if (trine_s2_encode(model, sim_a, strlen(sim_a), depth, ea) != 0 ||
        trine_s2_encode(model, sim_b, strlen(sim_b), depth, eb) != 0) {
        fprintf(stderr, "Error: encode failed\n");
        trine_s2_free(model); trine_hebbian_free(trainer); return 1;
    }
    printf("    Similar:    \"%s\" vs \"%s\"\n", sim_a, sim_b);
    printf("    S2 score:   %.4f\n", (double)trine_s2_compare(ea, eb, NULL));

    if (trine_s2_encode(model, dif_a, strlen(dif_a), depth, ea) != 0 ||
        trine_s2_encode(model, dif_b, strlen(dif_b), depth, eb) != 0) {
        fprintf(stderr, "Error: encode failed\n");
        trine_s2_free(model); trine_hebbian_free(trainer); return 1;
    }
    printf("    Dissimilar: \"%s\" vs \"%s\"\n", dif_a, dif_b);
    printf("    S2 score:   %.4f\n\n", (double)trine_s2_compare(ea, eb, NULL));

    /* Save encoding for verification after reload */
    uint8_t emb_before[TRINE_S2_DIM];
    trine_s2_encode(model, sim_a, strlen(sim_a), depth, emb_before);

    /* Step 6: Save the model to a .trine2 file.
     * Format: 72-byte header + K*DIM*DIM weights + 8-byte FNV-1a checksum.
     * Total 172,880 bytes for K=3, DIM=240. */
    printf("[6] Saving model to %s...\n", MODEL_PATH);

    trine_s2_save_config_t save_cfg = {
        .similarity_threshold = config.similarity_threshold,
        .density              = config.freeze_target_density,
        .topo_seed            = 0
    };
    if (trine_s2_save(model, MODEL_PATH, &save_cfg) != 0) {
        fprintf(stderr, "Error: save failed\n");
        trine_s2_free(model); trine_hebbian_free(trainer); return 1;
    }
    printf("    Saved successfully.\n\n");

    /* Step 7: Load the model back and verify identical encoding.
     * trine_s2_validate() checks header + payload checksums.
     * trine_s2_load() also validates internally before returning. */
    printf("[7] Loading model from %s...\n", MODEL_PATH);

    if (trine_s2_validate(MODEL_PATH) != 0) {
        fprintf(stderr, "Error: validation failed\n");
        trine_s2_free(model); trine_hebbian_free(trainer); return 1;
    }

    trine_s2_model_t *loaded = trine_s2_load(MODEL_PATH);
    if (!loaded) {
        fprintf(stderr, "Error: load failed\n");
        trine_s2_free(model); trine_hebbian_free(trainer); return 1;
    }
    trine_s2_set_projection_mode(loaded, TRINE_S2_PROJ_DIAGONAL);

    uint8_t emb_after[TRINE_S2_DIM];
    if (trine_s2_encode(loaded, sim_a, strlen(sim_a), depth, emb_after) != 0) {
        fprintf(stderr, "Error: encode with loaded model failed\n");
        trine_s2_free(loaded); trine_s2_free(model);
        trine_hebbian_free(trainer); return 1;
    }

    int match = (memcmp(emb_before, emb_after, TRINE_S2_DIM) == 0);
    printf("    Encoding match: %s\n\n", match ? "PASS" : "FAIL");

    if (!match) {
        fprintf(stderr, "Error: saved/loaded model produces different encoding\n");
        trine_s2_free(loaded); trine_s2_free(model);
        trine_hebbian_free(trainer); return 1;
    }

    /* Step 8: Clean up all resources. */
    printf("[8] Cleaning up...\n");
    trine_s2_free(loaded);
    trine_s2_free(model);
    trine_hebbian_free(trainer);

    printf("    Done. All resources freed.\n\n");
    printf("Training workflow complete.\n");
    return 0;
}
