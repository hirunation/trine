/* =====================================================================
 * train_block_diagonal.c -- Block-Diagonal Hebbian Training Demo
 * =====================================================================
 *
 * Demonstrates the block-diagonal projection mode introduced in v1.0.3.
 *
 * TRINE's 240-dimensional embeddings are organized as 4 chains of 60
 * channels each (Edit, Morph, Phrase, Vocab).  In the standard diagonal
 * mode, each of the 240 channels receives an independent keep/flip/zero
 * gate.  Block-diagonal mode goes further: it learns a separate 60x60
 * ternary projection matrix *per chain*, enabling cross-channel mixing
 * within each chain while preserving chain independence.
 *
 * Layout:
 *   Diagonal:       240 gates       (240 parameters)
 *   Block-diagonal: 4 x 60 x 60    (14,400 parameters per copy, K=3 -> 43,200)
 *   Full-matrix:    240 x 240       (57,600 parameters per copy, K=3 -> 172,800)
 *
 * Block-diagonal is a middle ground: more expressive than diagonal
 * (allows within-chain mixing) but sparser than the full 240x240 matrix
 * (no cross-chain leakage).  The .trine2 file is 43,280 bytes vs
 * 172,880 bytes for full-matrix mode.
 *
 * This example:
 *   1. Creates 30 positive + 30 negative synthetic training pairs
 *   2. Trains with block_diagonal=1
 *   3. Prints training metrics
 *   4. Freezes to a block-diagonal model
 *   5. Encodes and compares sample texts
 *   6. Saves the model to a .trine2 file
 *   7. Loads it back and verifies consistency
 *   8. Compares identity, diagonal, and block-diagonal models side by side
 *   9. Cleans up all resources
 *
 * Build (from project root):
 *   cc -O2 -Wall -Wextra -Isrc/encode -Isrc/compare -Isrc/index \
 *      -Isrc/canon -Isrc/algebra -Isrc/model -Isrc/stage2/projection \
 *      -Isrc/stage2/cascade -Isrc/stage2/inference -Isrc/stage2/hebbian \
 *      -Isrc/stage2/persist -o build/train_block_diagonal \
 *      examples/train_block_diagonal.c build/libtrine.a -lm
 *
 * Run:   ./build/train_block_diagonal
 * ===================================================================== */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "trine_hebbian.h"
#include "trine_stage2.h"
#include "trine_s2_persist.h"
#include "trine_stage1.h"

/* ---- Training data -------------------------------------------------- */

/* Each pair has two texts and a human-readable label.
 * The first 30 are "positive" (semantically similar) pairs --
 * synonyms, paraphrases, rephrasings in various domains.
 * The next 30 are "negative" (unrelated topics) pairs.
 *
 * The Hebbian trainer does NOT use these labels directly; instead it
 * computes Stage-1 cosine similarity and applies a positive update
 * when s1 > threshold, negative otherwise.  The labels here are for
 * human readability only. */

typedef struct { const char *a, *b; const char *label; } pair_t;

static const pair_t POS_PAIRS[] = {
    /* Synonym swaps */
    {"the car is fast",                 "the automobile is quick",           "synonym: car/auto"},
    {"a big house",                     "a large home",                      "synonym: big/large"},
    {"she is happy",                    "she is joyful",                     "synonym: happy/joyful"},
    {"the movie was excellent",         "the film was outstanding",          "synonym: movie/film"},
    {"the child is clever",             "the kid is smart",                  "synonym: child/kid"},
    /* Paraphrases */
    {"the weather is cold today",       "it is freezing outside today",      "paraphrase: weather"},
    {"he runs every morning",           "he jogs each morning",              "paraphrase: running"},
    {"please send me the report",       "could you email the report to me",  "paraphrase: request"},
    {"the project is complete",         "the project has been finished",     "paraphrase: completion"},
    {"prices have increased sharply",   "prices rose dramatically",          "paraphrase: prices"},
    /* Word reordering */
    {"she opened the door slowly",      "she slowly opened the door",        "reorder: adverb"},
    {"carefully he placed the cup",     "he placed the cup carefully",       "reorder: adverb 2"},
    {"quickly the dog ran away",        "the dog ran away quickly",          "reorder: adverb 3"},
    /* Tense and article variation */
    {"the cat sat on the mat",          "a cat was sitting on the mat",      "tense+article"},
    {"she writes code daily",           "she has been writing code daily",   "tense: progressive"},
    /* Verb synonyms */
    {"turn off the lights",             "switch off the lights",             "verb: turn/switch"},
    {"he fixed the car engine",         "he repaired the car engine",        "verb: fix/repair"},
    {"she built a wooden table",        "she constructed a wooden table",    "verb: build/construct"},
    /* Technical / domain */
    {"the server crashed unexpectedly", "the server went down without warning", "tech: crash"},
    {"allocate memory for the buffer",  "reserve memory for the buffer",     "tech: allocate/reserve"},
    {"the patient has a high fever",    "the patient is running a temperature", "medical: fever"},
    {"the defendant was found guilty",  "the accused was convicted",         "legal: guilty"},
    {"quarterly revenue grew by 12%",   "quarterly earnings increased 12%",  "finance: revenue"},
    /* Longer paraphrases */
    {"the quick brown fox jumps over",  "a fast brown fox leaps over",       "phrase: quick/fast"},
    {"we need to reduce our costs",     "we must cut our expenses",          "phrase: reduce/cut"},
    {"the test results were positive",  "the test outcomes were favorable",  "phrase: results"},
    {"upload the file to the server",   "put the file on the server",        "phrase: upload/put"},
    {"the algorithm runs in O(n)",      "the algorithm has linear time",     "tech: complexity"},
    {"he graduated with honors",        "he finished his degree with distinction", "education"},
    {"the bridge spans the river",      "the bridge crosses the river",      "geography: spans"},
};

static const pair_t NEG_PAIRS[] = {
    /* Completely unrelated topics */
    {"the stock market crashed",        "banana bread recipe with walnuts",    "finance vs cooking"},
    {"quantum entanglement experiment", "how to train a puppy at home",        "physics vs pets"},
    {"annual revenue exceeded targets", "the garden needs more sunlight",      "business vs garden"},
    {"install the software update now", "the renaissance period in europe",    "tech vs history"},
    {"the patient has a fever",         "solving quadratic equations",         "medical vs math"},
    {"flight departure delayed",        "watercolor painting techniques",      "travel vs art"},
    {"the defendant pleaded not guilty","photosynthesis in plant cells",       "legal vs biology"},
    {"configure the firewall rules",    "yoga poses for beginners",            "network vs fitness"},
    {"the bridge collapsed in the storm","chocolate cake frosting recipe",     "disaster vs baking"},
    {"compile with debug flags",        "migration patterns of arctic terns",  "code vs ornithology"},
    {"database schema migration",       "how to knit a sweater pattern",       "db vs knitting"},
    {"mutual fund performance report",  "origami crane folding instructions",  "finance vs craft"},
    {"tcp packet retransmission",       "ancient egyptian hieroglyphics",      "network vs history"},
    {"insulin resistance diagnosis",    "best surfing beaches in australia",   "medical vs travel"},
    {"react component lifecycle",       "medieval castle architecture",        "frontend vs history"},
    {"corporate merger announcement",   "the lifecycle of a butterfly",        "business vs nature"},
    {"docker container orchestration",  "traditional japanese tea ceremony",   "devops vs culture"},
    {"pension fund allocation strategy","how to grow tomatoes indoors",        "finance vs garden"},
    {"ssl certificate renewal",         "impressionist painting techniques",   "security vs art"},
    {"blood pressure medication dosage","assembly of flat pack furniture",     "medical vs diy"},
    {"recursive binary search tree",    "sourdough starter maintenance",       "algo vs baking"},
    {"kubernetes pod scheduling",       "baroque music composition rules",     "devops vs music"},
    {"mortgage interest rate forecast", "caring for pet goldfish properly",    "finance vs pets"},
    {"compiler optimization passes",    "types of volcanic eruptions",         "compiler vs geology"},
    {"neural network backpropagation",  "history of the olympic games",        "ml vs sports"},
    {"load balancer configuration",     "traditional bread baking methods",    "infra vs cooking"},
    {"gene sequencing methodology",     "how to play chess openings",          "bio vs games"},
    {"http response status codes",      "rose garden pest control guide",      "web vs garden"},
    {"cardiac surgery recovery time",   "deep sea fishing equipment list",     "medical vs hobby"},
    {"agile sprint retrospective",      "penguin colony behavior patterns",    "pm vs zoology"},
};

#define N_POS ((int)(sizeof(POS_PAIRS) / sizeof(POS_PAIRS[0])))
#define N_NEG ((int)(sizeof(NEG_PAIRS) / sizeof(NEG_PAIRS[0])))
#define MODEL_PATH "/tmp/train_block_diagonal_example.trine2"

/* ---- Helpers -------------------------------------------------------- */

/* Print a horizontal separator line of width w. */
static void sep(int w)
{
    for (int i = 0; i < w; i++) putchar('-');
    putchar('\n');
}

/* Encode a pair with a given model and return the S2 similarity score. */
static float encode_and_compare(const trine_s2_model_t *model,
                                const char *text_a, const char *text_b,
                                uint32_t depth)
{
    uint8_t ea[TRINE_S2_DIM], eb[TRINE_S2_DIM];
    if (trine_s2_encode(model, text_a, strlen(text_a), depth, ea) != 0 ||
        trine_s2_encode(model, text_b, strlen(text_b), depth, eb) != 0) {
        return -1.0f;
    }
    return trine_s2_compare(ea, eb, NULL);
}

/* ---- Main ----------------------------------------------------------- */

int main(void)
{
    printf("=========================================================\n");
    printf("TRINE Stage-2 Block-Diagonal Training Demo (v1.0.3)\n");
    printf("=========================================================\n\n");

    /* =================================================================
     * Step 1: Create synthetic training pairs
     * =================================================================
     *
     * We have 30 positive (similar) and 30 negative (dissimilar) pairs.
     * The Hebbian trainer will encode both texts via Stage-1, compute
     * their cosine similarity, and apply:
     *   - Positive update if s1 > similarity_threshold
     *   - Negative update if s1 <= similarity_threshold
     *
     * This is self-supervised: no explicit labels needed.  The Stage-1
     * surface similarity IS the training signal. */

    printf("[1] Training data: %d positive + %d negative = %d pairs\n\n",
           N_POS, N_NEG, N_POS + N_NEG);

    /* =================================================================
     * Step 2: Create Hebbian trainer with block_diagonal=1
     * =================================================================
     *
     * Key config fields:
     *   block_diagonal = 1
     *     Tells the freezer to produce 4 independent 60x60 weight
     *     matrices (one per chain) rather than one 240-element diagonal
     *     or one 240x240 full matrix.
     *
     *   projection_mode = 1 (diagonal)
     *     During accumulation, updates are diagonal.  The block_diagonal
     *     flag takes effect at freeze time, shaping how the accumulated
     *     counters are quantized into the projection weights.
     *
     *   freeze_target_density = 0.20
     *     Controls sparsity: ~20% of weights will be non-zero after
     *     freezing.  Lower = sparser = faster inference but less signal.
     *
     *   similarity_threshold = 0.85
     *     Stage-1 cosine above this -> positive Hebbian update.
     *     Below -> negative update.
     *
     *   cascade_depth = 0, cascade_cells = 0
     *     No cascade mixing -- projection only.  This keeps the demo
     *     focused on the block-diagonal projection itself. */

    printf("[2] Creating Hebbian trainer (block_diagonal=1)...\n");

    trine_hebbian_config_t config = TRINE_HEBBIAN_CONFIG_DEFAULT;
    config.projection_mode       = 1;       /* diagonal accumulation     */
    config.block_diagonal        = 1;       /* freeze to block-diagonal  */
    config.freeze_target_density = 0.20f;   /* 20% non-zero weights      */
    config.similarity_threshold  = 0.85f;   /* positive/negative cutoff  */
    config.cascade_depth         = 0;       /* projection only           */
    config.cascade_cells         = 0;       /* no cascade cells          */

    trine_hebbian_state_t *trainer = trine_hebbian_create(&config);
    if (!trainer) {
        fprintf(stderr, "Error: failed to create Hebbian trainer\n");
        return 1;
    }

    printf("    Mode:      block_diagonal (4 x 60x60 per copy)\n");
    printf("    Threshold: %.2f\n", (double)config.similarity_threshold);
    printf("    Density:   %.2f\n", (double)config.freeze_target_density);
    printf("    Depth:     %u (projection only)\n\n",
           (unsigned)config.cascade_depth);

    /* =================================================================
     * Step 3: Observe all pairs
     * =================================================================
     *
     * observe_text() internally:
     *   1. Stage-1 encodes both texts -> 240-trit vectors
     *   2. Computes Stage-1 cosine similarity
     *   3. If s1 > threshold: positive Hebbian update (co-activation)
     *      If s1 <= threshold: negative Hebbian update (anti-Hebbian)
     *
     * The accumulator tracks per-channel counters.  Positive pairs
     * increment, negative pairs decrement.  After many observations,
     * channels that reliably co-activate for similar texts will have
     * large positive counters; channels dominated by noise will be
     * near zero and get pruned at freeze time. */

    printf("[3] Observing %d training pairs...\n", N_POS + N_NEG);

    /* Feed positive pairs */
    printf("    Positive pairs:\n");
    for (int i = 0; i < N_POS; i++) {
        trine_hebbian_observe_text(trainer,
            POS_PAIRS[i].a, strlen(POS_PAIRS[i].a),
            POS_PAIRS[i].b, strlen(POS_PAIRS[i].b));
        printf("      [%2d] %s\n", i + 1, POS_PAIRS[i].label);
    }

    /* Feed negative pairs */
    printf("    Negative pairs:\n");
    for (int i = 0; i < N_NEG; i++) {
        trine_hebbian_observe_text(trainer,
            NEG_PAIRS[i].a, strlen(NEG_PAIRS[i].a),
            NEG_PAIRS[i].b, strlen(NEG_PAIRS[i].b));
        printf("      [%2d] %s\n", i + 1, NEG_PAIRS[i].label);
    }
    printf("\n");

    /* =================================================================
     * Step 4: Print training metrics
     * =================================================================
     *
     * Metrics are computed from the raw accumulator state (pre-freeze).
     *   pairs_observed   = total observe() calls
     *   max_abs_counter  = largest |counter| (signal strength indicator)
     *   n_positive/neg/zero = distribution of counter signs
     *   weight_density   = fraction that would survive freezing
     *   effective_threshold = quantization threshold (auto-tuned) */

    printf("[4] Training metrics:\n");

    trine_hebbian_metrics_t metrics;
    if (trine_hebbian_metrics(trainer, &metrics) != 0) {
        fprintf(stderr, "Error: failed to get training metrics\n");
        trine_hebbian_free(trainer);
        return 1;
    }

    printf("    Pairs observed:     %lld\n",  (long long)metrics.pairs_observed);
    printf("    Max |counter|:      %d\n",    (int)metrics.max_abs_counter);
    printf("    Positive weights:   %u\n",    (unsigned)metrics.n_positive_weights);
    printf("    Negative weights:   %u\n",    (unsigned)metrics.n_negative_weights);
    printf("    Zero weights:       %u\n",    (unsigned)metrics.n_zero_weights);
    printf("    Weight density:     %.4f\n",  (double)metrics.weight_density);
    printf("    Effective threshold: %d\n\n", (int)metrics.effective_threshold);

    /* =================================================================
     * Step 5: Freeze to block-diagonal model
     * =================================================================
     *
     * trine_hebbian_freeze() quantizes the raw counters into ternary
     * weights {0, 1, 2} using the effective threshold:
     *   counter >  T  ->  2 (keep: pass-through)
     *   counter < -T  ->  1 (flip: Z3 negation)
     *   |counter| <= T -> 0 (zero: uninformative, pruned)
     *
     * With block_diagonal=1, the frozen weights have shape:
     *   K copies x 4 chains x 60 x 60 = 43,200 bytes total
     * (vs 172,800 bytes for the full 240x240 matrix)
     *
     * After freezing, we set projection mode to TRINE_S2_PROJ_BLOCK_DIAG
     * to tell the forward pass to apply chain-local 60x60 matmul. */

    printf("[5] Freezing trainer to block-diagonal model...\n");

    trine_s2_model_t *block_model = trine_hebbian_freeze(trainer);
    if (!block_model) {
        fprintf(stderr, "Error: freeze failed\n");
        trine_hebbian_free(trainer);
        return 1;
    }

    /* Set block-diagonal projection mode for inference. */
    trine_s2_set_projection_mode(block_model, TRINE_S2_PROJ_BLOCK_DIAG);

    trine_s2_info_t info;
    if (trine_s2_info(block_model, &info) == 0) {
        printf("    Projection:  K=%u, dims=%u, mode=block-diagonal\n",
               info.projection_k, info.projection_dims);
        printf("    Cascade:     %u cells\n", info.cascade_cells);
        printf("    Identity:    %s\n\n", info.is_identity ? "yes" : "no");
    }

    /* =================================================================
     * Step 6: Encode and compare sample texts
     * =================================================================
     *
     * Demonstrate that the block-diagonal model produces meaningful
     * similarity scores: similar texts should score higher than
     * dissimilar texts. */

    printf("[6] Inference with block-diagonal model:\n\n");

    const uint32_t depth = 0;  /* projection only, no cascade */

    /* Test pairs for demonstration */
    const char *test_pairs[][3] = {
        {"the car is very fast",
         "the automobile is really quick",
         "Similar: synonym swap"},

        {"she opened the door gently",
         "she gently opened the door",
         "Similar: word reorder"},

        {"the server crashed last night",
         "the server went down yesterday evening",
         "Similar: tech paraphrase"},

        {"the car is very fast",
         "baking sourdough bread at home",
         "Dissimilar: auto vs cooking"},

        {"database query optimization",
         "the lifecycle of a butterfly",
         "Dissimilar: tech vs nature"},

        {"the patient needs surgery",
         "how to play chess openings",
         "Dissimilar: medical vs games"},
    };
    int n_tests = (int)(sizeof(test_pairs) / sizeof(test_pairs[0]));

    for (int i = 0; i < n_tests; i++) {
        float score = encode_and_compare(block_model,
                                          test_pairs[i][0],
                                          test_pairs[i][1],
                                          depth);
        printf("    [%d] %s\n", i + 1, test_pairs[i][2]);
        printf("        A: \"%s\"\n", test_pairs[i][0]);
        printf("        B: \"%s\"\n", test_pairs[i][1]);
        printf("        S2 score: %.4f\n\n", (double)score);
    }

    /* Save one encoding for round-trip verification later. */
    uint8_t emb_before[TRINE_S2_DIM];
    trine_s2_encode(block_model, test_pairs[0][0],
                    strlen(test_pairs[0][0]), depth, emb_before);

    /* =================================================================
     * Step 7: Save model to file
     * =================================================================
     *
     * The .trine2 format stores:
     *   - 72-byte header (magic, version, flags, training params)
     *   - Projection weights (43,200 bytes for block-diagonal)
     *   - 8-byte payload checksum (FNV-1a)
     *
     * Total: 43,280 bytes -- about 4x smaller than full-matrix mode.
     * The FLAG_BLOCK_DIAG bit (bit 2) in the header flags tells the
     * loader to reconstruct a block-diagonal model. */

    printf("[7] Saving model to %s...\n", MODEL_PATH);

    trine_s2_save_config_t save_cfg = {
        .similarity_threshold = config.similarity_threshold,
        .density              = config.freeze_target_density,
        .topo_seed            = 0
    };

    if (trine_s2_save(block_model, MODEL_PATH, &save_cfg) != 0) {
        fprintf(stderr, "Error: save failed\n");
        trine_s2_free(block_model);
        trine_hebbian_free(trainer);
        return 1;
    }
    printf("    Saved successfully.\n\n");

    /* =================================================================
     * Step 8: Load model and verify consistency
     * =================================================================
     *
     * A round-trip test: load the saved model and verify that encoding
     * the same text produces the exact same 240-trit embedding.  This
     * confirms deterministic persistence. */

    printf("[8] Loading model from %s...\n", MODEL_PATH);

    /* Validate file integrity (header + payload checksums). */
    if (trine_s2_validate(MODEL_PATH) != 0) {
        fprintf(stderr, "Error: file validation failed\n");
        trine_s2_free(block_model);
        trine_hebbian_free(trainer);
        return 1;
    }

    trine_s2_model_t *loaded_model = trine_s2_load(MODEL_PATH);
    if (!loaded_model) {
        fprintf(stderr, "Error: load failed\n");
        trine_s2_free(block_model);
        trine_hebbian_free(trainer);
        return 1;
    }

    /* The loaded model should already have block-diagonal mode set
     * from the FLAG_BLOCK_DIAG flag in the file header.  We set it
     * explicitly here for clarity. */
    trine_s2_set_projection_mode(loaded_model, TRINE_S2_PROJ_BLOCK_DIAG);

    uint8_t emb_after[TRINE_S2_DIM];
    if (trine_s2_encode(loaded_model, test_pairs[0][0],
                        strlen(test_pairs[0][0]), depth, emb_after) != 0) {
        fprintf(stderr, "Error: encode with loaded model failed\n");
        trine_s2_free(loaded_model);
        trine_s2_free(block_model);
        trine_hebbian_free(trainer);
        return 1;
    }

    int match = (memcmp(emb_before, emb_after, TRINE_S2_DIM) == 0);
    printf("    Round-trip encoding match: %s\n\n", match ? "PASS" : "FAIL");

    if (!match) {
        fprintf(stderr, "Error: saved/loaded model produces different encoding\n");
        trine_s2_free(loaded_model);
        trine_s2_free(block_model);
        trine_hebbian_free(trainer);
        return 1;
    }

    /* =================================================================
     * Step 9: Compare identity, diagonal, and block-diagonal models
     * =================================================================
     *
     * To understand what block-diagonal adds, we compare three models:
     *
     *   Identity:       S2 = S1 (pass-through, no learned weights)
     *   Diagonal:       Per-channel gate: keep / flip / zero
     *   Block-diagonal: Per-chain 60x60 matmul: cross-channel mixing
     *                   within each chain, no cross-chain leakage
     *
     * For similar pairs, we want higher scores (boosted signal).
     * For dissimilar pairs, we want lower scores (suppressed noise).
     * The gap between similar and dissimilar scores indicates how
     * well each model separates the classes.
     *
     * Diagonal is fast but limited to element-wise gating.
     * Block-diagonal can express within-chain rotations and mixtures,
     * giving it more capacity to separate similar from dissimilar. */

    printf("[9] Model comparison: identity vs diagonal vs block-diagonal\n\n");

    /* Create identity model (pass-through). */
    trine_s2_model_t *identity_model = trine_s2_create_identity();
    if (!identity_model) {
        fprintf(stderr, "Error: failed to create identity model\n");
        trine_s2_free(loaded_model);
        trine_s2_free(block_model);
        trine_hebbian_free(trainer);
        return 1;
    }

    /* Create diagonal model: retrain with same data but diagonal mode.
     * We reuse the existing trainer by resetting and re-observing. */
    trine_hebbian_config_t diag_config = TRINE_HEBBIAN_CONFIG_DEFAULT;
    diag_config.projection_mode       = 1;      /* diagonal */
    diag_config.block_diagonal        = 0;      /* NOT block-diagonal */
    diag_config.freeze_target_density = 0.20f;
    diag_config.similarity_threshold  = 0.85f;
    diag_config.cascade_depth         = 0;
    diag_config.cascade_cells         = 0;

    trine_hebbian_state_t *diag_trainer = trine_hebbian_create(&diag_config);
    if (!diag_trainer) {
        fprintf(stderr, "Error: failed to create diagonal trainer\n");
        trine_s2_free(identity_model);
        trine_s2_free(loaded_model);
        trine_s2_free(block_model);
        trine_hebbian_free(trainer);
        return 1;
    }

    /* Train the diagonal model on the same data. */
    for (int i = 0; i < N_POS; i++) {
        trine_hebbian_observe_text(diag_trainer,
            POS_PAIRS[i].a, strlen(POS_PAIRS[i].a),
            POS_PAIRS[i].b, strlen(POS_PAIRS[i].b));
    }
    for (int i = 0; i < N_NEG; i++) {
        trine_hebbian_observe_text(diag_trainer,
            NEG_PAIRS[i].a, strlen(NEG_PAIRS[i].a),
            NEG_PAIRS[i].b, strlen(NEG_PAIRS[i].b));
    }

    trine_s2_model_t *diag_model = trine_hebbian_freeze(diag_trainer);
    if (!diag_model) {
        fprintf(stderr, "Error: diagonal freeze failed\n");
        trine_s2_free(identity_model);
        trine_s2_free(loaded_model);
        trine_s2_free(block_model);
        trine_hebbian_free(diag_trainer);
        trine_hebbian_free(trainer);
        return 1;
    }
    trine_s2_set_projection_mode(diag_model, TRINE_S2_PROJ_DIAGONAL);

    /* Comparison table header. */
    sep(78);
    printf("%-36s  %10s  %10s  %10s\n",
           "Pair", "Identity", "Diagonal", "BlockDiag");
    sep(78);

    /* Compare a set of representative pairs across all three models. */
    const char *cmp_pairs[][3] = {
        {"the car is very fast",
         "the automobile is really quick",
         "Similar: car/auto"},

        {"the server crashed last night",
         "the server went down yesterday",
         "Similar: crash/down"},

        {"she is very happy today",
         "she is feeling joyful today",
         "Similar: happy/joyful"},

        {"the car is very fast",
         "baking sourdough bread at home",
         "Dissimilar: auto/bake"},

        {"database query optimization",
         "the lifecycle of a butterfly",
         "Dissimilar: tech/nature"},

        {"neural network training",
         "penguin colony behavior",
         "Dissimilar: ML/zoology"},
    };
    int n_cmp = (int)(sizeof(cmp_pairs) / sizeof(cmp_pairs[0]));

    float sum_id = 0, sum_diag = 0, sum_block = 0;
    int n_sim = 0, n_dis = 0;
    float sim_id = 0, sim_diag = 0, sim_block = 0;
    float dis_id = 0, dis_diag = 0, dis_block = 0;

    for (int i = 0; i < n_cmp; i++) {
        float s_id    = encode_and_compare(identity_model,
                                            cmp_pairs[i][0], cmp_pairs[i][1], depth);
        float s_diag  = encode_and_compare(diag_model,
                                            cmp_pairs[i][0], cmp_pairs[i][1], depth);
        float s_block = encode_and_compare(block_model,
                                            cmp_pairs[i][0], cmp_pairs[i][1], depth);

        printf("%-36s  %10.4f  %10.4f  %10.4f\n",
               cmp_pairs[i][2], (double)s_id, (double)s_diag, (double)s_block);

        sum_id    += s_id;
        sum_diag  += s_diag;
        sum_block += s_block;

        /* Track similar vs dissimilar averages (first 3 are similar). */
        if (i < 3) {
            sim_id    += s_id;
            sim_diag  += s_diag;
            sim_block += s_block;
            n_sim++;
        } else {
            dis_id    += s_id;
            dis_diag  += s_diag;
            dis_block += s_block;
            n_dis++;
        }
    }

    sep(78);

    /* Print averages and separation gap. */
    printf("\n  Average (similar):      %10.4f  %10.4f  %10.4f\n",
           (double)(sim_id / n_sim), (double)(sim_diag / n_sim),
           (double)(sim_block / n_sim));
    printf("  Average (dissimilar):   %10.4f  %10.4f  %10.4f\n",
           (double)(dis_id / n_dis), (double)(dis_diag / n_dis),
           (double)(dis_block / n_dis));

    float gap_id    = (sim_id / n_sim) - (dis_id / n_dis);
    float gap_diag  = (sim_diag / n_sim) - (dis_diag / n_dis);
    float gap_block = (sim_block / n_sim) - (dis_block / n_dis);

    printf("  Separation gap:         %10.4f  %10.4f  %10.4f\n\n",
           (double)gap_id, (double)gap_diag, (double)gap_block);

    /* Suppress unused variable warnings for sum_* accumulators. */
    (void)sum_id;
    (void)sum_diag;
    (void)sum_block;

    /* =================================================================
     * Step 10: Clean up all resources
     * ================================================================= */

    printf("[10] Cleaning up...\n");

    trine_s2_free(identity_model);
    trine_s2_free(diag_model);
    trine_hebbian_free(diag_trainer);
    trine_s2_free(loaded_model);
    trine_s2_free(block_model);
    trine_hebbian_free(trainer);

    printf("     Done. All resources freed.\n\n");

    /* Summary notes. */
    printf("Notes:\n");
    printf("  - Block-diagonal: 4 independent 60x60 projections (one per chain).\n");
    printf("  - Model size: 43,280 bytes (vs 172,880 for full-matrix).\n");
    printf("  - Cross-channel mixing within each chain, no cross-chain leakage.\n");
    printf("  - Separation gap measures how well similar and dissimilar pairs\n");
    printf("    are distinguished: higher = better class separation.\n");
    printf("  - Diagonal is fastest; block-diagonal adds within-chain capacity;\n");
    printf("    full-matrix is most expressive but largest and slowest.\n");

    return 0;
}
