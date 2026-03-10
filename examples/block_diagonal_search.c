/* =====================================================================
 * block_diagonal_search.c -- Block-Diagonal Semantic Search Demo
 * =====================================================================
 *
 * Demonstrates Stage-2 block-diagonal projection for semantic search:
 *   1. Create a block-diagonal model with random weights (K=3)
 *   2. Encode 10 sample documents using Stage-2
 *   3. Encode a query and compare against all documents (S2 blend)
 *   4. Sort by similarity and print top-3 results
 *   5. Adaptive alpha: per-S1-bucket blending
 *   6. Side-by-side: identity model vs block-diagonal model
 *
 * Block-diagonal projection uses 4 independent 60x60 ternary matmuls,
 * one per TRINE chain (Edit, Morph, Phrase, Vocab).  This preserves
 * chain locality -- each chain's 60 dimensions are projected within
 * their own subspace, never mixing across chains.  The result is a 4x
 * smaller weight footprint (43,200 vs 172,800 bytes for K=3) while
 * maintaining the multi-scale structure of TRINE's shingle encoding.
 *
 * Build (from project root):
 *   cc -O2 -Wall -Wextra -Isrc/encode -Isrc/compare -Isrc/index \
 *      -Isrc/canon -Isrc/algebra -Isrc/model -Isrc/stage2/projection \
 *      -Isrc/stage2/cascade -Isrc/stage2/inference -Isrc/stage2/hebbian \
 *      -Isrc/stage2/persist -o build/block_diagonal_search \
 *      examples/block_diagonal_search.c build/libtrine.a -lm
 *
 * Run:   ./build/block_diagonal_search
 * ===================================================================== */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "trine_stage1.h"
#include "trine_stage2.h"
#include "trine_project.h"

/* ── Corpus: 10 sample documents spanning diverse topics ────────────── */

typedef struct {
    const char *id;     /* short label for display */
    const char *text;   /* document body           */
} document_t;

static const document_t CORPUS[] = {
    { "D01-cardiac",
      "Heart failure is a chronic condition where the heart cannot pump "
      "enough blood to meet the body's needs." },
    { "D02-neuro",
      "Alzheimer's disease causes progressive memory loss and cognitive "
      "decline due to neuronal degeneration." },
    { "D03-cardio",
      "Myocardial infarction occurs when blood flow to the heart muscle "
      "is blocked, causing tissue damage." },
    { "D04-python",
      "Python is a high-level programming language known for its readable "
      "syntax and dynamic typing." },
    { "D05-sorting",
      "Quicksort is a divide-and-conquer sorting algorithm with average "
      "case O(n log n) time complexity." },
    { "D06-ml",
      "Neural networks learn representations by adjusting connection "
      "weights through backpropagation of gradients." },
    { "D07-cooking",
      "Sourdough bread requires a fermented starter culture and long "
      "proofing times to develop its distinctive tangy flavor." },
    { "D08-history",
      "The Renaissance was a cultural movement spanning the 14th to 17th "
      "centuries that transformed European art and science." },
    { "D09-legal",
      "Contract law governs enforceable agreements between parties, "
      "requiring offer, acceptance, and consideration." },
    { "D10-climate",
      "Global warming is driven by greenhouse gas emissions, primarily "
      "carbon dioxide from fossil fuel combustion." },
};

#define N_DOCS ((int)(sizeof(CORPUS) / sizeof(CORPUS[0])))

/* ── Formatting helpers ─────────────────────────────────────────────── */

static void sep(int w)
{
    for (int i = 0; i < w; i++)
        putchar('-');
    putchar('\n');
}

static void banner(const char *title)
{
    printf("\n");
    sep(64);
    printf("  %s\n", title);
    sep(64);
    printf("\n");
}

/* ── Ranked result for sorting ──────────────────────────────────────── */

typedef struct {
    int   doc_idx;
    float score;
} ranked_t;

static int cmp_ranked_desc(const void *a, const void *b)
{
    float sa = ((const ranked_t *)a)->score;
    float sb = ((const ranked_t *)b)->score;
    if (sb > sa) return  1;
    if (sb < sa) return -1;
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(void)
{
    printf("============================================================\n");
    printf("TRINE Stage-2 Block-Diagonal Semantic Search Demo\n");
    printf("============================================================\n");

    /* ================================================================
     * Step 1: Create a block-diagonal model with random weights (K=3)
     * ================================================================
     *
     * trine_s2_create_random() builds a full 240x240 projection.
     * For block-diagonal, we instead call trine_s2_create_block_diagonal()
     * with K * 4 * 60 * 60 = 43,200 bytes of per-chain weights.
     *
     * Block-diagonal means each chain's 60 dims are projected
     * independently: chain 0 (Edit) uses block[0], chain 1 (Morph)
     * uses block[1], etc.  No cross-chain mixing at the projection
     * stage -- the cascade (if depth > 0) handles mixing later.
     *
     * We also create an identity model for comparison. */

    banner("Step 1: Model creation");

    const int K = TRINE_PROJECT_K;   /* 3 projection copies              */
    const uint32_t N_CELLS = 256;    /* cascade cells (moderate)         */
    const uint64_t SEED    = 42;     /* deterministic seed               */

    /* Allocate and fill K * 4 * 60 * 60 = 43,200 bytes of block weights.
     * trine_projection_block_random() fills each of the K copies with
     * pseudorandom ternary values {0,1,2} from a seeded LCG. */
    size_t bw_size = (size_t)K * 4 * 60 * 60;
    uint8_t *block_weights = malloc(bw_size);
    if (!block_weights) {
        fprintf(stderr, "Error: failed to allocate block weights\n");
        return 1;
    }
    trine_projection_block_random(block_weights, K, SEED);

    /* Create the block-diagonal model.
     * trine_s2_create_block_diagonal() copies the weights internally
     * and sets projection_mode = TRINE_S2_PROJ_BLOCK_DIAG automatically. */
    trine_s2_model_t *bd_model = trine_s2_create_block_diagonal(
        block_weights, K, N_CELLS, SEED);
    free(block_weights);  /* model owns its own copy now */

    if (!bd_model) {
        fprintf(stderr, "Error: failed to create block-diagonal model\n");
        return 1;
    }

    trine_s2_info_t info;
    if (trine_s2_info(bd_model, &info) == 0) {
        printf("Block-diagonal model:\n");
        printf("  K=%u  dims=%u  cascade=%u cells  identity=%s\n",
               info.projection_k, info.projection_dims,
               info.cascade_cells, info.is_identity ? "yes" : "no");
        printf("  Projection mode: block-diagonal (4 x 60x60 per copy)\n");
    }

    /* Identity model for baseline comparison */
    trine_s2_model_t *id_model = trine_s2_create_identity();
    if (!id_model) {
        fprintf(stderr, "Error: failed to create identity model\n");
        trine_s2_free(bd_model);
        return 1;
    }
    printf("\nIdentity model (baseline): projection = pass-through\n");

    /* ================================================================
     * Step 2: Encode all 10 documents with both models
     * ================================================================
     *
     * For each document we store:
     *   - s1[i]:  Stage-1 embedding (raw surface fingerprint)
     *   - s2[i]:  Stage-2 embedding (block-diagonal projected, depth=0)
     *
     * Depth 0 = projection only (no cascade ticks).  This isolates
     * the block-diagonal projection effect from cascade mixing. */

    banner("Step 2: Encode corpus (10 documents)");

    const uint32_t DEPTH = 0;  /* projection only */

    uint8_t s1_docs[N_DOCS][240];
    uint8_t s2_docs[N_DOCS][240];
    uint8_t id_docs[N_DOCS][240];  /* identity model embeddings */

    for (int i = 0; i < N_DOCS; i++) {
        const char *text = CORPUS[i].text;
        size_t len = strlen(text);

        /* Stage-1: surface fingerprint */
        trine_s1_encode(text, len, s1_docs[i]);

        /* Stage-2 block-diagonal: projected embedding */
        trine_s2_encode(bd_model, text, len, DEPTH, s2_docs[i]);

        /* Identity model: should equal Stage-1 */
        trine_s2_encode(id_model, text, len, DEPTH, id_docs[i]);

        printf("  Encoded [%s] (%zu chars)\n", CORPUS[i].id, len);
    }
    printf("\n  All %d documents encoded.\n", N_DOCS);

    /* ================================================================
     * Step 3: Encode query and compare against all documents
     * ================================================================
     *
     * Query: "heart disease and blood circulation problems"
     * We expect documents about cardiac/cardiovascular topics (D01, D03)
     * to rank highest. */

    banner("Step 3: Query comparison (S2 blend)");

    const char *QUERY = "heart disease and blood circulation problems";
    const float ALPHA = 0.65f;  /* 65% S1 + 35% S2 (best STS-B blend) */

    printf("Query: \"%s\"\n", QUERY);
    printf("Blend: alpha=%.2f (%.0f%% S1 + %.0f%% S2)\n\n",
           (double)ALPHA, (double)(ALPHA * 100), (double)((1.0f - ALPHA) * 100));

    uint8_t q_s1[240], q_s2[240];
    trine_s1_encode(QUERY, strlen(QUERY), q_s1);
    trine_s2_encode(bd_model, QUERY, strlen(QUERY), DEPTH, q_s2);

    trine_s1_lens_t lens = TRINE_S1_LENS_UNIFORM;

    ranked_t results[N_DOCS];

    printf("  %-14s  %8s  %8s  %8s\n", "Document", "S1", "S2(BD)", "Blend");
    sep(48);

    for (int i = 0; i < N_DOCS; i++) {
        float sim_s1 = trine_s1_compare(q_s1, s1_docs[i], &lens);
        float sim_s2 = trine_s2_compare(q_s2, s2_docs[i], &lens);
        float blended = ALPHA * sim_s1 + (1.0f - ALPHA) * sim_s2;

        results[i].doc_idx = i;
        results[i].score   = blended;

        printf("  %-14s  %8.4f  %8.4f  %8.4f\n",
               CORPUS[i].id, (double)sim_s1, (double)sim_s2, (double)blended);
    }

    /* ================================================================
     * Step 4: Sort by similarity and print top-3 results
     * ================================================================ */

    banner("Step 4: Top-3 results (sorted by blend score)");

    qsort(results, N_DOCS, sizeof(ranked_t), cmp_ranked_desc);

    for (int r = 0; r < 3 && r < N_DOCS; r++) {
        int idx = results[r].doc_idx;
        printf("  #%d  [%s]  score=%.4f\n", r + 1, CORPUS[idx].id,
               (double)results[r].score);
        printf("       \"%s\"\n\n", CORPUS[idx].text);
    }

    /* ================================================================
     * Step 5: Adaptive alpha (per-S1-bucket blending)
     * ================================================================
     *
     * Instead of a single global alpha, adaptive blending picks alpha
     * based on the S1 similarity bucket.  This allows different blend
     * ratios for different similarity ranges:
     *   - Low S1 (dissimilar): rely more on S2 (lower alpha)
     *   - High S1 (near-duplicate): trust S1 (higher alpha)
     *
     * Buckets cover [0.0-0.1), [0.1-0.2), ..., [0.9-1.0] */

    banner("Step 5: Adaptive alpha blending");

    /* Bucket alphas: trust S2 more at low similarity, S1 more at high */
    float buckets[10] = {
        0.30f,  /* [0.0-0.1): very dissimilar -- lean on S2        */
        0.35f,  /* [0.1-0.2)                                        */
        0.40f,  /* [0.2-0.3)                                        */
        0.50f,  /* [0.3-0.4)                                        */
        0.55f,  /* [0.4-0.5)                                        */
        0.60f,  /* [0.5-0.6)                                        */
        0.65f,  /* [0.6-0.7): moderate similarity                   */
        0.75f,  /* [0.7-0.8): similar -- trust S1                   */
        0.85f,  /* [0.8-0.9): very similar                          */
        0.95f,  /* [0.9-1.0]: near-duplicate -- almost pure S1      */
    };

    trine_s2_set_adaptive_alpha(bd_model, buckets);

    printf("  Adaptive alpha buckets:\n");
    for (int b = 0; b < 10; b++) {
        printf("    [%.1f-%.1f%s  alpha=%.2f\n",
               (double)(b * 0.1f), (double)((b + 1) * 0.1f),
               b < 9 ? ")" : "]", (double)buckets[b]);
    }
    printf("\n");

    /* Compare query against all docs using adaptive blend */
    printf("  %-14s  %8s  %8s  %8s\n", "Document", "Fixed", "Adaptive", "Delta");
    sep(48);

    ranked_t adaptive_results[N_DOCS];

    for (int i = 0; i < N_DOCS; i++) {
        float sim_s1 = trine_s1_compare(q_s1, s1_docs[i], &lens);
        float sim_s2 = trine_s2_compare(q_s2, s2_docs[i], &lens);
        float fixed = ALPHA * sim_s1 + (1.0f - ALPHA) * sim_s2;

        /* Adaptive blend uses the model's bucket table */
        float adaptive = trine_s2_compare_adaptive_blend(
            bd_model, q_s1, s1_docs[i], q_s2, s2_docs[i]);

        adaptive_results[i].doc_idx = i;
        adaptive_results[i].score   = adaptive;

        printf("  %-14s  %8.4f  %8.4f  %+7.4f\n",
               CORPUS[i].id, (double)fixed, (double)adaptive,
               (double)(adaptive - fixed));
    }

    /* Show top-3 with adaptive blend */
    printf("\n  Top-3 (adaptive blend):\n");
    qsort(adaptive_results, N_DOCS, sizeof(ranked_t), cmp_ranked_desc);
    for (int r = 0; r < 3 && r < N_DOCS; r++) {
        int idx = adaptive_results[r].doc_idx;
        printf("    #%d  [%s]  score=%.4f\n", r + 1, CORPUS[idx].id,
               (double)adaptive_results[r].score);
    }

    /* Disable adaptive alpha for subsequent operations */
    trine_s2_set_adaptive_alpha(bd_model, NULL);

    /* ================================================================
     * Step 6: Identity vs block-diagonal comparison
     * ================================================================
     *
     * The identity model passes Stage-1 trits through unchanged, so
     * S2(identity) == S1.  The block-diagonal model transforms each
     * chain independently, potentially reshaping the similarity
     * landscape.  With random weights (untrained), the block-diagonal
     * model scrambles the embedding space -- the real benefit emerges
     * after Hebbian training learns which channels to amplify/suppress.
     *
     * Here we show both side by side so the user can see the effect. */

    banner("Step 6: Identity vs block-diagonal model");

    uint8_t q_id[240];
    trine_s2_encode(id_model, QUERY, strlen(QUERY), DEPTH, q_id);

    printf("  Query: \"%s\"\n\n", QUERY);
    printf("  %-14s  %8s  %8s  %8s\n",
           "Document", "Identity", "BlockDiag", "Delta");
    sep(52);

    for (int i = 0; i < N_DOCS; i++) {
        float sim_id = trine_s2_compare(q_id, id_docs[i], &lens);
        float sim_bd = trine_s2_compare(q_s2, s2_docs[i], &lens);
        float delta  = sim_bd - sim_id;

        printf("  %-14s  %8.4f  %8.4f  %+7.4f\n",
               CORPUS[i].id, (double)sim_id, (double)sim_bd, (double)delta);
    }

    printf("\n  Note: With random (untrained) weights, block-diagonal\n"
           "  projection scrambles the similarity space. After Hebbian\n"
           "  training, learned per-chain gates selectively amplify\n"
           "  informative channels and suppress noisy ones.\n");

    /* ================================================================
     * Step 7: Depth sweep with block-diagonal model
     * ================================================================
     *
     * Show how cascade depth affects similarity for a specific pair.
     * At depth 0, only the block-diagonal projection is applied.
     * Higher depths add cascade mixing ticks that blend information
     * across chains via the random topology network. */

    banner("Step 7: Depth sweep (block-diagonal + cascade)");

    const char *pair_a = "heart disease and blood circulation problems";
    const char *pair_b = "myocardial infarction causes tissue damage";

    printf("  A: \"%s\"\n", pair_a);
    printf("  B: \"%s\"\n\n", pair_b);
    printf("  %-8s  %10s  %10s\n", "Depth", "BD Model", "Identity");
    sep(34);

    for (uint32_t d = 0; d <= 8; d++) {
        uint8_t ea_bd[240], eb_bd[240], ea_id[240], eb_id[240];

        trine_s2_encode(bd_model, pair_a, strlen(pair_a), d, ea_bd);
        trine_s2_encode(bd_model, pair_b, strlen(pair_b), d, eb_bd);
        trine_s2_encode(id_model, pair_a, strlen(pair_a), d, ea_id);
        trine_s2_encode(id_model, pair_b, strlen(pair_b), d, eb_id);

        float sim_bd = trine_s2_compare(ea_bd, eb_bd, &lens);
        float sim_id = trine_s2_compare(ea_id, eb_id, &lens);

        printf("  %-8u  %10.4f  %10.4f\n", d, (double)sim_bd, (double)sim_id);
    }

    /* ── Summary ────────────────────────────────────────────────────── */

    banner("Summary");

    printf("  Block-diagonal projection:\n");
    printf("    - 4 independent 60x60 matmuls (one per chain)\n");
    printf("    - Weight size: K*4*60*60 = 43,200 bytes (4x smaller)\n");
    printf("    - Preserves chain locality: no cross-chain mixing\n");
    printf("    - Cascade (depth > 0) adds controlled cross-chain flow\n");
    printf("\n");
    printf("  Adaptive alpha:\n");
    printf("    - Per-bucket blend ratio based on S1 similarity\n");
    printf("    - Low S1 -> trust S2 more (alpha=0.30)\n");
    printf("    - High S1 -> trust S1 more (alpha=0.95)\n");
    printf("\n");
    printf("  For best results, train with trine_train --block-diagonal\n");
    printf("  to learn which per-chain channels to amplify/suppress.\n");

    /* ── Cleanup ────────────────────────────────────────────────────── */

    trine_s2_free(bd_model);
    trine_s2_free(id_model);

    printf("\nDone.\n");
    return 0;
}
