/* =====================================================================
 * TRINE Stage-2 — Self-Supervised Deepening
 * =====================================================================
 *
 * Recursive partial-order bootstrapping (SPEC Section 2.4).
 *
 * After initial training on the Stage-1 partial order, the model
 * uses its own depth-D embeddings to generate new training signal
 * for depth-(D+1).  Each deepening round:
 *
 *   1. Freeze current accumulators -> model at depth D
 *   2. Re-read the training JSONL file
 *   3. For each pair: encode with Stage-2 at depth D
 *   4. Compute Stage-2 cosine between the embeddings
 *   5. Reset accumulators
 *   6. Re-accumulate using Stage-2 trits and Stage-2 similarity
 *   7. Free intermediate model
 *
 * After the final round, freeze and return the deepened model.
 *
 * ===================================================================== */

/* Enable getline() / ssize_t on POSIX systems */
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L
#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "trine_hebbian.h"
#include "trine_jsonl.h"
#include "trine_accumulator.h"
#include "trine_stage2.h"
#include "trine_encode.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* --------------------------------------------------------------------- */
/* Uniform cosine similarity between two 240-trit vectors                 */
/* --------------------------------------------------------------------- */

/* Centered cosine: treats trits {0,1,2} as {-1,0,+1}.
 * Correct metric for sign-based/diagonal projection outputs. */
static float centered_cosine_240(const uint8_t a[240], const uint8_t b[240])
{
    double dot = 0.0, ma = 0.0, mb = 0.0;
    for (int i = 0; i < 240; i++) {
        double va = (double)a[i] - 1.0;
        double vb = (double)b[i] - 1.0;
        dot += va * vb;
        ma  += va * va;
        mb  += vb * vb;
    }
    if (ma < 1e-12 || mb < 1e-12) return 0.0f;
    return (float)(dot / (sqrt(ma) * sqrt(mb)));
}

/* --------------------------------------------------------------------- */
/* Self-supervised deepening                                              */
/* --------------------------------------------------------------------- */

#define SD_MAX_TEXT  4096

struct trine_s2_model *trine_self_deepen(trine_hebbian_state_t *state,
                                          const char *data_path,
                                          uint32_t n_rounds)
{
    if (!state || !data_path || n_rounds == 0) return NULL;

    struct trine_s2_model *model = NULL;
    trine_hebbian_config_t cfg = trine_hebbian_get_config(state);
    uint32_t depth = cfg.cascade_depth;

    /* Dynamically-growing line buffer, reused across rounds */
    char *line = NULL;
    size_t line_cap = 0;

    for (uint32_t r = 0; r < n_rounds; r++) {

        /* Step 1: Freeze current accumulators to an intermediate model */
        model = trine_hebbian_freeze(state);
        if (!model) { free(line); return NULL; }

        /* Propagate projection mode to intermediate model */
        if (cfg.projection_mode != 0) {
            trine_s2_set_projection_mode(model, cfg.projection_mode);
        }

        /* Step 2: Reset accumulators for the next round */
        trine_hebbian_reset(state);

        /* Step 3: Re-read training data, re-accumulate with Stage-2 signal */
        FILE *fp = fopen(data_path, "r");
        if (!fp) {
            trine_s2_free(model);
            free(line);
            return NULL;
        }

        char text_a[SD_MAX_TEXT];
        char text_b[SD_MAX_TEXT];

        while (getline(&line, &line_cap, fp) != -1) {
            if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
                continue;

            int len_a = trine_jsonl_extract_string(line, 0, "text_a",
                                                    text_a, sizeof(text_a));
            int len_b = trine_jsonl_extract_string(line, 0, "text_b",
                                                    text_b, sizeof(text_b));

            if (len_a <= 0 || len_b <= 0) continue;

            /* Encode both texts through the Stage-2 model at current depth */
            uint8_t emb_a[240], emb_b[240];
            if (trine_s2_encode(model, text_a, (size_t)len_a, depth, emb_a) != 0)
                continue;
            if (trine_s2_encode(model, text_b, (size_t)len_b, depth, emb_b) != 0)
                continue;

            /* Use labeled score from data if available, else fall back to
             * Stage-2 cosine.  Human labels provide better training signal
             * for deepening rounds than self-predicted similarity. */
            float sim;
            if (!trine_jsonl_extract_float(line, 0, "score", &sim)) {
                sim = centered_cosine_240(emb_a, emb_b);
            }

            /* Accumulate with Stage-2 embeddings and labeled similarity */
            trine_hebbian_observe(state, emb_a, emb_b, sim);
        }

        fclose(fp);

        /* Free the intermediate model (unless this is the last round,
         * in which case we still free it and freeze a fresh one below) */
        trine_s2_free(model);
        model = NULL;
    }

    free(line);

    /* Final freeze after the last round of re-accumulation */
    model = trine_hebbian_freeze(state);
    return model;
}
