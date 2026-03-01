/* =====================================================================
 * TRINE CS-IDF — Corpus-Specific Inverse Document Frequency Weighting
 * Implementation
 * =====================================================================
 *
 * Per-channel document-frequency tracking and IDF weight computation.
 * Channel i's DF is the count of documents whose embedding[i] != 0.
 * IDF weight = log2(N / (1 + DF[i])), normalized to [0.01, 1.0].
 *
 * Build:
 *   cc -O2 -Wall -Wextra -c trine_csidf.c -o trine_csidf.o
 *
 * ===================================================================== */

#include "trine_csidf.h"

#include <string.h>
#include <math.h>
#include <stdio.h>

/* =====================================================================
 * I. INITIALIZATION
 * ===================================================================== */

void trine_csidf_init(trine_csidf_t *csidf)
{
    if (!csidf) return;
    memset(csidf, 0, sizeof(trine_csidf_t));
}

/* =====================================================================
 * II. OBSERVATION — Track document frequencies
 * ===================================================================== */

void trine_csidf_observe(trine_csidf_t *csidf, const uint8_t emb[240])
{
    if (!csidf || !emb) return;

    csidf->doc_count++;

    for (int i = 0; i < TRINE_CSIDF_DIMS; i++) {
        if (emb[i] != 0) {
            csidf->channel_df[i]++;
        }
    }

    /* Invalidate cached weights since corpus changed */
    csidf->computed = 0;
}

/* =====================================================================
 * III. WEIGHT COMPUTATION
 * ===================================================================== */

int trine_csidf_compute(trine_csidf_t *csidf)
{
    if (!csidf) return -1;
    if (csidf->doc_count == 0) {
        /* No documents — set uniform weights */
        for (int i = 0; i < TRINE_CSIDF_DIMS; i++) {
            csidf->weights[i] = 1.0f;
        }
        csidf->computed = 1;
        return -1;
    }

    float N = (float)csidf->doc_count;
    float max_idf = 0.0f;

    /* Pass 1: compute raw IDF values */
    for (int i = 0; i < TRINE_CSIDF_DIMS; i++) {
        float df = (float)csidf->channel_df[i];
        float raw = log2f(N / (1.0f + df));
        csidf->weights[i] = raw;
        if (raw > max_idf) max_idf = raw;
    }

    /* Pass 2: normalize to [CSIDF_MIN_WEIGHT, 1.0] */
    if (max_idf > 0.0f) {
        for (int i = 0; i < TRINE_CSIDF_DIMS; i++) {
            csidf->weights[i] /= max_idf;
            if (csidf->weights[i] < TRINE_CSIDF_MIN_WEIGHT) {
                csidf->weights[i] = TRINE_CSIDF_MIN_WEIGHT;
            }
        }
    } else {
        /* All channels have the same DF — uniform weights */
        for (int i = 0; i < TRINE_CSIDF_DIMS; i++) {
            csidf->weights[i] = 1.0f;
        }
    }

    csidf->computed = 1;
    return 0;
}

/* =====================================================================
 * IV. MERGE (for append mode)
 * ===================================================================== */

int trine_csidf_merge(trine_csidf_t *dst, const trine_csidf_t *src)
{
    if (!dst || !src) return -1;

    dst->doc_count += src->doc_count;
    for (int i = 0; i < TRINE_CSIDF_DIMS; i++) {
        dst->channel_df[i] += src->channel_df[i];
    }

    /* Invalidate cached weights */
    dst->computed = 0;

    return 0;
}

/* =====================================================================
 * V. COMPARISON — CS-IDF weighted cosine
 * ===================================================================== */

float trine_csidf_cosine(const uint8_t a[240], const uint8_t b[240],
                          const trine_csidf_t *csidf)
{
    if (!a || !b || !csidf || !csidf->computed) return 0.0f;

    float dot = 0.0f;
    float mag_a = 0.0f;
    float mag_b = 0.0f;

    for (int i = 0; i < TRINE_CSIDF_DIMS; i++) {
        float w  = csidf->weights[i];
        float ai = (float)a[i];
        float bi = (float)b[i];

        dot   += w * ai * bi;
        mag_a += w * ai * ai;
        mag_b += w * bi * bi;
    }

    float denom = sqrtf(mag_a) * sqrtf(mag_b);
    if (denom < 1e-12f) return 0.0f;

    float sim = dot / denom;
    if (sim > 1.0f) sim = 1.0f;
    if (sim < 0.0f) sim = 0.0f;

    return sim;
}

float trine_csidf_cosine_lens(const uint8_t a[240], const uint8_t b[240],
                               const trine_csidf_t *csidf,
                               const float lens[4])
{
    if (!a || !b || !csidf || !csidf->computed || !lens) return 0.0f;

    double weighted_sum = 0.0;
    double weight_sum   = 0.0;

    for (int chain = 0; chain < 4; chain++) {
        double lw = (double)lens[chain];
        if (lw <= 0.0) continue;

        int base = chain * 60;
        int end  = base + 60;

        float dot = 0.0f;
        float mag_a_c = 0.0f;
        float mag_b_c = 0.0f;

        for (int i = base; i < end; i++) {
            float w  = csidf->weights[i];
            float ai = (float)a[i];
            float bi = (float)b[i];

            dot     += w * ai * bi;
            mag_a_c += w * ai * ai;
            mag_b_c += w * bi * bi;
        }

        float denom = sqrtf(mag_a_c) * sqrtf(mag_b_c);
        float cos_c = 0.0f;
        if (denom >= 1e-12f) {
            cos_c = dot / denom;
            if (cos_c > 1.0f) cos_c = 1.0f;
            if (cos_c < 0.0f) cos_c = 0.0f;
        }

        weighted_sum += lw * (double)cos_c;
        weight_sum   += lw;
    }

    if (weight_sum == 0.0) return 0.0f;
    return (float)(weighted_sum / weight_sum);
}

/* =====================================================================
 * VI. SERIALIZATION
 * ===================================================================== */

int trine_csidf_write(const trine_csidf_t *csidf, void *fp_void)
{
    FILE *fp = (FILE *)fp_void;
    if (!csidf || !fp) return -1;

    if (fwrite(&csidf->doc_count, sizeof(uint32_t), 1, fp) != 1)
        return -1;
    if (fwrite(csidf->channel_df, sizeof(uint32_t), TRINE_CSIDF_DIMS, fp)
        != TRINE_CSIDF_DIMS)
        return -1;
    if (fwrite(csidf->weights, sizeof(float), TRINE_CSIDF_DIMS, fp)
        != TRINE_CSIDF_DIMS)
        return -1;

    return 0;
}

int trine_csidf_read(trine_csidf_t *csidf, void *fp_void)
{
    FILE *fp = (FILE *)fp_void;
    if (!csidf || !fp) return -1;

    if (fread(&csidf->doc_count, sizeof(uint32_t), 1, fp) != 1)
        return -1;
    if (fread(csidf->channel_df, sizeof(uint32_t), TRINE_CSIDF_DIMS, fp)
        != TRINE_CSIDF_DIMS)
        return -1;
    if (fread(csidf->weights, sizeof(float), TRINE_CSIDF_DIMS, fp)
        != TRINE_CSIDF_DIMS)
        return -1;

    csidf->computed = 1;
    return 0;
}
