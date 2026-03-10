/* =====================================================================
 * TRINE — Ternary Resonance Interference Network Embedding
 * Public Runtime API v1.0.1
 * =====================================================================
 *
 * Complete text embedding library backed by ternary algebraic cascade.
 * Loads a self-contained .trine model file and computes embeddings
 * with ZERO external dependencies.
 *
 * USAGE
 *   trine_model_t *model = trine_load("model.trine");
 *   trine_embedding_t emb;
 *   trine_embed(model, "hello", 5, TRINE_STANDARD, &emb);
 *   // emb.trits[0..emb.dims-1] contains the embedding
 *   trine_embedding_free(&emb);
 *   trine_free(model);
 *
 * THREAD SAFETY
 *   A loaded model is read-only and may be shared across threads.
 *   Each trine_embed() call allocates its own cascade workspace
 *   on the heap, so concurrent calls on the same model are safe.
 *
 * RESOLUTION TIERS
 *   Screening:   68 dims (~11 bytes)   — fast coarse filter
 *   Standard:  1053 dims (~166 bytes)  — balanced quality/speed
 *   Deep:     17410 dims (~2.7 KB)     — maximum fidelity
 *
 * ===================================================================== */

#ifndef TRINE_H
#define TRINE_H

#include <stdint.h>
#include <stddef.h>

/* =====================================================================
 * Opaque Model Handle
 * ===================================================================== */

typedef struct trine_model trine_model_t;

/* =====================================================================
 * Embedding Output
 * ===================================================================== */

typedef struct {
    uint8_t *trits;       /* Trit values (0, 1, or 2), heap-allocated      */
    int      dims;        /* Number of dimensions                          */
    int      resolution;  /* 0=screening, 1=standard, 2=deep               */
    uint32_t consistency; /* Self-consistency score from VERIFY layer (0-2) */
    int      ticks;       /* Cascade ticks used                            */
} trine_embedding_t;

/* =====================================================================
 * Resolution Levels
 * ===================================================================== */

#define TRINE_SCREENING  0  /*   68 dims (~11 bytes)   — cascade fingerprint */
#define TRINE_STANDARD   1  /* 1053 dims (~166 bytes)  — cascade fingerprint */
#define TRINE_DEEP       2  /* 17410 dims (~2.7 KB)    — cascade fingerprint */
#define TRINE_SHINGLE    3  /*  240 dims (~38 bytes)   — similarity search   */

/* =====================================================================
 * Model Lifecycle
 * ===================================================================== */

/*
 * trine_load — Load a .trine model file from disk.
 *
 * Reads the file at `path`, validates magic/version/checksums, and
 * returns an opaque model handle. The model is immutable after load.
 *
 * Returns NULL on error (message printed to stderr).
 */
trine_model_t *trine_load(const char *path);

/*
 * trine_free — Release all resources held by a loaded model.
 *
 * Frees the snap arena, I/O maps, and the model struct itself.
 * Safe to call with NULL (no-op).
 */
void trine_free(trine_model_t *model);

/* =====================================================================
 * Core Embedding
 * ===================================================================== */

/*
 * trine_embed — Compute a text embedding.
 *
 * @param model       Loaded model (from trine_load).
 * @param text        Input text (ASCII). Characters >= 128 are masked.
 * @param len         Length of input text in bytes.
 * @param resolution  TRINE_SCREENING, TRINE_STANDARD, or TRINE_DEEP.
 * @param out         Output embedding (caller-provided struct).
 *
 * The function allocates out->trits. Caller must free with
 * trine_embedding_free() when done.
 *
 * Returns 0 on success, negative on error:
 *   -1: invalid argument (NULL model, NULL out, bad resolution)
 *   -2: allocation failure
 *   -3: cascade engine failure
 */
int trine_embed(trine_model_t *model,
                const char *text, size_t len,
                int resolution,
                trine_embedding_t *out);

/*
 * trine_embedding_free — Release memory held by an embedding.
 *
 * Frees out->trits and zeroes the struct.
 * Safe to call on a zeroed or already-freed embedding (no-op).
 */
void trine_embedding_free(trine_embedding_t *emb);

/* =====================================================================
 * Comparison Functions
 * ===================================================================== */

/*
 * trine_compare — Cosine similarity in trit space.
 *
 * Treats trit vectors as elements of {0, 1, 2}^N and computes:
 *   similarity = dot(a, b) / (|a| * |b|)
 *
 * Returns 0.0 (orthogonal) to 1.0 (identical).
 * Returns 0.0 if either embedding has zero magnitude.
 * Returns -1.0 on error (mismatched dimensions, NULL input).
 */
double trine_compare(const trine_embedding_t *a, const trine_embedding_t *b);

/*
 * trine_hamming — Hamming distance between two embeddings.
 *
 * Counts the number of positions where a->trits[i] != b->trits[i].
 *
 * Returns the count, or -1 on error (mismatched dimensions, NULL input).
 */
int trine_hamming(const trine_embedding_t *a, const trine_embedding_t *b);

/* =====================================================================
 * Lens System — Weighted Per-Chain Similarity
 * ===================================================================== */

/* Chain indices for lens weights */
#define TRINE_CHAIN_EDIT    0   /* Chain 1: char unigrams + bigrams      */
#define TRINE_CHAIN_MORPH   1   /* Chain 2: trigrams                     */
#define TRINE_CHAIN_PHRASE  2   /* Chain 3: 5-grams                      */
#define TRINE_CHAIN_VOCAB   3   /* Chain 4: word unigrams                */
#define TRINE_NUM_CHAINS    4
#define TRINE_CHAIN_WIDTH   60  /* Channels per chain                    */

typedef struct {
    float weights[TRINE_NUM_CHAINS];  /* Per-chain importance weights */
} trine_lens_t;

/* Predefined lenses (11 total: 6 generic + 5 domain-specific) */
#define TRINE_LENS_UNIFORM  {{1.0f, 1.0f, 1.0f, 1.0f}}
#define TRINE_LENS_EDIT     {{1.0f, 0.3f, 0.1f, 0.0f}}
#define TRINE_LENS_MORPH    {{0.3f, 1.0f, 0.5f, 0.2f}}
#define TRINE_LENS_PHRASE   {{0.1f, 0.5f, 1.0f, 0.3f}}
#define TRINE_LENS_VOCAB    {{0.0f, 0.2f, 0.3f, 1.0f}}
#define TRINE_LENS_DEDUP    {{0.5f, 0.5f, 0.7f, 1.0f}}

/* Domain-specific lenses */
#define TRINE_LENS_CODE     {{1.0f, 0.8f, 0.4f, 0.2f}}  /* Source code: identifier-level */
#define TRINE_LENS_LEGAL    {{0.2f, 0.4f, 1.0f, 0.8f}}  /* Legal: clause & term matching */
#define TRINE_LENS_MEDICAL  {{0.3f, 1.0f, 0.6f, 0.5f}}  /* Medical: morpheme prefixes   */
#define TRINE_LENS_SUPPORT  {{0.2f, 0.4f, 0.7f, 1.0f}}  /* Support: product vocabulary  */
#define TRINE_LENS_POLICY   {{0.1f, 0.3f, 1.0f, 0.8f}}  /* Policy: regulatory phrasing  */

/* Detailed similarity result with per-chain breakdown */
typedef struct {
    float chain[TRINE_NUM_CHAINS];  /* Per-chain cosine similarities    */
    float combined;                  /* Weighted combined score          */
    float uniform;                   /* Unweighted (flat) cosine         */
} trine_similarity_t;

/*
 * trine_compare_lens — Weighted per-chain cosine similarity.
 *
 * Computes cosine similarity independently for each of the 4 chains
 * (60 channels each), then combines with lens weights:
 *   combined = sum(weight[i] * chain_cosine[i]) / sum(weight[i])
 *
 * Only works with TRINE_SHINGLE (240-dim) embeddings. Returns -1.0
 * on error (wrong dimensions, NULL input).
 */
double trine_compare_lens(const trine_embedding_t *a,
                           const trine_embedding_t *b,
                           const trine_lens_t *lens);

/*
 * trine_compare_detail — Full per-chain similarity breakdown.
 *
 * Computes per-chain cosine, weighted combination, and flat cosine.
 * Writes results to `out`. Returns 0 on success, -1 on error.
 */
int trine_compare_detail(const trine_embedding_t *a,
                          const trine_embedding_t *b,
                          const trine_lens_t *lens,
                          trine_similarity_t *out);

/* =====================================================================
 * Model Introspection
 * ===================================================================== */

typedef struct {
    uint32_t    snap_count;   /* Total snaps in the model topology         */
    uint32_t    version;      /* .trine format version                     */
    int         layer_count;  /* Number of processing layers (13)          */
    const char *model_name;   /* "TRINE v1"                                */
} trine_info_t;

/*
 * trine_info — Query model metadata.
 *
 * @param model  Loaded model.
 * @param info   Output struct (caller-provided).
 *
 * Returns 0 on success, -1 on error (NULL arguments).
 */
int trine_info(const trine_model_t *model, trine_info_t *info);

/* =====================================================================
 * Version String
 * ===================================================================== */

/*
 * trine_version — Return the library version string.
 *
 * Returns a pointer to a static string, e.g. "1.0.1".
 */
const char *trine_version(void);

#endif /* TRINE_H */
