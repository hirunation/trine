/* ═══════════════════════════════════════════════════════════════════════
 * TRINE — Ternary Resonance Interference Network Embedding
 * Batch Compare — Cache-Friendly Block Processing Implementation
 * ═══════════════════════════════════════════════════════════════════════
 *
 * CACHE STRATEGY
 *   The inner loop processes corpus vectors in blocks of 16. Each block
 *   iteration:
 *     1. The query vector (240 bytes) remains hot in L1 cache.
 *     2. 16 corpus vectors (16 * 240 = 3840 bytes) are streamed in.
 *     3. Total working set: ~4 KB, well within 32-64 KB L1d.
 *
 *   Within each block, we compute the three dot-product accumulators
 *   (dot_ab, mag_a, mag_b) for all 240 dimensions per corpus vector.
 *   The mag_a (query magnitude squared) is precomputed once outside
 *   the block loop, saving redundant computation across all n vectors.
 *
 * TOP-K
 *   Uses a binary min-heap of size k. After the heap is filled, each
 *   new candidate is compared against the heap minimum (O(1)). If it
 *   exceeds the minimum, we replace the root and sift down (O(log k)).
 *   Final extraction is done by repeatedly removing the root, yielding
 *   results in ascending order; we then reverse for descending output.
 *   Total: O(n log k) instead of O(n log n) for a full sort.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -Werror -Isrc/encode -Isrc/compare \
 *      -c -o build/trine_batch_compare.o src/compare/trine_batch_compare.c
 *
 * ═══════════════════════════════════════════════════════════════════════ */

#include "trine_batch_compare.h"
#include "trine_stage1.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * I. INTERNAL HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Compute uniform cosine similarity between two 240-byte trit vectors.
 *
 * Treats trit values {0, 1, 2} as real-valued vector components and
 * computes standard cosine = dot(a,b) / (|a| * |b|) over all 240 dims.
 *
 * If query_mag_sq is provided (non-zero), it is used as the precomputed
 * squared magnitude of vector a, avoiding redundant recomputation when
 * comparing one query against many corpus vectors.
 *
 * Returns 0.0f if either vector has zero magnitude.
 */
static float batch_cosine(const uint8_t *a, const uint8_t *b,
                           uint64_t query_mag_sq)
{
    uint64_t dot_ab = 0;
    uint64_t mag_b  = 0;

    for (int i = 0; i < TRINE_S1_DIMS; i++) {
        uint64_t va = a[i];
        uint64_t vb = b[i];
        dot_ab += va * vb;
        mag_b  += vb * vb;
    }

    if (query_mag_sq == 0 || mag_b == 0) return 0.0f;

    double denom = sqrt((double)query_mag_sq) * sqrt((double)mag_b);
    if (denom == 0.0) return 0.0f;

    double sim = (double)dot_ab / denom;

    /* Clamp to [0, 1] for floating-point rounding */
    if (sim > 1.0) sim = 1.0;
    if (sim < 0.0) sim = 0.0;

    return (float)sim;
}

/*
 * Precompute the squared magnitude of a 240-byte trit vector.
 * Returns sum of (trit[i])^2 for i in [0, 240).
 */
static uint64_t compute_mag_sq(const uint8_t *vec)
{
    uint64_t mag = 0;
    for (int i = 0; i < TRINE_S1_DIMS; i++) {
        uint64_t v = vec[i];
        mag += v * v;
    }
    return mag;
}

/* ═══════════════════════════════════════════════════════════════════════
 * II. MIN-HEAP FOR TOP-K TRACKING
 * ═══════════════════════════════════════════════════════════════════════
 *
 * A min-heap ordered by similarity. The root is the smallest similarity
 * in the top-k set, so we can quickly decide whether a new candidate
 * should be inserted (if it exceeds the root).
 */

typedef struct {
    float  sim;
    size_t idx;
} heap_entry_t;

/*
 * Sift down the element at position pos in a min-heap of size heap_size.
 * Maintains the min-heap invariant: parent.sim <= child.sim.
 */
static void heap_sift_down(heap_entry_t *heap, size_t heap_size, size_t pos)
{
    while (1) {
        size_t smallest = pos;
        size_t left  = 2 * pos + 1;
        size_t right = 2 * pos + 2;

        if (left < heap_size && heap[left].sim < heap[smallest].sim)
            smallest = left;
        if (right < heap_size && heap[right].sim < heap[smallest].sim)
            smallest = right;

        if (smallest == pos) break;

        /* Swap */
        heap_entry_t tmp = heap[pos];
        heap[pos] = heap[smallest];
        heap[smallest] = tmp;

        pos = smallest;
    }
}

/*
 * Sift up the element at position pos in a min-heap.
 * Used during initial heap construction (insertion phase).
 */
static void heap_sift_up(heap_entry_t *heap, size_t pos)
{
    while (pos > 0) {
        size_t parent = (pos - 1) / 2;
        if (heap[pos].sim < heap[parent].sim) {
            heap_entry_t tmp = heap[pos];
            heap[pos] = heap[parent];
            heap[parent] = tmp;
            pos = parent;
        } else {
            break;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * III. BATCH COMPARE — FULL RESULTS
 * ═══════════════════════════════════════════════════════════════════════ */

int trine_batch_compare(
    const uint8_t *query,
    const uint8_t *corpus,
    size_t n,
    float *sims)
{
    if (!query || !corpus || !sims) return -1;
    if (n == 0) return 0;

    /* Precompute query magnitude squared once */
    uint64_t query_mag_sq = compute_mag_sq(query);

    /* Process in blocks of TRINE_BATCH_BLOCK_SIZE for cache locality.
     * The query vector stays in L1 while we iterate through a block
     * of corpus vectors. */
    size_t full_blocks = n / TRINE_BATCH_BLOCK_SIZE;
    size_t remainder   = n % TRINE_BATCH_BLOCK_SIZE;

    /* Full blocks */
    for (size_t b = 0; b < full_blocks; b++) {
        size_t base = b * TRINE_BATCH_BLOCK_SIZE;
        const uint8_t *block_start = corpus + base * TRINE_S1_DIMS;

        for (size_t j = 0; j < TRINE_BATCH_BLOCK_SIZE; j++) {
            const uint8_t *vec = block_start + j * TRINE_S1_DIMS;
            sims[base + j] = batch_cosine(query, vec, query_mag_sq);
        }
    }

    /* Remainder (< TRINE_BATCH_BLOCK_SIZE vectors) */
    if (remainder > 0) {
        size_t base = full_blocks * TRINE_BATCH_BLOCK_SIZE;
        const uint8_t *block_start = corpus + base * TRINE_S1_DIMS;

        for (size_t j = 0; j < remainder; j++) {
            const uint8_t *vec = block_start + j * TRINE_S1_DIMS;
            sims[base + j] = batch_cosine(query, vec, query_mag_sq);
        }
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * IV. BATCH COMPARE — TOP-K WITH MIN-HEAP
 * ═══════════════════════════════════════════════════════════════════════ */

size_t trine_batch_compare_topk(
    const uint8_t *query,
    const uint8_t *corpus,
    size_t n,
    size_t top_k,
    size_t *top_k_idx,
    float *top_k_sim)
{
    if (!query || !corpus || !top_k_idx || !top_k_sim)
        return 0;
    if (n == 0 || top_k == 0)
        return 0;

    /* Clamp top_k to n */
    if (top_k > n) top_k = n;

    /* Precompute query magnitude squared once */
    uint64_t query_mag_sq = compute_mag_sq(query);

    /* Use the output arrays as the heap storage.
     * heap[i].sim  -> top_k_sim[i]
     * heap[i].idx  -> top_k_idx[i]
     * We use a local array of heap_entry_t for clean heap operations,
     * then copy results to the output arrays at the end. */

    /* Stack-allocate heap for small top_k, otherwise the caller is
     * expected to use reasonable values. We use a VLA-safe approach
     * with a fixed maximum stack allocation and fallback. */
    heap_entry_t heap_stack[256];
    heap_entry_t *heap;
    int heap_on_stack = 1;

    if (top_k <= 256) {
        heap = heap_stack;
    } else {
        heap = (heap_entry_t *)malloc(top_k * sizeof(heap_entry_t));
        if (!heap) return 0;
        heap_on_stack = 0;
    }

    size_t heap_size = 0;

    /* Process in blocks of TRINE_BATCH_BLOCK_SIZE */
    size_t full_blocks = n / TRINE_BATCH_BLOCK_SIZE;
    size_t remainder   = n % TRINE_BATCH_BLOCK_SIZE;

    /* Macro-like inline: process a single corpus vector at index i */
    #define PROCESS_VECTOR(i, vec_ptr) do {                              \
        float sim = batch_cosine(query, (vec_ptr), query_mag_sq);        \
        if (heap_size < top_k) {                                         \
            /* Filling phase: insert into heap */                        \
            heap[heap_size].sim = sim;                                   \
            heap[heap_size].idx = (i);                                   \
            heap_size++;                                                 \
            heap_sift_up(heap, heap_size - 1);                           \
        } else if (sim > heap[0].sim) {                                  \
            /* Replace min and restore heap property */                  \
            heap[0].sim = sim;                                           \
            heap[0].idx = (i);                                           \
            heap_sift_down(heap, heap_size, 0);                          \
        }                                                                \
    } while (0)

    /* Full blocks */
    for (size_t b = 0; b < full_blocks; b++) {
        size_t base = b * TRINE_BATCH_BLOCK_SIZE;
        const uint8_t *block_start = corpus + base * TRINE_S1_DIMS;

        for (size_t j = 0; j < TRINE_BATCH_BLOCK_SIZE; j++) {
            const uint8_t *vec = block_start + j * TRINE_S1_DIMS;
            PROCESS_VECTOR(base + j, vec);
        }
    }

    /* Remainder */
    if (remainder > 0) {
        size_t base = full_blocks * TRINE_BATCH_BLOCK_SIZE;
        const uint8_t *block_start = corpus + base * TRINE_S1_DIMS;

        for (size_t j = 0; j < remainder; j++) {
            const uint8_t *vec = block_start + j * TRINE_S1_DIMS;
            PROCESS_VECTOR(base + j, vec);
        }
    }

    #undef PROCESS_VECTOR

    /* Extract results from heap in descending order.
     * The heap is a min-heap, so repeated extraction gives ascending
     * order. We extract to the end of the output arrays and the
     * result is naturally in descending order. */
    size_t result_count = heap_size;

    for (size_t i = result_count; i > 0; i--) {
        /* heap[0] is the current minimum — place it at position i-1
         * (filling from the back gives descending order) */
        top_k_idx[i - 1] = heap[0].idx;
        top_k_sim[i - 1] = heap[0].sim;

        /* Move last element to root and shrink heap */
        heap_size--;
        if (heap_size > 0) {
            heap[0] = heap[heap_size];
            heap_sift_down(heap, heap_size, 0);
        }
    }

    if (!heap_on_stack) {
        free(heap);
    }

    return result_count;
}
