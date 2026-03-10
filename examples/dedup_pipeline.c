/* =====================================================================
 * dedup_pipeline.c -- TRINE Stage-1 Deduplication Pipeline Demo
 * =====================================================================
 *
 * Demonstrates:
 *   - Building an in-memory embedding index incrementally
 *   - Checking each new line against existing entries for near-duplicates
 *   - Reporting duplicates with similarity scores and matched entries
 *   - Summary statistics at the end
 *
 * Usage:
 *   ./dedup_pipeline                 # Run with hardcoded demo lines
 *   echo "lines..." | ./dedup_pipeline --stdin   # Read from stdin
 *
 * The program processes lines one at a time. For each line:
 *   1. Encode it to a 240-dim ternary embedding
 *   2. Query the index for the best match
 *   3. If the match exceeds the threshold, flag as duplicate
 *   4. Otherwise, add to the index as a new unique entry
 *
 * Build (from project root):
 *   cc -O2 -Wall -Wextra -o dedup_pipeline examples/dedup_pipeline.c \
 *      src/encode/trine_encode.c src/compare/trine_stage1.c \
 *      -Isrc/encode -Isrc/compare -lm
 *
 * ===================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trine_stage1.h"

/* ---- Hardcoded demo corpus ------------------------------------------ */

/* A mix of unique lines and near-duplicates to demonstrate detection.
 * Duplicates are intentional variations: reworded, reordered, or with
 * minor edits. */
static const char *DEMO_LINES[] = {
    /* Group 1: Server error reports */
    "ERROR: database connection timeout after 30 seconds",
    "WARNING: disk usage at 92 percent on server alpha",
    "ERROR: database connection timed out after 30 sec",        /* dup of 0 */

    /* Group 2: Meeting notes */
    "Meeting scheduled for Friday at 2pm in conference room B",
    "Quarterly review with the engineering team next Monday",
    "Meeting scheduled Friday 2pm conference room B",           /* dup of 3 */

    /* Group 3: Customer support */
    "Customer requested refund for order number 98712",
    "How do I reset my password on the admin portal",
    "Customer is requesting a refund for order 98712",          /* dup of 6 */

    /* Group 4: Code review comments */
    "Please add error handling for the null pointer case",
    "The variable naming could be more descriptive here",
    "Please add error handling for null pointer cases",         /* dup of 9 */

    /* Group 5: Unrelated unique lines */
    "New firmware version 3.2.1 available for download",
    "The annual company picnic will be held on June 15th",
    "WARNING: disk usage at 92 percent on server alpha node",   /* dup of 1 */
};

#define NUM_DEMO_LINES ((int)(sizeof(DEMO_LINES) / sizeof(DEMO_LINES[0])))

/* ---- Configuration -------------------------------------------------- */

/* Threshold for raw cosine similarity. With calibration disabled,
 * raw scores are not inflated by sparsity adjustment. A threshold
 * of 0.60 catches true near-duplicates (reworded, reordered) while
 * rejecting topically unrelated texts. Tune this value for your
 * domain: lower = more aggressive dedup, higher = stricter. */
#define DEDUP_THRESHOLD 0.60f

/* ---- Processing ----------------------------------------------------- */

typedef struct {
    int total;
    int unique;
    int duplicates;
} dedup_stats_t;

/*
 * process_line -- Check a single line against the index, print result.
 *
 * If the line matches an existing entry above the threshold, it is
 * flagged as a duplicate. Otherwise, it is added to the index.
 */
static void process_line(trine_s1_index_t *idx, const char *line,
                         int line_num, dedup_stats_t *stats)
{
    /* Skip empty lines */
    size_t len = strlen(line);
    if (len == 0) return;

    /* Strip trailing newline if present */
    char buf[4096];
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    memcpy(buf, line, len);
    if (len > 0 && buf[len - 1] == '\n') {
        buf[len - 1] = '\0';
        len--;
    }
    buf[len] = '\0';

    if (len == 0) return;

    stats->total++;

    /* Encode the line */
    uint8_t emb[240];
    trine_s1_encode(buf, len, emb);

    /* Query the index for a match */
    trine_s1_result_t result = trine_s1_index_query(idx, emb);

    if (result.is_duplicate) {
        /* Found a near-duplicate */
        stats->duplicates++;
        const char *match_tag = trine_s1_index_tag(idx, result.matched_index);

        printf("  [DUP]  line %2d: \"%.50s%s\"\n",
               line_num, buf, len > 50 ? "..." : "");
        printf("         match #%d (sim=%.3f cal=%.3f): \"%.50s%s\"\n",
               result.matched_index,
               result.similarity,
               result.calibrated,
               match_tag ? match_tag : "(no tag)",
               match_tag && strlen(match_tag) > 50 ? "..." : "");
        printf("\n");
    } else {
        /* Unique -- add to index */
        stats->unique++;
        int idx_num = trine_s1_index_add(idx, emb, buf);
        printf("  [NEW]  line %2d: #%-3d \"%.60s%s\"\n",
               line_num, idx_num, buf, len > 60 ? "..." : "");
    }
}

/* ---- Main ----------------------------------------------------------- */

int main(int argc, char *argv[])
{
    int use_stdin = 0;

    /* Check for --stdin flag */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = 1;
        }
    }

    printf("TRINE Stage-1 Deduplication Pipeline\n");
    printf("=====================================\n\n");

    /* Configure the index with our threshold and the DEDUP lens */
    trine_s1_config_t config = {
        .threshold = DEDUP_THRESHOLD,
        .lens = TRINE_S1_LENS_DEDUP,
        .calibrate_length = 0
    };

    printf("Config: threshold=%.2f, lens=DEDUP, calibration=off\n\n",
           config.threshold);

    trine_s1_index_t *idx = trine_s1_index_create(&config);
    if (!idx) {
        fprintf(stderr, "Failed to create index\n");
        return 1;
    }

    dedup_stats_t stats = {0, 0, 0};

    if (use_stdin) {
        /* Read lines from stdin */
        printf("Reading from stdin (one line per entry, Ctrl-D to end):\n\n");
        char line[4096];
        int line_num = 1;
        while (fgets(line, sizeof(line), stdin)) {
            process_line(idx, line, line_num, &stats);
            line_num++;
        }
    } else {
        /* Use hardcoded demo lines */
        printf("Processing %d demo lines:\n\n", NUM_DEMO_LINES);
        for (int i = 0; i < NUM_DEMO_LINES; i++) {
            process_line(idx, DEMO_LINES[i], i + 1, &stats);
        }
    }

    /* Print summary */
    printf("\n");
    printf("Summary\n");
    printf("-------\n");
    printf("  Total lines processed:  %d\n", stats.total);
    printf("  Unique entries:         %d\n", stats.unique);
    printf("  Duplicates found:       %d\n", stats.duplicates);
    printf("  Duplicate rate:         %.1f%%\n",
           stats.total > 0 ? 100.0 * stats.duplicates / stats.total : 0.0);
    printf("  Index size:             %d entries\n",
           trine_s1_index_count(idx));

    trine_s1_index_free(idx);

    return 0;
}
