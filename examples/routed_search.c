/* =====================================================================
 * routed_search.c -- TRINE Routed Index Demo (Band-LSH Sublinear Query)
 * =====================================================================
 *
 * Demonstrates:
 *   - Building a routed index from a sample corpus
 *   - Querying with routing statistics (candidates checked, speedup)
 *   - Comparing brute-force (linear scan) vs routed query performance
 *
 * The routed index uses Band-LSH (Locality-Sensitive Hashing) to
 * partition embeddings into buckets by chain. At query time, only
 * entries sharing at least one bucket with the query are compared,
 * reducing the number of full cosine comparisons from O(N) to a
 * much smaller candidate set.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -o routed_search routed_search.c \
 *      ../trine_encode.c ../trine_stage1.c ../trine_route.c -I.. -lm
 *
 * ===================================================================== */

#include <stdio.h>
#include <string.h>

#include "trine_stage1.h"
#include "trine_route.h"

/* ---- Sample corpus -------------------------------------------------- */

/* 50 short texts across several topic clusters. The routed index should
 * be able to find near-matches efficiently by only scanning candidates
 * that share LSH bucket keys with the query. */
static const char *CORPUS[] = {
    /* Cluster: Software engineering (0-9) */
    "implement binary search tree insertion",
    "fix null pointer dereference in parser",
    "add unit tests for authentication module",
    "refactor database connection pool logic",
    "optimize query execution plan for large tables",
    "update dependencies to latest stable versions",
    "implement rate limiting for the REST API",
    "fix memory leak in image processing pipeline",
    "add logging and monitoring for microservices",
    "implement websocket support for real-time updates",

    /* Cluster: Cooking (10-19) */
    "preheat oven to 350 degrees fahrenheit",
    "dice the onions and mince the garlic cloves",
    "simmer the tomato sauce for thirty minutes",
    "fold the egg whites gently into the batter",
    "season with salt pepper and fresh herbs",
    "let the dough rise for two hours",
    "grill the chicken until internal temp reaches 165",
    "whisk together flour sugar and baking powder",
    "saute the vegetables in olive oil until tender",
    "garnish with fresh parsley and a squeeze of lemon",

    /* Cluster: Finance (20-29) */
    "quarterly earnings exceeded analyst expectations",
    "the federal reserve raised interest rates by 25 basis points",
    "diversify your portfolio across multiple asset classes",
    "year over year revenue growth reached twelve percent",
    "the bond market is pricing in a rate cut next quarter",
    "startup raised series B funding of forty million dollars",
    "inflation data came in below consensus forecast",
    "the stock dropped seven percent after the earnings miss",
    "compound interest is the most powerful force in investing",
    "hedge fund liquidated its position in technology stocks",

    /* Cluster: Medicine (30-39) */
    "patient presents with acute lower back pain",
    "prescribe ibuprofen 400mg three times daily",
    "blood pressure reading was 140 over 90 mmHg",
    "schedule a follow up MRI in six weeks",
    "the lab results show elevated white blood cell count",
    "recommend physical therapy twice per week",
    "patient reports persistent headache for three days",
    "review family history of cardiovascular disease",
    "administer flu vaccine before the start of season",
    "allergic reaction to penicillin noted in chart",

    /* Cluster: Travel (40-49) */
    "flight departs at 7am from terminal B gate 42",
    "book a hotel room near the city center for three nights",
    "the train from Paris to London takes about two hours",
    "pack light and bring a universal power adapter",
    "visa application requires proof of accommodation",
    "the museum is closed on Mondays but open weekends",
    "exchange currency at the airport or use ATM abroad",
    "rent a car for the drive along the coastal highway",
    "travel insurance covers medical emergencies abroad",
    "check in online 24 hours before departure",
};

#define CORPUS_SIZE ((int)(sizeof(CORPUS) / sizeof(CORPUS[0])))

/* ---- Queries -------------------------------------------------------- */

/* Queries designed to match various corpus entries. Some are close
 * paraphrases, others share topical vocabulary. */
static const char *QUERIES[] = {
    "fix a null pointer bug in the parser module",
    "preheat the oven to 350 degrees",
    "the federal reserve increased interest rates",
    "patient has acute pain in the lower back",
    "book a hotel near downtown for three nights",
    "implement rate limiting on the API endpoints",
    "whisk flour and sugar together with baking powder",
    "the stock fell after missing earnings estimates",
};

#define NUM_QUERIES ((int)(sizeof(QUERIES) / sizeof(QUERIES[0])))

/* ---- Helpers -------------------------------------------------------- */

static void print_separator(int width)
{
    for (int i = 0; i < width; i++)
        putchar('=');
    putchar('\n');
}

static void print_dash(int width)
{
    for (int i = 0; i < width; i++)
        putchar('-');
    putchar('\n');
}

/* ---- Main ----------------------------------------------------------- */

int main(void)
{
    printf("TRINE Routed Index Search Demo\n");
    print_separator(60);
    printf("\n");

    /* Configuration: use DEDUP lens for semantic matching.
     * Calibration is off so raw cosine is used directly for ranking. */
    trine_s1_config_t config = {
        .threshold = 0.50f,
        .lens = TRINE_S1_LENS_DEDUP,
        .calibrate_length = 0
    };

    /* Create both a routed index and a brute-force index */
    trine_route_t *routed = trine_route_create(&config);
    trine_s1_index_t *brute = trine_s1_index_create(&config);

    if (!routed || !brute) {
        fprintf(stderr, "Failed to create indexes\n");
        return 1;
    }

    /* Populate both indexes with the same corpus */
    printf("Building indexes from %d corpus entries...\n", CORPUS_SIZE);

    for (int i = 0; i < CORPUS_SIZE; i++) {
        uint8_t emb[240];
        trine_s1_encode(CORPUS[i], strlen(CORPUS[i]), emb);

        trine_route_add(routed, emb, CORPUS[i]);
        trine_s1_index_add(brute, emb, CORPUS[i]);
    }

    printf("  Routed index:      %d entries\n", trine_route_count(routed));
    printf("  Brute-force index: %d entries\n", trine_s1_index_count(brute));
    printf("\n");

    /* Run queries against both indexes and compare results */
    print_separator(60);
    printf("Query Results\n");
    print_separator(60);
    printf("\n");

    int total_routed_checks = 0;
    int total_brute_checks = 0;

    for (int q = 0; q < NUM_QUERIES; q++) {
        const char *query = QUERIES[q];

        /* Encode the query */
        uint8_t qemb[240];
        trine_s1_encode(query, strlen(query), qemb);

        /* Routed query */
        trine_route_stats_t rstats;
        memset(&rstats, 0, sizeof(rstats));
        trine_s1_result_t rresult = trine_route_query(routed, qemb, &rstats);

        /* Brute-force query */
        trine_s1_result_t bresult = trine_s1_index_query(brute, qemb);

        total_routed_checks += rstats.candidates_checked;
        total_brute_checks += CORPUS_SIZE;

        printf("Query %d: \"%s\"\n", q + 1, query);
        print_dash(60);

        /* Routed result */
        if (rresult.is_duplicate && rresult.matched_index >= 0) {
            const char *tag = trine_route_tag(routed, rresult.matched_index);
            printf("  ROUTED  -> #%-2d (cal=%.3f) \"%s\"\n",
                   rresult.matched_index, rresult.calibrated,
                   tag ? tag : "(null)");
        } else {
            printf("  ROUTED  -> no match above threshold\n");
        }
        printf("            checked %d/%d candidates (%.1fx speedup)\n",
               rstats.candidates_checked, rstats.total_entries,
               rstats.speedup);

        /* Brute-force result */
        if (bresult.is_duplicate && bresult.matched_index >= 0) {
            const char *tag = trine_s1_index_tag(brute, bresult.matched_index);
            printf("  BRUTE   -> #%-2d (cal=%.3f) \"%s\"\n",
                   bresult.matched_index, bresult.calibrated,
                   tag ? tag : "(null)");
        } else {
            printf("  BRUTE   -> no match above threshold\n");
        }
        printf("            checked %d/%d candidates (1.0x)\n",
               CORPUS_SIZE, CORPUS_SIZE);

        /* Agreement check */
        if (rresult.matched_index == bresult.matched_index) {
            printf("  STATUS: AGREE (same best match)\n");
        } else if (rresult.is_duplicate && bresult.is_duplicate) {
            printf("  STATUS: DIFFER (different best match -- routing may "
                   "miss the global best)\n");
        } else {
            printf("  STATUS: DIFFER (match/no-match disagreement)\n");
        }

        printf("\n");
    }

    /* Overall routing statistics */
    print_separator(60);
    printf("Routing Summary\n");
    print_separator(60);
    printf("\n");
    printf("  Total queries:              %d\n", NUM_QUERIES);
    printf("  Corpus size:                %d entries\n", CORPUS_SIZE);
    printf("  Brute-force comparisons:    %d (%.0f per query)\n",
           total_brute_checks,
           (double)total_brute_checks / NUM_QUERIES);
    printf("  Routed comparisons:         %d (%.1f per query)\n",
           total_routed_checks,
           (double)total_routed_checks / NUM_QUERIES);

    if (total_routed_checks > 0) {
        printf("  Overall speedup:            %.1fx fewer comparisons\n",
               (double)total_brute_checks / total_routed_checks);
    } else {
        printf("  Overall speedup:            N/A (no candidates checked)\n");
    }

    printf("\n");
    printf("Notes:\n");
    printf("  - Routed search uses Band-LSH with %d bands of %d buckets each.\n",
           TRINE_ROUTE_BANDS, TRINE_ROUTE_BUCKETS);
    printf("  - Multi-probe with %d probes per band catches near-miss hashes.\n",
           TRINE_ROUTE_PROBES);
    printf("  - Speedup grows with corpus size. At 50 entries the advantage\n");
    printf("    is modest; at 10K+ entries routing typically checks <5%% of\n");
    printf("    the corpus while maintaining high recall.\n");
    printf("  - Routing is approximate: it may occasionally miss the global\n");
    printf("    best match if it lands in a non-overlapping bucket set.\n");

    /* Cleanup */
    trine_route_free(routed);
    trine_s1_index_free(brute);

    return 0;
}
