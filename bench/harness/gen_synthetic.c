/* =====================================================================
 * GEN_SYNTHETIC — Synthetic JSONL Training Data Generator
 * =====================================================================
 *
 * Generates synthetic JSONL training pairs for the Hebbian training
 * pipeline without requiring external datasets. Output format matches
 * trine_train expectations:
 *
 *   {"text_a": "...", "text_b": "...", "score": 0.85}
 *
 * Three pair categories:
 *   - Positive (score > 0.7): near-duplicate with minor edits
 *   - Negative (score < 0.3): unrelated sentences from different topics
 *   - Medium  (0.3 - 0.7):   partially overlapping sentences
 *
 * Usage:
 *   gen_synthetic [--count N] [--seed N]
 *
 * Build:
 *   cc -O2 -Wall -Wextra -Werror -o build/gen_synthetic \
 *      bench/harness/gen_synthetic.c -lm
 *
 * ===================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* =====================================================================
 * Configuration
 * ===================================================================== */

#define DEFAULT_COUNT    1000
#define DEFAULT_SEED     42
#define MAX_SENTENCE     512
#define NUM_TOPICS       10

/* =====================================================================
 * Deterministic RNG (xoshiro128**)
 * ===================================================================== */

static uint32_t rng_state[4];

static uint32_t rotl(uint32_t x, int k)
{
    return (x << k) | (x >> (32 - k));
}

static uint32_t rng_next(void)
{
    uint32_t result = rotl(rng_state[1] * 5, 7) * 9;
    uint32_t t = rng_state[1] << 9;

    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];

    rng_state[2] ^= t;
    rng_state[3] = rotl(rng_state[3], 11);

    return result;
}

static void rng_seed(uint32_t seed)
{
    /* SplitMix32 to initialize xoshiro state from a single seed */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b9u;
        uint32_t z = seed;
        z = (z ^ (z >> 16)) * 0x85ebca6bu;
        z = (z ^ (z >> 13)) * 0xc2b2ae35u;
        z = z ^ (z >> 16);
        rng_state[i] = z;
    }
}

/* Random integer in [0, max) */
static uint32_t rng_range(uint32_t max)
{
    if (max <= 1) return 0;
    return rng_next() % max;
}

/* Random float in [lo, hi] */
static float rng_float(float lo, float hi)
{
    float t = (float)(rng_next() & 0x00FFFFFFu) / (float)0x00FFFFFFu;
    return lo + t * (hi - lo);
}

/* =====================================================================
 * Vocabulary (~200 common English words organized by topic)
 * ===================================================================== */

/* Topic 0: Technology */
static const char *vocab_tech[] = {
    "computer", "software", "network", "database", "algorithm",
    "program", "system", "server", "memory", "processor",
    "binary", "digital", "interface", "protocol", "storage",
    "hardware", "circuit", "cache", "kernel", "compiler"
};
#define N_TECH 20

/* Topic 1: Nature */
static const char *vocab_nature[] = {
    "mountain", "river", "forest", "ocean", "valley",
    "desert", "island", "meadow", "glacier", "canyon",
    "waterfall", "prairie", "volcano", "reef", "tundra",
    "swamp", "plateau", "lagoon", "ridge", "cliff"
};
#define N_NATURE 20

/* Topic 2: Food */
static const char *vocab_food[] = {
    "bread", "cheese", "apple", "chicken", "pasta",
    "salad", "pepper", "onion", "tomato", "garlic",
    "butter", "cream", "sugar", "flour", "olive",
    "lemon", "ginger", "basil", "rice", "potato"
};
#define N_FOOD 20

/* Topic 3: Science */
static const char *vocab_science[] = {
    "molecule", "particle", "element", "reaction", "compound",
    "energy", "gravity", "electron", "spectrum", "pressure",
    "velocity", "mass", "density", "nucleus", "photon",
    "quantum", "neutron", "plasma", "isotope", "catalyst"
};
#define N_SCIENCE 20

/* Topic 4: Music */
static const char *vocab_music[] = {
    "melody", "rhythm", "chord", "harmony", "tempo",
    "guitar", "piano", "violin", "trumpet", "drums",
    "singer", "concert", "album", "studio", "microphone",
    "orchestra", "symphony", "chorus", "sonata", "ballad"
};
#define N_MUSIC 20

/* Topic 5: Sports */
static const char *vocab_sports[] = {
    "football", "basketball", "tennis", "swimming", "running",
    "soccer", "baseball", "volleyball", "hockey", "cycling",
    "stadium", "champion", "athlete", "coach", "tournament",
    "victory", "defense", "offense", "player", "referee"
};
#define N_SPORTS 20

/* Topic 6: Weather */
static const char *vocab_weather[] = {
    "sunshine", "rainfall", "thunder", "lightning", "blizzard",
    "hurricane", "tornado", "drought", "breeze", "hailstorm",
    "overcast", "humidity", "forecast", "climate", "monsoon",
    "cyclone", "rainbow", "frost", "sleet", "snowfall"
};
#define N_WEATHER 20

/* Topic 7: Architecture */
static const char *vocab_arch[] = {
    "building", "bridge", "tower", "column", "ceiling",
    "foundation", "staircase", "balcony", "corridor", "dome",
    "archway", "facade", "terrace", "skylight", "basement",
    "pillar", "rooftop", "courtyard", "entrance", "chamber"
};
#define N_ARCH 20

/* Topic 8: Travel */
static const char *vocab_travel[] = {
    "airport", "railway", "passport", "luggage", "destination",
    "journey", "tourist", "hotel", "ticket", "departure",
    "customs", "terminal", "itinerary", "boarding", "transit",
    "arrival", "cruise", "backpack", "excursion", "souvenir"
};
#define N_TRAVEL 20

/* Topic 9: Education */
static const char *vocab_edu[] = {
    "student", "teacher", "lecture", "textbook", "classroom",
    "homework", "diploma", "research", "library", "campus",
    "semester", "professor", "thesis", "curriculum", "workshop",
    "graduate", "scholar", "syllabus", "tutorial", "knowledge"
};
#define N_EDU 20

/* Master vocabulary table */
static const char **vocab_topics[NUM_TOPICS] = {
    vocab_tech, vocab_nature, vocab_food, vocab_science, vocab_music,
    vocab_sports, vocab_weather, vocab_arch, vocab_travel, vocab_edu
};

static const int vocab_sizes[NUM_TOPICS] = {
    N_TECH, N_NATURE, N_FOOD, N_SCIENCE, N_MUSIC,
    N_SPORTS, N_WEATHER, N_ARCH, N_TRAVEL, N_EDU
};

/* =====================================================================
 * Common adjectives and verbs (shared across topics)
 * ===================================================================== */

static const char *adjectives[] = {
    "large", "small", "bright", "dark", "fast",
    "slow", "heavy", "light", "strong", "gentle",
    "ancient", "modern", "complex", "simple", "elegant"
};
#define N_ADJ 15

static const char *verbs[] = {
    "creates", "explores", "produces", "measures", "requires",
    "displays", "contains", "supports", "combines", "generates",
    "reveals", "transforms", "processes", "maintains", "delivers"
};
#define N_VERB 15

/* =====================================================================
 * Sentence templates (~50 patterns)
 * ===================================================================== */

/* Template slots:
 *   %N = topic noun, %A = adjective, %V = verb, %N2 = second topic noun
 */

typedef enum {
    SLOT_NOUN,
    SLOT_ADJ,
    SLOT_VERB,
    SLOT_NOUN2
} slot_type_t;

typedef struct {
    const char *prefix;
    slot_type_t slot1;
    const char *mid1;
    slot_type_t slot2;
    const char *mid2;
    slot_type_t slot3;
    const char *suffix;
    int num_slots;  /* 2 or 3 */
} template_t;

static const template_t templates[] = {
    /* 2-slot templates */
    {"the ",       SLOT_NOUN, " is very ",     SLOT_ADJ,  "",            SLOT_NOUN, "",        2},
    {"a ",         SLOT_ADJ,  " ",             SLOT_NOUN, " appeared",   SLOT_NOUN, "",        2},
    {"every ",     SLOT_NOUN, " needs proper ",SLOT_NOUN2,"",            SLOT_NOUN, "",        2},
    {"this ",      SLOT_NOUN, " ",             SLOT_VERB, " results",    SLOT_NOUN, "",        2},
    {"the ",       SLOT_ADJ,  " ",             SLOT_NOUN, " is useful",  SLOT_NOUN, "",        2},
    {"observe the ",SLOT_NOUN," carefully and ",SLOT_VERB,"",            SLOT_NOUN, "",        2},
    {"we found a ",SLOT_NOUN, " near the ",    SLOT_NOUN2,"",            SLOT_NOUN, "",        2},
    {"the old ",   SLOT_NOUN, " was quite ",   SLOT_ADJ,  "",            SLOT_NOUN, "",        2},
    {"consider the ",SLOT_NOUN," and its ",    SLOT_NOUN2,"",            SLOT_NOUN, "",        2},
    {"the ",       SLOT_NOUN, " ",             SLOT_VERB, " slowly",     SLOT_NOUN, "",        2},

    /* 3-slot templates */
    {"the ",       SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " well",   3},
    {"a ",         SLOT_NOUN, " ",             SLOT_VERB, " the ",       SLOT_NOUN2," today",  3},
    {"our ",       SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " here",   3},
    {"the ",       SLOT_NOUN, " and ",         SLOT_NOUN2," are ",       SLOT_ADJ,  "",        3},
    {"many ",      SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " daily",  3},
    {"each ",      SLOT_NOUN, " ",             SLOT_VERB, " a ",         SLOT_ADJ,  " way",    3},
    {"the ",       SLOT_NOUN, " was ",         SLOT_ADJ,  " and ",       SLOT_VERB, "",        3},
    {"inside the ",SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " often",  3},
    {"without a ", SLOT_NOUN, " the ",         SLOT_NOUN2," ",           SLOT_VERB, "",        3},
    {"beyond the ",SLOT_NOUN, " lies a ",      SLOT_ADJ,  " ",           SLOT_NOUN2,"",        3},

    {"that ",      SLOT_NOUN, " ",             SLOT_VERB, " the ",       SLOT_ADJ,  " part",   3},
    {"under the ", SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " softly", 3},
    {"through the ",SLOT_NOUN," one can ",     SLOT_VERB, " the ",       SLOT_NOUN2,"",        3},
    {"despite the ",SLOT_ADJ, " ",             SLOT_NOUN, " it ",        SLOT_VERB, "",        3},
    {"near the ",  SLOT_NOUN, " a ",           SLOT_ADJ,  " ",           SLOT_NOUN2," stood",  3},

    {"the ",       SLOT_NOUN, " ",             SLOT_VERB, " with ",      SLOT_ADJ,  " force",  3},
    {"once a ",    SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, "",        3},
    {"above the ", SLOT_NOUN, " the ",         SLOT_NOUN2," ",           SLOT_VERB, "",        3},
    {"from the ",  SLOT_ADJ,  " ",             SLOT_NOUN, " came a ",    SLOT_NOUN2,"",        3},
    {"along the ", SLOT_NOUN, " ",             SLOT_VERB, " a ",         SLOT_ADJ,  " note",   3},

    {"the ",       SLOT_NOUN, " always ",      SLOT_VERB, " ",           SLOT_ADJ,  " things", 3},
    {"before the ",SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " away",   3},
    {"with every ",SLOT_NOUN, " the ",         SLOT_NOUN2," ",           SLOT_VERB, "",        3},
    {"after the ", SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " again",  3},
    {"only the ",  SLOT_NOUN, " can ",         SLOT_VERB, " the ",       SLOT_ADJ,  " one",    3},

    {"behind the ",SLOT_NOUN, " ",             SLOT_VERB, " a ",         SLOT_NOUN2,"",        3},
    {"among the ", SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " silently",3},
    {"within the ",SLOT_NOUN, " the ",         SLOT_ADJ,  " ",           SLOT_NOUN2," waits",  3},
    {"upon the ",  SLOT_NOUN, " ",             SLOT_VERB, " the ",       SLOT_ADJ,  " form",   3},
    {"against the ",SLOT_ADJ, " ",             SLOT_NOUN, " ",           SLOT_VERB, " firmly", 3},

    {"like a ",    SLOT_ADJ,  " ",             SLOT_NOUN, " it ",        SLOT_VERB, "",        3},
    {"over the ",  SLOT_NOUN, " the ",         SLOT_NOUN2," ",           SLOT_VERB, " high",   3},
    {"beside the ",SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " gently", 3},
    {"around the ",SLOT_NOUN, " ",             SLOT_VERB, " a ",         SLOT_NOUN2,"",        3},
    {"across the ",SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " twice",  3},

    {"toward the ",SLOT_NOUN, " the ",         SLOT_ADJ,  " ",           SLOT_NOUN2," moved",  3},
    {"until the ", SLOT_NOUN, " ",             SLOT_VERB, " the ",       SLOT_ADJ,  " edge",   3},
    {"for every ", SLOT_NOUN, " there is a ",  SLOT_ADJ,  " ",           SLOT_NOUN2,"",        3},
    {"during the ",SLOT_ADJ,  " ",             SLOT_NOUN, " ",           SLOT_VERB, " slowly", 3},
    {"about the ", SLOT_NOUN, " we ",          SLOT_VERB, " ",           SLOT_ADJ,  " facts",  3},
};
#define N_TEMPLATES 50

/* =====================================================================
 * Sentence generation
 * ===================================================================== */

/* Fill a slot from the given topic */
static const char *fill_slot(slot_type_t stype, int topic)
{
    switch (stype) {
    case SLOT_NOUN:
        return vocab_topics[topic][rng_range((uint32_t)vocab_sizes[topic])];
    case SLOT_ADJ:
        return adjectives[rng_range(N_ADJ)];
    case SLOT_VERB:
        return verbs[rng_range(N_VERB)];
    case SLOT_NOUN2:
        return vocab_topics[topic][rng_range((uint32_t)vocab_sizes[topic])];
    }
    return "thing";
}

/* Generate a sentence from a template using words from a given topic.
 * Returns the length written (excluding NUL). */
static int generate_sentence(char *buf, size_t buf_size, int topic)
{
    int tmpl_idx = (int)rng_range(N_TEMPLATES);
    const template_t *t = &templates[tmpl_idx];

    const char *w1 = fill_slot(t->slot1, topic);
    const char *w2 = fill_slot(t->slot2, topic);

    int len;
    if (t->num_slots == 2) {
        len = snprintf(buf, buf_size, "%s%s%s%s%s",
                       t->prefix, w1, t->mid1, w2, t->mid2);
    } else {
        const char *w3 = fill_slot(t->slot3, topic);
        len = snprintf(buf, buf_size, "%s%s%s%s%s%s%s",
                       t->prefix, w1, t->mid1, w2, t->mid2, w3, t->suffix);
    }

    if (len < 0 || (size_t)len >= buf_size)
        len = (int)(buf_size - 1);

    return len;
}

/* =====================================================================
 * Sentence perturbation (for positive pairs)
 * ===================================================================== */

/* Copy src into dst, applying minor character-level perturbations:
 *   - swap adjacent characters (~10% probability per position)
 *   - delete a character (~3% probability per position)
 *   - duplicate a character (~3% probability per position)
 * Returns the length of the perturbed string. */
static int perturb_chars(const char *src, char *dst, size_t dst_size)
{
    size_t slen = strlen(src);
    size_t out = 0;

    for (size_t i = 0; i < slen && out < dst_size - 2; i++) {
        uint32_t r = rng_range(100);
        if (r < 10 && i + 1 < slen && out + 1 < dst_size - 1) {
            /* Swap adjacent characters */
            dst[out++] = src[i + 1];
            dst[out++] = src[i];
            i++;  /* skip next, already used */
        } else if (r < 13 && slen > 10) {
            /* Delete character (skip it) */
            continue;
        } else if (r < 16 && out + 1 < dst_size - 1) {
            /* Duplicate character */
            dst[out++] = src[i];
            dst[out++] = src[i];
        } else {
            dst[out++] = src[i];
        }
    }

    dst[out] = '\0';
    return (int)out;
}

/* Word-level reordering: split on spaces, shuffle some words.
 * Returns the length written. */
static int reorder_words(const char *src, char *dst, size_t dst_size)
{
    /* Tokenize a copy */
    char copy[MAX_SENTENCE];
    strncpy(copy, src, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = '\0';

    char *words[64];
    int nwords = 0;
    char *tok = strtok(copy, " ");
    while (tok && nwords < 64) {
        words[nwords++] = tok;
        tok = strtok(NULL, " ");
    }

    /* Swap 1-2 random adjacent word pairs */
    int swaps = 1 + (int)rng_range(2);
    for (int s = 0; s < swaps && nwords > 1; s++) {
        int idx = (int)rng_range((uint32_t)(nwords - 1));
        char *tmp = words[idx];
        words[idx] = words[idx + 1];
        words[idx + 1] = tmp;
    }

    /* Reassemble */
    size_t out = 0;
    for (int w = 0; w < nwords && out < dst_size - 1; w++) {
        if (w > 0 && out < dst_size - 1) dst[out++] = ' ';
        size_t wlen = strlen(words[w]);
        if (out + wlen >= dst_size - 1) wlen = dst_size - 1 - out;
        memcpy(dst + out, words[w], wlen);
        out += wlen;
    }
    dst[out] = '\0';
    return (int)out;
}

/* Create a near-duplicate of src using one of several perturbation strategies */
static int make_near_duplicate(const char *src, char *dst, size_t dst_size)
{
    uint32_t strategy = rng_range(3);
    switch (strategy) {
    case 0:
        return perturb_chars(src, dst, dst_size);
    case 1:
        return reorder_words(src, dst, dst_size);
    case 2: {
        /* Combine both: reorder then perturb */
        char tmp[MAX_SENTENCE];
        reorder_words(src, tmp, sizeof(tmp));
        return perturb_chars(tmp, dst, dst_size);
    }
    default:
        return perturb_chars(src, dst, dst_size);
    }
}

/* =====================================================================
 * Medium-similarity pair generation
 *
 * Shares some words from the same topic but uses a different template,
 * producing partial lexical overlap.
 * ===================================================================== */

static int generate_medium_pair(char *buf_a, size_t size_a,
                                char *buf_b, size_t size_b,
                                int topic)
{
    /* Generate two sentences from the same topic but different templates.
     * This naturally creates partial word overlap. */
    generate_sentence(buf_a, size_a, topic);

    /* For the second, pick a nearby topic occasionally for variety */
    int topic_b = topic;
    if (rng_range(3) == 0) {
        topic_b = (topic + 1 + (int)rng_range((uint32_t)(NUM_TOPICS - 1))) % NUM_TOPICS;
    }
    generate_sentence(buf_b, size_b, topic_b);

    return 0;
}

/* =====================================================================
 * JSON string escaping
 * ===================================================================== */

static void emit_json_string(FILE *fp, const char *s)
{
    fputc('"', fp);
    while (*s) {
        switch (*s) {
        case '"':  fputs("\\\"", fp); break;
        case '\\': fputs("\\\\", fp); break;
        case '\n': fputs("\\n", fp);  break;
        case '\r': fputs("\\r", fp);  break;
        case '\t': fputs("\\t", fp);  break;
        default:
            if ((unsigned char)*s < 0x20) {
                fprintf(fp, "\\u%04x", (unsigned char)*s);
            } else {
                fputc(*s, fp);
            }
            break;
        }
        s++;
    }
    fputc('"', fp);
}

/* =====================================================================
 * JSONL emission
 * ===================================================================== */

static void emit_pair(FILE *fp, const char *text_a, const char *text_b,
                      float score)
{
    fputs("{\"text_a\": ", fp);
    emit_json_string(fp, text_a);
    fputs(", \"text_b\": ", fp);
    emit_json_string(fp, text_b);
    fprintf(fp, ", \"score\": %.4f}\n", (double)score);
}

/* =====================================================================
 * Main generator
 * ===================================================================== */

static void generate_pairs(int count, FILE *fp)
{
    /* Distribution: ~33% positive, ~33% negative, ~34% medium */
    int n_pos = count / 3;
    int n_neg = count / 3;
    int n_med = count - n_pos - n_neg;

    char buf_a[MAX_SENTENCE];
    char buf_b[MAX_SENTENCE];

    int pair_idx = 0;

    /* ── Positive pairs (score > 0.7) ─────────────────────────────── */
    for (int i = 0; i < n_pos; i++) {
        int topic = (int)rng_range(NUM_TOPICS);
        generate_sentence(buf_a, sizeof(buf_a), topic);
        make_near_duplicate(buf_a, buf_b, sizeof(buf_b));
        float score = rng_float(0.71f, 0.98f);
        emit_pair(fp, buf_a, buf_b, score);
        pair_idx++;
    }

    /* ── Negative pairs (score < 0.3) ─────────────────────────────── */
    for (int i = 0; i < n_neg; i++) {
        int topic_a = (int)rng_range(NUM_TOPICS);
        int topic_b = (topic_a + 1 + (int)rng_range((uint32_t)(NUM_TOPICS - 1))) % NUM_TOPICS;
        generate_sentence(buf_a, sizeof(buf_a), topic_a);
        generate_sentence(buf_b, sizeof(buf_b), topic_b);
        float score = rng_float(0.01f, 0.29f);
        emit_pair(fp, buf_a, buf_b, score);
        pair_idx++;
    }

    /* ── Medium pairs (0.3 - 0.7) ─────────────────────────────────── */
    for (int i = 0; i < n_med; i++) {
        int topic = (int)rng_range(NUM_TOPICS);
        generate_medium_pair(buf_a, sizeof(buf_a), buf_b, sizeof(buf_b), topic);
        float score = rng_float(0.30f, 0.70f);
        emit_pair(fp, buf_a, buf_b, score);
        pair_idx++;
    }

    (void)pair_idx;
}

/* =====================================================================
 * CLI
 * ===================================================================== */

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [--count N] [--seed N]\n"
        "\n"
        "Generate synthetic JSONL training data for trine_train.\n"
        "\n"
        "Options:\n"
        "  --count N   Number of pairs to generate (default: %d)\n"
        "  --seed  N   RNG seed for reproducibility (default: %d)\n"
        "  --help      Show this message\n",
        prog, DEFAULT_COUNT, DEFAULT_SEED);
}

int main(int argc, char *argv[])
{
    int count = DEFAULT_COUNT;
    uint32_t seed = DEFAULT_SEED;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--count") == 0 && i + 1 < argc) {
            count = atoi(argv[++i]);
            if (count < 1) {
                fprintf(stderr, "error: --count must be >= 1\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Initialize RNG */
    rng_seed(seed);

    /* Generate to stdout */
    generate_pairs(count, stdout);

    return 0;
}
