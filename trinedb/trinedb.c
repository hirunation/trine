/* =====================================================================
 * TrineDB — Minimal REST Server for the TRINE Embedding Library
 * Version 1.0.1
 * =====================================================================
 *
 * Single-file HTTP server exposing TRINE shingle embeddings, lens-weighted
 * comparison, and routed-index operations via REST/JSON endpoints.
 *
 * ZERO external dependencies: POSIX sockets + libtrine (4 C files).
 *
 * Endpoints:
 *   POST /embed          — Encode text to 240-trit embedding
 *   POST /compare        — Compare two texts with lens weighting
 *   POST /index/add      — Add document to index
 *   POST /index/add_batch— Batch add documents
 *   POST /query          — Query index for nearest match
 *   POST /dedup          — Batch dedup with cluster output
 *   GET  /health         — Health check
 *   GET  /stats          — Index statistics
 *   POST /index/save     — Persist index to disk
 *   POST /index/load     — Load index from disk
 *   POST /index/clear    — Clear index
 *
 * Build:
 *   make          (see companion Makefile)
 *
 * Usage:
 *   trinedb -p 7319 -d ./trinedb_data
 *
 * ===================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "trine_encode.h"
#include "trine_stage1.h"
#include "trine_route.h"
#include "trine_canon.h"

/* =====================================================================
 * Constants
 * ===================================================================== */

#define TRINEDB_VERSION       "1.0.1"
#define TRINEDB_DEFAULT_PORT  7319       /* "TRIN" on phone keypad        */
#define TRINEDB_MAX_BODY      (10 * 1024 * 1024)  /* 10 MB max request  */
#define TRINEDB_RECV_BUF      (16 * 1024)          /* 16 KB recv chunk  */
#define TRINEDB_HEADER_MAX    8192       /* Max HTTP header size          */
#define TRINEDB_PATH_MAX      256        /* Max URL path length           */
#define TRINEDB_RESP_BUF      (1024 * 1024)  /* 1 MB response buffer    */
#define TRINEDB_JSON_BUF      4096       /* Per-field JSON buffer         */
#define TRINEDB_CANON_BUF     (256 * 1024)  /* 256 KB canon buffer      */
#define TRINEDB_TAG_MAX       1024       /* Max tag length                */
#define TRINEDB_FILEPATH_MAX  4096       /* Max file path length          */
#define TRINEDB_FULLPATH_MAX  8192       /* Composed path (dir + file)    */
#define TRINEDB_LISTEN_BACKLOG 64        /* Listen queue depth            */

/* =====================================================================
 * Global State
 * ===================================================================== */

static volatile sig_atomic_t g_running = 1;

typedef struct {
    /* Server config */
    int              port;
    char             data_dir[TRINEDB_FILEPATH_MAX];
    char             load_path[TRINEDB_FILEPATH_MAX];
    float            threshold;
    int              canon_preset;
    int              recall_mode;
    int              quiet;

    /* Runtime state */
    trine_route_t   *index;
    trine_s1_config_t config;
    time_t           start_time;
    int              listen_fd;
    char             lens_name[32];
} trinedb_state_t;

static trinedb_state_t g_state;

/* =====================================================================
 * Logging
 * ===================================================================== */

static void log_info(const char *fmt, ...)
{
    if (g_state.quiet) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[trinedb] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

static void log_error(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[trinedb ERROR] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

/* =====================================================================
 * Signal Handling
 * ===================================================================== */

static void signal_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

static void install_signals(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    /* Ignore SIGPIPE from closed connections */
    signal(SIGPIPE, SIG_IGN);
}

/* =====================================================================
 * JSON Helpers — Minimal Parser & Generator
 * =====================================================================
 *
 * Same approach as trine_dedup.c: scan for keys, extract values.
 * No full JSON parser — just enough for our specific request formats.
 *
 * ===================================================================== */

/* --- JSON Output Escaping --- */

static size_t json_escape(char *out, size_t out_cap,
                           const char *src, size_t src_len)
{
    size_t w = 0;
    for (size_t i = 0; i < src_len && w + 6 < out_cap; i++) {
        unsigned char c = (unsigned char)src[i];
        switch (c) {
        case '"':  out[w++] = '\\'; out[w++] = '"';  break;
        case '\\': out[w++] = '\\'; out[w++] = '\\'; break;
        case '\n': out[w++] = '\\'; out[w++] = 'n';  break;
        case '\r': out[w++] = '\\'; out[w++] = 'r';  break;
        case '\t': out[w++] = '\\'; out[w++] = 't';  break;
        default:
            if (c < 0x20) {
                w += (size_t)snprintf(out + w, out_cap - w,
                                      "\\u%04x", (unsigned)c);
            } else {
                out[w++] = (char)c;
            }
            break;
        }
    }
    if (w < out_cap) out[w] = '\0';
    return w;
}

/* --- JSON Input Parsing --- */

/*
 * json_find_string — Find "key": "value" in JSON text.
 * Returns pointer to value start (after opening quote), sets *vlen.
 * Returns NULL if not found.
 */
static const char *json_find_string(const char *json, size_t json_len,
                                     const char *key, size_t *vlen)
{
    if (!json || !key || !vlen) return NULL;
    size_t klen = strlen(key);
    const char *end = json + json_len;
    const char *p = json;

    while (p < end) {
        const char *q1 = memchr(p, '"', (size_t)(end - p));
        if (!q1 || q1 + klen + 1 >= end) return NULL;

        if (memcmp(q1 + 1, key, klen) == 0 && q1[klen + 1] == '"') {
            const char *after_key = q1 + klen + 2;
            while (after_key < end && (*after_key == ' ' || *after_key == '\t'
                   || *after_key == '\n' || *after_key == '\r'))
                after_key++;
            if (after_key >= end || *after_key != ':') { p = after_key; continue; }
            after_key++;
            while (after_key < end && (*after_key == ' ' || *after_key == '\t'
                   || *after_key == '\n' || *after_key == '\r'))
                after_key++;
            if (after_key >= end || *after_key != '"') { p = after_key; continue; }
            after_key++;
            const char *vstart = after_key;
            const char *vp = vstart;
            while (vp < end) {
                if (*vp == '\\' && vp + 1 < end) { vp += 2; continue; }
                if (*vp == '"') { *vlen = (size_t)(vp - vstart); return vstart; }
                vp++;
            }
            return NULL;
        }
        q1++;
        while (q1 < end) {
            if (*q1 == '\\' && q1 + 1 < end) { q1 += 2; continue; }
            if (*q1 == '"') { q1++; break; }
            q1++;
        }
        p = q1;
    }
    return NULL;
}

/*
 * json_find_number — Find "key": <number> in JSON text.
 * Returns 1 if found and sets *out. Returns 0 if not found.
 */
static int json_find_number(const char *json, size_t json_len,
                             const char *key, double *out)
{
    if (!json || !key || !out) return 0;
    size_t klen = strlen(key);
    const char *end = json + json_len;
    const char *p = json;

    while (p < end) {
        const char *q1 = memchr(p, '"', (size_t)(end - p));
        if (!q1 || q1 + klen + 1 >= end) return 0;

        if (memcmp(q1 + 1, key, klen) == 0 && q1[klen + 1] == '"') {
            const char *after_key = q1 + klen + 2;
            while (after_key < end && (*after_key == ' ' || *after_key == '\t'
                   || *after_key == '\n' || *after_key == '\r'))
                after_key++;
            if (after_key >= end || *after_key != ':') { p = after_key; continue; }
            after_key++;
            while (after_key < end && (*after_key == ' ' || *after_key == '\t'
                   || *after_key == '\n' || *after_key == '\r'))
                after_key++;
            if (after_key >= end) return 0;
            char *endp = NULL;
            *out = strtod(after_key, &endp);
            if (endp > after_key) return 1;
            return 0;
        }
        q1++;
        while (q1 < end) {
            if (*q1 == '\\' && q1 + 1 < end) { q1 += 2; continue; }
            if (*q1 == '"') { q1++; break; }
            q1++;
        }
        p = q1;
    }
    return 0;
}

/*
 * json_unescape — Unescape a JSON string value in-place.
 * Returns unescaped length.
 */
static size_t json_unescape(char *buf, size_t len)
{
    size_t r = 0, w = 0;
    while (r < len) {
        if (buf[r] == '\\' && r + 1 < len) {
            char c = buf[r + 1];
            switch (c) {
            case '"':  buf[w++] = '"';  r += 2; break;
            case '\\': buf[w++] = '\\'; r += 2; break;
            case 'n':  buf[w++] = '\n'; r += 2; break;
            case 't':  buf[w++] = '\t'; r += 2; break;
            case 'r':  buf[w++] = '\r'; r += 2; break;
            case '/':  buf[w++] = '/';  r += 2; break;
            default:   buf[w++] = buf[r++]; break;
            }
        } else {
            buf[w++] = buf[r++];
        }
    }
    buf[w] = '\0';
    return w;
}

/*
 * json_extract_string — Extract a string field from JSON, heap-allocated.
 * Caller must free the result. Returns NULL if not found.
 */
static char *json_extract_string(const char *json, size_t json_len,
                                  const char *key)
{
    size_t vlen = 0;
    const char *v = json_find_string(json, json_len, key, &vlen);
    if (!v) return NULL;
    char *out = malloc(vlen + 1);
    if (!out) return NULL;
    memcpy(out, v, vlen);
    out[vlen] = '\0';
    json_unescape(out, vlen);
    return out;
}

/*
 * json_find_array — Find the start of a JSON array value for "key": [...].
 * Returns pointer to the '[' character, sets *arr_len to length
 * including brackets. Returns NULL if not found.
 */
static const char *json_find_array(const char *json, size_t json_len,
                                    const char *key, size_t *arr_len)
{
    if (!json || !key || !arr_len) return NULL;
    size_t klen = strlen(key);
    const char *end = json + json_len;
    const char *p = json;

    while (p < end) {
        const char *q1 = memchr(p, '"', (size_t)(end - p));
        if (!q1 || q1 + klen + 1 >= end) return NULL;

        if (memcmp(q1 + 1, key, klen) == 0 && q1[klen + 1] == '"') {
            const char *after_key = q1 + klen + 2;
            while (after_key < end && (*after_key == ' ' || *after_key == '\t'
                   || *after_key == '\n' || *after_key == '\r'))
                after_key++;
            if (after_key >= end || *after_key != ':') { p = after_key; continue; }
            after_key++;
            while (after_key < end && (*after_key == ' ' || *after_key == '\t'
                   || *after_key == '\n' || *after_key == '\r'))
                after_key++;
            if (after_key >= end || *after_key != '[') { p = after_key; continue; }

            /* Find matching ']', accounting for nesting and strings */
            const char *arr_start = after_key;
            const char *ap = after_key + 1;
            int depth = 1;
            int in_string = 0;
            while (ap < end && depth > 0) {
                if (in_string) {
                    if (*ap == '\\' && ap + 1 < end) { ap += 2; continue; }
                    if (*ap == '"') in_string = 0;
                } else {
                    if (*ap == '"') in_string = 1;
                    else if (*ap == '[') depth++;
                    else if (*ap == ']') depth--;
                }
                ap++;
            }
            if (depth == 0) {
                *arr_len = (size_t)(ap - arr_start);
                return arr_start;
            }
            return NULL;
        }
        q1++;
        while (q1 < end) {
            if (*q1 == '\\' && q1 + 1 < end) { q1 += 2; continue; }
            if (*q1 == '"') { q1++; break; }
            q1++;
        }
        p = q1;
    }
    return NULL;
}

/*
 * json_iter_array_objects — Iterate over objects in a JSON array.
 *
 * Given a pointer to '[', calls callback for each {...} object found.
 * The callback receives a pointer to the '{' and the object's length
 * (including braces).
 *
 * Returns the number of objects iterated.
 */
typedef void (*json_object_cb)(const char *obj, size_t obj_len, void *ctx);

static int json_iter_array_objects(const char *arr, size_t arr_len,
                                    json_object_cb cb, void *ctx)
{
    if (!arr || arr_len < 2 || arr[0] != '[') return 0;
    const char *end = arr + arr_len;
    const char *p = arr + 1; /* skip '[' */
    int count = 0;

    while (p < end) {
        /* Skip whitespace and commas */
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n'
               || *p == '\r' || *p == ','))
            p++;
        if (p >= end || *p == ']') break;

        if (*p == '{') {
            /* Find matching '}' */
            const char *obj_start = p;
            p++;
            int depth = 1;
            int in_string = 0;
            while (p < end && depth > 0) {
                if (in_string) {
                    if (*p == '\\' && p + 1 < end) { p += 2; continue; }
                    if (*p == '"') in_string = 0;
                } else {
                    if (*p == '"') in_string = 1;
                    else if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                }
                p++;
            }
            if (depth == 0) {
                size_t obj_len_val = (size_t)(p - obj_start);
                if (cb) cb(obj_start, obj_len_val, ctx);
                count++;
            }
        } else {
            /* Skip non-object value */
            p++;
        }
    }
    return count;
}

/* =====================================================================
 * Lens Name Mapping
 * ===================================================================== */

typedef struct {
    const char   *name;
    trine_s1_lens_t lens;
} lens_entry_t;

static const lens_entry_t LENS_TABLE[] = {
    { "uniform", {.weights = {1.0f, 1.0f, 1.0f, 1.0f}} },
    { "dedup",   {.weights = {0.5f, 0.5f, 0.7f, 1.0f}} },
    { "edit",    {.weights = {1.0f, 0.3f, 0.1f, 0.0f}} },
    { "vocab",   {.weights = {0.0f, 0.2f, 0.3f, 1.0f}} },
    { "code",    {.weights = {1.0f, 0.8f, 0.4f, 0.2f}} },
    { "legal",   {.weights = {0.2f, 0.4f, 1.0f, 0.8f}} },
    { "medical", {.weights = {0.3f, 1.0f, 0.6f, 0.5f}} },
    { "support", {.weights = {0.2f, 0.4f, 0.7f, 1.0f}} },
    { "policy",  {.weights = {0.1f, 0.3f, 1.0f, 0.8f}} },
    { NULL,      {.weights = {0}} }
};

static int resolve_lens(const char *name, trine_s1_lens_t *out)
{
    if (!name || !out) return -1;
    for (int i = 0; LENS_TABLE[i].name; i++) {
        if (strcmp(name, LENS_TABLE[i].name) == 0) {
            *out = LENS_TABLE[i].lens;
            return 0;
        }
    }
    return -1;
}

static const char *recall_mode_str(int mode)
{
    switch (mode) {
    case TRINE_RECALL_FAST:     return "fast";
    case TRINE_RECALL_BALANCED: return "balanced";
    case TRINE_RECALL_STRICT:   return "strict";
    default:                    return "unknown";
    }
}

static int parse_recall_mode(const char *s)
{
    if (!s) return -1;
    if (strcmp(s, "fast") == 0)     return TRINE_RECALL_FAST;
    if (strcmp(s, "balanced") == 0) return TRINE_RECALL_BALANCED;
    if (strcmp(s, "strict") == 0)   return TRINE_RECALL_STRICT;
    return -1;
}

/* =====================================================================
 * Canonicalization Helper
 * ===================================================================== */

/*
 * apply_canon — Apply canonicalization to text, returning a heap-allocated
 * result. If preset is 0 (NONE), returns a copy of the input.
 * Caller must free the result. Sets *out_len.
 */
static char *apply_canon(const char *text, size_t len, int preset, size_t *out_len)
{
    if (preset == TRINE_CANON_NONE) {
        char *copy = malloc(len + 1);
        if (!copy) return NULL;
        memcpy(copy, text, len);
        copy[len] = '\0';
        *out_len = len;
        return copy;
    }

    size_t cap = len + 1;
    if (cap < 256) cap = 256;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    size_t olen = 0;
    if (trine_canon_apply(text, len, preset, buf, cap, &olen) != 0) {
        free(buf);
        return NULL;
    }
    *out_len = olen;
    return buf;
}

/* =====================================================================
 * HTTP Parser — Minimal HTTP/1.1
 * =====================================================================
 *
 * Parses method, path, Content-Length, and reads body for POST.
 * Single-connection model: Connection: close after every response.
 *
 * ===================================================================== */

typedef struct {
    char     method[16];
    char     path[TRINEDB_PATH_MAX];
    size_t   content_length;
    int      has_content_length;
    char    *body;
    size_t   body_len;
    int      valid;
} http_request_t;

/*
 * recv_all — Receive exactly n bytes from socket.
 * Returns 0 on success, -1 on error/disconnect.
 */
static int recv_all(int fd, char *buf, size_t n)
{
    size_t total = 0;
    while (total < n) {
        ssize_t r = recv(fd, buf + total, n - total, 0);
        if (r <= 0) return -1;
        total += (size_t)r;
    }
    return 0;
}

/*
 * parse_request — Read and parse an HTTP request from socket.
 * Allocates body on heap if POST with content. Caller must free body.
 */
static void parse_request(int client_fd, http_request_t *req)
{
    memset(req, 0, sizeof(*req));
    req->valid = 0;

    /* Read headers into a buffer, byte-by-byte looking for \r\n\r\n.
     * In practice we read in chunks for efficiency. */
    char header_buf[TRINEDB_HEADER_MAX];
    size_t header_len = 0;
    int found_end = 0;

    while (header_len < TRINEDB_HEADER_MAX - 1) {
        ssize_t r = recv(client_fd, header_buf + header_len, 1, 0);
        if (r <= 0) return;
        header_len++;

        /* Check for \r\n\r\n end of headers */
        if (header_len >= 4 &&
            header_buf[header_len - 4] == '\r' &&
            header_buf[header_len - 3] == '\n' &&
            header_buf[header_len - 2] == '\r' &&
            header_buf[header_len - 1] == '\n') {
            found_end = 1;
            break;
        }
    }
    if (!found_end) return;
    header_buf[header_len] = '\0';

    /* Parse request line: METHOD PATH HTTP/1.x\r\n */
    char *line_end = strstr(header_buf, "\r\n");
    if (!line_end) return;

    /* Method */
    char *sp = memchr(header_buf, ' ', (size_t)(line_end - header_buf));
    if (!sp) return;
    size_t mlen = (size_t)(sp - header_buf);
    if (mlen >= sizeof(req->method)) return;
    memcpy(req->method, header_buf, mlen);
    req->method[mlen] = '\0';

    /* Path */
    sp++;
    char *sp2 = memchr(sp, ' ', (size_t)(line_end - sp));
    if (!sp2) return;
    size_t plen = (size_t)(sp2 - sp);
    if (plen >= sizeof(req->path)) return;
    memcpy(req->path, sp, plen);
    req->path[plen] = '\0';

    /* Strip query string for routing */
    char *qmark = strchr(req->path, '?');
    if (qmark) *qmark = '\0';

    /* Parse Content-Length header */
    const char *cl_header = "Content-Length:";
    size_t cl_hlen = strlen(cl_header);
    const char *h = header_buf;
    while ((h = strstr(h, "\r\n")) != NULL) {
        h += 2;
        if (*h == '\r') break; /* end of headers */
        if (strncasecmp(h, cl_header, cl_hlen) == 0) {
            const char *val = h + cl_hlen;
            while (*val == ' ' || *val == '\t') val++;
            req->content_length = (size_t)strtoul(val, NULL, 10);
            req->has_content_length = 1;
            break;
        }
    }

    req->valid = 1;

    /* Read body if POST/PUT with Content-Length */
    if (req->has_content_length && req->content_length > 0) {
        if (req->content_length > TRINEDB_MAX_BODY) {
            req->valid = 0; /* oversized */
            return;
        }
        req->body = malloc(req->content_length + 1);
        if (!req->body) { req->valid = 0; return; }

        if (recv_all(client_fd, req->body, req->content_length) != 0) {
            free(req->body);
            req->body = NULL;
            req->valid = 0;
            return;
        }
        req->body[req->content_length] = '\0';
        req->body_len = req->content_length;
    }
}

/* =====================================================================
 * HTTP Response Helpers
 * ===================================================================== */

static void send_response(int fd, int status, const char *status_text,
                           const char *body, size_t body_len)
{
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, body_len);

    /* Send header + body. Ignore send errors on closed connections. */
    (void)send(fd, header, (size_t)hlen, MSG_NOSIGNAL);
    if (body_len > 0 && body) {
        (void)send(fd, body, body_len, MSG_NOSIGNAL);
    }
}

static void send_json(int fd, int status, const char *status_text,
                       const char *json)
{
    send_response(fd, status, status_text, json, strlen(json));
}

static void send_error(int fd, int status, const char *status_text,
                        const char *message)
{
    char buf[512];
    char escaped[256];
    json_escape(escaped, sizeof(escaped), message, strlen(message));
    int n = snprintf(buf, sizeof(buf),
        "{\"error\":\"%s\",\"status\":%d}", escaped, status);
    send_response(fd, status, status_text, buf, (size_t)n);
}

static void send_options(int fd)
{
    const char *resp =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Connection: close\r\n"
        "\r\n";
    (void)send(fd, resp, strlen(resp), MSG_NOSIGNAL);
}

/* =====================================================================
 * Endpoint Handlers
 * ===================================================================== */

/* --- POST /embed --- */
static void handle_embed(int fd, const http_request_t *req)
{
    if (!req->body || req->body_len == 0) {
        send_error(fd, 400, "Bad Request", "Missing request body");
        return;
    }

    char *text = json_extract_string(req->body, req->body_len, "text");
    if (!text) {
        send_error(fd, 400, "Bad Request", "Missing 'text' field");
        return;
    }

    /* Optional canon preset */
    int preset = g_state.canon_preset;
    double d_preset;
    if (json_find_number(req->body, req->body_len, "canon", &d_preset)) {
        preset = (int)d_preset;
        if (preset < 0 || preset > 4) {
            free(text);
            send_error(fd, 400, "Bad Request", "Invalid canon preset (0-4)");
            return;
        }
    }

    /* Canonicalize */
    size_t canon_len = 0;
    char *canon = apply_canon(text, strlen(text), preset, &canon_len);
    free(text);
    if (!canon) {
        send_error(fd, 500, "Internal Server Error", "Canonicalization failed");
        return;
    }

    /* Encode */
    uint8_t emb[240];
    trine_encode_shingle(canon, canon_len, emb);
    float fill = trine_s1_fill_ratio(emb);
    free(canon);

    /* Build response: {"trits":[...], "fill_ratio": 0.42} */
    char *resp = malloc(TRINEDB_RESP_BUF);
    if (!resp) { send_error(fd, 500, "Internal Server Error", "OOM"); return; }

    int pos = 0;
    pos += snprintf(resp + pos, TRINEDB_RESP_BUF - (size_t)pos, "{\"trits\":[");
    for (int i = 0; i < 240; i++) {
        if (i > 0) resp[pos++] = ',';
        pos += snprintf(resp + pos, TRINEDB_RESP_BUF - (size_t)pos, "%d", emb[i]);
    }
    pos += snprintf(resp + pos, TRINEDB_RESP_BUF - (size_t)pos,
        "],\"fill_ratio\":%.4f}", fill);

    send_response(fd, 200, "OK", resp, (size_t)pos);
    free(resp);
}

/* --- POST /compare --- */
static void handle_compare(int fd, const http_request_t *req)
{
    if (!req->body || req->body_len == 0) {
        send_error(fd, 400, "Bad Request", "Missing request body");
        return;
    }

    char *text_a = json_extract_string(req->body, req->body_len, "a");
    char *text_b = json_extract_string(req->body, req->body_len, "b");
    if (!text_a || !text_b) {
        free(text_a); free(text_b);
        send_error(fd, 400, "Bad Request", "Missing 'a' and/or 'b' fields");
        return;
    }

    /* Optional lens */
    trine_s1_lens_t lens = g_state.config.lens;
    char *lens_name = json_extract_string(req->body, req->body_len, "lens");
    if (lens_name) {
        if (resolve_lens(lens_name, &lens) != 0) {
            free(text_a); free(text_b); free(lens_name);
            send_error(fd, 400, "Bad Request", "Unknown lens name");
            return;
        }
        free(lens_name);
    }

    /* Optional canon */
    int preset = g_state.canon_preset;
    double d_preset;
    if (json_find_number(req->body, req->body_len, "canon", &d_preset)) {
        preset = (int)d_preset;
        if (preset < 0 || preset > 4) {
            free(text_a); free(text_b);
            send_error(fd, 400, "Bad Request", "Invalid canon preset (0-4)");
            return;
        }
    }

    /* Canonicalize */
    size_t len_a, len_b;
    char *ca = apply_canon(text_a, strlen(text_a), preset, &len_a);
    char *cb = apply_canon(text_b, strlen(text_b), preset, &len_b);
    free(text_a); free(text_b);
    if (!ca || !cb) {
        free(ca); free(cb);
        send_error(fd, 500, "Internal Server Error", "Canonicalization failed");
        return;
    }

    /* Encode both */
    uint8_t emb_a[240], emb_b[240];
    trine_encode_shingle(ca, len_a, emb_a);
    trine_encode_shingle(cb, len_b, emb_b);
    free(ca); free(cb);

    float sim = trine_s1_compare(emb_a, emb_b, &lens);
    float fill_a = trine_s1_fill_ratio(emb_a);
    float fill_b = trine_s1_fill_ratio(emb_b);

    char resp[256];
    int n = snprintf(resp, sizeof(resp),
        "{\"similarity\":%.6f,\"fill_a\":%.4f,\"fill_b\":%.4f}",
        sim, fill_a, fill_b);
    send_response(fd, 200, "OK", resp, (size_t)n);
}

/* --- POST /index/add --- */
static void handle_index_add(int fd, const http_request_t *req)
{
    if (!req->body || req->body_len == 0) {
        send_error(fd, 400, "Bad Request", "Missing request body");
        return;
    }

    char *text = json_extract_string(req->body, req->body_len, "text");
    if (!text) {
        send_error(fd, 400, "Bad Request", "Missing 'text' field");
        return;
    }

    char *tag = json_extract_string(req->body, req->body_len, "tag");

    /* Optional canon */
    int preset = g_state.canon_preset;
    double d_preset;
    if (json_find_number(req->body, req->body_len, "canon", &d_preset)) {
        preset = (int)d_preset;
        if (preset < 0 || preset > 4) {
            free(text); free(tag);
            send_error(fd, 400, "Bad Request", "Invalid canon preset (0-4)");
            return;
        }
    }

    /* Canonicalize + encode */
    size_t canon_len;
    char *canon = apply_canon(text, strlen(text), preset, &canon_len);
    free(text);
    if (!canon) {
        free(tag);
        send_error(fd, 500, "Internal Server Error", "Canonicalization failed");
        return;
    }

    uint8_t emb[240];
    trine_encode_shingle(canon, canon_len, emb);
    free(canon);

    int id = trine_route_add(g_state.index, emb, tag);
    int count = trine_route_count(g_state.index);
    free(tag);

    if (id < 0) {
        send_error(fd, 500, "Internal Server Error", "Failed to add to index");
        return;
    }

    char resp[128];
    int n = snprintf(resp, sizeof(resp), "{\"id\":%d,\"count\":%d}", id, count);
    send_response(fd, 200, "OK", resp, (size_t)n);
}

/* --- POST /index/add_batch --- */

typedef struct {
    trine_route_t *index;
    int            preset;
    int            added;
    int            errors;
} batch_add_ctx_t;

static void batch_add_cb(const char *obj, size_t obj_len, void *ctx_ptr)
{
    batch_add_ctx_t *ctx = (batch_add_ctx_t *)ctx_ptr;

    char *text = json_extract_string(obj, obj_len, "text");
    if (!text) { ctx->errors++; return; }

    char *tag = json_extract_string(obj, obj_len, "tag");

    size_t canon_len;
    char *canon = apply_canon(text, strlen(text), ctx->preset, &canon_len);
    free(text);
    if (!canon) { free(tag); ctx->errors++; return; }

    uint8_t emb[240];
    trine_encode_shingle(canon, canon_len, emb);
    free(canon);

    int id = trine_route_add(ctx->index, emb, tag);
    free(tag);

    if (id >= 0) ctx->added++;
    else         ctx->errors++;
}

static void handle_index_add_batch(int fd, const http_request_t *req)
{
    if (!req->body || req->body_len == 0) {
        send_error(fd, 400, "Bad Request", "Missing request body");
        return;
    }

    size_t arr_len = 0;
    const char *arr = json_find_array(req->body, req->body_len,
                                       "documents", &arr_len);
    if (!arr) {
        send_error(fd, 400, "Bad Request", "Missing 'documents' array");
        return;
    }

    /* Optional canon */
    int preset = g_state.canon_preset;
    double d_preset;
    if (json_find_number(req->body, req->body_len, "canon", &d_preset)) {
        preset = (int)d_preset;
        if (preset < 0 || preset > 4) preset = g_state.canon_preset;
    }

    batch_add_ctx_t ctx = { g_state.index, preset, 0, 0 };
    json_iter_array_objects(arr, arr_len, batch_add_cb, &ctx);

    int count = trine_route_count(g_state.index);

    char resp[128];
    int n = snprintf(resp, sizeof(resp),
        "{\"added\":%d,\"count\":%d}", ctx.added, count);
    send_response(fd, 200, "OK", resp, (size_t)n);
}

/* --- POST /query --- */
static void handle_query(int fd, const http_request_t *req)
{
    if (!req->body || req->body_len == 0) {
        send_error(fd, 400, "Bad Request", "Missing request body");
        return;
    }

    char *text = json_extract_string(req->body, req->body_len, "text");
    if (!text) {
        send_error(fd, 400, "Bad Request", "Missing 'text' field");
        return;
    }

    /* Optional canon */
    int preset = g_state.canon_preset;
    double d_preset;
    if (json_find_number(req->body, req->body_len, "canon", &d_preset)) {
        preset = (int)d_preset;
        if (preset < 0 || preset > 4) preset = g_state.canon_preset;
    }

    size_t canon_len;
    char *canon = apply_canon(text, strlen(text), preset, &canon_len);
    free(text);
    if (!canon) {
        send_error(fd, 500, "Internal Server Error", "Canonicalization failed");
        return;
    }

    uint8_t emb[240];
    trine_encode_shingle(canon, canon_len, emb);
    free(canon);

    trine_route_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    trine_s1_result_t result = trine_route_query(g_state.index, emb, &stats);

    /* Get tag of matched entry */
    const char *tag = "";
    if (result.matched_index >= 0) {
        const char *t = trine_route_tag(g_state.index, result.matched_index);
        if (t) tag = t;
    }

    char *resp = malloc(1024);
    if (!resp) { send_error(fd, 500, "Internal Server Error", "OOM"); return; }

    char tag_esc[TRINEDB_TAG_MAX * 6];
    json_escape(tag_esc, sizeof(tag_esc), tag, strlen(tag));

    int n = snprintf(resp, 1024,
        "{\"is_duplicate\":%s,"
        "\"similarity\":%.6f,"
        "\"calibrated\":%.6f,"
        "\"matched_index\":%d,"
        "\"tag\":\"%s\","
        "\"stats\":{"
            "\"candidates\":%d,"
            "\"total\":%d,"
            "\"speedup\":%.1f"
        "}}",
        result.is_duplicate ? "true" : "false",
        result.similarity,
        result.calibrated,
        result.matched_index,
        tag_esc,
        stats.candidates_checked,
        stats.total_entries,
        stats.speedup);

    send_response(fd, 200, "OK", resp, (size_t)n);
    free(resp);
}

/* --- POST /dedup --- */

/*
 * Dedup cluster tracking. We build a temporary routed index and detect
 * duplicate clusters as we add documents.
 */

#define DEDUP_MAX_CLUSTERS 10000

typedef struct {
    char  *canonical_tag;
    char **dup_tags;
    int    dup_count;
    int    dup_capacity;
} dedup_cluster_t;

typedef struct {
    dedup_cluster_t *clusters;
    int              count;
    int             *cluster_map; /* doc index -> cluster index, -1 if canonical */
    int              unique;
    int              duplicates;
} dedup_state_t;

static void dedup_state_free(dedup_state_t *ds, int doc_count)
{
    if (!ds) return;
    for (int i = 0; i < ds->count; i++) {
        free(ds->clusters[i].canonical_tag);
        for (int j = 0; j < ds->clusters[i].dup_count; j++) {
            free(ds->clusters[i].dup_tags[j]);
        }
        free(ds->clusters[i].dup_tags);
    }
    free(ds->clusters);
    free(ds->cluster_map);
    (void)doc_count;
}

typedef struct {
    /* Input parsing */
    char **texts;
    char **tags;
    int   count;
    int   capacity;
    int   canon_preset;
} dedup_parse_ctx_t;

static void dedup_doc_cb(const char *obj, size_t obj_len, void *ctx_ptr)
{
    dedup_parse_ctx_t *ctx = (dedup_parse_ctx_t *)ctx_ptr;
    if (ctx->count >= ctx->capacity) {
        int new_cap = ctx->capacity * 2;
        char **new_texts = realloc(ctx->texts, sizeof(char *) * (size_t)new_cap);
        char **new_tags = realloc(ctx->tags, sizeof(char *) * (size_t)new_cap);
        if (!new_texts || !new_tags) return;
        ctx->texts = new_texts;
        ctx->tags = new_tags;
        ctx->capacity = new_cap;
    }

    char *text = json_extract_string(obj, obj_len, "text");
    char *tag = json_extract_string(obj, obj_len, "tag");
    if (!text) { free(tag); return; }

    ctx->texts[ctx->count] = text;
    ctx->tags[ctx->count] = tag;
    ctx->count++;
}

static void handle_dedup(int fd, const http_request_t *req)
{
    if (!req->body || req->body_len == 0) {
        send_error(fd, 400, "Bad Request", "Missing request body");
        return;
    }

    size_t arr_len = 0;
    const char *arr = json_find_array(req->body, req->body_len,
                                       "documents", &arr_len);
    if (!arr) {
        send_error(fd, 400, "Bad Request", "Missing 'documents' array");
        return;
    }

    /* Optional threshold override */
    float threshold = g_state.config.threshold;
    double d_thresh;
    if (json_find_number(req->body, req->body_len, "threshold", &d_thresh)) {
        threshold = (float)d_thresh;
        if (threshold < 0.0f || threshold > 1.0f) threshold = g_state.config.threshold;
    }

    /* Optional canon */
    int preset = g_state.canon_preset;
    double d_preset;
    if (json_find_number(req->body, req->body_len, "canon", &d_preset)) {
        preset = (int)d_preset;
        if (preset < 0 || preset > 4) preset = g_state.canon_preset;
    }

    /* Parse all documents */
    dedup_parse_ctx_t parse = { NULL, NULL, 0, 256, preset };
    parse.texts = malloc(sizeof(char *) * (size_t)parse.capacity);
    parse.tags = malloc(sizeof(char *) * (size_t)parse.capacity);
    if (!parse.texts || !parse.tags) {
        free(parse.texts); free(parse.tags);
        send_error(fd, 500, "Internal Server Error", "OOM");
        return;
    }

    json_iter_array_objects(arr, arr_len, dedup_doc_cb, &parse);

    if (parse.count == 0) {
        free(parse.texts); free(parse.tags);
        send_json(fd, 200, "OK",
            "{\"unique\":0,\"duplicates\":0,\"clusters\":[]}");
        return;
    }

    /* Create temporary index for dedup */
    trine_s1_config_t dedup_cfg = g_state.config;
    dedup_cfg.threshold = threshold;
    trine_route_t *dedup_idx = trine_route_create(&dedup_cfg);
    if (!dedup_idx) {
        for (int i = 0; i < parse.count; i++) { free(parse.texts[i]); free(parse.tags[i]); }
        free(parse.texts); free(parse.tags);
        send_error(fd, 500, "Internal Server Error", "Failed to create dedup index");
        return;
    }

    /* Dedup state */
    dedup_state_t ds;
    memset(&ds, 0, sizeof(ds));
    ds.clusters = calloc((size_t)parse.count, sizeof(dedup_cluster_t));
    ds.cluster_map = malloc(sizeof(int) * (size_t)parse.count);
    if (!ds.clusters || !ds.cluster_map) {
        trine_route_free(dedup_idx);
        for (int i = 0; i < parse.count; i++) { free(parse.texts[i]); free(parse.tags[i]); }
        free(parse.texts); free(parse.tags);
        free(ds.clusters); free(ds.cluster_map);
        send_error(fd, 500, "Internal Server Error", "OOM");
        return;
    }
    for (int i = 0; i < parse.count; i++) ds.cluster_map[i] = -1;

    /* Process each document */
    for (int i = 0; i < parse.count; i++) {
        size_t canon_len;
        char *canon = apply_canon(parse.texts[i], strlen(parse.texts[i]),
                                   preset, &canon_len);
        if (!canon) continue;

        uint8_t emb[240];
        trine_encode_shingle(canon, canon_len, emb);
        free(canon);

        /* Query existing index for duplicates */
        trine_s1_result_t result = trine_route_query(dedup_idx, emb, NULL);

        if (result.is_duplicate && result.matched_index >= 0) {
            /* Found a duplicate — add to the matched entry's cluster */
            int matched = result.matched_index;
            int cluster_idx = ds.cluster_map[matched];

            if (cluster_idx < 0) {
                /* Matched entry has no cluster yet — create one */
                cluster_idx = ds.count++;
                ds.cluster_map[matched] = cluster_idx;
                const char *ctag = parse.tags[matched] ? parse.tags[matched] : "";
                ds.clusters[cluster_idx].canonical_tag = strdup(ctag);
                ds.clusters[cluster_idx].dup_tags = NULL;
                ds.clusters[cluster_idx].dup_count = 0;
                ds.clusters[cluster_idx].dup_capacity = 0;
            }

            /* Add this doc as a duplicate in the cluster */
            dedup_cluster_t *cl = &ds.clusters[cluster_idx];
            if (cl->dup_count >= cl->dup_capacity) {
                int new_cap = cl->dup_capacity < 4 ? 4 : cl->dup_capacity * 2;
                char **new_tags = realloc(cl->dup_tags,
                    sizeof(char *) * (size_t)new_cap);
                if (!new_tags) continue;
                cl->dup_tags = new_tags;
                cl->dup_capacity = new_cap;
            }
            const char *dtag = parse.tags[i] ? parse.tags[i] : "";
            cl->dup_tags[cl->dup_count++] = strdup(dtag);
            ds.cluster_map[i] = cluster_idx;
            ds.duplicates++;
        } else {
            ds.unique++;
        }

        /* Add to the temporary index regardless */
        const char *tag = parse.tags[i] ? parse.tags[i] : "";
        trine_route_add(dedup_idx, emb, tag);
    }

    trine_route_free(dedup_idx);

    /* Build response JSON */
    size_t resp_cap = (size_t)(parse.count * 256 + 1024);
    if (resp_cap > 64 * 1024 * 1024) resp_cap = 64 * 1024 * 1024;
    char *resp = malloc(resp_cap);
    if (!resp) {
        dedup_state_free(&ds, parse.count);
        for (int i = 0; i < parse.count; i++) { free(parse.texts[i]); free(parse.tags[i]); }
        free(parse.texts); free(parse.tags);
        send_error(fd, 500, "Internal Server Error", "OOM");
        return;
    }

    int pos = 0;
    pos += snprintf(resp + pos, resp_cap - (size_t)pos,
        "{\"unique\":%d,\"duplicates\":%d,\"clusters\":[",
        ds.unique, ds.duplicates);

    for (int c = 0; c < ds.count; c++) {
        if (c > 0) resp[pos++] = ',';
        pos += snprintf(resp + pos, resp_cap - (size_t)pos, "{\"canonical\":");

        char esc[TRINEDB_TAG_MAX * 6];
        const char *ctag = ds.clusters[c].canonical_tag ?
                           ds.clusters[c].canonical_tag : "";
        json_escape(esc, sizeof(esc), ctag, strlen(ctag));
        pos += snprintf(resp + pos, resp_cap - (size_t)pos, "\"%s\"", esc);

        pos += snprintf(resp + pos, resp_cap - (size_t)pos, ",\"duplicates\":[");
        for (int d = 0; d < ds.clusters[c].dup_count; d++) {
            if (d > 0) resp[pos++] = ',';
            const char *dtag = ds.clusters[c].dup_tags[d] ?
                               ds.clusters[c].dup_tags[d] : "";
            json_escape(esc, sizeof(esc), dtag, strlen(dtag));
            pos += snprintf(resp + pos, resp_cap - (size_t)pos, "\"%s\"", esc);
        }
        pos += snprintf(resp + pos, resp_cap - (size_t)pos, "]}");
    }

    pos += snprintf(resp + pos, resp_cap - (size_t)pos, "]}");

    send_response(fd, 200, "OK", resp, (size_t)pos);

    free(resp);
    dedup_state_free(&ds, parse.count);
    for (int i = 0; i < parse.count; i++) {
        free(parse.texts[i]);
        free(parse.tags[i]);
    }
    free(parse.texts);
    free(parse.tags);
}

/* --- GET /health --- */
static void handle_health(int fd)
{
    time_t now = time(NULL);
    long uptime = (long)(now - g_state.start_time);
    int count = trine_route_count(g_state.index);

    char resp[256];
    int n = snprintf(resp, sizeof(resp),
        "{\"status\":\"ok\",\"version\":\"%s\","
        "\"index_count\":%d,\"uptime_seconds\":%ld}",
        TRINEDB_VERSION, count, uptime);
    send_response(fd, 200, "OK", resp, (size_t)n);
}

/* --- GET /stats --- */
static void handle_stats(int fd)
{
    int count = trine_route_count(g_state.index);
    int recall = trine_route_get_recall(g_state.index);

    char resp[256];
    int n = snprintf(resp, sizeof(resp),
        "{\"count\":%d,\"recall_mode\":\"%s\","
        "\"threshold\":%.2f,\"lens\":\"%s\"}",
        count, recall_mode_str(recall),
        g_state.config.threshold, g_state.lens_name);
    send_response(fd, 200, "OK", resp, (size_t)n);
}

/* --- POST /index/save --- */
static void handle_index_save(int fd, const http_request_t *req)
{
    char path[TRINEDB_FULLPATH_MAX];

    /* Default save path */
    snprintf(path, sizeof(path), "%s/index.trrt", g_state.data_dir);

    /* Optional custom path from body */
    if (req->body && req->body_len > 0) {
        char *custom = json_extract_string(req->body, req->body_len, "path");
        if (custom) {
            /* If path doesn't contain '/', prepend data_dir */
            if (!strchr(custom, '/')) {
                snprintf(path, sizeof(path), "%s/%s", g_state.data_dir, custom);
            } else {
                snprintf(path, sizeof(path), "%s", custom);
            }
            free(custom);
        }
    }

    int count = trine_route_count(g_state.index);

    if (trine_route_save(g_state.index, path) != 0) {
        send_error(fd, 500, "Internal Server Error", "Failed to save index");
        return;
    }

    log_info("Index saved: %s (%d entries)", path, count);

    char path_esc[TRINEDB_FULLPATH_MAX * 2];
    json_escape(path_esc, sizeof(path_esc), path, strlen(path));

    char resp[512];
    int n = snprintf(resp, sizeof(resp),
        "{\"saved\":true,\"path\":\"%s\",\"count\":%d}",
        path_esc, count);
    send_response(fd, 200, "OK", resp, (size_t)n);
}

/* --- POST /index/load --- */
static void handle_index_load(int fd, const http_request_t *req)
{
    char path[TRINEDB_FULLPATH_MAX];
    path[0] = '\0';

    if (req->body && req->body_len > 0) {
        char *custom = json_extract_string(req->body, req->body_len, "path");
        if (custom) {
            if (!strchr(custom, '/')) {
                snprintf(path, sizeof(path), "%s/%s", g_state.data_dir, custom);
            } else {
                snprintf(path, sizeof(path), "%s", custom);
            }
            free(custom);
        }
    }

    if (path[0] == '\0') {
        snprintf(path, sizeof(path), "%s/index.trrt", g_state.data_dir);
    }

    trine_route_t *new_idx = trine_route_load(path);
    if (!new_idx) {
        send_error(fd, 404, "Not Found", "Failed to load index file");
        return;
    }

    /* Replace current index */
    trine_route_free(g_state.index);
    g_state.index = new_idx;

    /* Restore recall mode */
    trine_route_set_recall(g_state.index, g_state.recall_mode);

    int count = trine_route_count(g_state.index);
    log_info("Index loaded: %s (%d entries)", path, count);

    char resp[128];
    int n = snprintf(resp, sizeof(resp),
        "{\"loaded\":true,\"count\":%d}", count);
    send_response(fd, 200, "OK", resp, (size_t)n);
}

/* --- POST /index/clear --- */
static void handle_index_clear(int fd)
{
    /* Free old index and create fresh one */
    trine_route_free(g_state.index);
    g_state.index = trine_route_create(&g_state.config);

    if (!g_state.index) {
        send_error(fd, 500, "Internal Server Error",
                   "Failed to recreate index after clear");
        return;
    }

    trine_route_set_recall(g_state.index, g_state.recall_mode);
    log_info("Index cleared");

    send_json(fd, 200, "OK", "{\"cleared\":true}");
}

/* =====================================================================
 * Request Router
 * ===================================================================== */

static void route_request(int client_fd, const http_request_t *req)
{
    /* OPTIONS preflight */
    if (strcmp(req->method, "OPTIONS") == 0) {
        send_options(client_fd);
        return;
    }

    /* GET endpoints */
    if (strcmp(req->method, "GET") == 0) {
        if (strcmp(req->path, "/health") == 0) {
            handle_health(client_fd);
            return;
        }
        if (strcmp(req->path, "/stats") == 0) {
            handle_stats(client_fd);
            return;
        }
        send_error(client_fd, 404, "Not Found", "Unknown endpoint");
        return;
    }

    /* POST endpoints */
    if (strcmp(req->method, "POST") == 0) {
        if (strcmp(req->path, "/embed") == 0) {
            handle_embed(client_fd, req);
            return;
        }
        if (strcmp(req->path, "/compare") == 0) {
            handle_compare(client_fd, req);
            return;
        }
        if (strcmp(req->path, "/index/add") == 0) {
            handle_index_add(client_fd, req);
            return;
        }
        if (strcmp(req->path, "/index/add_batch") == 0) {
            handle_index_add_batch(client_fd, req);
            return;
        }
        if (strcmp(req->path, "/query") == 0) {
            handle_query(client_fd, req);
            return;
        }
        if (strcmp(req->path, "/dedup") == 0) {
            handle_dedup(client_fd, req);
            return;
        }
        if (strcmp(req->path, "/index/save") == 0) {
            handle_index_save(client_fd, req);
            return;
        }
        if (strcmp(req->path, "/index/load") == 0) {
            handle_index_load(client_fd, req);
            return;
        }
        if (strcmp(req->path, "/index/clear") == 0) {
            handle_index_clear(client_fd);
            return;
        }
        send_error(client_fd, 404, "Not Found", "Unknown endpoint");
        return;
    }

    send_error(client_fd, 405, "Method Not Allowed", "Use GET or POST");
}

/* =====================================================================
 * Startup Banner
 * ===================================================================== */

static void print_banner(void)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "  TrineDB v%s — TRINE Embedding REST Server\n", TRINEDB_VERSION);
    fprintf(stderr, "  ======================================\n");
    fprintf(stderr, "  Port:          %d\n", g_state.port);
    fprintf(stderr, "  Data dir:      %s\n", g_state.data_dir);
    fprintf(stderr, "  Threshold:     %.2f\n", g_state.config.threshold);
    fprintf(stderr, "  Lens:          %s\n", g_state.lens_name);
    fprintf(stderr, "  Canon:         %s (%d)\n",
            trine_canon_preset_name(g_state.canon_preset), g_state.canon_preset);
    fprintf(stderr, "  Recall:        %s\n", recall_mode_str(g_state.recall_mode));
    fprintf(stderr, "  Index entries: %d\n",
            trine_route_count(g_state.index));
    fprintf(stderr, "  Max body:      %d MB\n",
            (int)(TRINEDB_MAX_BODY / (1024 * 1024)));
    fprintf(stderr, "\n");
    fprintf(stderr, "  Endpoints:\n");
    fprintf(stderr, "    POST /embed            Encode text\n");
    fprintf(stderr, "    POST /compare          Compare two texts\n");
    fprintf(stderr, "    POST /index/add        Add document\n");
    fprintf(stderr, "    POST /index/add_batch  Batch add documents\n");
    fprintf(stderr, "    POST /query            Query index\n");
    fprintf(stderr, "    POST /dedup            Batch dedup\n");
    fprintf(stderr, "    GET  /health           Health check\n");
    fprintf(stderr, "    GET  /stats            Index stats\n");
    fprintf(stderr, "    POST /index/save       Save index\n");
    fprintf(stderr, "    POST /index/load       Load index\n");
    fprintf(stderr, "    POST /index/clear      Clear index\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  Listening on http://0.0.0.0:%d\n", g_state.port);
    fprintf(stderr, "  Press Ctrl-C to stop (auto-saves index)\n\n");
}

/* =====================================================================
 * Auto-Save on Shutdown
 * ===================================================================== */

static void auto_save(void)
{
    int count = trine_route_count(g_state.index);
    if (count == 0) {
        log_info("Index empty, skipping auto-save.");
        return;
    }

    char path[TRINEDB_FULLPATH_MAX];
    snprintf(path, sizeof(path), "%s/index.trrt", g_state.data_dir);

    log_info("Auto-saving index (%d entries) to %s ...", count, path);
    if (trine_route_save(g_state.index, path) == 0) {
        log_info("Saved successfully.");
    } else {
        log_error("Auto-save failed!");
    }
}

/* =====================================================================
 * Usage / Help
 * ===================================================================== */

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "TrineDB v%s — TRINE Embedding REST Server\n"
        "\n"
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  -p PORT      Listen port (default: %d)\n"
        "  -d PATH      Data directory for persistence (default: ./trinedb_data)\n"
        "  -l PATH      Load existing .trrt index at startup\n"
        "  -t THRESH    Similarity threshold 0.0-1.0 (default: 0.60)\n"
        "  --lens NAME  Default lens (default: dedup)\n"
        "               Names: uniform dedup edit vocab code legal medical support policy\n"
        "  --canon N    Default canonicalization preset 0-4 (default: 0)\n"
        "               0=none, 1=support, 2=code, 3=policy, 4=general\n"
        "  --recall M   Recall mode: fast/balanced/strict (default: balanced)\n"
        "  -q           Quiet mode (suppress request logging)\n"
        "  -h           Show this help\n"
        "\n"
        "Examples:\n"
        "  %s                              # defaults, port 7319\n"
        "  %s -p 8080 -t 0.70             # port 8080, threshold 0.70\n"
        "  %s -l my_corpus.trrt            # load existing index\n"
        "  %s -d /data/trine --recall strict\n"
        "\n",
        TRINEDB_VERSION, prog, TRINEDB_DEFAULT_PORT,
        prog, prog, prog, prog);
}

/* =====================================================================
 * Main
 * ===================================================================== */

int main(int argc, char **argv)
{
    /* Default state */
    memset(&g_state, 0, sizeof(g_state));
    g_state.port = TRINEDB_DEFAULT_PORT;
    snprintf(g_state.data_dir, sizeof(g_state.data_dir), "./trinedb_data");
    g_state.threshold = 0.60f;
    g_state.canon_preset = TRINE_CANON_NONE;
    g_state.recall_mode = TRINE_RECALL_BALANCED;
    g_state.quiet = 0;
    snprintf(g_state.lens_name, sizeof(g_state.lens_name), "dedup");

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(arg, "-p") == 0 && i + 1 < argc) {
            g_state.port = atoi(argv[++i]);
            if (g_state.port <= 0 || g_state.port > 65535) {
                fprintf(stderr, "Error: invalid port %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(arg, "-d") == 0 && i + 1 < argc) {
            snprintf(g_state.data_dir, sizeof(g_state.data_dir), "%s", argv[++i]);
        }
        else if (strcmp(arg, "-l") == 0 && i + 1 < argc) {
            snprintf(g_state.load_path, sizeof(g_state.load_path), "%s", argv[++i]);
        }
        else if (strcmp(arg, "-t") == 0 && i + 1 < argc) {
            g_state.threshold = (float)atof(argv[++i]);
            if (g_state.threshold < 0.0f || g_state.threshold > 1.0f) {
                fprintf(stderr, "Error: threshold must be 0.0-1.0\n");
                return 1;
            }
        }
        else if (strcmp(arg, "--lens") == 0 && i + 1 < argc) {
            snprintf(g_state.lens_name, sizeof(g_state.lens_name), "%s", argv[++i]);
        }
        else if (strcmp(arg, "--canon") == 0 && i + 1 < argc) {
            g_state.canon_preset = atoi(argv[++i]);
            if (g_state.canon_preset < 0 || g_state.canon_preset > 4) {
                fprintf(stderr, "Error: canon preset must be 0-4\n");
                return 1;
            }
        }
        else if (strcmp(arg, "--recall") == 0 && i + 1 < argc) {
            int mode = parse_recall_mode(argv[++i]);
            if (mode < 0) {
                fprintf(stderr, "Error: recall mode must be fast/balanced/strict\n");
                return 1;
            }
            g_state.recall_mode = mode;
        }
        else if (strcmp(arg, "-q") == 0) {
            g_state.quiet = 1;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Resolve lens */
    trine_s1_lens_t lens;
    if (resolve_lens(g_state.lens_name, &lens) != 0) {
        fprintf(stderr, "Error: unknown lens '%s'\n", g_state.lens_name);
        return 1;
    }

    /* Build config */
    g_state.config.threshold = g_state.threshold;
    g_state.config.lens = lens;
    g_state.config.calibrate_length = 1;

    /* Create data directory */
    mkdir(g_state.data_dir, 0755);

    /* Create or load index */
    if (g_state.load_path[0] != '\0') {
        g_state.index = trine_route_load(g_state.load_path);
        if (!g_state.index) {
            fprintf(stderr, "Error: failed to load index from '%s'\n",
                    g_state.load_path);
            return 1;
        }
        trine_route_set_recall(g_state.index, g_state.recall_mode);
        fprintf(stderr, "Loaded index: %d entries from %s\n",
                trine_route_count(g_state.index), g_state.load_path);
    } else {
        g_state.index = trine_route_create(&g_state.config);
        if (!g_state.index) {
            fprintf(stderr, "Error: failed to create index\n");
            return 1;
        }
        trine_route_set_recall(g_state.index, g_state.recall_mode);

        /* Try to auto-load from data dir */
        char auto_path[TRINEDB_FULLPATH_MAX];
        snprintf(auto_path, sizeof(auto_path), "%s/index.trrt", g_state.data_dir);
        if (access(auto_path, R_OK) == 0) {
            trine_route_t *loaded = trine_route_load(auto_path);
            if (loaded) {
                trine_route_free(g_state.index);
                g_state.index = loaded;
                trine_route_set_recall(g_state.index, g_state.recall_mode);
                fprintf(stderr, "Auto-loaded index: %d entries from %s\n",
                        trine_route_count(g_state.index), auto_path);
            }
        }
    }

    /* Install signal handlers */
    install_signals();
    g_state.start_time = time(NULL);

    /* Create listen socket */
    g_state.listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_state.listen_fd < 0) {
        perror("socket");
        trine_route_free(g_state.index);
        return 1;
    }

    /* Allow port reuse */
    int opt = 1;
    setsockopt(g_state.listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)g_state.port);

    if (bind(g_state.listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(g_state.listen_fd);
        trine_route_free(g_state.index);
        return 1;
    }

    if (listen(g_state.listen_fd, TRINEDB_LISTEN_BACKLOG) < 0) {
        perror("listen");
        close(g_state.listen_fd);
        trine_route_free(g_state.index);
        return 1;
    }

    /* Print banner */
    print_banner();

    /* Main event loop */
    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(g_state.listen_fd,
                               (struct sockaddr *)&client_addr,
                               &client_len);
        if (client_fd < 0) {
            if (g_running) {
                /* Could be EINTR from signal — check g_running */
                if (errno == EINTR) continue;
                perror("accept");
            }
            break;
        }

        /* Set receive timeout to avoid hanging on slow clients */
        struct timeval tv;
        tv.tv_sec = 10;
        tv.tv_usec = 0;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        /* Parse request */
        http_request_t req;
        parse_request(client_fd, &req);

        if (req.valid) {
            if (!g_state.quiet) {
                log_info("%s %s (%zu bytes)",
                         req.method, req.path, req.body_len);
            }
            route_request(client_fd, &req);
        } else {
            if (req.content_length > TRINEDB_MAX_BODY) {
                send_error(client_fd, 413, "Payload Too Large",
                           "Request body exceeds 10 MB limit");
            } else {
                send_error(client_fd, 400, "Bad Request",
                           "Malformed HTTP request");
            }
        }

        free(req.body);
        close(client_fd);
    }

    /* Shutdown */
    fprintf(stderr, "\nShutting down...\n");
    close(g_state.listen_fd);

    /* Auto-save */
    auto_save();

    /* Cleanup */
    trine_route_free(g_state.index);
    fprintf(stderr, "Goodbye.\n");

    return 0;
}
