#ifndef TRINE_ERROR_H
#define TRINE_ERROR_H

/* TRINE error codes — returned by functions that can fail.
 * 0 = success, negative values = errors. */
typedef enum {
    TRINE_OK            =  0,   /* Success */
    TRINE_ERR_NULL      = -1,   /* NULL pointer argument */
    TRINE_ERR_ALLOC     = -2,   /* Memory allocation failed */
    TRINE_ERR_IO        = -3,   /* File I/O error */
    TRINE_ERR_FORMAT    = -4,   /* Invalid file format or magic bytes */
    TRINE_ERR_CHECKSUM  = -5,   /* Checksum mismatch */
    TRINE_ERR_VERSION   = -6,   /* Unsupported format version */
    TRINE_ERR_RANGE     = -7,   /* Value out of valid range */
    TRINE_ERR_CAPACITY  = -8,   /* Buffer or index capacity exceeded */
    TRINE_ERR_CORRUPT   = -9,   /* Data corruption detected */
    TRINE_ERR_CONFIG    = -10   /* Invalid configuration */
} trine_error_t;

/* Return human-readable error description */
static inline const char *trine_strerror(int code) {
    switch (code) {
        case TRINE_OK:           return "success";
        case TRINE_ERR_NULL:     return "null pointer";
        case TRINE_ERR_ALLOC:    return "allocation failed";
        case TRINE_ERR_IO:       return "I/O error";
        case TRINE_ERR_FORMAT:   return "invalid format";
        case TRINE_ERR_CHECKSUM: return "checksum mismatch";
        case TRINE_ERR_VERSION:  return "unsupported version";
        case TRINE_ERR_RANGE:    return "value out of range";
        case TRINE_ERR_CAPACITY: return "capacity exceeded";
        case TRINE_ERR_CORRUPT:  return "data corruption";
        case TRINE_ERR_CONFIG:   return "invalid config";
        default:                 return "unknown error";
    }
}

#endif /* TRINE_ERROR_H */
