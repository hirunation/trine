//! Raw FFI bindings to the TRINE C library.
//!
//! All types and functions here map 1:1 to their C counterparts.
//! Users should prefer the safe wrappers in the parent module.

use std::os::raw::{c_char, c_float, c_int};

// ── Constants ────────────────────────────────────────────────────────

pub const TRINE_CHANNELS: usize = 240;
pub const TRINE_S1_PACKED_SIZE: usize = 48;
pub const TRINE_S1_CHAINS: usize = 4;

// Canon presets
pub const TRINE_CANON_NONE: c_int = 0;
pub const TRINE_CANON_SUPPORT: c_int = 1;
pub const TRINE_CANON_CODE: c_int = 2;
pub const TRINE_CANON_POLICY: c_int = 3;
pub const TRINE_CANON_GENERAL: c_int = 4;

// Recall modes
pub const TRINE_RECALL_FAST: c_int = 0;
pub const TRINE_RECALL_BALANCED: c_int = 1;
pub const TRINE_RECALL_STRICT: c_int = 2;

// ── C Structs ────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct trine_s1_lens_t {
    pub weights: [c_float; TRINE_S1_CHAINS],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct trine_s1_config_t {
    pub threshold: c_float,
    pub lens: trine_s1_lens_t,
    pub calibrate_length: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct trine_s1_result_t {
    pub is_duplicate: c_int,
    pub similarity: c_float,
    pub calibrated: c_float,
    pub matched_index: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct trine_route_stats_t {
    pub candidates_checked: c_int,
    pub total_entries: c_int,
    pub candidate_ratio: c_float,
    pub speedup: c_float,
    pub recall_mode: *const c_char,
}

// ── Opaque types ─────────────────────────────────────────────────────

/// Opaque handle to a Stage-1 in-memory index.
#[repr(C)]
pub struct trine_s1_index_t {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Opaque handle to a Band-LSH routed index.
#[repr(C)]
pub struct trine_route_t {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

// ── Extern C functions ───────────────────────────────────────────────

extern "C" {
    // ── Encode ───────────────────────────────────────────────────────

    /// Shingle encoder: arbitrary-length, case-insensitive, locality-preserving.
    pub fn trine_encode_shingle(
        text: *const c_char,
        len: usize,
        channels: *mut u8,
    );

    /// Stage-1 convenience encoder (wraps trine_encode_shingle). Returns 0 on success.
    pub fn trine_s1_encode(
        text: *const c_char,
        len: usize,
        out: *mut u8,
    ) -> c_int;

    /// Batch encode N texts. `out` must be pre-allocated: count * 240 bytes.
    pub fn trine_s1_encode_batch(
        texts: *const *const c_char,
        lens: *const usize,
        count: c_int,
        out: *mut u8,
    ) -> c_int;

    // ── Compare ──────────────────────────────────────────────────────

    /// Lens-weighted cosine similarity between two 240-byte embeddings.
    pub fn trine_s1_compare(
        a: *const u8,
        b: *const u8,
        lens: *const trine_s1_lens_t,
    ) -> c_float;

    /// Full dedup check with threshold and calibration.
    pub fn trine_s1_check(
        candidate: *const u8,
        reference: *const u8,
        config: *const trine_s1_config_t,
    ) -> trine_s1_result_t;

    /// Fill ratio: fraction of non-zero channels in [0.0, 1.0].
    pub fn trine_s1_fill_ratio(emb: *const u8) -> c_float;

    // ── Pack/Unpack ──────────────────────────────────────────────────

    /// Pack 240 trits into 48 bytes. Returns 0 on success.
    pub fn trine_s1_pack(trits: *const u8, packed: *mut u8) -> c_int;

    /// Unpack 48 bytes back to 240 trits. Returns 0 on success.
    pub fn trine_s1_unpack(packed: *const u8, trits: *mut u8) -> c_int;

    // ── Index ────────────────────────────────────────────────────────

    /// Create an empty Stage-1 index. Returns NULL on allocation failure.
    pub fn trine_s1_index_create(
        config: *const trine_s1_config_t,
    ) -> *mut trine_s1_index_t;

    /// Add an embedding with optional tag. Returns assigned index or -1.
    pub fn trine_s1_index_add(
        idx: *mut trine_s1_index_t,
        emb: *const u8,
        tag: *const c_char,
    ) -> c_int;

    /// Query for best match. Returns result with matched_index (-1 if none).
    pub fn trine_s1_index_query(
        idx: *const trine_s1_index_t,
        candidate: *const u8,
    ) -> trine_s1_result_t;

    /// Number of entries in the index.
    pub fn trine_s1_index_count(idx: *const trine_s1_index_t) -> c_int;

    /// Tag for entry at given index (may return NULL).
    pub fn trine_s1_index_tag(
        idx: *const trine_s1_index_t,
        index: c_int,
    ) -> *const c_char;

    /// Free the index and all entries.
    pub fn trine_s1_index_free(idx: *mut trine_s1_index_t);

    /// Save index to binary file. Returns 0 on success.
    pub fn trine_s1_index_save(
        idx: *const trine_s1_index_t,
        path: *const c_char,
    ) -> c_int;

    /// Load index from binary file. Returns NULL on error.
    pub fn trine_s1_index_load(path: *const c_char) -> *mut trine_s1_index_t;

    // ── Routed Index ─────────────────────────────────────────────────

    /// Create a routed index. Returns NULL on allocation failure.
    pub fn trine_route_create(
        config: *const trine_s1_config_t,
    ) -> *mut trine_route_t;

    /// Add embedding with optional tag. Returns assigned index or -1.
    pub fn trine_route_add(
        rt: *mut trine_route_t,
        emb: *const u8,
        tag: *const c_char,
    ) -> c_int;

    /// Query routed index. `stats` may be NULL.
    pub fn trine_route_query(
        rt: *const trine_route_t,
        candidate: *const u8,
        stats: *mut trine_route_stats_t,
    ) -> trine_s1_result_t;

    /// Number of entries in the routed index.
    pub fn trine_route_count(rt: *const trine_route_t) -> c_int;

    /// Tag for entry at given index.
    pub fn trine_route_tag(
        rt: *const trine_route_t,
        index: c_int,
    ) -> *const c_char;

    /// Free the routed index.
    pub fn trine_route_free(rt: *mut trine_route_t);

    /// Save routed index to binary file. Returns 0 on success.
    pub fn trine_route_save(
        rt: *const trine_route_t,
        path: *const c_char,
    ) -> c_int;

    /// Load routed index from binary file. Returns NULL on error.
    pub fn trine_route_load(path: *const c_char) -> *mut trine_route_t;

    /// Set recall mode (FAST/BALANCED/STRICT). Returns 0 on success.
    pub fn trine_route_set_recall(
        rt: *mut trine_route_t,
        mode: c_int,
    ) -> c_int;

    // ── Canon ────────────────────────────────────────────────────────

    /// Apply canonicalization preset. Returns 0 on success.
    pub fn trine_canon_apply(
        text: *const c_char,
        len: usize,
        preset: c_int,
        out: *mut c_char,
        out_cap: usize,
        out_len: *mut usize,
    ) -> c_int;
}
