//! Raw FFI bindings to the TRINE C library.
//!
//! All types and functions here map 1:1 to their C counterparts.
//! Users should prefer the safe wrappers in the parent module.

use std::os::raw::{c_char, c_float, c_int, c_void};

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
    /// Returns 0 on success, -1 on memory allocation failure.
    pub fn trine_encode_shingle(
        text: *const c_char,
        len: usize,
        channels: *mut u8,
    ) -> c_int;

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

// ═════════════════════════════════════════════════════════════════════
// Stage-2: Semantic Projection Layer
// ═════════════════════════════════════════════════════════════════════

pub const TRINE_S2_DIM: usize = 240;

// Projection modes
pub const TRINE_S2_PROJ_SIGN: c_int = 0;
pub const TRINE_S2_PROJ_DIAGONAL: c_int = 1;
pub const TRINE_S2_PROJ_SPARSE: c_int = 2;
pub const TRINE_S2_PROJ_BLOCK_DIAG: c_int = 3;

// ── Stage-2 Opaque types ────────────────────────────────────────────

/// Opaque handle to a Stage-2 model.
#[repr(C)]
pub struct trine_s2_model_t {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Opaque handle to a Hebbian training state.
#[repr(C)]
pub struct trine_hebbian_state_t {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

// ── Stage-2 C Structs ───────────────────────────────────────────────

/// Model introspection info.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct trine_s2_info_t {
    pub projection_k: u32,
    pub projection_dims: u32,
    pub cascade_cells: u32,
    pub max_depth: u32,
    pub is_identity: c_int,
}

/// Per-source training weight entry.
pub const TRINE_MAX_SOURCES: usize = 8;
pub const TRINE_SOURCE_NAME_LEN: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct trine_source_weight_t {
    pub name: [u8; TRINE_SOURCE_NAME_LEN],
    pub weight: c_float,
}

/// Hebbian training configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct trine_hebbian_config_t {
    pub similarity_threshold: c_float,
    pub freeze_threshold: i32,
    pub freeze_target_density: c_float,
    pub cascade_cells: u32,
    pub cascade_depth: u32,
    pub projection_mode: c_int,
    // Phase A1: Weighted Hebbian
    pub weighted_mode: c_int,
    pub pos_scale: c_float,
    pub neg_scale: c_float,
    // Phase A2: Dataset rebalancing
    pub source_weights: [trine_source_weight_t; TRINE_MAX_SOURCES],
    pub n_source_weights: c_int,
    // Phase D1: Sparse projection
    pub sparse_k: u32,
    // v1.0.3: Block-diagonal projection
    pub block_diagonal: c_int,
    // RNG seed for deterministic downsampling
    pub rng_seed: u64,
}

/// Hebbian training metrics.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct trine_hebbian_metrics_t {
    pub pairs_observed: i64,
    pub max_abs_counter: i32,
    pub n_positive_weights: u32,
    pub n_negative_weights: u32,
    pub n_zero_weights: u32,
    pub weight_density: c_float,
    pub effective_threshold: i32,
}

/// Save configuration for .trine2 files.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct trine_s2_save_config_t {
    pub similarity_threshold: c_float,
    pub density: c_float,
    pub topo_seed: u64,
}

// ── Stage-2 Extern C functions ──────────────────────────────────────

extern "C" {
    // ── Lifecycle ───────────────────────────────────────────────────

    /// Create an identity model (pass-through, no projection).
    pub fn trine_s2_create_identity() -> *mut trine_s2_model_t;

    /// Create a random model with given cascade cells and seed.
    pub fn trine_s2_create_random(n_cells: u32, seed: u64) -> *mut trine_s2_model_t;

    /// Create a model from pre-trained projection weights.
    pub fn trine_s2_create_from_parts(
        proj: *const c_void,
        n_cells: u32,
        topo_seed: u64,
    ) -> *mut trine_s2_model_t;

    /// Create a model with block-diagonal projection weights.
    /// block_weights: K * 4 * 60 * 60 bytes of per-chain ternary weights (copied).
    /// K: number of projection copies (typically 3).
    /// n_cells: number of cascade mixing cells.
    /// topo_seed: PRNG seed for deterministic cascade topology.
    /// Returns NULL on allocation failure.
    pub fn trine_s2_create_block_diagonal(
        block_weights: *const u8,
        k: c_int,
        n_cells: u32,
        topo_seed: u64,
    ) -> *mut trine_s2_model_t;

    /// Free a Stage-2 model.
    pub fn trine_s2_free(model: *mut trine_s2_model_t);

    // ── Forward Pass ────────────────────────────────────────────────

    /// Full pipeline: text -> Stage-1 encode -> projection -> cascade.
    /// Returns 0 on success, -1 on error.
    pub fn trine_s2_encode(
        model: *const trine_s2_model_t,
        text: *const c_char,
        len: usize,
        depth: u32,
        out: *mut u8,
    ) -> c_int;

    /// From pre-computed Stage-1 trits: projection -> cascade.
    /// Returns 0 on success, -1 on error.
    pub fn trine_s2_encode_from_trits(
        model: *const trine_s2_model_t,
        stage1: *const u8,
        depth: u32,
        out: *mut u8,
    ) -> c_int;

    // ── Comparison ──────────────────────────────────────────────────

    /// Compare two Stage-2 embeddings using a lens.
    /// lens may be NULL (uniform weights) or a trine_s1_lens_t pointer.
    /// Returns similarity in [0.0, 1.0], or -1.0 on error.
    pub fn trine_s2_compare(
        a: *const u8,
        b: *const u8,
        lens: *const c_void,
    ) -> c_float;

    // ── Gate-Aware Comparison (Phase B1) ───────────────────────────

    /// Compare using only channels with active diagonal gates.
    /// Skips uninformative channels for noise-reduced cosine similarity.
    pub fn trine_s2_compare_gated(
        model: *const trine_s2_model_t,
        a: *const u8,
        b: *const u8,
    ) -> c_float;

    // ── Per-Chain Blend Comparison (Phase B2) ───────────────────

    /// Blend S1 and S2 per-chain similarities with independent alpha weights.
    pub fn trine_s2_compare_chain_blend(
        s1_a: *const u8,
        s1_b: *const u8,
        s2_a: *const u8,
        s2_b: *const u8,
        alpha: *const c_float,
    ) -> c_float;

    // ── Adaptive Blend (v1.0.3) ────────────────────────────────────

    /// Set per-S1-bucket alpha values for adaptive blending.
    /// buckets[10]: alpha for S1 similarity in [0.0-0.1), [0.1-0.2), ..., [0.9-1.0].
    /// Pass NULL to disable adaptive blending.
    pub fn trine_s2_set_adaptive_alpha(
        model: *mut trine_s2_model_t,
        buckets: *const c_float,
    );

    /// Adaptive blend: alpha selected based on S1 similarity bucket.
    /// Computes S1 similarity, looks up alpha, then blends S1 and S2 scores.
    /// Returns 0.0 if adaptive_alpha is not set or on error.
    pub fn trine_s2_compare_adaptive_blend(
        model: *const trine_s2_model_t,
        s1_a: *const u8,
        s1_b: *const u8,
        s2_a: *const u8,
        s2_b: *const u8,
    ) -> c_float;

    // ── Block-Diagonal Introspection (v1.0.3) ───────────────────

    /// Get a read-only pointer to the block-diagonal projection weights.
    /// Returns NULL if model is NULL or not in block-diagonal mode.
    /// Layout: K * 4 * 60 * 60 bytes = 43,200 bytes total.
    pub fn trine_s2_get_block_projection(
        model: *const trine_s2_model_t,
    ) -> *const u8;

    // ── Introspection ───────────────────────────────────────────────

    /// Query model parameters. Returns 0 on success, -1 on null model.
    pub fn trine_s2_info(
        model: *const trine_s2_model_t,
        info: *mut trine_s2_info_t,
    ) -> c_int;

    /// Set projection mode (0=sign, 1=diagonal, 2=sparse).
    pub fn trine_s2_set_projection_mode(model: *mut trine_s2_model_t, mode: c_int);

    /// Get projection mode. Returns -1 on null model.
    pub fn trine_s2_get_projection_mode(model: *const trine_s2_model_t) -> c_int;

    /// Enable/disable stacked depth (re-apply projection instead of cascade).
    pub fn trine_s2_set_stacked_depth(model: *mut trine_s2_model_t, enable: c_int);

    /// Get stacked depth flag. Returns 0 if disabled or null model.
    pub fn trine_s2_get_stacked_depth(model: *const trine_s2_model_t) -> c_int;

    /// Get cascade cell count. Returns 0 on null model.
    pub fn trine_s2_get_cascade_cells(model: *const trine_s2_model_t) -> u32;

    /// Get default depth. Returns 0 on null model.
    pub fn trine_s2_get_default_depth(model: *const trine_s2_model_t) -> u32;

    /// Get read-only pointer to projection weights.
    pub fn trine_s2_get_projection(model: *const trine_s2_model_t) -> *const c_void;

    // ── Persistence ─────────────────────────────────────────────────

    /// Save a trained Stage-2 model to a .trine2 file.
    /// Returns 0 on success, -1 on error.
    pub fn trine_s2_save(
        model: *const trine_s2_model_t,
        path: *const c_char,
        config: *const trine_s2_save_config_t,
    ) -> c_int;

    /// Load a Stage-2 model from a .trine2 file.
    /// Returns NULL on error.
    pub fn trine_s2_load(path: *const c_char) -> *mut trine_s2_model_t;

    /// Validate a .trine2 file without loading it.
    /// Returns 0 if valid, -1 on error.
    pub fn trine_s2_validate(path: *const c_char) -> c_int;

    // ── Hebbian Training ────────────────────────────────────────────

    /// Create training state. Returns NULL on failure.
    pub fn trine_hebbian_create(
        config: *const trine_hebbian_config_t,
    ) -> *mut trine_hebbian_state_t;

    /// Free training state. Safe to call with NULL.
    pub fn trine_hebbian_free(state: *mut trine_hebbian_state_t);

    /// Observe a single training pair (raw trits).
    pub fn trine_hebbian_observe(
        state: *mut trine_hebbian_state_t,
        a: *const u8,
        b: *const u8,
        similarity: c_float,
    );

    /// Observe a pair from text (encode + compare + accumulate).
    pub fn trine_hebbian_observe_text(
        state: *mut trine_hebbian_state_t,
        text_a: *const c_char,
        len_a: usize,
        text_b: *const c_char,
        len_b: usize,
    );

    /// Train from a JSONL file. Returns pairs processed, or -1 on error.
    pub fn trine_hebbian_train_file(
        state: *mut trine_hebbian_state_t,
        path: *const c_char,
        epochs: u32,
    ) -> i64;

    /// Freeze current state to a Stage-2 model.
    /// Returns NULL on failure.
    pub fn trine_hebbian_freeze(
        state: *const trine_hebbian_state_t,
    ) -> *mut trine_s2_model_t;

    /// Get training metrics. Returns 0 on success.
    pub fn trine_hebbian_metrics(
        state: *const trine_hebbian_state_t,
        out: *mut trine_hebbian_metrics_t,
    ) -> c_int;

    /// Reset accumulators (start fresh, keep config).
    pub fn trine_hebbian_reset(state: *mut trine_hebbian_state_t);

    /// Update similarity threshold between epochs (threshold schedule).
    pub fn trine_hebbian_set_threshold(state: *mut trine_hebbian_state_t, threshold: c_float);
}
