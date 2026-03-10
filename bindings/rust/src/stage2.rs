//! # Stage-2 Semantic Projection Layer
//!
//! Safe wrappers for TRINE's Hebbian-trained semantic projection.
//! Stage-2 transforms Stage-1 surface-form embeddings into semantic
//! embeddings via learned ternary projection and cascade mixing.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use trine::{Embedding, Lens};
//! use trine::stage2::{Stage2Model, stage2_compare, stage2_blend};
//!
//! // Load a trained model
//! let model = Stage2Model::load("model.trine2").unwrap();
//!
//! // Encode with Stage-2 projection
//! let a = model.encode("hello world", 0);
//! let b = model.encode("hello earth", 0);
//!
//! // Compare Stage-2 embeddings
//! let sim = stage2_compare(&a, &b, &Lens::UNIFORM);
//! println!("Stage-2 similarity: {sim:.3}");
//!
//! // Blend Stage-1 and Stage-2 scores
//! let blended = stage2_blend("hello world", "hello earth", &model, 0.65, &Lens::UNIFORM, 0);
//! println!("Blended similarity: {blended:.3}");
//! ```

use super::{ffi, Embedding, Error, Lens};

use std::ffi::CString;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;

// ═════════════════════════════════════════════════════════════════════
// ProjectionMode
// ═════════════════════════════════════════════════════════════════════

/// Projection mode for the Stage-2 forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectionMode {
    /// Full 240x240 centered dot product + sign (default).
    Sign,
    /// Per-channel keep/flip/zero using diagonal entries only.
    Diagonal,
    /// Sparse cross-channel: top-K per output row, W=0 entries skipped.
    Sparse,
    /// Block-diagonal: 4 independent 60x60 chain-local projections.
    BlockDiagonal,
}

impl ProjectionMode {
    fn to_c_int(self) -> std::os::raw::c_int {
        match self {
            ProjectionMode::Sign => ffi::TRINE_S2_PROJ_SIGN,
            ProjectionMode::Diagonal => ffi::TRINE_S2_PROJ_DIAGONAL,
            ProjectionMode::Sparse => ffi::TRINE_S2_PROJ_SPARSE,
            ProjectionMode::BlockDiagonal => ffi::TRINE_S2_PROJ_BLOCK_DIAG,
        }
    }

    fn from_c_int(v: std::os::raw::c_int) -> Self {
        match v {
            1 => ProjectionMode::Diagonal,
            2 => ProjectionMode::Sparse,
            3 => ProjectionMode::BlockDiagonal,
            _ => ProjectionMode::Sign,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Stage2Info
// ═════════════════════════════════════════════════════════════════════

/// Introspection info for a Stage-2 model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stage2Info {
    /// Number of projection copies (always 3).
    pub projection_k: u32,
    /// Projection dimensionality (always 240).
    pub projection_dims: u32,
    /// Number of cascade mixing cells.
    pub cascade_cells: u32,
    /// Maximum cascade depth.
    pub max_depth: u32,
    /// Whether this model is an identity (pass-through).
    pub is_identity: bool,
}

// ═════════════════════════════════════════════════════════════════════
// HebbianConfig
// ═════════════════════════════════════════════════════════════════════

/// Configuration for Hebbian training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HebbianConfig {
    /// Stage-1 similarity above this threshold = "similar" pair (positive update).
    pub similarity_threshold: f32,
    /// Quantization threshold for freeze. 0 = auto.
    pub freeze_threshold: i32,
    /// Target density for auto-threshold freeze.
    pub freeze_target_density: f32,
    /// Number of cascade mixing cells.
    pub cascade_cells: u32,
    /// Default inference depth.
    pub cascade_depth: u32,
    /// Projection mode: Sign, Diagonal, Sparse, or BlockDiagonal.
    pub projection_mode: ProjectionMode,
    /// Weighted Hebbian mode (0=binary sign, 1=weighted magnitude).
    pub weighted_mode: bool,
    /// Positive magnitude scale (default 10.0).
    pub pos_scale: f32,
    /// Negative magnitude scale (default 3.0).
    pub neg_scale: f32,
    /// Sparse top-K per output row (0=disabled).
    pub sparse_k: u32,
    /// Enable block-diagonal projection mode for training.
    pub block_diagonal: bool,
    /// RNG seed for deterministic downsampling (0 = derive from pairs_observed).
    pub rng_seed: u64,
}

impl Default for HebbianConfig {
    /// Mirrors TRINE_HEBBIAN_CONFIG_DEFAULT from the C header.
    fn default() -> Self {
        HebbianConfig {
            similarity_threshold: 0.5,
            freeze_threshold: 0,
            freeze_target_density: 0.33,
            cascade_cells: 512,
            cascade_depth: 4,
            projection_mode: ProjectionMode::Sign,
            weighted_mode: false,
            pos_scale: 10.0,
            neg_scale: 3.0,
            sparse_k: 0,
            block_diagonal: false,
            rng_seed: 0,
        }
    }
}

impl HebbianConfig {
    fn as_ffi(&self) -> ffi::trine_hebbian_config_t {
        ffi::trine_hebbian_config_t {
            similarity_threshold: self.similarity_threshold,
            freeze_threshold: self.freeze_threshold,
            freeze_target_density: self.freeze_target_density,
            cascade_cells: self.cascade_cells,
            cascade_depth: self.cascade_depth,
            projection_mode: self.projection_mode.to_c_int(),
            weighted_mode: if self.weighted_mode { 1 } else { 0 },
            pos_scale: self.pos_scale,
            neg_scale: self.neg_scale,
            source_weights: [ffi::trine_source_weight_t {
                name: [0u8; ffi::TRINE_SOURCE_NAME_LEN],
                weight: 0.0,
            }; ffi::TRINE_MAX_SOURCES],
            n_source_weights: 0,
            sparse_k: self.sparse_k,
            block_diagonal: if self.block_diagonal { 1 } else { 0 },
            rng_seed: self.rng_seed,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// SaveConfig
// ═════════════════════════════════════════════════════════════════════

/// Configuration embedded in the .trine2 file header when saving.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaveConfig {
    /// Training similarity threshold.
    pub similarity_threshold: f32,
    /// Freeze target density.
    pub density: f32,
    /// Cascade topology PRNG seed.
    pub topo_seed: u64,
}

impl Default for SaveConfig {
    fn default() -> Self {
        SaveConfig {
            similarity_threshold: 0.5,
            density: 0.33,
            topo_seed: 0,
        }
    }
}

impl SaveConfig {
    fn as_ffi(&self) -> ffi::trine_s2_save_config_t {
        ffi::trine_s2_save_config_t {
            similarity_threshold: self.similarity_threshold,
            density: self.density,
            topo_seed: self.topo_seed,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// TrainerMetrics
// ═════════════════════════════════════════════════════════════════════

/// Metrics from Hebbian training.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainerMetrics {
    /// Total pairs fed to observe().
    pub pairs_observed: i64,
    /// Maximum |counter| across all weights.
    pub max_abs_counter: i32,
    /// Number of positive-weighted counters.
    pub n_positive_weights: u32,
    /// Number of negative-weighted counters.
    pub n_negative_weights: u32,
    /// Number of zero-weighted counters.
    pub n_zero_weights: u32,
    /// Fraction non-zero after freeze.
    pub weight_density: f32,
    /// Threshold used (explicit or auto).
    pub effective_threshold: i32,
}

// ═════════════════════════════════════════════════════════════════════
// Stage2Error — extends parent Error
// ═════════════════════════════════════════════════════════════════════

/// Stage-2 specific error variants.
///
/// These are returned as `Error` variants via `From` conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage2Error {
    /// Failed to load a .trine2 model file.
    LoadFailed(String),
    /// Failed to save a .trine2 model file.
    SaveFailed(String),
    /// .trine2 file validation failed.
    ValidationFailed(String),
    /// Stage-2 encoding failed (null model or text).
    EncodeFailed,
    /// Hebbian trainer creation failed.
    TrainerCreateFailed,
    /// Hebbian freeze failed (returned NULL).
    FreezeFailed,
    /// Training file processing failed.
    TrainFileFailed(String),
    /// Model info query failed.
    InfoFailed,
}

impl fmt::Display for Stage2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stage2Error::LoadFailed(p) => write!(f, "failed to load Stage-2 model from {p}"),
            Stage2Error::SaveFailed(p) => write!(f, "failed to save Stage-2 model to {p}"),
            Stage2Error::ValidationFailed(p) => write!(f, "Stage-2 file validation failed: {p}"),
            Stage2Error::EncodeFailed => write!(f, "Stage-2 encoding failed"),
            Stage2Error::TrainerCreateFailed => write!(f, "Hebbian trainer creation failed"),
            Stage2Error::FreezeFailed => write!(f, "Hebbian freeze failed"),
            Stage2Error::TrainFileFailed(p) => write!(f, "failed to train from file: {p}"),
            Stage2Error::InfoFailed => write!(f, "Stage-2 model info query failed"),
        }
    }
}

impl std::error::Error for Stage2Error {}

impl From<Stage2Error> for Error {
    fn from(e: Stage2Error) -> Error {
        Error::IoError(e.to_string())
    }
}

// ═════════════════════════════════════════════════════════════════════
// Stage2Model
// ═════════════════════════════════════════════════════════════════════

/// A trained Stage-2 semantic projection model.
///
/// Owns the underlying C model handle and frees it on drop.
/// Provides encode, compare, and save/load operations.
///
/// This type is NOT `Send` or `Sync` because the underlying C library
/// is not guaranteed thread-safe for all operations.
pub struct Stage2Model {
    ptr: *mut ffi::trine_s2_model_t,
    _not_send_sync: PhantomData<*const ()>,
}

impl Stage2Model {
    /// Create an identity model (pass-through).
    ///
    /// The identity model returns the Stage-1 encoding unchanged.
    /// This is the backward-compatibility baseline.
    pub fn identity() -> Result<Self, Error> {
        // SAFETY: trine_s2_create_identity allocates a new model.
        // Returns NULL only on allocation failure.
        let ptr = unsafe { ffi::trine_s2_create_identity() };
        if ptr.is_null() {
            return Err(Error::AllocationFailed);
        }
        Ok(Stage2Model { ptr, _not_send_sync: PhantomData })
    }

    /// Create a random model with given cascade cells and PRNG seed.
    ///
    /// Useful for baselines and testing. Not semantically meaningful.
    pub fn random(cells: u32, seed: u64) -> Result<Self, Error> {
        // SAFETY: trine_s2_create_random allocates a new model.
        // Returns NULL only on allocation failure.
        let ptr = unsafe { ffi::trine_s2_create_random(cells, seed) };
        if ptr.is_null() {
            return Err(Error::AllocationFailed);
        }
        Ok(Stage2Model { ptr, _not_send_sync: PhantomData })
    }

    /// Create a model with block-diagonal projection weights.
    ///
    /// Block-diagonal projection uses 4 independent 60x60 chain-local
    /// projections instead of a single 240x240 matrix, reducing the
    /// parameter count while preserving per-chain structure.
    ///
    /// `weights` must contain exactly `k * 4 * 60 * 60` bytes of per-chain
    /// ternary weights. `k` is the number of projection copies (typically 3).
    /// `n_cells` is the number of cascade mixing cells.
    /// `topo_seed` is the PRNG seed for deterministic cascade topology.
    ///
    /// Returns `None` if the C library returns NULL (allocation failure or
    /// invalid parameters).
    pub fn create_block_diagonal(
        weights: &[u8],
        k: i32,
        n_cells: u32,
        topo_seed: u64,
    ) -> Option<Self> {
        // Expected size: k * 4 chains * 60 * 60 per-chain weights
        let expected = k as usize * 4 * 60 * 60;
        if weights.len() != expected {
            return None;
        }
        // SAFETY: trine_s2_create_block_diagonal copies the weight data.
        // weights pointer is valid for weights.len() bytes.
        // Returns NULL on allocation failure.
        let ptr = unsafe {
            ffi::trine_s2_create_block_diagonal(
                weights.as_ptr(),
                k as std::os::raw::c_int,
                n_cells,
                topo_seed,
            )
        };
        if ptr.is_null() {
            return None;
        }
        Some(Stage2Model { ptr, _not_send_sync: PhantomData })
    }

    /// Load a trained model from a .trine2 file.
    pub fn load(path: &str) -> Result<Self, Error> {
        let c_path = CString::new(path).map_err(|_| Error::InvalidPath)?;
        // SAFETY: trine_s2_load reads and validates the file.
        // Returns NULL on any error (corrupt file, I/O failure, etc.).
        let ptr = unsafe { ffi::trine_s2_load(c_path.as_ptr()) };
        if ptr.is_null() {
            return Err(Stage2Error::LoadFailed(path.to_string()).into());
        }
        Ok(Stage2Model { ptr, _not_send_sync: PhantomData })
    }

    /// Save this model to a .trine2 file with default config.
    pub fn save(&self, path: &str) -> Result<(), Error> {
        self.save_with_config(path, &SaveConfig::default())
    }

    /// Save this model to a .trine2 file with explicit config.
    pub fn save_with_config(&self, path: &str, config: &SaveConfig) -> Result<(), Error> {
        let c_path = CString::new(path).map_err(|_| Error::InvalidPath)?;
        let c_config = config.as_ffi();
        // SAFETY: trine_s2_save writes the model to disk.
        // self.ptr is valid (non-null) because we only construct from valid pointers.
        let rc = unsafe { ffi::trine_s2_save(self.ptr, c_path.as_ptr(), &c_config) };
        if rc != 0 {
            return Err(Stage2Error::SaveFailed(path.to_string()).into());
        }
        Ok(())
    }

    /// Encode text into a Stage-2 embedding.
    ///
    /// The full pipeline: Stage-1 shingle encode -> projection -> cascade.
    /// `depth` controls the number of cascade ticks (0 = projection only).
    pub fn encode(&self, text: &str, depth: i32) -> Embedding {
        let mut out = [0u8; 240];
        // SAFETY: trine_s2_encode writes exactly 240 bytes to `out`.
        // self.ptr is valid (non-null). Text pointer + len is valid for the str slice.
        let _rc = unsafe {
            ffi::trine_s2_encode(
                self.ptr,
                text.as_ptr() as *const std::os::raw::c_char,
                text.len(),
                depth as u32,
                out.as_mut_ptr(),
            )
        };
        Embedding::from_trits(out)
    }

    /// Encode from pre-computed Stage-1 trits.
    ///
    /// Useful when the Stage-1 embedding is already available,
    /// avoiding redundant re-encoding.
    pub fn encode_from_trits(&self, stage1: &Embedding, depth: i32) -> Embedding {
        let mut out = [0u8; 240];
        // SAFETY: trine_s2_encode_from_trits reads 240 bytes from stage1,
        // writes 240 bytes to out. Both buffers are 240 bytes.
        let _rc = unsafe {
            ffi::trine_s2_encode_from_trits(
                self.ptr,
                stage1.trits().as_ptr(),
                depth as u32,
                out.as_mut_ptr(),
            )
        };
        Embedding::from_trits(out)
    }

    /// Query model parameters.
    pub fn info(&self) -> Stage2Info {
        let mut c_info = ffi::trine_s2_info_t {
            projection_k: 0,
            projection_dims: 0,
            cascade_cells: 0,
            max_depth: 0,
            is_identity: 0,
        };
        // SAFETY: trine_s2_info fills the info struct.
        // self.ptr is valid (non-null).
        unsafe { ffi::trine_s2_info(self.ptr, &mut c_info) };
        Stage2Info {
            projection_k: c_info.projection_k,
            projection_dims: c_info.projection_dims,
            cascade_cells: c_info.cascade_cells,
            max_depth: c_info.max_depth,
            is_identity: c_info.is_identity != 0,
        }
    }

    /// Set the projection mode (Sign or Diagonal).
    pub fn set_projection_mode(&mut self, mode: ProjectionMode) {
        // SAFETY: trine_s2_set_projection_mode mutates the model's mode flag.
        // self.ptr is valid (non-null), and we have &mut self.
        unsafe { ffi::trine_s2_set_projection_mode(self.ptr, mode.to_c_int()) };
    }

    /// Get the current projection mode.
    pub fn projection_mode(&self) -> ProjectionMode {
        // SAFETY: trine_s2_get_projection_mode reads the model's mode flag.
        let mode = unsafe { ffi::trine_s2_get_projection_mode(self.ptr) };
        ProjectionMode::from_c_int(mode)
    }

    /// Enable or disable stacked depth.
    ///
    /// When enabled, each depth tick re-applies the learned projection
    /// instead of running the random cascade network.
    pub fn set_stacked_depth(&mut self, enable: bool) {
        unsafe { ffi::trine_s2_set_stacked_depth(self.ptr, if enable { 1 } else { 0 }) };
    }

    /// Check if stacked depth is enabled.
    pub fn stacked_depth(&self) -> bool {
        unsafe { ffi::trine_s2_get_stacked_depth(self.ptr) != 0 }
    }

    /// Gate-aware comparison using only channels with active diagonal gates.
    ///
    /// Channels where the majority of K=3 diagonal gates are zero
    /// (uninformative) are excluded from the cosine similarity.
    pub fn compare_gated(&self, a: &Embedding, b: &Embedding) -> f32 {
        unsafe {
            ffi::trine_s2_compare_gated(
                self.ptr,
                a.trits().as_ptr(),
                b.trits().as_ptr(),
            )
        }
    }

    /// Set per-S1-bucket alpha values for adaptive blending.
    ///
    /// `buckets` contains 10 alpha values, one per S1 similarity bucket:
    /// `[0.0-0.1)`, `[0.1-0.2)`, ..., `[0.9-1.0]`.
    ///
    /// When set, `compare_adaptive_blend()` uses the bucket lookup to
    /// select the blend weight based on the S1 similarity of the pair.
    pub fn set_adaptive_alpha(&mut self, buckets: &[f32; 10]) {
        // SAFETY: trine_s2_set_adaptive_alpha reads exactly 10 floats
        // from the buckets pointer. self.ptr is valid (non-null),
        // and we have &mut self.
        unsafe {
            ffi::trine_s2_set_adaptive_alpha(self.ptr, buckets.as_ptr());
        }
    }

    /// Adaptive blend: alpha selected based on S1 similarity bucket.
    ///
    /// Computes the Stage-1 cosine similarity (uniform), looks up the
    /// corresponding alpha from the model's adaptive alpha buckets, and
    /// blends: `result = alpha * s1_sim + (1 - alpha) * s2_centered_cosine`.
    ///
    /// Returns 0.0 if adaptive alpha is not set or on error.
    pub fn compare_adaptive_blend(
        &self,
        s1_a: &[u8; 240],
        s1_b: &[u8; 240],
        s2_a: &[u8; 240],
        s2_b: &[u8; 240],
    ) -> f32 {
        // SAFETY: trine_s2_compare_adaptive_blend reads 240 bytes from
        // each of the 4 embedding pointers. self.ptr is valid (non-null).
        unsafe {
            ffi::trine_s2_compare_adaptive_blend(
                self.ptr,
                s1_a.as_ptr(),
                s1_b.as_ptr(),
                s2_a.as_ptr(),
                s2_b.as_ptr(),
            )
        }
    }

    /// Get a read-only view of the block-diagonal projection weights.
    ///
    /// Returns `None` if the model is not in block-diagonal mode or is NULL.
    /// The returned slice has layout `K * 4 * 60 * 60` bytes and is valid
    /// for the lifetime of this `Stage2Model`.
    pub fn get_block_projection(&self) -> Option<&[u8]> {
        // SAFETY: trine_s2_get_block_projection returns a pointer to
        // internal model memory valid for the model's lifetime.
        // Returns NULL if not in block-diagonal mode.
        let ptr = unsafe { ffi::trine_s2_get_block_projection(self.ptr) };
        if ptr.is_null() {
            return None;
        }
        let info = self.info();
        let len = info.projection_k as usize * 4 * 60 * 60;
        // SAFETY: The returned pointer is valid for len bytes and lives
        // as long as the model (tied to &self lifetime).
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Access the raw C model pointer (for advanced FFI use).
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of this `Stage2Model`.
    /// Do not free the pointer manually.
    pub unsafe fn as_ptr(&self) -> *const ffi::trine_s2_model_t {
        self.ptr
    }
}

impl Drop for Stage2Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: trine_s2_free is safe to call with a valid pointer.
            // We own this pointer and are dropping it.
            unsafe { ffi::trine_s2_free(self.ptr); }
            self.ptr = ptr::null_mut();
        }
    }
}

impl fmt::Debug for Stage2Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let info = self.info();
        write!(
            f,
            "Stage2Model(k={}, dims={}, cascade={}, identity={})",
            info.projection_k, info.projection_dims, info.cascade_cells, info.is_identity
        )
    }
}

// ═════════════════════════════════════════════════════════════════════
// Free functions
// ═════════════════════════════════════════════════════════════════════

/// Compare two Stage-2 embeddings using a lens.
///
/// Returns a similarity value in [0.0, 1.0].
pub fn stage2_compare(a: &Embedding, b: &Embedding, lens: &Lens) -> f32 {
    let c_lens = ffi::trine_s1_lens_t { weights: lens.weights };
    // SAFETY: trine_s2_compare reads 240 bytes from each embedding.
    // The lens pointer is valid for the duration of the call.
    unsafe {
        ffi::trine_s2_compare(
            a.trits().as_ptr(),
            b.trits().as_ptr(),
            &c_lens as *const ffi::trine_s1_lens_t as *const std::os::raw::c_void,
        )
    }
}

/// Convenience: blend Stage-1 and Stage-2 similarity scores.
///
/// Computes `alpha * s1_similarity + (1 - alpha) * s2_similarity`.
/// `alpha` should be in [0.0, 1.0]. Values like 0.65 (65% S1, 35% S2)
/// are typical for best blend performance.
pub fn stage2_blend(
    text_a: &str,
    text_b: &str,
    model: &Stage2Model,
    alpha: f32,
    lens: &Lens,
    depth: i32,
) -> f32 {
    // Stage-1 embeddings
    let s1_a = Embedding::encode(text_a);
    let s1_b = Embedding::encode(text_b);
    let s1_sim = s1_a.similarity(&s1_b, lens);

    // Stage-2 embeddings (from pre-computed Stage-1 trits)
    let s2_a = model.encode_from_trits(&s1_a, depth);
    let s2_b = model.encode_from_trits(&s1_b, depth);
    let s2_sim = stage2_compare(&s2_a, &s2_b, lens);

    alpha * s1_sim + (1.0 - alpha) * s2_sim
}

/// Validate a .trine2 file without loading it.
///
/// Returns `Ok(())` if the file is valid, or an error describing the problem.
pub fn stage2_validate(path: &str) -> Result<(), Error> {
    let c_path = CString::new(path).map_err(|_| Error::InvalidPath)?;
    // SAFETY: trine_s2_validate reads and checks the file.
    let rc = unsafe { ffi::trine_s2_validate(c_path.as_ptr()) };
    if rc != 0 {
        return Err(Stage2Error::ValidationFailed(path.to_string()).into());
    }
    Ok(())
}

// ═════════════════════════════════════════════════════════════════════
// HebbianTrainer
// ═════════════════════════════════════════════════════════════════════

/// Hebbian training harness for learning Stage-2 projections.
///
/// Feed text pairs (or raw embeddings with similarity scores), then
/// freeze the accumulated statistics into a trained `Stage2Model`.
///
/// This type is NOT `Send` or `Sync` because the underlying C library
/// is not thread-safe for mutation.
pub struct HebbianTrainer {
    ptr: *mut ffi::trine_hebbian_state_t,
    _not_send_sync: PhantomData<*const ()>,
}

impl HebbianTrainer {
    /// Create a new Hebbian trainer with the given configuration.
    pub fn new(config: HebbianConfig) -> Result<Self, Error> {
        let c_config = config.as_ffi();
        // SAFETY: trine_hebbian_create allocates training state.
        // Returns NULL only on allocation failure.
        let ptr = unsafe { ffi::trine_hebbian_create(&c_config) };
        if ptr.is_null() {
            return Err(Stage2Error::TrainerCreateFailed.into());
        }
        Ok(HebbianTrainer { ptr, _not_send_sync: PhantomData })
    }

    /// Create a new Hebbian trainer configured for block-diagonal mode.
    ///
    /// This is a convenience constructor that takes a base `HebbianConfig`
    /// and overrides `block_diagonal = true` and
    /// `projection_mode = BlockDiagonal`. All other config fields are
    /// preserved from the provided config.
    pub fn new_block_diagonal(config: HebbianConfig) -> Result<Self, Error> {
        let mut bd_config = config;
        bd_config.block_diagonal = true;
        bd_config.projection_mode = ProjectionMode::BlockDiagonal;
        let c_config = bd_config.as_ffi();
        // SAFETY: trine_hebbian_create allocates training state.
        // Returns NULL only on allocation failure.
        let ptr = unsafe { ffi::trine_hebbian_create(&c_config) };
        if ptr.is_null() {
            return Err(Stage2Error::TrainerCreateFailed.into());
        }
        Ok(HebbianTrainer { ptr, _not_send_sync: PhantomData })
    }

    /// Observe a single training pair from raw embeddings.
    ///
    /// `similarity` is the Stage-1 cosine similarity, used to determine
    /// the sign of the Hebbian update (positive if above threshold,
    /// negative otherwise).
    pub fn observe(&mut self, a: &Embedding, b: &Embedding, similarity: f32) {
        // SAFETY: trine_hebbian_observe reads 240 bytes from each embedding.
        // self.ptr is valid (non-null), and we have &mut self.
        unsafe {
            ffi::trine_hebbian_observe(
                self.ptr,
                a.trits().as_ptr(),
                b.trits().as_ptr(),
                similarity,
            );
        }
    }

    /// Observe a pair from text (encode + compare + accumulate).
    ///
    /// The trainer internally encodes both texts with Stage-1,
    /// computes their similarity, and applies the Hebbian update.
    pub fn observe_text(&mut self, text_a: &str, text_b: &str) {
        // SAFETY: trine_hebbian_observe_text reads text_a[..len_a] and text_b[..len_b].
        // self.ptr is valid (non-null), and we have &mut self.
        unsafe {
            ffi::trine_hebbian_observe_text(
                self.ptr,
                text_a.as_ptr() as *const std::os::raw::c_char,
                text_a.len(),
                text_b.as_ptr() as *const std::os::raw::c_char,
                text_b.len(),
            );
        }
    }

    /// Train from a JSONL file.
    ///
    /// Each line must have `text_a`, `text_b`, `score`, `label` fields.
    /// `epochs` controls how many times the file is re-read.
    /// Returns the number of pairs processed.
    pub fn train_file(&mut self, path: &str, epochs: i32) -> Result<i64, Error> {
        let c_path = CString::new(path).map_err(|_| Error::InvalidPath)?;
        // SAFETY: trine_hebbian_train_file reads the JSONL file and
        // updates training state. self.ptr is valid.
        let result = unsafe {
            ffi::trine_hebbian_train_file(self.ptr, c_path.as_ptr(), epochs as u32)
        };
        if result < 0 {
            return Err(Stage2Error::TrainFileFailed(path.to_string()).into());
        }
        Ok(result)
    }

    /// Freeze the accumulated Hebbian statistics into a Stage-2 model.
    ///
    /// Returns a new `Stage2Model` ready for inference. The trainer
    /// retains its state and can continue accumulating.
    pub fn freeze(&self) -> Result<Stage2Model, Error> {
        // SAFETY: trine_hebbian_freeze reads the accumulator state and
        // creates a new model. Returns NULL on failure.
        let ptr = unsafe { ffi::trine_hebbian_freeze(self.ptr) };
        if ptr.is_null() {
            return Err(Stage2Error::FreezeFailed.into());
        }
        Ok(Stage2Model { ptr, _not_send_sync: PhantomData })
    }

    /// Get current training metrics.
    pub fn metrics(&self) -> TrainerMetrics {
        let mut c_metrics = ffi::trine_hebbian_metrics_t {
            pairs_observed: 0,
            max_abs_counter: 0,
            n_positive_weights: 0,
            n_negative_weights: 0,
            n_zero_weights: 0,
            weight_density: 0.0,
            effective_threshold: 0,
        };
        // SAFETY: trine_hebbian_metrics fills the metrics struct.
        // self.ptr is valid (non-null).
        unsafe { ffi::trine_hebbian_metrics(self.ptr, &mut c_metrics) };
        TrainerMetrics {
            pairs_observed: c_metrics.pairs_observed,
            max_abs_counter: c_metrics.max_abs_counter,
            n_positive_weights: c_metrics.n_positive_weights,
            n_negative_weights: c_metrics.n_negative_weights,
            n_zero_weights: c_metrics.n_zero_weights,
            weight_density: c_metrics.weight_density,
            effective_threshold: c_metrics.effective_threshold,
        }
    }

    /// Reset accumulators (start fresh, keep config).
    pub fn reset(&mut self) {
        // SAFETY: trine_hebbian_reset clears the accumulator state.
        // self.ptr is valid (non-null), and we have &mut self.
        unsafe { ffi::trine_hebbian_reset(self.ptr) };
    }
}

impl Drop for HebbianTrainer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: trine_hebbian_free is safe to call with a valid pointer.
            // We own this pointer and are dropping it.
            unsafe { ffi::trine_hebbian_free(self.ptr); }
            self.ptr = ptr::null_mut();
        }
    }
}

impl fmt::Debug for HebbianTrainer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = self.metrics();
        write!(
            f,
            "HebbianTrainer({} pairs, density={:.3})",
            m.pairs_observed, m.weight_density
        )
    }
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_model_creation() {
        let model = Stage2Model::identity().expect("identity model creation should succeed");
        let info = model.info();
        assert!(info.is_identity, "identity model should report is_identity=true");
        assert_eq!(info.projection_k, 3);
        assert_eq!(info.projection_dims, 240);
    }

    #[test]
    fn test_identity_passthrough() {
        let model = Stage2Model::identity().expect("create identity model");
        let text = "hello world";
        let s1 = Embedding::encode(text);
        let s2 = model.encode(text, 0);
        assert_eq!(
            s1, s2,
            "identity model at depth=0 should return Stage-1 encoding unchanged"
        );
    }

    #[test]
    fn test_random_model_creation() {
        let model = Stage2Model::random(512, 42).expect("random model creation should succeed");
        let info = model.info();
        assert!(!info.is_identity, "random model should not be identity");
        assert_eq!(info.cascade_cells, 512);
    }

    #[test]
    fn test_random_model_deterministic() {
        let m1 = Stage2Model::random(64, 1234).expect("create model 1");
        let m2 = Stage2Model::random(64, 1234).expect("create model 2");

        let text = "deterministic encoding test";
        let e1 = m1.encode(text, 0);
        let e2 = m2.encode(text, 0);
        assert_eq!(e1, e2, "same seed should produce same encoding");
    }

    #[test]
    fn test_encode_from_trits() {
        let model = Stage2Model::random(64, 99).expect("create model");
        let text = "encode from trits test";
        let s1 = Embedding::encode(text);

        // Full encode vs encode_from_trits should match
        let full = model.encode(text, 0);
        let from_trits = model.encode_from_trits(&s1, 0);
        assert_eq!(
            full, from_trits,
            "encode() and encode_from_trits() should produce same result"
        );
    }

    #[test]
    fn test_stage2_compare() {
        let model = Stage2Model::identity().expect("create identity model");
        let a = model.encode("the quick brown fox", 0);
        let b = model.encode("the quick brown fox", 0);
        let sim = stage2_compare(&a, &b, &Lens::UNIFORM);
        assert!(
            (sim - 1.0).abs() < 0.001,
            "self-similarity should be ~1.0, got {sim}"
        );
    }

    #[test]
    fn test_stage2_blend() {
        let model = Stage2Model::identity().expect("create identity model");
        let sim = stage2_blend("hello world", "hello world", &model, 0.65, &Lens::UNIFORM, 0);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "identical texts should blend to ~1.0, got {sim}"
        );
    }

    #[test]
    fn test_projection_mode_roundtrip() {
        let mut model = Stage2Model::random(64, 42).expect("create model");

        model.set_projection_mode(ProjectionMode::Diagonal);
        assert_eq!(model.projection_mode(), ProjectionMode::Diagonal);

        model.set_projection_mode(ProjectionMode::Sign);
        assert_eq!(model.projection_mode(), ProjectionMode::Sign);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("trine_rust_test_s2.trine2");
        let path_str = path.to_str().unwrap();

        // Create and save
        let model = Stage2Model::random(64, 42).expect("create model");
        let info_before = model.info();
        model.save(path_str).expect("save should succeed");

        // Load and verify
        let loaded = Stage2Model::load(path_str).expect("load should succeed");
        let info_after = loaded.info();
        assert_eq!(info_before.projection_k, info_after.projection_k);
        assert_eq!(info_before.projection_dims, info_after.projection_dims);

        // Encoding should be identical
        let text = "save load roundtrip test";
        let e1 = model.encode(text, 0);
        let e2 = loaded.encode(text, 0);
        assert_eq!(e1, e2, "loaded model should produce same encoding");

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_validate_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("trine_rust_test_validate.trine2");
        let path_str = path.to_str().unwrap();

        let model = Stage2Model::random(64, 42).expect("create model");
        model.save(path_str).expect("save should succeed");

        stage2_validate(path_str).expect("validation should succeed");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_validate_nonexistent() {
        let result = stage2_validate("/tmp/nonexistent_trine2_file.trine2");
        assert!(result.is_err(), "validation of nonexistent file should fail");
    }

    #[test]
    fn test_hebbian_trainer_lifecycle() {
        let config = HebbianConfig::default();
        let mut trainer = HebbianTrainer::new(config).expect("trainer creation should succeed");

        // Observe some pairs
        let a = Embedding::encode("hello world");
        let b = Embedding::encode("hello earth");
        let sim = a.similarity(&b, &Lens::UNIFORM);
        trainer.observe(&a, &b, sim);

        let metrics = trainer.metrics();
        assert_eq!(metrics.pairs_observed, 1, "should have 1 observed pair");

        // Observe text pair
        trainer.observe_text("foo bar", "foo baz");
        let metrics = trainer.metrics();
        assert_eq!(metrics.pairs_observed, 2, "should have 2 observed pairs");

        // Freeze to model
        let model = trainer.freeze().expect("freeze should succeed");
        let info = model.info();
        assert!(!info.is_identity, "trained model should not be identity");
    }

    #[test]
    fn test_hebbian_reset() {
        let config = HebbianConfig::default();
        let mut trainer = HebbianTrainer::new(config).expect("create trainer");

        trainer.observe_text("hello", "world");
        assert_eq!(trainer.metrics().pairs_observed, 1);

        trainer.reset();
        assert_eq!(trainer.metrics().pairs_observed, 0, "reset should clear pairs");
    }

    #[test]
    fn test_hebbian_config_default() {
        let config = HebbianConfig::default();
        assert!((config.similarity_threshold - 0.5).abs() < 0.001);
        assert_eq!(config.freeze_threshold, 0);
        assert!((config.freeze_target_density - 0.33).abs() < 0.01);
        assert_eq!(config.cascade_cells, 512);
        assert_eq!(config.cascade_depth, 4);
        assert_eq!(config.projection_mode, ProjectionMode::Sign);
        assert!(!config.weighted_mode);
        assert!((config.pos_scale - 10.0).abs() < 0.001);
        assert!((config.neg_scale - 3.0).abs() < 0.001);
        assert_eq!(config.sparse_k, 0);
        assert!(!config.block_diagonal);
        assert_eq!(config.rng_seed, 0);
    }

    #[test]
    fn test_save_config_default() {
        let config = SaveConfig::default();
        assert!((config.similarity_threshold - 0.5).abs() < 0.001);
        assert!((config.density - 0.33).abs() < 0.01);
        assert_eq!(config.topo_seed, 0);
    }

    #[test]
    fn test_model_debug_format() {
        let model = Stage2Model::identity().expect("create model");
        let debug = format!("{model:?}");
        assert!(debug.starts_with("Stage2Model("), "debug: {debug}");
        assert!(debug.contains("identity=true"), "debug: {debug}");
    }

    #[test]
    fn test_trainer_debug_format() {
        let config = HebbianConfig::default();
        let trainer = HebbianTrainer::new(config).expect("create trainer");
        let debug = format!("{trainer:?}");
        assert!(debug.starts_with("HebbianTrainer("), "debug: {debug}");
        assert!(debug.contains("0 pairs"), "debug: {debug}");
    }

    #[test]
    fn test_save_with_config() {
        let dir = std::env::temp_dir();
        let path = dir.join("trine_rust_test_s2_config.trine2");
        let path_str = path.to_str().unwrap();

        let model = Stage2Model::random(64, 42).expect("create model");
        let save_cfg = SaveConfig {
            similarity_threshold: 0.9,
            density: 0.15,
            topo_seed: 12345,
        };
        model.save_with_config(path_str, &save_cfg).expect("save should succeed");

        let loaded = Stage2Model::load(path_str).expect("load should succeed");
        let text = "config roundtrip test";
        let e1 = model.encode(text, 0);
        let e2 = loaded.encode(text, 0);
        assert_eq!(e1, e2, "loaded model should produce same encoding");

        let _ = std::fs::remove_file(&path);
    }

    // ── Block-diagonal tests ──────────────────────────────────────────

    #[test]
    fn test_create_block_diagonal() {
        // K=3 projection copies, 4 chains, 60x60 per chain
        let k: i32 = 3;
        let expected_len = k as usize * 4 * 60 * 60; // 43200 bytes
        // Fill with valid ternary weights (cycling 0, 1, 2)
        let weights: Vec<u8> = (0..expected_len).map(|i| (i % 3) as u8).collect();
        let n_cells: u32 = 64;
        let topo_seed: u64 = 42;

        let model = Stage2Model::create_block_diagonal(&weights, k, n_cells, topo_seed);
        assert!(model.is_some(), "create_block_diagonal should succeed with valid weights");

        let model = model.unwrap();
        let info = model.info();
        assert!(!info.is_identity, "block-diagonal model should not be identity");
        assert_eq!(info.projection_k, k as u32);
        assert_eq!(info.projection_dims, 240);
    }

    #[test]
    fn test_create_block_diagonal_wrong_size() {
        // Provide weights with wrong length — should return None
        let bad_weights = vec![0u8; 100]; // too short
        let result = Stage2Model::create_block_diagonal(&bad_weights, 3, 64, 42);
        assert!(result.is_none(), "wrong-sized weights should return None");
    }

    #[test]
    fn test_block_diagonal_encode() {
        let k: i32 = 3;
        let weight_len = k as usize * 4 * 60 * 60;
        let weights: Vec<u8> = (0..weight_len).map(|i| (i % 3) as u8).collect();

        let model = Stage2Model::create_block_diagonal(&weights, k, 64, 42)
            .expect("create block-diagonal model");

        let emb = model.encode("block diagonal encoding test", 0);
        let trits = emb.trits();

        // Verify all 240 trits are in the valid ternary set {0, 1, 2}
        assert_eq!(trits.len(), 240, "embedding must have exactly 240 dimensions");
        for (i, &t) in trits.iter().enumerate() {
            assert!(
                t <= 2,
                "trit[{i}] = {t}, expected value in {{0, 1, 2}}"
            );
        }

        // Determinism: same text produces same embedding
        let emb2 = model.encode("block diagonal encoding test", 0);
        assert_eq!(emb, emb2, "same text should produce identical embeddings");
    }

    #[test]
    fn test_block_diagonal_persistence() {
        let k: i32 = 3;
        let weight_len = k as usize * 4 * 60 * 60;
        let weights: Vec<u8> = (0..weight_len).map(|i| (i % 3) as u8).collect();

        let model = Stage2Model::create_block_diagonal(&weights, k, 64, 42)
            .expect("create block-diagonal model");

        // Save
        let dir = std::env::temp_dir();
        let path = dir.join("trine_rust_test_block_diag.trine2");
        let path_str = path.to_str().unwrap();
        model.save(path_str).expect("save block-diagonal model should succeed");

        // Load
        let loaded = Stage2Model::load(path_str).expect("load block-diagonal model should succeed");
        let info_before = model.info();
        let info_after = loaded.info();
        assert_eq!(info_before.projection_k, info_after.projection_k);
        assert_eq!(info_before.projection_dims, info_after.projection_dims);
        assert_eq!(info_before.cascade_cells, info_after.cascade_cells);

        // Encoding round-trip: original and loaded models produce same result
        let text = "block diagonal persistence roundtrip";
        let e1 = model.encode(text, 0);
        let e2 = loaded.encode(text, 0);
        assert_eq!(e1, e2, "loaded block-diagonal model should produce same encoding");

        // Validate the saved file
        stage2_validate(path_str).expect("saved block-diagonal file should validate");

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_adaptive_alpha() {
        let mut model = Stage2Model::random(64, 42).expect("create model");

        // Set per-bucket alpha values (10 buckets for S1 similarity ranges)
        let buckets: [f32; 10] = [
            0.90, 0.85, 0.80, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40,
        ];
        model.set_adaptive_alpha(&buckets);

        // Encode two texts with Stage-1 and Stage-2
        let s1_a = Embedding::encode("adaptive alpha test one");
        let s1_b = Embedding::encode("adaptive alpha test two");
        let s2_a = model.encode("adaptive alpha test one", 0);
        let s2_b = model.encode("adaptive alpha test two", 0);

        // Call adaptive blend — should not panic, returns a finite float
        let result = model.compare_adaptive_blend(
            s1_a.trits(),
            s1_b.trits(),
            s2_a.trits(),
            s2_b.trits(),
        );
        assert!(
            result.is_finite(),
            "adaptive blend should return a finite value, got {result}"
        );

        // Self-similarity should be high
        let self_sim = model.compare_adaptive_blend(
            s1_a.trits(),
            s1_a.trits(),
            s2_a.trits(),
            s2_a.trits(),
        );
        assert!(
            self_sim.is_finite(),
            "self-similarity adaptive blend should be finite, got {self_sim}"
        );
    }

    #[test]
    fn test_hebbian_block_diagonal() {
        let config = HebbianConfig {
            cascade_cells: 64,
            cascade_depth: 2,
            ..HebbianConfig::default()
        };

        // Create trainer in block-diagonal mode via convenience constructor
        let mut trainer = HebbianTrainer::new_block_diagonal(config)
            .expect("block-diagonal trainer creation should succeed");

        // Observe some training pairs
        let a = Embedding::encode("block diagonal training alpha");
        let b = Embedding::encode("block diagonal training beta");
        let sim = a.similarity(&b, &Lens::UNIFORM);
        trainer.observe(&a, &b, sim);

        trainer.observe_text("block diagonal text one", "block diagonal text two");

        let metrics = trainer.metrics();
        assert_eq!(metrics.pairs_observed, 2, "should have 2 observed pairs");

        // Freeze into a model
        let model = trainer.freeze().expect("freeze block-diagonal trainer should succeed");
        let info = model.info();
        assert!(!info.is_identity, "frozen block-diagonal model should not be identity");
        assert_eq!(info.projection_k, 3);
        assert_eq!(info.projection_dims, 240);

        // Reset and verify clean state
        trainer.reset();
        assert_eq!(
            trainer.metrics().pairs_observed, 0,
            "reset should clear pairs in block-diagonal trainer"
        );
    }

    #[test]
    fn test_projection_mode_enum() {
        // Verify to_c_int mapping
        assert_eq!(ProjectionMode::Sign.to_c_int(), ffi::TRINE_S2_PROJ_SIGN);
        assert_eq!(ProjectionMode::Diagonal.to_c_int(), ffi::TRINE_S2_PROJ_DIAGONAL);
        assert_eq!(ProjectionMode::Sparse.to_c_int(), ffi::TRINE_S2_PROJ_SPARSE);
        assert_eq!(ProjectionMode::BlockDiagonal.to_c_int(), ffi::TRINE_S2_PROJ_BLOCK_DIAG);

        // Verify from_c_int round-trip for all variants
        assert_eq!(ProjectionMode::from_c_int(0), ProjectionMode::Sign);
        assert_eq!(ProjectionMode::from_c_int(1), ProjectionMode::Diagonal);
        assert_eq!(ProjectionMode::from_c_int(2), ProjectionMode::Sparse);
        assert_eq!(ProjectionMode::from_c_int(3), ProjectionMode::BlockDiagonal);

        // Unknown values should fall back to Sign
        assert_eq!(ProjectionMode::from_c_int(99), ProjectionMode::Sign);
        assert_eq!(ProjectionMode::from_c_int(-1), ProjectionMode::Sign);

        // Full round-trip: variant -> c_int -> variant
        let modes = [
            ProjectionMode::Sign,
            ProjectionMode::Diagonal,
            ProjectionMode::Sparse,
            ProjectionMode::BlockDiagonal,
        ];
        for mode in modes {
            let c_val = mode.to_c_int();
            let back = ProjectionMode::from_c_int(c_val);
            assert_eq!(
                mode, back,
                "round-trip failed for {mode:?}: to_c_int={c_val}, from_c_int={back:?}"
            );
        }

        // Verify the concrete integer values (C ABI contract)
        assert_eq!(ffi::TRINE_S2_PROJ_SIGN, 0);
        assert_eq!(ffi::TRINE_S2_PROJ_DIAGONAL, 1);
        assert_eq!(ffi::TRINE_S2_PROJ_SPARSE, 2);
        assert_eq!(ffi::TRINE_S2_PROJ_BLOCK_DIAG, 3);
    }
}
