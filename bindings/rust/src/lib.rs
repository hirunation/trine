//! # TRINE — Ternary Resonance Interference Network Embedding
//!
//! Safe Rust bindings for the TRINE C embedding library. Provides fast,
//! deterministic text-to-trit encoding, lens-weighted similarity comparison,
//! in-memory indexing, and Band-LSH routed search.
//!
//! ## Quick Start
//!
//! ```rust
//! use trine::{Embedding, Lens, Index, Config};
//!
//! // Encode two texts
//! let a = Embedding::encode("hello world");
//! let b = Embedding::encode("hello world!");
//!
//! // Compare with a lens
//! let sim = a.similarity(&b, &Lens::DEDUP);
//! println!("similarity: {sim:.3}");
//!
//! // Build an index
//! let mut idx = Index::new(Config::default()).unwrap();
//! idx.add(&a, Some("greeting")).unwrap();
//! let result = idx.query(&b);
//! println!("duplicate: {}", result.is_duplicate);
//! ```

pub mod ffi;
pub mod stage2;

#[cfg(test)]
mod tests;

pub use stage2::{Stage2Model, HebbianTrainer, stage2_compare, stage2_blend};

use std::ffi::{CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr;

// ═════════════════════════════════════════════════════════════════════
// Error
// ═════════════════════════════════════════════════════════════════════

/// Error type for TRINE operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Memory allocation failure in the C library.
    AllocationFailed,
    /// A path contained an interior NUL byte.
    InvalidPath,
    /// File I/O error during save or load.
    IoError(String),
    /// Encoding error.
    EncodeFailed,
    /// Canon buffer too small (should not happen with our sizing).
    CanonBufferTooSmall,
    /// Pack/unpack error.
    PackError,
    /// Invalid recall mode.
    InvalidRecallMode,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::AllocationFailed => write!(f, "allocation failed in TRINE C library"),
            Error::InvalidPath => write!(f, "path contains interior NUL byte"),
            Error::IoError(msg) => write!(f, "I/O error: {msg}"),
            Error::EncodeFailed => write!(f, "encoding failed"),
            Error::CanonBufferTooSmall => write!(f, "canonicalization buffer too small"),
            Error::PackError => write!(f, "pack/unpack error"),
            Error::InvalidRecallMode => write!(f, "invalid recall mode"),
        }
    }
}

impl std::error::Error for Error {}

// ═════════════════════════════════════════════════════════════════════
// Canon
// ═════════════════════════════════════════════════════════════════════

/// Canonicalization preset applied before encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Canon {
    /// No transforms (passthrough).
    None,
    /// Whitespace + timestamps + UUIDs + numbers.
    Support,
    /// Whitespace + identifier normalization.
    Code,
    /// Whitespace + number bucketing.
    Policy,
    /// Whitespace normalization only.
    General,
}

impl Canon {
    /// Apply this canonicalization preset to the given text.
    ///
    /// Returns the canonicalized string. The `None` variant returns
    /// the input unchanged (no allocation beyond the return value).
    pub fn apply(&self, text: &str) -> String {
        let preset = self.to_c_int();
        if preset == ffi::TRINE_CANON_NONE {
            return text.to_string();
        }

        // Code preset may expand (camelCase splitting inserts spaces),
        // so allocate 2x + 1. Other presets only shrink or preserve.
        let cap = text.len() * 2 + 1;
        let mut out_buf: Vec<u8> = vec![0u8; cap];
        let mut out_len: usize = 0;

        let rc = unsafe {
            ffi::trine_canon_apply(
                text.as_ptr() as *const std::os::raw::c_char,
                text.len(),
                preset,
                out_buf.as_mut_ptr() as *mut std::os::raw::c_char,
                cap,
                &mut out_len,
            )
        };

        if rc != 0 {
            // Fallback: return original text if canon fails
            return text.to_string();
        }

        // Safety: trine_canon_apply writes out_len bytes of valid content
        out_buf.truncate(out_len);
        String::from_utf8_lossy(&out_buf).into_owned()
    }

    fn to_c_int(self) -> std::os::raw::c_int {
        match self {
            Canon::None => ffi::TRINE_CANON_NONE,
            Canon::Support => ffi::TRINE_CANON_SUPPORT,
            Canon::Code => ffi::TRINE_CANON_CODE,
            Canon::Policy => ffi::TRINE_CANON_POLICY,
            Canon::General => ffi::TRINE_CANON_GENERAL,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Lens
// ═════════════════════════════════════════════════════════════════════

/// Lens weights for similarity comparison.
///
/// The four weights correspond to the four TRINE chains:
/// - `edit`:   Character-level features (chain 1)
/// - `morph`:  Morphological features (chain 2)
/// - `phrase`: Phrase-level features (chain 3)
/// - `vocab`:  Vocabulary features (chain 4)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lens {
    /// Raw weights array: [edit, morph, phrase, vocab].
    pub weights: [f32; 4],
}

impl Lens {
    /// Equal weight across all chains.
    pub const UNIFORM: Lens = Lens { weights: [1.0, 1.0, 1.0, 1.0] };
    /// Optimized for near-duplicate detection.
    pub const DEDUP: Lens = Lens { weights: [0.5, 0.5, 0.7, 1.0] };
    /// Emphasizes character-level edit similarity.
    pub const EDIT: Lens = Lens { weights: [1.0, 0.3, 0.1, 0.0] };
    /// Emphasizes vocabulary overlap.
    pub const VOCAB: Lens = Lens { weights: [0.0, 0.2, 0.3, 1.0] };
    /// Optimized for source code.
    pub const CODE: Lens = Lens { weights: [1.0, 0.8, 0.4, 0.2] };
    /// Optimized for legal documents.
    pub const LEGAL: Lens = Lens { weights: [0.2, 0.4, 1.0, 0.8] };
    /// Optimized for medical texts.
    pub const MEDICAL: Lens = Lens { weights: [0.3, 1.0, 0.6, 0.5] };
    /// Optimized for support tickets.
    pub const SUPPORT: Lens = Lens { weights: [0.2, 0.4, 0.7, 1.0] };
    /// Optimized for policy documents.
    pub const POLICY: Lens = Lens { weights: [0.1, 0.3, 1.0, 0.8] };

    /// Create a lens with explicit weights.
    pub fn new(edit: f32, morph: f32, phrase: f32, vocab: f32) -> Self {
        Lens { weights: [edit, morph, phrase, vocab] }
    }

    fn as_ffi(&self) -> ffi::trine_s1_lens_t {
        ffi::trine_s1_lens_t { weights: self.weights }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Embedding
// ═════════════════════════════════════════════════════════════════════

/// A 240-trit TRINE embedding.
///
/// Each element is a trit value (0, 1, or 2) across 4 chains of 60 channels.
/// Embeddings are small (240 bytes) and implement `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Embedding([u8; 240]);

impl Embedding {
    /// Encode text into a TRINE embedding.
    ///
    /// Uses the shingle encoder: arbitrary-length input, case-insensitive,
    /// locality-preserving. Deterministic for the same input.
    pub fn encode(text: &str) -> Self {
        let mut channels = [0u8; 240];
        let rc = unsafe {
            ffi::trine_encode_shingle(
                text.as_ptr() as *const std::os::raw::c_char,
                text.len(),
                channels.as_mut_ptr(),
            )
        };
        if rc != 0 {
            // OOM: return all-zero embedding (channels already zeroed)
            return Embedding(channels);
        }
        Embedding(channels)
    }

    /// Encode text with canonicalization applied first.
    ///
    /// Equivalent to `Canon::apply` followed by `Embedding::encode`,
    /// but expressed as a single call.
    pub fn encode_with_canon(text: &str, preset: Canon) -> Self {
        let canonical = preset.apply(text);
        Self::encode(&canonical)
    }

    /// Compute lens-weighted cosine similarity with another embedding.
    ///
    /// Returns a value in [0.0, 1.0], or -1.0 on error.
    pub fn similarity(&self, other: &Embedding, lens: &Lens) -> f32 {
        let c_lens = lens.as_ffi();
        unsafe {
            ffi::trine_s1_compare(
                self.0.as_ptr(),
                other.0.as_ptr(),
                &c_lens,
            )
        }
    }

    /// Fraction of non-zero channels in [0.0, 1.0].
    ///
    /// Sparse embeddings (short texts) have lower fill ratios.
    pub fn fill_ratio(&self) -> f32 {
        unsafe { ffi::trine_s1_fill_ratio(self.0.as_ptr()) }
    }

    /// Pack this embedding into 48 bytes (5 trits per byte).
    pub fn pack(&self) -> Result<PackedEmbedding, Error> {
        let mut packed = [0u8; 48];
        let rc = unsafe { ffi::trine_s1_pack(self.0.as_ptr(), packed.as_mut_ptr()) };
        if rc != 0 {
            return Err(Error::PackError);
        }
        Ok(PackedEmbedding(packed))
    }

    /// Access the raw 240-byte trit array.
    pub fn trits(&self) -> &[u8; 240] {
        &self.0
    }

    /// Create an embedding from a raw trit array.
    ///
    /// No validation is performed. Each byte should be 0, 1, or 2.
    pub fn from_trits(trits: [u8; 240]) -> Self {
        Embedding(trits)
    }
}

impl fmt::Debug for Embedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nonzero = self.0.iter().filter(|&&t| t != 0).count();
        write!(f, "Embedding({nonzero}/240 active)")
    }
}

/// Batch-encode multiple texts into embeddings.
///
/// More efficient than encoding one at a time for large batches.
pub fn encode_batch(texts: &[&str]) -> Result<Vec<Embedding>, Error> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let count = texts.len();
    let ptrs: Vec<*const std::os::raw::c_char> = texts
        .iter()
        .map(|t| t.as_ptr() as *const std::os::raw::c_char)
        .collect();
    let lens: Vec<usize> = texts.iter().map(|t| t.len()).collect();

    let mut out = vec![0u8; count * 240];
    let rc = unsafe {
        ffi::trine_s1_encode_batch(
            ptrs.as_ptr(),
            lens.as_ptr(),
            count as std::os::raw::c_int,
            out.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(Error::EncodeFailed);
    }

    let embeddings = (0..count)
        .map(|i| {
            let mut trits = [0u8; 240];
            trits.copy_from_slice(&out[i * 240..(i + 1) * 240]);
            Embedding(trits)
        })
        .collect();
    Ok(embeddings)
}

// ═════════════════════════════════════════════════════════════════════
// PackedEmbedding
// ═════════════════════════════════════════════════════════════════════

/// A packed TRINE embedding (48 bytes, 5 trits per byte).
///
/// Useful for storage-constrained scenarios. Unpack to compare.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedEmbedding([u8; 48]);

impl PackedEmbedding {
    /// Unpack back to a full 240-trit embedding.
    pub fn unpack(&self) -> Result<Embedding, Error> {
        let mut trits = [0u8; 240];
        let rc = unsafe { ffi::trine_s1_unpack(self.0.as_ptr(), trits.as_mut_ptr()) };
        if rc != 0 {
            return Err(Error::PackError);
        }
        Ok(Embedding(trits))
    }

    /// Access the raw 48-byte packed array.
    pub fn bytes(&self) -> &[u8; 48] {
        &self.0
    }

    /// Create a packed embedding from raw bytes.
    pub fn from_bytes(bytes: [u8; 48]) -> Self {
        PackedEmbedding(bytes)
    }
}

impl fmt::Debug for PackedEmbedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PackedEmbedding(48 bytes)")
    }
}

// ═════════════════════════════════════════════════════════════════════
// Config
// ═════════════════════════════════════════════════════════════════════

/// Configuration for duplicate detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Cosine similarity threshold for "duplicate" (0.0-1.0).
    pub threshold: f32,
    /// Lens weights for comparison.
    pub lens: Lens,
    /// Whether to apply length-aware calibration.
    pub calibrate: bool,
}

impl Config {
    fn as_ffi(&self) -> ffi::trine_s1_config_t {
        ffi::trine_s1_config_t {
            threshold: self.threshold,
            lens: self.lens.as_ffi(),
            calibrate_length: if self.calibrate { 1 } else { 0 },
        }
    }
}

impl Default for Config {
    /// Default config: threshold=0.60, DEDUP lens, calibration on.
    fn default() -> Self {
        Config {
            threshold: 0.60,
            lens: Lens::DEDUP,
            calibrate: true,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// QueryResult
// ═════════════════════════════════════════════════════════════════════

/// Result of a duplicate-detection query.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    /// Whether the candidate was flagged as a duplicate.
    pub is_duplicate: bool,
    /// Raw lens-weighted cosine similarity.
    pub similarity: f32,
    /// Length-calibrated similarity (if calibration was enabled).
    pub calibrated: f32,
    /// Index of the best match in the index, if any.
    pub matched_index: Option<usize>,
    /// Tag of the matched entry, if available.
    pub tag: Option<String>,
}

impl QueryResult {
    fn from_ffi(r: ffi::trine_s1_result_t, tag: Option<String>) -> Self {
        QueryResult {
            is_duplicate: r.is_duplicate != 0,
            similarity: r.similarity,
            calibrated: r.calibrated,
            matched_index: if r.matched_index >= 0 {
                Some(r.matched_index as usize)
            } else {
                None
            },
            tag,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Index
// ═════════════════════════════════════════════════════════════════════

/// In-memory embedding index for batch deduplication.
///
/// Uses linear scan (O(N) per query). For larger corpora, use [`Router`]
/// which adds Band-LSH routing for sub-linear query time.
///
/// This type is NOT `Send` or `Sync` because the underlying C library
/// is not thread-safe for mutation.
pub struct Index {
    ptr: *mut ffi::trine_s1_index_t,
    // *const () is !Send + !Sync, preventing cross-thread use
    _not_send_sync: PhantomData<*const ()>,
}

impl Index {
    /// Create a new empty index with the given configuration.
    pub fn new(config: Config) -> Result<Self, Error> {
        let c_config = config.as_ffi();
        let ptr = unsafe { ffi::trine_s1_index_create(&c_config) };
        if ptr.is_null() {
            return Err(Error::AllocationFailed);
        }
        Ok(Index { ptr, _not_send_sync: PhantomData })
    }

    /// Add an embedding to the index with an optional tag.
    ///
    /// Returns the assigned zero-based index of the new entry.
    pub fn add(&mut self, embedding: &Embedding, tag: Option<&str>) -> Result<usize, Error> {
        let c_tag = match tag {
            Some(s) => {
                let cs = CString::new(s).map_err(|_| Error::InvalidPath)?;
                cs.into_raw()
            }
            None => ptr::null_mut(),
        };

        let result = unsafe {
            ffi::trine_s1_index_add(self.ptr, embedding.0.as_ptr(), c_tag)
        };

        // Free the CString if we allocated one
        if !c_tag.is_null() {
            unsafe { drop(CString::from_raw(c_tag)); }
        }

        if result < 0 {
            return Err(Error::AllocationFailed);
        }
        Ok(result as usize)
    }

    /// Query the index for the best match to a candidate embedding.
    pub fn query(&self, candidate: &Embedding) -> QueryResult {
        let r = unsafe { ffi::trine_s1_index_query(self.ptr, candidate.0.as_ptr()) };

        let tag = if r.matched_index >= 0 {
            self.tag(r.matched_index as usize)
                .map(|s| s.to_string())
        } else {
            None
        };

        QueryResult::from_ffi(r, tag)
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        unsafe { ffi::trine_s1_index_count(self.ptr) as usize }
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the tag for an entry by index. Returns `None` if no tag was set
    /// or the index is out of bounds.
    pub fn tag(&self, index: usize) -> Option<&str> {
        let ptr = unsafe { ffi::trine_s1_index_tag(self.ptr, index as std::os::raw::c_int) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// Save the index to a binary file.
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let c_path = path_to_cstring(path)?;
        let rc = unsafe { ffi::trine_s1_index_save(self.ptr, c_path.as_ptr()) };
        if rc != 0 {
            return Err(Error::IoError(format!("failed to save index to {}", path.display())));
        }
        Ok(())
    }

    /// Load an index from a binary file.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let c_path = path_to_cstring(path)?;
        let ptr = unsafe { ffi::trine_s1_index_load(c_path.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::IoError(format!("failed to load index from {}", path.display())));
        }
        Ok(Index { ptr, _not_send_sync: PhantomData })
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::trine_s1_index_free(self.ptr); }
            self.ptr = ptr::null_mut();
        }
    }
}

impl fmt::Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Index({} entries)", self.len())
    }
}

// ═════════════════════════════════════════════════════════════════════
// RecallMode
// ═════════════════════════════════════════════════════════════════════

/// Recall mode for the routed index, trading speed for completeness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecallMode {
    /// 1 probe/band, cap 200. Maximum speed.
    Fast,
    /// 3 probes/band, cap 500. Default.
    Balanced,
    /// 5 probes/band, cap 2000. Maximum recall.
    Strict,
}

impl RecallMode {
    fn to_c_int(self) -> std::os::raw::c_int {
        match self {
            RecallMode::Fast => ffi::TRINE_RECALL_FAST,
            RecallMode::Balanced => ffi::TRINE_RECALL_BALANCED,
            RecallMode::Strict => ffi::TRINE_RECALL_STRICT,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// RouteStats
// ═════════════════════════════════════════════════════════════════════

/// Statistics from a routed query.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteStats {
    /// Number of full cosine comparisons performed.
    pub candidates_checked: usize,
    /// Total entries in the index.
    pub total_entries: usize,
    /// Speedup factor (total_entries / candidates_checked).
    pub speedup: f32,
}

impl RouteStats {
    fn from_ffi(s: &ffi::trine_route_stats_t) -> Self {
        RouteStats {
            candidates_checked: s.candidates_checked as usize,
            total_entries: s.total_entries as usize,
            speedup: s.speedup,
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// RouterResult
// ═════════════════════════════════════════════════════════════════════

/// Result of a routed query, including routing statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct RouterResult {
    /// The query result (duplicate flag, similarity, matched index/tag).
    pub result: QueryResult,
    /// Routing statistics (candidates checked, speedup).
    pub stats: RouteStats,
}

// ═════════════════════════════════════════════════════════════════════
// Router
// ═════════════════════════════════════════════════════════════════════

/// Band-LSH routed embedding index.
///
/// Wraps the Stage-1 index with locality-sensitive hashing for sub-linear
/// query time. Reduces per-query comparisons by 10-50x without hurting
/// recall significantly.
///
/// This type is NOT `Send` or `Sync` because the underlying C library
/// is not thread-safe for mutation.
pub struct Router {
    ptr: *mut ffi::trine_route_t,
    // *const () is !Send + !Sync, preventing cross-thread use
    _not_send_sync: PhantomData<*const ()>,
}

impl Router {
    /// Create a new empty routed index with the given configuration.
    pub fn new(config: Config) -> Result<Self, Error> {
        let c_config = config.as_ffi();
        let ptr = unsafe { ffi::trine_route_create(&c_config) };
        if ptr.is_null() {
            return Err(Error::AllocationFailed);
        }
        Ok(Router { ptr, _not_send_sync: PhantomData })
    }

    /// Add an embedding to the routed index with an optional tag.
    ///
    /// Returns the assigned zero-based index of the new entry.
    pub fn add(&mut self, embedding: &Embedding, tag: Option<&str>) -> Result<usize, Error> {
        let c_tag = match tag {
            Some(s) => {
                let cs = CString::new(s).map_err(|_| Error::InvalidPath)?;
                cs.into_raw()
            }
            None => ptr::null_mut(),
        };

        let result = unsafe {
            ffi::trine_route_add(self.ptr, embedding.0.as_ptr(), c_tag)
        };

        if !c_tag.is_null() {
            unsafe { drop(CString::from_raw(c_tag)); }
        }

        if result < 0 {
            return Err(Error::AllocationFailed);
        }
        Ok(result as usize)
    }

    /// Query the routed index for the best match, with routing statistics.
    pub fn query(&self, candidate: &Embedding) -> RouterResult {
        let mut stats = ffi::trine_route_stats_t {
            candidates_checked: 0,
            total_entries: 0,
            candidate_ratio: 0.0,
            speedup: 0.0,
            recall_mode: ptr::null(),
        };

        let r = unsafe {
            ffi::trine_route_query(self.ptr, candidate.0.as_ptr(), &mut stats)
        };

        let tag = if r.matched_index >= 0 {
            self.tag(r.matched_index as usize)
                .map(|s| s.to_string())
        } else {
            None
        };

        RouterResult {
            result: QueryResult::from_ffi(r, tag),
            stats: RouteStats::from_ffi(&stats),
        }
    }

    /// Number of entries in the routed index.
    pub fn len(&self) -> usize {
        unsafe { ffi::trine_route_count(self.ptr) as usize }
    }

    /// Whether the routed index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the tag for an entry by index.
    pub fn tag(&self, index: usize) -> Option<&str> {
        let ptr = unsafe { ffi::trine_route_tag(self.ptr, index as std::os::raw::c_int) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// Set the recall mode. Takes effect on the next query.
    pub fn set_recall(&mut self, mode: RecallMode) -> Result<(), Error> {
        let rc = unsafe { ffi::trine_route_set_recall(self.ptr, mode.to_c_int()) };
        if rc != 0 {
            return Err(Error::InvalidRecallMode);
        }
        Ok(())
    }

    /// Save the routed index to a binary file.
    pub fn save(&self, path: &Path) -> Result<(), Error> {
        let c_path = path_to_cstring(path)?;
        let rc = unsafe { ffi::trine_route_save(self.ptr, c_path.as_ptr()) };
        if rc != 0 {
            return Err(Error::IoError(format!("failed to save router to {}", path.display())));
        }
        Ok(())
    }

    /// Load a routed index from a binary file.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let c_path = path_to_cstring(path)?;
        let ptr = unsafe { ffi::trine_route_load(c_path.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::IoError(format!("failed to load router from {}", path.display())));
        }
        Ok(Router { ptr, _not_send_sync: PhantomData })
    }
}

impl Drop for Router {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::trine_route_free(self.ptr); }
            self.ptr = ptr::null_mut();
        }
    }
}

impl fmt::Debug for Router {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Router({} entries)", self.len())
    }
}

// ═════════════════════════════════════════════════════════════════════
// Helpers
// ═════════════════════════════════════════════════════════════════════

fn path_to_cstring(path: &Path) -> Result<CString, Error> {
    let s = path.to_str().ok_or(Error::InvalidPath)?;
    CString::new(s).map_err(|_| Error::InvalidPath)
}
