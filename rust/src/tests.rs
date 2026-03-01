//! Integration tests for the TRINE Rust bindings.

use super::*;

#[test]
fn test_encode_deterministic() {
    let a = Embedding::encode("hello world");
    let b = Embedding::encode("hello world");
    assert_eq!(a, b, "same input must produce identical embeddings");
}

#[test]
fn test_encode_case_insensitive() {
    let lower = Embedding::encode("Hello World");
    let upper = Embedding::encode("hello world");
    let mixed = Embedding::encode("HELLO WORLD");

    // Shingle encoder is case-insensitive: all forms must be identical
    assert_eq!(lower, upper, "encoding must be case-insensitive");
    assert_eq!(upper, mixed, "encoding must be case-insensitive");
}

#[test]
fn test_similarity_identity() {
    let emb = Embedding::encode("the quick brown fox");
    let sim = emb.similarity(&emb, &Lens::UNIFORM);
    // Self-similarity should be 1.0 (or very close due to float precision)
    assert!(
        (sim - 1.0).abs() < 0.001,
        "self-similarity should be ~1.0, got {sim}"
    );
}

#[test]
fn test_similarity_dissimilar() {
    let a = Embedding::encode("the quick brown fox jumps over the lazy dog");
    let b = Embedding::encode("zzzzzxqjwvkp");
    let sim = a.similarity(&b, &Lens::UNIFORM);
    // Unrelated texts should have low similarity
    assert!(
        sim < 0.6,
        "dissimilar texts should have low similarity, got {sim}"
    );
}

#[test]
fn test_similarity_similar() {
    let a = Embedding::encode("the quick brown fox");
    let b = Embedding::encode("the quick brown fox jumps");
    let sim = a.similarity(&b, &Lens::DEDUP);
    // Overlapping texts should have moderate-to-high similarity
    assert!(
        sim > 0.3,
        "similar texts should have moderate similarity, got {sim}"
    );
}

#[test]
fn test_fill_ratio() {
    let empty = Embedding::from_trits([0u8; 240]);
    assert!(
        empty.fill_ratio() < 0.001,
        "all-zero embedding should have ~0 fill ratio"
    );

    let full = Embedding::encode("a]moderately long piece of text to fill channels");
    let ratio = full.fill_ratio();
    assert!(
        ratio > 0.0 && ratio <= 1.0,
        "fill ratio should be in (0, 1], got {ratio}"
    );
}

#[test]
fn test_packed_roundtrip() {
    let original = Embedding::encode("pack me up");
    let packed = original.pack().expect("pack should succeed");
    let unpacked = packed.unpack().expect("unpack should succeed");
    assert_eq!(
        original, unpacked,
        "pack/unpack roundtrip must preserve embedding"
    );
}

#[test]
fn test_packed_bytes() {
    let emb = Embedding::encode("test packing");
    let packed = emb.pack().expect("pack should succeed");
    assert_eq!(packed.bytes().len(), 48);
}

#[test]
fn test_index_lifecycle() {
    let config = Config::default();
    let mut idx = Index::new(config).expect("index creation should succeed");
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);

    // Add entries
    let e1 = Embedding::encode("first document");
    let e2 = Embedding::encode("second document");
    let e3 = Embedding::encode("something completely different");

    let i0 = idx.add(&e1, Some("doc-1")).expect("add should succeed");
    let i1 = idx.add(&e2, Some("doc-2")).expect("add should succeed");
    let i2 = idx.add(&e3, None).expect("add should succeed");

    assert_eq!(i0, 0);
    assert_eq!(i1, 1);
    assert_eq!(i2, 2);
    assert_eq!(idx.len(), 3);
    assert!(!idx.is_empty());

    // Tags
    assert_eq!(idx.tag(0), Some("doc-1"));
    assert_eq!(idx.tag(1), Some("doc-2"));
    assert_eq!(idx.tag(2), None);

    // Query with a near-duplicate of the first document
    let candidate = Embedding::encode("first document");
    let result = idx.query(&candidate);
    assert!(result.is_duplicate, "exact match should be flagged as duplicate");
    assert_eq!(result.matched_index, Some(0));
    assert_eq!(result.tag.as_deref(), Some("doc-1"));
}

#[test]
fn test_index_save_load() {
    let dir = std::env::temp_dir();
    let path = dir.join("trine_rust_test_index.trs1");

    // Create and populate
    let config = Config::default();
    let mut idx = Index::new(config).expect("create should succeed");
    let e1 = Embedding::encode("save me");
    idx.add(&e1, Some("saved-doc")).expect("add should succeed");

    // Save
    idx.save(&path).expect("save should succeed");

    // Load
    let loaded = Index::load(&path).expect("load should succeed");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded.tag(0), Some("saved-doc"));

    // Query should still work
    let candidate = Embedding::encode("save me");
    let result = loaded.query(&candidate);
    assert!(result.is_duplicate);

    // Cleanup
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_router_lifecycle() {
    let config = Config::default();
    let mut router = Router::new(config).expect("router creation should succeed");
    assert!(router.is_empty());

    // Add entries
    let texts = [
        "the quick brown fox",
        "jumped over the lazy dog",
        "a completely unrelated sentence about quantum physics",
    ];

    for (i, text) in texts.iter().enumerate() {
        let emb = Embedding::encode(text);
        let tag = format!("doc-{i}");
        router.add(&emb, Some(&tag)).expect("add should succeed");
    }
    assert_eq!(router.len(), 3);

    // Query
    let candidate = Embedding::encode("the quick brown fox");
    let rr = router.query(&candidate);
    assert!(rr.result.is_duplicate);
    assert_eq!(rr.result.matched_index, Some(0));

    // Stats should be populated
    assert_eq!(rr.stats.total_entries, 3);
    assert!(rr.stats.candidates_checked > 0);
}

#[test]
fn test_router_save_load() {
    let dir = std::env::temp_dir();
    let path = dir.join("trine_rust_test_router.trrt");

    let config = Config::default();
    let mut router = Router::new(config).expect("create should succeed");
    let emb = Embedding::encode("persistent data");
    router.add(&emb, Some("persistent")).expect("add should succeed");

    router.save(&path).expect("save should succeed");

    let loaded = Router::load(&path).expect("load should succeed");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded.tag(0), Some("persistent"));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_router_recall_modes() {
    let config = Config::default();
    let mut router = Router::new(config).expect("create should succeed");

    router.set_recall(RecallMode::Fast).expect("fast should succeed");
    router.set_recall(RecallMode::Balanced).expect("balanced should succeed");
    router.set_recall(RecallMode::Strict).expect("strict should succeed");
}

#[test]
fn test_canon_presets() {
    // None should pass through
    let text = "  Hello   World  ";
    let none_result = Canon::None.apply(text);
    assert_eq!(none_result, text);

    // General normalizes whitespace
    let general_result = Canon::General.apply(text);
    // Collapsed whitespace: leading/trailing trimmed, runs collapsed
    assert!(!general_result.starts_with(' '), "should be trimmed");
    assert!(!general_result.ends_with(' '), "should be trimmed");
    assert!(!general_result.contains("  "), "runs should be collapsed");

    // Support strips timestamps and UUIDs
    let with_ts = "Error at 2024-01-15 14:30:00 in module";
    let support_result = Canon::Support.apply(with_ts);
    assert!(!support_result.contains("2024-01-15"), "timestamp should be stripped");

    // Code normalizes identifiers (camelCase -> lowercase words)
    let with_camel = "processUserData and getValueFromMap";
    let code_result = Canon::Code.apply(with_camel);
    // Should split camelCase and lowercase
    assert!(
        code_result.contains("process"),
        "camelCase should be split, got: {code_result}"
    );
    assert!(
        code_result.contains("user"),
        "camelCase should be split, got: {code_result}"
    );
}

#[test]
fn test_canon_encode_integration() {
    let text = "  HELLO   world  ";
    let with_canon = Embedding::encode_with_canon(text, Canon::General);
    let manual = Embedding::encode(&Canon::General.apply(text));
    assert_eq!(
        with_canon, manual,
        "encode_with_canon should equal manual canon + encode"
    );
}

#[test]
fn test_batch_encode() {
    let texts = vec!["hello", "world", "foo bar"];
    let batch = encode_batch(&texts).expect("batch encode should succeed");

    assert_eq!(batch.len(), 3);

    // Each should match individual encoding
    for (i, text) in texts.iter().enumerate() {
        let individual = Embedding::encode(text);
        assert_eq!(
            batch[i], individual,
            "batch[{i}] should match individual encoding of \"{text}\""
        );
    }
}

#[test]
fn test_batch_encode_empty() {
    let texts: Vec<&str> = vec![];
    let batch = encode_batch(&texts).expect("empty batch should succeed");
    assert!(batch.is_empty());
}

#[test]
fn test_lens_presets() {
    let a = Embedding::encode("test lens weighting behavior");
    let b = Embedding::encode("test lens weighting");

    let uniform_sim = a.similarity(&b, &Lens::UNIFORM);
    let edit_sim = a.similarity(&b, &Lens::EDIT);
    let vocab_sim = a.similarity(&b, &Lens::VOCAB);

    // All should be valid similarities
    assert!(uniform_sim >= 0.0 && uniform_sim <= 1.0, "uniform: {uniform_sim}");
    assert!(edit_sim >= -1.0 && edit_sim <= 1.0, "edit: {edit_sim}");
    assert!(vocab_sim >= -1.0 && vocab_sim <= 1.0, "vocab: {vocab_sim}");

    // Different lenses should generally give different scores
    // (not strictly required but expected for this input)
    let all_same = (uniform_sim - edit_sim).abs() < 0.0001
        && (edit_sim - vocab_sim).abs() < 0.0001;
    assert!(!all_same, "different lenses should generally give different scores");
}

#[test]
fn test_config_default() {
    let config = Config::default();
    assert!((config.threshold - 0.60).abs() < 0.001);
    assert_eq!(config.lens.weights, Lens::DEDUP.weights);
    assert!(config.calibrate);
}

#[test]
fn test_embedding_debug() {
    let emb = Embedding::encode("debug test");
    let debug = format!("{emb:?}");
    assert!(debug.starts_with("Embedding("));
    assert!(debug.contains("/240 active)"));
}

#[test]
fn test_error_display() {
    let err = Error::AllocationFailed;
    assert_eq!(format!("{err}"), "allocation failed in TRINE C library");

    let err = Error::IoError("disk full".into());
    assert_eq!(format!("{err}"), "I/O error: disk full");
}
