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

// ═════════════════════════════════════════════════════════════════════
// Stage-1 + Stage-2 interaction tests
// ═════════════════════════════════════════════════════════════════════

#[test]
fn test_stage2_identity_matches_stage1() {
    // Encode same text with S1 and S2 identity model
    // Verify the embeddings are identical
    let model = Stage2Model::identity().expect("create identity model");
    let texts = [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "short",
        "a moderately long piece of text with several words to test encoding",
    ];

    for text in &texts {
        let s1 = Embedding::encode(text);
        let s2 = model.encode(text, 0);
        assert_eq!(
            s1, s2,
            "identity Stage-2 model must produce identical embedding to Stage-1 for \"{text}\""
        );

        // Also verify via encode_from_trits path
        let s2_from_trits = model.encode_from_trits(&s1, 0);
        assert_eq!(
            s1, s2_from_trits,
            "identity encode_from_trits must match Stage-1 for \"{text}\""
        );

        // Similarity scores should also match
        let s1_sim = s1.similarity(&Embedding::encode("hello world"), &Lens::UNIFORM);
        let s2_sim = stage2::stage2_compare(&s2, &Stage2Model::identity().unwrap().encode("hello world", 0), &Lens::UNIFORM);
        assert!(
            (s1_sim - s2_sim).abs() < 0.001,
            "identity S2 similarity should match S1 similarity for \"{text}\": s1={s1_sim}, s2={s2_sim}"
        );
    }
}

#[test]
fn test_stage2_from_trainer() {
    // Create trainer, observe a few pairs, freeze, verify model works
    let config = stage2::HebbianConfig {
        similarity_threshold: 0.5,
        freeze_threshold: 0,
        freeze_target_density: 0.33,
        cascade_cells: 64,
        cascade_depth: 0,
        projection_mode: stage2::ProjectionMode::Diagonal,
        weighted_mode: false,
        pos_scale: 10.0,
        neg_scale: 3.0,
        sparse_k: 0,
    };
    let mut trainer = stage2::HebbianTrainer::new(config).expect("create trainer");

    // Feed several similar and dissimilar pairs
    let similar_pairs = [
        ("the cat sat on the mat", "the cat sat on a mat"),
        ("quick brown fox", "quick brown fox jumps"),
        ("hello world program", "hello world application"),
    ];
    let dissimilar_pairs = [
        ("the cat sat on the mat", "quantum physics theory"),
        ("quick brown fox", "zxywvutsrqponm"),
        ("hello world program", "financial market analysis report"),
    ];

    for (a, b) in &similar_pairs {
        let ea = Embedding::encode(a);
        let eb = Embedding::encode(b);
        let sim = ea.similarity(&eb, &Lens::UNIFORM);
        trainer.observe(&ea, &eb, sim);
    }
    for (a, b) in &dissimilar_pairs {
        let ea = Embedding::encode(a);
        let eb = Embedding::encode(b);
        let sim = ea.similarity(&eb, &Lens::UNIFORM);
        trainer.observe(&ea, &eb, sim);
    }

    // Verify metrics
    let metrics = trainer.metrics();
    assert_eq!(
        metrics.pairs_observed, 6,
        "should have observed 6 pairs, got {}",
        metrics.pairs_observed
    );

    // Freeze into a model
    let model = trainer.freeze().expect("freeze should succeed");
    let info = model.info();
    assert!(!info.is_identity, "trained model should not be identity");
    assert_eq!(info.projection_k, 3, "projection K should be 3");
    assert_eq!(info.projection_dims, 240, "projection dims should be 240");

    // Model should produce valid (non-all-zero) embeddings
    let emb = model.encode("test text for frozen model", 0);
    let ratio = emb.fill_ratio();
    assert!(
        ratio > 0.0,
        "frozen model embedding should have non-zero fill ratio, got {ratio}"
    );

    // Model should be usable for comparison
    let a = model.encode("hello world", 0);
    let b = model.encode("hello world", 0);
    let sim = stage2::stage2_compare(&a, &b, &Lens::UNIFORM);
    assert!(
        (sim - 1.0).abs() < 0.001,
        "self-similarity via trained model should be ~1.0, got {sim}"
    );
}

#[test]
fn test_stage2_blend_bounds() {
    // Verify blend with alpha=1.0 gives pure S1 and alpha=0.0 gives pure S2
    let model = Stage2Model::identity().expect("create identity model");
    let text_a = "the quick brown fox";
    let text_b = "the slow brown fox";
    let lens = &Lens::UNIFORM;
    let depth = 0;

    // Compute Stage-1 and Stage-2 similarities independently
    let s1_a = Embedding::encode(text_a);
    let s1_b = Embedding::encode(text_b);
    let s1_sim = s1_a.similarity(&s1_b, lens);

    let s2_a = model.encode_from_trits(&s1_a, depth);
    let s2_b = model.encode_from_trits(&s1_b, depth);
    let s2_sim = stage2::stage2_compare(&s2_a, &s2_b, lens);

    // alpha=1.0 should give pure Stage-1
    let blend_pure_s1 = stage2::stage2_blend(text_a, text_b, &model, 1.0, lens, depth);
    assert!(
        (blend_pure_s1 - s1_sim).abs() < 0.001,
        "alpha=1.0 should give pure S1: blend={blend_pure_s1}, s1={s1_sim}"
    );

    // alpha=0.0 should give pure Stage-2
    let blend_pure_s2 = stage2::stage2_blend(text_a, text_b, &model, 0.0, lens, depth);
    assert!(
        (blend_pure_s2 - s2_sim).abs() < 0.001,
        "alpha=0.0 should give pure S2: blend={blend_pure_s2}, s2={s2_sim}"
    );

    // alpha=0.5 should be midpoint
    let blend_mid = stage2::stage2_blend(text_a, text_b, &model, 0.5, lens, depth);
    let expected_mid = 0.5 * s1_sim + 0.5 * s2_sim;
    assert!(
        (blend_mid - expected_mid).abs() < 0.001,
        "alpha=0.5 should be midpoint: blend={blend_mid}, expected={expected_mid}"
    );

    // Also verify with a non-identity model to ensure formula holds
    let random_model = Stage2Model::random(64, 42).expect("create random model");
    let r_s2_a = random_model.encode_from_trits(&s1_a, depth);
    let r_s2_b = random_model.encode_from_trits(&s1_b, depth);
    let r_s2_sim = stage2::stage2_compare(&r_s2_a, &r_s2_b, lens);

    let r_blend_s1 = stage2::stage2_blend(text_a, text_b, &random_model, 1.0, lens, depth);
    assert!(
        (r_blend_s1 - s1_sim).abs() < 0.001,
        "alpha=1.0 with random model should still give pure S1: blend={r_blend_s1}, s1={s1_sim}"
    );

    let r_blend_s2 = stage2::stage2_blend(text_a, text_b, &random_model, 0.0, lens, depth);
    assert!(
        (r_blend_s2 - r_s2_sim).abs() < 0.001,
        "alpha=0.0 with random model should give pure S2: blend={r_blend_s2}, s2={r_s2_sim}"
    );
}
