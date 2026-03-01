//! Basic example: encode, compare, and index with TRINE.
//!
//! Run with: cargo run --example basic

use trine::{Canon, Config, Embedding, Index, Lens, Router, RecallMode, encode_batch};

fn main() {
    println!("=== TRINE Rust Bindings Example ===\n");

    // ── Encoding ─────────────────────────────────────────────────────

    let texts = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumped over the lazy dogs",
        "A completely different sentence about quantum physics",
        "Error at 2024-01-15: connection refused from 192.168.1.1",
        "Error at 2024-02-20: connection refused from 10.0.0.5",
    ];

    println!("--- Encoding ---");
    for text in &texts {
        let emb = Embedding::encode(text);
        println!(
            "  [{:.0}% fill] \"{}\"",
            emb.fill_ratio() * 100.0,
            if text.len() > 50 { &text[..50] } else { text }
        );
    }

    // ── Batch encoding ───────────────────────────────────────────────

    println!("\n--- Batch Encoding ---");
    let text_refs: Vec<&str> = texts.iter().copied().collect();
    let embeddings = encode_batch(&text_refs).expect("batch encode failed");
    println!("  Encoded {} texts in batch", embeddings.len());

    // ── Pairwise similarity ──────────────────────────────────────────

    println!("\n--- Pairwise Similarity (DEDUP lens) ---");
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = embeddings[i].similarity(&embeddings[j], &Lens::DEDUP);
            let label = if sim > 0.6 { " ** DUPLICATE" } else { "" };
            println!(
                "  [{i}] vs [{j}]: {sim:.3}{label}"
            );
        }
    }

    // ── Canonicalization ─────────────────────────────────────────────

    println!("\n--- Canonicalization ---");
    let raw = texts[3];
    let canonical = Canon::Support.apply(raw);
    println!("  Raw:   \"{raw}\"");
    println!("  Canon: \"{canonical}\"");

    let raw_emb = Embedding::encode(raw);
    let canon_emb = Embedding::encode_with_canon(texts[4], Canon::Support);
    let sim_raw = raw_emb.similarity(&Embedding::encode(texts[4]), &Lens::SUPPORT);
    let sim_canon = Embedding::encode_with_canon(raw, Canon::Support)
        .similarity(&canon_emb, &Lens::SUPPORT);
    println!("  Raw similarity:   {sim_raw:.3}");
    println!("  Canon similarity: {sim_canon:.3}");

    // ── Pack/Unpack ──────────────────────────────────────────────────

    println!("\n--- Pack/Unpack ---");
    let original = &embeddings[0];
    let packed = original.pack().expect("pack failed");
    let unpacked = packed.unpack().expect("unpack failed");
    println!(
        "  240 bytes -> {} packed bytes -> 240 bytes",
        packed.bytes().len()
    );
    println!("  Roundtrip OK: {}", original == &unpacked);

    // ── Index ────────────────────────────────────────────────────────

    println!("\n--- Index ---");
    let config = Config {
        threshold: 0.60,
        lens: Lens::DEDUP,
        calibrate: true,
    };

    let mut index = Index::new(config).expect("index creation failed");
    for (i, emb) in embeddings.iter().enumerate() {
        let tag = format!("doc-{i}");
        index.add(emb, Some(&tag)).expect("add failed");
    }
    println!("  Index size: {}", index.len());

    // Query with a near-duplicate
    let query_text = "The quick brown fox jumps over the lazy dog";
    let query_emb = Embedding::encode(query_text);
    let result = index.query(&query_emb);
    println!("  Query: \"{query_text}\"");
    println!(
        "  Result: duplicate={}, sim={:.3}, match={:?}, tag={:?}",
        result.is_duplicate, result.similarity, result.matched_index, result.tag
    );

    // ── Router ───────────────────────────────────────────────────────

    println!("\n--- Router (Band-LSH) ---");
    let mut router = Router::new(Config::default()).expect("router creation failed");
    for (i, emb) in embeddings.iter().enumerate() {
        let tag = format!("doc-{i}");
        router.add(emb, Some(&tag)).expect("add failed");
    }
    router.set_recall(RecallMode::Strict).expect("set recall failed");

    let rr = router.query(&query_emb);
    println!("  Query: \"{query_text}\"");
    println!(
        "  Result: duplicate={}, sim={:.3}, tag={:?}",
        rr.result.is_duplicate, rr.result.similarity, rr.result.tag
    );
    println!(
        "  Stats: checked={}/{}, speedup={:.1}x",
        rr.stats.candidates_checked, rr.stats.total_entries, rr.stats.speedup
    );

    println!("\n=== Done ===");
}
