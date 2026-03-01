# Changelog

All notable changes to TRINE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-03-01 — First Official Release

This is the first GitHub-ready release of TRINE, packaging all four completed development phases.

### Added (Phase 4 — CS-IDF & Field-Aware)
- Corpus-specific IDF (CS-IDF): auto-computed document-frequency weights during index build
- Field-aware indexing: embed title/body/code separately, route by field weights
- Append mode: atomic save for streaming index builds
- CS-IDF reduces noise floor by ~8-14%, zero overhead vs static IDF
- Field-aware scoring: 3.0M comparisons/sec (2 fields)

### Added (Phase 3 — Adoption & Bindings)
- Python package (pytrine): 14 files, 4,104 lines, ctypes FFI
  - TrineEncoder, TrineIndex, TrineRouter, Embedding, Lens, Canon classes
  - LangChain retriever integration (TrineRetriever, TrineEmbeddings)
  - LlamaIndex retriever integration (TrineRetriever, TrineEmbedding)
  - 57 Python tests
- Rust crate (trine): safe FFI wrapper, 21 tests + 1 doc-test
- TrineDB: REST server (11 endpoints, pure POSIX C, zero deps), 40 integration tests
- Docker: multi-stage Alpine image, docker-compose orchestration

### Added (Phase 2 — Standard Corpora Benchmarks)
- Benchmark suite across 3 domains: code (F1 0.985), legal (F1 0.990), text (F1 0.851)
- STS semantic correlation baseline (Spearman rho 0.476 — honest surface-form boundary)
- Cost equivalence model: 96% savings at 10K docs, 99.2% at 50K
- Routing performance: 100% recall, 159-381x candidate reduction

### Added (Phase 1 — Core Library)
- 240-dimensional ternary fingerprints, 4 semantic chains
- Stage-1 API: encode, compare, index, packed storage
- Band-LSH routing: sublinear query via locality-sensitive hashing
- 11 lens presets (6 generic + 5 domain-specific)
- 5 canonicalization presets (NONE, GENERAL, SUPPORT, CODE, POLICY)
- Production hardening: FNV-1a checksums, endianness markers, version gating
- CLI tools: trine_embed, trine_dedup, trine_recall, trine_bench
- 3 example programs (basic_embed, dedup_pipeline, routed_search)
- Zero external dependencies (C99 + libc + libm)

### Performance
- Encode: ~4.7M embeddings/sec (short text)
- Compare: ~6.9M raw cosine/sec, ~6.0M lens-weighted/sec
- CS-IDF compare: ~4.1M/sec (zero overhead vs static IDF)
- Routed query: 152x speedup at 10K entries
- Packed storage: 48 bytes per embedding (5x compression)
- Static library: <100 KB

### Tests
- 163 C core tests (31 categories)
- 57 Python tests (10 classes)
- 21 Rust tests
- 40 TrineDB integration tests
- **281 total, all pass**

## [1.0.0] - 2025 — Internal Development

Internal development releases spanning Phases 1-3. Not publicly distributed.
