# Changelog

All notable changes to TRINE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.3] - 2026-03-02

### Added

- **Block-Diagonal Projection**: 4 independent 60x60 ternary matrix blocks (one per chain)
  - K=3 majority-vote block-diagonal matmul (`TRINE_S2_PROJ_BLOCK_DIAG = 3`)
  - 43,200 parameters vs 720 (diagonal) — captures intra-chain cross-channel correlations
  - Preserves chain independence (C1 constraint)
  - Block-diagonal accumulator and freeze for Hebbian training
  - Block-diagonal model persistence (.trine2 format, 43,280 bytes)
  - Block-diagonal accumulator persistence (.trine2a format)

- **Adaptive Blend Alpha**: per-S1-bucket alpha values for optimal S1/S2 blend weighting
  - `trine_s2_set_adaptive_alpha()` / `trine_s2_compare_adaptive_blend()` API
  - Automatic alpha optimization from validation data in `trine_train`

- **Batch & SIMD APIs**: cache-friendly batch operations and hardware acceleration
  - `trine_encode_shingle_batch()` — batch encoding for throughput
  - `trine_batch_compare()` / `trine_batch_compare_topk()` — cache-friendly batch comparison
  - `trine_simd_cosine_sse2()` — SSE2-accelerated ternary cosine

- **Packed Trit Storage**: `trine_pack_trits()` / `trine_unpack_trits()` — 4x compression
  for trit storage

- **Structured Error Codes**: `trine_error_t` enum for machine-readable error reporting

- **Block-Diagonal Model Creation**: `trine_s2_create_block_diagonal()` API

- **CLI Updates**
  - `trine_train`: `--block-diagonal`, `--adaptive-alpha`, `--checkpoint-dir` flags
  - `trine_embed`: `--block-diagonal` flag for block-diag random model
  - `trine_dedup`: `--block-diagonal` flag

- **Python bindings**: `Stage2Model.create_block_diagonal()`, HebbianTrainer block mode,
  adaptive alpha support
  - LangChain integration: `block_diagonal` and `adaptive_alpha` parameters
  - LlamaIndex integration: `block_diagonal` and `adaptive_alpha` parameters

- **Rust bindings**: `Stage2Model::create_block_diagonal()`, adaptive alpha support

- **Shared JSONL parser** (`trine_jsonl.h/c`): eliminated code duplication across tools

- **Golden vector test suite** (37 tests): deterministic encode/decode round-trip verification

- **Backward compatibility tests** (24 assertions): ensures .trine2 format stability

- **Block-diagonal test suite** (174 new assertions across 5 test groups: 46 + 36 + 34 + 26 + 32)

- **Training pipeline integration test**: end-to-end train/checkpoint/resume/evaluate

- **Benchmark harnesses**: throughput benchmarks, projection mode comparison

- **Pipeline automation scripts**: `prepare_data.sh`, `train.sh`, `evaluate.sh`, `release.sh`

- **Model directory structure**: `model/checkpoints/`, `model/release/`

- **Examples**: updated for block-diagonal and adaptive alpha workflows

### Fixed

- **Block-diagonal accumulator sign inversion**: `trine_block_accumulator_update()`
  treated negative sign values as positive due to C truthiness of `-1`. All block-
  diagonal Hebbian training with dissimilar pairs was accumulating with inverted sign.
  Fixed to use `(sign > 0) ? 1 : -1`, consistent with the full-matrix accumulator.
- **Buffer overflow in canon SUPPORT/POLICY presets**: `trine_canon_bucket_numbers()`
  can expand single digits to 3-byte `<N>` placeholders (3x expansion). SUPPORT and
  POLICY presets allocated only `len+1` bytes, insufficient for worst-case input.
  Now allocates `3*len+1` for all presets that invoke bucket_numbers.
- **Defensive Z3 reduction in trine_project_single()**: changed `acc % 3` to
  `((acc % 3) + 3) % 3` for consistency with block-diagonal path. Not a functional
  bug (acc is always non-negative for valid trits) but eliminates a maintenance risk.
- **Thread-safe RNG**: removed static LCG state in Hebbian training; each trainer
  instance now carries its own RNG state
- **Buffer overflow in encode_depths**: added bounds check for depth parameter
- **Buffer overflow in canon CODE**: 3x buffer allocation to prevent overrun on
  heavily-escaped input
- **Trit validation on .trine2 load**: rejects models with out-of-range trit values
  instead of silently accepting corrupt data
- **Realloc atomicity in index growth**: both arrays succeed or neither commits,
  preventing half-grown state on OOM
- **OOM propagation from trine_encode_shingle()**: now returns `int` error code
  (was `void`), callers can detect allocation failure
- **JSON escaped quote handling**: field parser correctly handles `\"` inside
  JSON string values
- **JSONL dynamic line reading**: `getline` replaces fixed-size `fgets`, supporting
  arbitrarily long JSONL records

## [1.0.2] - 2026-03-02

### Added

- **Stage-2 Semantic Projection**: Hebbian diagonal gating with K=3 majority vote
  - Ternary matmul projection (src/stage2/projection/)
  - ENDO-based cascade mixing network (src/stage2/cascade/)
  - Unified forward pass with depth-parameterized embedding (src/stage2/inference/)
  - Hebbian training harness: accumulate, freeze, self-deepen (src/stage2/hebbian/)
  - .trine2 binary persistence format with FNV-1a checksums (src/stage2/persist/)
  - +28.8% Spearman rho over Stage-1 baseline (0.2812 blend vs 0.2184)
  - ~1M encodes/sec at depth=0, 172,880 byte model

- **Training CLI** (`trine_train`): --data, --epochs, --diagonal, --density,
  --similarity-threshold, --depth, --save, --load, --val, --deepen

- **Semantic dedup** (`trine_dedup --semantic`): --blend, --s2-only, --s2-depth
  for Stage-2 enhanced near-duplicate detection

- **Model loading** (`trine_embed --model`): load trained .trine2 models for
  Stage-2 embedding from the CLI

- **Python Stage-2 bindings**: Stage2Model, Stage2Encoder, HebbianTrainer
  - LangChain integration with optional stage2_model parameter
  - LlamaIndex integration with optional stage2_model parameter
  - 37 new tests (94 total Python tests)

- **Rust Stage-2 bindings**: Stage2Model, HebbianTrainer, stage2_compare, stage2_blend
  - 21 new tests (44 total Rust tests)

- **Examples**: semantic_search.c (S1 vs S2 comparison), train_custom.c (Hebbian workflow)

- **Data foundation**: 7 datasets (STS-B, SICK, MRPC, QQP, SNLI, SimLex, WordSim),
  174K pairs, stratified train/val/test splits

- **159 Stage-2 C tests**: projection(14), majority(10), cascade(10), pipeline(30),
  hebbian(45), freeze(20), self-deepen(13), persistence(17)

### Added (Phase A-D — v1.0.2 Improvements)

- **Weighted Hebbian mode** (Phase A1): magnitude-scaled updates with configurable
  positive/negative scaling factors for stronger gradient signal

- **Dataset rebalancing** (Phase A2): per-source training weights with probabilistic
  downsampling for under-weighted sources (up to 8 named sources)

- **Threshold schedule** (Phase A3): `trine_hebbian_set_threshold()` for adjusting
  similarity threshold between epochs during curriculum learning

- **Gate-aware comparison** (Phase B1): `trine_s2_compare_gated()` skips
  uninformative channels (where majority of diagonal gates are zero) for
  noise-reduced cosine similarity

- **Per-chain blend comparison** (Phase B2): `trine_s2_compare_chain_blend()`
  blends S1 and S2 per-chain similarities with independent alpha weights

- **Accumulator persistence** (Phase C): `.trine2a` binary format (691,272 bytes)
  for saving/loading Hebbian accumulator state. Enables warm-start training,
  curriculum learning, and incremental model updates
  - `--save-accum PATH` and `--load-accum PATH` CLI flags for `trine_train`
  - `trine_accumulator_from_frozen()` reconstructs accumulators from .trine2 models

- **Sparse cross-channel projection** (Phase D1): top-K per output row sparse
  freeze with W=0 entries skipped during projection
  - `--sparse K` CLI flag for `trine_train`
  - New projection mode: `TRINE_S2_PROJ_SPARSE` (mode=2)

- **Stacked projection depth** (Phase D2): re-apply learned projection at each
  depth tick instead of random cascade mixing
  - `--stacked` CLI flag for `trine_train`

- **112 new C tests**: Phase A-B tests (63 assertions), sparse projection tests
  (49 assertions) — total 434 C tests

### Changed

- Project restructured from flat root to `src/` module layout
- Makefile updated for Stage-2 builds (16 object files in libtrine.a)
- Python bindings version bumped to 1.0.2, Makefile compiles Stage-2 sources
- Rust bindings build.rs compiles Stage-2 sources

### Changed (Phase A-D)

- TrineHebbianConfig struct expanded: weighted_mode, pos_scale, neg_scale,
  source_weights[8], n_source_weights, sparse_k fields
- trine_s2_model struct expanded: projection_mode, stacked_depth fields
- Makefile: trine_accumulator_persist.o added to PERSIST_OBJS

### Fixed

- (none)

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
