# TRINE Benchmark Report

Standard corpora benchmarks for TRINE v2.0.0 across three domains: code, general text, and legal/policy.

All results are deterministic and reproducible via `prepare.sh` + `trine_corpus_bench`.

---

## Corpora

| Corpus | Domain | Entries | Originals | Synthetic Dups | Source |
|--------|--------|---------|-----------|----------------|--------|
| `code_corpus.jsonl` | Code | 1,036 | 259 | 777 (typo/ws/rename) | CPython stdlib (10 files) |
| `text_corpus.jsonl` | General | 2,592 | 864 | 1,728 (edit/reorder) | Gutenberg (Frankenstein, Pride & Prejudice) |
| `legal_corpus.jsonl` | Legal/Policy | 1,161 | 387 | 774 (number/punct) | US Constitution, Plato's Republic |
| `sts_pairs.jsonl` | Mixed (STS) | 8,628 pairs | -- | -- | SemEval STS Benchmark 2012-2017 |

---

## 1. STS Benchmark -- Semantic Correlation

TRINE is a **surface-form** system. This benchmark honestly measures its correlation with human semantic similarity judgments. TRINE is not expected to match neural models here -- this establishes the contract boundary.

| Lens | Spearman rho | Mean Abs Error | High-Pair Corr | Low-Pair Corr |
|------|-------------|----------------|----------------|---------------|
| **uniform** | **0.476** | 0.230 | 0.228 | 0.338 |
| dedup | 0.476 | 0.227 | 0.217 | 0.325 |
| support | 0.470 | 0.225 | 0.201 | 0.315 |
| legal | 0.465 | 0.228 | 0.192 | 0.321 |
| policy | 0.458 | 0.227 | 0.183 | 0.314 |
| vocab | 0.452 | 0.221 | 0.192 | 0.284 |
| phrase | 0.442 | 0.230 | 0.175 | 0.309 |
| code | 0.412 | 0.236 | 0.248 | 0.307 |

**Interpretation**: Spearman rho ~0.48 confirms TRINE captures surface-form overlap, not semantic meaning. Neural models (sentence-transformers) achieve rho ~0.85 on this benchmark. TRINE's role is Stage-1 filtering, not semantic understanding -- this result validates the two-stage architecture.

---

## 2. Dedup Precision/Recall

Ground-truth near-duplicates created via controlled transformations. Measured with the `dedup` lens.

### Code Domain (1,036 entries)

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.50 | 0.024 | 1.000 | 0.046 |
| 0.60 | 0.065 | 1.000 | 0.122 |
| 0.65 | 0.394 | 0.999 | 0.565 |
| **0.70** | **0.982** | **0.988** | **0.985** |
| 0.75 | 0.999 | 0.892 | 0.942 |
| 0.80 | 0.998 | 0.718 | 0.835 |
| 0.90 | 1.000 | 0.432 | 0.604 |

Peak F1: **0.985** at threshold 0.70

### General Text Domain (2,592 entries)

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.50 | 0.005 | 1.000 | 0.010 |
| 0.60 | 0.021 | 0.997 | 0.040 |
| 0.65 | 0.191 | 0.974 | 0.320 |
| **0.70** | **0.902** | **0.805** | **0.851** |
| 0.75 | 0.995 | 0.591 | 0.742 |
| 0.80 | 1.000 | 0.509 | 0.675 |
| 0.90 | 1.000 | 0.109 | 0.196 |

Peak F1: **0.851** at threshold 0.70

### Legal/Policy Domain (1,161 entries)

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.50 | 0.012 | 1.000 | 0.023 |
| 0.60 | 0.069 | 1.000 | 0.129 |
| 0.65 | 0.534 | 1.000 | 0.697 |
| **0.70** | **0.988** | **0.992** | **0.990** |
| 0.75 | 1.000 | 0.955 | 0.977 |
| 0.80 | 1.000 | 0.908 | 0.952 |
| 0.90 | 1.000 | 0.682 | 0.811 |

Peak F1: **0.990** at threshold 0.70

### Summary

| Domain | Peak F1 | Best Threshold | Precision@Best | Recall@Best |
|--------|---------|----------------|----------------|-------------|
| Code | 0.985 | 0.70 | 0.982 | 0.988 |
| Legal | 0.990 | 0.70 | 0.988 | 0.992 |
| Text | 0.851 | 0.70 | 0.902 | 0.805 |

The default threshold of 0.60 is conservative (recall=1.0 everywhere, precision low). Threshold 0.70 is the sweet spot across all domains.

---

## 3. Canonicalization Impact

F1 improvement (delta) when applying domain-specific canonicalization presets.

### Code Domain + CODE Canon

| Threshold | F1 (raw) | F1 (canon) | Delta |
|-----------|---------|-----------|-------|
| 0.65 | 0.565 | 0.608 | **+0.043** |
| 0.70 | 0.985 | 0.981 | -0.004 |
| 0.80 | 0.835 | 0.858 | **+0.023** |
| 0.85 | 0.721 | 0.805 | **+0.084** |
| 0.90 | 0.604 | 0.800 | **+0.196** |

CODE canon dramatically improves recall at high thresholds (+0.196 F1 at 0.90) by normalizing identifiers (camelCase/snake_case unification).

### Legal Domain + POLICY Canon

| Threshold | F1 (raw) | F1 (canon) | Delta |
|-----------|---------|-----------|-------|
| 0.60 | 0.129 | 0.163 | **+0.033** |
| 0.70 | 0.990 | 0.885 | -0.106 |
| 0.90 | 0.811 | 0.850 | **+0.039** |

POLICY canon helps at extreme thresholds by collapsing numbers, but hurts at 0.70 due to increased false positives from over-normalization. Recommendation: use POLICY only at high thresholds (0.85+).

### General Text + SUPPORT Canon

| Threshold | F1 (raw) | F1 (canon) | Delta |
|-----------|---------|-----------|-------|
| 0.70 | 0.851 | 0.774 | -0.077 |
| 0.85 | 0.613 | 0.633 | **+0.020** |

SUPPORT canon provides marginal improvement on literary text (no timestamps/UUIDs to strip). Best suited for log/ticket data.

---

## 4. Routing Performance

Band-LSH routing reduces per-query comparisons to sublinear cost.

### Near-Duplicate Recall (trine_recall --validate)

| Mode | Corpus Size | Recall | Avg Candidates | Verdict |
|------|-------------|--------|----------------|---------|
| FAST | 5,000 | **100.0%** | 9.9 | PASS |
| BALANCED | 5,000 | **100.0%** | 19.4 | PASS |
| STRICT | 5,000 | **100.0%** | 29.3 | PASS |

100% recall for true near-duplicates across all modes.

### Candidate Reduction on Real Corpora

| Corpus | Entries | Mode | Avg Candidates | Reduction |
|--------|---------|------|----------------|-----------|
| Code | 1,036 | FAST | 4.5 | **230x** |
| Code | 1,036 | BALANCED | 6.5 | **159x** |
| Legal | 1,161 | FAST | 4.5 | **258x** |
| Legal | 1,161 | BALANCED | 6.9 | **168x** |
| Text | 2,592 | FAST | 6.8 | **381x** |
| Text | 2,592 | BALANCED | 11.9 | **218x** |

### Query Latency (BALANCED mode)

| Corpus | Entries | Brute QPS | Routed QPS | p50 | p95 | p99 |
|--------|---------|-----------|------------|-----|-----|-----|
| Code | 1,036 | 4,025 | -- | 2.2 us | 3.8 us | 4.9 us |
| Legal | 1,161 | 3,637 | -- | 3.0 us | 6.1 us | 10.1 us |
| Text | 2,592 | 1,554 | -- | 3.8 us | 5.7 us | 8.1 us |

### Scaling Curve (BALANCED, text corpus)

| Corpus Size | Avg Candidates | Candidate Ratio | p50 Latency |
|-------------|----------------|-----------------|-------------|
| 100 | 100.0 | 1.000 (fallback) | 21.9 us |
| 500 | 3.5 | 0.007 | 1.4 us |
| 1,000 | 5.7 | 0.006 | 3.8 us |
| 2,000 | 9.6 | 0.005 | 3.2 us |

Candidate ratio decreases as corpus grows -- routing becomes more valuable at scale.

---

## 5. Cost Equivalence

Comparison: neural-only pipeline vs TRINE Stage-1 + neural Stage-2.

**Assumptions**: Embedding cost $0.0001/call, DB query cost $0.00001/query, 3 chunks per document, ~40 candidates after TRINE filtering.

| Documents | Neural-Only | TRINE + Neural | Savings |
|-----------|-------------|----------------|---------|
| 100 | $0.11 | $0.40 | -264% (TRINE overhead > savings) |
| 500 | $2.55 | $2.00 | **22%** |
| 1,000 | $10.10 | $4.00 | **60%** |
| 5,000 | $250.50 | $20.00 | **92%** |
| 10,000 | $1,001.00 | $40.00 | **96%** |
| 50,000 | $25,005.00 | $200.00 | **99.2%** |

**Break-even point**: ~400 documents. Below that, the fixed cost of TRINE Stage-2 exceeds brute-force neural cost. Above that, TRINE's O(N) Stage-1 vs O(N^2) all-pairs comparison dominates.

---

## 6. Phase 4: CS-IDF & Field-Aware Performance (v2.0.0)

### CS-IDF Compare Throughput

Per-comparison cost of CS-IDF weighted cosine vs static IDF and raw cosine.

| Method | Throughput | Latency | Overhead vs Raw |
|--------|-----------|---------|-----------------|
| Raw cosine (240-dim) | ~15M/sec | 66 ns | baseline |
| Static IDF cosine | 4.1M/sec | 247 ns | -- |
| **CS-IDF cosine (240-dim)** | **4.2M/sec** | **239 ns** | ~0% vs static IDF |
| **CS-IDF + lens cosine** | **4.0M/sec** | **253 ns** | ~3% vs static IDF |

CS-IDF weighted cosine adds **zero measurable overhead** compared to the existing static IDF cosine. Both use the same per-channel multiply-accumulate loop, just with different weight sources.

### Field-Aware Scoring Throughput

| Method | Throughput | Latency | Note |
|--------|-----------|---------|------|
| Field cosine (2 fields) | 3.0M/sec | 332 ns | 2x 240-dim chain cosines |
| Field + IDF (2 fields) | 2.0M/sec | 491 ns | 2x IDF-weighted chain cosines |

Field-aware scoring is ~1.5-2x slower than single-field cosine due to per-field iteration. Still fast enough for sub-millisecond query latency on indices up to 100K entries.

### CS-IDF Noise Floor

Pairwise similarity of disjoint (unrelated) random documents. Lower noise floor = better separation between duplicates and non-duplicates.

| Metric | Uniform Cosine | CS-IDF Cosine | Improvement |
|--------|---------------|---------------|-------------|
| Average similarity | 0.557 | 0.514 | **-7.7%** |
| Min similarity | 0.444 | 0.394 | -11.4% |
| Max similarity | 0.685 | 0.655 | -4.3% |
| Downweighted channels | 0/240 | **180/240** | 75% of channels suppressed |

CS-IDF reduces the noise floor by ~8%, pushing unrelated documents further below the duplicate threshold. This widens the gap between true duplicates (sim > 0.7) and noise (sim ~0.5), improving precision without changing recall.

### CS-IDF Routed Query Latency

End-to-end query latency with CS-IDF scoring through the Band-LSH routing layer. 5,000 entries, BALANCED recall mode.

| Query Type | p50 | p95 | p99 | Avg | Overhead |
|-----------|-----|-----|-----|-----|----------|
| Standard (lens cosine) | 6.3 us | 9.1 us | 15.7 us | 6.6 us | baseline |
| **CS-IDF (IDF-weighted)** | **6.0 us** | **8.3 us** | **13.6 us** | **6.2 us** | **-6.5%** |

CS-IDF query is actually slightly faster due to the IDF weights pruning low-information channels from the cosine inner loop. The overhead is within measurement noise — **well within the 10-20% latency budget**.

---

## 7. Recommended Configuration

Based on benchmark results:

| Parameter | Recommended | Rationale |
|-----------|------------|-----------|
| Threshold | 0.70 | Peak F1 across all domains |
| Lens | dedup | Best balance of precision/recall |
| Recall mode | BALANCED | 100% recall, ~20 candidates |
| CS-IDF | ON for corpora > 100 docs | ~8% noise reduction, zero overhead |
| Field-aware | 2-3 fields (title, body) | Improves precision on structured docs |
| Canon (code) | CODE, threshold 0.85+ | +0.084 F1 improvement |
| Canon (logs) | SUPPORT | Strips timestamps/UUIDs |
| Canon (legal) | POLICY, threshold 0.85+ | +0.039 F1 improvement |
| Canon (general) | NONE or GENERAL | Minimal benefit on clean text |

---

## Reproducibility

```bash
cd hteb/benchmarks
bash prepare.sh                                                    # Prepare datasets
cd ..
make all                                                           # Build tools

# Core benchmarks (Sections 1-5)
./build/trine_corpus_bench --sts benchmarks/prepared/sts_pairs.jsonl
./build/trine_corpus_bench --dedup benchmarks/prepared/code_corpus.jsonl
./build/trine_corpus_bench --dedup benchmarks/prepared/code_corpus.jsonl --canon code
./build/trine_corpus_bench --dedup benchmarks/prepared/text_corpus.jsonl
./build/trine_corpus_bench --dedup benchmarks/prepared/legal_corpus.jsonl
./build/trine_corpus_bench --dedup benchmarks/prepared/legal_corpus.jsonl --canon policy
./build/trine_corpus_bench --routing benchmarks/prepared/text_corpus.jsonl
./build/trine_corpus_bench --cost --docs 10000 --embedding-cost 0.0001 --db-cost 0.00001

# Phase 4 benchmarks (Section 6)
./build/trine_bench --phase4-only              # Full Phase 4 benchmarks
./build/trine_bench --phase4-only --quick       # Quick mode (10x fewer iterations)
```
