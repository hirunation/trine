#!/bin/bash
# evaluate.sh — Evaluate a TRINE build: benchmarks + full test suite
# Runs quick benchmarks and all 434 C tests (Stage-1 + Stage-2).
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# ── Defaults ─────────────────────────────────────────────────────────

MODEL_PATH="${1:-model/release/model.trine2}"

echo "=== TRINE Evaluation Pipeline ==="
echo "  Project root: $PROJ_ROOT"
echo "  Model path:   $MODEL_PATH"
echo ""

# ── Step 1: Check prerequisites ─────────────────────────────────────

echo "=== Step 1: Checking prerequisites ==="

if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "  Model found: $MODEL_PATH ($MODEL_SIZE)"
else
    echo "  Model not found: $MODEL_PATH (benchmarks will run without Stage-2 model)"
fi
echo ""

# ── Step 2: Build if needed ──────────────────────────────────────────

echo "=== Step 2: Building project ==="

BENCH_BIN="build/trine_bench"
TEST_BIN="build/trine_test_sim"

if [ -x "$BENCH_BIN" ] && [ -x "$TEST_BIN" ]; then
    echo "  Binaries already built (skipping build)"
else
    echo "  Building project..."
    make
    echo "  Build complete."
fi
echo ""

# ── Step 3: Run benchmark suite ──────────────────────────────────────

echo "=== Step 3: Running benchmark suite ==="

if [ ! -x "$BENCH_BIN" ]; then
    echo "  ERROR: $BENCH_BIN not found."
    exit 1
fi

START_BENCH=$(date +%s)
./"$BENCH_BIN" --quick
END_BENCH=$(date +%s)
BENCH_ELAPSED=$((END_BENCH - START_BENCH))

echo ""
echo "  Benchmark duration: ${BENCH_ELAPSED}s"
echo ""

# ── Step 4: Run full test suite ──────────────────────────────────────

echo "=== Step 4: Running full test suite (434 C tests) ==="

START_TEST=$(date +%s)
make test
END_TEST=$(date +%s)
TEST_ELAPSED=$((END_TEST - START_TEST))

echo ""
echo "  Test suite duration: ${TEST_ELAPSED}s"
echo ""

# ── Summary ──────────────────────────────────────────────────────────

TOTAL_ELAPSED=$((BENCH_ELAPSED + TEST_ELAPSED))

echo "=== Evaluation Summary ==="
echo "  Benchmark duration: ${BENCH_ELAPSED}s"
echo "  Test suite duration: ${TEST_ELAPSED}s"
echo "  Total duration:     ${TOTAL_ELAPSED}s"
if [ -f "$MODEL_PATH" ]; then
    echo "  Model evaluated:    $MODEL_PATH"
fi
echo ""
echo "  Status: SUCCESS"
echo "  All tests passed. Ready for release."
