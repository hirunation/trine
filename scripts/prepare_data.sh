#!/bin/bash
# prepare_data.sh — Prepare and split TRINE training data
# Converts raw downloads to JSONL, then creates stratified train/val/test splits.
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

echo "=== TRINE Data Preparation Pipeline ==="
echo "  Project root: $PROJ_ROOT"
echo ""

# ── Step 1: Check for prepared JSONL data ────────────────────────────

echo "=== Step 1: Checking prepared data ==="

PREPARED_DIR="data/prepared/semantic"
PREPARE_SCRIPT="data/prepare/prepare_all.sh"

if [ -d "$PREPARED_DIR" ]; then
    JSONL_COUNT=$(find "$PREPARED_DIR" -maxdepth 1 -name "*.jsonl" 2>/dev/null | wc -l)
else
    JSONL_COUNT=0
fi

if [ "$JSONL_COUNT" -gt 0 ]; then
    echo "  Found $JSONL_COUNT JSONL files in $PREPARED_DIR (skipping preparation)"
    echo "  Files:"
    for f in "$PREPARED_DIR"/*.jsonl; do
        ROWS=$(wc -l < "$f")
        echo "    $(basename "$f"): $ROWS rows"
    done
else
    echo "  No prepared JSONL files found."
    if [ -f "$PREPARE_SCRIPT" ]; then
        echo "  Running $PREPARE_SCRIPT to prepare data..."
        bash "$PREPARE_SCRIPT"
        echo "  Preparation complete."
    else
        echo "  ERROR: $PREPARE_SCRIPT not found. Cannot prepare data."
        echo "  Please download raw data first (see data/download/fetch_all.sh)."
        exit 1
    fi
fi
echo ""

# ── Step 2: Create train/val/test splits ─────────────────────────────

echo "=== Step 2: Creating train/val/test splits ==="

SPLITS_DIR="data/splits"
SPLIT_SCRIPT="data/prepare/split_data.py"

if [ ! -f "$SPLIT_SCRIPT" ]; then
    echo "  ERROR: $SPLIT_SCRIPT not found."
    exit 1
fi

# Create splits directory if missing
mkdir -p "$SPLITS_DIR"

# Check if splits already exist and are non-empty
EXISTING_SPLITS=0
for split in train val test; do
    if [ -f "$SPLITS_DIR/${split}.jsonl" ] && [ -s "$SPLITS_DIR/${split}.jsonl" ]; then
        EXISTING_SPLITS=$((EXISTING_SPLITS + 1))
    fi
done

if [ "$EXISTING_SPLITS" -eq 3 ]; then
    echo "  All 3 split files already exist. Re-generating for idempotency..."
fi

echo "  Running $SPLIT_SCRIPT..."
python3 "$SPLIT_SCRIPT"
echo "  Split complete."
echo ""

# ── Step 3: Validate output ──────────────────────────────────────────

echo "=== Step 3: Validating output ==="

TOTAL_ROWS=0
ALL_OK=true

for split in train val test; do
    FPATH="$SPLITS_DIR/${split}.jsonl"
    if [ ! -f "$FPATH" ]; then
        echo "  ERROR: $FPATH was not created!"
        ALL_OK=false
        continue
    fi
    ROWS=$(wc -l < "$FPATH")
    SIZE=$(du -h "$FPATH" | cut -f1)
    TOTAL_ROWS=$((TOTAL_ROWS + ROWS))
    echo "  ${split}.jsonl: $ROWS rows ($SIZE)"
done

echo ""

# Run the validation script if available
VALIDATE_SCRIPT="data/prepare/validate_data.py"
if [ -f "$VALIDATE_SCRIPT" ]; then
    echo "  Running validation checks..."
    python3 "$VALIDATE_SCRIPT" || ALL_OK=false
fi

echo ""

# ── Summary ──────────────────────────────────────────────────────────

echo "=== Data Preparation Summary ==="
echo "  Total rows across splits: $TOTAL_ROWS"
echo "  Splits directory: $PROJ_ROOT/$SPLITS_DIR/"
for split in train val test; do
    if [ -f "$SPLITS_DIR/${split}.jsonl" ]; then
        ROWS=$(wc -l < "$SPLITS_DIR/${split}.jsonl")
        PCT=$(awk "BEGIN { printf \"%.1f\", $ROWS / $TOTAL_ROWS * 100 }")
        echo "    ${split}: $ROWS rows (${PCT}%)"
    fi
done

if [ "$ALL_OK" = true ]; then
    echo ""
    echo "  Status: SUCCESS"
    echo "  Next step: run scripts/train.sh"
else
    echo ""
    echo "  Status: COMPLETED WITH WARNINGS (check output above)"
    exit 1
fi
