#!/bin/bash
# prepare_all.sh — Run all dataset preparation scripts
# Converts raw downloads to standardized JSONL format
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== TRINE Data Preparation Pipeline ==="
echo ""

TOTAL=0
FAILED=0

for script in prepare_sts.py prepare_sick.py prepare_mrpc.py prepare_qqp.py prepare_snli.py prepare_simlex.py prepare_wordsim.py; do
    echo "--- Running $script ---"
    if python3 "$script"; then
        TOTAL=$((TOTAL + 1))
        echo "  OK"
    else
        echo "  FAILED (skipping)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=== Preparation Summary ==="
echo "  Succeeded: $TOTAL / $((TOTAL + FAILED))"
echo "  Failed:    $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED datasets failed preparation. Check raw downloads."
    exit 1
fi

echo "All datasets prepared. Run split_data.py to create train/val/test splits."
