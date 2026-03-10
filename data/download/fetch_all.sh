#!/bin/bash
# fetch_all.sh — Master download script for all TRINE datasets
# Runs all individual fetch scripts, then prepares and validates data.
#
# Usage: ./fetch_all.sh
# Runs from zero state to fully prepared, validated dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREPARE_DIR="$SCRIPT_DIR/../prepare"

echo "=== TRINE Dataset Download Pipeline ==="
echo "  Script directory: $SCRIPT_DIR"
echo ""

FAILED=0

for script in fetch_sts.sh fetch_sick.sh fetch_simlex.sh fetch_wordsim.sh fetch_mrpc.sh fetch_qqp.sh fetch_snli.sh; do
    SCRIPT_PATH="$SCRIPT_DIR/$script"
    if [ -f "$SCRIPT_PATH" ]; then
        echo "--- Running $script ---"
        if bash "$SCRIPT_PATH"; then
            echo "  OK"
        else
            echo "  FAILED"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    else
        echo "WARNING: $script not found, skipping."
        FAILED=$((FAILED + 1))
    fi
done

echo "=== Download Phase Complete ==="
echo "  Failed: $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "ERROR: $FAILED downloads failed. Fix and re-run."
    exit 1
fi

echo "=== Running Data Preparation ==="
if [ -f "$PREPARE_DIR/prepare_all.sh" ]; then
    bash "$PREPARE_DIR/prepare_all.sh"
else
    echo "Running individual preparation scripts..."
    for prep in prepare_sts.py prepare_sick.py prepare_mrpc.py prepare_qqp.py prepare_snli.py prepare_simlex.py prepare_wordsim.py; do
        if [ -f "$PREPARE_DIR/$prep" ]; then
            python3 "$PREPARE_DIR/$prep"
        fi
    done
fi

echo ""
echo "=== Creating Train/Val/Test Splits ==="
python3 "$PREPARE_DIR/split_data.py"

echo ""
echo "=== Validating Data ==="
python3 "$PREPARE_DIR/validate_data.py"

echo ""
echo "=== Pipeline Complete ==="
