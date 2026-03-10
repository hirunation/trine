#!/bin/bash
# =====================================================================
# Fetch SNLI Dataset (Stanford Natural Language Inference)
# =====================================================================
#
# ~570K sentence pairs with entailment/neutral/contradiction labels.
# Source: Bowman et al. (2015), Stanford NLP.
# Format: JSONL with fields: gold_label, sentence1, sentence2, ...
#
# Usage:  ./fetch_snli.sh
# Output: data/raw/snli/snli_1.0_train.jsonl, snli_1.0_dev.jsonl, snli_1.0_test.jsonl
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/raw/snli"
ZIPFILE="$DATA_DIR/snli_1.0.zip"
MARKER="$DATA_DIR/snli_1.0_train.jsonl"

URL="https://nlp.stanford.edu/projects/snli/snli_1.0.zip"

echo "=== Fetching SNLI dataset ==="

# Idempotent: skip if already downloaded and non-empty
if [ -f "$MARKER" ] && [ -s "$MARKER" ]; then
    LINES=$(wc -l < "$MARKER")
    echo "Already exists: $MARKER ($LINES lines). Skipping."
    exit 0
fi

mkdir -p "$DATA_DIR"

# Download zip (~90MB)
echo "Downloading from: $URL"
echo "(This is ~90MB, may take a minute...)"
if ! curl -L -f -o "$ZIPFILE" --connect-timeout 30 --max-time 600 --progress-bar "$URL"; then
    echo "ERROR: Download failed for SNLI."
    rm -f "$ZIPFILE"
    exit 1
fi

# Verify zip is non-empty
if [ ! -s "$ZIPFILE" ]; then
    echo "ERROR: Downloaded zip is empty."
    rm -f "$ZIPFILE"
    exit 1
fi

# Extract
echo "Extracting..."
# SNLI zip contains a snli_1.0/ top-level directory
unzip -o -q "$ZIPFILE" -d "$DATA_DIR"

# If extraction created snli_1.0/ subdirectory, move contents up
if [ -d "$DATA_DIR/snli_1.0" ]; then
    mv "$DATA_DIR/snli_1.0/"* "$DATA_DIR/" 2>/dev/null || true
    rmdir "$DATA_DIR/snli_1.0" 2>/dev/null || true
fi

# Clean up zip
rm -f "$ZIPFILE"

# Verify at least the train file exists
if [ ! -s "$DATA_DIR/snli_1.0_train.jsonl" ]; then
    echo "ERROR: Extraction failed — snli_1.0_train.jsonl not found or empty."
    echo "Contents of $DATA_DIR:"
    ls -la "$DATA_DIR"
    exit 1
fi

# Report
echo ""
echo "=== SNLI dataset files ==="
for f in "$DATA_DIR"/snli_1.0_*.jsonl; do
    if [ -f "$f" ]; then
        LINES=$(wc -l < "$f")
        SIZE=$(du -h "$f" | cut -f1)
        echo "  $(basename "$f"): $LINES lines, $SIZE"
    fi
done
echo ""
echo "=== SNLI fetch complete ==="
