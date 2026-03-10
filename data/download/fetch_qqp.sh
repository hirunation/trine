#!/bin/bash
# =====================================================================
# Fetch QQP Dataset (Quora Question Pairs)
# =====================================================================
#
# ~400K question pairs with binary duplicate labels (0/1).
# Source: GLUE benchmark distribution (Facebook AI / NYU).
# Format: TSV with columns: id, qid1, qid2, question1, question2, is_duplicate
#
# Usage:  ./fetch_qqp.sh
# Output: data/raw/qqp/train.tsv, dev.tsv, test.tsv
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/raw/qqp"
ZIPFILE="$DATA_DIR/QQP.zip"
MARKER="$DATA_DIR/train.tsv"

# GLUE distribution URLs (tried in order)
URL_PRIMARY="https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip"
URL_FALLBACK="https://dl.fbaipublicfiles.com/glue/data/QQP.zip"

echo "=== Fetching QQP dataset ==="

# Idempotent: skip if already downloaded and non-empty
if [ -f "$MARKER" ] && [ -s "$MARKER" ]; then
    LINES=$(wc -l < "$MARKER")
    echo "Already exists: $MARKER ($LINES lines). Skipping."
    exit 0
fi

mkdir -p "$DATA_DIR"

# Download zip
download_ok=0
for URL in "$URL_PRIMARY" "$URL_FALLBACK"; do
    echo "Downloading from: $URL"
    if curl -L -f -o "$ZIPFILE" --connect-timeout 30 --max-time 600 --progress-bar "$URL"; then
        echo "Download complete."
        download_ok=1
        break
    else
        echo "Failed: $URL"
    fi
done

if [ "$download_ok" -eq 0 ]; then
    echo "ERROR: All download URLs failed for QQP."
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
# QQP zip may contain a top-level QQP/ directory or flat files — handle both
unzip -o -q "$ZIPFILE" -d "$DATA_DIR"

# If extraction created a QQP/ subdirectory, move contents up
if [ -d "$DATA_DIR/QQP" ]; then
    mv "$DATA_DIR/QQP/"* "$DATA_DIR/" 2>/dev/null || true
    rmdir "$DATA_DIR/QQP" 2>/dev/null || true
fi

# Clean up zip
rm -f "$ZIPFILE"

# Verify at least train.tsv exists
if [ ! -s "$DATA_DIR/train.tsv" ]; then
    echo "ERROR: Extraction failed — train.tsv not found or empty."
    echo "Contents of $DATA_DIR:"
    ls -la "$DATA_DIR"
    exit 1
fi

# Report
echo ""
echo "=== QQP dataset files ==="
for f in "$DATA_DIR"/*.tsv; do
    if [ -f "$f" ]; then
        LINES=$(wc -l < "$f")
        SIZE=$(du -h "$f" | cut -f1)
        echo "  $(basename "$f"): $LINES lines, $SIZE"
    fi
done
echo ""
echo "=== QQP fetch complete ==="
