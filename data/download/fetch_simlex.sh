#!/bin/bash
# =====================================================================
# Fetch SimLex-999 Dataset
# =====================================================================
#
# 999 word pairs with human similarity scores (0-10).
# Source: Hill et al. (2015).
#
# Usage:  ./fetch_simlex.sh
# Output: data/raw/simlex/SimLex-999.txt
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/raw/simlex"
OUTFILE="$DATA_DIR/SimLex-999.txt"

PRIMARY_URL="https://raw.githubusercontent.com/benathi/word2gm/master/evaluation_data/SimLex-999/SimLex-999.txt"
FALLBACK_URL="https://raw.githubusercontent.com/yumeng5/Spherical-Text-Embedding/master/datasets/SimLex-999/SimLex-999.txt"

echo "=== Fetching SimLex-999 dataset ==="

# Idempotent: skip if already downloaded and non-empty
if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
    LINES=$(wc -l < "$OUTFILE")
    echo "Already exists: $OUTFILE ($LINES lines). Skipping."
    exit 0
fi

mkdir -p "$DATA_DIR"

# Try primary URL (direct .txt download)
echo "Downloading from: $PRIMARY_URL"
if curl -L -f -o "$OUTFILE" --connect-timeout 30 --max-time 120 "$PRIMARY_URL" 2>/dev/null; then
    echo "Download complete (primary source)."
else
    echo "Primary URL failed. Trying fallback..."
    if curl -L -f -o "$OUTFILE" --connect-timeout 30 --max-time 120 "$FALLBACK_URL" 2>/dev/null; then
        echo "Download complete (fallback source)."
    else
        echo "ERROR: Both URLs failed. Cannot download SimLex-999 dataset."
        rm -f "$OUTFILE"
        exit 1
    fi
fi

# Verify download has content
if [ ! -s "$OUTFILE" ]; then
    echo "ERROR: Downloaded file is empty."
    rm -f "$OUTFILE"
    exit 1
fi

LINES=$(wc -l < "$OUTFILE")
SIZE=$(du -h "$OUTFILE" | cut -f1)
echo "Saved: $OUTFILE ($LINES lines, $SIZE)"
echo "=== SimLex-999 fetch complete ==="
