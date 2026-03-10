#!/bin/bash
# =====================================================================
# Fetch WordSim-353 Dataset
# =====================================================================
#
# 353 word pairs with human relatedness scores (0-10).
# Source: Finkelstein et al. (2002).
#
# Usage:  ./fetch_wordsim.sh
# Output: data/raw/wordsim/combined.tab  (combined set, tab-separated)
# Also:   data/raw/wordsim/set1.tab, set2.tab (subsets)
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/raw/wordsim"
OUTFILE="$DATA_DIR/combined.tab"
ZIPFILE="$DATA_DIR/wordsim353.zip"

PRIMARY_URL="https://gabrilovich.com/resources/data/wordsim353/wordsim353.zip"
FALLBACK_URL="http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip"

echo "=== Fetching WordSim-353 dataset ==="

# Idempotent: skip if already downloaded and non-empty
if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
    LINES=$(wc -l < "$OUTFILE")
    echo "Already exists: $OUTFILE ($LINES lines). Skipping."
    exit 0
fi

mkdir -p "$DATA_DIR"

DOWNLOADED=0

# Try primary URL (zip)
echo "Downloading from: $PRIMARY_URL"
if curl -L -f -o "$ZIPFILE" --connect-timeout 30 --max-time 120 "$PRIMARY_URL" 2>/dev/null; then
    echo "Download complete (primary source)."
    DOWNLOADED=1
fi

# Try fallback zip
if [ $DOWNLOADED -eq 0 ]; then
    echo "Primary URL failed. Trying fallback..."
    if curl -L -f -o "$ZIPFILE" --connect-timeout 30 --max-time 120 "$FALLBACK_URL" 2>/dev/null; then
        echo "Download complete (fallback source)."
        DOWNLOADED=1
    fi
fi

# Verify zip and extract
if [ $DOWNLOADED -eq 1 ] && [ -s "$ZIPFILE" ]; then
    echo "Extracting..."
    unzip -o -q "$ZIPFILE" -d "$DATA_DIR"
    rm -f "$ZIPFILE"
else
    echo "ERROR: All URLs failed. Cannot download WordSim-353 dataset."
    rm -f "$ZIPFILE"
    exit 1
fi

# Verify the combined file exists
if [ ! -s "$OUTFILE" ]; then
    echo "ERROR: Extracted combined.tab not found or empty."
    exit 1
fi

LINES=$(wc -l < "$OUTFILE")
SIZE=$(du -h "$OUTFILE" | cut -f1)
echo "Saved: $OUTFILE ($LINES lines, $SIZE)"
echo "=== WordSim-353 fetch complete ==="
