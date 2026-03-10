#!/bin/bash
# =====================================================================
# Fetch SICK Dataset (Sentences Involving Compositional Knowledge)
# =====================================================================
#
# ~9,840 sentence pairs with relatedness scores (1-5) and entailment labels.
# Source: Marelli et al. (2014), hosted on Zenodo.
#
# Usage:  ./fetch_sick.sh
# Output: data/raw/sick/SICK.txt
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/raw/sick"
OUTFILE="$DATA_DIR/SICK.txt"

PRIMARY_URL="https://raw.githubusercontent.com/text-machine-lab/MUTT/master/data/sick/SICK.txt"
FALLBACK_URL="https://zenodo.org/records/2787612/files/SICK.txt"

echo "=== Fetching SICK dataset ==="

# Idempotent: skip if already downloaded and non-empty
if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
    LINES=$(wc -l < "$OUTFILE")
    echo "Already exists: $OUTFILE ($LINES lines). Skipping."
    exit 0
fi

mkdir -p "$DATA_DIR"

# Try primary URL
echo "Downloading from: $PRIMARY_URL"
if curl -L -f -o "$OUTFILE" --connect-timeout 30 --max-time 120 "$PRIMARY_URL" 2>/dev/null; then
    echo "Download complete (primary source)."
else
    echo "Primary URL failed. Trying fallback..."
    if curl -L -f -o "$OUTFILE" --connect-timeout 30 --max-time 120 "$FALLBACK_URL" 2>/dev/null; then
        echo "Download complete (fallback source)."
    else
        echo "ERROR: Both URLs failed. Cannot download SICK dataset."
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
echo "=== SICK fetch complete ==="
