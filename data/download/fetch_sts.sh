#!/bin/bash
# =====================================================================
# Fetch STS-B (Semantic Textual Similarity Benchmark)
# =====================================================================
#
# Downloads train/dev/test splits from the GLUE mirror (Facebook AI)
# and falls back to the original STS Wiki source if needed.
#
# Output: data/raw/sts/{train,dev,test}.tsv
#
# Usage:  ./data/download/fetch_sts.sh        (from project root)
#         bash data/download/fetch_sts.sh
# =====================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw/sts"
TMP_DIR="$(mktemp -d)"

trap 'rm -rf "$TMP_DIR"' EXIT

# ── Sources ──────────────────────────────────────────────────────────

GLUE_URL="https://dl.fbaipublicfiles.com/glue/data/STS-B.zip"
WIKI_URL="http://ixa2.si.ehu.eus/stswiki/images/4/48/Stsbenchmark.tar.gz"

# ── Helpers ──────────────────────────────────────────────────────────

log()  { echo "  [STS]  $*"; }
die()  { echo "  [STS]  FATAL: $*" >&2; exit 1; }

check_existing() {
    if [ -f "$RAW_DIR/train.tsv" ] && [ -f "$RAW_DIR/dev.tsv" ] && [ -f "$RAW_DIR/test.tsv" ]; then
        local train_lines dev_lines test_lines
        train_lines=$(wc -l < "$RAW_DIR/train.tsv")
        dev_lines=$(wc -l < "$RAW_DIR/dev.tsv")
        test_lines=$(wc -l < "$RAW_DIR/test.tsv")
        if [ "$train_lines" -gt 100 ] && [ "$dev_lines" -gt 100 ] && [ "$test_lines" -gt 100 ]; then
            log "Already downloaded (train=$train_lines dev=$dev_lines test=$test_lines lines)"
            log "Skipping. Delete $RAW_DIR to re-download."
            exit 0
        fi
    fi
}

verify_file() {
    local path="$1"
    local name="$2"
    if [ ! -f "$path" ]; then
        die "$name not found at $path"
    fi
    local lines
    lines=$(wc -l < "$path")
    if [ "$lines" -lt 10 ]; then
        die "$name has only $lines lines -- download likely failed"
    fi
    log "$name: $lines lines, $(du -h "$path" | cut -f1) on disk"
}

# ── Main ─────────────────────────────────────────────────────────────

log "STS-B dataset fetcher"
log "Target directory: $RAW_DIR"
echo ""

# Idempotency: skip if already present
check_existing

mkdir -p "$RAW_DIR"

# ── Attempt 1: GLUE mirror (STS-B.zip) ──────────────────────────────

log "Trying GLUE mirror: $GLUE_URL"
GLUE_OK=0

if curl -fSL --connect-timeout 15 --max-time 120 -o "$TMP_DIR/STS-B.zip" "$GLUE_URL" 2>/dev/null; then
    log "Downloaded STS-B.zip ($(du -h "$TMP_DIR/STS-B.zip" | cut -f1))"
    if unzip -q -o "$TMP_DIR/STS-B.zip" -d "$TMP_DIR/glue" 2>/dev/null; then
        # GLUE layout: STS-B/{train,dev,test}.tsv
        GLUE_DIR=$(find "$TMP_DIR/glue" -type d -name "STS-B" 2>/dev/null | head -1)
        if [ -z "$GLUE_DIR" ]; then
            # Maybe flat layout
            GLUE_DIR="$TMP_DIR/glue"
        fi
        log "Extracted to $GLUE_DIR"
        ls -la "$GLUE_DIR/" 2>/dev/null || true

        # Look for the split files
        for split in train dev test; do
            src=$(find "$GLUE_DIR" -iname "${split}.tsv" -type f 2>/dev/null | head -1)
            if [ -n "$src" ] && [ -f "$src" ]; then
                cp "$src" "$RAW_DIR/${split}.tsv"
                GLUE_OK=$((GLUE_OK + 1))
            fi
        done
    fi
fi

if [ "$GLUE_OK" -eq 3 ]; then
    log "GLUE source: all 3 splits obtained"
else
    log "GLUE source incomplete ($GLUE_OK/3 splits), trying Wiki source..."

    # ── Attempt 2: Original STS Wiki (tar.gz) ───────────────────────

    log "Trying Wiki source: $WIKI_URL"

    if ! curl -fSL --connect-timeout 15 --max-time 120 -o "$TMP_DIR/Stsbenchmark.tar.gz" "$WIKI_URL" 2>/dev/null; then
        die "Both GLUE and Wiki downloads failed. Check your network."
    fi

    log "Downloaded Stsbenchmark.tar.gz ($(du -h "$TMP_DIR/Stsbenchmark.tar.gz" | cut -f1))"
    tar xzf "$TMP_DIR/Stsbenchmark.tar.gz" -C "$TMP_DIR/wiki" 2>/dev/null || \
        tar xzf "$TMP_DIR/Stsbenchmark.tar.gz" -C "$TMP_DIR" 2>/dev/null

    # Wiki layout: stsbenchmark/sts-{train,dev,test}.csv
    WIKI_DIR=$(find "$TMP_DIR" -type d -name "stsbenchmark" 2>/dev/null | head -1)
    if [ -z "$WIKI_DIR" ]; then
        WIKI_DIR="$TMP_DIR"
    fi

    for split in train dev test; do
        src=$(find "$WIKI_DIR" -iname "sts-${split}.csv" -type f 2>/dev/null | head -1)
        if [ -n "$src" ] && [ -f "$src" ]; then
            cp "$src" "$RAW_DIR/${split}.tsv"
        else
            die "Could not find ${split} split in Wiki archive"
        fi
    done

    log "Wiki source: all 3 splits obtained"
fi

# ── Verify ───────────────────────────────────────────────────────────

echo ""
log "Verifying downloads..."
verify_file "$RAW_DIR/train.tsv" "train"
verify_file "$RAW_DIR/dev.tsv"   "dev"
verify_file "$RAW_DIR/test.tsv"  "test"

echo ""
log "Sample from train.tsv (first 3 lines):"
head -3 "$RAW_DIR/train.tsv" | while IFS= read -r line; do
    echo "    $line"
done

echo ""
log "Done. STS-B data is ready at: $RAW_DIR/"
