#!/usr/bin/env bash
# fetch_mrpc.sh — Download MRPC (Microsoft Research Paraphrase Corpus)
#
# MRPC contains ~5,801 sentence pairs labeled as paraphrase (1) or not (0).
# Part of the GLUE benchmark.
#
# Sources tried (in order):
#   1. MegEngine/Models GitHub mirror (direct txt files, verified working)
#   2. magsail/MRPC GitHub mirror (direct txt files, verified working)
#   3. Facebook AI senteval mirror (tar.gz, may be blocked/403)
#
# Output: data/raw/mrpc/msr_paraphrase_train.txt
#         data/raw/mrpc/msr_paraphrase_test.txt
#
# Idempotent: skips download if files already exist and pass verification.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RAW_DIR="$PROJECT_ROOT/data/raw/mrpc"
TMP_DIR="$PROJECT_ROOT/data/raw/mrpc/.tmp"

TRAIN_FILE="$RAW_DIR/msr_paraphrase_train.txt"
TEST_FILE="$RAW_DIR/msr_paraphrase_test.txt"

# Expected line counts (including header)
TRAIN_LINES_MIN=4070
TRAIN_LINES_MAX=4080
TEST_LINES_MIN=1720
TEST_LINES_MAX=1730

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[MRPC]${NC} $*"; }
warn()  { echo -e "${YELLOW}[MRPC]${NC} $*"; }
error() { echo -e "${RED}[MRPC]${NC} $*" >&2; }

# --- Verification ---
verify_file() {
    local file="$1"
    local min_lines="$2"
    local max_lines="$3"
    local label="$4"

    if [ ! -f "$file" ]; then
        return 1
    fi

    local count
    count=$(wc -l < "$file")
    if [ "$count" -lt "$min_lines" ] || [ "$count" -gt "$max_lines" ]; then
        warn "$label: unexpected line count $count (expected $min_lines-$max_lines)"
        return 1
    fi

    # Check header format: should contain "Quality" (may have UTF-8 BOM prefix)
    local header
    header=$(head -1 "$file")
    if [[ "$header" != *Quality* ]]; then
        warn "$label: unexpected header format (missing 'Quality' column)"
        return 1
    fi

    return 0
}

verify_both() {
    verify_file "$TRAIN_FILE" "$TRAIN_LINES_MIN" "$TRAIN_LINES_MAX" "train" && \
    verify_file "$TEST_FILE"  "$TEST_LINES_MIN"  "$TEST_LINES_MAX"  "test"
}

# --- Idempotency check ---
if verify_both 2>/dev/null; then
    info "MRPC data already exists and passes verification. Skipping download."
    info "  Train: $TRAIN_FILE ($(wc -l < "$TRAIN_FILE") lines)"
    info "  Test:  $TEST_FILE ($(wc -l < "$TEST_FILE") lines)"
    exit 0
fi

info "Downloading MRPC dataset..."
mkdir -p "$RAW_DIR"

# --- Method 1: MegEngine/Models GitHub mirror (verified working) ---
try_megengine_mirror() {
    local base="https://raw.githubusercontent.com/MegEngine/Models/master/official/nlp/bert/glue_data/MRPC"
    info "Trying MegEngine GitHub mirror: $base"

    local train_url="$base/msr_paraphrase_train.txt"
    local test_url="$base/msr_paraphrase_test.txt"

    if curl -L -f -o "$TRAIN_FILE" --connect-timeout 15 --max-time 120 "$train_url" 2>/dev/null && \
       curl -L -f -o "$TEST_FILE"  --connect-timeout 15 --max-time 120 "$test_url"  2>/dev/null; then
        return 0
    else
        warn "MegEngine mirror download failed"
        rm -f "$TRAIN_FILE" "$TEST_FILE"
        return 1
    fi
}

# --- Method 2: magsail/MRPC GitHub mirror (verified working) ---
try_magsail_mirror() {
    local base="https://raw.githubusercontent.com/magsail/MRPC/master/MRPC"
    info "Trying magsail GitHub mirror: $base"

    local train_url="$base/msr_paraphrase_train.txt"
    local test_url="$base/msr_paraphrase_test.txt"

    if curl -L -f -o "$TRAIN_FILE" --connect-timeout 15 --max-time 120 "$train_url" 2>/dev/null && \
       curl -L -f -o "$TEST_FILE"  --connect-timeout 15 --max-time 120 "$test_url"  2>/dev/null; then
        return 0
    else
        warn "magsail mirror download failed"
        rm -f "$TRAIN_FILE" "$TEST_FILE"
        return 1
    fi
}

# --- Method 3: Facebook AI senteval tar.gz mirror ---
try_senteval_mirror() {
    local url="https://dl.fbaipublicfiles.com/senteval/senteval_data/msr-paraphrase-corpus.tar.gz"
    info "Trying senteval mirror: $url"

    mkdir -p "$TMP_DIR"
    local tarball="$TMP_DIR/msr-paraphrase-corpus.tar.gz"

    if curl -L -f -o "$tarball" --connect-timeout 15 --max-time 120 "$url" 2>/dev/null; then
        info "Download succeeded, extracting..."
        tar xzf "$tarball" -C "$TMP_DIR" 2>/dev/null || { warn "Extraction failed"; return 1; }

        local train_found test_found
        train_found=$(find "$TMP_DIR" -name "msr_paraphrase_train.txt" -type f 2>/dev/null | head -1)
        test_found=$(find "$TMP_DIR" -name "msr_paraphrase_test.txt" -type f 2>/dev/null | head -1)

        if [ -n "$train_found" ] && [ -n "$test_found" ]; then
            cp "$train_found" "$TRAIN_FILE"
            cp "$test_found" "$TEST_FILE"
            rm -rf "$TMP_DIR"
            return 0
        else
            warn "Expected files not found in tarball"
            rm -rf "$TMP_DIR"
            return 1
        fi
    else
        warn "senteval mirror download failed"
        rm -rf "$TMP_DIR"
        return 1
    fi
}

# --- Method 4: JepsonWong/BERT GitHub mirror ---
try_jepsonwong_mirror() {
    local base="https://raw.githubusercontent.com/JepsonWong/BERT/master/BERT_Classification_English/glue_data/MRPC"
    info "Trying JepsonWong GitHub mirror: $base"

    local train_url="$base/msr_paraphrase_train.txt"
    local test_url="$base/msr_paraphrase_test.txt"

    if curl -L -f -o "$TRAIN_FILE" --connect-timeout 15 --max-time 120 "$train_url" 2>/dev/null && \
       curl -L -f -o "$TEST_FILE"  --connect-timeout 15 --max-time 120 "$test_url"  2>/dev/null; then
        return 0
    else
        warn "JepsonWong mirror download failed"
        rm -f "$TRAIN_FILE" "$TEST_FILE"
        return 1
    fi
}

# --- Try each method in order ---
downloaded=false

if try_megengine_mirror; then
    downloaded=true
elif try_magsail_mirror; then
    downloaded=true
elif try_senteval_mirror; then
    downloaded=true
elif try_jepsonwong_mirror; then
    downloaded=true
fi

if [ "$downloaded" = false ]; then
    error "All download methods failed."
    error "Please download MRPC manually and place files in: $RAW_DIR/"
    error "  Expected: msr_paraphrase_train.txt, msr_paraphrase_test.txt"
    exit 1
fi

# --- Final verification ---
info "Verifying downloaded files..."

if verify_both; then
    train_lines=$(wc -l < "$TRAIN_FILE")
    test_lines=$(wc -l < "$TEST_FILE")
    train_size=$(du -h "$TRAIN_FILE" | cut -f1)
    test_size=$(du -h "$TEST_FILE" | cut -f1)

    info "MRPC download complete and verified."
    info "  Train: $TRAIN_FILE ($train_lines lines, $train_size)"
    info "  Test:  $TEST_FILE ($test_lines lines, $test_size)"
    info ""
    info "Format: TSV with columns: Quality | #1 ID | #2 ID | #1 String | #2 String"
    info "  Quality=1 means paraphrase, Quality=0 means not paraphrase"
else
    error "Verification failed. Files may be corrupted."
    error "  Train: $TRAIN_FILE"
    error "  Test:  $TEST_FILE"
    exit 1
fi
