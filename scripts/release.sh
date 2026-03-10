#!/bin/bash
# release.sh — Build, test, and package a TRINE release tarball
# Creates a clean tarball with source, docs, examples, bindings, and built tools.
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# ── Defaults ─────────────────────────────────────────────────────────

VERSION="${1:-1.0.3}"
RELEASE_NAME="trine-${VERSION}"
STAGING_DIR="/tmp/${RELEASE_NAME}"
TARBALL="/mnt/d/${RELEASE_NAME}.tar"

echo "=== TRINE Release Builder ==="
echo "  Project root: $PROJ_ROOT"
echo "  Version:      $VERSION"
echo "  Tarball:      $TARBALL"
echo ""

# ── Step 1: Clean build ──────────────────────────────────────────────

echo "=== Step 1: Clean build ==="

make clean
make

echo ""
echo "  Clean build complete."
echo ""

# ── Step 2: Run full test suite ──────────────────────────────────────

echo "=== Step 2: Running full test suite ==="

make test

echo ""
echo "  All tests passed."
echo ""

# ── Step 3: Prepare staging directory ────────────────────────────────

echo "=== Step 3: Preparing release staging directory ==="

# Clean up any previous staging directory
if [ -d "$STAGING_DIR" ]; then
    echo "  Cleaning previous staging directory..."
    rm -rf "$STAGING_DIR"
fi

mkdir -p "$STAGING_DIR"

# Copy source
echo "  Copying source..."
cp -r src/ "$STAGING_DIR/src/"

# Copy docs
echo "  Copying docs..."
if [ -d docs/ ]; then
    cp -r docs/ "$STAGING_DIR/docs/"
fi

# Copy examples
echo "  Copying examples..."
if [ -d examples/ ]; then
    cp -r examples/ "$STAGING_DIR/examples/"
fi

# Copy bindings
echo "  Copying bindings..."
if [ -d bindings/ ]; then
    cp -r bindings/ "$STAGING_DIR/bindings/"
fi

# Copy Makefile
echo "  Copying Makefile..."
cp Makefile "$STAGING_DIR/"

# Copy markdown files from project root
echo "  Copying documentation files..."
for mdfile in *.md; do
    if [ -f "$mdfile" ]; then
        cp "$mdfile" "$STAGING_DIR/"
    fi
done

# Copy LICENSE if it exists
if [ -f LICENSE ]; then
    cp LICENSE "$STAGING_DIR/"
fi

# Copy .gitignore if it exists
if [ -f .gitignore ]; then
    cp .gitignore "$STAGING_DIR/"
fi

# Copy tests
echo "  Copying tests..."
if [ -d tests/ ]; then
    cp -r tests/ "$STAGING_DIR/tests/"
fi

# Copy bench
echo "  Copying benchmarks..."
if [ -d bench/ ]; then
    cp -r bench/ "$STAGING_DIR/bench/"
fi

# Copy data preparation scripts (not raw data)
echo "  Copying data preparation scripts..."
if [ -d data/prepare/ ]; then
    mkdir -p "$STAGING_DIR/data/prepare/"
    cp -r data/prepare/ "$STAGING_DIR/data/prepare/"
fi
if [ -d data/download/ ]; then
    mkdir -p "$STAGING_DIR/data/download/"
    cp -r data/download/ "$STAGING_DIR/data/download/"
fi

# Copy automation scripts
echo "  Copying automation scripts..."
if [ -d scripts/ ]; then
    cp -r scripts/ "$STAGING_DIR/scripts/"
fi

# Copy built tools and library
echo "  Copying built artifacts..."
mkdir -p "$STAGING_DIR/build/"
for tool in trine_embed trine_dedup trine_bench trine_train trine_test_sim trine_recall trine_corpus_bench; do
    if [ -f "build/$tool" ]; then
        cp "build/$tool" "$STAGING_DIR/build/"
    fi
done
if [ -f build/libtrine.a ]; then
    cp build/libtrine.a "$STAGING_DIR/build/"
fi

# Copy model file if present
if [ -f model.trine ]; then
    echo "  Copying model.trine..."
    cp model.trine "$STAGING_DIR/"
fi

echo ""
echo "  Staging directory: $STAGING_DIR"
echo ""

# ── Step 4: Create tarball ───────────────────────────────────────────

echo "=== Step 4: Creating tarball ==="

tar cf "$TARBALL" -C /tmp "$RELEASE_NAME/"

TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
echo "  Tarball created: $TARBALL ($TARBALL_SIZE)"
echo ""

# ── Step 5: Tarball contents summary ─────────────────────────────────

echo "=== Step 5: Tarball contents summary ==="

# Count files by type
TOTAL_FILES=$(tar tf "$TARBALL" | grep -v '/$' | wc -l)
C_FILES=$(tar tf "$TARBALL" | grep -c '\.c$' || true)
H_FILES=$(tar tf "$TARBALL" | grep -c '\.h$' || true)
PY_FILES=$(tar tf "$TARBALL" | grep -c '\.py$' || true)
RS_FILES=$(tar tf "$TARBALL" | grep -c '\.rs$' || true)
MD_FILES=$(tar tf "$TARBALL" | grep -c '\.md$' || true)
SH_FILES=$(tar tf "$TARBALL" | grep -c '\.sh$' || true)

echo "  Total files:    $TOTAL_FILES"
echo "  C source (.c):  $C_FILES"
echo "  C headers (.h): $H_FILES"
echo "  Python (.py):   $PY_FILES"
echo "  Rust (.rs):     $RS_FILES"
echo "  Markdown (.md): $MD_FILES"
echo "  Shell (.sh):    $SH_FILES"
echo ""

# List top-level contents
echo "  Top-level entries:"
tar tf "$TARBALL" | sed "s|^${RELEASE_NAME}/||" | grep -v '/' | sort | while read -r entry; do
    if [ -n "$entry" ]; then
        echo "    $entry"
    fi
done

echo ""
echo "  Directories:"
tar tf "$TARBALL" | sed "s|^${RELEASE_NAME}/||" | grep '/' | cut -d'/' -f1 | sort -u | while read -r dir; do
    if [ -n "$dir" ]; then
        DIR_COUNT=$(tar tf "$TARBALL" | grep "^${RELEASE_NAME}/${dir}/" | grep -v '/$' | wc -l)
        echo "    ${dir}/ ($DIR_COUNT files)"
    fi
done
echo ""

# ── Step 6: Clean up staging directory ───────────────────────────────

echo "=== Step 6: Cleaning up ==="

rm -rf "$STAGING_DIR"
echo "  Staging directory removed."
echo ""

# ── Summary ──────────────────────────────────────────────────────────

echo "=== Release Summary ==="
echo "  Version: $VERSION"
echo "  Tarball: $TARBALL"
echo "  Size:    $TARBALL_SIZE"
echo "  Files:   $TOTAL_FILES"
echo ""
echo "  Status: SUCCESS"
echo "  Release tarball ready at $TARBALL"
