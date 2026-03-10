#!/bin/bash
# train.sh — Train a TRINE Stage-2 Hebbian model
# Best-known config: diagonal gating, density 0.15, similarity-threshold 0.90, depth 0
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# ── Defaults ─────────────────────────────────────────────────────────

MODE="--diagonal"
EPOCHS=1
DATA_FILE="data/splits/train.jsonl"
VAL_FILE="data/splits/val.jsonl"
DENSITY=0.15
SIM_THRESHOLD=0.90
DEPTH=0
MODEL_DIR="model/release"
MODEL_PATH="$MODEL_DIR/model.trine2"
EXTRA_ARGS=()

# ── Parse arguments ──────────────────────────────────────────────────

while [ $# -gt 0 ]; do
    case "$1" in
        --diagonal)
            MODE="--diagonal"
            shift
            ;;
        --block-diagonal)
            MODE=""
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --val)
            VAL_FILE="$2"
            shift 2
            ;;
        --density)
            DENSITY="$2"
            shift 2
            ;;
        --similarity-threshold)
            SIM_THRESHOLD="$2"
            shift 2
            ;;
        --depth)
            DEPTH="$2"
            shift 2
            ;;
        --save)
            MODEL_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --diagonal              Use diagonal gating (default)"
            echo "  --block-diagonal        Use block-diagonal (full) projection"
            echo "  --epochs N              Number of training epochs (default: 1)"
            echo "  --data PATH             Training data JSONL (default: data/splits/train.jsonl)"
            echo "  --val PATH              Validation data JSONL (default: data/splits/val.jsonl)"
            echo "  --density D             Target density (default: 0.15)"
            echo "  --similarity-threshold S  Stage-1 similarity threshold (default: 0.90)"
            echo "  --depth N               Cascade depth (default: 0)"
            echo "  --save PATH             Model output path (default: model/release/model.trine2)"
            echo "  -h, --help              Show this help"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=== TRINE Stage-2 Hebbian Training ==="
echo "  Project root: $PROJ_ROOT"
echo "  Mode:         ${MODE:-block-diagonal}"
echo "  Epochs:       $EPOCHS"
echo "  Density:      $DENSITY"
echo "  Sim threshold: $SIM_THRESHOLD"
echo "  Depth:        $DEPTH"
echo "  Data:         $DATA_FILE"
echo "  Validation:   $VAL_FILE"
echo "  Model output: $MODEL_PATH"
echo ""

# ── Step 1: Check prerequisites ─────────────────────────────────────

echo "=== Step 1: Checking prerequisites ==="

if [ ! -f "$DATA_FILE" ]; then
    echo "  ERROR: Training data not found: $DATA_FILE"
    echo "  Run scripts/prepare_data.sh first."
    exit 1
fi
TRAIN_ROWS=$(wc -l < "$DATA_FILE")
echo "  Training data: $DATA_FILE ($TRAIN_ROWS rows)"

VAL_ARG=""
if [ -f "$VAL_FILE" ]; then
    VAL_ROWS=$(wc -l < "$VAL_FILE")
    echo "  Validation data: $VAL_FILE ($VAL_ROWS rows)"
    VAL_ARG="--val $VAL_FILE"
else
    echo "  Validation data: not found (training without validation)"
fi
echo ""

# ── Step 2: Build if needed ──────────────────────────────────────────

echo "=== Step 2: Building trine_train ==="

TRAIN_BIN="build/trine_train"

if [ -x "$TRAIN_BIN" ]; then
    echo "  $TRAIN_BIN already built (skipping build)"
else
    echo "  Building project..."
    make
    echo "  Build complete."
fi

if [ ! -x "$TRAIN_BIN" ]; then
    echo "  ERROR: $TRAIN_BIN not found after build."
    exit 1
fi
echo ""

# ── Step 3: Train ────────────────────────────────────────────────────

echo "=== Step 3: Training model ==="

# Create model output directory
mkdir -p "$(dirname "$MODEL_PATH")"

# Build the command
CMD="./$TRAIN_BIN"
CMD+=" --data $DATA_FILE"
CMD+=" $VAL_ARG"
CMD+=" --epochs $EPOCHS"
CMD+=" --density $DENSITY"
CMD+=" --similarity-threshold $SIM_THRESHOLD"
CMD+=" --depth $DEPTH"
if [ -n "$MODE" ]; then
    CMD+=" $MODE"
fi
CMD+=" --save $MODEL_PATH"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=" ${EXTRA_ARGS[*]}"
fi

echo "  Command: $CMD"
echo ""

START_TIME=$(date +%s)
eval "$CMD"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""

# ── Step 4: Verify output ────────────────────────────────────────────

echo "=== Step 4: Verifying output ==="

if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    MODEL_BYTES=$(wc -c < "$MODEL_PATH")
    echo "  Model saved: $MODEL_PATH"
    echo "  Model size:  $MODEL_SIZE ($MODEL_BYTES bytes)"
else
    echo "  WARNING: Model file not created at $MODEL_PATH"
fi
echo ""

# ── Summary ──────────────────────────────────────────────────────────

echo "=== Training Summary ==="
echo "  Mode:         ${MODE:-block-diagonal}"
echo "  Epochs:       $EPOCHS"
echo "  Training rows: $TRAIN_ROWS"
echo "  Duration:     ${ELAPSED}s"
if [ -f "$MODEL_PATH" ]; then
    echo "  Model:        $MODEL_PATH ($MODEL_SIZE)"
    echo ""
    echo "  Status: SUCCESS"
    echo "  Next step: run scripts/evaluate.sh"
else
    echo ""
    echo "  Status: FAILED (no model file produced)"
    exit 1
fi
