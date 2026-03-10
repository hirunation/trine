#!/bin/bash
# =====================================================================
# TrineDB Integration Test Suite
# =====================================================================
#
# Starts trinedb on a test port, runs curl tests against all endpoints,
# verifies JSON responses, reports pass/fail.
#
# Usage:  ./test_trinedb.sh
# =====================================================================

set -e

PORT=17319  # Test port (offset from default to avoid conflicts)
TRINEDB="./trinedb"
DATA_DIR="/tmp/trinedb_test_$$"
PASS=0
FAIL=0
TOTAL=0

# ── Helpers ──────────────────────────────────────────────────────────

cleanup() {
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        wait "$PID" 2>/dev/null || true
    fi
    rm -rf "$DATA_DIR"
}
trap cleanup EXIT

check_response() {
    local test_name="$1"
    local response="$2"
    local expected_field="$3"
    TOTAL=$((TOTAL + 1))

    if echo "$response" | grep -q "$expected_field"; then
        PASS=$((PASS + 1))
        echo "  PASS  $test_name"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL  $test_name"
        echo "        Expected field: $expected_field"
        echo "        Got: $response"
    fi
}

check_status() {
    local test_name="$1"
    local http_code="$2"
    local expected_code="$3"
    TOTAL=$((TOTAL + 1))

    if [ "$http_code" = "$expected_code" ]; then
        PASS=$((PASS + 1))
        echo "  PASS  $test_name"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL  $test_name"
        echo "        Expected HTTP $expected_code, got HTTP $http_code"
    fi
}

# ── Build ────────────────────────────────────────────────────────────

echo "========================================"
echo "TrineDB Integration Tests"
echo "========================================"
echo ""

if [ ! -f "$TRINEDB" ]; then
    echo "Building trinedb..."
    make -s
    echo ""
fi

if [ ! -f "$TRINEDB" ]; then
    echo "FATAL: trinedb binary not found after build"
    exit 1
fi

# ── Start server ─────────────────────────────────────────────────────

echo "Starting trinedb on port $PORT..."
mkdir -p "$DATA_DIR"
$TRINEDB -p $PORT -d "$DATA_DIR" -q &
PID=$!

# Wait for server to be ready
for i in $(seq 1 20); do
    if curl -s -o /dev/null "http://localhost:$PORT/health" 2>/dev/null; then
        break
    fi
    sleep 0.25
done

# Verify server is up
if ! curl -s -o /dev/null "http://localhost:$PORT/health" 2>/dev/null; then
    echo "FATAL: trinedb failed to start"
    exit 1
fi

echo "Server started (PID $PID)"
echo ""

# ── Test Suite ───────────────────────────────────────────────────────

echo "--- Health & Stats ---"

RESP=$(curl -s "http://localhost:$PORT/health")
check_response "GET /health - status ok" "$RESP" '"status":"ok"'
check_response "GET /health - version"   "$RESP" '"version":"1.0.1"'
check_response "GET /health - uptime"    "$RESP" '"uptime_seconds"'

RESP=$(curl -s "http://localhost:$PORT/stats")
check_response "GET /stats - count"       "$RESP" '"count":0'
check_response "GET /stats - recall_mode" "$RESP" '"recall_mode":"balanced"'
check_response "GET /stats - threshold"   "$RESP" '"threshold"'

echo ""
echo "--- Embed ---"

RESP=$(curl -s -X POST "http://localhost:$PORT/embed" \
    -H "Content-Type: application/json" \
    -d '{"text":"hello world"}')
check_response "POST /embed - trits array" "$RESP" '"trits":\['
check_response "POST /embed - fill_ratio"  "$RESP" '"fill_ratio"'

RESP=$(curl -s -X POST "http://localhost:$PORT/embed" \
    -H "Content-Type: application/json" \
    -d '{"text":"hello world", "canon": 4}')
check_response "POST /embed with canon" "$RESP" '"trits":\['

echo ""
echo "--- Compare ---"

RESP=$(curl -s -X POST "http://localhost:$PORT/compare" \
    -H "Content-Type: application/json" \
    -d '{"a":"hello world","b":"hello world"}')
check_response "POST /compare - identical texts" "$RESP" '"similarity"'

RESP=$(curl -s -X POST "http://localhost:$PORT/compare" \
    -H "Content-Type: application/json" \
    -d '{"a":"hello world","b":"goodbye moon","lens":"edit"}')
check_response "POST /compare - different texts with lens" "$RESP" '"fill_a"'

RESP=$(curl -s -X POST "http://localhost:$PORT/compare" \
    -H "Content-Type: application/json" \
    -d '{"a":"the quick brown fox","b":"the quick brown fox","lens":"dedup"}')
check_response "POST /compare - near-duplicate" "$RESP" '"similarity"'

echo ""
echo "--- Index Add ---"

RESP=$(curl -s -X POST "http://localhost:$PORT/index/add" \
    -H "Content-Type: application/json" \
    -d '{"text":"the quick brown fox jumps over the lazy dog","tag":"doc-001"}')
check_response "POST /index/add - first doc" "$RESP" '"id":0'
check_response "POST /index/add - count"     "$RESP" '"count":1'

RESP=$(curl -s -X POST "http://localhost:$PORT/index/add" \
    -H "Content-Type: application/json" \
    -d '{"text":"a completely different document about quantum mechanics","tag":"doc-002"}')
check_response "POST /index/add - second doc" "$RESP" '"id":1'
check_response "POST /index/add - count 2"    "$RESP" '"count":2'

echo ""
echo "--- Index Add Batch ---"

RESP=$(curl -s -X POST "http://localhost:$PORT/index/add_batch" \
    -H "Content-Type: application/json" \
    -d '{"documents":[{"text":"batch doc one about cats","tag":"batch-001"},{"text":"batch doc two about dogs","tag":"batch-002"},{"text":"batch doc three about birds","tag":"batch-003"}]}')
check_response "POST /index/add_batch - added 3" "$RESP" '"added":3'
check_response "POST /index/add_batch - count 5" "$RESP" '"count":5'

echo ""
echo "--- Query ---"

RESP=$(curl -s -X POST "http://localhost:$PORT/query" \
    -H "Content-Type: application/json" \
    -d '{"text":"the quick brown fox jumps over the lazy dog"}')
check_response "POST /query - is_duplicate"   "$RESP" '"is_duplicate"'
check_response "POST /query - similarity"     "$RESP" '"similarity"'
check_response "POST /query - matched_index"  "$RESP" '"matched_index"'
check_response "POST /query - tag"            "$RESP" '"tag"'
check_response "POST /query - stats"          "$RESP" '"stats"'

RESP=$(curl -s -X POST "http://localhost:$PORT/query" \
    -H "Content-Type: application/json" \
    -d '{"text":"something completely novel and unique that has never been seen"}')
check_response "POST /query - no match" "$RESP" '"is_duplicate"'

echo ""
echo "--- Dedup ---"

RESP=$(curl -s -X POST "http://localhost:$PORT/dedup" \
    -H "Content-Type: application/json" \
    -d '{"documents":[{"text":"the cat sat on the mat","tag":"d1"},{"text":"the cat sat on the mat","tag":"d2"},{"text":"dogs are great pets","tag":"d3"},{"text":"the cat sat on a mat","tag":"d4"}],"threshold":0.60}')
check_response "POST /dedup - unique field"     "$RESP" '"unique"'
check_response "POST /dedup - duplicates field"  "$RESP" '"duplicates"'
check_response "POST /dedup - clusters array"    "$RESP" '"clusters":\['

echo ""
echo "--- Index Save/Load/Clear ---"

RESP=$(curl -s -X POST "http://localhost:$PORT/index/save" \
    -H "Content-Type: application/json" \
    -d '{}')
check_response "POST /index/save - saved"  "$RESP" '"saved":true'
check_response "POST /index/save - path"   "$RESP" '"path"'
check_response "POST /index/save - count"  "$RESP" '"count":5'

RESP=$(curl -s -X POST "http://localhost:$PORT/index/clear" \
    -H "Content-Type: application/json")
check_response "POST /index/clear - cleared" "$RESP" '"cleared":true'

RESP=$(curl -s "http://localhost:$PORT/stats")
check_response "GET /stats after clear - count 0" "$RESP" '"count":0'

RESP=$(curl -s -X POST "http://localhost:$PORT/index/load" \
    -H "Content-Type: application/json" \
    -d '{}')
check_response "POST /index/load - loaded" "$RESP" '"loaded":true'
check_response "POST /index/load - count"  "$RESP" '"count":5'

echo ""
echo "--- Error Handling ---"

CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/nonexistent")
check_status "GET unknown path - 404" "$CODE" "404"

CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:$PORT/embed" \
    -H "Content-Type: application/json" -d '')
check_status "POST /embed no body - 400" "$CODE" "400"

CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:$PORT/embed" \
    -H "Content-Type: application/json" -d '{"wrong":"field"}')
check_status "POST /embed wrong field - 400" "$CODE" "400"

CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:$PORT/compare" \
    -H "Content-Type: application/json" -d '{"a":"hello"}')
check_status "POST /compare missing b - 400" "$CODE" "400"

CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://localhost:$PORT/index/load" \
    -H "Content-Type: application/json" -d '{"path":"nonexistent.trrt"}')
check_status "POST /index/load missing file - 404" "$CODE" "404"

echo ""
echo "--- CORS Preflight ---"

RESP=$(curl -s -o /dev/null -w "%{http_code}" -X OPTIONS "http://localhost:$PORT/embed")
check_status "OPTIONS preflight - 204" "$RESP" "204"

# ── Report ───────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "Results: $PASS passed, $FAIL failed, $TOTAL total"
echo "========================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
