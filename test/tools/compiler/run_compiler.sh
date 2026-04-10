#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
source examples/env.sh

COMPILER=./build/bin/compiler

# Test 1: binary exists and --help exits 0
echo "--- Test 1: --help ---"
set +e
"$COMPILER" --help > /dev/null 2>&1
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "PASS: --help exits 0"
else
  echo "FAIL: --help exited $rc"
  exit 1
fi

# Test 2: missing --kernel file → exit 4 (input error: file not found)
echo "--- Test 2: missing kernel file ---"
set +e
"$COMPILER" --kernel /nonexistent/kernel.cpp --output /tmp/compiler_test 2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 4 ]; then
  echo "PASS: missing kernel file → exit 4"
else
  echo "FAIL: expected exit 4, got $rc"
  exit 1
fi

# Test 3: --num-outputs 2 → exit 4 (unsupported, input error per spec)
echo "--- Test 3: --num-outputs 2 rejected ---"
echo "// dummy" > /tmp/dummy_kernel.cpp
set +e
"$COMPILER" --kernel /tmp/dummy_kernel.cpp \
            --output /tmp/compiler_test \
          --num-outputs 2 2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 4 ]; then
  echo "PASS: --num-outputs 2 → exit 4"
else
  echo "FAIL: expected exit 4, got $rc"
  exit 1
fi

echo "ALL TESTS PASSED"
