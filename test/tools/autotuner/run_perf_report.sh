#!/usr/bin/env bash
set -e
cd /home/niu/code/Ascend-MLIR
source examples/env.sh

AUTOTUNER=./build/bin/autotuner

# Test 1: --perf-report in --help
echo "--- Test 1: --perf-report in --help ---"
set +e
"$AUTOTUNER" --help 2>&1 | grep -q "perf-report"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "PASS: --perf-report present in --help"
else
  echo "FAIL: --perf-report not found in --help"
  exit 1
fi

# Test 2: --sim-report rejected as unknown option
echo "--- Test 2: --sim-report rejected ---"
set +e
err=$("$AUTOTUNER" --sim-report=trace 2>&1)
rc=$?
set -e
if [ "$rc" -ne 0 ] && echo "$err" | grep -q "sim-report"; then
  echo "PASS: --sim-report rejected (exit $rc)"
else
  echo "FAIL: expected --sim-report to be rejected with error mentioning 'sim-report', got: $err"
  exit 1
fi

# Test 3: --perf-report-out in --help
echo "--- Test 3: --perf-report-out in --help ---"
set +e
"$AUTOTUNER" --help 2>&1 | grep -q "perf-report-out"
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "PASS: --perf-report-out present in --help"
else
  echo "FAIL: --perf-report-out not found in --help"
  exit 1
fi

echo "ALL TESTS PASSED"
