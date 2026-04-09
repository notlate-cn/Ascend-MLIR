#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
source "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh"

ASCEND_HOME="$(resolve_ascend_home || true)"
if [ -z "${ASCEND_HOME}" ]; then
  echo "Error: set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME before running runner tests"
  exit 1
fi
export ASCEND_HOME_PATH="${ASCEND_HOME}"

LLVM_BUILD="${LLVM_BUILD_DIR:-$PROJECT_ROOT/../llvm-project/llvm/build}"
if [ ! -d "$LLVM_BUILD" ]; then
  echo "Error: LLVM_BUILD directory not found: $LLVM_BUILD"
  echo "Please set LLVM_BUILD_DIR environment variable"
  exit 1
fi

# Build AscendCRuntime
rm -f build/lib/libAscendCRuntime.a
cd build && cmake --build . --target AscendCRuntime -j4 && cd ..

# Compile test driver
g++ -std=c++17 \
    -I include/ \
    -I $LLVM_BUILD/include \
    -I ~/code/llvm-project/llvm/include \
    test/tools/runner/test_runner_gen.cpp \
    build/lib/libAscendCRuntime.a \
    $($LLVM_BUILD/bin/llvm-config --ldflags --libs support) \
    -ldl \
    -o /tmp/test_runner_gen

# Run test driver
/tmp/test_runner_gen

# Verify single-output runner enforces --bin required
echo "--- Testing --bin required (single output) ---"
if /tmp/runner_gen_test/runner --inputs /dev/null 2>&1 | grep -q "\-\-bin required"; then
  echo "PASS: --bin required check (single output)"
else
  echo "FAIL: --bin required check (single output)"
  exit 1
fi

# Verify multi-output runner was generated and also enforces --bin required
echo "--- Testing multi-output runner (num_outputs=2) ---"
if [ -x /tmp/runner_gen_test2/runner ]; then
  if /tmp/runner_gen_test2/runner --inputs /dev/null 2>&1 | grep -q "\-\-bin required"; then
    echo "PASS: multi-output runner compiled and enforces --bin"
  else
    echo "FAIL: multi-output runner --bin check"
    exit 1
  fi
else
  echo "FAIL: multi-output runner not found at /tmp/runner_gen_test2/runner"
  exit 1
fi

# Verify workspace_size runner was generated
echo "--- Testing workspace_size=65536 runner ---"
if [ -x /tmp/runner_gen_test4/runner ]; then
  if grep -q "65536" /tmp/runner_gen_test4/runner.cpp; then
    echo "PASS: workspace_size=65536 appears in runner.cpp"
  else
    echo "FAIL: workspace_size=65536 not in runner.cpp"
    exit 1
  fi
else
  echo "FAIL: workspace_size runner not found at /tmp/runner_gen_test4/runner"
  exit 1
fi

echo "ALL TESTS PASSED"
