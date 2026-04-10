#!/usr/bin/env bash
# test/tools/runtime/run_runtime.sh
# Builds and runs the lib/Runtime unit test suite.
# Does NOT require a simulator or .bin file.
#
# Usage:
#   cd /path/to/Ascend-MLIR
#   bash test/tools/runtime/run_runtime.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
source "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh"
source "${PROJECT_ROOT}/scripts/resolve_llvm_env.sh"

ASCEND_HOME="$(resolve_ascend_home || true)"
if [ -z "${ASCEND_HOME}" ]; then
  echo "Error: set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME before running runtime tests"
  exit 1
fi
export ASCEND_HOME_PATH="${ASCEND_HOME}"

LLVM_BUILD="$(require_llvm_build_dir || true)"
if [ -z "$LLVM_BUILD" ]; then
  exit 1
fi

# Build AscendCRuntime
echo "--- Building AscendCRuntime ---"
rm -f build/lib/libAscendCRuntime.a
cd build && cmake --build . --target AscendCRuntime -j4 && cd ..

# Compile test drivers
echo "--- Compiling runtime tests ---"
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    test/tools/runtime/test_runtime.cpp \
    build/lib/libAscendCRuntime.a \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support) \
    -ldl \
    -o /tmp/test_runtime
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    test/tools/runtime/test_taskgraph_runtime.cpp \
    build/lib/libAscendCRuntime.a \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support) \
    -ldl \
    -o /tmp/test_taskgraph_runtime

# Run
echo "--- Running test_taskgraph_runtime ---"
/tmp/test_taskgraph_runtime
echo "--- Running test_runtime ---"
/tmp/test_runtime
