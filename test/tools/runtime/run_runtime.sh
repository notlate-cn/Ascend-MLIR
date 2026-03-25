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

LLVM_BUILD="${LLVM_BUILD_DIR:-$PROJECT_ROOT/../llvm-project/llvm/build}"
if [ ! -d "$LLVM_BUILD" ]; then
  echo "Error: LLVM_BUILD directory not found: $LLVM_BUILD"
  echo "Please set LLVM_BUILD_DIR environment variable"
  exit 1
fi

# Build AscendCRuntime
echo "--- Building AscendCRuntime ---"
cd build && cmake --build . --target AscendCRuntime -j4 && cd ..

# Compile test driver
echo "--- Compiling test_runtime ---"
g++ -std=c++17 \
    -I include/ \
    -I "$LLVM_BUILD/include" \
    -I ~/code/llvm-project/llvm/include \
    test/tools/runtime/test_runtime.cpp \
    build/lib/libAscendCRuntime.a \
    $("$LLVM_BUILD/bin/llvm-config" --ldflags --libs support) \
    -ldl \
    -o /tmp/test_runtime

# Run
echo "--- Running test_runtime ---"
/tmp/test_runtime
