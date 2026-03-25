#!/usr/bin/env bash
# test/tools/runtime/run_runtime.sh
# Builds and runs the lib/Runtime unit test suite.
# Does NOT require a simulator or .bin file.
#
# Usage:
#   cd /home/niu/code/Ascend-MLIR
#   bash test/tools/runtime/run_runtime.sh
set -e
cd /home/niu/code/Ascend-MLIR

LLVM_BUILD=~/code/llvm-project/llvm/build

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
