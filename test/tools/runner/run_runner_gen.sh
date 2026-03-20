#!/usr/bin/env bash
set -e
cd /home/niu/code/Ascend-MLIR
source examples/env.sh

LLVM_BUILD=~/code/llvm-project/llvm/build

# Build AscendCRuntime
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

# Verify runner enforces --bin required
echo "--- Testing --bin required ---"
if /tmp/runner_gen_test/runner --inputs /dev/null 2>&1 | grep -q "\-\-bin required"; then
  echo "PASS: --bin required check"
else
  echo "FAIL: --bin required check"
  exit 1
fi

echo "ALL TESTS PASSED"
