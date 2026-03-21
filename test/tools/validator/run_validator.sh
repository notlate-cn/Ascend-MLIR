#!/usr/bin/env bash
set -e
cd /home/niu/code/Ascend-MLIR
source examples/env.sh

VALIDATOR=./build/bin/validator

# Test 1: --help exits 0
echo "--- Test 1: --help ---"
set +e
"$VALIDATOR" --help > /dev/null 2>&1
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "PASS: --help exits 0"
else
  echo "FAIL: --help exited $rc"
  exit 1
fi

# Test 2: --npu → exit 4 (not implemented)
echo "--- Test 2: --npu not implemented ---"
python3 -c "
import numpy as np
np.save('/tmp/dummy_in.npy', np.zeros((1,), dtype=np.float16))
np.save('/tmp/dummy_exp.npy', np.zeros((1,), dtype=np.float16))
"
set +e
"$VALIDATOR" --bin /tmp/nonexistent.bin \
             --name dummy \
             --inputs /tmp/dummy_in.npy \
             --expected /tmp/dummy_exp.npy \
             --npu 2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 4 ]; then
  echo "PASS: --npu → exit 4"
else
  echo "FAIL: expected exit 4, got $rc"
  exit 1
fi

# Test 3: missing .bin file → exit 3 (RegisterBinary fails)
echo "--- Test 3: missing .bin file ---"
set +e
"$VALIDATOR" --bin /tmp/nonexistent.bin \
             --name dummy \
             --inputs /tmp/dummy_in.npy \
             --expected /tmp/dummy_exp.npy \
             2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 3 ]; then
  echo "PASS: missing .bin → exit 3"
else
  echo "FAIL: expected exit 3, got $rc"
  exit 1
fi

echo "ALL TESTS PASSED"
