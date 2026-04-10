#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
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

# Test 4: LoadNpy / SaveNpy round-trip for new dtypes (BF16, INT8, INT64)
# Uses python3 to create .npy files then verifies validator can load them
echo "--- Test 4: NpyIO dtype round-trip (BF16, INT8, INT64) ---"
python3 -c "
import numpy as np
import struct, os

# BF16: numpy doesn't have bf16, but we can store as void2 (<V2)
# Create a simple array of known bf16 bit patterns
# float32 1.0 = 0x3F800000 → bf16 = 0x3F80
data_bf16 = np.array([0x3F80, 0x4000, 0x4040], dtype=np.uint16).view(np.dtype('<V2'))
np.save('/tmp/test_bf16.npy', data_bf16)

# INT8
data_i8 = np.array([1, -2, 127, -128], dtype=np.int8)
np.save('/tmp/test_i8.npy', data_i8)

# INT64
data_i64 = np.array([1000000, -2000000, 0], dtype=np.int64)
np.save('/tmp/test_i64.npy', data_i64)
print('dtype test arrays created')
"

# LoadNpy BF16: validator with bf16 input should load without error (will fail on missing bin)
set +e
"$VALIDATOR" --bin /tmp/nonexistent.bin \
             --name dummy \
             --inputs /tmp/test_bf16.npy \
             --expected /tmp/test_bf16.npy \
             2>&1 | grep -q "Error\|Cannot open"
rc=$?
set -e
# We expect exit 3 (binary not found), not exit 4 (load error)
set +e
"$VALIDATOR" --bin /tmp/nonexistent.bin \
             --name dummy \
             --inputs /tmp/test_bf16.npy \
             --expected /tmp/test_bf16.npy \
             2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 3 ]; then
  echo "PASS: BF16 .npy loaded successfully (fails at binary, not at load)"
else
  echo "FAIL: BF16 .npy load failed unexpectedly (exit $rc)"
  exit 1
fi

# INT8
set +e
"$VALIDATOR" --bin /tmp/nonexistent.bin \
             --name dummy \
             --inputs /tmp/test_i8.npy \
             --expected /tmp/test_i8.npy \
             2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 3 ]; then
  echo "PASS: INT8 .npy loaded successfully"
else
  echo "FAIL: INT8 .npy load failed unexpectedly (exit $rc)"
  exit 1
fi

# INT64
set +e
"$VALIDATOR" --bin /tmp/nonexistent.bin \
             --name dummy \
             --inputs /tmp/test_i64.npy \
             --expected /tmp/test_i64.npy \
             2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 3 ]; then
  echo "PASS: INT64 .npy loaded successfully"
else
  echo "FAIL: INT64 .npy load failed unexpectedly (exit $rc)"
  exit 1
fi

# Test 5: unsupported dtype → exit 4 (load error)
echo "--- Test 5: unsupported dtype (.npy with float64) → exit 4 ---"
python3 -c "
import numpy as np
np.save('/tmp/test_f64.npy', np.zeros((4,), dtype=np.float64))
"
set +e
"$VALIDATOR" --bin /tmp/nonexistent.bin \
             --name dummy \
             --inputs /tmp/test_f64.npy \
             --expected /tmp/test_f64.npy \
             2>/dev/null
rc=$?
set -e
if [ "$rc" -eq 4 ]; then
  echo "PASS: float64 dtype → exit 4"
else
  echo "FAIL: expected exit 4 for float64, got $rc"
  exit 1
fi

echo "ALL TESTS PASSED"
