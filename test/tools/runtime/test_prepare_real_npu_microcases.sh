#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HELPER="${PROJECT_ROOT}/examples/real-npu-microcases/prepare.sh"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/examples/real-npu-microcases/cases.sh"

TMP_DIR="$(mktemp -d /tmp/runtime-real-npu-microcases-test.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

bash "${HELPER}" --out-dir "${TMP_DIR}/case" --skip-compile

expected_order=(
  const640
  copy640
  copy_tbuf640
  copy_scalar640
  copy_params640
  copy_wait640
  const_with_input640
  relu_only
  broadcast_add
)
test "${REAL_NPU_MICROCASES[*]}" = "${expected_order[*]}"

relu_diagnostic_order=(
  relu_diag_broadcast_store
  relu_diag_strided_copy
  relu_diag_broadcast_add
  relu_diag_generated_buffers
  relu_diag_tiling_abi
)
test "${REAL_NPU_RELU_DIAGNOSTIC_MICROCASES[*]}" = "${relu_diagnostic_order[*]}"

for case_name in "${REAL_NPU_MICROCASES[@]}"; do
  manifest="${TMP_DIR}/case/${case_name}/run_manifest.json"
  test -f "${manifest}"
  grep -q '"backend": "npu"' "${manifest}"
  grep -q '"task_id": "'"${case_name}"'"' "${manifest}"
  grep -q '"block_dim": 1' "${manifest}"
  grep -q '"workspace_size": 8192' "${manifest}"
  grep -q '"expected_outputs"' "${manifest}"
  grep -q '"profiling": true' "${manifest}"
  test -f "${TMP_DIR}/case/${case_name}/expected.npy"
  test -f "${TMP_DIR}/case/${case_name}/${case_name}.cpp"
done

grep -q '"inputs": \[\]' "${TMP_DIR}/case/const640/run_manifest.json"
grep -q '"name": "input"' "${TMP_DIR}/case/copy640/run_manifest.json"
grep -q '"name": "input"' "${TMP_DIR}/case/copy_tbuf640/run_manifest.json"
grep -q '"name": "input"' "${TMP_DIR}/case/copy_scalar640/run_manifest.json"
grep -q '"name": "input"' "${TMP_DIR}/case/copy_params640/run_manifest.json"
grep -q '"name": "input"' "${TMP_DIR}/case/copy_wait640/run_manifest.json"
grep -q '"name": "input"' "${TMP_DIR}/case/const_with_input640/run_manifest.json"
grep -q '"name": "input"' "${TMP_DIR}/case/relu_only/run_manifest.json"
grep -q '"name": "bias"' "${TMP_DIR}/case/broadcast_add/run_manifest.json"

python3 - <<PY
import ast
import struct
from pathlib import Path

root = Path("${TMP_DIR}/case")

def load_f16_npy(path):
    data = path.read_bytes()
    assert data[:6] == b"\\x93NUMPY", path
    assert data[6:8] == b"\\x01\\x00", path
    header_len = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10:10 + header_len].decode("latin1"))
    assert header["descr"] == "<f2", path
    values = [
        struct.unpack("<e", data[i:i + 2])[0]
        for i in range(10 + header_len, len(data), 2)
    ]
    return header["shape"], values

def round_f16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]

shape, const_expected = load_f16_npy(root / "const640" / "expected.npy")
assert shape == (640,)
assert const_expected == [1.0] * 640

_, copy_input = load_f16_npy(root / "copy640" / "input.npy")
_, copy_expected = load_f16_npy(root / "copy640" / "expected.npy")
assert copy_input == copy_expected

_, copy_tbuf_input = load_f16_npy(root / "copy_tbuf640" / "input.npy")
_, copy_tbuf_expected = load_f16_npy(root / "copy_tbuf640" / "expected.npy")
assert copy_tbuf_input == copy_tbuf_expected
assert copy_tbuf_input == copy_input

_, copy_scalar_input = load_f16_npy(root / "copy_scalar640" / "input.npy")
_, copy_scalar_expected = load_f16_npy(root / "copy_scalar640" / "expected.npy")
assert copy_scalar_input == copy_scalar_expected
assert copy_scalar_input == copy_input

_, copy_params_input = load_f16_npy(root / "copy_params640" / "input.npy")
_, copy_params_expected = load_f16_npy(root / "copy_params640" / "expected.npy")
assert copy_params_input == copy_params_expected
assert copy_params_input == copy_input

_, copy_wait_input = load_f16_npy(root / "copy_wait640" / "input.npy")
_, copy_wait_expected = load_f16_npy(root / "copy_wait640" / "expected.npy")
assert copy_wait_input == copy_wait_expected
assert copy_wait_input == copy_input

_, const_with_input = load_f16_npy(root / "const_with_input640" / "input.npy")
_, const_with_input_expected = load_f16_npy(
    root / "const_with_input640" / "expected.npy"
)
assert const_with_input == copy_input
assert const_with_input_expected == [2.0] * 640

_, relu_input = load_f16_npy(root / "relu_only" / "input.npy")
_, relu_expected = load_f16_npy(root / "relu_only" / "expected.npy")
assert [max(x, 0.0) for x in relu_input] == relu_expected

shape, broadcast = load_f16_npy(root / "broadcast_add" / "input.npy")
assert shape == (1, 640)
_, bias = load_f16_npy(root / "broadcast_add" / "bias.npy")
shape, expected = load_f16_npy(root / "broadcast_add" / "expected.npy")
assert shape == (2, 640)
assert [round_f16(broadcast[i % 640] + bias[i]) for i in range(1280)] == expected
PY

bash "${HELPER}" --out-dir "${TMP_DIR}/diagnostic" --skip-compile --include-relu-diagnostics

for case_name in "${REAL_NPU_RELU_DIAGNOSTIC_MICROCASES[@]}"; do
  manifest="${TMP_DIR}/diagnostic/${case_name}/run_manifest.json"
  test -f "${manifest}"
  grep -q '"backend": "npu"' "${manifest}"
  grep -q '"task_id": "'"${case_name}"'"' "${manifest}"
  grep -q '"block_dim": 20' "${manifest}"
  if [[ "${case_name}" == "relu_diag_tiling_abi" ]]; then
    grep -q '"workspace_size": 0' "${manifest}"
  else
    grep -q '"workspace_size": 8192' "${manifest}"
  fi
  grep -q '"expected_outputs"' "${manifest}"
  grep -q '"shape": \[500, 640\]' "${manifest}"
  test -f "${TMP_DIR}/diagnostic/${case_name}/expected.npy"
  test -f "${TMP_DIR}/diagnostic/${case_name}/${case_name}.cpp"
done

grep -q '"name": "data0"' "${TMP_DIR}/diagnostic/relu_diag_broadcast_store/run_manifest.json"
grep -q '"name": "data1"' "${TMP_DIR}/diagnostic/relu_diag_strided_copy/run_manifest.json"
grep -q '"name": "data0"' "${TMP_DIR}/diagnostic/relu_diag_broadcast_add/run_manifest.json"
grep -q '"name": "data1"' "${TMP_DIR}/diagnostic/relu_diag_broadcast_add/run_manifest.json"
grep -q '"name": "data0"' "${TMP_DIR}/diagnostic/relu_diag_generated_buffers/run_manifest.json"
grep -q '"name": "data1"' "${TMP_DIR}/diagnostic/relu_diag_generated_buffers/run_manifest.json"
grep -q '"name": "data0"' "${TMP_DIR}/diagnostic/relu_diag_tiling_abi/run_manifest.json"
grep -q '"name": "data1"' "${TMP_DIR}/diagnostic/relu_diag_tiling_abi/run_manifest.json"
grep -q '"workspace_size": 0' "${TMP_DIR}/diagnostic/relu_diag_tiling_abi/run_manifest.json"
grep -q '"tiling"' "${TMP_DIR}/diagnostic/relu_diag_tiling_abi/run_manifest.json"
grep -q '"binary": "'"${TMP_DIR}"'/diagnostic/relu_diag_tiling_abi/tiling.bin"' "${TMP_DIR}/diagnostic/relu_diag_tiling_abi/run_manifest.json"
test -f "${TMP_DIR}/diagnostic/relu_diag_tiling_abi/tiling.bin"

python3 - <<PY
import ast
import struct
from pathlib import Path

root = Path("${TMP_DIR}/diagnostic")

def load_f16_npy(path):
    data = path.read_bytes()
    assert data[:6] == b"\\x93NUMPY", path
    assert data[6:8] == b"\\x01\\x00", path
    header_len = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10:10 + header_len].decode("latin1"))
    assert header["descr"] == "<f2", path
    values = [
        struct.unpack("<e", data[i:i + 2])[0]
        for i in range(10 + header_len, len(data), 2)
    ]
    return header["shape"], values

def round_f16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]

shape, data0 = load_f16_npy(root / "relu_diag_broadcast_store" / "input_data0.npy")
assert shape == (640,)
shape, broadcast_expected = load_f16_npy(root / "relu_diag_broadcast_store" / "expected.npy")
assert shape == (500, 640)
assert broadcast_expected == [max(data0[i % 640], 0.0) for i in range(500 * 640)]

shape, data1 = load_f16_npy(root / "relu_diag_strided_copy" / "input_data1.npy")
assert shape == (500, 640)
shape, strided_expected = load_f16_npy(root / "relu_diag_strided_copy" / "expected.npy")
assert shape == (500, 640)
assert strided_expected == data1

_, full_data0 = load_f16_npy(root / "relu_diag_broadcast_add" / "input_data0.npy")
_, full_data1 = load_f16_npy(root / "relu_diag_broadcast_add" / "input_data1.npy")
shape, full_expected = load_f16_npy(root / "relu_diag_broadcast_add" / "expected.npy")
assert shape == (500, 640)
assert full_data0 == data0
assert full_data1 == data1
assert full_expected == [
    round_f16(max(full_data0[i % 640], 0.0) + full_data1[i])
    for i in range(500 * 640)
]

_, generated_data0 = load_f16_npy(root / "relu_diag_generated_buffers" / "input_data0.npy")
_, generated_data1 = load_f16_npy(root / "relu_diag_generated_buffers" / "input_data1.npy")
shape, generated_expected = load_f16_npy(root / "relu_diag_generated_buffers" / "expected.npy")
assert shape == (500, 640)
assert generated_data0 == full_data0
assert generated_data1 == full_data1
assert generated_expected == full_expected

_, tiling_data0 = load_f16_npy(root / "relu_diag_tiling_abi" / "input_data0.npy")
_, tiling_data1 = load_f16_npy(root / "relu_diag_tiling_abi" / "input_data1.npy")
shape, tiling_expected = load_f16_npy(root / "relu_diag_tiling_abi" / "expected.npy")
assert shape == (500, 640)
assert tiling_data0 == full_data0
assert tiling_data1 == full_data1
assert tiling_expected == full_expected
assert struct.unpack("<6q", (root / "relu_diag_tiling_abi" / "tiling.bin").read_bytes()) == (
    32,
    32,
    640,
    500,
    640,
    1,
)
PY

echo "real npu microcase helper test passed"
