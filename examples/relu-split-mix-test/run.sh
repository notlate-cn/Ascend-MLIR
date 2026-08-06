#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/home/niu/code/llvm-project/llvm/build}"
BOOTSTRAP_BUILD_DIR="${BOOTSTRAP_BUILD_DIR:-${REPO_ROOT}/build/runtime-mix-bootstrap}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/build/runtime-mix-relu-split}"
DATA_DIR="${DATA_DIR:-${ARTIFACT_DIR}/testdata}"

KERNEL_SRC="${REPO_ROOT}/examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp"
KERNEL_NAME="${KERNEL_NAME:-fc_relu_split}"

if [[ ! -d "${LLVM_BUILD_DIR}" ]]; then
  echo "LLVM build dir not found: ${LLVM_BUILD_DIR}" >&2
  exit 2
fi

mkdir -p "${BOOTSTRAP_BUILD_DIR}/bin"
LLVM_CONFIG="${LLVM_BUILD_DIR}/bin/llvm-config"
if [[ ! -x "${LLVM_CONFIG}" ]]; then
  LLVM_CONFIG="$(command -v llvm-config || true)"
fi
if [[ -z "${LLVM_CONFIG}" ]]; then
  echo "llvm-config not found in ${LLVM_BUILD_DIR}/bin or PATH" >&2
  exit 2
fi
LLVM_FLAGS="$("${LLVM_CONFIG}" --cxxflags --ldflags --libs support --system-libs)"

if [[ ! -x "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" ]]; then
  clang++ \
    "${REPO_ROOT}/tools/mix-compiler/mix_compiler_main.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixDirectBackend.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixCommandBuilder.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixSourceAnalyzer.cpp" \
    "${REPO_ROOT}/lib/Runtime/MixStubTemplate.cpp" \
    ${LLVM_FLAGS} \
    -std=c++17 \
    -I"${REPO_ROOT}/include" \
    -o "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler"
fi

if [[ ! -x "${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" ]]; then
  clang++ \
    "${REPO_ROOT}/tools/mix-validator/mix_validator_main.cpp" \
    "${REPO_ROOT}/lib/Runtime/Executor.cpp" \
    ${LLVM_FLAGS} \
    -std=c++17 \
    -I"${REPO_ROOT}/include" \
    -o "${BOOTSTRAP_BUILD_DIR}/bin/mix-validator"
fi

rm -rf "${ARTIFACT_DIR}"
"${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" \
  --kernel "${KERNEL_SRC}" \
  --name "${KERNEL_NAME}" \
  --output "${ARTIFACT_DIR}" \
  --soc "${SOC_VERSION}"

mkdir -p "${DATA_DIR}/input" "${DATA_DIR}/output" "${DATA_DIR}/npy"
python3 "${REPO_ROOT}/examples/matmul-add-relu-sum/gen_data_fp16ab_biasn_relu.py" \
  "${DATA_DIR}/npy"

python3 - "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path
import numpy as np

data_dir = Path(sys.argv[1])
npy_dir = data_dir / "npy"
input_dir = data_dir / "input"
output_dir = data_dir / "output"

mapping = [
    ("input_a.npy", input_dir / "fc_relu_split_input_a.bin"),
    ("input_b.npy", input_dir / "fc_relu_split_input_b.bin"),
    ("input_bias.npy", input_dir / "fc_relu_split_input_bias.bin"),
]
for src_name, dst in mapping:
    np.load(npy_dir / src_name).tofile(dst)

golden = np.load(npy_dir / "output.npy")
golden.tofile(output_dir / "fc_relu_split_output.bin")
golden.tofile(output_dir / "golden.bin")
PY

if [[ ! -f "${ARTIFACT_DIR}/out/tiling.bin" ]]; then
  echo "Expected tiling artifact missing: ${ARTIFACT_DIR}/out/tiling.bin" >&2
  exit 3
fi

"${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" \
  --artifact-root "${ARTIFACT_DIR}" \
  --input-dir "${DATA_DIR}/input" \
  --golden "${DATA_DIR}/output/golden.bin" \
  --output-file "${DATA_DIR}/output/actual.bin" \
  --soc "${SOC_VERSION}"

"${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" \
  --artifact-root "${ARTIFACT_DIR}" \
  --input-dir "${DATA_DIR}/input" \
  --golden "${DATA_DIR}/output/golden.bin" \
  --output-file "${DATA_DIR}/output/direct-actual.bin" \
  --soc "${SOC_VERSION}" \
  --force-direct-packed

python3 - "${DATA_DIR}/output/golden.bin" "${DATA_DIR}/output/actual.bin" "${DATA_DIR}/output/direct-actual.bin" <<'PY'
import hashlib
import sys
import numpy as np

golden_path, actual_path, direct_path = sys.argv[1:]

def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def verify(path: str, label: str) -> None:
    golden = np.fromfile(golden_path, dtype=np.float32)
    actual = np.fromfile(path, dtype=np.float32)
    if golden.shape != actual.shape:
      raise SystemExit(f"{label}: shape mismatch {actual.shape} vs {golden.shape}")
    diff = np.abs(actual - golden)
    print(f"{label}_max_abs_diff={diff.max():.6e}")
    print(f"{label}_mean_abs_diff={diff.mean():.6e}")
    if not np.array_equal(actual, golden):
      raise SystemExit(f"{label}: binary mismatch")
    print(f"{label}_md5={md5(path)}")

print(f"golden_md5={md5(golden_path)}")
verify(actual_path, "runner")
verify(direct_path, "direct")
print("test pass")
PY

echo "artifact_dir=${ARTIFACT_DIR}"
echo "data_dir=${DATA_DIR}"
