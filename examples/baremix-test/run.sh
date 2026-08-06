#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/home/niu/code/llvm-project/llvm/build}"
BOOTSTRAP_BUILD_DIR="${BOOTSTRAP_BUILD_DIR:-${REPO_ROOT}/build/runtime-mix-bootstrap}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/build/runtime-mix-baremix}"
DATA_DIR="${DATA_DIR:-${ARTIFACT_DIR}/testdata}"

if [[ ! -d "${LLVM_BUILD_DIR}" ]]; then
  echo "LLVM build dir not found: ${LLVM_BUILD_DIR}" >&2
  exit 2
fi

mkdir -p "${BOOTSTRAP_BUILD_DIR}/bin"
LLVM_FLAGS="$(llvm-config --cxxflags --ldflags --libs support --system-libs)"

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
  --kernel "${REPO_ROOT}/examples/baremix-test/baremix_custom.cpp" \
  --name baremix_custom \
  --output "${ARTIFACT_DIR}" \
  --soc "${SOC_VERSION}"

mkdir -p "${DATA_DIR}"
(
  cd "${DATA_DIR}"
  python3 "${REPO_ROOT}/examples/baremix-test/scripts/gen_data.py"
)

"${BOOTSTRAP_BUILD_DIR}/bin/mix-validator" \
  --artifact-root "${ARTIFACT_DIR}" \
  --input-dir "${DATA_DIR}/input" \
  --golden "${DATA_DIR}/output/golden.bin" \
  --output-file "${DATA_DIR}/output/actual.bin" \
  --soc "${SOC_VERSION}"

(
  cd "${DATA_DIR}"
  python3 "${REPO_ROOT}/examples/baremix-test/scripts/verify_result.py" \
    output/actual.bin output/golden.bin
)

echo "artifact_dir=${ARTIFACT_DIR}"
echo "data_dir=${DATA_DIR}"
