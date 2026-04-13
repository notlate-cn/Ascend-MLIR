#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

source "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh"
source "${PROJECT_ROOT}/scripts/resolve_llvm_env.sh"

ASCEND_HOME="$(resolve_ascend_home || true)"
if [ -z "${ASCEND_HOME}" ]; then
  echo "Error: set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME before running mix repeat test"
  exit 1
fi
export ASCEND_HOME_PATH="${ASCEND_HOME}"
source "${PROJECT_ROOT}/examples/env.sh" >/dev/null

LLVM_BUILD="$(require_llvm_build_dir || true)"
if [ -z "${LLVM_BUILD}" ]; then
  exit 1
fi

cmake -S . -B build -DLLVM_BUILD_DIR="${LLVM_BUILD}" >/dev/null
cmake --build build --target runtime-session mix-compiler -j2 >/dev/null

EXAMPLE_LOG="$(mktemp /tmp/runtime-mix-repeat-example.XXXXXX.log)"
cleanup() {
  rm -f "${EXAMPLE_LOG}"
}
trap cleanup EXIT

bash examples/matmul-add-leakyrelu/run.sh >"${EXAMPLE_LOG}" 2>&1

ARTIFACT_DIR="${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu"
DATA_DIR="${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu-data"
MANIFEST="${DATA_DIR}/runtime-manifest.json"
RUNTIME_SESSION="${PROJECT_ROOT}/build/runtime-mix-bootstrap/bin/runtime-session"

CANN_ARCH="$(resolve_cann_arch_dir)"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
DAV_SIM_VERSION="${ASCEND_DAV_SIM_VERSION:-dav_3002}"
ASCEND_LIB64="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
SOC_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC_VERSION}/lib"
DAV_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${DAV_SIM_VERSION}/lib"
DEVICE_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64/device/lib64"

for i in $(seq 1 20); do
  if ! ASCEND_DAV_SIM_VERSION="${DAV_SIM_VERSION}" \
    LD_LIBRARY_PATH="${ARTIFACT_DIR}/out:${ASCEND_LIB64}:${SOC_SIM_LIB}:${DAV_SIM_LIB}:${DEVICE_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${RUNTIME_SESSION}" --run-manifest "${MANIFEST}" --run \
    >/tmp/runtime-mix-repeat.log 2>&1; then
    cat /tmp/runtime-mix-repeat.log >&2 || true
    echo "FAIL: mix runtime-session repeat failed at iteration ${i}" >&2
    exit 1
  fi
done

echo "mix runtime-session repeat passed"
