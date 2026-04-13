#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

source "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh"
source "${PROJECT_ROOT}/scripts/resolve_llvm_env.sh"

ASCEND_HOME="$(resolve_ascend_home || true)"
if [ -z "${ASCEND_HOME}" ]; then
  echo "Error: set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME before running mix reuse test"
  exit 1
fi
export ASCEND_HOME_PATH="${ASCEND_HOME}"
source "${PROJECT_ROOT}/examples/env.sh" >/dev/null

LLVM_BUILD="$(require_llvm_build_dir || true)"
if [ -z "${LLVM_BUILD}" ]; then
  exit 1
fi

echo "--- Preparing mix example artifact and manifest ---"
bash examples/matmul-add-leakyrelu/run.sh >/tmp/runtime_mix_output_reuse_setup.log 2>&1 || true

MANIFEST="${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu-data/runtime-manifest.json"
ARTIFACT_DIR="${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu"
RUNTIME_SESSION="${PROJECT_ROOT}/build/runtime-mix-bootstrap/bin/runtime-session"

test -f "${MANIFEST}"
test -x "${RUNTIME_SESSION}"

SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
ASCEND_DAV_SIM_VERSION="${ASCEND_DAV_SIM_VERSION:-dav_3002}"
CANN_ARCH="$(resolve_cann_arch_dir)"
ASCEND_LIB64="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
SOC_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC_VERSION}/lib"
DAV_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${ASCEND_DAV_SIM_VERSION}/lib"
DEVICE_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64/device/lib64"

for i in $(seq 1 10); do
  echo "--- Reusing mix output path: run ${i} ---"
  ASCEND_DAV_SIM_VERSION="${ASCEND_DAV_SIM_VERSION}" \
  LD_LIBRARY_PATH="${ARTIFACT_DIR}/out:${ASCEND_LIB64}:${SOC_SIM_LIB}:${DAV_SIM_LIB}:${DEVICE_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${RUNTIME_SESSION}" --run-manifest "${MANIFEST}" --run \
    >/tmp/runtime_mix_output_reuse.log 2>&1
done

echo "Mix output reuse regression passed"
