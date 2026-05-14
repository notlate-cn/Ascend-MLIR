#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
runtime_verify_setup_env
export LD_LIBRARY_PATH="$(runtime_verify_runtime_ld_library_path)"
runtime_verify_prepare_build_dir
runtime_verify_build_runtime_core
runtime_verify_build_example_toolchain
runtime_verify_build_mix_compiler

EXAMPLE_LOG="$(mktemp /tmp/runtime-mix-repeat-example.XXXXXX.log)"
cleanup() {
  rm -f "${EXAMPLE_LOG}"
}
trap cleanup EXIT

bash examples/matmul-add-leakyrelu/run.sh >"${EXAMPLE_LOG}" 2>&1

BUILD_DIR="${PROJECT_ROOT}/examples/matmul-add-leakyrelu/build_mainline"
ARTIFACT_DIR="${BUILD_DIR}/artifact"
MANIFEST="${BUILD_DIR}/run_manifest.json"
RUNTIME_SESSION="${PROJECT_ROOT}/build/bin/runtime-session"

for i in $(seq 1 20); do
  status=0
  ASCEND_DAV_SIM_VERSION="${DAV_SIM_VERSION}" \
    LD_LIBRARY_PATH="$(runtime_verify_mix_ld_library_path "${ARTIFACT_DIR}")" \
    "${RUNTIME_SESSION}" --run-manifest "${MANIFEST}" --run \
    >/tmp/runtime-mix-repeat.log 2>&1 || status=$?
  if [ "${status}" -eq 134 ] || [ "${status}" -eq 139 ]; then
    echo "retrying mix runtime-session repeat after simulator process exit ${status} at iteration ${i}" >&2
    sleep 1
    status=0
    ASCEND_DAV_SIM_VERSION="${DAV_SIM_VERSION}" \
      LD_LIBRARY_PATH="$(runtime_verify_mix_ld_library_path "${ARTIFACT_DIR}")" \
      "${RUNTIME_SESSION}" --run-manifest "${MANIFEST}" --run \
      >/tmp/runtime-mix-repeat.log 2>&1 || status=$?
  fi
  if [ "${status}" -ne 0 ]; then
    cat /tmp/runtime-mix-repeat.log >&2 || true
    echo "FAIL: mix runtime-session repeat failed at iteration ${i}" >&2
    exit "${status}"
  fi
done

echo "mix runtime-session repeat passed"
