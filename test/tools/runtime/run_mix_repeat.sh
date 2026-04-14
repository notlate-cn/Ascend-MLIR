#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
runtime_verify_setup_env
runtime_verify_prepare_build_dir
runtime_verify_build_runtime_core
runtime_verify_build_mix_compiler

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

for i in $(seq 1 20); do
  if ! ASCEND_DAV_SIM_VERSION="${DAV_SIM_VERSION}" \
    LD_LIBRARY_PATH="$(runtime_verify_mix_ld_library_path "${ARTIFACT_DIR}")" \
    "${RUNTIME_SESSION}" --run-manifest "${MANIFEST}" --run \
    >/tmp/runtime-mix-repeat.log 2>&1; then
    cat /tmp/runtime-mix-repeat.log >&2 || true
    echo "FAIL: mix runtime-session repeat failed at iteration ${i}" >&2
    exit 1
  fi
done

echo "mix runtime-session repeat passed"
