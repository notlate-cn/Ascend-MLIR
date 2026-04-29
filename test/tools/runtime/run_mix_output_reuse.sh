#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
runtime_verify_setup_env

echo "--- Preparing mix example artifact and manifest ---"
bash examples/matmul-add-leakyrelu/run.sh >/tmp/runtime_mix_output_reuse_setup.log 2>&1 || true

MANIFEST="${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu-data/runtime-manifest.json"
ARTIFACT_DIR="${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu"
RUNTIME_SESSION="${PROJECT_ROOT}/build/runtime-mix-bootstrap/bin/runtime-session"

test -f "${MANIFEST}"
test -x "${RUNTIME_SESSION}"

for i in $(seq 1 10); do
  echo "--- Reusing mix output path: run ${i} ---"
  ASCEND_DAV_SIM_VERSION="${DAV_SIM_VERSION}" \
  LD_LIBRARY_PATH="$(runtime_verify_mix_ld_library_path "${ARTIFACT_DIR}")" \
    "${RUNTIME_SESSION}" --run-manifest "${MANIFEST}" --run \
    >/tmp/runtime_mix_output_reuse.log 2>&1
done

echo "Mix output reuse regression passed"
