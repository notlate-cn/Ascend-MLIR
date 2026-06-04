#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_ROOT}"

bash scripts/sync-and-submit.sh --help | grep -q -- "--skip-sim"
bash scripts/sync-and-submit.sh --help | grep -q -- "--npu-timeout"
bash scripts/real-npu-ci/docker-run.sh --help | grep -q -- "--skip-sim"
bash scripts/real-npu-ci/docker-run.sh --help | grep -q -- "--npu-timeout"
bash scripts/real-npu-ci/docker-run.sh --help | grep -q -- "relu-broadcast-diagnostics"
bash scripts/real-npu-ci/run-real-npu-job.sh --help | grep -q "ASCEND_MLIR_CI_SKIP_SIM"
bash scripts/real-npu-ci/run-real-npu-job.sh --help | grep -q "ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS"
bash scripts/sync-and-submit.sh --help | grep -q -- "relu-broadcast-diagnostics"
bash scripts/sync-and-submit.sh --list-cases | grep -q -- "^relu-broadcast-diagnostics$"

grep -q "ASCEND_MLIR_CI_SKIP_SIM" scripts/sync-and-submit.sh
grep -q "ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS" scripts/sync-and-submit.sh
grep -q "ASCEND_MLIR_CI_SKIP_SIM" scripts/real-npu-ci/docker-run.sh
grep -q "ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS" scripts/real-npu-ci/docker-run.sh
grep -q -- "--prepare-runtime-artifacts" scripts/real-npu-ci/run-real-npu-job.sh
grep -q "env -i" scripts/real-npu-ci/run-real-npu-job.sh
grep -q "timeout --kill-after" scripts/real-npu-ci/run-real-npu-job.sh
grep -q "run_relu_broadcast_diagnostics" scripts/real-npu-ci/run-real-npu-job.sh

for case_name in \
  add-broadcast-concat \
  broadcast-add-reduce \
  gather-elementwise-fusion \
  matmul-add-leakyrelu \
  relu-broadcast-transpose \
  split-relu-brc-add-mul; do
  grep -q -- "--prepare-runtime-artifacts" "examples/${case_name}/run-mainline.sh"
  bash -n "examples/${case_name}/run-mainline.sh"
done

bash -n scripts/sync-and-submit.sh
bash -n scripts/real-npu-ci/docker-run.sh
bash -n scripts/real-npu-ci/run-real-npu-job.sh
