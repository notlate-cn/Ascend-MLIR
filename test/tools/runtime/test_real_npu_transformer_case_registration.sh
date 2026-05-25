#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CASE_NAME="transformer-real-npu"

cd "${PROJECT_ROOT}"

bash scripts/sync-and-submit.sh --list-cases | grep -qx "${CASE_NAME}"
bash scripts/real-npu-ci/docker-run.sh --help | grep -q "${CASE_NAME}"
bash scripts/real-npu-ci/run-real-npu-job.sh --help | grep -q "${CASE_NAME}"

grep -q -- "--prepare-runtime-artifacts" examples/transformer/run-mainline.sh

bash -n examples/transformer/run-mainline.sh
bash -n scripts/sync-and-submit.sh
bash -n scripts/real-npu-ci/docker-run.sh
bash -n scripts/real-npu-ci/run-real-npu-job.sh
