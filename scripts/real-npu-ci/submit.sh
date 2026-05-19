#!/usr/bin/env bash
set -euo pipefail

REMOTE="${ASCEND_MLIR_CI_REMOTE:-root@<real-npu-host>}"
PORT="${ASCEND_MLIR_CI_REMOTE_PORT:-141}"
REMOTE_DIR="${ASCEND_MLIR_CI_REMOTE_DIR:-/data/nyh/Codex-Ascend-MLIR}"
REPO_URL="${ASCEND_MLIR_CI_REPO_URL:-}"
REF="${ASCEND_MLIR_CI_REF:-HEAD}"
CASE_NAME="${ASCEND_MLIR_CI_CASE:-relu-broadcast-transpose}"
CMD="${ASCEND_MLIR_CI_CMD:-}"
DEVICE_ID="${ASCEND_DEVICE_ID:-7}"
IMAGE="${ASCEND_MLIR_CI_IMAGE:-}"
JOB_ROOT="${ASCEND_MLIR_CI_JOB_ROOT:-}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-}"
CANN_HOME="${ASCEND_HOME_PATH:-}"
REMOTE_SOURCE_DIR="${ASCEND_MLIR_CI_REMOTE_SOURCE_DIR:-}"
JOBS="${ASCEND_MLIR_CI_JOBS:-6}"

usage() {
  cat <<'EOF'
Usage: submit.sh [OPTIONS]

Submit a real-NPU validation job to the shared real-NPU host over SSH.

Options:
  --remote USER@HOST        SSH target. Default: root@<real-npu-host>
  --port PORT               SSH port. Default: 141
  --remote-dir DIR          Repo path on the real-NPU host. Default: /data/nyh/Codex-Ascend-MLIR
  --repo-url URL            Git repository URL for the container to clone.
  --ref REF                 Git ref, branch, tag, or commit to test. Default: HEAD
  --case NAME               Example case to run. Default: relu-broadcast-transpose
  --cmd COMMAND             Custom command to run after build, from repo root.
                            Takes precedence over --case.
  --device-id ID            NPU device id. Default: ASCEND_DEVICE_ID or 7
  --image IMAGE             Builder image tag on the real-NPU host.
  --job-root DIR            Job root on the real-NPU host/container.
  --llvm-build-dir DIR      LLVM build dir inside the container.
  --cann-home DIR           CANN toolkit root inside the container.
  --remote-source-dir DIR   Use a source tree already present on the real-NPU host.
  --jobs N                  Build parallelism inside the container. Default: 6.
  --help                    Show this help.

Pass either --repo-url or --remote-source-dir. For x86 developer machines,
prefer --repo-url plus --ref so the build happens entirely on the real-NPU host.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
      shift 2
      ;;
    --case)
      CASE_NAME="$2"
      shift 2
      ;;
    --cmd)
      CMD="$2"
      shift 2
      ;;
    --device-id)
      DEVICE_ID="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --job-root)
      JOB_ROOT="$2"
      shift 2
      ;;
    --llvm-build-dir)
      LLVM_BUILD_DIR="$2"
      shift 2
      ;;
    --cann-home)
      CANN_HOME="$2"
      shift 2
      ;;
    --remote-source-dir)
      REMOTE_SOURCE_DIR="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${REPO_URL}" && -z "${REMOTE_SOURCE_DIR}" ]]; then
  echo "one of --repo-url or --remote-source-dir is required" >&2
  usage >&2
  exit 2
fi

runner_args=()
if [[ -n "${REPO_URL}" ]]; then
  runner_args+=(--repo-url "${REPO_URL}")
fi
if [[ -n "${REMOTE_SOURCE_DIR}" ]]; then
  runner_args+=(--source-dir "${REMOTE_SOURCE_DIR}")
fi
runner_args+=(--ref "${REF}")
runner_args+=(--case "${CASE_NAME}")
if [[ -n "${CMD}" ]]; then
  runner_args+=(--cmd "${CMD}")
fi
runner_args+=(--device-id "${DEVICE_ID}")
runner_args+=(--jobs "${JOBS}")
if [[ -n "${IMAGE}" ]]; then
  runner_args+=(--image "${IMAGE}")
fi
if [[ -n "${JOB_ROOT}" ]]; then
  runner_args+=(--job-root "${JOB_ROOT}")
fi
if [[ -n "${LLVM_BUILD_DIR}" ]]; then
  runner_args+=(--llvm-build-dir "${LLVM_BUILD_DIR}")
fi
if [[ -n "${CANN_HOME}" ]]; then
  runner_args+=(--cann-home "${CANN_HOME}")
fi

quote() {
  printf '%q' "$1"
}

remote_cmd="cd $(quote "${REMOTE_DIR}") && exec scripts/real-npu-ci/docker-run.sh"
for arg in "${runner_args[@]}"; do
  remote_cmd+=" $(quote "${arg}")"
done

exec ssh -p "${PORT}" "${REMOTE}" "${remote_cmd}"
