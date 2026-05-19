#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ASCEND_MLIR_CI_IMAGE:-ascend-mlir-builder:aarch64-ubuntu22.04}"
REPO_URL="${ASCEND_MLIR_CI_REPO_URL:-}"
REF="${ASCEND_MLIR_CI_REF:-HEAD}"
CASE_NAME="${ASCEND_MLIR_CI_CASE:-relu-broadcast-transpose}"
JOB_ROOT="${ASCEND_MLIR_CI_JOB_ROOT:-/data/nyh/real-npu-jobs}"
DEVICE_ID="${ASCEND_DEVICE_ID:-7}"
SOURCE_DIR="${ASCEND_MLIR_CI_SOURCE_DIR:-}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/opt/llvm/build}"
CANN_HOME="${ASCEND_HOME_PATH:-/data/nyh/Ascend/latest}"
JOBS="${ASCEND_MLIR_CI_JOBS:-6}"
EXTRA_DOCKER_ARGS=()

usage() {
  cat <<'EOF'
Usage: docker-run-910c.sh [OPTIONS]

Run an Ascend-MLIR real-NPU job inside the 910C host container.

Options:
  --image IMAGE          Builder image tag. Default: ascend-mlir-builder:aarch64-ubuntu22.04
  --repo-url URL         Git repository URL to clone inside the container.
  --ref REF              Git ref, branch, tag, or commit to test. Default: HEAD
  --case NAME            Example case to run. Default: relu-broadcast-transpose
  --job-root DIR         Host/container job root. Default: /data/nyh/real-npu-jobs
  --device-id ID         NPU device id. Default: ASCEND_DEVICE_ID or 7
  --source-dir DIR       Use a mounted local source tree instead of cloning.
  --llvm-build-dir DIR   LLVM build dir inside the container. Default: /opt/llvm/build
  --cann-home DIR        CANN toolkit root inside the container. Default: /data/nyh/Ascend/latest
  --jobs N               Build parallelism inside the container. Default: 6
  --docker-arg ARG       Extra argument passed to docker run. May be repeated.
  --help                 Show this help.

Either --repo-url or --source-dir is required.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
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
    --job-root)
      JOB_ROOT="$2"
      shift 2
      ;;
    --device-id)
      DEVICE_ID="$2"
      shift 2
      ;;
    --source-dir)
      SOURCE_DIR="$2"
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
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --docker-arg)
      EXTRA_DOCKER_ARGS+=("$2")
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

if [[ -z "${REPO_URL}" && -z "${SOURCE_DIR}" ]]; then
  echo "one of --repo-url or --source-dir is required" >&2
  usage >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on the 910C host" >&2
  exit 1
fi

if [[ ! -e "/dev/davinci${DEVICE_ID}" ]]; then
  echo "NPU device node not found: /dev/davinci${DEVICE_ID}" >&2
  echo "check ASCEND_DEVICE_ID, host driver state, and npu-smi info" >&2
  exit 1
fi

if [[ ! -d "${CANN_HOME}" ]]; then
  echo "CANN toolkit root not found: ${CANN_HOME}" >&2
  exit 1
fi

mkdir -p "${JOB_ROOT}"

DOCKER_MOUNTS=(
  -v "${JOB_ROOT}:${JOB_ROOT}"
  -v /data/nyh:/data/nyh
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
)

if [[ -n "${SOURCE_DIR}" ]]; then
  SOURCE_DIR="$(cd "${SOURCE_DIR}" && pwd)"
  DOCKER_MOUNTS+=(-v "${SOURCE_DIR}:/workspace/source:ro")
fi

if [[ -e /var/log/npu ]]; then
  DOCKER_MOUNTS+=(-v /var/log/npu:/var/log/npu)
fi
if [[ -e /usr/local/bin/npu-smi ]]; then
  DOCKER_MOUNTS+=(-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro)
fi
if [[ -e /etc/ascend_install.info ]]; then
  DOCKER_MOUNTS+=(-v /etc/ascend_install.info:/etc/ascend_install.info:ro)
fi

DOCKER_DEVICES=(
  --device="/dev/davinci${DEVICE_ID}:/dev/davinci${DEVICE_ID}"
)

for dev in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  if [[ -e "${dev}" ]]; then
    DOCKER_DEVICES+=(--device="${dev}:${dev}")
  fi
done

exec docker run --rm \
  --privileged \
  --network host \
  --ipc host \
  "${DOCKER_DEVICES[@]}" \
  "${DOCKER_MOUNTS[@]}" \
  "${EXTRA_DOCKER_ARGS[@]}" \
  -e ASCEND_DEVICE_ID="${DEVICE_ID}" \
  -e ASCEND_HOME_PATH="${CANN_HOME}" \
  -e ASCEND_TOOLKIT_HOME="${CANN_HOME}" \
  -e LLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
  -e ASCEND_MLIR_CI_JOBS="${JOBS}" \
  -e ASCEND_MLIR_CI_REPO_URL="${REPO_URL}" \
  -e ASCEND_MLIR_CI_REF="${REF}" \
  -e ASCEND_MLIR_CI_CASE="${CASE_NAME}" \
  -e ASCEND_MLIR_CI_JOB_ROOT="${JOB_ROOT}" \
  -e ASCEND_MLIR_CI_SOURCE_DIR="${SOURCE_DIR:+/workspace/source}" \
  "${IMAGE}"
