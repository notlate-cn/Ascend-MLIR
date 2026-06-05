#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/versions.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/versions.env"
fi

IMAGE="${ASCEND_MLIR_CI_IMAGE:-${ASCEND_MLIR_CI_DEFAULT_REMOTE_IMAGE:-ascend-mlir-builder:aarch64-ubuntu22.04}}"
REPO_URL="${ASCEND_MLIR_CI_REPO_URL:-}"
REF="${ASCEND_MLIR_CI_REF:-HEAD}"
CASE_NAME="${ASCEND_MLIR_CI_CASE:-relu-broadcast-transpose}"
CMD="${ASCEND_MLIR_CI_CMD:-}"
SKIP_SIM="${ASCEND_MLIR_CI_SKIP_SIM:-0}"
NPU_RUN_TIMEOUT_SECONDS="${ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS:-600}"
JOB_ROOT="${ASCEND_MLIR_CI_JOB_ROOT:-/data/nyh/real-npu-jobs}"
DEVICE_ID="${ASCEND_DEVICE_ID:-7}"
SOURCE_DIR="${ASCEND_MLIR_CI_SOURCE_DIR:-}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/opt/llvm/build}"
CANN_HOME="${ASCEND_HOME_PATH:-/data/nyh/Ascend/latest}"
JOBS="${ASCEND_MLIR_CI_JOBS:-6}"
INCREMENTAL_SOURCE="${ASCEND_MLIR_CI_INCREMENTAL_SOURCE:-0}"
USE_CCACHE="${ASCEND_MLIR_CI_USE_CCACHE:-1}"
CCACHE_DIR_HOST="${ASCEND_MLIR_CI_CCACHE_DIR:-}"
CLEAN="${ASCEND_MLIR_CI_CLEAN:-0}"
BUILD_PROFILE="${ASCEND_MLIR_CI_BUILD_PROFILE:-ascend}"
EXTRA_DOCKER_ARGS=()
ENTRYPOINT_ARGS=()
REAL_NPU_ALL_CASES=(
  add-broadcast-concat
  broadcast-add-reduce
  gather-elementwise-fusion
  matmul-add-leakyrelu
  relu-broadcast-transpose
  split-relu-brc-add-mul
  real-npu-multikernel
)

usage() {
  cat <<'EOF'
Usage: docker-run.sh [OPTIONS]

Run an Ascend-MLIR real-NPU job inside the real-NPU host container.

Options:
  --image IMAGE          Builder image tag. Defaults to ASCEND_MLIR_CI_DEFAULT_REMOTE_IMAGE
                         from versions.env, or ascend-mlir-builder:aarch64-ubuntu22.04.
  --repo-url URL         Git repository URL to clone inside the container.
  --ref REF              Git ref, branch, tag, or commit to test. Default: HEAD
  --case NAME            Example case to run. Special cases include microcases,
                         relu-broadcast-diagnostics, real-npu-multikernel,
                         transformer-real-npu, and all.
                         all runs each suite case in a separate container.
                         Default: relu-broadcast-transpose
  --cmd COMMAND          Custom command to run after build, from repo root.
                         Takes precedence over --case.
  --skip-sim             Prepare ordinary example artifacts, then run only the
                         real NPU phase. Diagnostic only; not a readiness gate.
  --npu-timeout SECONDS  Per-manifest real NPU runtime-session timeout.
                         Default: ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS or 600.
  --job-root DIR         Host/container job root. Default: /data/nyh/real-npu-jobs
  --device-id ID         NPU device id. Default: ASCEND_DEVICE_ID or 7
  --source-dir DIR       Use a mounted local source tree instead of cloning.
  --incremental-source   Build directly in --source-dir so its build dirs are reused.
  --no-incremental-source
                         Copy --source-dir into the job directory before building.
  --llvm-build-dir DIR   LLVM build dir inside the container. Default: /opt/llvm/build
  --cann-home DIR        CANN toolkit root inside the container. Default: /data/nyh/Ascend/latest
  --jobs N               Build parallelism inside the container. Default: 6
  --ccache-dir DIR       Host ccache directory to mount at /ccache.
  --no-ccache            Do not mount or use ccache.
  --build-profile NAME   Project build profile inside the container: ascend or full.
                         Default: ascend.
  --clean                Remove build dirs before building.
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
    --cmd)
      CMD="$2"
      shift 2
      ;;
    --skip-sim)
      SKIP_SIM=1
      shift
      ;;
    --npu-timeout)
      NPU_RUN_TIMEOUT_SECONDS="$2"
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
    --incremental-source)
      INCREMENTAL_SOURCE=1
      shift
      ;;
    --no-incremental-source)
      INCREMENTAL_SOURCE=0
      shift
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
    --ccache-dir)
      CCACHE_DIR_HOST="$2"
      shift 2
      ;;
    --no-ccache)
      USE_CCACHE=0
      shift
      ;;
    --build-profile)
      BUILD_PROFILE="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
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

case "${BUILD_PROFILE}" in
  ascend|full) ;;
  *)
    echo "--build-profile must be 'ascend' or 'full': ${BUILD_PROFILE}" >&2
    exit 2
    ;;
esac

if [[ -z "${REPO_URL}" && -z "${SOURCE_DIR}" ]]; then
  echo "one of --repo-url or --source-dir is required" >&2
  usage >&2
  exit 2
fi

if ! [[ "${NPU_RUN_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--npu-timeout must be a positive integer: ${NPU_RUN_TIMEOUT_SECONDS}" >&2
  exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required on the real-NPU host" >&2
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
  if [[ "${INCREMENTAL_SOURCE}" == "1" ]]; then
    DOCKER_MOUNTS+=(-v "${SOURCE_DIR}:/workspace/source")
  else
    DOCKER_MOUNTS+=(-v "${SOURCE_DIR}:/workspace/source:ro")
  fi
  if [[ -f "${SOURCE_DIR}/scripts/real-npu-ci/run-real-npu-job.sh" ]]; then
    ENTRYPOINT_ARGS=(--entrypoint /workspace/source/scripts/real-npu-ci/run-real-npu-job.sh)
  fi
fi

if [[ "${USE_CCACHE}" == "1" ]]; then
  if [[ -z "${CCACHE_DIR_HOST}" ]]; then
    safe_cache_name="$(basename "${SOURCE_DIR:-${REF}}")"
    safe_cache_name="$(echo "${safe_cache_name}" | tr '/:@ ' '____' | tr -cd '[:alnum:]_.-')"
    [[ -n "${safe_cache_name}" ]] || safe_cache_name="default"
    CCACHE_DIR_HOST="${ASCEND_MLIR_CI_DEFAULT_CCACHE_ROOT:-/data/nyh/ccache}/${safe_cache_name}"
  fi
  mkdir -p "${CCACHE_DIR_HOST}"
  DOCKER_MOUNTS+=(-v "${CCACHE_DIR_HOST}:/ccache")
fi

if [[ "${CASE_NAME}" == "all" && -z "${CMD}" ]]; then
  first_case=1
  for suite_case in "${REAL_NPU_ALL_CASES[@]}"; do
    echo "=== real-NPU suite case: ${suite_case} ==="
    suite_args=(
      --image "${IMAGE}"
      --ref "${REF}"
      --case "${suite_case}"
      --npu-timeout "${NPU_RUN_TIMEOUT_SECONDS}"
      --job-root "${JOB_ROOT}"
      --device-id "${DEVICE_ID}"
      --llvm-build-dir "${LLVM_BUILD_DIR}"
      --cann-home "${CANN_HOME}"
      --jobs "${JOBS}"
      --build-profile "${BUILD_PROFILE}"
    )
    if [[ -n "${REPO_URL}" ]]; then
      suite_args+=(--repo-url "${REPO_URL}")
    fi
    if [[ -n "${SOURCE_DIR}" ]]; then
      suite_args+=(--source-dir "${SOURCE_DIR}")
    fi
    if [[ "${INCREMENTAL_SOURCE}" == "1" ]]; then
      suite_args+=(--incremental-source)
    else
      suite_args+=(--no-incremental-source)
    fi
    if [[ "${USE_CCACHE}" == "1" ]]; then
      suite_args+=(--ccache-dir "${CCACHE_DIR_HOST}")
    else
      suite_args+=(--no-ccache)
    fi
    if [[ "${CLEAN}" == "1" && "${first_case}" == "1" ]]; then
      suite_args+=(--clean)
    fi
    if [[ "${SKIP_SIM}" == "1" ]]; then
      suite_args+=(--skip-sim)
    fi
    for extra_arg in "${EXTRA_DOCKER_ARGS[@]}"; do
      suite_args+=(--docker-arg "${extra_arg}")
    done
    "${BASH_SOURCE[0]}" "${suite_args[@]}"
    first_case=0
  done
  exit 0
fi

DOCKER_ENV=(
  -e ASCEND_DEVICE_ID="${DEVICE_ID}"
  -e ASCEND_HOME_PATH="${CANN_HOME}"
  -e ASCEND_TOOLKIT_HOME="${CANN_HOME}"
  -e LLVM_BUILD_DIR="${LLVM_BUILD_DIR}"
  -e ASCEND_MLIR_CI_JOBS="${JOBS}"
  -e ASCEND_MLIR_CI_REPO_URL="${REPO_URL}"
  -e ASCEND_MLIR_CI_REF="${REF}"
  -e ASCEND_MLIR_CI_CASE="${CASE_NAME}"
  -e ASCEND_MLIR_CI_CMD="${CMD}"
  -e ASCEND_MLIR_CI_SKIP_SIM="${SKIP_SIM}"
  -e ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS="${NPU_RUN_TIMEOUT_SECONDS}"
  -e ASCEND_MLIR_CI_JOB_ROOT="${JOB_ROOT}"
  -e ASCEND_MLIR_CI_SOURCE_DIR="${SOURCE_DIR:+/workspace/source}"
  -e ASCEND_MLIR_CI_INCREMENTAL_SOURCE="${INCREMENTAL_SOURCE}"
  -e ASCEND_MLIR_CI_USE_CCACHE="${USE_CCACHE}"
  -e ASCEND_MLIR_CI_CLEAN="${CLEAN}"
  -e ASCEND_MLIR_CI_BUILD_PROFILE="${BUILD_PROFILE}"
)

if [[ "${USE_CCACHE}" == "1" ]]; then
  DOCKER_ENV+=(
    -e CCACHE_DIR=/ccache
    -e CCACHE_BASEDIR="${SOURCE_DIR:+/workspace/source}"
    -e CCACHE_COMPILERCHECK=content
  )
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
  "${ENTRYPOINT_ARGS[@]}" \
  "${EXTRA_DOCKER_ARGS[@]}" \
  "${DOCKER_ENV[@]}" \
  "${IMAGE}"
