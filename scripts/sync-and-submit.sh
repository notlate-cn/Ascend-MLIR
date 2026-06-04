#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_NPU_CI_DIR="${SCRIPT_DIR}/real-npu-ci"

if [[ -f "${REAL_NPU_CI_DIR}/versions.env" ]]; then
  # shellcheck source=/dev/null
  source "${REAL_NPU_CI_DIR}/versions.env"
fi

REMOTE="${ASCEND_MLIR_CI_REMOTE:-}"
PORT="${ASCEND_MLIR_CI_REMOTE_PORT:-141}"
REMOTE_DIR="${ASCEND_MLIR_CI_REMOTE_DIR:-${ASCEND_MLIR_CI_REMOTE_SOURCE_DIR:-/data/nyh/Codex-Ascend-MLIR-current}}"
IMAGE="${ASCEND_MLIR_CI_IMAGE:-${ASCEND_MLIR_CI_DEFAULT_REMOTE_IMAGE:-${ASCEND_MLIR_CI_DEFAULT_LLVM_IMAGE:-ascend-mlir-builder:aarch64-ubuntu22.04-llvm21}}}"
CASE_NAME="${ASCEND_MLIR_CI_CASE:-relu-broadcast-transpose}"
CMD="${ASCEND_MLIR_CI_CMD:-}"
SKIP_SIM="${ASCEND_MLIR_CI_SKIP_SIM:-0}"
NPU_RUN_TIMEOUT_SECONDS="${ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS:-600}"
DEVICE_ID="${ASCEND_DEVICE_ID:-7}"
REF="${ASCEND_MLIR_CI_REF:-}"
JOB_ROOT="${ASCEND_MLIR_CI_JOB_ROOT:-}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-}"
CANN_HOME="${ASCEND_HOME_PATH:-}"
JOBS="${ASCEND_MLIR_CI_JOBS:-6}"
LIST_CASES=0
SSH_PASSWORD="${ASCEND_MLIR_CI_SSH_PASSWORD:-}"
INCREMENTAL="${ASCEND_MLIR_CI_INCREMENTAL_SOURCE:-1}"
USE_CCACHE="${ASCEND_MLIR_CI_USE_CCACHE:-1}"
CCACHE_DIR="${ASCEND_MLIR_CI_CCACHE_DIR:-}"
CLEAN="${ASCEND_MLIR_CI_CLEAN:-0}"

usage() {
  cat <<'EOF'
Usage: sync-and-submit.sh [OPTIONS]

Package the local git worktree, sync it to a real-NPU host directory, then run a
real-NPU validation job with an image that already exists on the real-NPU host.

Options:
  --remote USER@HOST        SSH target. Required unless ASCEND_MLIR_CI_REMOTE is set.
  --port PORT               SSH port. Default: 141
  --remote-dir DIR          Remote source directory to replace and run.
                            Default: /data/nyh/Codex-Ascend-MLIR-current
  --image IMAGE             Existing builder image tag on the real-NPU host.
                            Defaults to ASCEND_MLIR_CI_DEFAULT_REMOTE_IMAGE
                            from versions.env.
  --case NAME               Example case to run. NAME comes from
                            examples/<NAME>/run.sh. Special: microcases,
                            relu-broadcast-diagnostics,
                            real-npu-multikernel, transformer-real-npu, all.
                            all runs each case in a separate real-NPU container.
                            Default: relu-broadcast-transpose
  --cmd COMMAND             Custom command to run after build, from repo root.
                            Takes precedence over --case.
  --skip-sim                Prepare ordinary example artifacts, then run only
                            the real NPU phase. Diagnostic only; not a
                            readiness gate for candidate kernel fixes.
  --npu-timeout SECONDS     Per-manifest real NPU runtime-session timeout.
                            Default: ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS
                            or 600.
  --list-cases              Print local case names and exit.
  --device-id ID            NPU device id. Default: ASCEND_DEVICE_ID or 7
  --ref REF                 Label recorded in job output. Default: <HEAD>-local
                            when the worktree is dirty, otherwise <HEAD>.
  --job-root DIR            Job root on the real-NPU host/container.
  --llvm-build-dir DIR      LLVM build dir inside the container.
  --cann-home DIR           CANN toolkit root inside the container.
  --jobs N                  Build parallelism inside the container. Default: 6.
  --clean                   Remove remote build dirs and this worktree's ccache
                            before running.
  --no-incremental          Copy source into each job dir and build from scratch.
  --ccache-dir DIR          Host ccache directory. Default is derived from
                            --remote-dir under /data/nyh/ccache/.
  --no-ccache               Do not mount or use ccache.
  --help                    Show this help.

The sync step includes tracked files, populated submodule files, and unignored
untracked files from the local git worktree. Ignored artifacts such as build/,
out/, generated .npy files, and macOS metadata files are not synced.

For the shared real-NPU host, the remote source directory must be under /data/nyh.

SSH authentication uses normal ssh keys first. If ASCEND_MLIR_CI_SSH_PASSWORD
is set in the local environment, sshpass is used for password authentication.
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
    --remote-source-dir)
      echo "warning: --remote-source-dir is deprecated; use --remote-dir" >&2
      REMOTE_DIR="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
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
    --device-id)
      DEVICE_ID="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
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
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --no-incremental)
      INCREMENTAL=0
      shift
      ;;
    --ccache-dir)
      CCACHE_DIR="$2"
      shift 2
      ;;
    --no-ccache)
      USE_CCACHE=0
      shift
      ;;
    --list-cases)
      LIST_CASES=1
      shift
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

list_cases() {
  find examples -mindepth 2 -maxdepth 2 -name run.sh -print |
    sed 's#^examples/##; s#/run.sh$##' |
    sort
  if [[ -f examples/real-npu-microcases/prepare.sh ]]; then
    echo microcases
    echo relu-broadcast-diagnostics
  fi
  if [[ -f examples/real-npu-multikernel/run.sh ]]; then
    echo real-npu-multikernel
  fi
  if [[ -f examples/transformer/run-mainline.sh ]]; then
    echo transformer-real-npu
  fi
  echo all
}

if [[ "${LIST_CASES}" == "1" ]]; then
  if repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    cd "${repo_root}"
  fi
  list_cases | sort -u
  exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

if [[ -z "${REMOTE}" ]]; then
  echo "--remote is required, or set ASCEND_MLIR_CI_REMOTE" >&2
  usage >&2
  exit 2
fi

if ! [[ "${NPU_RUN_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--npu-timeout must be a positive integer: ${NPU_RUN_TIMEOUT_SECONDS}" >&2
  exit 2
fi

SSH_BASE=(ssh -p "${PORT}" -o StrictHostKeyChecking=accept-new)
if [[ -n "${SSH_PASSWORD}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "ssh password is configured, but sshpass is not installed" >&2
    echo "install sshpass, configure an ssh key, or unset ASCEND_MLIR_CI_SSH_PASSWORD for interactive ssh" >&2
    exit 1
  fi
  SSH_BASE=(
    sshpass -e ssh
    -p "${PORT}"
    -o PreferredAuthentications=password
    -o PubkeyAuthentication=no
    -o StrictHostKeyChecking=accept-new
  )
fi

run_ssh() {
  if [[ -n "${SSH_PASSWORD}" ]]; then
    SSHPASS="${SSH_PASSWORD}" "${SSH_BASE[@]}" "${REMOTE}" "$@"
  else
    "${SSH_BASE[@]}" "${REMOTE}" "$@"
  fi
}

if [[ -z "${REMOTE_DIR}" || "${REMOTE_DIR}" == "/" || "${REMOTE_DIR}" == "/data/nyh" ]]; then
  echo "unsafe --remote-dir: ${REMOTE_DIR}" >&2
  exit 2
fi

case "${REMOTE_DIR}" in
  /data/nyh/*) ;;
  *)
    echo "--remote-dir must be under /data/nyh for the shared real-NPU host: ${REMOTE_DIR}" >&2
    exit 2
    ;;
esac

if [[ -z "${CCACHE_DIR}" && "${USE_CCACHE}" == "1" ]]; then
  cache_name="${REMOTE_DIR#/data/nyh/}"
  cache_name="$(echo "${cache_name}" | tr '/:@ ' '____' | tr -cd '[:alnum:]_.-')"
  [[ -n "${cache_name}" ]] || cache_name="default"
  CCACHE_DIR="${ASCEND_MLIR_CI_DEFAULT_CCACHE_ROOT:-/data/nyh/ccache}/${cache_name}"
fi

if [[ -z "${REF}" ]]; then
  short_head="$(git rev-parse --short HEAD 2>/dev/null || echo local-tree)"
  REF="${short_head}"
  if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    REF="${short_head}-local"
  fi
fi

file_list="$(mktemp)"
trap 'rm -f "${file_list}"' EXIT

append_git_files() {
  local prefix="$1"
  while IFS= read -r -d '' file_path; do
    case "${file_path}" in
      .DS_Store|._*|*/.DS_Store|*/._*) continue ;;
    esac
    local repo_path="${prefix}${file_path}"
    if [[ -f "${repo_path}" || -L "${repo_path}" ]]; then
      printf '%s\0' "${repo_path}"
    fi
  done
  return 0
}

git ls-files -co --exclude-standard -z | append_git_files "" >"${file_list}"

if [[ -f .gitmodules ]]; then
  git config --file .gitmodules --get-regexp 'submodule\..*\.path' |
    awk '{print $2}' |
    while IFS= read -r submodule_path; do
      if [[ -d "${submodule_path}" ]]; then
        git -C "${submodule_path}" ls-files -co --exclude-standard -z |
          append_git_files "${submodule_path}/" >>"${file_list}"
      fi
    done
fi

if [[ ! -s "${file_list}" ]]; then
  echo "no files selected for sync" >&2
  exit 1
fi

quote() {
  printf '%q' "$1"
}

remote_source_q="$(quote "${REMOTE_DIR}")"
if [[ "${CLEAN}" == "1" || "${INCREMENTAL}" != "1" ]]; then
  remote_prep_cmd="set -euo pipefail; rm -rf -- ${remote_source_q}; mkdir -p -- ${remote_source_q}; tar --no-xattrs -C ${remote_source_q} -xzf -"
else
  remote_prep_cmd="set -euo pipefail; mkdir -p -- ${remote_source_q}; find ${remote_source_q} -mindepth 1 -maxdepth 1 \\( -path ${remote_source_q}/build -o -path ${remote_source_q}/build-runtime-session-run-only \\) -prune -o -exec rm -rf -- {} +; tar --no-xattrs -C ${remote_source_q} -xzf -"
fi
if [[ "${CLEAN}" == "1" && -n "${CCACHE_DIR}" ]]; then
  remote_ccache_q="$(quote "${CCACHE_DIR}")"
  remote_prep_cmd="set -euo pipefail; rm -rf -- ${remote_ccache_q}; ${remote_prep_cmd}"
fi

echo "sync local worktree -> ${REMOTE}:${REMOTE_DIR}"
COPYFILE_DISABLE=1 tar --no-xattrs -czf - --null -T "${file_list}" |
  run_ssh "${remote_prep_cmd}"

runner_args=(
  --image "${IMAGE}"
  --source-dir "${REMOTE_DIR}"
  --ref "${REF}"
  --case "${CASE_NAME}"
  --npu-timeout "${NPU_RUN_TIMEOUT_SECONDS}"
  --device-id "${DEVICE_ID}"
  --jobs "${JOBS}"
)
if [[ -n "${CMD}" ]]; then
  runner_args+=(--cmd "${CMD}")
fi
if [[ "${SKIP_SIM}" == "1" ]]; then
  runner_args+=(--skip-sim)
fi
if [[ "${INCREMENTAL}" == "1" ]]; then
  runner_args+=(--incremental-source)
else
  runner_args+=(--no-incremental-source)
fi
if [[ "${USE_CCACHE}" == "1" ]]; then
  runner_args+=(--ccache-dir "${CCACHE_DIR}")
else
  runner_args+=(--no-ccache)
fi
if [[ "${CLEAN}" == "1" ]]; then
  runner_args+=(--clean)
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

remote_run_cmd="cd ${remote_source_q} && exec scripts/real-npu-ci/docker-run.sh"
for arg in "${runner_args[@]}"; do
  remote_run_cmd+=" $(quote "${arg}")"
done

if [[ -n "${CMD}" ]]; then
  echo "run remote command '${CMD}' with image '${IMAGE}' on device ${DEVICE_ID}"
else
  echo "run real-NPU case '${CASE_NAME}' with image '${IMAGE}' on device ${DEVICE_ID}"
fi
run_ssh "${remote_run_cmd}"
