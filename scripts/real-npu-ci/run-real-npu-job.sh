#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${ASCEND_MLIR_CI_REPO_URL:-}"
REF="${ASCEND_MLIR_CI_REF:-HEAD}"
CASE_NAME="${ASCEND_MLIR_CI_CASE:-relu-broadcast-transpose}"
JOB_ROOT="${ASCEND_MLIR_CI_JOB_ROOT:-/data/nyh/real-npu-jobs}"
SOURCE_DIR="${ASCEND_MLIR_CI_SOURCE_DIR:-}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/opt/llvm/build}"
BUILD_LLVM="${ASCEND_MLIR_CI_BUILD_LLVM:-0}"
JOBS="${ASCEND_MLIR_CI_JOBS:-6}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: run-real-npu-job.sh

This script is normally run as the container ENTRYPOINT. Configure it with:
  ASCEND_MLIR_CI_REPO_URL    Git repository URL to clone, unless SOURCE_DIR is set.
  ASCEND_MLIR_CI_REF         Git ref, branch, tag, or commit. Default: HEAD.
  ASCEND_MLIR_CI_CASE        Example case name. Default: relu-broadcast-transpose.
  ASCEND_MLIR_CI_JOB_ROOT    Output root. Default: /data/nyh/real-npu-jobs.
  ASCEND_MLIR_CI_SOURCE_DIR  Optional mounted source tree.
  ASCEND_MLIR_CI_JOBS        Build parallelism. Default: 6.
  LLVM_BUILD_DIR             Required LLVM build dir, unless ASCEND_MLIR_CI_BUILD_LLVM=1.
  ASCEND_HOME_PATH           CANN toolkit root.
  ASCEND_DEVICE_ID           NPU device id.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
safe_ref="$(echo "${REF}" | tr '/:@ ' '____' | tr -cd '[:alnum:]_.-')"
safe_case="$(echo "${CASE_NAME}" | tr '/:@ ' '____' | tr -cd '[:alnum:]_.-')"
JOB_DIR="${JOB_ROOT}/${timestamp}-${safe_ref}-${safe_case}"
SRC_DIR="${JOB_DIR}/src"
LOG_DIR="${JOB_DIR}/logs"
OUT_DIR="${JOB_DIR}/out"
RUN_ONLY_BUILD_DIR="${SRC_DIR}/build-runtime-session-run-only"
RUN_RUNTIME_SESSION="${RUN_ONLY_BUILD_DIR}/bin/runtime-session"
mkdir -p "${SRC_DIR}" "${LOG_DIR}" "${OUT_DIR}"

log() {
  echo "[$(date -Is)] $*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

run_logged() {
  local name="$1"
  shift
  log "run ${name}: $*"
  "$@" >"${LOG_DIR}/${name}.log" 2>&1
}

source_if_exists() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    # shellcheck source=/dev/null
    set +u
    source "${file}"
    set -u
  fi
}

record_job_env() {
  {
    echo "timestamp=${timestamp}"
    echo "repo_url=${REPO_URL}"
    echo "ref=${REF}"
    echo "case=${CASE_NAME}"
    echo "job_dir=${JOB_DIR}"
    echo "source_dir=${SOURCE_DIR}"
    echo "llvm_build_dir=${LLVM_BUILD_DIR}"
    echo "jobs=${JOBS}"
    echo "ascend_home_path=${ASCEND_HOME_PATH:-}"
    echo "ascend_device_id=${ASCEND_DEVICE_ID:-}"
    uname -a || true
    command -v npu-smi >/dev/null 2>&1 && npu-smi info || true
  } >"${JOB_DIR}/job-env.txt" 2>&1
}

cleanup_collect() {
  "${SCRIPT_DIR}/collect-plog.sh" --out-dir "${LOG_DIR}/plog" || true
}
trap cleanup_collect EXIT

record_job_env

source_if_exists /usr/local/Ascend/driver/bin/setenv.bash
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  source_if_exists "${ASCEND_HOME_PATH}/set_env.sh"
fi

if [[ -n "${SOURCE_DIR}" ]]; then
  log "copy source from mounted tree: ${SOURCE_DIR}"
  rsync -a --delete \
    --exclude build \
    --exclude .git \
    --exclude out \
    --exclude .DS_Store \
    --exclude '._*' \
    "${SOURCE_DIR}/" "${SRC_DIR}/"
  if [[ -d "${SOURCE_DIR}/.git" ]]; then
    git -C "${SOURCE_DIR}" rev-parse HEAD >"${JOB_DIR}/source-head.txt" 2>/dev/null || true
  fi
else
  [[ -n "${REPO_URL}" ]] || fail "ASCEND_MLIR_CI_REPO_URL is required when ASCEND_MLIR_CI_SOURCE_DIR is not set"
  log "clone ${REPO_URL}"
  git clone --recursive "${REPO_URL}" "${SRC_DIR}" >"${LOG_DIR}/git-clone.log" 2>&1
  git -C "${SRC_DIR}" checkout "${REF}" >"${LOG_DIR}/git-checkout.log" 2>&1
  git -C "${SRC_DIR}" submodule update --init --recursive >"${LOG_DIR}/git-submodule.log" 2>&1
fi

git -C "${SRC_DIR}" rev-parse HEAD >"${JOB_DIR}/commit.txt" 2>/dev/null || true
git -C "${SRC_DIR}" status --short >"${JOB_DIR}/source-status.txt" 2>/dev/null || true

if [[ ! -d "${LLVM_BUILD_DIR}/lib/cmake/mlir" ]]; then
  if [[ "${BUILD_LLVM}" == "1" ]]; then
    log "LLVM_BUILD_DIR is missing; building LLVM because ASCEND_MLIR_CI_BUILD_LLVM=1"
    run_logged build-llvm bash "${SRC_DIR}/scripts/build_llvm.sh"
    LLVM_BUILD_DIR="${SRC_DIR}/externals/llvm-project/build"
  else
    fail "LLVM_BUILD_DIR is not usable: ${LLVM_BUILD_DIR}. Bake LLVM into the image, mount it, or set ASCEND_MLIR_CI_BUILD_LLVM=1."
  fi
fi

export LLVM_BUILD_DIR
export ASCEND_RUNTIME_TRACE_LAUNCH="${ASCEND_RUNTIME_TRACE_LAUNCH:-1}"
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-7}"

log "build project"
(
  cd "${SRC_DIR}"
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    source_if_exists "${ASCEND_HOME_PATH}/set_env.sh"
  fi
  BUILD_DIR="${SRC_DIR}/build" \
  LLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
  NUM_JOBS="${JOBS}" \
  bash scripts/build.sh --build-project --jobs "${JOBS}"
) >"${LOG_DIR}/build-project.log" 2>&1

log "build run-only runtime-session"
(
  cd "${SRC_DIR}"
  cmake -G Ninja -S "${SRC_DIR}" -B "${RUN_ONLY_BUILD_DIR}" \
    -DLLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
    -DASCEND_RUNTIME_SESSION_RUN_ONLY=ON
  cmake --build "${RUN_ONLY_BUILD_DIR}" --target runtime-session -j"${JOBS}"
) >"${LOG_DIR}/build-run-only.log" 2>&1

run_real_manifest() {
  local source_manifest="$1"
  local real_manifest="$2"
  local real_output="$3"
  python3 - "${source_manifest}" "${real_manifest}" "${real_output}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
out = sys.argv[3]
data = json.loads(src.read_text())
data["backend"] = "npu"
for binding in data.get("outputs", []):
    binding["path"] = out
dst.write_text(json.dumps(data, indent=2) + "\n")
PY
  "${RUN_RUNTIME_SESSION}" --run-manifest "${real_manifest}" --run
}

run_example_case() {
  local case_name="$1"
  local case_dir="${SRC_DIR}/examples/${case_name}"
  [[ -f "${case_dir}/run.sh" ]] || fail "example run.sh not found: ${case_dir}/run.sh"

  log "run xvm-style sim pipeline for ${case_name}"
  (
    cd "${SRC_DIR}"
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
      source_if_exists "${ASCEND_HOME_PATH}/set_env.sh"
    fi
    # shellcheck source=/dev/null
    source examples/env.sh
    export PATH="${SRC_DIR}/build/bin:${PATH}"
    export AFIR_OPT="${SRC_DIR}/build/bin/afir-opt"
    export AFIR_TRANSLATE="${SRC_DIR}/build/bin/afir-translate"
    export RUNTIME_SESSION="${SRC_DIR}/build/bin/runtime-session"
    bash "examples/${case_name}/run.sh" --log
  ) >"${LOG_DIR}/${case_name}-sim.log" 2>&1

  local manifest="${case_dir}/build_e2e/run_manifest.json"
  [[ -f "${manifest}" ]] || fail "example did not produce run manifest: ${manifest}"

  log "run real NPU for ${case_name}"
  mkdir -p "${OUT_DIR}/${case_name}"
  run_real_manifest \
    "${manifest}" \
    "${OUT_DIR}/${case_name}/run_manifest.npu.json" \
    "${OUT_DIR}/${case_name}/output.npy" \
    >"${LOG_DIR}/${case_name}-npu.log" 2>&1
}

run_microcases() {
  local micro_out="${OUT_DIR}/microcases"
  mkdir -p "${micro_out}"
  log "prepare real-npu microcases"
  (
    cd "${SRC_DIR}"
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
      source_if_exists "${ASCEND_HOME_PATH}/set_env.sh"
    fi
    # shellcheck source=/dev/null
    source examples/env.sh
    export RUNTIME_SESSION="${SRC_DIR}/build/bin/runtime-session"
    bash examples/real-npu-microcases/prepare.sh --out-dir "${micro_out}"
  ) >"${LOG_DIR}/microcases-prepare.log" 2>&1

  for manifest in "${micro_out}"/*/run_manifest.json; do
    [[ -f "${manifest}" ]] || continue
    local case_dir
    case_dir="$(dirname "${manifest}")"
    local case_name
    case_name="$(basename "${case_dir}")"
    mkdir -p "${OUT_DIR}/microcases-real/${case_name}"
    log "run real NPU microcase ${case_name}"
    run_real_manifest \
      "${manifest}" \
      "${OUT_DIR}/microcases-real/${case_name}/run_manifest.npu.json" \
      "${OUT_DIR}/microcases-real/${case_name}/output.npy" \
      >"${LOG_DIR}/microcase-${case_name}-npu.log" 2>&1
  done
}

case "${CASE_NAME}" in
  microcases)
    run_microcases
    ;;
  all)
    fail "case=all is intentionally not enabled yet; run named examples until all real-NPU cases are closed"
    ;;
  *)
    run_example_case "${CASE_NAME}"
    ;;
esac

log "job complete: ${JOB_DIR}"
echo "${JOB_DIR}" >"${JOB_ROOT}/latest-job.txt"
