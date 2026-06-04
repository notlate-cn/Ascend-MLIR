#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${ASCEND_MLIR_CI_REPO_URL:-}"
REF="${ASCEND_MLIR_CI_REF:-HEAD}"
CASE_NAME="${ASCEND_MLIR_CI_CASE:-relu-broadcast-transpose}"
CMD="${ASCEND_MLIR_CI_CMD:-}"
SKIP_SIM="${ASCEND_MLIR_CI_SKIP_SIM:-0}"
NPU_RUN_TIMEOUT_SECONDS="${ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS:-600}"
JOB_ROOT="${ASCEND_MLIR_CI_JOB_ROOT:-/data/nyh/real-npu-jobs}"
SOURCE_DIR="${ASCEND_MLIR_CI_SOURCE_DIR:-}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-/opt/llvm/build}"
BUILD_LLVM="${ASCEND_MLIR_CI_BUILD_LLVM:-0}"
JOBS="${ASCEND_MLIR_CI_JOBS:-6}"
INCREMENTAL_SOURCE="${ASCEND_MLIR_CI_INCREMENTAL_SOURCE:-0}"
USE_CCACHE="${ASCEND_MLIR_CI_USE_CCACHE:-1}"
CLEAN="${ASCEND_MLIR_CI_CLEAN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: run-real-npu-job.sh

This script is normally run as the container ENTRYPOINT. Configure it with:
  ASCEND_MLIR_CI_REPO_URL    Git repository URL to clone, unless SOURCE_DIR is set.
  ASCEND_MLIR_CI_REF         Git ref, branch, tag, or commit. Default: HEAD.
  ASCEND_MLIR_CI_CASE        Example case name. Use docker-run.sh --case all
                              for the real-NPU suite. Special cases include
                              microcases, real-npu-multikernel, and
                              transformer-real-npu.
                              Default: relu-broadcast-transpose.
  ASCEND_MLIR_CI_CMD         Optional custom command to run after build. When set,
                              it takes precedence over ASCEND_MLIR_CI_CASE.
  ASCEND_MLIR_CI_SKIP_SIM    Set to 1 to skip ordinary example sim execution and
                              run the real NPU phase only. Diagnostic only; not a
                              readiness gate for candidate kernel fixes.
  ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS
                              Per-manifest real NPU runtime-session timeout in
                              seconds. Default: 600.
  ASCEND_MLIR_CI_JOB_ROOT    Output root. Default: /data/nyh/real-npu-jobs.
  ASCEND_MLIR_CI_SOURCE_DIR  Optional mounted source tree.
  ASCEND_MLIR_CI_INCREMENTAL_SOURCE
                              Build directly in SOURCE_DIR when set to 1.
  ASCEND_MLIR_CI_JOBS        Build parallelism. Default: 6.
  ASCEND_MLIR_CI_USE_CCACHE   Use ccache when available. Default: 1.
  ASCEND_MLIR_CI_CLEAN        Remove local build dirs before building. Default: 0.
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
if [[ -n "${CMD}" ]]; then
  safe_case="cmd"
else
  safe_case="$(echo "${CASE_NAME}" | tr '/:@ ' '____' | tr -cd '[:alnum:]_.-')"
fi
JOB_DIR="${JOB_ROOT}/${timestamp}-${safe_ref}-${safe_case}"
JOB_SRC_DIR="${JOB_DIR}/src"
LOG_DIR="${JOB_DIR}/logs"
OUT_DIR="${JOB_DIR}/out"
if [[ -n "${SOURCE_DIR}" && "${INCREMENTAL_SOURCE}" == "1" ]]; then
  SRC_DIR="${SOURCE_DIR}"
else
  SRC_DIR="${JOB_SRC_DIR}"
fi
RUN_ONLY_BUILD_DIR="${SRC_DIR}/build-runtime-session-run-only"
RUN_RUNTIME_SESSION="${RUN_ONLY_BUILD_DIR}/bin/runtime-session"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"
if [[ "${SRC_DIR}" == "${JOB_SRC_DIR}" ]]; then
  mkdir -p "${SRC_DIR}"
fi

log() {
  echo "[$(date -Is)] $*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

if ! [[ "${NPU_RUN_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  fail "ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS must be a positive integer: ${NPU_RUN_TIMEOUT_SECONDS}"
fi
command -v timeout >/dev/null 2>&1 || fail "timeout command is required for real NPU run guarding"

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
    echo "cmd=${CMD}"
    echo "skip_sim=${SKIP_SIM}"
    echo "npu_run_timeout_seconds=${NPU_RUN_TIMEOUT_SECONDS}"
    echo "job_dir=${JOB_DIR}"
    echo "source_dir=${SOURCE_DIR}"
    echo "src_dir=${SRC_DIR}"
    echo "llvm_build_dir=${LLVM_BUILD_DIR}"
    echo "jobs=${JOBS}"
    echo "incremental_source=${INCREMENTAL_SOURCE}"
    echo "use_ccache=${USE_CCACHE}"
    echo "ccache_dir=${CCACHE_DIR:-}"
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

if [[ -n "${SOURCE_DIR}" && "${INCREMENTAL_SOURCE}" == "1" ]]; then
  log "use mounted source incrementally: ${SOURCE_DIR}"
  [[ -w "${SRC_DIR}" ]] || fail "incremental source dir is not writable: ${SRC_DIR}"
elif [[ -n "${SOURCE_DIR}" ]]; then
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

if [[ "${CLEAN}" == "1" ]]; then
  log "clean build directories"
  rm -rf "${SRC_DIR}/build" "${RUN_ONLY_BUILD_DIR}"
fi

cmake_launcher_args=()
if [[ "${USE_CCACHE}" == "1" ]] && command -v ccache >/dev/null 2>&1; then
  export CCACHE_DIR="${CCACHE_DIR:-/ccache}"
  export CCACHE_BASEDIR="${CCACHE_BASEDIR:-${SRC_DIR}}"
  export CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"
  export CCACHE_NOHASHDIR="${CCACHE_NOHASHDIR:-true}"
  mkdir -p "${CCACHE_DIR}" || true
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
  cmake_launcher_args=(
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  )
  log "ccache enabled: ${CCACHE_DIR}"
elif [[ "${USE_CCACHE}" == "1" ]]; then
  log "ccache requested but not found; continuing without ccache"
fi

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
    -DASCEND_RUNTIME_SESSION_RUN_ONLY=ON \
    "${cmake_launcher_args[@]}"
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
rewritten = 0

def output_path(index):
    if index == 0:
        return out
    base = Path(out)
    return str(base.with_name(f"{base.stem}.{index}{base.suffix}"))

def rewrite_outputs(outputs):
    global rewritten
    for binding in outputs or []:
        if "path" not in binding:
            continue
        binding["path"] = output_path(rewritten)
        rewritten += 1

rewrite_outputs(data.get("outputs"))
for task in data.get("tasks", []):
    rewrite_outputs(task.get("outputs"))
dst.write_text(json.dumps(data, indent=2) + "\n")
PY
  local rc=0
  timeout --kill-after=30s "${NPU_RUN_TIMEOUT_SECONDS}s" \
    env -i \
    HOME="${HOME:-/root}" \
    USER="${USER:-root}" \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-7}" \
    ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}" \
    ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-}}" \
    ASCEND_RUNTIME_TRACE_LAUNCH="${ASCEND_RUNTIME_TRACE_LAUNCH:-1}" \
    RUN_RUNTIME_SESSION="${RUN_RUNTIME_SESSION}" \
    bash -lc '
      source /usr/local/Ascend/driver/bin/setenv.bash >/dev/null 2>&1 || true
      if [ -n "${ASCEND_HOME_PATH:-}" ]; then
        source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1 || true
      fi
      "${RUN_RUNTIME_SESSION}" --run-manifest "$1" --run
    ' bash "${real_manifest}" || rc=$?
  if [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then
    echo "ERROR: real NPU runtime-session timed out after ${NPU_RUN_TIMEOUT_SECONDS}s for manifest: ${source_manifest}" >&2
  fi
  return "${rc}"
}

run_example_case() {
  local case_name="$1"
  local case_dir="${SRC_DIR}/examples/${case_name}"
  [[ -f "${case_dir}/run.sh" ]] || fail "example run.sh not found: ${case_dir}/run.sh"

  if [[ "${SKIP_SIM}" == "1" ]]; then
    log "prepare real-NPU artifacts without sim for ${case_name}"
  else
    log "run xvm-style sim pipeline for ${case_name}"
  fi
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
    if [[ "${SKIP_SIM}" == "1" ]]; then
      bash "examples/${case_name}/run.sh" --prepare-runtime-artifacts --log
    else
      bash "examples/${case_name}/run.sh" --log
    fi
  ) >"${LOG_DIR}/${case_name}-$([[ "${SKIP_SIM}" == "1" ]] && echo prepare || echo sim).log" 2>&1

  local manifest=""
  local manifest_candidates=(
    "${case_dir}/build_mainline/run_manifest.json"
    "${case_dir}/build_mainline/run_manifest.prepared.json"
    "${case_dir}/build_e2e/run_manifest.json"
    "${SRC_DIR}/build/runtime-mix-${case_name}-data/runtime-manifest.json"
  )
  for candidate in "${manifest_candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      manifest="${candidate}"
      break
    fi
  done
  if [[ -z "${manifest}" ]]; then
    printf 'checked manifest candidates:\n' >&2
    printf '  %s\n' "${manifest_candidates[@]}" >&2
    fail "example did not produce a run manifest for ${case_name}"
  fi

  log "run real NPU for ${case_name}"
  mkdir -p "${OUT_DIR}/${case_name}"
  run_real_manifest \
    "${manifest}" \
    "${OUT_DIR}/${case_name}/run_manifest.npu.json" \
    "${OUT_DIR}/${case_name}/output.npy" \
    >"${LOG_DIR}/${case_name}-npu.log" 2>&1
}

run_custom_cmd() {
  local cmd="$1"
  log "run custom command"
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
    export RUN_ONLY_RUNTIME_SESSION="${RUN_RUNTIME_SESSION}"
    bash -lc "${cmd}"
  ) >"${LOG_DIR}/custom-cmd.log" 2>&1
}

run_microcases() {
  local micro_out="${OUT_DIR}/microcases"
  mkdir -p "${micro_out}"
  # shellcheck source=/dev/null
  source "${SRC_DIR}/examples/real-npu-microcases/cases.sh"
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

  for case_name in "${REAL_NPU_MICROCASES[@]}"; do
    local manifest="${micro_out}/${case_name}/run_manifest.json"
    [[ -f "${manifest}" ]] || fail "microcase manifest not found: ${manifest}"
    mkdir -p "${OUT_DIR}/microcases-real/${case_name}"
    log "run real NPU microcase ${case_name}"
    run_real_manifest \
      "${manifest}" \
      "${OUT_DIR}/microcases-real/${case_name}/run_manifest.npu.json" \
      "${OUT_DIR}/microcases-real/${case_name}/output.npy" \
      >"${LOG_DIR}/microcase-${case_name}-npu.log" 2>&1
  done
}

run_multikernel() {
  local multi_out="${OUT_DIR}/real-npu-multikernel"
  mkdir -p "${multi_out}"
  log "run real-NPU multi-kernel scheduling cases"
  (
    cd "${SRC_DIR}"
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
      source_if_exists "${ASCEND_HOME_PATH}/set_env.sh"
    fi
    # shellcheck source=/dev/null
    source examples/env.sh
    export RUNTIME_SESSION="${SRC_DIR}/build/bin/runtime-session"
    export RUN_ONLY_RUNTIME_SESSION="${RUN_RUNTIME_SESSION}"
    multikernel_args=(--out-dir "${multi_out}")
    if [[ "${SKIP_SIM}" == "1" ]]; then
      multikernel_args+=(--skip-sim)
    fi
    bash examples/real-npu-multikernel/run.sh "${multikernel_args[@]}"
  ) >"${LOG_DIR}/real-npu-multikernel.log" 2>&1
}

run_transformer_real_npu() {
  log "prepare transformer runtime artifacts for real NPU"
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
    bash examples/transformer/run-mainline.sh \
      --prepare-runtime-artifacts \
      --batch "${ASCEND_MLIR_TRANSFORMER_BATCH:-1}" \
      --seq "${ASCEND_MLIR_TRANSFORMER_SEQ:-1}" \
      --log
  ) >"${LOG_DIR}/transformer-real-npu-prepare.log" 2>&1

  local manifest="${SRC_DIR}/examples/transformer/build_mainline/run_manifest.json"
  [[ -f "${manifest}" ]] || fail "transformer run manifest not found: ${manifest}"

  log "run real NPU for transformer-real-npu"
  mkdir -p "${OUT_DIR}/transformer-real-npu"
  run_real_manifest \
    "${manifest}" \
    "${OUT_DIR}/transformer-real-npu/run_manifest.npu.json" \
    "${OUT_DIR}/transformer-real-npu/output.npy" \
    >"${LOG_DIR}/transformer-real-npu-npu.log" 2>&1
}

if [[ -n "${CMD}" ]]; then
  run_custom_cmd "${CMD}"
else
  case "${CASE_NAME}" in
    microcases)
      run_microcases
      ;;
    real-npu-multikernel|multikernel)
      run_multikernel
      ;;
    transformer-real-npu|transformer)
      run_transformer_real_npu
      ;;
    all)
      fail "case=all is a host-wrapper mode; use docker-run.sh or sync-and-submit.sh --case all so each real-NPU case runs in a separate container"
      ;;
    *)
      run_example_case "${CASE_NAME}"
      ;;
  esac
fi

log "job complete: ${JOB_DIR}"
echo "${JOB_DIR}" >"${JOB_ROOT}/latest-job.txt"
