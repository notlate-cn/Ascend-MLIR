#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CASE="all"
OUT_DIR="${PROJECT_ROOT}/build/examples/real-npu-multikernel"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
RUNNER="${RUN_ONLY_RUNTIME_SESSION:-${RUNTIME_SESSION}}"
SKIP_SIM=false

usage() {
  cat <<'EOF' >&2
Usage:
  run.sh [--case serial-two-kernel|fork-join|all] [--out-dir <dir>] [--skip-sim]

Prepares and runs real-NPU multi-kernel runtime-session manifests. Artifact
compilation uses RUNTIME_SESSION. Manifest execution uses RUN_ONLY_RUNTIME_SESSION
when set, otherwise RUNTIME_SESSION. By default each case runs a simulation
manifest first, then the NPU manifest.
EOF
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --case)
      [ "$#" -ge 2 ] || usage
      CASE="$2"
      shift 2
      ;;
    --out-dir)
      [ "$#" -ge 2 ] || usage
      OUT_DIR="$2"
      shift 2
      ;;
    --skip-sim)
      SKIP_SIM=true
      shift
      ;;
    --help|-h)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

case "${CASE}" in
  serial-two-kernel|fork-join|all)
    ;;
  *)
    echo "unknown real-npu-multikernel case: ${CASE}" >&2
    usage
    ;;
esac

mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

RUNTIME_SESSION="${RUNTIME_SESSION}" \
  bash "${SCRIPT_DIR}/prepare.sh" --out-dir "${OUT_DIR}"

write_backend_manifest() {
  local source_manifest="$1"
  local target_manifest="$2"
  local backend="$3"
  local output_suffix="$4"
  "${PYTHON:-python3}" - "${source_manifest}" "${target_manifest}" \
    "${backend}" "${output_suffix}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
backend = sys.argv[3]
suffix = sys.argv[4]
data = json.loads(src.read_text())
data["backend"] = backend

for task in data.get("tasks", []):
    for binding in task.get("outputs", []):
        path = binding.get("path")
        if path:
            p = Path(path)
            binding["path"] = str(p.with_name(p.stem + suffix + p.suffix))

dst.write_text(json.dumps(data, indent=2) + "\n")
PY
}

verify_log() {
  local log_file="$1"
  local backend="$2"
  local task_count="$3"
  grep -q "^session.backend=${backend}$" "${log_file}"
  grep -q '^session.result=success$' "${log_file}"
  grep -q '^session.validation=pass$' "${log_file}"
  grep -q '^session.runtime.attribute.scheduler_scope=global$' "${log_file}"
  grep -q "^session.runtime.counter.planned_task_count=${task_count}$" \
    "${log_file}"
  if grep -q "^session.runtime.counter.serialized_launch_count=${task_count}$" \
      "${log_file}"; then
    return
  fi
  if [ "${backend}" = "npu" ]; then
    local launch_count
    launch_count="$(grep -c '^\[npu-launch\] kernel=' "${log_file}" || true)"
    [ "${launch_count}" = "${task_count}" ]
    return
  fi
  return 1
}

run_case() {
  local case_name="$1"
  local task_count="$2"
  local base_manifest="${OUT_DIR}/${case_name}/run_manifest.json"
  echo "=== real-npu-multikernel: ${case_name} ==="
  if ! "${SKIP_SIM}"; then
    local sim_manifest="${OUT_DIR}/${case_name}/run_manifest.sim.json"
    local sim_log="${OUT_DIR}/${case_name}/runtime-session.sim.log"
    write_backend_manifest "${base_manifest}" "${sim_manifest}" sim ".sim"
    "${RUNTIME_SESSION}" --run-manifest "${sim_manifest}" --run 2>&1 | tee "${sim_log}"
    verify_log "${sim_log}" sim "${task_count}"
  fi
  local npu_log="${OUT_DIR}/${case_name}/runtime-session.npu.log"
  "${RUNNER}" \
    --run-manifest "${base_manifest}" \
    --run 2>&1 | tee "${npu_log}"
  verify_log "${npu_log}" npu "${task_count}"
}

if [ "${CASE}" = "all" ] || [ "${CASE}" = "serial-two-kernel" ]; then
  run_case serial-two-kernel 2
fi
if [ "${CASE}" = "all" ] || [ "${CASE}" = "fork-join" ]; then
  run_case fork-join 3
fi
