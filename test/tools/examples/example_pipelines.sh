#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

EXAMPLES=(
  "add-broadcast-concat"
  "broadcast-add-reduce"
  "gather-elementwise-fusion"
  "relu-broadcast-transpose"
  "split-relu-brc-add-mul"
  "matmul-add-leakyrelu"
  "two-kernel-dag"
)

require_tool() {
  command -v "$1" >/dev/null 2>&1
}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

echo "INFO: example pipeline test entry"

if [[ ! -f "${REPO_ROOT}/examples/env.sh" ]]; then
  fail "examples/env.sh not found"
fi

if [[ ! -f "${REPO_ROOT}/scripts/resolve_ascend_env.sh" ]]; then
  fail "scripts/resolve_ascend_env.sh not found"
fi

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/resolve_ascend_env.sh"
if ! resolve_ascend_home >/dev/null 2>&1; then
  fail "Ascend environment is unavailable; run 'source ~/Ascend/latest/set_env.sh' or export ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME"
fi

# shellcheck source=/dev/null
source "${REPO_ROOT}/examples/env.sh"

export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64:${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC_VERSION:-Ascend910B1}/lib:${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64/device/lib64:${ASCEND_HOME_PATH}/runtime/lib64/stub${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for tool in bash python3 afir-opt afir-translate runtime-session mix-compiler; do
  if ! require_tool "${tool}"; then
    fail "required tool '${tool}' is unavailable"
  fi
done

if ! python3 -c 'import numpy' >/dev/null 2>&1; then
  fail "python3 numpy module is unavailable"
fi

echo "INFO: executing ${#EXAMPLES[@]} example pipelines"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/afir-example-pipelines.XXXXXX")"
trap 'rm -rf "${workdir}"' EXIT

failures=()

run_runtime_session_smoke() {
  local name="$1"
  local manifest_source="$2"
  local manifest_file="${workdir}/smoke-${name}.json"
  local log_file="${workdir}/smoke-${name}.log"

  if [[ ! -f "${manifest_source}" ]]; then
    echo "FAIL [smoke:${name}] missing manifest: ${manifest_source}"
    failures+=("smoke:${name}:missing-manifest")
    return
  fi

  python3 - "${manifest_source}" "${manifest_file}" "${REPO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

src_path = Path(sys.argv[1])
dst_path = Path(sys.argv[2])
repo_root = sys.argv[3]

data = json.loads(src_path.read_text())
artifact_root = data.get("artifact_root", "")
marker = "/examples/"
source_root = artifact_root.split(marker, 1)[0] if marker in artifact_root else None

def rewrite(obj):
    if isinstance(obj, dict):
        return {k: rewrite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v) for v in obj]
    if isinstance(obj, str) and source_root and obj.startswith(source_root):
        return repo_root + obj[len(source_root):]
    return obj

dst_path.write_text(json.dumps(rewrite(data), indent=2) + "\n")
PY

  echo "== SMOKE ${name} =="
  if ! runtime-session --run-manifest "${manifest_file}" --run >"${log_file}" 2>&1; then
    echo "FAIL [smoke:${name}] exited nonzero"
    tail -n 80 "${log_file}" || true
    failures+=("smoke:${name}:exit")
    return
  fi

  if ! grep -q '^session.result=success$' "${log_file}" || \
     ! grep -q '^session.validation=pass$' "${log_file}"; then
    echo "FAIL [smoke:${name}] missing runtime-session success markers"
    tail -n 80 "${log_file}" || true
    failures+=("smoke:${name}:markers")
    return
  fi

  echo "PASS [smoke:${name}]"
}

run_example() {
  local name="$1"
  local log_file="${workdir}/${name}.log"
  local script="${REPO_ROOT}/examples/${name}/run.sh"

  if [[ ! -x "${script}" && ! -f "${script}" ]]; then
    echo "FAIL [${name}] missing script: ${script}"
    failures+=("${name}:missing-script")
    return
  fi

  echo "== RUN ${name} =="
  if ! bash "${script}" --log >"${log_file}" 2>&1; then
    echo "FAIL [${name}] exited nonzero"
    tail -n 80 "${log_file}" || true
    failures+=("${name}:exit")
    return
  fi

  if [[ "${name}" == "matmul-add-leakyrelu" ]]; then
    if ! grep -q '^PASS$' "${log_file}" || \
       ! grep -q 'max_abs_diff=0.000000e+00' "${log_file}" || \
       ! grep -q 'mean_abs_diff=0.000000e+00' "${log_file}"; then
      echo "FAIL [${name}] missing mix validation markers"
      tail -n 80 "${log_file}" || true
      failures+=("${name}:mix-markers")
      return
    fi
  else
    if ! grep -q '^session.result=success$' "${log_file}" || \
       ! grep -q '^session.validation=pass$' "${log_file}"; then
      echo "FAIL [${name}] missing runtime-session success markers"
      tail -n 80 "${log_file}" || true
      failures+=("${name}:runtime-session-markers")
      return
    fi
  fi

  echo "PASS [${name}]"
}

echo "INFO: executing example pipelines"
for example in "${EXAMPLES[@]}"; do
  run_example "${example}"
done

if ((${#failures[@]} == 0)); then
  echo "INFO: executing focused cross-session runtime-session smoke"
  run_runtime_session_smoke "session-a" "${REPO_ROOT}/examples/add-broadcast-concat/build_mainline/run_manifest.json"
  run_runtime_session_smoke "session-b" "${REPO_ROOT}/examples/broadcast-add-reduce/build_mainline/run_manifest.json"
fi
if ((${#failures[@]} == 0)); then
  echo "PASS [cross-session smoke]"
fi

if ((${#failures[@]} > 0)); then
  printf 'FAILED examples: %s\n' "${failures[*]}" >&2
  exit 1
fi

echo "EXECUTED: ${#EXAMPLES[@]} example pipelines"
echo "ALL EXAMPLE PIPELINES PASSED"
