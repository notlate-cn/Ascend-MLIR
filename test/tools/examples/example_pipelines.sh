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
)

require_tool() {
  command -v "$1" >/dev/null 2>&1
}

skip() {
  echo "SKIP: $*"
  exit 0
}

echo "INFO: example pipeline test entry"

if [[ ! -f "${REPO_ROOT}/examples/env.sh" ]]; then
  skip "examples/env.sh not found"
fi

if [[ ! -f "${REPO_ROOT}/scripts/resolve_ascend_env.sh" ]]; then
  skip "scripts/resolve_ascend_env.sh not found"
fi

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/resolve_ascend_env.sh"
if ! resolve_ascend_home >/dev/null 2>&1; then
  skip "ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME is not configured"
fi

# shellcheck source=/dev/null
source "${REPO_ROOT}/examples/env.sh"

for tool in bash python3 afir-opt afir-translate; do
  if ! require_tool "${tool}"; then
    skip "required tool '${tool}' is unavailable"
  fi
done

if ! require_tool compiler; then
  skip "compiler is unavailable"
fi

if ! require_tool validator; then
  skip "validator is unavailable"
fi

if ! python3 -c 'import numpy' >/dev/null 2>&1; then
  skip "python3 numpy module is unavailable"
fi

echo "INFO: executing ${#EXAMPLES[@]} example pipelines"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/afir-example-pipelines.XXXXXX")"
trap 'rm -rf "${workdir}"' EXIT

failures=()

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

  if ! grep -q '^PASS$' "${log_file}"; then
    echo "FAIL [${name}] missing PASS marker"
    tail -n 80 "${log_file}" || true
    failures+=("${name}:pass-marker")
    return
  fi

  if [[ "${name}" == "matmul-add-leakyrelu" ]]; then
    if ! grep -q 'max_abs_diff=0.000000e+00' "${log_file}" || \
       ! grep -q 'mean_abs_diff=0.000000e+00' "${log_file}"; then
      echo "FAIL [${name}] missing zero-diff markers"
      tail -n 80 "${log_file}" || true
      failures+=("${name}:zero-diff")
      return
    fi
  fi

  echo "PASS [${name}]"
}

for example in "${EXAMPLES[@]}"; do
  run_example "${example}"
done

if ((${#failures[@]} > 0)); then
  printf 'FAILED examples: %s\n' "${failures[*]}" >&2
  exit 1
fi

echo "EXECUTED: ${#EXAMPLES[@]} example pipelines"
echo "ALL EXAMPLE PIPELINES PASSED"
