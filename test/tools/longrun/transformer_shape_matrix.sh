#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_SCRIPT="${REPO_ROOT}/examples/transformer/run-mainline.sh"

DEFAULT_CASES="1x1 1x2 1x4 1x16 2x2 2x4"
CASE_LIST="${TRANSFORMER_SHAPE_MATRIX_CASES:-${DEFAULT_CASES}}"
CASE_TIMEOUT="${TRANSFORMER_SHAPE_TIMEOUT:-2400s}"
RUN_TIMEOUT="${RUN_TIMEOUT:-1200s}"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

require_tool() {
  command -v "$1" >/dev/null 2>&1 || fail "required tool '$1' is unavailable"
}

parse_case() {
  local spec="$1"
  if [[ "$spec" =~ ^([1-9][0-9]*)[x:]([1-9][0-9]*)$ ]]; then
    BASH_REMATCH_BATCH="${BASH_REMATCH[1]}"
    BASH_REMATCH_SEQ="${BASH_REMATCH[2]}"
    return
  fi
  fail "invalid transformer shape case '${spec}', expected BxS"
}

check_marker() {
  local log_file="$1"
  local marker="$2"
  grep -Fxq "${marker}" "${log_file}" || {
    echo "FAIL: missing marker '${marker}'" >&2
    tail -n 120 "${log_file}" >&2 || true
    exit 1
  }
}

run_case() {
  local spec="$1"
  parse_case "${spec}"
  local batch="${BASH_REMATCH_BATCH}"
  local seq="${BASH_REMATCH_SEQ}"
  local case_name="batch${batch}_seq${seq}"
  local log_file="${WORKDIR}/${case_name}.log"

  echo "transformer_shape.case=${case_name}.start"
  if ! RUN_TIMEOUT="${RUN_TIMEOUT}" timeout "${CASE_TIMEOUT}" bash "${RUN_SCRIPT}" \
      --runtime-e2e --batch "${batch}" --seq "${seq}" --log \
      >"${log_file}" 2>&1; then
    echo "FAIL [transformer-shape batch=${batch} seq=${seq}]"
    tail -n 200 "${log_file}" || true
    return 1
  fi

  check_marker "${log_file}" "transformer_data.batch=${batch}"
  check_marker "${log_file}" "transformer_data.seq=${seq}"
  check_marker "${log_file}" "transformer_dynamic.artifact_compile=pass"
  check_marker "${log_file}" "transformer_dynamic.run_plan.tasks=53"
  check_marker "${log_file}" "session.result=success"
  check_marker "${log_file}" "session.validation=pass"
  check_marker "${log_file}" "transformer_dynamic.runtime_session=pass"
  check_marker "${log_file}" "transformer_dynamic.validation=pass"

  echo "transformer_shape.case=${case_name}.artifact_compile=pass"
  echo "transformer_shape.case=${case_name}.runtime_session=pass"
  echo "transformer_shape.case=${case_name}.validation=pass"
  echo "PASS [transformer-shape batch=${batch} seq=${seq}]"
}

echo "INFO: transformer shape matrix test entry"

[[ -f "${RUN_SCRIPT}" ]] || fail "transformer run script not found: ${RUN_SCRIPT}"
require_tool bash
require_tool timeout

read -r -a CASES <<<"${CASE_LIST//,/ }"
if ((${#CASES[@]} == 0)); then
  fail "empty TRANSFORMER_SHAPE_MATRIX_CASES"
fi

echo "INFO: executing ${#CASES[@]} transformer runtime shapes"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/afir-transformer-shape-matrix.XXXXXX")"
trap 'rm -rf "${WORKDIR}"' EXIT

for shape_case in "${CASES[@]}"; do
  run_case "${shape_case}"
done

echo "EXECUTED: ${#CASES[@]} transformer runtime shapes"
echo "ALL TRANSFORMER SHAPE MATRIX PASSED"
