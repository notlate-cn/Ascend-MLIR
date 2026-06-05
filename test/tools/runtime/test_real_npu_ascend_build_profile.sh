#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/real-npu-ascend-profile.XXXXXX")"
cleanup() {
  local rc=$?
  if [[ "${rc}" -ne 0 && -d "${TMP_DIR}/jobs" ]]; then
    find "${TMP_DIR}/jobs" -type f -maxdepth 4 -print -exec sh -c 'echo "== $1 =="; sed -n "1,120p" "$1"' sh {} \; >&2 || true
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

SRC_DIR="${TMP_DIR}/src"
JOB_ROOT="${TMP_DIR}/jobs"
LLVM_DIR="${TMP_DIR}/llvm"
FAKE_BIN="${TMP_DIR}/bin"
mkdir -p \
  "${SRC_DIR}/scripts/real-npu-ci" \
  "${SRC_DIR}/scripts" \
  "${SRC_DIR}/examples/relu-broadcast-transpose" \
  "${SRC_DIR}/examples" \
  "${LLVM_DIR}/lib/cmake/mlir" \
  "${FAKE_BIN}" \
  "${JOB_ROOT}"

cat >"${SRC_DIR}/scripts/real-npu-ci/collect-plog.sh" <<'SH'
#!/usr/bin/env bash
exit 0
SH

cat >"${SRC_DIR}/scripts/build.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${BUILD_DIR}"
printf '%s\n' "$*" >"${BUILD_DIR}/build.args"
if [[ "$*" == *"--build-project"* ]]; then
  echo "real-NPU runner unexpectedly requested full --build-project" >&2
  exit 23
fi
if [[ "$*" != *"--build-ascend"* ]]; then
  echo "real-NPU runner did not request --build-ascend" >&2
  exit 24
fi
mkdir -p "${BUILD_DIR}/bin"
for tool in ascend-mlir-opt ascend-mlir-translate runtime-session ascend-debug; do
  cat >"${BUILD_DIR}/bin/${tool}" <<'TOOL'
#!/usr/bin/env bash
exit 0
TOOL
  chmod +x "${BUILD_DIR}/bin/${tool}"
done
SH

cat >"${SRC_DIR}/examples/env.sh" <<'SH'
#!/usr/bin/env bash
export PATH="$PWD/build/bin:${PATH}"
SH

cat >"${SRC_DIR}/examples/relu-broadcast-transpose/run.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[[ "${AFIR_OPT:-}" == "$PWD/build/bin/ascend-mlir-opt" ]]
[[ "${AFIR_TRANSLATE:-}" == "$PWD/build/bin/ascend-mlir-translate" ]]
[[ "${RUNTIME_SESSION:-}" == "$PWD/build/bin/runtime-session" ]]
mkdir -p examples/relu-broadcast-transpose/build_mainline
cat >examples/relu-broadcast-transpose/build_mainline/run_manifest.json <<'JSON'
{"backend":"sim","outputs":[{"path":"output.npy"}],"tasks":[]}
JSON
SH

cat >"${FAKE_BIN}/cmake" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--build" ]]; then
  build_dir="$2"
  mkdir -p "${build_dir}/bin"
  cat >"${build_dir}/bin/runtime-session" <<'TOOL'
#!/usr/bin/env bash
exit 0
TOOL
  chmod +x "${build_dir}/bin/runtime-session"
  exit 0
fi
build_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -B)
      build_dir="$2"
      shift 2
      ;;
    -B*)
      build_dir="${1#-B}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done
[[ -n "${build_dir}" ]]
mkdir -p "${build_dir}/bin"
SH

cat >"${FAKE_BIN}/timeout" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kill-after=*)
      shift
      ;;
    *s)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done
exec "$@"
SH

cat >"${FAKE_BIN}/date" <<'SH'
#!/usr/bin/env bash
case "${1:-}" in
  -Is) echo "2026-06-05T00:00:00+00:00" ;;
  +%Y%m%d-%H%M%S) echo "20260605-000000" ;;
  *) /bin/date "$@" ;;
esac
SH

chmod +x \
  "${SRC_DIR}/scripts/real-npu-ci/collect-plog.sh" \
  "${SRC_DIR}/scripts/build.sh" \
  "${SRC_DIR}/examples/env.sh" \
  "${SRC_DIR}/examples/relu-broadcast-transpose/run.sh" \
  "${FAKE_BIN}/cmake" \
  "${FAKE_BIN}/timeout" \
  "${FAKE_BIN}/date"

PATH="${FAKE_BIN}:${PATH}" \
ASCEND_MLIR_CI_SOURCE_DIR="${SRC_DIR}" \
ASCEND_MLIR_CI_INCREMENTAL_SOURCE=1 \
ASCEND_MLIR_CI_JOB_ROOT="${JOB_ROOT}" \
ASCEND_MLIR_CI_CASE=relu-broadcast-transpose \
ASCEND_MLIR_CI_REF=test-real-npu-ascend-profile \
ASCEND_MLIR_CI_USE_CCACHE=0 \
ASCEND_MLIR_CI_NPU_RUN_TIMEOUT_SECONDS=30 \
LLVM_BUILD_DIR="${LLVM_DIR}" \
bash "${ROOT}/scripts/real-npu-ci/run-real-npu-job.sh"

LATEST_JOB="$(cat "${JOB_ROOT}/latest-job.txt")"
grep -q "build_profile=ascend" "${LATEST_JOB}/job-env.txt"
grep -q -- "--build-ascend" "${SRC_DIR}/build/build.args"
