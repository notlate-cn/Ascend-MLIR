#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CASE_DIR="${ROOT}/examples/split-relu-brc-add-mul"
RUNTIME_SESSION="${RUNTIME_SESSION:-${ROOT}/build/bin/runtime-session}"
RUN_ONLY_RUNTIME_SESSION="${RUN_ONLY_RUNTIME_SESSION:-${RUNTIME_SESSION}}"
PYTHON="${PYTHON:-python3}"

M=64
N=80
BACKEND=sim

usage() {
  cat <<'EOF'
Usage: run_split_post_add_checkpoint.sh [--backend sim|npu|both] [--m M] [--n N]

Build split-relu-brc-add-mul, patch the generated kernel so the final Mul ops
are skipped, and compare the output against the relu+bias intermediate.
This is a diagnostic checkpoint, not a correctness proof for the full kernel.
EOF
}

require_arg() {
  local opt="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "missing value for ${opt}" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      require_arg "$1" "${2:-}"
      BACKEND="$2"
      shift 2
      ;;
    --m)
      require_arg "$1" "${2:-}"
      M="$2"
      shift 2
      ;;
    --n)
      require_arg "$1" "${2:-}"
      N="$2"
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

case "${BACKEND}" in
  sim|npu|both) ;;
  *)
    echo "unsupported backend: ${BACKEND}" >&2
    exit 2
    ;;
esac

cd "${ROOT}"

if [[ -f "${ROOT}/examples/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/examples/env.sh"
fi

export RUNTIME_SESSION
echo "checkpoint.shape.M=${M}"
echo "checkpoint.shape.N=${N}"
echo "checkpoint.backend=${BACKEND}"
echo "checkpoint.ascend_device_id=${ASCEND_DEVICE_ID:-}"

npu_ld_library_path() {
  local cleaned=""
  local part=""
  local old_ifs="${IFS}"
  IFS=:
  for part in ${LD_LIBRARY_PATH:-}; do
    case "${part}" in
      */simulator/*|*/runtime/lib64/stub) continue ;;
      "") continue ;;
    esac
    if [[ -z "${cleaned}" ]]; then
      cleaned="${part}"
    else
      cleaned="${cleaned}:${part}"
    fi
  done
  IFS="${old_ifs}"

  local cann_base="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}"
  if [[ -n "${cann_base}" ]]; then
    local cann_arch="${CANN_ARCH:-$(resolve_cann_arch_dir)}"
    local base_lib="${cann_base}/${cann_arch}/lib64"
    local device_lib="${base_lib}/device/lib64"
    cleaned="${base_lib}:${device_lib}${cleaned:+:${cleaned}}"
  fi
  echo "${cleaned}"
}

bash "${CASE_DIR}/run-mainline.sh" --m "${M}" --n "${N}" --log

BUILD_DIR="${CASE_DIR}/build_mainline"
SRC_KERNEL="${BUILD_DIR}/step10_kernel.cpp"
CHECK_KERNEL="${BUILD_DIR}/step10_post_add_checkpoint.cpp"
CHECK_ARTIFACT="${BUILD_DIR}/artifact_post_add_checkpoint"
CHECK_EXPECTED="${BUILD_DIR}/post_add_expected.npy"

cp "${SRC_KERNEL}" "${CHECK_KERNEL}"

"${PYTHON}" - "${CHECK_KERNEL}" "${BUILD_DIR}" <<'PY'
import re
import sys
from pathlib import Path

import numpy as np

kernel = Path(sys.argv[1])
build_dir = Path(sys.argv[2])
text = kernel.read_text()
pattern = re.compile(
    r"^(\s*)AscendC::Mul\(([^;]+)\);\n",
    re.MULTILINE,
)

def replace(match):
    return (
        f"{match.group(1)}// post-add checkpoint: keep the Add accumulator.\n"
        f"{match.group(1)}(void)_afir_chunk;\n"
    )

patched, count = pattern.subn(replace, text)
if count != 2:
    raise SystemExit(f"expected to patch 2 Mul ops, patched {count}")
kernel.write_text(patched)

input_a = np.load(build_dir / "input_a.npy").astype(np.float32)
bias0 = np.load(build_dir / "bias0.npy").astype(np.float32)
bias1 = np.load(build_dir / "bias1.npy").astype(np.float32)
hm = input_a.shape[0] // 2
out0 = np.maximum(input_a[:hm, :], 0.0) + bias0[:, None]
out1 = np.maximum(input_a[hm:, :], 0.0) + bias1[:, None]
expected = np.concatenate([out0, out1], axis=0).astype(np.float16)
np.save(build_dir / "post_add_expected.npy", expected)
print(f"checkpoint.patch.mul_ops={count}")
print(f"checkpoint.expected={build_dir / 'post_add_expected.npy'}")
PY

"${RUNTIME_SESSION}" \
  --kernel "${CHECK_KERNEL}" \
  --kernel-kind vec \
  --output "${CHECK_ARTIFACT}" \
  --name ewop_broadcast_split

make_manifest() {
  local backend="$1"
  local output="$2"
  local manifest="$3"
  "${PYTHON}" - "${BUILD_DIR}/run_manifest.json" "${manifest}" \
    "${CHECK_ARTIFACT}" "${backend}" "${output}" "${M}" "${N}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
artifact = sys.argv[3]
backend = sys.argv[4]
output = sys.argv[5]
m = int(sys.argv[6])
n = int(sys.argv[7])

data = json.loads(src.read_text())
data["backend"] = backend
data["artifact_root"] = artifact
data["outputs"] = [{"name": "out", "path": output, "shape": [m, n], "dtype": "f16"}]
data["expected_outputs"] = []
dst.write_text(json.dumps(data, indent=2) + "\n")
PY
}

compare_output() {
  local label="$1"
  local actual="$2"
  local require_pass="$3"
  "${PYTHON}" - "${label}" "${actual}" "${CHECK_EXPECTED}" "${N}" "${require_pass}" <<'PY'
import sys
import numpy as np

label, actual_path, expected_path = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4])
require_pass = sys.argv[5] == "1"
actual = np.load(actual_path).astype(np.float32)
expected = np.load(expected_path).astype(np.float32)
diff = np.abs(actual - expected)
tail_start = (n // 64) * 64

def stats(name, arr):
    print(f"checkpoint.{label}.{name}.max_abs_diff={float(arr.max()) if arr.size else 0.0:.8g}")
    print(f"checkpoint.{label}.{name}.mean_abs_diff={float(arr.mean()) if arr.size else 0.0:.8g}")

stats("all", diff)
if 0 < tail_start < n:
    stats(f"head_cols_0_{tail_start - 1}", diff[:, :tail_start])
    stats(f"tail_cols_{tail_start}_{n - 1}", diff[:, tail_start:])
match = bool(np.allclose(actual, expected, atol=1e-2, rtol=1e-2))
print(f"checkpoint.{label}.match={int(match)}")
if require_pass and not match:
    raise SystemExit(1)
PY
}

run_one() {
  local backend="$1"
  local runner="$2"
  local output="${BUILD_DIR}/post_add_${backend}_actual.npy"
  local manifest="${BUILD_DIR}/post_add_${backend}_manifest.json"
  local log="${BUILD_DIR}/post_add_${backend}.log"
  local require_pass=0
  if [[ "${backend}" == "sim" ]]; then
    require_pass=1
  fi
  make_manifest "${backend}" "${output}" "${manifest}"
  echo "checkpoint.${backend}.manifest=${manifest}"
  if [[ "${backend}" == "npu" ]]; then
    local npu_ld
    npu_ld="$(npu_ld_library_path)"
    case "${npu_ld}" in
      *"/simulator/"*) echo "checkpoint.npu.ld.has_sim=1" ;;
      *) echo "checkpoint.npu.ld.has_sim=0" ;;
    esac
    case "${npu_ld}" in
      *"/runtime/lib64/stub"*) echo "checkpoint.npu.ld.has_stub=1" ;;
      *) echo "checkpoint.npu.ld.has_stub=0" ;;
    esac
    env -i \
      HOME="${HOME:-/root}" \
      PATH="${PATH}" \
      ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}" \
      ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH:-}" \
      ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}" \
      ASCEND_RUNTIME_TRACE_LAUNCH="${ASCEND_RUNTIME_TRACE_LAUNCH:-1}" \
      LD_LIBRARY_PATH="${npu_ld}" \
      "${runner}" --run-manifest "${manifest}" --run >"${log}" 2>&1
  else
    "${runner}" --run-manifest "${manifest}" --run >"${log}" 2>&1
  fi
  grep -E '^session\.(backend|result|validation|error_stage|error)=' "${log}" || true
  compare_output "${backend}" "${output}" "${require_pass}"
}

if [[ "${BACKEND}" == "sim" || "${BACKEND}" == "both" ]]; then
  run_one sim "${RUNTIME_SESSION}"
fi

if [[ "${BACKEND}" == "npu" || "${BACKEND}" == "both" ]]; then
  run_one npu "${RUN_ONLY_RUNTIME_SESSION}"
fi
