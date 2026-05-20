#!/usr/bin/env bash
# examples/matmul-relu-fused-e2e/run.sh
# CV-fusion (matmul + relu) end-to-end via auto-fuse-codegen.
#
# This is the first CV-fusion gate that does NOT use the transform-interpreter
# path used by matmul-add-leakyrelu; it consumes raw `linalg.matmul + linalg.generic relu`
# and lets auto-fuse handle group analysis, cube tile plan, and emission.
#
# Usage:
#   source examples/env.sh
#   bash examples/matmul-relu-fused-e2e/run.sh [--log]
#
# STAGE summary (compared to matmul-add-leakyrelu):
#   our STAGE 1-7: replaced by single `--auto-fuse-codegen` invocation
#   our STAGE 8:   afir-translate -mlir-to-cann (unchanged)
#   STAGE 9/10:    sim run (TODO: mix-compiler integration for no-bias matmul)
set -euo pipefail
export ASCEND_DAV_SIM_VERSION=dav_3002

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"

VERBOSE=false
for arg in "$@"; do [[ $arg == "--log" ]] && VERBOSE=true; done
log() { $VERBOSE && echo "$@" || true; }

echo "=========================================================="
echo " matmul + relu CV-fusion via auto-fuse-codegen"
echo "=========================================================="

# ── STAGE 1-7: auto-fuse codegen (replaces transform+bufferize+place chain)
echo "=== [STAGE 1-7] auto-fuse-codegen ==="
$AFIR_OPT --auto-fuse-codegen \
  "$SCRIPT_DIR/step0_input.mlir" \
  -o "$SCRIPT_DIR/step7_cann.mlir"
log "  step7_cann.mlir done"

# ── STAGE 8: mlir → cann (.cpp)
echo "=== [STAGE 8] mlir-to-cann codegen ==="
$AFIR_TRANSLATE --mlir-to-cann \
  "$SCRIPT_DIR/step7_cann.mlir" \
  -o "$SCRIPT_DIR/step8_kernel.cpp"

# Validate the emitted kernel contains the cube + vector epilogue.
if ! grep -q 'mm.template IterateAll' "$SCRIPT_DIR/step8_kernel.cpp"; then
  echo "FAIL: emitted kernel missing IterateAll" >&2
  exit 2
fi
if ! grep -q 'Relu(' "$SCRIPT_DIR/step8_kernel.cpp"; then
  echo "FAIL: emitted kernel missing Relu epilogue" >&2
  exit 2
fi
echo "  step8_kernel.cpp emitted (cube + relu epilogue)"

# ── STAGE 9: test data
echo "=== [STAGE 9] gen test data ==="
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
mkdir -p "${DATA_DIR}"
python3 "$SCRIPT_DIR/gen_data.py" --out-dir "${DATA_DIR}"

# ── STAGE 10: RuntimeMix compile
echo "=== [STAGE 10] RuntimeMix compile ==="
ARTIFACT_DIR="${ARTIFACT_DIR:-$SCRIPT_DIR/artifact}"
rm -rf "${ARTIFACT_DIR}"
"${MIX_COMPILER:-mix-compiler}" \
  --kernel       "$SCRIPT_DIR/step8_kernel.cpp" \
  --cann-mlir    "$SCRIPT_DIR/step7_cann.mlir" \
  --npy-dir      "${DATA_DIR}" \
  --output       "${ARTIFACT_DIR}" \
  --soc          "${SOC_VERSION:-Ascend910B1}" > "${ARTIFACT_DIR}.log" 2>&1 || {
    echo "FAIL: mix-compiler failed" >&2
    tail -n 40 "${ARTIFACT_DIR}.log" >&2
    exit 2
  }
log "  artifact: ${ARTIFACT_DIR}"

# ── STAGE 11: sim run + validation
echo "=== [STAGE 11] runtime-session sim ==="
RUN_MANIFEST="${ARTIFACT_DIR}/run_manifest.json"
ACTUAL_OUTPUT="${ARTIFACT_DIR}/actual_output.npy"
python3 "$SCRIPT_DIR/build_run_manifest.py" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --data-dir     "${DATA_DIR}" \
  --out-manifest "${RUN_MANIFEST}" \
  --out-npy      "${ACTUAL_OUTPUT}" > /dev/null

CANN_ARCH="$(uname -m)"
[[ "${CANN_ARCH}" == "x86_64" ]] && CANN_ARCH=x86_64-linux || CANN_ARCH=aarch64-linux
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64:${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC_VERSION:-Ascend910B1}/lib:${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64/device/lib64:${ASCEND_HOME_PATH}/runtime/lib64/stub${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

SIM_LOG="${ARTIFACT_DIR}.sim.log"
"${RUNTIME_SESSION:-runtime-session}" --run-manifest "${RUN_MANIFEST}" --run > "${SIM_LOG}" 2>&1 || {
    echo "FAIL: runtime-session exited nonzero" >&2
    tail -n 30 "${SIM_LOG}" >&2
    exit 2
  }
grep -q '^session.result=success$'     "${SIM_LOG}" || { echo "FAIL: missing session.result=success"     >&2; exit 2; }
grep -q '^session.validation=pass$'    "${SIM_LOG}" || { echo "FAIL: missing session.validation=pass"    >&2; exit 2; }

# Numerical cross-check against the golden npy.
python3 - "${ACTUAL_OUTPUT}" "${DATA_DIR}/output.npy" <<'PY'
import sys
import numpy as np
actual = np.load(sys.argv[1])
golden = np.load(sys.argv[2])
diff = float(np.max(np.abs(actual - golden)))
print(f"max_abs_diff = {diff}")
if diff > 1e-3:
    sys.exit(f"FAIL: numerical mismatch (max_abs_diff={diff})")
PY

echo "PASS [matmul-relu-fused-e2e]"
