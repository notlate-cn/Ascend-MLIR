#!/usr/bin/env bash
# New Ascend mainline pipeline for matmul + bias add + leaky_relu.
#
# This path starts from the linalg/tensor graph in step0_input.mlir and runs
# through Normalize -> Kernelize -> Schedule -> Realize -> Phase 5 mix codegen.

set -euo pipefail
export ASCEND_DAV_SIM_VERSION="${ASCEND_DAV_SIM_VERSION:-dav_3002}"

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"

AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

M=128
K=256
N=128
SEED=42
BLOCK_DIM=1
SOC="${SOC_VERSION:-Ascend910B1}"
VERBOSE=false

require_arg() {
  local opt="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "missing value for ${opt}" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --m)
      require_arg "$1" "${2:-}"
      M="$2"
      shift 2
      ;;
    --k)
      require_arg "$1" "${2:-}"
      K="$2"
      shift 2
      ;;
    --n)
      require_arg "$1" "${2:-}"
      N="$2"
      shift 2
      ;;
    --seed)
      require_arg "$1" "${2:-}"
      SEED="$2"
      shift 2
      ;;
    --block-dim)
      require_arg "$1" "${2:-}"
      BLOCK_DIM="$2"
      shift 2
      ;;
    --soc)
      require_arg "$1" "${2:-}"
      SOC="$2"
      shift 2
      ;;
    --log)
      VERBOSE=true
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "unsupported shape: ${name} must be >= 1" >&2
    exit 2
  fi
}

require_positive_int "M" "$M"
require_positive_int "K" "$K"
require_positive_int "N" "$N"
require_positive_int "BLOCK_DIM" "$BLOCK_DIM"

log() {
  if $VERBOSE; then
    echo "$@"
  fi
}

BUILD_DIR="$DIR/build_mainline"
NPY_DIR="$BUILD_DIR/npy"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output_actual.npy"
EXPECTED_OUTPUT="$NPY_DIR/output.npy"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

PHASE5_TILING_SPACE="$BUILD_DIR/phase5_tiling_space.json"
PHASE5_RUNTIME_MANIFEST="$BUILD_DIR/phase5_runtime_manifest.json"
PHASE5_HOST_TILING="$BUILD_DIR/host_tiling.cpp"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$NPY_DIR"

echo "========================================================"
echo " matmul + add(bias[N]) + leaky_relu Ascend mainline pipeline"
echo "========================================================"
echo "shape.M=$M"
echo "shape.K=$K"
echo "shape.N=$N"
echo "block_dim=$BLOCK_DIM"

echo ""
echo "==================== [STAGE 1] Ascend normalize ===================="
"$AFIR_OPT" "$DIR/step0_input.mlir" \
  --ascend-normalize \
  -o "$BUILD_DIR/step1_normalized.mlir"
log "  output: $BUILD_DIR/step1_normalized.mlir"

echo ""
echo "==================== [STAGE 2] Ascend kernelize ===================="
"$AFIR_OPT" "$BUILD_DIR/step1_normalized.mlir" \
  --ascend-kernelize \
  -o "$BUILD_DIR/step2_kernelized.mlir"
log "  output: $BUILD_DIR/step2_kernelized.mlir"

echo ""
echo "==================== [STAGE 3] Ascend schedule ===================="
"$AFIR_OPT" "$BUILD_DIR/step2_kernelized.mlir" \
  --ascend-schedule='target-tile-policy=legacy-default' \
  -o "$BUILD_DIR/step3_scheduled.mlir"
log "  output: $BUILD_DIR/step3_scheduled.mlir"

echo ""
echo "==================== [STAGE 4] Ascend realize ===================="
"$AFIR_OPT" "$BUILD_DIR/step3_scheduled.mlir" \
  --ascend-realize='materialization-mode=memory-space-annotate' \
  -o "$BUILD_DIR/step4_realized.mlir"
log "  output: $BUILD_DIR/step4_realized.mlir"

echo ""
echo "==================== [STAGE 5] Ascend compute lower ===================="
"$AFIR_OPT" "$BUILD_DIR/step4_realized.mlir" \
  --ascend-compute-lower \
  --annotate-ascendc-kernel-kind \
  -o "$BUILD_DIR/step5_ascendc.mlir"
log "  output: $BUILD_DIR/step5_ascendc.mlir"

echo ""
echo "==================== [STAGE 6] Ascend parallelize ===================="
"$AFIR_OPT" "$BUILD_DIR/step5_ascendc.mlir" \
  --ascend-parallelize \
  -o "$BUILD_DIR/step6_parallelized.mlir"
log "  output: $BUILD_DIR/step6_parallelized.mlir"

echo ""
echo "==================== [STAGE 7] Ascend prepare for emit ===================="
"$AFIR_OPT" "$BUILD_DIR/step6_parallelized.mlir" \
  --ascend-prepare-for-emit \
  -o "$BUILD_DIR/step7_kernel_ir.mlir"
log "  output: $BUILD_DIR/step7_kernel_ir.mlir"

echo ""
echo "==================== [STAGE 8] CANN signature ===================="
"$AFIR_OPT" "$BUILD_DIR/step7_kernel_ir.mlir" \
  --ascend-canonicalize-cann-signature \
  --canonicalize \
  --cse \
  -o "$BUILD_DIR/step8_cann.mlir"
log "  output: $BUILD_DIR/step8_cann.mlir"

echo ""
echo "==================== [STAGE 9] CANN codegen ===================="
"$AFIR_TRANSLATE" -mlir-to-cann "$BUILD_DIR/step8_cann.mlir" \
  --tiling-space-out="$PHASE5_TILING_SPACE" \
  --runtime-manifest-out="$PHASE5_RUNTIME_MANIFEST" \
  --host-tiling-out="$PHASE5_HOST_TILING" \
  --cann-soc="$SOC" \
  -o "$BUILD_DIR/step10_kernel.cpp"
test -s "$BUILD_DIR/step10_kernel.cpp"
test -s "$PHASE5_TILING_SPACE"
test -s "$PHASE5_RUNTIME_MANIFEST"
test -s "$PHASE5_HOST_TILING"
log "  output: $BUILD_DIR/step10_kernel.cpp"

echo ""
echo "==================== [STAGE 10] Generate data ===================="
"$PYTHON" "$DIR/gen_data.py" \
  --M "$M" --K "$K" --N "$N" --seed "$SEED" \
  --out-dir "$NPY_DIR"
log "  output: $NPY_DIR"

echo ""
echo "==================== [STAGE 11] runtime-session compile ===================="
rm -rf "$ARTIFACT_ROOT"
ASCEND_DAV_SIM_VERSION="$ASCEND_DAV_SIM_VERSION" "$RUNTIME_SESSION" \
  --kernel "$BUILD_DIR/step10_kernel.cpp" \
  --kernel-kind mix \
  --name matmul_add_leakyrelu \
  --cann-mlir "$BUILD_DIR/step8_cann.mlir" \
  --npy-dir "$NPY_DIR" \
  --soc "$SOC" \
  --output "$ARTIFACT_ROOT"
test -s "$ARTIFACT_ROOT/out/tiling.bin"
log "  output: $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 12] runtime-session sim ===================="
cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "a", "path": "${NPY_DIR}/input_a.npy" },
    { "name": "b", "path": "${NPY_DIR}/input_b.npy" },
    { "name": "bias", "path": "${NPY_DIR}/input_bias.npy" }
  ],
  "outputs": [
    { "name": "out", "path": "${ACTUAL_OUTPUT}" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "${EXPECTED_OUTPUT}" }
  ],
  "tiling": {
    "binary": "${ARTIFACT_ROOT}/out/tiling.bin"
  },
  "block_dim": ${BLOCK_DIM},
  "workspace_size": 16777216,
  "profiling": true,
  "atol": 1.0,
  "rtol": 1e-2
}
EOF

CANN_ARCH="$(uname -m)"
if [[ "$CANN_ARCH" == "x86_64" ]]; then
  CANN_ARCH="x86_64-linux"
else
  CANN_ARCH="aarch64-linux"
fi
ASCEND_LIB64="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
SOC_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC}/lib"
DAV_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${ASCEND_DAV_SIM_VERSION}/lib"
DEVICE_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64/device/lib64"

ASCEND_DAV_SIM_VERSION="$ASCEND_DAV_SIM_VERSION" \
LD_LIBRARY_PATH="${ARTIFACT_ROOT}/out:${ASCEND_LIB64}:${SOC_SIM_LIB}:${DAV_SIM_LIB}:${DEVICE_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "$RUNTIME_SESSION" \
  --run-manifest "$RUN_MANIFEST" \
  --run >"$VALIDATION_LOG" 2>&1
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]\|^\[DRVSTUB_LOG\]\|^\[FuncCache\]\|^ \|^=\|^\[TmSim\]\|^>>>>' \
  "$VALIDATION_LOG" || true
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"

"$PYTHON" - "$EXPECTED_OUTPUT" "$ACTUAL_OUTPUT" <<'PY'
import sys
import numpy as np

expected = np.load(sys.argv[1])
actual = np.load(sys.argv[2])
if expected.shape != actual.shape:
    raise SystemExit(f"shape mismatch: {actual.shape} vs {expected.shape}")
if expected.dtype != actual.dtype:
    raise SystemExit(f"dtype mismatch: {actual.dtype} vs {expected.dtype}")
diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
print(f"max_abs_diff={diff.max():.6e}")
print(f"mean_abs_diff={diff.mean():.6e}")
if not np.allclose(actual, expected, atol=1.0, rtol=1e-2):
    raise SystemExit("FAIL: outputs differ beyond tolerance")
print("PASS")
PY

echo ""
echo "========================================================"
echo " mainline pipeline complete"
echo "   build_mainline/step1_normalized.mlir"
echo "   build_mainline/step2_kernelized.mlir"
echo "   build_mainline/step3_scheduled.mlir"
echo "   build_mainline/step4_realized.mlir"
echo "   build_mainline/step5_ascendc.mlir"
echo "   build_mainline/step6_parallelized.mlir"
echo "   build_mainline/step7_kernel_ir.mlir"
echo "   build_mainline/step8_cann.mlir"
echo "   build_mainline/step10_kernel.cpp"
echo "   build_mainline/phase5_tiling_space.json"
echo "   build_mainline/phase5_runtime_manifest.json"
echo "   build_mainline/host_tiling.cpp"
echo "   build_mainline/artifact"
echo "========================================================"
