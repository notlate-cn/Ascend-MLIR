#!/usr/bin/env bash
# Ascend mainline runtime-session slices for transformer fragments.

set -euo pipefail
export ASCEND_DAV_SIM_VERSION="${ASCEND_DAV_SIM_VERSION:-dav_3002}"

DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../mainline-target-env.sh
source "$DIR/../mainline-target-env.sh"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

FRAGMENT="layernorm"
M=4
BATCH=1
SEQ=16
SEED=42
BLOCK_DIM=1
SOC="${SOC_VERSION:-Ascend910B1}"
VERBOSE=false
RUNTIME_E2E=""

require_arg() {
  local opt="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "missing value for ${opt}" >&2
    exit 2
  fi
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "unsupported shape: ${name} must be >= 1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fragment)
      require_arg "$1" "${2:-}"
      FRAGMENT="$2"
      shift 2
      ;;
    --m)
      require_arg "$1" "${2:-}"
      M="$2"
      shift 2
      ;;
    --batch)
      require_arg "$1" "${2:-}"
      BATCH="$2"
      shift 2
      ;;
    --seq)
      require_arg "$1" "${2:-}"
      SEQ="$2"
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
    --runtime-e2e)
      RUNTIME_E2E=true
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$FRAGMENT" in
  layernorm|qkv)
    ;;
  *)
    echo "unknown fragment: $FRAGMENT" >&2
    exit 2
    ;;
esac

require_positive_int "M" "$M"
require_positive_int "BATCH" "$BATCH"
require_positive_int "SEQ" "$SEQ"
require_positive_int "BLOCK_DIM" "$BLOCK_DIM"

if [[ -z "$RUNTIME_E2E" ]]; then
  if [[ "$FRAGMENT" == "qkv" ]]; then
    RUNTIME_E2E=false
  else
    RUNTIME_E2E=true
  fi
fi

log() {
  if $VERBOSE; then
    echo "$@"
  fi
}

BUILD_DIR="$DIR/build_${FRAGMENT}"
NPY_DIR="$BUILD_DIR/npy"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output_actual.npy"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"
PHASE5_TILING_SPACE="$BUILD_DIR/phase5_tiling_space.json"
PHASE5_RUNTIME_MANIFEST="$BUILD_DIR/phase5_runtime_manifest.json"
PHASE5_HOST_TILING="$BUILD_DIR/host_tiling.cpp"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$NPY_DIR"

echo "========================================================"
echo " transformer fragment ${FRAGMENT} Ascend mainline pipeline"
echo "========================================================"
if [[ "$FRAGMENT" == "layernorm" ]]; then
  echo "shape.M=$M"
  echo "shape.hidden=128"
else
  echo "shape.batch=$BATCH"
  echo "shape.seq=$SEQ"
  echo "shape.tokens=$((BATCH * SEQ))"
  echo "shape.hidden=16"
  echo "shape.qkv=48"
fi
echo "block_dim=$BLOCK_DIM"

echo ""
echo "==================== [STAGE 1] Ascend normalize ===================="
"$AFIR_OPT" "$DIR/${FRAGMENT}.mlir" \
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
  --ascend-schedule="target-tile-policy=target-aware cann-root=${CANN_ROOT} soc=${SOC}" \
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
if [[ "$FRAGMENT" == "qkv" ]]; then
  "$AFIR_OPT" "$BUILD_DIR/step4_realized.mlir" \
    --ascend-compute-lower \
    --annotate-ascendc-kernel-kind \
    -o "$BUILD_DIR/step5_ascendc.mlir"
else
  "$AFIR_OPT" "$BUILD_DIR/step4_realized.mlir" \
    --ascend-compute-lower \
    -o "$BUILD_DIR/step5_ascendc.mlir"
fi
log "  output: $BUILD_DIR/step5_ascendc.mlir"
echo "transformer_fragment.${FRAGMENT}.full_codegen=pass"

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
  -o "$BUILD_DIR/step9_kernel.cpp"
test -s "$BUILD_DIR/step9_kernel.cpp"
test -s "$PHASE5_TILING_SPACE"
test -s "$PHASE5_RUNTIME_MANIFEST"
test -s "$PHASE5_HOST_TILING"
log "  output: $BUILD_DIR/step9_kernel.cpp"
echo "transformer_fragment.${FRAGMENT}.phase5_translate=pass"

echo ""
echo "==================== [STAGE 10] Generate data ===================="
"$PYTHON" "$DIR/gen_data.py" \
  --fragment "$FRAGMENT" \
  --m "$M" \
  --batch "$BATCH" \
  --seq "$SEQ" \
  --seed "$SEED" \
  --out-dir "$NPY_DIR" \
  --artifact-root "$ARTIFACT_ROOT" \
  --tiling-schema "$PHASE5_TILING_SPACE" \
  --run-manifest "$RUN_MANIFEST" \
  --actual-output "$ACTUAL_OUTPUT" \
  --block-dim "$BLOCK_DIM"
log "  output: $NPY_DIR"

echo ""
echo "==================== [STAGE 11] runtime-session compile ===================="
rm -rf "$ARTIFACT_ROOT"
if [[ "$FRAGMENT" == "qkv" ]]; then
  ASCEND_DAV_SIM_VERSION="$ASCEND_DAV_SIM_VERSION" "$RUNTIME_SESSION" \
    --kernel "$BUILD_DIR/step9_kernel.cpp" \
    --kernel-kind mix \
    --name kernel \
    --cann-mlir "$BUILD_DIR/step8_cann.mlir" \
    --npy-dir "$NPY_DIR" \
    --soc "$SOC" \
    --output "$ARTIFACT_ROOT"
  test -s "$ARTIFACT_ROOT/out/tiling.bin"
  cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "x", "path": "${NPY_DIR}/input_x.npy" },
    { "name": "weight", "path": "${NPY_DIR}/input_weight.npy" },
    { "name": "bias", "path": "${NPY_DIR}/input_bias.npy" }
  ],
  "outputs": [
    { "name": "out", "path": "${ACTUAL_OUTPUT}" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "${NPY_DIR}/expected_out.npy" }
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
else
  "$RUNTIME_SESSION" \
    --kernel "$BUILD_DIR/step9_kernel.cpp" \
    --kernel-kind vec \
    --output "$ARTIFACT_ROOT" \
    --name kernel
fi
log "  output: $ARTIFACT_ROOT"
echo "transformer_fragment.${FRAGMENT}.artifact_compile=pass"

if [[ "$FRAGMENT" == "qkv" && "$RUNTIME_E2E" != true ]]; then
  echo "transformer_fragment.qkv.runtime_session=deferred"
  echo "transformer_fragment.qkv.next_gap=qkv_runtime_sim"
  exit 0
fi

echo ""
echo "==================== [STAGE 12] runtime-session sim ===================="
if [[ "$FRAGMENT" == "qkv" ]]; then
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
else
  "$RUNTIME_SESSION" \
    --run-manifest "$RUN_MANIFEST" \
    --run >"$VALIDATION_LOG" 2>&1
fi
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]\|^\[DRVSTUB_LOG\]\|^\[FuncCache\]\|^ \|^=\|^\[TmSim\]\|^>>>>' \
  "$VALIDATION_LOG" || true
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"

echo "transformer_fragment.${FRAGMENT}.runtime_session=pass"
echo "transformer_fragment.${FRAGMENT}.validation=pass"

echo ""
echo "========================================================"
echo " transformer fragment ${FRAGMENT} pipeline complete"
echo "   build_${FRAGMENT}/step1_normalized.mlir"
echo "   build_${FRAGMENT}/step2_kernelized.mlir"
echo "   build_${FRAGMENT}/step3_scheduled.mlir"
echo "   build_${FRAGMENT}/step4_realized.mlir"
echo "   build_${FRAGMENT}/step5_ascendc.mlir"
echo "   build_${FRAGMENT}/step6_parallelized.mlir"
echo "   build_${FRAGMENT}/step7_kernel_ir.mlir"
echo "   build_${FRAGMENT}/step8_cann.mlir"
echo "   build_${FRAGMENT}/step9_kernel.cpp"
echo "   build_${FRAGMENT}/phase5_tiling_space.json"
echo "   build_${FRAGMENT}/phase5_runtime_manifest.json"
echo "   build_${FRAGMENT}/host_tiling.cpp"
echo "   build_${FRAGMENT}/artifact"
echo "========================================================"
