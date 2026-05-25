#!/usr/bin/env bash
# New Ascend mainline pipeline for split + relu/broadcast elementwise + concat.
#
# This path starts from the linalg/tensor graph in step0_input.mlir and runs
# through Normalize -> Kernelize -> Schedule -> Realize -> Phase 5 codegen.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../mainline-target-env.sh
source "$DIR/../mainline-target-env.sh"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

M=64
N=80
TILE_M=32
BLOCK_DIM=""
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
    --n)
      require_arg "$1" "${2:-}"
      N="$2"
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
require_positive_int "N" "$N"
if (( M % 2 != 0 )); then
  echo "unsupported shape: M must be even for this split example" >&2
  exit 2
fi

HM=$((M / 2))
if [[ -z "$BLOCK_DIM" ]]; then
  BLOCK_DIM=$(((HM + TILE_M - 1) / TILE_M))
fi
require_positive_int "BLOCK_DIM" "$BLOCK_DIM"

log() {
  if $VERBOSE; then
    echo "$@"
  fi
}

BUILD_DIR="$DIR/build_mainline"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

PHASE5_TILING_SPACE="$BUILD_DIR/phase5_tiling_space.json"
PHASE5_RUNTIME_MANIFEST="$BUILD_DIR/phase5_runtime_manifest.json"
PHASE5_HOST_TILING="$BUILD_DIR/host_tiling.cpp"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output_actual.npy"
EXPECTED_OUTPUT="$BUILD_DIR/output.npy"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

echo "========================================================"
echo " split + relu/broadcast elementwise + concat Ascend mainline pipeline"
echo "========================================================"
echo "shape.M=$M"
echo "shape.N=$N"
echo "shape.HM=$HM"
echo "block_dim=$BLOCK_DIM"

echo ""
echo "==================== [STAGE 1] Linalg fusion ===================="
"$AFIR_OPT" "$DIR/step0_input.mlir" \
  --linalg-fuse-elementwise-ops \
  --canonicalize \
  --cse \
  -o "$BUILD_DIR/step1_fused.mlir"
log "  output: $BUILD_DIR/step1_fused.mlir"

echo ""
echo "==================== [STAGE 2] Ascend normalize ===================="
"$AFIR_OPT" "$BUILD_DIR/step1_fused.mlir" \
  --ascend-normalize \
  -o "$BUILD_DIR/step2_normalized.mlir"
log "  output: $BUILD_DIR/step2_normalized.mlir"

echo ""
echo "==================== [STAGE 3] Ascend kernelize ===================="
"$AFIR_OPT" "$BUILD_DIR/step2_normalized.mlir" \
  --ascend-kernelize \
  -o "$BUILD_DIR/step3_kernelized.mlir"
log "  output: $BUILD_DIR/step3_kernelized.mlir"

echo ""
echo "==================== [STAGE 4] Ascend schedule ===================="
"$AFIR_OPT" "$BUILD_DIR/step3_kernelized.mlir" \
  --ascend-schedule="target-tile-policy=target-aware cann-root=${CANN_ROOT} soc=${SOC}" \
  -o "$BUILD_DIR/step4_scheduled.mlir"
log "  output: $BUILD_DIR/step4_scheduled.mlir"

echo ""
echo "==================== [STAGE 5] Ascend realize ===================="
"$AFIR_OPT" "$BUILD_DIR/step4_scheduled.mlir" \
  --ascend-realize='materialization-mode=memory-space-annotate' \
  -o "$BUILD_DIR/step5_realized.mlir"
log "  output: $BUILD_DIR/step5_realized.mlir"

echo ""
echo "==================== [STAGE 6] Ascend compute lower ===================="
"$AFIR_OPT" "$BUILD_DIR/step5_realized.mlir" \
  --ascend-compute-lower \
  -o "$BUILD_DIR/step6_ascendc.mlir"
log "  output: $BUILD_DIR/step6_ascendc.mlir"

echo ""
echo "==================== [STAGE 7] Ascend parallelize ===================="
"$AFIR_OPT" "$BUILD_DIR/step6_ascendc.mlir" \
  --ascend-parallelize \
  -o "$BUILD_DIR/step7_parallelized.mlir"
log "  output: $BUILD_DIR/step7_parallelized.mlir"

echo ""
echo "==================== [STAGE 8] Ascend prepare for emit ===================="
"$AFIR_OPT" "$BUILD_DIR/step7_parallelized.mlir" \
  --ascend-prepare-for-emit \
  -o "$BUILD_DIR/step8_kernel_ir.mlir"
log "  output: $BUILD_DIR/step8_kernel_ir.mlir"

echo ""
echo "==================== [STAGE 9] CANN signature ===================="
"$AFIR_OPT" "$BUILD_DIR/step8_kernel_ir.mlir" \
  --ascend-canonicalize-cann-signature \
  --canonicalize \
  --cse \
  -o "$BUILD_DIR/step9_cann.mlir"
log "  output: $BUILD_DIR/step9_cann.mlir"

echo ""
echo "==================== [STAGE 10] CANN codegen ===================="
"$AFIR_TRANSLATE" -mlir-to-cann "$BUILD_DIR/step9_cann.mlir" \
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
echo "==================== [STAGE 11] Generate data ===================="
"$PYTHON" "$DIR/gen_data.py" --M "$M" --N "$N" --outdir "$BUILD_DIR"
log "  output: input_a.npy bias0.npy bias1.npy scale0.npy scale1.npy output.npy"

echo ""
echo "==================== [STAGE 12] runtime-session compile ===================="
"$RUNTIME_SESSION" \
  --kernel "$BUILD_DIR/step10_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name ewop_broadcast_split
log "  output: $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 13] runtime-session sim ===================="
TILING_PARAMS="$("$PYTHON" - "$PHASE5_TILING_SPACE" "$M" "$N" "$HM" <<'PY'
import json
import sys

schema_path, m, n, hm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(schema_path, "r", encoding="utf-8") as f:
    root = json.load(f)

shape_values = {
    "arg0_dim0": m,
    "arg0_dim1": n,
    "arg1_dim0": hm,
    "arg2_dim0": hm,
    "arg3_dim0": n,
    "arg4_dim0": n,
}

params = []
for field in root.get("tiling_params", []):
    name = field["name"]
    if field.get("fixed"):
        shape_key = field.get("shape_key")
        if shape_key not in shape_values:
            raise SystemExit(f"unsupported fixed shape key: {shape_key}")
        value = shape_values[shape_key]
    elif "fixed_value" in field:
        value = int(field["fixed_value"])
    elif field.get("values"):
        value = int(field["values"][0])
    else:
        raise SystemExit(f"unsupported free tiling field: {name}")
    params.append(f"{name}={value}")

print(",".join(params))
PY
)"

cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "input_a", "path": "${BUILD_DIR}/input_a.npy" },
    { "name": "bias0", "path": "${BUILD_DIR}/bias0.npy" },
    { "name": "bias1", "path": "${BUILD_DIR}/bias1.npy" },
    { "name": "scale0", "path": "${BUILD_DIR}/scale0.npy" },
    { "name": "scale1", "path": "${BUILD_DIR}/scale1.npy" }
  ],
  "outputs": [
    { "name": "out", "path": "${ACTUAL_OUTPUT}" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "${EXPECTED_OUTPUT}" }
  ],
  "tiling": {
    "schema": "${PHASE5_TILING_SPACE}",
    "params": "${TILING_PARAMS}"
  },
  "block_dim": ${BLOCK_DIM},
  "workspace_size": 16777216,
  "profiling": true,
  "atol": 1e-2,
  "rtol": 1e-2
}
EOF

"$RUNTIME_SESSION" \
  --run-manifest "$RUN_MANIFEST" \
  --run >"$VALIDATION_LOG" 2>&1
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]' \
  "$VALIDATION_LOG" || true
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"

echo ""
echo "========================================================"
echo " mainline pipeline complete"
echo "   build_mainline/step1_fused.mlir"
echo "   build_mainline/step2_normalized.mlir"
echo "   build_mainline/step3_kernelized.mlir"
echo "   build_mainline/step4_scheduled.mlir"
echo "   build_mainline/step5_realized.mlir"
echo "   build_mainline/step6_ascendc.mlir"
echo "   build_mainline/step7_parallelized.mlir"
echo "   build_mainline/step8_kernel_ir.mlir"
echo "   build_mainline/step9_cann.mlir"
echo "   build_mainline/step10_kernel.cpp"
echo "   build_mainline/phase5_tiling_space.json"
echo "   build_mainline/phase5_runtime_manifest.json"
echo "   build_mainline/host_tiling.cpp"
echo "   build_mainline/artifact"
echo "========================================================"
