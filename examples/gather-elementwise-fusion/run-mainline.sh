#!/usr/bin/env bash
# New Ascend mainline pipeline for gather + relu + bias add.
#
# This script starts from the linalg/tensor graph in step0_input.mlir and uses
# the Ascend mainline passes through CANN codegen and runtime-session sim.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../mainline-target-env.sh
source "$DIR/../mainline-target-env.sh"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
ASCEND_MLIR_TRANSLATE="${ASCEND_MLIR_TRANSLATE:-${AFIR_TRANSLATE:-ascend-mlir-translate}}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

M=16
N=640
K=128
SEED=42
BLOCK_DIM=1
INDEX_TAIL_GUARD=16
INDEX_HIGH=""
SOC="${SOC_VERSION:-Ascend910B1}"
VERBOSE=false
PREPARE_RUNTIME_ARTIFACTS=false

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
    --prepare-runtime-artifacts)
      PREPARE_RUNTIME_ARTIFACTS=true
      shift
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
    --k)
      require_arg "$1" "${2:-}"
      K="$2"
      shift 2
      ;;
    --seed)
      require_arg "$1" "${2:-}"
      SEED="$2"
      shift 2
      ;;
    --index-high)
      require_arg "$1" "${2:-}"
      INDEX_HIGH="$2"
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
require_positive_int "K" "$K"
require_positive_int "BLOCK_DIM" "$BLOCK_DIM"
if (( K > N )); then
  echo "unsupported shape: K must be <= N" >&2
  exit 2
fi
if [[ -n "$INDEX_HIGH" ]]; then
  require_positive_int "INDEX_HIGH" "$INDEX_HIGH"
  EFFECTIVE_INDEX_HIGH="$INDEX_HIGH"
else
  EFFECTIVE_INDEX_HIGH=$((N - INDEX_TAIL_GUARD))
fi
if (( EFFECTIVE_INDEX_HIGH < 1 )); then
  echo "unsupported shape: index_high must be >= 1" >&2
  exit 2
fi
if (( EFFECTIVE_INDEX_HIGH > N )); then
  echo "unsupported shape: index_high must be <= N" >&2
  exit 2
fi
if (( K > EFFECTIVE_INDEX_HIGH )); then
  echo "unsupported shape: K must be <= index_high (${EFFECTIVE_INDEX_HIGH})" >&2
  exit 2
fi

log() {
  if $VERBOSE; then
    echo "$@"
  fi
}

BUILD_DIR="$DIR/build_mainline"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

PHASE5_TILING_SPACE="$BUILD_DIR/phase5_tiling_space.json"
PHASE5_ARTIFACT_MANIFEST="$BUILD_DIR/phase5_artifact_manifest.json"
PHASE5_HOST_TILING="$BUILD_DIR/host_tiling.cpp"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output_actual.npy"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

echo "========================================================"
echo " gather + relu + bias add Ascend mainline pipeline"
echo "========================================================"
echo "shape.M=$M"
echo "shape.N=$N"
echo "shape.K=$K"
echo "index_high=$EFFECTIVE_INDEX_HIGH"
echo "block_dim=$BLOCK_DIM"

echo ""
echo "==================== [STAGE 3] Ascend normalize ===================="
"$AFIR_OPT" "$DIR/step0_input.mlir" \
  --ascend-normalize \
  -o "$BUILD_DIR/step3_normalized.mlir"
log "  output: $BUILD_DIR/step3_normalized.mlir"

echo ""
echo "==================== [STAGE 4] Ascend kernelize ===================="
"$AFIR_OPT" "$BUILD_DIR/step3_normalized.mlir" \
  --ascend-kernelize \
  -o "$BUILD_DIR/step4_kernelized.mlir"
log "  output: $BUILD_DIR/step4_kernelized.mlir"

echo ""
echo "==================== [STAGE 5] Ascend schedule ===================="
"$AFIR_OPT" "$BUILD_DIR/step4_kernelized.mlir" \
  --ascend-schedule="target-tile-policy=target-aware cann-root=${CANN_ROOT} soc=${SOC}" \
  -o "$BUILD_DIR/step5_scheduled.mlir"
log "  output: $BUILD_DIR/step5_scheduled.mlir"

echo ""
echo "==================== [STAGE 6] Ascend realize ===================="
"$AFIR_OPT" "$BUILD_DIR/step5_scheduled.mlir" \
  --ascend-realize='materialization-mode=memory-space-annotate' \
  -o "$BUILD_DIR/step6_realized.mlir"
log "  output: $BUILD_DIR/step6_realized.mlir"

echo ""
echo "==================== [STAGE 7] Ascend compute lower ===================="
"$AFIR_OPT" "$BUILD_DIR/step6_realized.mlir" \
  --ascend-compute-lower \
  -o "$BUILD_DIR/step7_ascendc.mlir"
log "  output: $BUILD_DIR/step7_ascendc.mlir"

echo ""
echo "==================== [STAGE 8] Ascend parallelize ===================="
"$AFIR_OPT" "$BUILD_DIR/step7_ascendc.mlir" \
  --ascend-parallelize \
  -o "$BUILD_DIR/step8_parallelized.mlir"
log "  output: $BUILD_DIR/step8_parallelized.mlir"

echo ""
echo "==================== [STAGE 9] Ascend prepare for emit ===================="
"$AFIR_OPT" "$BUILD_DIR/step8_parallelized.mlir" \
  --ascend-prepare-for-emit \
  -o "$BUILD_DIR/step9_kernel_ir.mlir"
log "  output: $BUILD_DIR/step9_kernel_ir.mlir"

echo ""
echo "==================== [STAGE 10] CANN signature ===================="
"$AFIR_OPT" "$BUILD_DIR/step9_kernel_ir.mlir" \
  --ascend-canonicalize-cann-signature \
  --canonicalize \
  --cse \
  -o "$BUILD_DIR/step10_cann.mlir"
log "  output: $BUILD_DIR/step10_cann.mlir"

echo ""
echo "==================== [STAGE 11] CANN codegen ===================="
"$ASCEND_MLIR_TRANSLATE" -mlir-to-cann "$BUILD_DIR/step10_cann.mlir" \
  --tiling-space-out="$PHASE5_TILING_SPACE" \
  --artifact-manifest-out="$PHASE5_ARTIFACT_MANIFEST" \
  --host-tiling-out="$PHASE5_HOST_TILING" \
  --cann-soc="$SOC" \
  -o "$BUILD_DIR/step11_kernel.cpp"
test -s "$BUILD_DIR/step11_kernel.cpp"
test -s "$PHASE5_TILING_SPACE"
test -s "$PHASE5_ARTIFACT_MANIFEST"
test -s "$PHASE5_HOST_TILING"
log "  output: $BUILD_DIR/step11_kernel.cpp"

echo ""
echo "==================== [STAGE 12] Generate data ===================="
"$PYTHON" "$DIR/gen_data.py" --m "$M" --n "$N" --k "$K" \
  --index-high "$EFFECTIVE_INDEX_HIGH" --seed "$SEED" \
  --out-dir "$BUILD_DIR"
log "  output: input_data.npy input_indices.npy input_bias.npy output_out.npy"

echo ""
echo "==================== [STAGE 13] runtime-session compile ===================="
"$RUNTIME_SESSION" \
  --kernel "$BUILD_DIR/step11_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name relu_index_select_add
log "  output: $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 14] runtime-session sim ===================="
TILING_PARAMS="$("$PYTHON" - "$PHASE5_TILING_SPACE" "$M" "$N" "$K" <<'PY'
import json
import sys

schema_path, m, n, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(schema_path, "r", encoding="utf-8") as f:
    root = json.load(f)

shape_values = {
    "arg0_dim0": m,
    "arg0_dim1": n,
    "arg1_dim0": k,
    "arg2_dim0": k,
}

schedule_defaults = {}
for kernel in root.get("kernels", []):
    for entry in kernel.get("scheduleEntries", []):
        for tile_param in entry.get("tilingParams", {}).get("tile_params", []):
            name = tile_param.get("name")
            if name and "default" in tile_param:
                schedule_defaults.setdefault(name, int(tile_param["default"]))
for entry in root.get("scheduleEntries", []):
    for tile_param in entry.get("tilingParams", {}).get("tile_params", []):
        name = tile_param.get("name")
        if name and "default" in tile_param:
            schedule_defaults.setdefault(name, int(tile_param["default"]))

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
    elif "default" in field:
        value = int(field["default"])
    elif name in schedule_defaults:
        value = schedule_defaults[name]
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
    { "name": "arg0", "path": "${BUILD_DIR}/input_data.npy" },
    { "name": "arg1", "path": "${BUILD_DIR}/input_indices.npy" },
    { "name": "arg2", "path": "${BUILD_DIR}/input_bias.npy" }
  ],
  "outputs": [
    { "name": "out", "path": "${ACTUAL_OUTPUT}" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "${BUILD_DIR}/output_out.npy" }
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

if $PREPARE_RUNTIME_ARTIFACTS; then
  echo "gather-elementwise-fusion: prepared runtime artifacts"
  exit 0
fi

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
echo "   build_mainline/step3_normalized.mlir"
echo "   build_mainline/step4_kernelized.mlir"
echo "   build_mainline/step5_scheduled.mlir"
echo "   build_mainline/step6_realized.mlir"
echo "   build_mainline/step7_ascendc.mlir"
echo "   build_mainline/step8_parallelized.mlir"
echo "   build_mainline/step9_kernel_ir.mlir"
echo "   build_mainline/step10_cann.mlir"
echo "   build_mainline/step11_kernel.cpp"
echo "   build_mainline/phase5_tiling_space.json"
echo "   build_mainline/phase5_artifact_manifest.json"
echo "   build_mainline/host_tiling.cpp"
echo "   build_mainline/artifact"
echo "========================================================"
