#!/usr/bin/env bash
# Three-kernel Ascend mainline example.
#
# The source module contains two high-level linalg functions. The pipeline
# lowers them into two CANN kernel entries, compiles each entry into a runtime
# artifact, and executes them as a runtime-session DAG:
#   kernel_a: a + b -> mid
#   kernel_b: mid * c -> mid1
#   kernel_c: mid1 + d -> out

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../mainline-target-env.sh
source "$DIR/../mainline-target-env.sh"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

N=64
SEED=42
BLOCK_DIM=20
SOC="${SOC_VERSION:-Ascend910B1}"
VERBOSE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n)
      N="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --block-dim)
      BLOCK_DIM="$2"
      shift 2
      ;;
    --soc)
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
ARTIFACT_A="$BUILD_DIR/artifact_kernel_a"
ARTIFACT_B="$BUILD_DIR/artifact_kernel_b"
ARTIFACT_C="$BUILD_DIR/artifact_kernel_c"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output.npy"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

echo "========================================================"
echo " three-kernel DAG Ascend mainline pipeline"
echo "========================================================"
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
"$AFIR_OPT" "$BUILD_DIR/step4_realized.mlir" \
  --ascend-compute-lower \
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
  -o "$BUILD_DIR/step8_cann.mlir"
log "  output: $BUILD_DIR/step8_cann.mlir"

echo ""
echo "==================== [STAGE 9] CANN codegen ===================="
"$AFIR_TRANSLATE" -mlir-to-cann "$BUILD_DIR/step8_cann.mlir" \
  --tiling-space-out="$PHASE5_TILING_SPACE" \
  --artifact-manifest-out="$PHASE5_ARTIFACT_MANIFEST" \
  --cann-soc="$SOC" \
  -o "$BUILD_DIR/step9_kernel.cpp"
test -s "$BUILD_DIR/step9_kernel.cpp"
test -s "$PHASE5_TILING_SPACE"
test -s "$PHASE5_ARTIFACT_MANIFEST"

"$PYTHON" - "$PHASE5_ARTIFACT_MANIFEST" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
with open(manifest_path, "r", encoding="utf-8") as f:
    root = json.load(f)

entries = root.get("kernel_entries", [])
nodes = root.get("kernelGraph", {}).get("nodes", [])
names = [entry.get("kernel_id") for entry in entries]
if names != ["kernel_a", "kernel_b", "kernel_c"]:
    raise SystemExit(f"expected kernel_entries for kernel_a/kernel_b/kernel_c, got {names}")
if len(nodes) != 3:
    raise SystemExit(f"expected three kernelGraph nodes, got {len(nodes)}")
print("manifest.kernel_entries=3")
print("manifest.kernelGraph.nodes=3")
PY
log "  output: $BUILD_DIR/step9_kernel.cpp"

echo ""
echo "==================== [STAGE 10] Generate data ===================="
"$PYTHON" "$DIR/gen_data.py" --n "$N" --seed "$SEED" --out-dir "$BUILD_DIR"

echo ""
echo "==================== [STAGE 11] runtime-session compile ===================="
"$RUNTIME_SESSION" \
  --kernel "$BUILD_DIR/step9_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_A" \
  --name kernel_a
"$RUNTIME_SESSION" \
  --kernel "$BUILD_DIR/step9_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_B" \
  --name kernel_b
"$RUNTIME_SESSION" \
  --kernel "$BUILD_DIR/step9_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_C" \
  --name kernel_c

echo ""
echo "==================== [STAGE 12] runtime-session DAG sim ===================="
TILING_PARAMS="$("$PYTHON" - "$PHASE5_TILING_SPACE" "$N" <<'PY'
import json
import sys

schema_path, n = sys.argv[1], int(sys.argv[2])
with open(schema_path, "r", encoding="utf-8") as f:
    root = json.load(f)

shape_values = {
    "arg0_dim0": n,
    "arg1_dim0": n,
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
  "backend": "sim",
  "tasks": [
    {
      "task_id": "kernel_a",
      "artifact_root": "${ARTIFACT_A}",
      "inputs": [
        { "name": "a", "path": "${BUILD_DIR}/input_a.npy" },
        { "name": "b", "path": "${BUILD_DIR}/input_b.npy" }
      ],
      "outputs": [
        { "name": "mid", "shape": [${N}], "dtype": "f16" }
      ],
      "tiling": {
        "schema": "${PHASE5_TILING_SPACE}",
        "params": "${TILING_PARAMS}"
      },
      "block_dim": ${BLOCK_DIM},
      "workspace_size": 16777216,
      "profiling": true
    },
    {
      "task_id": "kernel_b",
      "dependencies": ["kernel_a"],
      "artifact_root": "${ARTIFACT_B}",
      "inputs": [
        { "name": "mid", "source": "task_output", "upstream_task": "kernel_a", "upstream_output": "mid" },
        { "name": "c", "path": "${BUILD_DIR}/input_c.npy" }
      ],
      "outputs": [
        { "name": "mid1", "shape": [${N}], "dtype": "f16" }
      ],
      "tiling": {
        "schema": "${PHASE5_TILING_SPACE}",
        "params": "${TILING_PARAMS}"
      },
      "block_dim": ${BLOCK_DIM},
      "workspace_size": 16777216,
      "profiling": true
    },
    {
      "task_id": "kernel_c",
      "dependencies": ["kernel_b"],
      "artifact_root": "${ARTIFACT_C}",
      "inputs": [
        { "name": "mid1", "source": "task_output", "upstream_task": "kernel_b", "upstream_output": "mid1" },
        { "name": "d", "path": "${BUILD_DIR}/input_d.npy" }
      ],
      "outputs": [
        { "name": "out", "path": "${ACTUAL_OUTPUT}", "shape": [${N}], "dtype": "f16" }
      ],
      "tiling": {
        "schema": "${PHASE5_TILING_SPACE}",
        "params": "${TILING_PARAMS}"
      },
      "block_dim": ${BLOCK_DIM},
      "workspace_size": 16777216,
      "profiling": true
    }
  ]
}
EOF

"$RUNTIME_SESSION" \
  --run-manifest "$RUN_MANIFEST" \
  --run >"$VALIDATION_LOG" 2>&1
"$PYTHON" - "$ACTUAL_OUTPUT" "$BUILD_DIR/output_expected.npy" >>"$VALIDATION_LOG" <<'PY'
import sys

import numpy as np

actual = np.load(sys.argv[1])
expected = np.load(sys.argv[2])
diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
max_abs = float(diff.max()) if diff.size else 0.0
mean_abs = float(diff.mean()) if diff.size else 0.0
print(f"max_abs_diff={max_abs:.6e}")
print(f"mean_abs_diff={mean_abs:.6e}")
if not np.allclose(actual, expected, atol=1e-2, rtol=1e-2):
    raise SystemExit(1)
print("session.validation=pass")
PY
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]' \
  "$VALIDATION_LOG" || true
grep -q '^session.plan.tasks=3$' "$VALIDATION_LOG"
grep -q '^session.plan\[0\]=kernel_a$' "$VALIDATION_LOG"
grep -q '^session.plan\[1\]=kernel_b$' "$VALIDATION_LOG"
grep -q '^session.plan\[2\]=kernel_c$' "$VALIDATION_LOG"
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"
grep -q '^session.profile.count=3$' "$VALIDATION_LOG"
grep -q '^session.runtime.counter.planned_task_count=3$' "$VALIDATION_LOG"
grep -q '^session.runtime.counter.serialized_launch_count=3$' "$VALIDATION_LOG"

echo ""
echo "========================================================"
echo " three-kernel DAG pipeline complete"
echo "   build_mainline/step1_normalized.mlir"
echo "   build_mainline/step2_kernelized.mlir"
echo "   build_mainline/step3_scheduled.mlir"
echo "   build_mainline/step4_realized.mlir"
echo "   build_mainline/step5_ascendc.mlir"
echo "   build_mainline/step6_parallelized.mlir"
echo "   build_mainline/step7_kernel_ir.mlir"
echo "   build_mainline/step8_cann.mlir"
echo "   build_mainline/step9_kernel.cpp"
echo "   build_mainline/phase5_tiling_space.json"
echo "   build_mainline/phase5_artifact_manifest.json"
echo "   build_mainline/artifact_kernel_a"
echo "   build_mainline/artifact_kernel_b"
echo "   build_mainline/artifact_kernel_c"
echo "   build_mainline/output.npy"
echo "========================================================"
