#!/bin/bash
# ============================================================
# 2D reduce-sum (leading axis) 端到端流水线 Demo — FullLoad + RA layout
#
# 用法:
#   source examples/env.sh
#   bash examples/leading-reduce-e2e/run.sh [--log]
#
# 计算图:
#   out[d1] = sum_{d0}( x[d0, d1] )
#
# Shape:  x[D0, D1] f32  →  out[D1] f32   (axis=0 reduce)
#
# Tiling:
#   D0 (= R)         全载到 UB —— TilePlanGen 选 FullLoad (afir.reduce_template).
#   D1 (= 并行)      分核 (每核 XBLOCK 行), 核内按 XBLOCK_SUB 切片.
#   reduce_sum_2d_l2 layout = RA (= 1), 因为 reduce 在 leading.
#
# block_dim = ceil(D1 / XBLOCK).
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

D0=8; D1=1024
XBLOCK=64; XBLOCK_SUB=16
BLOCK_DIM=$(( (D1 + XBLOCK - 1) / XBLOCK ))

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done
log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " leading-reduce E2E (RA + FullLoad): codegen → sim"
echo "========================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --d0 "$D0" --d1 "$D1"

echo ""
echo "==================== [STAGE 1] MLIR → AscendC C++ ===================="
"$AFIR_OPT" "$DIR/leading_reduce.mlir" --auto-fuse-codegen \
  -o "$DIR/leading_reduce_kernel.mlir" 2>&1
log "  ✓ MLIR codegen OK → leading_reduce_kernel.mlir"

# Lock that we actually landed on FullLoad + RA — protects this example from
# silent regression of the picker or the AR/RA detector.
grep -q 'afir.reduce_template = "FullLoad"' "$DIR/leading_reduce_kernel.mlir" \
  || { echo "ERROR: expected FullLoad template, picker drifted"; exit 1; }
grep -q 'reduce_sum_2d_l2.*layout = 1' "$DIR/leading_reduce_kernel.mlir" \
  || { echo "ERROR: expected RA layout (= 1), got AR or none"; exit 1; }
echo "  ✓ Template & layout assertions OK (FullLoad + RA)"

"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/leading_reduce_kernel.mlir" \
  -o "$DIR/leading_reduce_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
echo "  ✓ Translate OK → leading_reduce_kernel.cpp + tiling_space.json"

"$PYTHON" - <<PYEOF
import json, pathlib
p = pathlib.Path("$DIR/tiling_space.json")
ts = json.loads(p.read_text())
for param in ts["tiling_params"]:
    if param["name"] == "XBLOCK":
        param["values"] = [$XBLOCK]
    elif param["name"] == "XBLOCK_SUB":
        param["values"] = [$XBLOCK_SUB]
p.write_text(json.dumps(ts, indent=2))
print("  ✓ tiling_space.json patched (XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB)")
PYEOF

echo ""
echo "==================== [STAGE 2] runtime-session compile ===================="
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output.npy"

"$RUNTIME_SESSION" \
  --kernel "$DIR/leading_reduce_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name leading_reduce__v0 \
  2>&1
echo "  ✓ Compile OK → $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 3] Simulator Run + Verify ===================="
log "  XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB, block_dim=$BLOCK_DIM"
log "  shape: x=${D0}x${D1}, out=${D1}"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TILING_PARAMS+=",dim_arg0_0=${D0},dim_arg0_1=${D1}"

cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "x", "path": "${DIR}/x.npy" }
  ],
  "outputs": [
    { "name": "out", "path": "${ACTUAL_OUTPUT}" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "${DIR}/expected.npy" }
  ],
  "tiling": {
    "schema": "${DIR}/tiling_space.json",
    "params": "${TILING_PARAMS}"
  },
  "block_dim": ${BLOCK_DIM},
  "workspace_size": 16777216,
  "profiling": false,
  "atol": 1e-5,
  "rtol": 1e-5
}
EOF

"$RUNTIME_SESSION" \
  --run-manifest "$RUN_MANIFEST" \
  --run >"$VALIDATION_LOG" 2>&1
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]' "$VALIDATION_LOG" || true
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"

echo ""
echo "========================================================"
echo " Done. session.validation=pass"
echo "========================================================"
