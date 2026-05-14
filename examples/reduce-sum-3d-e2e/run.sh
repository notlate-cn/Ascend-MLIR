#!/bin/bash
# ============================================================
# 3D reduce-sum 端到端编译流水线 Demo —— 使用 --vector-plan-codegen
#
# 用法：
#   source examples/env.sh
#   bash examples/reduce-sum-3d-e2e/run.sh [--log]
#
# 计算图：
#   out[d0, d1] = sum_{d2}( x[d0, d1, d2] )
#
# 形状: x[D0, D1, D2] f32  →  out[D0, D1] f32   (axis=2 reduce)
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

D0=4; D1=8; D2=32

# Iteration-space layout after collapse([0,1], [2]):
#   rows = D0*D1 (parallel)   — outer loop sub-tile
#   cols = D2    (reduction)  — full-extent (RBLOCK_0 isn't tile-split here)
# Block dispatch: each block owns XBLOCK rows; inner loop walks them in
# XBLOCK_SUB chunks.  Pick BLOCK_DIM = ceil(rows / XBLOCK).
ROWS=$(( D0 * D1 ))
XBLOCK=8
XBLOCK_SUB=8
BLOCK_DIM=$(( (ROWS + XBLOCK - 1) / XBLOCK ))

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done
log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " reduce-sum-3D E2E: vector-plan-codegen → runtime-session → sim"
echo "========================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --d0 "$D0" --d1 "$D1" --d2 "$D2"

echo ""
echo "==================== [STAGE 1] MLIR → AscendC C++ ===================="
"$AFIR_OPT" "$DIR/reduce_sum_3d.mlir" --vector-plan-codegen \
  -o "$DIR/reduce_sum_3d_kernel.mlir" 2>&1
log "  ✓ MLIR codegen OK → reduce_sum_3d_kernel.mlir"

"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/reduce_sum_3d_kernel.mlir" \
  -o "$DIR/reduce_sum_3d_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
echo "  ✓ Translate OK → reduce_sum_3d_kernel.cpp + tiling_space.json"

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
  --kernel "$DIR/reduce_sum_3d_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name reduce_sum_3d__v0 \
  2>&1
echo "  ✓ Compile OK → $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 3] Simulator Run + Verify ===================="
log "  XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB, block_dim=$BLOCK_DIM"
log "  shape: x=${D0}x${D1}x${D2}, out=${D0}x${D1}"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

# Tiling params: XBLOCK + XBLOCK_SUB tunables, plus shape dims for both args.
TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TILING_PARAMS+=",dim_arg0_0=${D0},dim_arg0_1=${D1},dim_arg0_2=${D2}"
TILING_PARAMS+=",dim_arg1_0=${D0},dim_arg1_1=${D1}"

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
echo " Done. 生成文件："
echo "   reduce_sum_3d_kernel.mlir   → vector-plan-codegen 后 MLIR"
echo "   reduce_sum_3d_kernel.cpp    → AscendC C++ kernel"
echo "   tiling_space.json           → 自动生成+patched tiling schema"
echo "   build_e2e/artifact          → runtime-session 编译产物"
echo "   build_e2e/output.npy        → 仿真输出（session.validation=pass）"
echo "========================================================"
