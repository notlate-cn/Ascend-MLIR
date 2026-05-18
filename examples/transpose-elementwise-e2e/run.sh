#!/bin/bash
# ============================================================
# Tail-axis transpose (eliminate template) end-to-end Demo
#
# 用法:
#   source examples/env.sh
#   bash examples/transpose-elementwise-e2e/run.sh [--log]
#
# 计算图:
#   out = relu( transpose(x, [1,0]) )
#
# 形状: x[M, N] f16  ->  out[N, M] f16   (M=16, N=32)
#
# --linalg-fuse-elementwise-ops 把 named linalg.transpose 吸进 relu 的 indexing
# map (operand map (d0,d1)->(d1,d0)),所以 codegen 里没有 transpose 节点
# (= 消除模板)。片上:转置后的 x operand tile 是 x 的行带 stride 的 subview
# → 每行一条 DataCopy 装进 VECIN,再用 AscendC::Transpose 重排成输出布局后做
# relu。AscendC::Transpose 的基础 16x16 形态只支持 16-bit 数据,故用 f16 +
# 方阵内层 tile (XBLOCK_SUB == 16;另一根迭代轴整维全载 — §3.4)。
# block_dim = ceil(N / XBLOCK)。
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

M=16; N=32
XBLOCK=16; XBLOCK_SUB=16
BLOCK_DIM=$(( (N + XBLOCK - 1) / XBLOCK ))

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done
log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " transpose-elementwise E2E: vector-plan-codegen → runtime-session → sim"
echo "========================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --m "$M" --n "$N"

echo ""
echo "==================== [STAGE 1] MLIR → AscendC C++ ===================="
"$AFIR_OPT" "$DIR/transpose_relu.mlir" --vector-plan-codegen \
  -o "$DIR/transpose_relu_kernel.mlir" 2>&1
log "  ✓ MLIR codegen OK → transpose_relu_kernel.mlir"

"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/transpose_relu_kernel.mlir" \
  -o "$DIR/transpose_relu_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
echo "  ✓ Translate OK → transpose_relu_kernel.cpp + tiling_space.json"

"$PYTHON" - <<PYEOF
import json, pathlib
p = pathlib.Path("$DIR/tiling_space.json")
ts = json.loads(p.read_text())
pin = {"XBLOCK": $XBLOCK, "XBLOCK_SUB": $XBLOCK_SUB}
for param in ts["tiling_params"]:
    if param["name"] in pin:
        param["values"] = [pin[param["name"]]]
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
  --kernel "$DIR/transpose_relu_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name transpose_relu__v0 \
  2>&1
echo "  ✓ Compile OK → $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 3] Simulator Run + Verify ===================="
log "  XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB, block_dim=$BLOCK_DIM"
log "  shape: x=${M}x${N}, out=${N}x${M}"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"

cat > "$RUN_MANIFEST" <<MANIFEST
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
MANIFEST

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
