#!/bin/bash
# ============================================================
# Multi-use transpose ("preserve" template) end-to-end Demo
#
# 用法:
#   source examples/env.sh
#   bash examples/transpose-preserve-e2e/run.sh [--log]
#
# 计算图:
#   t      = transpose(x, [1,0])
#   out_a  = relu(t)
#   out_b  = t * 2
#
# 形状: x[M, N] f16  ->  out_a, out_b : [N, M] f16   (M=N=32)
#
# t 有两个消费者 → --linalg-fuse-elementwise-ops 吸不动 transpose，它留作独立
# op = 保留模板。classifyAxes(≈ AF GenTransposeTilingGroup) 把输入侧发散轴
# (d0 = 输出外维) 划成 X(自己的 inner tunable XBLOCK_X_0，永不当块轴)、输出侧
# (d1) 划成 Y(块轴)。片上(每个 16x16 tile)：transpose 把行带 stride 的 x slice
# 每行一条 DataCopy 装进 VECIN，AscendC::Transpose 重排成输出布局放在 VECCALC
# TBuf，两个消费者读同一个 TBuf(它留在片上，不外溢成 GM 参数)，再把行带 stride
# 的 out[d0_range, d1_range] tile 每行一条 DataCopy 写回 GM。f16 + 方阵 16x16
# 内层 tile (AscendC::Transpose 基础形态 16-bit / 16x16)。run.sh 钉 XBLOCK =
# XBLOCK_SUB = XBLOCK_X_0 = 16，M=32 → block_dim = ceil(32/16) = 2(多核)。
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

M=32; N=32
XBLOCK=16; XBLOCK_SUB=16; XBLOCK_X_0=16
BLOCK_DIM=$(( (M + XBLOCK - 1) / XBLOCK ))   # block axis = the Y axis (out dim1 = M)

VERBOSE=false
for arg in "$@"; do case $arg in --log) VERBOSE=true ;; esac; done
log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " transpose-preserve E2E: vector-plan-codegen → runtime-session → sim"
echo "========================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --m "$M" --n "$N"

echo ""
echo "==================== [STAGE 1] MLIR → AscendC C++ ===================="
"$AFIR_OPT" "$DIR/transpose_preserve.mlir" --vector-plan-codegen \
  -o "$DIR/transpose_preserve_kernel.mlir" 2>&1
log "  ✓ MLIR codegen OK"

"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/transpose_preserve_kernel.mlir" \
  -o "$DIR/transpose_preserve_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
echo "  ✓ Translate OK → transpose_preserve_kernel.cpp + tiling_space.json"

"$PYTHON" - <<PYEOF
import json, pathlib
p = pathlib.Path("$DIR/tiling_space.json")
ts = json.loads(p.read_text())
pin = {"XBLOCK_X_0": $XBLOCK_X_0, "XBLOCK": $XBLOCK, "XBLOCK_SUB": $XBLOCK_SUB}
for param in ts["tiling_params"]:
    if param["name"] in pin:
        param["values"] = [pin[param["name"]]]
p.write_text(json.dumps(ts, indent=2))
print("  ✓ tiling_space.json patched (XBLOCK_X_0=$XBLOCK_X_0, XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB)")
PYEOF

echo ""
echo "==================== [STAGE 2] runtime-session compile ===================="
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"

"$RUNTIME_SESSION" \
  --kernel "$DIR/transpose_preserve_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name transpose_preserve \
  2>&1
echo "  ✓ Compile OK → $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 3] Simulator Run + Verify ===================="
log "  XBLOCK_X_0=$XBLOCK_X_0, XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB, block_dim=$BLOCK_DIM"
log "  shape: x=${M}x${N}, out_a=out_b=${N}x${M}"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

TILING_PARAMS="XBLOCK_X_0=${XBLOCK_X_0},XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TILING_PARAMS+=",dim_arg0_1=${N},dim_arg4_1=${M},dim_arg5_1=${M}"

cat > "$RUN_MANIFEST" <<MANIFEST
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "x", "path": "${DIR}/x.npy" }
  ],
  "outputs": [
    { "name": "a", "path": "${BUILD_DIR}/a.npy" },
    { "name": "b", "path": "${BUILD_DIR}/b.npy" }
  ],
  "expected_outputs": [
    { "name": "a", "path": "${DIR}/expected_a.npy" },
    { "name": "b", "path": "${DIR}/expected_b.npy" }
  ],
  "tiling": {
    "schema": "${DIR}/tiling_space.json",
    "params": "${TILING_PARAMS}"
  },
  "block_dim": ${BLOCK_DIM},
  "workspace_size": 16777216,
  "profiling": false,
  "atol": 1e-2,
  "rtol": 1e-2
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
