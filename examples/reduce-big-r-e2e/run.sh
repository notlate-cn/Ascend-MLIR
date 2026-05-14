#!/bin/bash
# ============================================================
# 2D reduce-sum with a too-large reduction axis — 端到端编译流水线 Demo
#   使用 --vector-plan-codegen（自动启用 RBLOCK reduction-split）
#
# 用法：
#   source examples/env.sh
#   bash examples/reduce-big-r-e2e/run.sh [--log]
#
# 计算图：
#   out[a] = sum_{r}( x[a, r] )
#
# 形状: x[A, R] f32  →  out[A] f32   (axis=1 reduce, R=65536 太大需拆 RBLOCK)
#
# 验证 RBLOCK reduction-split codegen 的两个关键点：
#   1. 每个 chunk 的 GM→UB 搬运是二维带 stride 的 subview
#      （[XBLOCK_SUB 行 × RBLOCK_0 列]，行 stride = R），发成每行一次 DataCopy；
#   2. reduce 用的所有 UB scratch（chunk 累加器 / VECIN / partial / workspace）
#      在 R 块循环外只分配一次（否则 ~R/RBLOCK_0 次 InitBuffer 撑爆 UB）。
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

A=8; R=65536

# Iteration-space layout: A is parallel (block-dispatched, XBLOCK rows/core,
# inner step XBLOCK_SUB), R is reduction tile-split into RBLOCK_0 chunks.
#   block_dim = ceil(A / XBLOCK);  on-chip tile = XBLOCK_SUB · RBLOCK_0 · 4 B
# XBLOCK_SUB must divide XBLOCK (tail-peel precondition).
XBLOCK=4
XBLOCK_SUB=4
RBLOCK_0=512
BLOCK_DIM=$(( (A + XBLOCK - 1) / XBLOCK ))

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done
log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " reduce-big-r E2E: vector-plan-codegen → runtime-session → sim"
echo "========================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --a "$A" --r "$R"

echo ""
echo "==================== [STAGE 1] MLIR → AscendC C++ ===================="
"$AFIR_OPT" "$DIR/reduce_big_r.mlir" --vector-plan-codegen \
  -o "$DIR/reduce_big_r_kernel.mlir" 2>&1
log "  ✓ MLIR codegen OK → reduce_big_r_kernel.mlir"

"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/reduce_big_r_kernel.mlir" \
  -o "$DIR/reduce_big_r_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
echo "  ✓ Translate OK → reduce_big_r_kernel.cpp + tiling_space.json"

"$PYTHON" - <<PYEOF
import json, pathlib
p = pathlib.Path("$DIR/tiling_space.json")
ts = json.loads(p.read_text())
for param in ts["tiling_params"]:
    if param["name"] == "XBLOCK":
        param["values"] = [$XBLOCK]
    elif param["name"] == "XBLOCK_SUB":
        param["values"] = [$XBLOCK_SUB]
    elif param["name"] == "RBLOCK_0":
        param["values"] = [$RBLOCK_0]
p.write_text(json.dumps(ts, indent=2))
print("  ✓ tiling_space.json patched (XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB, RBLOCK_0=$RBLOCK_0)")
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
  --kernel "$DIR/reduce_big_r_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name reduce_big_r__v0 \
  2>&1
echo "  ✓ Compile OK → $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 3] Simulator Run + Verify ===================="
log "  XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB, RBLOCK_0=$RBLOCK_0, block_dim=$BLOCK_DIM"
log "  shape: x=${A}x${R}, out=${A}"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

# Tiling params: XBLOCK + XBLOCK_SUB + RBLOCK_0 tunables, plus the R-extent dim.
TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB},RBLOCK_0=${RBLOCK_0}"
TILING_PARAMS+=",dim_arg0_1=${R}"

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
echo "   reduce_big_r_kernel.mlir   → vector-plan-codegen 后 MLIR"
echo "   reduce_big_r_kernel.cpp    → AscendC C++ kernel"
echo "   tiling_space.json          → 自动生成+patched tiling schema"
echo "   build_e2e/artifact         → runtime-session 编译产物"
echo "   build_e2e/output.npy       → 仿真输出（session.validation=pass）"
echo "========================================================"
