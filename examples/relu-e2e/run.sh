#!/bin/bash
# ============================================================
# ReLU (max(x, 0)) 端到端编译流水线 Demo —— 使用 --auto-fuse-codegen
#
# 用法：
#   source examples/env.sh
#   bash examples/relu-e2e/run.sh [--log]
#
# 计算图：
#   output[1024] = max(input[1024], 0.0)  (element-wise ReLU, float32)
#
# 特点：
#   - 直接使用 --auto-fuse-codegen 一键走完 tile→bufferize→
#     buffer-placement→linalg-to-ascendc→parallelize→prepare-for-emit→
#     canonicalize-cann-signature 全流水
#   - TilePlanGen 自动生成 XBLOCK=128 / XBLOCK_SUB=16 tiling 策略
#   - block_dim = ceil(1024 / 128) = 8 个 AiCore 并行
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done

log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " ReLU E2E: auto-fuse-codegen → runtime-session → sim"
echo "========================================================"


# ── 生成测试数据 ──────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR"


# ── STAGE 1: auto-fuse-codegen ──────────────────────────────
echo ""
echo "==================== [STAGE 1] MLIR → AscendC C++ ===================="
log "  Pass: --auto-fuse-codegen (tile-fuse → bufferize → insert-tile-buffers"
log "        → buffer-placement → linalg-to-ascendc → parallelize → prepare-for-emit"
log "        → canonicalize-cann-signature → translate)"
"$AFIR_OPT" "$DIR/relu.mlir" --auto-fuse-codegen \
  -o "$DIR/relu_kernel.mlir" 2>&1
log "  ✓ MLIR codegen OK, output: relu_kernel.mlir"

"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/relu_kernel.mlir" \
  -o "$DIR/relu_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
echo "  ✓ Codegen OK → relu_kernel.cpp + tiling_space.json"

# Patch tiling values into the auto-generated schema (XBLOCK=128 / XBLOCK_SUB=16).
"$PYTHON" - <<PYEOF
import json, pathlib
p = pathlib.Path("$DIR/tiling_space.json")
ts = json.loads(p.read_text())
for param in ts["tiling_params"]:
    if param["name"] == "XBLOCK":
        param["values"] = [128]
    elif param["name"] == "XBLOCK_SUB":
        param["values"] = [16]
p.write_text(json.dumps(ts, indent=2))
print("  ✓ tiling_space.json patched (XBLOCK=128, XBLOCK_SUB=16)")
PYEOF
log ""
log "  [TilingData struct]"
log "$(grep -A6 'struct TilingData' "$DIR/relu_kernel.cpp")"


# ── STAGE 2: Compile ──────────────────────────────────────────
echo ""
echo "==================== [STAGE 2] runtime-session compile ===================="
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output.npy"
"$RUNTIME_SESSION" \
  --kernel "$DIR/relu_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name relu__v0 \
  2>&1
echo "  ✓ Compile OK → $ARTIFACT_ROOT"


# ── STAGE 3: Run + Verify ─────────────────────────────────────
echo ""
echo "==================== [STAGE 3] Simulator Run + Verify ===================="
log "  XBLOCK=128, XBLOCK_SUB=16, dim_arg0_0=1024, dim_arg1_0=1024"
log "  block_dim=8  (8 AiCores × 128 elements = 1024)"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"
cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "input", "path": "${DIR}/input.npy" }
  ],
  "outputs": [
    { "name": "output", "path": "${ACTUAL_OUTPUT}" }
  ],
  "expected_outputs": [
    { "name": "output", "path": "${DIR}/expected.npy" }
  ],
  "tiling": {
    "schema": "${DIR}/tiling_space.json",
    "params": "XBLOCK=128,XBLOCK_SUB=16,dim_arg0_0=1024,dim_arg1_0=1024"
  },
  "block_dim": 8,
  "workspace_size": 16777216,
  "profiling": true,
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
echo "   relu_kernel.mlir           → auto-fuse-codegen 后 MLIR"
echo "   relu_kernel.cpp            → AscendC C++ kernel"
echo "   build_e2e/artifact         → runtime-session 编译产物"
echo "   build_e2e/output.npy       → 仿真输出"
echo "========================================================"
