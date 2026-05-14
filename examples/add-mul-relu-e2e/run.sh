#!/bin/bash
# ============================================================
# add+mul+relu 端到端编译流水线 Demo —— 使用 --vector-plan-codegen
#
# 用法：
#   source examples/env.sh
#   bash examples/add-mul-relu-e2e/run.sh [--log]
#
# 计算图：
#   out[d0,d1,d2] = max(a + b*c, 0.0)   (dynamic 3D, float32)
#   示例形状: 4×8×32 = 1024 elements
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

D0=4; D1=8; D2=32
N=$((D0 * D1 * D2))
# After multi-op collapse, XBLOCK tiles over flat N = D0*D1*D2.
# BLOCK_DIM = ceil(N / XBLOCK).
XBLOCK=128
XBLOCK_SUB=16
BLOCK_DIM=$(( (N + XBLOCK - 1) / XBLOCK ))

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done
log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " add+mul+relu E2E: vector-plan-codegen → runtime-session → sim"
echo "========================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR"

echo ""
echo "==================== [STAGE 1] MLIR → AscendC C++ ===================="
"$AFIR_OPT" "$DIR/add_mul_relu.mlir" --vector-plan-codegen \
  -o "$DIR/add_mul_relu_kernel.mlir" 2>&1
log "  ✓ MLIR codegen OK → add_mul_relu_kernel.mlir"

"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/add_mul_relu_kernel.mlir" \
  -o "$DIR/add_mul_relu_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
echo "  ✓ Translate OK → add_mul_relu_kernel.cpp + tiling_space.json"

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
  --kernel "$DIR/add_mul_relu_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name add_mul_relu__v0 \
  2>&1
echo "  ✓ Compile OK → $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 3] Simulator Run + Verify ===================="
log "  XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB, shape=${D0}x${D1}x${D2}=$N, block_dim=$BLOCK_DIM"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"
INTER1_OUT="$BUILD_DIR/inter1.npy"
INTER2_OUT="$BUILD_DIR/inter2.npy"

# Tiling params — from auto-generated tiling_space.json.
# After multi-op collapse the loop iterates over flat N = D0*D1*D2.
# dim_arg{N}_{0,1,2} are the three 3D dimensions of each tensor arg:
#   arg0=a, arg1=b, arg3=out (arg2=c is broadcast-fused, no separate dim)
TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TILING_PARAMS+=",dim_arg1_0=${D0},dim_arg1_1=${D1},dim_arg1_2=${D2}"
TILING_PARAMS+=",dim_arg0_0=${D0},dim_arg0_1=${D1},dim_arg0_2=${D2}"
TILING_PARAMS+=",dim_arg3_0=${D0},dim_arg3_1=${D1},dim_arg3_2=${D2}"

cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "a",   "path": "${DIR}/a.npy" },
    { "name": "b",   "path": "${DIR}/b.npy" },
    { "name": "c",   "path": "${DIR}/c.npy" }
  ],
  "outputs": [
    { "name": "out",   "path": "${ACTUAL_OUTPUT}" },
    { "name": "inter1","path": "${INTER1_OUT}" },
    { "name": "inter2","path": "${INTER2_OUT}" }
  ],
  "expected_outputs": [
    { "name": "out",   "path": "${DIR}/expected.npy" },
    { "name": "inter1","path": "${DIR}/expected_inter1.npy" },
    { "name": "inter2","path": "${DIR}/expected_inter2.npy" }
  ],
  "tiling": {
    "schema": "${DIR}/tiling_space.json",
    "params": "${TILING_PARAMS}"
  },
  "block_dim": ${BLOCK_DIM},
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
echo "   add_mul_relu_kernel.mlir    → vector-plan-codegen 后 MLIR"
echo "   add_mul_relu_kernel.cpp     → AscendC C++ kernel"
echo "   tiling_space.json           → 自动生成+patched tiling schema"
echo "   build_e2e/artifact          → runtime-session 编译产物"
echo "   build_e2e/output.npy        → 仿真输出（session.validation=pass）"
echo "========================================================"
