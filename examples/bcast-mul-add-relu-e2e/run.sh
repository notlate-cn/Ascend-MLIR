#!/bin/bash
# ============================================================
# broadcast+mul+add+relu 端到端编译流水线 Demo —— 使用 --vector-plan-codegen
#
# 用法：
#   source examples/env.sh
#   bash examples/bcast-mul-add-relu-e2e/run.sh [--log]
#
# 计算图：
#   inter_mul[d0,d1,d2] = b[d0,0,d2] * c[d0,d1,d2]   (b broadcast on d1)
#   inter_add[d0,d1,d2] = a[d0,d1,d2] + inter_mul[d0,d1,d2]
#   out[d0,d1,d2]       = max(inter_add[d0,d1,d2], 0.0)
#
#   b has shape D0×1×D2 (size-1 on d1 axis, broadcast to D0×D1×D2)
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

D0=4; D1=8; D2=32

# Loop structure after tiling:
#   Parallel: block_idx * XBLOCK < D0  →  BLOCK_DIM = ceil(D0 / XBLOCK)
#   arg8 (d0 sub-tile): 0..XBLOCK step XBLOCK_SUB
#   arg9 (broadcast d1): 0..D1 step BCAST_TILE_0
#   arg10 (d2 sub-tile): 0..D2 step XBLOCK_SUB_0
#
# VECCALC init_buffer is inside arg9 but outside arg10.
# It is called arg8_iters × arg9_iters times per block.
# Set XBLOCK = XBLOCK_SUB and BCAST_TILE_0 = D1 to ensure exactly 1 call.
XBLOCK=1
XBLOCK_SUB=1
BCAST_TILE_0=${D1}
XBLOCK_SUB_0=${D2}
BLOCK_DIM=$(( (D0 + XBLOCK - 1) / XBLOCK ))

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done
log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " bcast+mul+add+relu E2E: vector-plan-codegen → runtime-session → sim"
echo "========================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR"

echo ""
echo "==================== [STAGE 1] MLIR → AscendC C++ ===================="
"$AFIR_OPT" "$DIR/bcast_mul_add_relu.mlir" --vector-plan-codegen \
  -o "$DIR/bcast_mul_add_relu_kernel.mlir" 2>&1
log "  ✓ MLIR codegen OK → bcast_mul_add_relu_kernel.mlir"

"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/bcast_mul_add_relu_kernel.mlir" \
  -o "$DIR/bcast_mul_add_relu_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
echo "  ✓ Translate OK → bcast_mul_add_relu_kernel.cpp + tiling_space.json"

"$PYTHON" - <<PYEOF
import json, pathlib
p = pathlib.Path("$DIR/tiling_space.json")
ts = json.loads(p.read_text())
for param in ts["tiling_params"]:
    if param["name"] == "XBLOCK":
        param["values"] = [$XBLOCK]
    elif param["name"] == "XBLOCK_SUB":
        param["values"] = [$XBLOCK_SUB]
    elif param["name"] == "BCAST_TILE_0":
        param["values"] = [$BCAST_TILE_0]
    elif param["name"] == "XBLOCK_SUB_0":
        param["values"] = [$XBLOCK_SUB_0]
p.write_text(json.dumps(ts, indent=2))
print("  ✓ tiling_space.json patched (XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB, BCAST_TILE_0=$BCAST_TILE_0, XBLOCK_SUB_0=$XBLOCK_SUB_0)")
PYEOF

echo ""
echo "==================== [STAGE 2] runtime-session compile ===================="
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output.npy"
INTER_MUL_OUT="$BUILD_DIR/inter_mul.npy"
INTER_ADD_OUT="$BUILD_DIR/inter_add.npy"

"$RUNTIME_SESSION" \
  --kernel "$DIR/bcast_mul_add_relu_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name bcast_mul_add_relu__v0 \
  2>&1
echo "  ✓ Compile OK → $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 3] Simulator Run + Verify ===================="
log "  XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB, BCAST_TILE_0=$BCAST_TILE_0, XBLOCK_SUB_0=$XBLOCK_SUB_0"
log "  shape: a=${D0}x${D1}x${D2}, b=${D0}x1x${D2}, block_dim=$BLOCK_DIM"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

# Tiling params: tunable params + fixed shape dims for all 6 memref args.
# dim_arg0_{0,1,2}: a's dims;  dim_arg1_{0,1,2}: b's dims (d1=1 always)
# dim_arg2_{1,2}: c's d1,d2;  dim_arg7/8/9_{1,2}: intermediate+output d1,d2
TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TILING_PARAMS+=",BCAST_TILE_0=${BCAST_TILE_0},XBLOCK_SUB_0=${XBLOCK_SUB_0}"
TILING_PARAMS+=",dim_arg0_0=${D0},dim_arg0_1=${D1},dim_arg0_2=${D2}"
TILING_PARAMS+=",dim_arg1_0=${D0},dim_arg2_1=${D1},dim_arg1_2=${D2}"
TILING_PARAMS+=",dim_arg1_1=1,dim_arg2_2=${D2}"
TILING_PARAMS+=",dim_arg7_1=${D1},dim_arg7_2=${D2}"
TILING_PARAMS+=",dim_arg8_1=${D1},dim_arg8_2=${D2}"
TILING_PARAMS+=",dim_arg9_1=${D1},dim_arg9_2=${D2}"

cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "a", "path": "${DIR}/a.npy" },
    { "name": "b", "path": "${DIR}/b.npy" },
    { "name": "c", "path": "${DIR}/c.npy" }
  ],
  "outputs": [
    { "name": "inter_mul", "path": "${INTER_MUL_OUT}" },
    { "name": "inter_add", "path": "${INTER_ADD_OUT}" },
    { "name": "out",       "path": "${ACTUAL_OUTPUT}" }
  ],
  "expected_outputs": [
    { "name": "inter_mul", "path": "${DIR}/expected_inter_mul.npy" },
    { "name": "inter_add", "path": "${DIR}/expected_inter_add.npy" },
    { "name": "out",       "path": "${DIR}/expected.npy" }
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
echo "   bcast_mul_add_relu_kernel.mlir   → vector-plan-codegen 后 MLIR"
echo "   bcast_mul_add_relu_kernel.cpp    → AscendC C++ kernel"
echo "   tiling_space.json               → 自动生成+patched tiling schema"
echo "   build_e2e/artifact              → runtime-session 编译产物"
echo "   build_e2e/output.npy            → 仿真输出（session.validation=pass）"
echo "========================================================"
