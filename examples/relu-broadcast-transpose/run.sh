#!/bin/bash
# ============================================================
# relu + broadcast + transpose 完整编译流水线 Demo
#
# 用法：
#   source examples/env.sh
#   bash examples/relu-broadcast-transpose/run.sh
#
# 计算图：
#   输入: data0[M,1], data1[N,M]
#   out[n,m] = relu(data0[m,0]) + data1[n,m]
#
# 融合后单 linalg.generic，indexing_map:
#   data0: (d0,d1)->(d1,0)  — 转置+广播（列向量广播至[N,M]）
#   data1: (d0,d1)->(d0,d1) — identity
#
# 形状: M=640, N=500, TB_N=64, block_dim=8（N不整除TB_N，含tail block）
#
# 各阶段说明：
#   step0_input.mlir          原始 High-Level IR（linalg/tensor，完全符号化）
#   step0_input_out.mlir      --linalg-generalize-named-ops --linalg-fuse-elementwise-ops 结果
#                             → 融合为单 linalg.generic（relu+transpose+broadcast+add）
#   step1_fused.mlir          --canonicalize --cse（基于 step0_input_out.mlir）
#   step2_tiled.mlir          --transform-interpreter tiling 结果
#                             → 单 generic TB/Tb 两级循环
#   step3_bufferized.mlir     --one-shot-bufferize 结果（tensor→memref）
#   step4_buffer_placement.mlir  --ascendc-buffer-placement 结果
#                             → 推导 on-chip memory_space（VECIN=9, VECOUT=10）
#   step5_ascendc.mlir        --linalg-to-ascendc 结果
#                             → data_copy_l2（GM→UB）+ relu + broadcast_l2 + add_l2
#   step6_parallelize.mlir    --ascendc-parallelize（get_block_idx 单维调度）
#   step7_kernel.mlir         --ascendc-prepare-for-emit（kernel IR）
#   step7_cann.mlir           --canonicalize-cann-signature（CANN 标准签名）
#   step8_kernel.cpp          afir-translate -mlir-to-cann（C++ kernel）
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
PYTHON="${PYTHON:-python3}"
COMPILER="${COMPILER:-compiler}"
VALIDATOR="${VALIDATOR:-validator}"

# 解析参数
VERBOSE=false
for arg in "$@"; do
  case $arg in
    --log) VERBOSE=true ;;
  esac
done

log() {
  if $VERBOSE; then
    echo "$@"
  fi
}

echo "========================================================"
echo " relu + broadcast + transpose 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "==================== [STAGE 0] 解析 + 融合 High-Level IR（relu + transpose + broadcast + add）===================="
log "  输入: step0_input.mlir"
$AFIR_OPT --linalg-generalize-named-ops \
  --linalg-fuse-elementwise-ops \
  --canonicalize --cse \
  "$DIR/step0_input.mlir" \
  -o "$DIR/step0_input_out.mlir" 2>&1
log "  ✓ 融合成功，输出: step0_input_out.mlir"

# ── STAGE 1: Canonicalize/CSE ──────────────────────────────
echo ""
echo "==================== [STAGE 1] Canonicalize/CSE ===================="
$AFIR_OPT --canonicalize --cse "$DIR/step0_input_out.mlir" \
  -o "$DIR/step1_fused.mlir" 2>&1
log "  ✓ 输出: step1_fused.mlir"

# ── STAGE 2: Transform Tiling ──────────────────────────────
echo ""
echo "==================== [STAGE 2] Tiling：--transform-interpreter ===================="
$AFIR_OPT --transform-interpreter "$DIR/step2_transform.mlir" \
  --canonicalize --cse \
  -o "$DIR/step2_tiled.mlir" 2>&1
log "  ✓ Tiling 成功，输出: step2_tiled.mlir"

# ── STAGE 3: Bufferize ─────────────────────────────────────
echo ""
echo "==================== [STAGE 3] Bufferize：--one-shot-bufferize ===================="
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
log "  ✓ Bufferize 成功，输出: step3_bufferized.mlir"

# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "==================== [STAGE 4] Buffer Placement：--ascendc-buffer-placement ===================="
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir" 2>&1
log "  ✓ Buffer Placement 成功，输出: step4_buffer_placement.mlir"

# ── STAGE 5: Linalg → AscendC Compute ─────────────────────
echo ""
echo "==================== [STAGE 5] Linalg → AscendC：--linalg-to-ascendc ===================="
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize \
  --cse \
  -o "$DIR/step5_ascendc.mlir" 2>&1
log "  ✓ Linalg→AscendC 成功，输出: step5_ascendc.mlir"

# ── STAGE 6: AscendC Parallelize ───────────────────────────
echo ""
echo "==================== [STAGE 6] Parallelize：--ascendc-parallelize ===================="
$AFIR_OPT "$DIR/step5_ascendc.mlir" \
  --ascendc-parallelize \
  --canonicalize \
  --cse \
  -o "$DIR/step6_parallelize.mlir" 2>&1
log "  ✓ Parallelize 成功，输出: step6_parallelize.mlir"

# ── STAGE 7: Prepare For Emit ──────────────────────────────
echo ""
echo "==================== [STAGE 7] Prepare For Emit：--ascendc-prepare-for-emit ===================="
$AFIR_OPT "$DIR/step6_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --canonicalize \
  --cse \
  -o "$DIR/step7_kernel.mlir" 2>&1
log "  ✓ Prepare For Emit 成功，输出: step7_kernel.mlir"

# ── STAGE 7b: Canonicalize CANN signature ──────────────────
echo ""
echo "==================== [STAGE 7b] CANN Signature：--canonicalize-cann-signature ===================="
$AFIR_OPT --canonicalize-cann-signature \
  "$DIR/step7_kernel.mlir" \
  -o "$DIR/step7_cann.mlir" 2>&1
log "  ✓ CANN 签名规范化成功，输出: step7_cann.mlir"

# ── STAGE 8: AscendC C++ Code Generation ───────────────────
echo ""
echo "==================== [STAGE 8] Codegen：afir-translate -mlir-to-cann ===================="
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/step7_cann.mlir" \
  -o "$DIR/step8_kernel.cpp" 2>&1
log "  ✓ Codegen 成功，输出: step8_kernel.cpp"

# ── STAGE 8b: Generate test data ───────────────────────────
echo ""
echo "==================== [STAGE 8b] 生成测试数据：gen_inputs.py ===================="
"$PYTHON" "$DIR/gen_inputs.py" --m 640 --n 500 --seed 42 --out-dir "$DIR" 2>&1
log "  ✓ 生成成功：input_data0.npy, input_data1.npy, output_expected.npy"

# ── STAGE 9: Compile AscendC kernel ────────────────────────
echo ""
echo "==================== [STAGE 9] Compile：bisheng C++ → .bin ===================="
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
"$COMPILER" \
  --kernel "$DIR/step8_kernel.cpp" \
  --output "$BUILD_DIR" \
  --name relu_transpose_broadcast_add \
  --num-inputs 2 2>&1
log "  ✓ Compile 成功，输出: $BUILD_DIR/relu_transpose_broadcast_add.bin"

# ── STAGE 10: Run and verify ────────────────────────────────
echo ""
echo "==================== [STAGE 10] Run + Verify ===================="
log "  使用参数：TB_M=64, TB_N=64, M=640, N=500, block-dim=8"
BIN="$BUILD_DIR/relu_transpose_broadcast_add.bin"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

if [ -f "$BIN" ]; then
  "$VALIDATOR" \
    --bin "$BIN" \
    --name relu_transpose_broadcast_add \
    --inputs "$DIR/input_data0.npy,$DIR/input_data1.npy" \
    --expected "$DIR/output_expected.npy" \
    --tiling-schema "$DIR/tiling_space.json" \
    --tiling-params 'TB_M=64,TB_N=64,dim_arg0_0=640,dim_arg1_0=500,dim_arg0_1=1,dim_arg1_1=640' \
    --block-dim 8 \
    --atol 1e-2 \
    --rtol 1e-2 \
    --dump-actual "$BUILD_DIR/actual.txt" \
    --dump-expected "$BUILD_DIR/expected.txt" \
    --precision 4 \
    2>&1 | tee "$VALIDATION_LOG" | grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]' || true
  grep -q '^session.backend=sim' "$VALIDATION_LOG"
  grep -q '^session.result=success' "$VALIDATION_LOG"
else
  echo "  ⚠ bin not found — skipping run"
fi

echo ""
echo "========================================================"
echo " 流水线完成！生成文件："
echo "   step0_input_out.mlir        → generalize+fuse 后单 linalg.generic IR"
echo "   step1_fused.mlir            → canonicalize+cse 清理后 IR"
echo "   step2_tiled.mlir            → Tiling 后（单 generic TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注（VECIN/VECOUT）"
echo "   step5_ascendc.mlir          → AscendC compute ops（data_copy+relu+broadcast+add）"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度（get_block_idx）"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step7_cann.mlir             → CANN 标准签名 IR"
echo "   step8_kernel.cpp            → AscendC C++ kernel 源码"
echo "   build_e2e/relu_transpose_broadcast_add.bin → 编译后二进制"
echo "========================================================"

rm -fr *.dump
rm -fr *.toml
