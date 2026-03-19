#!/bin/bash
# ============================================================
# elementwise + broadcast + transpose 完整编译流水线 Demo
#
# 用法：
#   source examples/env.sh
#   bash examples/ewop-broadcast-transpose/run.sh
#
# 计算图：
#   输入: A[M,N], bias[N], scale[M]
#   Op1: relu(A[M,N]) + broadcast_col(bias[N]) → B[M,N]
#        （逐元素 ReLU + 列方向广播加 bias）
#   Op2: Transpose(B[M,N]) → C[N,M]
#        （2D 转置，行列互换）
#   Op3: C[N,M] * broadcast_col(scale[M]) → D[N,M]
#        （逐元素乘 scale，scale[M] 沿 N 轴广播）
#
# Transpose 用 linalg.generic 表达：
#   ins 的 indexing_map = (d0,d1) -> (d1,d0)，body 直接 yield
#   → AscendC 转换：ascendc.transpose %dst, %src
#
# 与现有示例的统一之处：
#   - 同一套 pass pipeline（tiling → bufferize → buffer-placement → linalg-to-ascendc）
#   - 同一套内存层次（GM/VECIN/VECOUT）
#   - 同一套多核调度（get_block_idx）
#   - transpose generic 检测新增于 ComputeConversion.cpp
#
# 各阶段说明：
#   step0_input.mlir          原始 High-Level IR（linalg/tensor，完全符号化）
#   step1_fused.mlir          --linalg-fuse-elementwise-ops（尝试融合）
#   step2_tiled.mlir          --transform-interpreter tiling 结果
#                             → Op1/Op2/Op3 各自 TB/Tb 两级循环
#   step3_bufferized.mlir     --one-shot-bufferize 结果（tensor→memref）
#   step4_buffer_placement.mlir  --ascendc-buffer-placement 结果
#                             → 推导 on-chip memory_space（VECIN=9, VECOUT=10）
#   step5_ascendc.mlir        --linalg-to-ascendc 结果
#                             → Op1: broadcast_l2 + max_l2 + add_l2（relu+broadcast_col）
#                             → Op2: ascendc.transpose（转置）
#                             → Op3: broadcast_l2 + mul_l2（列广播乘 scale）
#   step6_parallelize.mlir    --ascendc-parallelize（get_block_idx 单维调度）
#   step7_kernel.mlir         --ascendc-prepare-for-emit（kernel IR）
#   step8_kernel.cpp          ascir-translate -mlir-to-ascendc（C++ kernel）
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"

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
echo " elementwise + broadcast + transpose 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "==================== [STAGE 0] 解析 High-Level IR（relu+broadcast_col+add, transpose, scale_mul）===================="
log "  输入: step0_input.mlir"
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir" 2>&1
log "  ✓ 解析成功，输出: step0_input_out.mlir"
log ""
log "  [计算图结构]"
log "    Op1 relu_bias_add:  iterator = [Parallel, Parallel], d0=M, d1=N"
log "         relu(A[M,N]) + broadcast_col(bias[N]) → B[M,N]"
log "    Op2 transpose:      iterator = [Parallel, Parallel], d0=N, d1=M"
log "         B[M,N] → C[N,M]（ins 访问 (d1,d0)，即 B[j,i]）"
log "    Op3 scale_mul:      iterator = [Parallel, Parallel], d0=N, d1=M"
log "         C[N,M] * broadcast_col(scale[M]) → D[N,M]"

# ── STAGE 1: 融合 ──────────────────────────────────────────
echo ""
echo "==================== [STAGE 1] 融合：--linalg-fuse-elementwise-ops ===================="
log "  Op1 与 Op2/Op3 之间存在 transpose 依赖，通常无法融合"
$AFIR_OPT --linalg-fuse-elementwise-ops "$DIR/step0_input.mlir" \
  --canonicalize --cse \
  -o "$DIR/step1_fused.mlir" 2>&1
log "  ✓ 输出: step1_fused.mlir"
log ""
log "  [融合后 linalg.generic 数量]"
log "$(grep -c "linalg.generic" "$DIR/step1_fused.mlir" || echo "  0")"

# ── STAGE 2: Transform Tiling ──────────────────────────────
echo ""
echo "==================== [STAGE 2] Tiling：--transform-interpreter ===================="
log "  输入: step2_transform.mlir（含 Transform 脚本）"
log "  策略: 对 Op1/Op2/Op3 分别沿第一轴（M 或 N）做两级切分 TB/Tb"
$AFIR_OPT --transform-interpreter "$DIR/step2_transform.mlir" \
  --canonicalize --cse \
  -o "$DIR/step2_tiled.mlir" 2>&1
log "  ✓ Tiling 成功，输出: step2_tiled.mlir"
log ""
log "  [循环结构]"
log "$(grep -E "scf\.for|ascendc\." "$DIR/step2_tiled.mlir" | head -15)"

# ── STAGE 3: Bufferize ─────────────────────────────────────
echo ""
echo "==================== [STAGE 3] Bufferize：--one-shot-bufferize ===================="
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
log "  ✓ Bufferize 成功，输出: step3_bufferized.mlir"
log ""
log "  [ascendc.* attrs 是否保留在 scf.for 上]"
log "$(grep "ascendc\." "$DIR/step3_bufferized.mlir" | grep -v "transform\." | head -10)"

# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "==================== [STAGE 4] Buffer Placement：--ascendc-buffer-placement ===================="
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir" 2>&1
log "  ✓ Buffer Placement 成功，输出: step4_buffer_placement.mlir"
log ""
log "  [on-chip memory_space 标注]"
log "$(grep -E "[0-9]+ : i32" "$DIR/step4_buffer_placement.mlir" | head -10 || \
  echo "  (检查输出)")"
log "  [memref.copy 占位搬运]"
log "$(grep "memref.copy" "$DIR/step4_buffer_placement.mlir" | head -8)"

# ── STAGE 5: Linalg → AscendC Compute ─────────────────────
echo ""
echo "==================== [STAGE 5] Linalg → AscendC：--linalg-to-ascendc ===================="
log "  转换规则："
log "    Op1 (relu+broadcast_col+add):"
log "       → broadcast_l2（列广播 bias）+ duplicate_l2（zero）+ max_l2（relu）+ add_l2"
log "    Op2 (transpose):"
log "       → ascendc.transpose %dst, %src（检测到置换 indexing_map）"
log "    Op3 (scale_mul):"
log "       → broadcast_l2（列广播 scale）+ mul_l2"
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize \
  --cse \
  -o "$DIR/step5_ascendc.mlir" 2>&1
log "  ✓ Linalg→AscendC 成功，输出: step5_ascendc.mlir"
log ""
log "  [生成的 AscendC ops]"
log "$(grep -E "ascendc\.(broadcast_l2|add_l2|mul_l2|max_l2|transpose|data_copy)" \
  "$DIR/step5_ascendc.mlir" | head -20 || \
  echo "  (未找到 ascendc compute ops，请检查输出)")"

# ── STAGE 6: AscendC Parallelize ───────────────────────────
echo ""
echo "==================== [STAGE 6] Parallelize：--ascendc-parallelize ===================="
$AFIR_OPT "$DIR/step5_ascendc.mlir" \
  --ascendc-parallelize \
  --canonicalize \
  --cse \
  -o "$DIR/step6_parallelize.mlir" 2>&1
log "  ✓ Parallelize 成功，输出: step6_parallelize.mlir"
log ""
log "  [get_block_idx dispatch]"
log "$(grep -E "get_block_idx|muli.*block" "$DIR/step6_parallelize.mlir" | head -5 || \
  echo "  (未找到多核调度 ops)")"

# ── STAGE 7: Prepare For Emit ──────────────────────────────
echo ""
echo "==================== [STAGE 7] Prepare For Emit：--ascendc-prepare-for-emit ===================="
$AFIR_OPT "$DIR/step6_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --canonicalize \
  --cse \
  -o "$DIR/step7_kernel.mlir" 2>&1
log "  ✓ Prepare For Emit 成功，输出: step7_kernel.mlir"
log ""
log "  [函数签名 + 属性]"
log "$(grep -E "func\.func|ascendc\.(aicore|global)|emitasc\." \
  "$DIR/step7_kernel.mlir" | head -8 || \
  echo "  (请检查输出)")"

# ── STAGE 8: AscendC C++ Code Generation ───────────────────
echo ""
echo "==================== [STAGE 8] Codegen：ascir-translate -mlir-to-ascendc ===================="
ASCIR_TRANSLATE="${ASCIR_TRANSLATE:-ascir-translate}"
if command -v "$ASCIR_TRANSLATE" &>/dev/null; then
  python3 -c "
import re, sys
content = open('$DIR/step7_kernel.mlir').read()
content = content.replace('module attributes {transform.with_named_sequence}', 'module')
content = re.sub(r'  transform\.named_sequence.*?^  \}\n', '', content, flags=re.DOTALL|re.MULTILINE)
sys.stdout.write(content)
" > "$DIR/step8_no_transform.mlir"
  "$ASCIR_TRANSLATE" -mlir-to-ascendc "$DIR/step8_no_transform.mlir" \
    -o "$DIR/step8_kernel.cpp" 2>&1
  log "  ✓ Codegen 成功，输出: step8_kernel.cpp"
  log ""
  log "  [生成的 C++ kernel 头部]"
  log "$(head -30 "$DIR/step8_kernel.cpp")"
else
  log "  (ascir-translate 未找到，跳过 Stage 8)"
  log "  若已构建 pyasc，请将 ascir-translate 加入 PATH 后重新运行。"
fi

echo ""
echo "========================================================"
echo " 流水线完成！生成文件："
echo "   step0_input_out.mlir        → 解析后 IR"
echo "   step1_fused.mlir            → 融合尝试（transpose 阻断融合）"
echo "   step2_tiled.mlir            → Tiling 后（Op1/Op2/Op3 各自 TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注（VECIN/VECOUT）"
echo "   step5_ascendc.mlir          → AscendC compute ops（含 ascendc.transpose）"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度（get_block_idx）"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step8_kernel.cpp            → AscendC C++ kernel 源码"
echo "========================================================"
