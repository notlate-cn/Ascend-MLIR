#!/bin/bash
# ============================================================
# elementwise + broadcast + concat 完整编译流水线 Demo
#
# 用法：
#   source examples/env.sh
#   bash examples/add-broadcast-concat/run.sh
#
# 计算图：
#   Op1: input_a[M] + input_b[M,N] → C[M,N]  （广播加法）
#   Op2: input_c[M] * input_d[M,N] → D[M,N]  （广播乘法）
#   Concat(C, D, axis=0) → output[2M, N]
#
# ── 设计说明 ────────────────────────────────────────────────
#
# 1. Concat 的表达与展开
#    step0 用 tensor.concat dim(0) 直接表达语义（与 stablehlo→linalg
#    的 enablePrimitiveOps 路径一致，语义清晰）。
#    step2 Transform sequence 开头通过
#      transform.apply_patterns.tensor.decompose_concat
#    将其展开为 empty + insert_slice，转为 tiling 友好的形式，
#    后续 tiling/bufferize 流程不变。
#    注：axis=0（首轴）concat 展开后内存连续，DMA 友好；
#        axis=N（尾轴）concat 展开后为 strided memref，不适合 AscendC。
#
# 2. Op 识别与 Tiling 策略
#    Op1 和 Op2 迭代器相同（[parallel, parallel]），tile 策略相同。
#    Transform sequence 用 transform.foreach 对所有 linalg.generic
#    统一做两级 tiling，无需 library_call 等手工标记区分——
#    这与 stablehlo→linalg 默认路径（HloBroadcastInDimConverter）
#    生成的 generic 完全兼容，不依赖上层标注。
#
# 3. 水平融合与 Kernel 边界
#    Op1 和 Op2 输入完全独立，无 producer-consumer 关系，
#    linalg-fuse-elementwise-ops 无法融合（step1 基本是 no-op）。
#    两者在同一 func.func 内串行执行，生成一个 AscendC kernel。
#
#    对于复杂图（如 MMoE），kernel 边界划分有以下方案：
#      a) func.func 粒度（当前方案）：上层按 kernel 粒度手工导出多个
#         函数，简单直接，适合现阶段。
#      b) flow.dispatch 区域（IREE 风格）：专门的 dispatch region op
#         标记 kernel 边界，编译器自动决定融合范围。
#      c) outline-kernels pass：上层用一个大 func 描述完整图，
#         由 pass 按数据依赖 + 内存约束自动切分成多个 func.func。
#    当前采用方案 a，每个 func.func 对应一个 fused kernel。
#
# ── 各阶段说明 ───────────────────────────────────────────────
#   step0_input.mlir          原始 High-Level IR（linalg/tensor，完全符号化）
#                             两个 linalg.generic + tensor.concat
#   step1_fused.mlir          --linalg-fuse-elementwise-ops
#                             两个 generic 独立，结果与 step0 相同
#   step2_tiled.mlir          --transform-interpreter tiling 结果
#                             decompose_concat → TB 层 / Tb 层 / 保留 N 轴
#   step3_bufferized.mlir     --one-shot-bufferize 结果（tensor→memref）
#   step4_buffer_placement.mlir  --ascendc-buffer-placement 结果
#                             推导 on-chip memory_space（VECIN=9, VECOUT=10）
#   step5_ascendc.mlir        --linalg-to-ascendc 结果
#                             broadcast+addf → broadcast_l2 + add_l2
#                             broadcast+mulf → broadcast_l2 + mul_l2
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
echo " elementwise + broadcast + concat 编译流水线"
echo "========================================================"


# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "==================== [STAGE 0] 解析 High-Level IR（broadcast+add, broadcast+mul, concat）===================="
log "  输入: step0_input.mlir"
log "  Pass: (仅解析，无变换)"
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir" 2>&1
log "  ✓ 解析成功，输出: step0_input_out.mlir"
log ""
log "  [计算图结构]"
log "    Op1 broadcast+add: iterator = [Parallel, Parallel]  input_a[M] + input_b[M,N] → C[M,N]"
log "    Op2 broadcast+mul: iterator = [Parallel, Parallel]  input_c[M] * input_d[M,N] → D[M,N]"
log "    Concat: tensor.concat dim(0) %C, %D → output[2M,N]"
log "    → 两个独立的 pure-parallel generic + tensor.concat"
log "    → tensor.concat 由 step2 Transform sequence 展开为 insert_slice"


# ── STAGE 1: 尝试融合 ─────────────────────────────────────
echo ""
echo "==================== [STAGE 1] 融合：--linalg-fuse-elementwise-ops ===================="
log "  注：Op1 和 Op2 输出独立（均写入不同的中间 tensor），无法互相融合"
log "  此 pass 对本场景基本是 no-op，保留阶段以与其他示例保持一致"
$AFIR_OPT --linalg-fuse-elementwise-ops "$DIR/step0_input.mlir" \
  --canonicalize --cse \
  -o "$DIR/step1_fused.mlir" 2>&1
log "  ✓ 输出: step1_fused.mlir"


# ── STAGE 2: Transform Tiling ──────────────────────────────
echo ""
echo "==================== [STAGE 2] Tiling：--transform-interpreter ===================="
log "  输入: step2_transform.mlir（含 Transform 脚本）"
log "  策略: 对 Op1 和 Op2 分别沿 M 轴做两级切分 TB/Tb"
log "         N 轴不切，保持完整 N 在 UB 内处理"
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
log "  输入: step2_tiled.mlir"
log "  tensor.insert_slice → memref.subview 写操作（Concat 的隐式表示）"
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
log "  ✓ Bufferize 成功，输出: step3_bufferized.mlir"
log ""
log "  [ascendc.* attrs 是否保留在 scf.for 上]"
log "$(grep "ascendc\." "$DIR/step3_bufferized.mlir" | grep -v "transform\." | head -10)"
log "  [subview（Concat 的 offset 表示）]"
log "$(grep "memref.subview" "$DIR/step3_bufferized.mlir" | head -6 || echo "  (检查 memref.copy 或直接写)")"


# ── STAGE 3b: Fold Concat Alloc ────────────────────────────
echo ""
echo "==================== [STAGE 3b] Fold Concat Alloc：--ascendc-fold-concat-alloc ===================="
log "  输入: step3_bufferized.mlir"
log "  消除由 decompose_concat → bufferize 引入的 GM→GM memref.copy："
log "    原始: Op1/Op2 各写独立 alloc，最终 memref.copy 到 output[2M,N] subview"
log "    变换: outAlloc + subview 提前到 for 循环之前，循环直接写 output slice"
$AFIR_OPT \
  --ascendc-fold-concat-alloc \
  --canonicalize --cse \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step3_bufferized.mlir" 2>&1
log "  ✓ Fold Concat Alloc 成功，output step3_bufferized.mlir 已原地更新"
log "  [验证无 memref.copy]"
log "$(grep 'memref.copy' "$DIR/step3_bufferized.mlir" || echo "  ✓ 无 memref.copy")"


# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "==================== [STAGE 4] Buffer Placement：--ascendc-buffer-placement ===================="
log "  输入: step3_bufferized.mlir"
log "  推导规则："
log "    prologue src:GM->VECIN → VECIN(9) 标注"
log "    epilogue dst:VECOUT->GM → VECOUT(10) 标注"
log "    insert_slice 目标（output 的 subview）保持 GM(0)"
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir" 2>&1
log "  ✓ Buffer Placement 成功，输出: step4_buffer_placement.mlir"
log ""
log "  [on-chip memory_space 标注]"
log "$(grep -E "[0-9]+ : i32" "$DIR/step4_buffer_placement.mlir" | head -10 || \
  echo "  (注：仅 VECIN/VECOUT，无 Cube 相关 memory_space)")"
log "  [memref.copy 占位搬运]"
log "$(grep "memref.copy" "$DIR/step4_buffer_placement.mlir" | head -8)"


# ── STAGE 5: Linalg → AscendC Compute ─────────────────────
echo ""
echo "==================== [STAGE 5] Linalg → AscendC：--linalg-to-ascendc ===================="
log "  输入: step4_buffer_placement.mlir"
log "  转换规则（ComputeConversion - pure-parallel generic 路径）："
log "    broadcast+addf generic → broadcast_l2 + add_l2"
log "    broadcast+mulf generic → broadcast_l2 + mul_l2"
log "    memref.copy GM→VECIN  → data_copy_l2"
log "    memref.copy VECOUT→GM → data_copy_l2（subview offset 携带 concat 位置）"
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize \
  --cse \
  -o "$DIR/step5_ascendc.mlir" 2>&1
log "  ✓ Linalg→AscendC 成功，输出: step5_ascendc.mlir"
log ""
log "  [生成的 AscendC ops]"
log "$(grep -E "ascendc\.(broadcast_l2|add_l2|mul_l2|data_copy)" \
  "$DIR/step5_ascendc.mlir" | head -20 || \
  echo "  (未找到 ascendc compute ops，请检查输出)")"


# ── STAGE 6: AscendC Parallelize ───────────────────────────
echo ""
echo "==================== [STAGE 6] Parallelize：--ascendc-parallelize ===================="
log "  输入: step5_ascendc.mlir"
log "  将最外层 scf.for（TB 层）转换为单维多核 AiCore 调度："
log "    %block_idx = ascendc.get_block_idx"
log "    %i         = arith.muli %block_idx, %TB  → row offset"
log "    scf.if (inbound)  ← 越界 block 直接跳过"
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
log "  输入: step6_parallelize.mlir"
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
log "  输入: step7_kernel.mlir"
log "  输出: step8_kernel.cpp（AscendC C++ kernel 源码）"
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
echo "   step1_fused.mlir            → 融合后（本场景基本不变）"
echo "   step2_tiled.mlir            → Tiling 后（Op1/Op2 各自 TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref，insert_slice→subview）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注（VECIN/VECOUT）"
echo "   step5_ascendc.mlir          → AscendC compute ops（broadcast_l2/add_l2/mul_l2）"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度（get_block_idx）"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step8_kernel.cpp            → AscendC C++ kernel 源码"
echo "========================================================"
