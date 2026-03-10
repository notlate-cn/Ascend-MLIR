#!/bin/bash
# ============================================================
# elementwise + broadcast + split 完整编译流水线 Demo
#
# 用法：
#   source examples/matmul-add-relu-sum/env.sh
#   bash examples/ewop-broadcast-split/run.sh
#
# 计算图：
#   Op1: relu(A[M,N]) + broadcast_row(bias[M]) → C[M,N]
#        （逐元素 ReLU + 行方向广播加 bias）
#   Op2: C[:, 0:N/2] * broadcast_col(scale0[N/2]) → out0[M, N/2]
#        （Split 前半列 + 列广播乘 scale）
#   Op3: C[:, N/2:N] * broadcast_col(scale1[N/2]) → out1[M, N/2]
#        （Split 后半列 + 列广播乘 scale）
#
# Split 语义通过 tensor.extract_slice 隐式实现（作为 ins）：
#   - Op2 读 C[:, 0:N/2]：subview offset=(0,0)
#   - Op3 读 C[:, N/2:N]：subview offset=(0,N/2)
# 与 add-broadcast-concat 的 Concat（insert_slice outs）完全对称
#
# 新增特性：列广播 affine_map<(d0,d1) -> (d1)>
#   - 与现有行广播 (d0,d1)->d0 对称，验证两种广播方向
#
# 与现有示例的统一之处：
#   - 同一套 pass pipeline（tiling → bufferize → buffer-placement → linalg-to-ascendc）
#   - 同一套内存层次（GM/VECIN/VECOUT）
#   - 同一套多核调度（get_block_idx）
#
# 各阶段说明：
#   step0_input.mlir          原始 High-Level IR（linalg/tensor，完全符号化）
#   step1_fused.mlir          --linalg-fuse-elementwise-ops（Op1 融合）
#   step2_tiled.mlir          --transform-interpreter tiling 结果
#                             → TB 层（ascendc.parallel）/ Tb 层 / 保留 N/N/2 轴
#   step3_bufferized.mlir     --one-shot-bufferize 结果（tensor→memref）
#   step4_buffer_placement.mlir  --ascendc-buffer-placement 结果
#                             → 推导 on-chip memory_space（VECIN=9, VECOUT=10）
#   step5_ascendc.mlir        --linalg-to-ascendc 结果
#                             → Op1: broadcast_l2 + add_l2（relu 内联）
#                             → Op2/Op3: broadcast_l2 + mul_l2（列广播）
#   step6_parallelize.mlir    --ascendc-parallelize（get_block_idx 单维调度）
#   step7_kernel.mlir         --ascendc-prepare-for-emit（kernel IR）
#   step8_kernel.cpp          ascir-translate -mlir-to-ascendc（C++ kernel）
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"

echo "========================================================"
echo " elementwise + broadcast + split 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "[STAGE 0] 解析 High-Level IR（relu+broadcast_add, split_scale0, split_scale1）"
echo "  输入: step0_input.mlir"
echo "  Pass: (仅解析，无变换)"
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir" 2>&1
echo "  ✓ 解析成功，输出: step0_input_out.mlir"
echo ""
echo "  [计算图结构]"
echo "    Op1 relu+broadcast_add: iterator = [Parallel, Parallel]"
echo "         relu(A[M,N]) + broadcast_row(bias[M]) → C[M,N]"
echo "    Op2 split_scale0:       iterator = [Parallel, Parallel]"
echo "         C[:, 0:N/2] * broadcast_col(scale0[N/2]) → out0[M, N/2]"
echo "    Op3 split_scale1:       iterator = [Parallel, Parallel]"
echo "         C[:, N/2:N] * broadcast_col(scale1[N/2]) → out1[M, N/2]"
echo "    → Split 通过 tensor.extract_slice（ins）隐式实现"
echo "    → 与 add-broadcast-concat Concat（insert_slice outs）对称"

# ── STAGE 1: 融合 ──────────────────────────────────────────
echo ""
echo "[STAGE 1] 融合：--linalg-fuse-elementwise-ops"
echo "  Op1（relu+broadcast_add）尝试与 Op2/Op3 融合"
echo "  由于 Op2/Op3 的 ins 是 Op1 的输出 slice，MLIR 可能将部分融合"
echo "  保留本阶段与其他示例保持结构一致"
$AFIR_OPT --linalg-fuse-elementwise-ops "$DIR/step0_input.mlir" \
  --canonicalize --cse \
  -o "$DIR/step1_fused.mlir" 2>&1
echo "  ✓ 输出: step1_fused.mlir"
echo ""
echo "  [融合后 linalg.generic 数量]"
grep -c "linalg.generic" "$DIR/step1_fused.mlir" || echo "  0"

# ── STAGE 2: Transform Tiling ──────────────────────────────
echo ""
echo "[STAGE 2] Tiling：--transform-interpreter"
echo "  输入: step2_transform.mlir（含 Transform 脚本）"
echo "  策略: 对 Op1/Op2/Op3 分别沿 M 轴做两级切分 TB/Tb"
echo "         N 轴（或 N/2 轴）不切，整体在 UB 内处理"
$AFIR_OPT --transform-interpreter "$DIR/step2_transform.mlir" \
  --canonicalize --cse \
  -o "$DIR/step2_tiled.mlir" 2>&1
echo "  ✓ Tiling 成功，输出: step2_tiled.mlir"
echo ""
echo "  [循环结构]"
grep -E "scf\.for|ascendc\." "$DIR/step2_tiled.mlir" | head -15

# ── STAGE 3: Bufferize ─────────────────────────────────────
echo ""
echo "[STAGE 3] Bufferize：--one-shot-bufferize"
echo "  输入: step2_tiled.mlir"
echo "  tensor.extract_slice → memref.subview（Split 的隐式表示）"
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
echo "  ✓ Bufferize 成功，输出: step3_bufferized.mlir"
echo ""
echo "  [ascendc.* attrs 是否保留在 scf.for 上]"
grep "ascendc\." "$DIR/step3_bufferized.mlir" | grep -v "transform\." | head -10
echo "  [subview（Split 的 offset 表示）]"
grep "memref.subview" "$DIR/step3_bufferized.mlir" | head -6 || echo "  (检查输出)"

# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "[STAGE 4] Buffer Placement：--ascendc-buffer-placement"
echo "  输入: step3_bufferized.mlir"
echo "  推导规则："
echo "    prologue src:GM->VECIN → VECIN(9) 标注"
echo "    epilogue dst:VECOUT->GM → VECOUT(10) 标注"
echo "    extract_slice 来源（C 的 subview）→ GM(0)（从 GM 读 slice）"
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir" 2>&1
echo "  ✓ Buffer Placement 成功，输出: step4_buffer_placement.mlir"
echo ""
echo "  [on-chip memory_space 标注]"
grep -E "[0-9]+ : i32" "$DIR/step4_buffer_placement.mlir" | head -10 || \
  echo "  (注：仅 VECIN/VECOUT，无 Cube 相关 memory_space)"
echo "  [memref.copy 占位搬运]"
grep "memref.copy" "$DIR/step4_buffer_placement.mlir" | head -8

# ── STAGE 5: Linalg → AscendC Compute ─────────────────────
echo ""
echo "[STAGE 5] Linalg → AscendC：--linalg-to-ascendc"
echo "  输入: step4_buffer_placement.mlir"
echo "  转换规则（ComputeConversion - pure-parallel generic 路径）："
echo "    Op1 (relu+broadcast_row+add):"
echo "       → broadcast_l2（行广播 bias）+ max_l2（relu，与 zero duplicate_l2）+ add_l2"
echo "    Op2/Op3 (broadcast_col+mul):"
echo "       → broadcast_l2（列广播 scale）+ mul_l2"
echo "    memref.copy GM→VECIN  → data_copy_l2"
echo "    memref.copy VECOUT→GM → data_copy_l2（subview offset 携带 split 位置）"
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize \
  --cse \
  -o "$DIR/step5_ascendc.mlir" 2>&1
echo "  ✓ Linalg→AscendC 成功，输出: step5_ascendc.mlir"
echo ""
echo "  [生成的 AscendC ops]"
grep -E "ascendc\.(broadcast_l2|add_l2|mul_l2|max_l2|data_copy)" \
  "$DIR/step5_ascendc.mlir" | head -20 || \
  echo "  (未找到 ascendc compute ops，请检查输出)"

# ── STAGE 6: AscendC Parallelize ───────────────────────────
echo ""
echo "[STAGE 6] Parallelize：--ascendc-parallelize"
echo "  输入: step5_ascendc.mlir"
echo "  将最外层 scf.for（TB 层）转换为单维多核 AiCore 调度："
echo "    %block_idx = ascendc.get_block_idx"
echo "    %i         = arith.muli %block_idx, %TB  → row offset"
echo "    scf.if (inbound)  ← 越界 block 直接跳过"
$AFIR_OPT "$DIR/step5_ascendc.mlir" \
  --ascendc-parallelize \
  --canonicalize \
  --cse \
  -o "$DIR/step6_parallelize.mlir" 2>&1
echo "  ✓ Parallelize 成功，输出: step6_parallelize.mlir"
echo ""
echo "  [get_block_idx dispatch]"
grep -E "get_block_idx|muli.*block" "$DIR/step6_parallelize.mlir" | head -5 || \
  echo "  (未找到多核调度 ops)"

# ── STAGE 7: Prepare For Emit ──────────────────────────────
echo ""
echo "[STAGE 7] Prepare For Emit：--ascendc-prepare-for-emit"
echo "  输入: step6_parallelize.mlir"
$AFIR_OPT "$DIR/step6_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --canonicalize \
  --cse \
  -o "$DIR/step7_kernel.mlir" 2>&1
echo "  ✓ Prepare For Emit 成功，输出: step7_kernel.mlir"
echo ""
echo "  [函数签名 + 属性]"
grep -E "func\.func|ascendc\.(aicore|global)|emitasc\." \
  "$DIR/step7_kernel.mlir" | head -8 || \
  echo "  (请检查输出)"

# ── STAGE 8: AscendC C++ Code Generation ───────────────────
echo ""
echo "[STAGE 8] Codegen：ascir-translate -mlir-to-ascendc"
echo "  输入: step7_kernel.mlir"
echo "  输出: step8_kernel.cpp（AscendC C++ kernel 源码）"
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
  echo "  ✓ Codegen 成功，输出: step8_kernel.cpp"
  echo ""
  echo "  [生成的 C++ kernel 头部]"
  head -30 "$DIR/step8_kernel.cpp"
else
  echo "  (ascir-translate 未找到，跳过 Stage 8)"
  echo "  若已构建 pyasc，请将 ascir-translate 加入 PATH 后重新运行。"
fi

echo ""
echo "========================================================"
echo " 流水线完成！生成文件："
echo "   step0_input_out.mlir        → 解析后 IR"
echo "   step1_fused.mlir            → 融合后（Op1 relu+broadcast_add 融合）"
echo "   step2_tiled.mlir            → Tiling 后（Op1/Op2/Op3 各自 TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref，extract_slice→subview）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注（VECIN/VECOUT）"
echo "   step5_ascendc.mlir          → AscendC compute ops（broadcast_l2/add_l2/mul_l2）"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度（get_block_idx）"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step8_kernel.cpp            → AscendC C++ kernel 源码"
echo "========================================================"
