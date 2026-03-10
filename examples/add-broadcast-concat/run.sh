#!/bin/bash
# ============================================================
# elementwise + broadcast + concat 完整编译流水线 Demo
#
# 用法：
#   source examples/matmul-add-relu-sum/env.sh
#   bash examples/ewop-broadcast-concat/run.sh
#
# 计算图：
#   Op1: input_a[M] + input_b[M,N] → C[M,N]  （广播加法）
#   Op2: input_c[M] * input_d[M,N] → D[M,N]  （广播乘法）
#   Concat(C, D, axis=0) → output[2M, N]
#
# Concat 语义通过 tensor.insert_slice + subview offset 隐式实现：
#   - C 写入 output[0:M, :]
#   - D 写入 output[M:2M, :]
# 无需新的 asc.concat pass
#
# 与现有示例的统一之处：
#   - 同一套 pass pipeline（tiling → bufferize → buffer-placement → linalg-to-ascendc）
#   - 同一套内存层次（GM/VECIN/VECOUT/VECCALC）
#   - 同一套多核调度（get_block_idx）
#
# 各阶段说明：
#   step0_input.mlir          原始 High-Level IR（linalg/tensor，完全符号化）
#   step1_fused.mlir          --linalg-fuse-elementwise-ops
#                             尝试融合，本场景两个 generic 独立，结果与 step0 相同
#   step2_tiled.mlir          --transform-interpreter tiling 结果
#                             → TB 层（ascendc.parallel）/ Tb 层 / 保留 N 轴
#   step3_bufferized.mlir     --one-shot-bufferize 结果（tensor→memref）
#   step4_buffer_placement.mlir  --ascendc-buffer-placement 结果
#                             → 推导 on-chip memory_space（VECIN=9, VECOUT=10）
#   step5_ascendc.mlir        --linalg-to-ascendc 结果
#                             → broadcast+addf → broadcast_l2 + add_l2
#                             → broadcast+mulf → broadcast_l2 + mul_l2
#   step6_parallelize.mlir    --ascendc-parallelize（get_block_idx 单维调度）
#   step7_kernel.mlir         --ascendc-prepare-for-emit（kernel IR）
#   step8_kernel.cpp          ascir-translate -mlir-to-ascendc（C++ kernel）
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"

echo "========================================================"
echo " elementwise + broadcast + concat 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "[STAGE 0] 解析 High-Level IR（broadcast+add, broadcast+mul, concat）"
echo "  输入: step0_input.mlir"
echo "  Pass: (仅解析，无变换)"
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir" 2>&1
echo "  ✓ 解析成功，输出: step0_input_out.mlir"
echo ""
echo "  [计算图结构]"
echo "    Op1 broadcast+add: iterator = [Parallel, Parallel]  input_a[M] + input_b[M,N] → C[M,N]"
echo "    Op2 broadcast+mul: iterator = [Parallel, Parallel]  input_c[M] * input_d[M,N] → D[M,N]"
echo "    Concat (insert_slice): C → output[0:M,:], D → output[M:2M,:]"
echo "    → 两个独立的 pure-parallel generic + 两个 tensor.insert_slice"
echo "    → Concat 通过 subview offset 隐式实现，无需显式 asc.concat"

# ── STAGE 1: 尝试融合 ─────────────────────────────────────
echo ""
echo "[STAGE 1] 融合：--linalg-fuse-elementwise-ops"
echo "  注：Op1 和 Op2 输出独立（均写入不同的中间 tensor），无法互相融合"
echo "  此 pass 对本场景基本是 no-op，保留阶段以与其他示例保持一致"
$AFIR_OPT --linalg-fuse-elementwise-ops "$DIR/step0_input.mlir" \
  --canonicalize --cse \
  -o "$DIR/step1_fused.mlir" 2>&1
echo "  ✓ 输出: step1_fused.mlir"

# ── STAGE 2: Transform Tiling ──────────────────────────────
echo ""
echo "[STAGE 2] Tiling：--transform-interpreter"
echo "  输入: step2_transform.mlir（含 Transform 脚本）"
echo "  策略: 对 Op1 和 Op2 分别沿 M 轴做两级切分 TB/Tb"
echo "         N 轴不切，保持完整 N 在 UB 内处理"
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
echo "  tensor.insert_slice → memref.subview 写操作（Concat 的隐式表示）"
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
echo "  ✓ Bufferize 成功，输出: step3_bufferized.mlir"
echo ""
echo "  [ascendc.* attrs 是否保留在 scf.for 上]"
grep "ascendc\." "$DIR/step3_bufferized.mlir" | grep -v "transform\." | head -10
echo "  [subview（Concat 的 offset 表示）]"
grep "memref.subview" "$DIR/step3_bufferized.mlir" | head -6 || echo "  (检查 memref.copy 或直接写)"

# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "[STAGE 4] Buffer Placement：--ascendc-buffer-placement"
echo "  输入: step3_bufferized.mlir"
echo "  推导规则："
echo "    prologue src:GM->VECIN → VECIN(9) 标注"
echo "    epilogue dst:VECOUT->GM → VECOUT(10) 标注"
echo "    insert_slice 目标（output 的 subview）保持 GM(0)"
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
echo "    broadcast+addf generic → broadcast_l2 + add_l2"
echo "    broadcast+mulf generic → broadcast_l2 + mul_l2"
echo "    memref.copy GM→VECIN  → data_copy_l2"
echo "    memref.copy VECOUT→GM → data_copy_l2（subview offset 携带 concat 位置）"
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize \
  --cse \
  -o "$DIR/step5_ascendc.mlir" 2>&1
echo "  ✓ Linalg→AscendC 成功，输出: step5_ascendc.mlir"
echo ""
echo "  [生成的 AscendC ops]"
grep -E "ascendc\.(broadcast_l2|add_l2|mul_l2|data_copy)" \
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
echo "   step1_fused.mlir            → 融合后（本场景基本不变）"
echo "   step2_tiled.mlir            → Tiling 后（Op1/Op2 各自 TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref，insert_slice→subview）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注（VECIN/VECOUT）"
echo "   step5_ascendc.mlir          → AscendC compute ops（broadcast_l2/add_l2/mul_l2）"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度（get_block_idx）"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step8_kernel.cpp            → AscendC C++ kernel 源码"
echo "========================================================"
