#!/bin/bash
# ============================================================
# broadcast + add + reducesum 完整编译流水线 Demo
#
# 用法：
#   source examples/matmul-add-relu-sum/env.sh
#   bash examples/broadcast-add-reduce/run.sh
#
# 各阶段说明：
#   step0_input.mlir          原始 High-Level IR（linalg/tensor，完全符号化）
#   step1_fused.mlir          --linalg-fuse-elementwise-ops 融合结果
#                             → Broadcast+Add 内联进 ReduceSum body
#                             → 从3个 generic → 1个 generic ["parallel","reduction"]
#   step2_tiled.mlir          --transform-interpreter tiling 结果
#                             → TB层（ascendc.parallel）/ Tb层 / 保留 Reduction 轴
#   step3_bufferized.mlir     --one-shot-bufferize 结果（tensor→memref）
#   step4_buffer_placement.mlir  --ascendc-buffer-placement 结果
#                             → 推导 on-chip memory_space（VECIN/VECOUT 等）
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"

echo "========================================================"
echo " broadcast + add + reducesum 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "[STAGE 0] 解析 High-Level IR（Broadcast + Add + ReduceSum）"
echo "  输入: step0_input.mlir"
echo "  Pass: (仅解析，无变换)"
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir" 2>&1
echo "  ✓ 解析成功，输出: step0_input_out.mlir"
echo ""
echo "  [轴分组分析]"
echo "    Op1 Broadcast: iterator = [Parallel, Parallel]   d0=P, d1=P"
echo "    Op2 Add:       iterator = [Parallel, Parallel]   d0=P, d1=P"
echo "    Op3 ReduceSum: iterator = [Parallel, Reduction]  d0=P, d1=R ← 冲突!"
echo "    → d1 类型不一致（P vs R）"
echo "    → 但 ReduceSum body 可内联 Broadcast+Add → 单 generic 融合"

# ── STAGE 1: 融合 ──────────────────────────────────────────
echo ""
echo "[STAGE 1] 融合：--linalg-fuse-elementwise-ops"
echo "  输入: step0_input.mlir"
$AFIR_OPT --linalg-fuse-elementwise-ops "$DIR/step0_input.mlir" -o "$DIR/step1_fused.mlir" 2>&1
echo "  ✓ 融合成功，输出: step1_fused.mlir"
echo ""
echo "  [融合结果]"
echo "    3个 linalg.generic → 1个 generic"
echo "    iterator = [\"parallel\", \"reduction\"]"
echo "    body: acc += (A[d0] + B[d0,d1])   // Broadcast+Add 内联到规约体内"
cat "$DIR/step1_fused.mlir" | grep -A8 "linalg.generic"

# ── STAGE 2: Transform Tiling ──────────────────────────────
echo ""
echo "[STAGE 2] Tiling：--transform-interpreter"
echo "  输入: step2_transform.mlir（含 Transform 脚本）"
echo "  策略: 沿 Parallel 轴(d0=M) 做两级切分 TB/Tb"
echo "         Reduction 轴(d1=N) 不切，保持完整 N 规约"
$AFIR_OPT --transform-interpreter "$DIR/step2_transform.mlir" --canonicalize --cse -o "$DIR/step2_tiled.mlir" 2>&1
echo "  ✓ Tiling 成功，输出: step2_tiled.mlir"
echo ""
echo "  [循环结构]"
grep -E "scf\.for|ascendc\." "$DIR/step2_tiled.mlir" | head -10

# ── STAGE 3: Bufferize ─────────────────────────────────────
echo ""
echo "[STAGE 3] Bufferize：--one-shot-bufferize"
echo "  输入: step2_tiled.mlir（tiling 已完成，ascendc.* attrs 已在 scf.for 上）"
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
echo "  ✓ Bufferize 成功，输出: step3_bufferized.mlir"
echo ""
echo "  [ascendc.* attrs 是否保留在 scf.for 上]"
grep "ascendc\." "$DIR/step3_bufferized.mlir" | grep -v "transform\." | head -10
echo "  [memref 类型]"
grep "memref" "$DIR/step3_bufferized.mlir" | grep "func.func" | head -3

# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "[STAGE 4] Buffer Placement：--ascendc-buffer-placement"
echo "  输入: step3_bufferized.mlir"
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir" 2>&1
echo "  ✓ Buffer Placement 成功，输出: step4_buffer_placement.mlir"
echo ""
echo "  [on-chip memory_space 标注]"
grep -E "memory_space|: i32" "$DIR/step4_buffer_placement.mlir" | head -10 || \
  echo "  (注：本算子无 matmul，buffer placement 主要标注 VECIN/VECOUT)"

# ── STAGE 5: Linalg → AscendC Compute ─────────────────────
echo ""
echo "[STAGE 5] Linalg → AscendC：--linalg-to-ascendc"
echo "  输入: step4_buffer_placement.mlir"
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize \
  --cse \
  -o "$DIR/step5_ascendc.mlir" 2>&1
echo "  ✓ Linalg→AscendC 成功，输出: step5_ascendc.mlir"
echo ""
echo "  [生成的 AscendC ops]"
grep -E "ascendc\.(broadcast|add_l2|reduce_sum_2d|data_copy)" "$DIR/step5_ascendc.mlir" | head -20 || \
  echo "  (未找到 ascendc compute ops，请检查输出)"

echo ""
echo "========================================================"
echo " 流水线完成！生成文件："
echo "   step0_input_out.mlir        → 解析后 IR"
echo "   step1_fused.mlir            → 融合后（3 generic → 1）"
echo "   step2_tiled.mlir            → Tiling 后（TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注"
echo "   step5_ascendc.mlir          → AscendC compute ops"
echo "========================================================"
