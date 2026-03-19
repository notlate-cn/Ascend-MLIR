#!/bin/bash
# ============================================================
# elementwise + broadcast + gather 完整编译流水线 Demo
#
# 用法：
#   source examples/env.sh
#   bash examples/ewop-broadcast-gather/run.sh
#
# 计算图：
#   Op1 (Gather):      data[M, N] 按 indices[K] 沿 N 轴抽取 → gathered[M, K]
#                      语义：gathered[i, j] = data[i, indices[j]]
#   Op2 (BroadcastAdd): gathered[M, K] + broadcast_row(bias[M]) → out[M, K]
#                      bias 沿 K 轴广播（行广播）
#
# Gather 转换策略（ComputeConversion 的 gather_by_index 路径）：
#   - 检测 library_call = "gather_by_index" 的 linalg.generic
#   - ins[0] = indices[K] (i32, VECIN), ins[1] = data[Tb_M, N] (f16, VECIN)
#   - 对每行 i in 0..Tb_M：
#     gather_l2(dst_row[K], data_row[N], indices[K], srcBaseAddr=0, count=K)
#   - 结果写入 VECOUT，后续 epilogue data_copy 写回 GM
#
# BroadcastAdd 转换（复用现有 pure-parallel generic 路径）：
#   - indices 输入按 col_broadcast_map 广播：broadcast_l2
#   - gathered 输入按 full_access_map：直接 data_copy
#   - body: addf → add_l2
#
# 与现有示例的统一之处：
#   - 同一套 pass pipeline（tiling → bufferize → buffer-placement → linalg-to-ascendc）
#   - 同一套内存层次（GM/VECIN/VECOUT）
#   - 同一套多核调度（get_block_idx）
#   - Gather 通过新增的 gather_by_index 检测路径，生成 gather_l2 而不是普通的 add_l2
#
# 各阶段说明：
#   step0_input.mlir          原始 High-Level IR（linalg/tensor，完全符号化）
#   step1_fused.mlir          --linalg-fuse-elementwise-ops（两个 generic 独立，no-op）
#   step2_tiled.mlir          --transform-interpreter tiling 结果
#   step3_bufferized.mlir     --one-shot-bufferize 结果（tensor→memref）
#   step4_buffer_placement.mlir  --ascendc-buffer-placement 结果
#   step5_ascendc.mlir        --linalg-to-ascendc 结果
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
echo " elementwise + broadcast + gather 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "==================== [STAGE 0] 解析 High-Level IR（gather + broadcast+add）===================="
log "  输入: step0_input.mlir"
log "  Pass: (仅解析，无变换)"
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir" 2>&1
log "  ✓ 解析成功，输出: step0_input_out.mlir"
log ""
log "  [计算图结构]"
log "    Op1 gather:        iterator = [Parallel, Parallel]  data[M,N] + indices[K] → gathered[M,K]"
log "    Op2 broadcast+add: iterator = [Parallel, Parallel]  gathered[M,K] + bias[M] → out[M,K]"
log "    → Op1 通过 library_call='gather_by_index' 标记 gather 语义"
log "    → Op2 复用现有 broadcast+addf → broadcast_l2 + add_l2 路径"

# ── STAGE 1: 尝试融合 ─────────────────────────────────────
echo ""
echo "==================== [STAGE 1] 融合：--linalg-fuse-elementwise-ops ===================="
log "  注：Op1 和 Op2 输出独立（Op1 输出 gathered，Op2 输出 out），无法融合"
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
log "         K 轴不切，整 K 在 UB 内处理"
log "  注意: Gather 场景建议 Tb_M=1（每次处理一行，N 轴扫全部）"
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
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
log "  ✓ Bufferize 成功，输出: step3_bufferized.mlir"
log ""
log "  [ascendc.* attrs 是否保留在 scf.for 上]"
log "$(grep "ascendc\." "$DIR/step3_bufferized.mlir" | grep -v "transform\." | head -10)"
log "  [memref.alloc 和 indices 处理]"
log "$(grep "memref.alloc\|memref.copy" "$DIR/step3_bufferized.mlir" | head -8)"

# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "==================== [STAGE 4] Buffer Placement：--ascendc-buffer-placement ===================="
log "  输入: step3_bufferized.mlir"
log "  推导规则："
log "    prologue src:GM->VECIN → VECIN(9) 标注"
log "    epilogue dst:VECOUT->GM → VECOUT(10) 标注"
log "    indices[K] (i32) 也被 prologue 搬入 VECIN（统一处理）"
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir" 2>&1
log "  ✓ Buffer Placement 成功，输出: step4_buffer_placement.mlir"
log ""
log "  [on-chip memory_space 标注]"
log "$(grep -E "[0-9]+ : i32" "$DIR/step4_buffer_placement.mlir" | head -10)"
log "  [memref.copy 占位搬运]"
log "$(grep "memref.copy" "$DIR/step4_buffer_placement.mlir" | head -8)"

# ── STAGE 5: Linalg → AscendC Compute ─────────────────────
echo ""
echo "==================== [STAGE 5] Linalg → AscendC：--linalg-to-ascendc ===================="
log "  输入: step4_buffer_placement.mlir"
log "  转换规则："
log "    gather_by_index generic → gather_l2（逐行：data_row[N] + indices[K] → gathered_row[K]）"
log "    broadcast+addf generic → broadcast_l2 + add_l2"
log "    memref.copy GM→VECIN  → data_copy_l2"
log "    memref.copy VECOUT→GM → data_copy_l2"
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize \
  --cse \
  -o "$DIR/step5_ascendc.mlir" 2>&1
log "  ✓ Linalg→AscendC 成功，输出: step5_ascendc.mlir"
log ""
log "  [生成的 AscendC ops]"
log "$(grep -E "ascendc\.(gather_l2|broadcast_l2|add_l2|data_copy)" \
  "$DIR/step5_ascendc.mlir" | head -20 || \
  echo "  (未找到 ascendc compute ops，请检查输出)")"

# ── STAGE 6: AscendC Parallelize ───────────────────────────
echo ""
echo "==================== [STAGE 6] Parallelize：--ascendc-parallelize ===================="
log "  输入: step5_ascendc.mlir"
$AFIR_OPT "$DIR/step5_ascendc.mlir" \
  --ascendc-parallelize \
  --canonicalize \
  --cse \
  -o "$DIR/step6_parallelize.mlir" 2>&1
log "  ✓ Parallelize 成功，输出: step6_parallelize.mlir"
log ""
log "  [get_block_idx dispatch]"
log "$(grep -E "get_block_idx|muli.*block" "$DIR/step6_parallelize.mlir" | head -5)"

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
  log "$(head -40 "$DIR/step8_kernel.cpp")"
else
  log "  (ascir-translate 未找到，跳过 Stage 8)"
  log "  若已构建 pyasc，请将 ascir-translate 加入 PATH 后重新运行。"
fi

echo ""
echo "========================================================"
echo " 流水线完成！生成文件："
echo "   step0_input_out.mlir        → 解析后 IR"
echo "   step1_fused.mlir            → 融合后（本场景不变）"
echo "   step2_tiled.mlir            → Tiling 后（Op1/Op2 各自 TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注（VECIN/VECOUT）"
echo "   step5_ascendc.mlir          → AscendC compute ops（gather_l2/broadcast_l2/add_l2）"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度（get_block_idx）"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step8_kernel.cpp            → AscendC C++ kernel 源码"
echo "========================================================"
