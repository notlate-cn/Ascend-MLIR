#!/bin/bash
# ============================================================
# matmul + add + relu 完整编译流水线 Demo
#
# 用法：
#   source examples/matmul-add-relu-sum/env.sh
#   bash examples/matmul-add-relu-sum/run.sh
#
# 各阶段说明：
#   fc_add_relu.mlir                   原始 High-Level IR（linalg-on-tensor，动态 shape）
#   output_step1_tile_and_fuse_3level  Transform Dialect 3级 Tile+Fuse 结果
#                                       → for_TB_M/for_TB_N（分核）/ for_Tb_M/for_Tb_N / for_K
#                                       → ascendc.parallel / prologue / epilogue / unit 标注落在 scf.for 上
#   output_step2_bufferized.mlir       --one-shot-bufferize 结果（tensor → memref）
#   output_step3_buffer_placement.mlir --ascendc-buffer-placement 结果
#                                       → 按 prologue/epilogue/unit 标注推导 on-chip memory_space
#                                       → 插入 memref.copy 占位搬运（GM↔A1/B1/VECIN/VECOUT、A1→A2、B1→B2、CO1→VECIN）
#                                       → 清除所有 ascendc.* 标注
#   output_step4_lowering_to_asc.mlir  --linalg-to-ascendc 结果
#                                       → linalg.matmul → ascendc.mmad（含 A1→A2 load_data_l0）
#                                       → linalg.elementwise add → ascendc.add_l2
#                                       → linalg.fill + linalg.elementwise max_signed → ascendc.duplicate_l2 + ascendc.max_l2
#                                       → memref.copy → ascendc.data_copy_l2 / data_copy_nd2nz / data_copy_co12dst
#                                       → 插入 TQue（ascendc.queue / que_bind.alloc/enque/deque/free）
#                                       → 插入 TPipe（ascendc.pipe / pipe.init_buffer）
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"

echo "========================================================"
echo " matmul + add + relu 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "[STAGE 0] 解析 High-Level IR（matmul + add + relu）"
echo "  输入: fc_add_relu.mlir"
echo "  Pass: (仅解析，无变换)"
echo ""
echo "  [原始计算图]"
echo "    linalg.matmul:              C = A * B              （Cube 单元）"
echo "    linalg.elementwise add:     D = C + bias           （Vector 单元）"
echo "    linalg.fill + linalg.elementwise max_signed: E = max(D, 0)  （Vector 单元 / ReLU）"
echo ""
echo "  [Tile 策略（3级嵌套）]"
echo "    Level 1（分核层）: TB_M × TB_N   → 映射到多 AiCore 并行（ascendc.parallel）"
echo "    Level 2（片上层）: Tb_M × Tb_N   → 每个 AiCore 的 UB tile"
echo "    Level 3（Cube K）: t_K           → 沿 K 轴分块，每次搬 A1→A2 / B1→B2"
echo ""
echo "  [内存层次]"
echo "    GM → A1/B1  （GM->L1，分核入口批量搬）"
echo "    A1 → A2     （L1->L0A，每次 K 迭代）"
echo "    B1 → B2     （L1->L0B，每次 K 迭代）"
echo "    matmul 累加 → CO1（L0C）"
echo "    CO1 → VECIN （fixpipe，K 轴完成后）"
echo "    VECIN + bias → VECCALC  （add）"
echo "    VECCALC vs VECIN(0) → VECOUT  （max / ReLU）"
echo "    VECOUT → GM  （分核出口写回）"

# ── STAGE 1: Transform Tiling ──────────────────────────────
echo ""
echo "[STAGE 1] 3级 Tile+Fuse：--transform-interpreter"
echo "  输入: fc_add_relu.mlir + transform_tile_and_fuse_3level.mlir"
echo "  策略:"
echo "    step 4-6: tile max [TB_M, TB_N] → for_TB_M/for_TB_N，fuse add/matmul into for_TB_N"
echo "              标注 ascendc.parallel, prologue(GM->A1/B1/VECIN), epilogue(VECOUT->GM)"
echo "    step 7-9: tile max [Tb_M, Tb_N] → for_Tb_M/for_Tb_N，fuse add/matmul into for_Tb_N"
echo "    step 10:  tile matmul [0, 0, t_K] → for_K"
echo "              标注 ascendc.unit=AiCore.Cube/Vector"
echo "              标注 for_K prologue(A1->A2,B1->B2), epilogue(CO1->VECIN)"
echo "    step 11:  hoist_loop_invariant_subsets（提升循环不变切片）"
$AFIR_OPT "$DIR/fc_add_relu.mlir" \
  --transform-preload-library="transform-library-paths=$DIR/transform_tile_and_fuse_3level.mlir" \
  --transform-interpreter="entry-point=__transform_main" \
  --canonicalize \
  --cse \
  -o "$DIR/output_step1_tile_and_fuse_3level.mlir" 2>&1
echo "  ✓ Tiling 成功，输出: output_step1_tile_and_fuse_3level.mlir"
echo ""
echo "  [循环结构]"
grep -E "scf\.for|ascendc\." "$DIR/output_step1_tile_and_fuse_3level.mlir" | head -15

# ── STAGE 2: Bufferize ─────────────────────────────────────
echo ""
echo "[STAGE 2] Bufferize：--one-shot-bufferize"
echo "  输入: output_step1_tile_and_fuse_3level.mlir"
echo "  将 tensor/extract_slice/insert_slice → memref/subview/copy"
echo "  ascendc.* 属性保留在 scf.for 上"
$AFIR_OPT "$DIR/output_step1_tile_and_fuse_3level.mlir" \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true" \
  --buffer-deallocation-pipeline \
  -o "$DIR/output_step2_bufferized.mlir" 2>&1
echo "  ✓ Bufferize 成功，输出: output_step2_bufferized.mlir"
echo ""
echo "  [ascendc.* attrs 是否保留在 scf.for 上]"
grep "ascendc\." "$DIR/output_step2_bufferized.mlir" | head -10
echo "  [memref 类型（函数签名）]"
grep "func.func" "$DIR/output_step2_bufferized.mlir" | head -3

# ── STAGE 3: Buffer Placement ──────────────────────────────
echo ""
echo "[STAGE 3] Buffer Placement：--ascendc-buffer-placement"
echo "  输入: output_step2_bufferized.mlir"
echo "  推导规则（AscendCBufferPlacementPass）："
echo "    Rule A: 解析 prologue/epilogue，建立各层搬运任务"
echo "    Rule B: linalg.matmul → A2/B2/CO1；linalg.elementwise → VECCALC/VECOUT"
echo "    Rule C: 传播 memory_space 到 subview，更新 alloc memory_space"
echo "    Rule D: 插入 memref.copy 占位（GM↔A1/B1/VECIN/VECOUT, CO1→VECIN 等）"
echo "    Rule E: 清除所有 ascendc.* 属性"
$AFIR_OPT "$DIR/output_step2_bufferized.mlir" \
  --ascendc-buffer-placement \
  --canonicalize \
  --cse \
  -o "$DIR/output_step3_buffer_placement.mlir" 2>&1
echo "  ✓ Buffer Placement 成功，输出: output_step3_buffer_placement.mlir"
echo ""
echo "  [on-chip memory_space 标注]"
grep -E "[0-9]+ : i32" "$DIR/output_step3_buffer_placement.mlir" | head -15
echo "  [memref.copy 占位搬运]"
grep "memref.copy" "$DIR/output_step3_buffer_placement.mlir" | head -10

# ── STAGE 4: Linalg → AscendC ──────────────────────────────
echo ""
echo "[STAGE 4] Linalg → AscendC：--linalg-to-ascendc"
echo "  输入: output_step3_buffer_placement.mlir"
echo "  转换规则（LinalgToAscendCPass）："
echo "    linalg.matmul（A∈A2,B∈B2,C∈CO1）→ ascendc.mmad + load_data_l0/with_transpose"
echo "    linalg.fill（outs∈CO1）           → （初始化由 mmad 内部处理，fill 消除）"
echo "    linalg.elementwise add             → ascendc.add_l2"
echo "    linalg.fill + elementwise max      → ascendc.duplicate_l2 + ascendc.max_l2"
echo "    memref.copy GM→A1/B1               → ascendc.data_copy_nd2nz（含 nd2nz_params）"
echo "    memref.copy GM→VECIN               → ascendc.data_copy_l2"
echo "    memref.copy A1→A2                  → ascendc.load_data_l0"
echo "    memref.copy B1→B2                  → ascendc.load_data_with_transpose"
echo "    memref.copy CO1→VECIN              → ascendc.data_copy_co12dst"
echo "    memref.copy VECOUT→GM              → ascendc.data_copy_l2（dst 为 global_tensor）"
echo "  同时插入 TQue/TPipe 管理（alloc/enque/deque/free/init_buffer）"
$AFIR_OPT "$DIR/output_step3_buffer_placement.mlir" \
  --linalg-to-ascendc \
  --canonicalize \
  --cse \
  -o "$DIR/output_step4_lowering_to_asc.mlir" 2>&1
echo "  ✓ Linalg→AscendC 成功，输出: output_step4_lowering_to_asc.mlir"
echo ""
echo "  [生成的 AscendC compute ops]"
grep -E "ascendc\.(mmad|add_l2|duplicate_l2|max_l2|data_copy|load_data)" \
  "$DIR/output_step4_lowering_to_asc.mlir" | head -20 || \
  echo "  (未找到 ascendc compute ops，请检查输出)"

# ── STAGE 5: AscendC Parallelize ───────────────────────────
echo ""
echo "[STAGE 5] Parallelize：--ascendc-parallelize"
echo "  输入: output_step4_lowering_to_asc.mlir"
echo "  将最外两层 scf.for（TB_M × TB_N）转换为多核 AiCore 调度："
echo "    %block_idx = ascendc.get_block_idx"
echo "    %i         = arith.divui %block_idx, %num_blks_N  (× TB_M → row offset)"
echo "    %j         = arith.remui %block_idx, %num_blks_N  (× TB_N → col offset)"
echo "    scf.if (inbound)  ← 越界 block 直接跳过（guard）"
$AFIR_OPT "$DIR/output_step4_lowering_to_asc.mlir" \
  --ascendc-parallelize \
  --canonicalize \
  --cse \
  -o "$DIR/output_step5_parallelize.mlir" 2>&1
echo "  ✓ Parallelize 成功，输出: output_step5_parallelize.mlir"
echo ""
echo "  [get_block_idx + divui/remui]"
grep -E "get_block_idx|divui|remui" "$DIR/output_step5_parallelize.mlir" | head -8 || \
  echo "  (未找到多核调度 ops)"

# ── STAGE 6: Prepare For Emit ──────────────────────────────
echo ""
echo "[STAGE 6] Prepare For Emit：--ascendc-prepare-for-emit"
echo "  输入: output_step5_parallelize.mlir"
echo "  转换规则："
echo "    memref<?, strided<...>> 参数 → memref<?, 22>（__gm__ 指针）"
echo "    i64 tiling 参数 → TilingData struct GM 指针"
echo "    在函数入口插入 emitasc.copy_struct + emitasc.member_ref 解包字段"
echo "    添加 {ascendc.aicore, ascendc.global} 属性"
echo "    去除函数返回值（kernel 返回 void）"
$AFIR_OPT "$DIR/output_step5_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --lower-affine \
  --canonicalize \
  --cse \
  -o "$DIR/output_step6_kernel.mlir" 2>&1
echo "  ✓ Prepare For Emit 成功，输出: output_step6_kernel.mlir"
echo ""
echo "  [函数签名 + 属性]"
grep -E "func\.func|ascendc\.(aicore|global)|emitasc\.(copy_struct|member_ref|declare)" \
  "$DIR/output_step6_kernel.mlir" | head -10 || \
  echo "  (请检查输出)"

# ── STAGE 7: AscendC C++ Code Generation ───────────────────
echo ""
echo "[STAGE 7] Codegen：ascir-translate -mlir-to-ascendc"
echo "  输入: output_step6_kernel.mlir"
echo "  输出: output_step7_kernel.cpp（AscendC C++ kernel 源码）"
ASCIR_TRANSLATE="${ASCIR_TRANSLATE:-ascir-translate}"
if command -v "$ASCIR_TRANSLATE" &>/dev/null; then
  "$ASCIR_TRANSLATE" -mlir-to-ascendc \
    "$DIR/output_step6_kernel.mlir" \
    -o "$DIR/output_step7_kernel.cpp" 2>&1
  echo "  ✓ Codegen 成功，输出: output_step7_kernel.cpp"
  echo ""
  echo "  [生成的 C++ kernel 头部]"
  head -30 "$DIR/output_step7_kernel.cpp"
else
  echo "  (ascir-translate 未找到，跳过 Stage 7)"
  echo "  若已构建 pyasc，请将 ascir-translate 加入 PATH 后重新运行。"
fi

echo ""
echo "========================================================"
echo " 流水线完成！生成文件："
echo "   output_step1_tile_and_fuse_3level.mlir → 3级 Tile+Fuse（linalg-on-tensor）"
echo "   output_step2_bufferized.mlir           → Bufferize 后（memref）"
echo "   output_step3_buffer_placement.mlir     → on-chip memory_space 标注 + memref.copy 占位"
echo "   output_step4_lowering_to_asc.mlir      → AscendC compute ops + TQue/TPipe"
echo "   output_step5_parallelize.mlir          → 多核 AiCore 调度（get_block_idx）"
echo "   output_step6_kernel.mlir               → 完整 AscendC kernel IR（可直接送 ascir-translate）"
echo "   output_step7_kernel.cpp                → AscendC C++ kernel 源码"
echo "========================================================"
