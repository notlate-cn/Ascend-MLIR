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
COMPILER="${COMPILER:-compiler}"
VALIDATOR="${VALIDATOR:-validator}"

M=512
K=256
N=640
TB_M=128
TB_N=128
Tb_M=64
Tb_N=128
t_K=64
BLOCK_DIM=20   # (M/TB_M) * (N/TB_N) = 4*5

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

clear

echo "========================================================"
echo " matmul + add + relu 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "==================== [STAGE 0] 解析 High-Level IR（matmul + add + relu）===================="
log "  输入: fc_add_relu.mlir"
log "  Pass: (仅解析，无变换)"
log ""
log "  [原始计算图]"
log "    linalg.matmul:              C = A * B              （Cube 单元）"
log "    linalg.elementwise add:     D = C + bias           （Vector 单元）"
log "    linalg.fill + linalg.elementwise max_signed: E = max(D, 0)  （Vector 单元 / ReLU）"
log ""
log "  [Tile 策略（3级嵌套）]"
log "    Level 1（分核层）: TB_M × TB_N   → 映射到多 AiCore 并行（ascendc.parallel）"
log "    Level 2（片上层）: Tb_M × Tb_N   → 每个 AiCore 的 UB tile"
log "    Level 3（Cube K）: t_K           → 沿 K 轴分块，每次搬 A1→A2 / B1→B2"
log ""
log "  [内存层次]"
log "    GM → A1/B1  （GM->L1，分核入口批量搬）"
log "    A1 → A2     （L1->L0A，每次 K 迭代）"
log "    B1 → B2     （L1->L0B，每次 K 迭代）"
log "    matmul 累加 → CO1（L0C）"
log "    CO1 → VECIN （fixpipe，K 轴完成后）"
log "    VECIN + bias → VECCALC  （add）"
log "    VECCALC vs VECIN(0) → VECOUT  （max / ReLU）"
log "    VECOUT → GM  （分核出口写回）"


# ── STAGE 1: Transform Tiling ──────────────────────────────
echo ""
echo "==================== [STAGE 1] 3级 Tile+Fuse：--transform-interpreter ===================="
log "  输入: fc_add_relu.mlir + transform_tile_and_fuse_3level.mlir"
log "  策略:"
log "    step 4-6: tile max [TB_M, TB_N] → for_TB_M/for_TB_N，fuse add/matmul into for_TB_N"
log "              标注 ascendc.parallel, prologue(GM->A1/B1/VECIN), epilogue(VECOUT->GM)"
log "    step 7-9: tile max [Tb_M, Tb_N] → for_Tb_M/for_Tb_N，fuse add/matmul into for_Tb_N"
log "    step 10:  tile matmul [0, 0, t_K] → for_K"
log "              标注 ascendc.unit=AiCore.Cube/Vector"
log "              标注 for_K prologue(A1->A2,B1->B2), epilogue(CO1->VECIN)"
log "    step 11:  hoist_loop_invariant_subsets（提升循环不变切片）"
$AFIR_OPT "$DIR/fc_add_relu.mlir" \
  --transform-preload-library="transform-library-paths=$DIR/transform_tile_and_fuse_3level.mlir" \
  --transform-interpreter="entry-point=__transform_main" \
  --canonicalize \
  --cse \
  -o "$DIR/output_step1_tile_and_fuse_3level.mlir" 2>&1
log "  ✓ Tiling 成功，输出: output_step1_tile_and_fuse_3level.mlir"
log ""
log "  [循环结构]"
log "$(grep -E "scf\.for|ascendc\." "$DIR/output_step1_tile_and_fuse_3level.mlir" | head -15)"


# ── STAGE 2: Bufferize ─────────────────────────────────────
echo ""
echo "==================== [STAGE 2] Bufferize：--one-shot-bufferize ===================="
log "  输入: output_step1_tile_and_fuse_3level.mlir"
log "  将 tensor/extract_slice/insert_slice → memref/subview/copy"
log "  ascendc.* 属性保留在 scf.for 上"
$AFIR_OPT "$DIR/output_step1_tile_and_fuse_3level.mlir" \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true" \
  --buffer-deallocation-pipeline \
  -o "$DIR/output_step2_bufferized.mlir" 2>&1
log "  ✓ Bufferize 成功，输出: output_step2_bufferized.mlir"
log ""
log "  [ascendc.* attrs 是否保留在 scf.for 上]"
log "$(grep "ascendc\." "$DIR/output_step2_bufferized.mlir" | head -10)"
log "  [memref 类型（函数签名）]"
log "$(grep "func.func" "$DIR/output_step2_bufferized.mlir" | head -3)"


# ── STAGE 3: Buffer Placement ──────────────────────────────
echo ""
echo "==================== [STAGE 3] Buffer Placement：--ascendc-buffer-placement ===================="
log "  输入: output_step2_bufferized.mlir"
log "  推导规则（AscendCBufferPlacementPass）："
log "    Rule A: 解析 prologue/epilogue，建立各层搬运任务"
log "    Rule B: linalg.matmul → A2/B2/CO1；linalg.elementwise → VECCALC/VECOUT"
log "    Rule C: 传播 memory_space 到 subview，更新 alloc memory_space"
log "    Rule D: 插入 memref.copy 占位（GM↔A1/B1/VECIN/VECOUT, CO1→VECIN 等）"
log "    Rule E: 清除所有 ascendc.* 属性"
$AFIR_OPT "$DIR/output_step2_bufferized.mlir" \
  --ascendc-buffer-placement \
  --canonicalize \
  --cse \
  -o "$DIR/output_step3_buffer_placement.mlir" 2>&1
log "  ✓ Buffer Placement 成功，输出: output_step3_buffer_placement.mlir"
log ""
log "  [on-chip memory_space 标注]"
log "$(grep -E "[0-9]+ : i32" "$DIR/output_step3_buffer_placement.mlir" | head -15)"
log "  [memref.copy 占位搬运]"
log "$(grep "memref.copy" "$DIR/output_step3_buffer_placement.mlir" | head -10)"


# ── STAGE 4: Linalg → AscendC ──────────────────────────────
echo ""
echo "==================== [STAGE 4] Linalg → AscendC：--linalg-to-ascendc ===================="
log "  输入: output_step3_buffer_placement.mlir"
log "  转换规则（LinalgToAscendCPass）："
log "    linalg.matmul（A∈A2,B∈B2,C∈CO1）→ ascendc.mmad + load_data_l0/with_transpose"
log "    linalg.fill（outs∈CO1）           → （初始化由 mmad 内部处理，fill 消除）"
log "    linalg.elementwise add             → ascendc.add_l2"
log "    linalg.fill + elementwise max      → ascendc.duplicate_l2 + ascendc.max_l2"
log "    memref.copy GM→A1/B1               → ascendc.data_copy_nd2nz（含 nd2nz_params）"
log "    memref.copy GM→VECIN               → ascendc.data_copy_l2"
log "    memref.copy A1→A2                  → ascendc.load_data_l0"
log "    memref.copy B1→B2                  → ascendc.load_data_with_transpose"
log "    memref.copy CO1→VECIN              → ascendc.data_copy_co12dst"
log "    memref.copy VECOUT→GM              → ascendc.data_copy_l2（dst 为 global_tensor）"
log "  同时插入 TQue/TPipe 管理（alloc/enque/deque/free/init_buffer）"
$AFIR_OPT "$DIR/output_step3_buffer_placement.mlir" \
  --linalg-to-ascendc \
  --canonicalize \
  --cse \
  -o "$DIR/output_step4_lowering_to_asc.mlir" 2>&1
log "  ✓ Linalg→AscendC 成功，输出: output_step4_lowering_to_asc.mlir"
log ""
log "  [生成的 AscendC compute ops]"
log "$(grep -E "ascendc\.(mmad|add_l2|duplicate_l2|max_l2|data_copy|load_data)" \
  "$DIR/output_step4_lowering_to_asc.mlir" | head -20 || \
  echo "  (未找到 ascendc compute ops，请检查输出)")"


# ── STAGE 5: AscendC Parallelize ───────────────────────────
echo ""
echo "==================== [STAGE 5] Parallelize：--ascendc-parallelize ===================="
log "  输入: output_step4_lowering_to_asc.mlir"
log "  将最外两层 scf.for（TB_M × TB_N）转换为多核 AiCore 调度："
log "    %block_idx = ascendc.get_block_idx"
log "    %i         = arith.divui %block_idx, %num_blks_N  (× TB_M → row offset)"
log "    %j         = arith.remui %block_idx, %num_blks_N  (× TB_N → col offset)"
log "    scf.if (inbound)  ← 越界 block 直接跳过（guard）"
$AFIR_OPT "$DIR/output_step4_lowering_to_asc.mlir" \
  --ascendc-parallelize \
  --canonicalize \
  --cse \
  -o "$DIR/output_step5_parallelize.mlir" 2>&1
log "  ✓ Parallelize 成功，输出: output_step5_parallelize.mlir"
log ""
log "  [get_block_idx + divui/remui]"
log "$(grep -E "get_block_idx|divui|remui" "$DIR/output_step5_parallelize.mlir" | head -8 || \
  echo "  (未找到多核调度 ops)")"


# ── STAGE 6: Prepare For Emit ──────────────────────────────
echo ""
echo "==================== [STAGE 6] Prepare For Emit：--ascendc-prepare-for-emit ===================="
log "  输入: output_step5_parallelize.mlir"
log "  转换规则："
log "    memref<?, strided<...>> 参数 → memref<?, 22>（__gm__ 指针）"
log "    i64 tiling 参数 → TilingData struct GM 指针"
log "    在函数入口插入 emitasc.copy_struct + emitasc.member_ref 解包字段"
log "    添加 {ascendc.aicore, ascendc.global} 属性"
log "    去除函数返回值（kernel 返回 void）"
$AFIR_OPT "$DIR/output_step5_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --lower-affine \
  --canonicalize \
  --cse \
  -o "$DIR/output_step6_kernel.mlir" 2>&1
log "  ✓ Prepare For Emit 成功，输出: output_step6_kernel.mlir"
log ""
log "  [函数签名 + 属性]"
log "$(grep -E "func\.func|ascendc\.(aicore|global)|emitasc\.(copy_struct|member_ref|declare)" \
  "$DIR/output_step6_kernel.mlir" | head -10 || \
  echo "  (请检查输出)")"


# ── STAGE 7: Canonicalize CANN Signature ───────────────────
echo ""
echo "==================== [STAGE 7] CANN Signature：--canonicalize-cann-signature ===================="
log "  输入: output_step6_kernel.mlir"
log "  输出: output_step7_cann.mlir（CANN 标准签名）"
$AFIR_OPT --canonicalize-cann-signature \
  "$DIR/output_step6_kernel.mlir" \
  -o "$DIR/output_step7_cann.mlir" 2>&1
log "  ✓ CANN 签名规范化成功，输出: output_step7_cann.mlir"

# ── STAGE 7b: AscendC C++ Code Generation ──────────────────
echo ""
echo "==================== [STAGE 7b] Codegen：afir-translate -mlir-to-cann ===================="
log "  输入: output_step7_cann.mlir"
log "  输出: output_step7_kernel.cpp（AscendC C++ kernel 源码）"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/output_step7_cann.mlir" \
  -o "$DIR/output_step7_kernel.cpp" 2>&1
log "  ✓ Codegen 成功，输出: output_step7_kernel.cpp"
log ""
log "  [生成的 C++ kernel 头部]"
log "$(head -30 "$DIR/output_step7_kernel.cpp")"

# ── STAGE 8: Generate Test Data ────────────────────────────
echo ""
echo "==================== [STAGE 8] Generate Test Data ===================="
DATA_DIR="$DIR/test_data"
mkdir -p "$DATA_DIR"
python3 "$DIR/gen_data.py" --M $M --K $K --N $N --out-dir "$DATA_DIR"
echo "  ✓ test_data/ (input_a input_b input_bias output)"

# ── STAGE 9: Compile AscendC Kernel ────────────────────────
echo ""
echo "==================== [STAGE 9] Compile ===================="
BUILD_DIR="$DIR/build_e2e"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
$COMPILER \
  --kernel "$DIR/output_step7_kernel.cpp" \
  --output "$BUILD_DIR" \
  --name fc_relu \
  --num-inputs 3 \
  --num-outputs 1 2>&1
echo "  ✓ $BUILD_DIR/fc_relu.bin"

# ── STAGE 10: Run + Verify ─────────────────────────────────
echo ""
echo "==================== [STAGE 10] Run + Verify ===================="
BIN="$BUILD_DIR/fc_relu.bin"

TILING_PARAMS="TB_M=${TB_M},TB_N=${TB_N},Tb_M=${Tb_M},Tb_N=${Tb_N},t_K=${t_K},dim_arg0_0=${M},dim_arg0_1=${K},dim_arg1_0=${K},dim_arg1_1=${N},dim_arg2_0=${M},dim_arg2_1=${N},dim_arg3_0=${M},dim_arg3_1=${N}"

$VALIDATOR \
  --bin "$BIN" \
  --name fc_relu \
  --inputs "$DATA_DIR/input_a.npy,$DATA_DIR/input_b.npy,$DATA_DIR/input_bias.npy" \
  --expected "$DATA_DIR/output.npy" \
  --tiling-schema "$DIR/tiling_space.json" \
  --tiling-params "$TILING_PARAMS" \
  --block-dim $BLOCK_DIM \
  --atol 1e-3 \
  --rtol 1e-3 \
  --dump-actual "$BUILD_DIR/actual.txt" \
  --dump-expected "$BUILD_DIR/expected.txt" \
  2>&1 | grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]' || true

echo ""
echo "========================================================"
echo " 全流程完成！"
echo " M=$M K=$K N=$N  TB_M=$TB_M TB_N=$TB_N Tb_M=${Tb_M} Tb_N=${Tb_N} t_K=${t_K}"
echo " block_dim=$BLOCK_DIM  CPU 仿真精度验证 atol=1e-3 rtol=1e-3"
echo "========================================================"

rm -fr *.dump
rm -fr *.toml
