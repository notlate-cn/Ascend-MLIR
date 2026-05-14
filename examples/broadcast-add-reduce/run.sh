#!/bin/bash
# ============================================================
# broadcast + add + reducesum 完整编译流水线 Demo
#
# 用法：
#   source examples/matmul-add-relu-sum/env.sh
#   bash examples/broadcast-add-reduce/run.sh [--log]
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
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

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

clear 2>/dev/null || true

echo "========================================================"
echo " broadcast + add + reducesum 编译流水线"
echo "========================================================"

# ── STAGE 0: 解析原始 IR ───────────────────────────────────
echo ""
echo "==================== [STAGE 0] 解析 High-Level IR（Broadcast + Add + ReduceSum）===================="
log "  输入: step0_input.mlir"
log "  Pass: (仅解析，无变换)"
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir" 2>&1
log "  ✓ 解析成功，输出: step0_input_out.mlir"
log ""
log "  [轴分组分析]"
log "    Op1 Broadcast: iterator = [Parallel, Parallel]   d0=P, d1=P"
log "    Op2 Add:       iterator = [Parallel, Parallel]   d0=P, d1=P"
log "    Op3 ReduceSum: iterator = [Parallel, Reduction]  d0=P, d1=R ← 冲突!"
log "    → d1 类型不一致（P vs R）"
log "    → 但 ReduceSum body 可内联 Broadcast+Add → 单 generic 融合"


# ── STAGE 1: 融合 ──────────────────────────────────────────
echo ""
echo "==================== [STAGE 1] 融合：--linalg-fuse-elementwise-ops ===================="
log "  输入: step0_input.mlir"
$AFIR_OPT --linalg-fuse-elementwise-ops "$DIR/step0_input.mlir" -o "$DIR/step1_fused.mlir" 2>&1
log "  ✓ 融合成功，输出: step1_fused.mlir"
log ""
log "  [融合结果]"
log "    3个 linalg.generic → 1个 generic"
log "    iterator = [\"parallel\", \"reduction\"]"
log "    body: acc += (A[d0] + B[d0,d1])   // Broadcast+Add 内联到规约体内"
log "$(cat "$DIR/step1_fused.mlir" | grep -A8 "linalg.generic")"


# ── STAGE 2: Transform Tiling ──────────────────────────────
echo ""
echo "==================== [STAGE 2] Tiling：--transform-interpreter ===================="
log "  输入: step2_transform.mlir（含 Transform 脚本）"
log "  策略: 沿 Parallel 轴(d0=M) 做两级切分 TB/Tb"
log "         Reduction 轴(d1=N) 按 chunk 切分并累加 partial sum"
$AFIR_OPT --transform-interpreter "$DIR/step2_transform.mlir" --canonicalize --cse -o "$DIR/step2_tiled.mlir" 2>&1
log "  ✓ Tiling 成功，输出: step2_tiled.mlir"
log ""
log "  [循环结构]"
log "$(grep -E "scf\.for|ascendc\." "$DIR/step2_tiled.mlir" | head -10)"


# ── STAGE 3: Bufferize ─────────────────────────────────────
# --one-shot-bufferize Pass参数介绍：
#   * bufferize-function-boundaries=true — 对函数边界也做 bufferize，即函数参数/返回值从 tensor 类型转为 memref 类型，否则只处理函数体内部
#   * allow-return-allocs-from-loops=true — 允许在循环内分配的 buffer 被返回（默认禁止，因为可能有性能问题）；
#                                           这里的 insert_slice/scf.yield 链会产生循环内 alloc，不开这个选项会报错
#   * function-boundary-type-conversion=identity-layout-map — 函数边界上的 memref 使用 identity layout（即 memref<?xf16> 而不是带 strided layout 的形式），
#                                           保持类型简洁；否则默认会带 fully-dynamic-layout-map
echo ""
echo "==================== [STAGE 3] Bufferize：--one-shot-bufferize ===================="
log "  输入: step2_tiled.mlir（tiling 已完成，ascendc.* attrs 已在 scf.for 上）"
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
log "  ✓ Bufferize 成功，输出: step3_bufferized.mlir"
log ""
log "  [ascendc.* attrs 是否保留在 scf.for 上]"
log "$(grep "ascendc\." "$DIR/step3_bufferized.mlir" | grep -v "transform\." | head -10)"
log "  [memref 类型]"
log "$(grep "memref" "$DIR/step3_bufferized.mlir" | grep "func.func" | head -3)"


# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "==================== [STAGE 4] Buffer Placement：--ascendc-buffer-placement ===================="
log "  输入: step3_bufferized.mlir"
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir" 2>&1
log "  ✓ Buffer Placement 成功，输出: step4_buffer_placement.mlir"
log ""
log "  [on-chip memory_space 标注]"
log "$(grep -E "memory_space|: i32" "$DIR/step4_buffer_placement.mlir" | head -10 || \
  echo "  (注：本算子无 matmul，buffer placement 主要标注 VECIN/VECOUT)")"


# ── STAGE 5: Linalg → AscendC Compute ─────────────────────
echo ""
echo "==================== [STAGE 5] Linalg → AscendC：--linalg-to-ascendc ===================="
log "  输入: step4_buffer_placement.mlir"
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize \
  --cse \
  -o "$DIR/step5_ascendc.mlir" 2>&1
log "  ✓ Linalg→AscendC 成功，输出: step5_ascendc.mlir"
log ""
log "  [生成的 AscendC ops]"
log "$(grep -E "ascendc\.(broadcast|add_l2|reduce_sum_2d|data_copy)" "$DIR/step5_ascendc.mlir" | head -20 || \
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


# ── STAGE 7b: Canonicalize CANN signature ──────────────────
echo ""
echo "==================== [STAGE 7b] CANN Signature：--canonicalize-cann-signature ===================="
log "  输入: step7_kernel.mlir"
log "  输出: step7_cann.mlir（CANN 标准签名，去除 transform ops）"
$AFIR_OPT --canonicalize-cann-signature \
  "$DIR/step7_kernel.mlir" \
  -o "$DIR/step7_cann.mlir" 2>&1
log "  ✓ CANN 签名规范化成功，输出: step7_cann.mlir"
log "$(grep -E 'func.func|cann.num_inputs|py_struct|memref<ui8>' "$DIR/step7_cann.mlir" | head -3)"

# ── STAGE 8: AscendC C++ Code Generation (CANN standard) ───
echo ""
echo "==================== [STAGE 8] Codegen：afir-translate -mlir-to-cann ===================="
log "  输入: step7_cann.mlir"
log "  输出: step8_kernel.cpp（CANN 标准 C++ kernel）"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/step7_cann.mlir" \
  -o "$DIR/step8_kernel.cpp" 2>&1
log "  ✓ Codegen 成功，输出: step8_kernel.cpp"
log ""
log "  [生成的 C++ kernel 头部]"
log "$(head -20 "$DIR/step8_kernel.cpp")"


# ── STAGE 8b: Generate test data ───────────────────────────────────────────
echo ""
echo "==================== [STAGE 8b] 生成测试数据：gen_data.py ===================="
log "  M=640, N=512, seed=42"
"$PYTHON" "$DIR/gen_data.py" --m 640 --n 512 --seed 42 --out-dir "$DIR" 2>&1
log "  ✓ 生成成功：input_a.npy, input_b.npy, output_c.npy"


# ── STAGE 9: Compile AscendC kernel ────────────────────────────────────────
echo ""
echo "==================== [STAGE 9] Compile：runtime-session ===================="
log "  输入: step8_kernel.cpp"
log "  输出: build_e2e/artifact"
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output.npy"
"$RUNTIME_SESSION" \
  --kernel "$DIR/step8_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name broadcast_add_reducesum__v0 \
  2>&1
log "  ✓ Compile 成功，输出: $ARTIFACT_ROOT"


# ── STAGE 10: Run and verify ────────────────────────────────────────────────
echo ""
echo "==================== [STAGE 10] Run + Verify ===================="
log "  使用参数：TB_M=16, TB_N=16, Tb_M=512, M=640, N=512, block-dim=40"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"
cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "a", "path": "${DIR}/input_a.npy" },
    { "name": "b", "path": "${DIR}/input_b.npy" }
  ],
  "outputs": [
    { "name": "out", "path": "${ACTUAL_OUTPUT}" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "${DIR}/output_c.npy" }
  ],
  "tiling": {
    "schema": "${DIR}/tiling_space.json",
    "params": "TB_M=16,TB_N=16,Tb_M=512,dim_arg0_0=640,dim_arg1_1=512,dim_arg1_0=640"
  },
  "block_dim": 40,
  "workspace_size": 16777216,
  "profiling": true,
  "atol": 10,
  "rtol": 1e-2
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
echo " 流水线完成！生成文件："
echo "   step0_input_out.mlir        → 解析后 IR"
echo "   step1_fused.mlir            → 融合后（3 generic → 1）"
echo "   step2_tiled.mlir            → Tiling 后（TB/Tb 两级循环）"
echo "   step3_bufferized.mlir       → Bufferize 后（memref）"
echo "   step4_buffer_placement.mlir → on-chip 内存标注"
echo "   step5_ascendc.mlir          → AscendC compute ops"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度（get_block_idx）"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step7_cann.mlir             → CANN 标准签名 IR（去除 transform ops）"
echo "   step8_kernel.cpp            → AscendC C++ kernel 源码"
echo "   tiling_space.json           → tiling 参数空间（手写维护）"
echo "   build_e2e/artifact               → runtime-session 编译产物"
echo "   build_e2e/output.npy            → 仿真输出"
echo "========================================================"

rm -fr *.dump
rm -fr *.toml
