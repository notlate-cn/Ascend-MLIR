#!/bin/bash
# ============================================================
# gather + elementwise fusion complete pipeline demo
#
# Usage:
#   source examples/env.sh
#   bash examples/gather-elementwise-fusion/run.sh [--log]
#
# Graph: relu -> index_select(dim=1) -> add
#   data[M,N] --relu--> gathered via index_select(dim=1, indices[K])
#   gathered[M,K] + bias[K] --> out[M,K]
#
# Shapes (参数设计规则):
#   M=16   (single TB_M=16 block; avoids current multi-block Gather UB on real NPU)
#   N=640  (gather source width, N >= K)
#   K=128  (gather output width, K >= 16 for DataCopy alignment)
#   indices in [0, 624), avoiding the final 32B half datablock on real NPU
#   TB_M=16, TB_N=16 (one 16-row tile per block; each row data[N] fits in UB VECCALC)
#   block_dim = M / TB_M = 1
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done

log() { if $VERBOSE; then echo "$@"; fi }

clear 2>/dev/null || true

echo "========================================================"
echo " gather + elementwise fusion pipeline"
echo "========================================================"

echo ""
echo "==================== [STAGE 0] Parse ===================="
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir"
log "  ok: step0_input_out.mlir"

echo ""
echo "==================== [STAGE 1] --mark-structured-ops ===================="
$AFIR_OPT --mark-structured-ops \
  "$DIR/step0_input.mlir" \
  -o "$DIR/step1_marked.mlir"
log "  ok: step1_marked.mlir"
log "  [gather_dim]"
log "$(grep 'gather_dim' "$DIR/step1_marked.mlir" || echo '  (not found)')"

echo ""
echo "==================== [STAGE 1.5] --fuse-gather-elementwise ===================="
$AFIR_OPT --fuse-gather-elementwise \
  "$DIR/step1_marked.mlir" \
  -o "$DIR/step1b_fused.mlir"
log "  ok: step1b_fused.mlir"

echo ""
echo "==================== [STAGE 2] --transform-interpreter ===================="
"$AFIR_OPT" "$DIR/step1b_fused.mlir" \
  "--transform-preload-library=transform-library-paths=$DIR/step2_transform.mlir" \
  "--transform-interpreter=entry-point=__transform_main" \
  --canonicalize --cse \
  -o "$DIR/step2_tiled.mlir"
log "  ok: step2_tiled.mlir"

echo ""
echo "==================== [STAGE 3] --one-shot-bufferize ===================="
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir"
log "  ok: step3_bufferized.mlir"

echo ""
echo "==================== [STAGE 4] --ascendc-buffer-placement ===================="
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir"
log "  ok: step4_buffer_placement.mlir"

echo ""
echo "==================== [STAGE 5] --linalg-to-ascendc ===================="
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize --cse \
  -o "$DIR/step5_ascendc.mlir"
log "  ok: step5_ascendc.mlir"

echo ""
echo "==================== [STAGE 6] --ascendc-parallelize ===================="
$AFIR_OPT "$DIR/step5_ascendc.mlir" \
  --ascendc-parallelize \
  --canonicalize --cse \
  -o "$DIR/step6_parallelize.mlir"
log "  ok: step6_parallelize.mlir"

echo ""
echo "==================== [STAGE 7] --ascendc-prepare-for-emit ===================="
$AFIR_OPT "$DIR/step6_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --canonicalize --cse \
  -o "$DIR/step7_kernel.mlir"
log "  ok: step7_kernel.mlir"

echo ""
echo "==================== [STAGE 7b] --canonicalize-cann-signature ===================="
$AFIR_OPT --canonicalize-cann-signature \
  "$DIR/step7_kernel.mlir" \
  -o "$DIR/step7_cann.mlir"
log "  ok: step7_cann.mlir"

echo ""
echo "==================== [STAGE 8] afir-translate -mlir-to-cann ===================="
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
# Generate into step8_kernel_gen.cpp and compile it directly.
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/step7_cann.mlir" \
  -o "$DIR/step8_kernel_gen.cpp"
log "  ok: step8_kernel_gen.cpp"

echo ""
echo "==================== [STAGE 8b] 生成测试数据：gen_data.py ===================="
log "  M=16, N=640, K=128, index_high=624, seed=42"
"$PYTHON" "$DIR/gen_data.py" --m 16 --n 640 --k 128 --index-high 624 --seed 42 --out-dir "$DIR"
log "  ok: input_data.npy, input_indices.npy, input_bias.npy, output_out.npy"

echo ""
echo "==================== [STAGE 9] Compile：runtime-session ===================="
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output.npy"
"$RUNTIME_SESSION" \
  --kernel "$DIR/step8_kernel_gen.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name relu_index_select_add \
  2>&1
log "  ok: $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 10] Run + Verify ===================="
log "  TB_M=16, TB_N=16, M=16, N=640, K=128, index_high=624, block-dim=1"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"
cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "data", "path": "${DIR}/input_data.npy" },
    { "name": "indices", "path": "${DIR}/input_indices.npy" },
    { "name": "bias", "path": "${DIR}/input_bias.npy" }
  ],
  "outputs": [
    { "name": "out", "path": "${ACTUAL_OUTPUT}" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "${DIR}/output_out.npy" }
  ],
  "tiling": {
    "schema": "${DIR}/tiling_space.json",
    "params": "TB_M=16,TB_N=16,dim_arg0_0=16,dim_arg1_0=128,dim_arg0_1=640,dim_arg2_0=128"
  },
  "block_dim": 1,
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
echo "   step1_marked.mlir           → mark-structured-ops (gather_dim stamped)"
echo "   step1b_fused.mlir           → fuse-gather-elementwise"
echo "   step2_tiled.mlir            → Tiling 后 (TB/Tb 两级循环)"
echo "   step3_bufferized.mlir       → Bufferize 后 (memref)"
echo "   step4_buffer_placement.mlir → on-chip 内存标注"
echo "   step5_ascendc.mlir          → AscendC compute ops"
echo "   step6_parallelize.mlir      → 多核 AiCore 调度 (get_block_idx)"
echo "   step7_kernel.mlir           → 完整 AscendC kernel IR"
echo "   step7_cann.mlir             → CANN 标准签名 IR"
echo "   step8_kernel_gen.cpp        → AscendC C++ kernel 源码"
echo "   tiling_space.json           → tiling 参数空间"
echo "   input_data.npy              → data[16,640] f16"
echo "   input_indices.npy           → indices[128] i64"
echo "   input_bias.npy              → bias[128] f16"
echo "   output_out.npy              → expected out[16,128] f16"
echo "   build_e2e/artifact               → runtime-session 编译产物"
echo "   build_e2e/output.npy            → 仿真输出"
echo "========================================================"

rm -fr *.dump *.toml
