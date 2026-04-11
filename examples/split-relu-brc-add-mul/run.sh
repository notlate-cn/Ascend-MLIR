#!/bin/bash
# ============================================================
# split-relu-brc-add-mul 完整编译 + CPU 仿真验证流水线
#
# 计算图：
#   input_a[M,N] → Split → a0[M/2,N], a1[M/2,N]
#   out0[M/2,N]  = (relu(a0) + bias0[M/2]_row_brc) * scale0[N]_col_brc
#   out1[M/2,N]  = (relu(a1) + bias1[M/2]_row_brc) * scale1[N]_col_brc
#   output[M,N]  = concat([out0, out1], axis=0)
#
# 输入 Shape (M=640, N=512, HM=M/2=320):
#   input_a [640, 512]  bias0 [320]  bias1 [320]
#   scale0  [512]       scale1[512]  output[640, 512]
#
# Tiling 策略 (UB 计算):
#   TB_M=16, TB_N=16 (inner batch = full TB), N 不切
#   block_dim = HM / TB_M = 320 / 16 = 20
#   UB 占用: 2 chains × (4 TQue + 8 TBuf) × 16 × 512 × 2B ≈ 196KB < 256KB
#
# 用法:
#   source examples/env.sh
#   bash examples/split-relu-brc-add-mul/run.sh [--log]
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
COMPILER="${COMPILER:-compiler}"
VALIDATOR="${VALIDATOR:-validator}"

M=640
N=512
HM=320   # M/2
TB_M=16
BLOCK_DIM=20  # HM / TB_M

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
echo " split-relu-brc-add-mul 编译 + CPU 仿真流水线"
echo " M=$M N=$N HM=$HM TB_M=$TB_M block_dim=$BLOCK_DIM"
echo "========================================================"

# ── STAGE 1: Fuse ──────────────────────────────────────────
echo ""
echo "==================== [STAGE 1] Fuse ===================="
$AFIR_OPT --linalg-fuse-elementwise-ops "$DIR/step0_input.mlir" \
  --canonicalize --cse \
  -o "$DIR/step1_fused.mlir" 2>&1
echo "  ✓ step1_fused.mlir"
log "  generics: $(grep -c 'linalg.generic' "$DIR/step1_fused.mlir")"

# ── STAGE 2: Transform Tiling ──────────────────────────────
echo ""
echo "==================== [STAGE 2] Tiling ===================="
$AFIR_OPT --transform-interpreter "$DIR/step2_transform.mlir" \
  --canonicalize --cse \
  -o "$DIR/step2_tiled.mlir" 2>&1
echo "  ✓ step2_tiled.mlir"
log "  return: $(grep 'return' "$DIR/step2_tiled.mlir" | grep -v transform | head -3)"

# ── STAGE 3: Bufferize ─────────────────────────────────────
echo ""
echo "==================== [STAGE 3] Bufferize ===================="
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir" 2>&1
echo "  ✓ step3_bufferized.mlir"
log "  allocs: $(grep -c 'memref.alloc' "$DIR/step3_bufferized.mlir")"

# ── STAGE 3b: Fold Concat Alloc ────────────────────────────
echo ""
echo "==================== [STAGE 3b] Fold Concat Alloc ===================="
$AFIR_OPT \
  --ascendc-fold-concat-alloc \
  --canonicalize --cse \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step3_bufferized.mlir" 2>&1
echo "  ✓ step3_bufferized.mlir (folded, 1 alloc)"
log "  allocs: $(grep -c 'memref.alloc' "$DIR/step3_bufferized.mlir")"
log "  memcpy: $(grep 'memref.copy' "$DIR/step3_bufferized.mlir" | wc -l)"

# ── STAGE 4: Buffer Placement ──────────────────────────────
echo ""
echo "==================== [STAGE 4] Buffer Placement ===================="
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir" 2>&1
echo "  ✓ step4_buffer_placement.mlir"

# ── STAGE 5: Linalg → AscendC ─────────────────────────────
echo ""
echo "==================== [STAGE 5] Linalg → AscendC ===================="
$AFIR_OPT \
  --linalg-to-ascendc \
  --canonicalize --cse \
  "$DIR/step4_buffer_placement.mlir" \
  -o "$DIR/step5_ascendc.mlir" 2>&1
echo "  ✓ step5_ascendc.mlir"

# ── STAGE 6: Parallelize ───────────────────────────────────
echo ""
echo "==================== [STAGE 6] Parallelize ===================="
$AFIR_OPT "$DIR/step5_ascendc.mlir" \
  --ascendc-parallelize \
  --canonicalize --cse \
  -o "$DIR/step6_parallelize.mlir" 2>&1
echo "  ✓ step6_parallelize.mlir"

# ── STAGE 7: Prepare For Emit ──────────────────────────────
echo ""
echo "==================== [STAGE 7] Prepare For Emit ===================="
$AFIR_OPT "$DIR/step6_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --canonicalize --cse \
  -o "$DIR/step7_kernel.mlir" 2>&1
echo "  ✓ step7_kernel.mlir"

# ── STAGE 7b: Canonicalize CANN Signature ──────────────────
echo ""
echo "==================== [STAGE 7b] CANN Signature ===================="
$AFIR_OPT --canonicalize-cann-signature \
  "$DIR/step7_kernel.mlir" \
  -o "$DIR/step7_cann.mlir" 2>&1
echo "  ✓ step7_cann.mlir"
log "  $(grep 'cann.num_inputs' "$DIR/step7_cann.mlir")"

# ── STAGE 8: C++ Code Generation ───────────────────────────
echo ""
echo "==================== [STAGE 8] Codegen ===================="
$AFIR_TRANSLATE -mlir-to-cann "$DIR/step7_cann.mlir" \
  -o "$DIR/step8_kernel.cpp" 2>&1
echo "  ✓ step8_kernel.cpp"

# ── STAGE 8b: Generate Test Data ───────────────────────────
echo ""
echo "==================== [STAGE 8b] Generate Test Data ===================="
DATA_DIR="$DIR"
python3 "$DIR/gen_data.py" --M $M --N $N --outdir "$DATA_DIR"
echo "  ✓ test_data/ (input_a bias0 bias1 scale0 scale1 output)"

# ── STAGE 9: Compile AscendC Kernel ────────────────────────
echo ""
echo "==================== [STAGE 9] Compile ===================="
BUILD_DIR="$DIR/build_e2e"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
$COMPILER \
  --kernel "$DIR/step8_kernel.cpp" \
  --output "$BUILD_DIR" \
  --name ewop_broadcast_split \
  --num-inputs 5 \
  --num-outputs 1 2>&1
echo "  ✓ $BUILD_DIR/ewop_broadcast_split.bin"

# ── STAGE 10: Run + Verify ─────────────────────────────────
echo ""
echo "==================== [STAGE 10] Run + Verify ===================="
BIN="$BUILD_DIR/ewop_broadcast_split.bin"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

TILING_PARAMS="TB_M=${TB_M},TB_N=${TB_M},dim_arg0_1=${N},dim_arg1_0=${HM},dim_arg0_0=${M},dim_arg1_1=${HM},dim_arg3_0=${N},dim_arg3_1=${N},dim_arg2_0=${HM},dim_arg2_1=${HM},dim_arg4_0=${N},dim_arg4_1=${N}"
# Note: N must be a multiple of 16 (AscendC DataCopy alignment for f16).

$VALIDATOR \
  --bin "$BIN" \
  --name ewop_broadcast_split \
  --inputs "$DATA_DIR/input_a.npy,$DATA_DIR/bias0.npy,$DATA_DIR/bias1.npy,$DATA_DIR/scale0.npy,$DATA_DIR/scale1.npy" \
  --expected "$DATA_DIR/output.npy" \
  --tiling-schema "$DIR/tiling_space.json" \
  --tiling-params "$TILING_PARAMS" \
  --block-dim $BLOCK_DIM \
  --atol 1e-2 \
  --rtol 1e-2 \
  --dump-actual "$BUILD_DIR/actual.txt" \
  --dump-expected "$BUILD_DIR/expected.txt" \
  2>&1 | tee "$VALIDATION_LOG"
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"

echo ""
echo "========================================================"
echo " 全流程完成！"
echo " M=$M N=$N HM=$HM TB_M=$TB_M block_dim=$BLOCK_DIM"
echo " CPU 仿真精度验证 atol=1e-2 rtol=1e-2"
echo "========================================================"

rm -fr *.dump
rm -fr *.toml
