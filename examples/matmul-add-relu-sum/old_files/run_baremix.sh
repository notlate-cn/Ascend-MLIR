#!/bin/bash
# ============================================================
# baremix_custom reference sample — run via our toolchain
#
# Goal: prove our compiler + validator can handle a known-good
# kernel from the AscendC samples.
#
# Graph: A[fp16,M,K] x B[fp16,K,N] + bias[fp32,N] → LeakyReLU → out[fp32,M,N]
# M=128 K=256 N=128, AIC_1_2 (1 AIC + 2 AIV), blockDim=1
#
# Usage:
#   source examples/env.sh
#   bash examples/matmul-add-relu-sum/run_baremix.sh
# ============================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
COMPILER="${COMPILER:-compiler}"
VALIDATOR="${VALIDATOR:-validator}"

M=128
K=256
N=128
BLOCK_DIM=1

echo "========================================================"
echo " baremix_custom reference via our toolchain"
echo " M=$M  K=$K  N=$N  block_dim=$BLOCK_DIM"
echo "========================================================"

# ── STAGE 1: Generate test data ──────────────────────────────
echo ""
echo "==================== [STAGE 1] Generate Test Data ===================="
DATA_DIR="$DIR/test_data_baremix"
mkdir -p "$DATA_DIR"
python3 "$DIR/gen_data_baremix.py" \
  --M $M --K $K --N $N \
  --out-dir "$DATA_DIR"

# ── STAGE 2: Generate TCubeTiling ────────────────────────────
echo ""
echo "==================== [STAGE 2] Generate TCubeTiling ===================="
TILING_JSON="$DIR/tiling_baremix.json"
python3 "$DIR/gen_tiling_leakyrelu.py" \
  --M $M --K $K --N $N \
  --block-dim $BLOCK_DIM \
  --kernel-name baremix_custom \
  --out-json "$TILING_JSON"
TILING_PARAMS=$(python3 "$DIR/gen_tiling_leakyrelu.py" \
  --M $M --K $K --N $N \
  --block-dim $BLOCK_DIM \
  --kernel-name baremix_custom \
  --out-json "$TILING_JSON" \
  --print-tiling 2>/dev/null | tail -1)
echo "  ✓ tiling_baremix.json"

# ── STAGE 3: Compile ─────────────────────────────────────────
echo ""
echo "==================== [STAGE 3] Compile baremix_custom_wrap.cpp ===================="
BUILD_DIR="$DIR/build_baremix"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
$COMPILER \
  --kernel  "$DIR/baremix_custom_wrap.cpp" \
  --output  "$BUILD_DIR" \
  --name    baremix_custom \
  --kernel-type mix \
  --num-inputs  3 \
  --num-outputs 1 2>&1
echo "  ✓ $BUILD_DIR/baremix_custom.bin"

# ── STAGE 4: Run + Verify ────────────────────────────────────
echo ""
echo "==================== [STAGE 4] Run + Verify ===================="
BIN="$BUILD_DIR/baremix_custom.bin"
VALIDATOR_LOG="$BUILD_DIR/validator.log"
rm -f "$BUILD_DIR/actual.txt" "$BUILD_DIR/expected.txt" "$VALIDATOR_LOG"

# Run validator in foreground so we see output directly.
# For sim mode with AIC_1_2, expect ~2-3 minutes.
timeout 300 $VALIDATOR \
  --bin       "$BIN" \
  --name      baremix_custom \
  --kernel-type mix \
  --inputs    "$DATA_DIR/input_a.npy,$DATA_DIR/input_b.npy,$DATA_DIR/input_bias.npy" \
  --expected  "$DATA_DIR/output.npy" \
  --tiling-schema "$TILING_JSON" \
  --tiling-params "$TILING_PARAMS" \
  --block-dim $BLOCK_DIM \
  --workspace-size 16777216 \
  --atol 1.0 \
  --rtol 1e-2 \
  --dump-actual   "$BUILD_DIR/actual.txt" \
  --dump-expected "$BUILD_DIR/expected.txt" \
  >"$VALIDATOR_LOG" 2>&1 &
VPID=$!
echo "  validator PID=$VPID"

# Wait for completion, show progress
while kill -0 $VPID 2>/dev/null; do
  sleep 5
  # Show if model finished
  if grep -q 'Model Stop Time' "$VALIDATOR_LOG" 2>/dev/null; then
    echo "  (kernel execution finished, validator processing...)"
    break
  fi
done
wait $VPID 2>/dev/null
VEXIT=$?

# Show results
grep -E 'max_abs_diff|mean_abs_diff|PASS|FAIL|Error' "$VALIDATOR_LOG" 2>/dev/null || true

echo ""
echo "========================================================"
echo " Done (validator exit=$VEXIT)"
echo "========================================================"

rm -f *.dump *.toml
