#!/bin/bash
# ============================================================
# fc_leakyrelu_mix kernel — step 2 validation
#
# 图结构 (对齐 BareMixInvocation 参考样例):
#   A[fp16, M×K] × B[fp16, K×N] + bias[fp32, N] → LeakyReLU(α=0.001) → out[fp32, M×N]
#
# 用法:
#   source examples/env.sh
#   bash examples/matmul-add-relu-sum/run_leakyrelu.sh
#
# ============================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
COMPILER="${COMPILER:-compiler}"
VALIDATOR="${VALIDATOR:-validator}"

M=128
K=256
N=128
BLOCK_DIM=1   # 1 AIC + 2 AIV

VERBOSE=false
for arg in "$@"; do
  case $arg in
    --log) VERBOSE=true ;;
  esac
done

log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " fc_leakyrelu_mix: fp16×fp16→fp32 + bias[N] + LeakyReLU"
echo " M=$M  K=$K  N=$N  block_dim=$BLOCK_DIM"
echo "========================================================"

# ── STAGE 1: Generate test data ──────────────────────────────
echo ""
echo "==================== [STAGE 1] Generate Test Data ===================="
DATA_DIR="$DIR/test_data_leakyrelu"
mkdir -p "$DATA_DIR"
python3 "$DIR/gen_data_leakyrelu.py" \
  --M $M --K $K --N $N \
  --out-dir "$DATA_DIR"
echo "  ✓ test_data_leakyrelu/ (input_a fp16, input_b fp16, input_bias fp32[N], output fp32)"

# ── STAGE 2: Generate TCubeTiling ────────────────────────────
echo ""
echo "==================== [STAGE 2] Generate TCubeTiling ===================="
TILING_JSON="$DIR/tiling_leakyrelu.json"
python3 "$DIR/gen_tiling_leakyrelu.py" \
  --M $M --K $K --N $N \
  --block-dim $BLOCK_DIM \
  --kernel-name fc_leakyrelu \
  --out-json "$TILING_JSON"
TILING_PARAMS=$(python3 "$DIR/gen_tiling_leakyrelu.py" \
  --M $M --K $K --N $N \
  --block-dim $BLOCK_DIM \
  --kernel-name fc_leakyrelu \
  --out-json "$TILING_JSON" \
  --print-tiling 2>/dev/null | tail -1)
echo "  ✓ tiling_leakyrelu.json"

# ── STAGE 3: Compile kernel ──────────────────────────────────
echo ""
echo "==================== [STAGE 3] Compile fc_leakyrelu_mix.cpp ===================="
BUILD_DIR="$DIR/build_leakyrelu"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
$COMPILER \
  --kernel  "$DIR/fc_leakyrelu_mix.cpp" \
  --output  "$BUILD_DIR" \
  --name    fc_leakyrelu \
  --kernel-type mix \
  --num-inputs  3 \
  --num-outputs 1 2>&1
echo "  ✓ $BUILD_DIR/fc_leakyrelu.bin"

# ── STAGE 4: Run + Verify ────────────────────────────────────
echo ""
echo "==================== [STAGE 4] Run + Verify ===================="
BIN="$BUILD_DIR/fc_leakyrelu.bin"
VALIDATOR_LOG="$BUILD_DIR/validator.log"
rm -f "$BUILD_DIR/actual.txt" "$BUILD_DIR/expected.txt" "$VALIDATOR_LOG"

$VALIDATOR \
  --bin       "$BIN" \
  --name      fc_leakyrelu \
  --kernel-type mix \
  --inputs    "$DATA_DIR/input_a.npy,$DATA_DIR/input_b.npy,$DATA_DIR/input_bias.npy" \
  --expected  "$DATA_DIR/output.npy" \
  --tiling-schema "$TILING_JSON" \
  --tiling-params "$TILING_PARAMS" \
  --block-dim $BLOCK_DIM \
  --atol 1.0 \
  --rtol 1e-2 \
  --dump-actual   "$BUILD_DIR/actual.txt" \
  --dump-expected "$BUILD_DIR/expected.txt" \
  >"$VALIDATOR_LOG" 2>&1 &
VALIDATOR_PID=$!
echo "  validator PID=$VALIDATOR_PID (log: $VALIDATOR_LOG)"

while kill -0 $VALIDATOR_PID 2>/dev/null; do
  if grep -q 'pem_lsu.cc:346\|pem_lsu.cc:381' "$VALIDATOR_LOG" 2>/dev/null; then
    kill $VALIDATOR_PID 2>/dev/null
    sleep 1
    break
  fi
  sleep 2
done
wait $VALIDATOR_PID 2>/dev/null || true

grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]\|pem_lsu\.cc' \
  "$VALIDATOR_LOG" 2>/dev/null || true

if [ -f "$BUILD_DIR/actual.txt" ]; then
  python3 - "$BUILD_DIR/actual.txt" "$BUILD_DIR/expected.txt" 1.0 1e-2 <<'PYEOF'
import sys, math
actual_f, expected_f, atol, rtol = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
def load(f):
    vals = []
    for l in open(f):
        l = l.strip()
        if l and not l.startswith('#'):
            vals.append(float(l))
    return vals
a, e = load(actual_f), load(expected_f)
if len(a) != len(e):
    print(f"FAIL  length mismatch: actual={len(a)} expected={len(e)}")
    sys.exit(1)
max_abs  = max(abs(ai - ei) for ai, ei in zip(a, e))
mean_abs = sum(abs(ai - ei) for ai, ei in zip(a, e)) / len(a)
passed   = all(abs(ai - ei) <= atol + rtol * abs(ei) for ai, ei in zip(a, e))
print(f"max_abs_diff:  {max_abs}")
print(f"mean_abs_diff: {mean_abs}")
print("PASS" if passed else "FAIL")
PYEOF
else
  echo "  (no actual.txt — kernel may not have produced output)"
fi

echo ""
echo "========================================================"
echo " 完成！M=$M K=$K N=$N  block_dim=$BLOCK_DIM"
echo "========================================================"

rm -f *.dump *.toml
