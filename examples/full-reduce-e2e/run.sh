#!/bin/bash
# ============================================================
# 1D full-reduce-sum 端到端流水线 Demo — RCore 模板 (多核 R 分核)
#
# 用法:
#   source examples/env.sh
#   bash examples/full-reduce-e2e/run.sh [--log]
#
# 计算图:  out = sum_{d0}( x[d0] )   (标量)
#
# Shape:  x[D0] f32  →  out f32
#
# Tiling:
#   D0 (= R, 唯一的 reduction 轴)  分核 (XBLOCK)、核内按 RBLOCK_0 切片.
#   block_dim = ceil(D0 / XBLOCK).
#
# 触发 RCore 路径:
#   1. linalg.generic 只有 reduction 迭代器、无 parallel 轴 → yAxes 空.
#   2. costEstimate L361: 对 reduceIsBlock + g.yAxes.empty() 返回 feasible.
#   3. TilePlan.reduceTemplate = RCore, R 轴 split 成 XBLOCK + RBLOCK_0.
#
# AscendCRCoreCombinePass 加的两段:
#   a. 每核 partial → workspace[block_idx + 64]  (workspace[0..256B] 给 soft-sync flag)
#   b. SyncAll<false>(soft sync via GM 计数器), block 0 scalar-sum + 写 out.
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

D0=256
XBLOCK=128; RBLOCK_0=64
BLOCK_DIM=$(( (D0 + XBLOCK - 1) / XBLOCK ))

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done
log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " full-reduce E2E (RCore + SyncAll soft sync): codegen → sim"
echo "========================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --d0 "$D0"

echo ""
echo "==================== [STAGE 1] MLIR → AscendC C++ ===================="
"$AFIR_OPT" "$DIR/full_reduce.mlir" --vector-plan-codegen \
  -o "$DIR/full_reduce_kernel.mlir" 2>&1
log "  ✓ MLIR codegen OK → full_reduce_kernel.mlir"

# Assert the picker landed on RCore and the combine pass ran.
grep -q 'afir.reduce_template = "RCore"' "$DIR/full_reduce_kernel.mlir" \
  || { echo "ERROR: expected RCore template, picker drifted"; exit 1; }
grep -q 'SyncAll<false>' "$DIR/full_reduce_kernel.mlir" \
  || { echo "ERROR: SyncAll soft-sync not emitted, RCoreCombinePass didn't fire"; exit 1; }
echo "  ✓ Template & combine pass assertions OK (RCore + SyncAll<false>)"

"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/full_reduce_kernel.mlir" \
  -o "$DIR/full_reduce_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
echo "  ✓ Translate OK → full_reduce_kernel.cpp + tiling_space.json"

"$PYTHON" - <<PYEOF
import json, pathlib
p = pathlib.Path("$DIR/tiling_space.json")
ts = json.loads(p.read_text())
for param in ts["tiling_params"]:
    if param["name"] == "XBLOCK":
        param["values"] = [$XBLOCK]
    elif param["name"] == "RBLOCK_0":
        param["values"] = [$RBLOCK_0]
p.write_text(json.dumps(ts, indent=2))
print("  ✓ tiling_space.json patched (XBLOCK=$XBLOCK, RBLOCK_0=$RBLOCK_0)")
PYEOF

echo ""
echo "==================== [STAGE 2] runtime-session compile ===================="
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output.npy"

"$RUNTIME_SESSION" \
  --kernel "$DIR/full_reduce_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name full_reduce__v0 \
  2>&1
echo "  ✓ Compile OK → $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 3] Simulator Run + Verify ===================="
log "  XBLOCK=$XBLOCK, RBLOCK_0=$RBLOCK_0, block_dim=$BLOCK_DIM"
log "  shape: x[$D0] → out (scalar)"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

TILING_PARAMS="XBLOCK=${XBLOCK},RBLOCK_0=${RBLOCK_0},dim_arg0_0=${D0}"

cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "x", "path": "${DIR}/x.npy" }
  ],
  "outputs": [
    { "name": "out", "path": "${ACTUAL_OUTPUT}" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "${DIR}/expected.npy" }
  ],
  "tiling": {
    "schema": "${DIR}/tiling_space.json",
    "params": "${TILING_PARAMS}"
  },
  "block_dim": ${BLOCK_DIM},
  "workspace_size": 16777216,
  "profiling": false,
  "atol": 1e-4,
  "rtol": 1e-4
}
EOF

"$RUNTIME_SESSION" \
  --run-manifest "$RUN_MANIFEST" \
  --run >"$VALIDATION_LOG" 2>&1
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]\|^\[DRVSTUB_LOG\]\|^\[DEBUG\]' "$VALIDATION_LOG" || true
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"

echo ""
echo "========================================================"
echo " Done. session.validation=pass"
echo "========================================================"
