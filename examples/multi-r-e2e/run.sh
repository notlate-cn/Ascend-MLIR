#!/bin/bash
# ============================================================
# Multi-reduce-axes (adjacent) 端到端流水线 Demo
#
# 用法:
#   source examples/env.sh
#   bash examples/multi-r-e2e/run.sh [--log]
#
# 计算图:  out[a] = sum_{r1, r2}( x[a, r1, r2] )
#
# Shape:  x[A, R1, R2] f32  →  out[A] f32  (axis=1,2 reduce)
#
# Iterator types: ["parallel", "reduction", "reduction"].  Adjacent reduce
# axes collapse to a single R via the Collapse pass:
#   tensor<8x16x32xf32> → tensor<8x512xf32>
# After collapse the kernel is 2D AR FullLoad — same path as
# leading-reduce / reduce-sum-3d.
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

A=8; R1=16; R2=32
XBLOCK=8; XBLOCK_SUB=8
BLOCK_DIM=$(( (A + XBLOCK - 1) / XBLOCK ))

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done
log() { $VERBOSE && echo "$@" || true; }

echo "========================================================"
echo " multi-R (adjacent) E2E: codegen → sim"
echo "========================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --a "$A" --r1 "$R1" --r2 "$R2"

echo ""
echo "==================== [STAGE 1] MLIR → AscendC C++ ===================="
"$AFIR_OPT" "$DIR/multi_r.mlir" --vector-plan-codegen \
  -o "$DIR/multi_r_kernel.mlir" 2>&1
log "  ✓ MLIR codegen OK → multi_r_kernel.mlir"

# Assert Collapse merged r1+r2 (tile-fuse sees 2D iter space) and FullLoad picked.
grep -q 'afir.reduce_template = "FullLoad"' "$DIR/multi_r_kernel.mlir" \
  || { echo "ERROR: expected FullLoad template (multi-R should collapse)"; exit 1; }
echo "  ✓ Template assertion OK (FullLoad after Collapse)"

"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/multi_r_kernel.mlir" \
  -o "$DIR/multi_r_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1
echo "  ✓ Translate OK → multi_r_kernel.cpp + tiling_space.json"

"$PYTHON" - <<PYEOF
import json, pathlib
p = pathlib.Path("$DIR/tiling_space.json")
ts = json.loads(p.read_text())
for param in ts["tiling_params"]:
    if param["name"] == "XBLOCK":
        param["values"] = [$XBLOCK]
    elif param["name"] == "XBLOCK_SUB":
        param["values"] = [$XBLOCK_SUB]
p.write_text(json.dumps(ts, indent=2))
print("  ✓ tiling_space.json patched (XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB)")
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
  --kernel "$DIR/multi_r_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name multi_r_adj__v0 \
  2>&1
echo "  ✓ Compile OK → $ARTIFACT_ROOT"

echo ""
echo "==================== [STAGE 3] Simulator Run + Verify ===================="
log "  XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB, block_dim=$BLOCK_DIM"
log "  shape: x[$A,$R1,$R2] → out[$A]"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB},dim_arg0_1=${R1},dim_arg0_2=${R2}"

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
