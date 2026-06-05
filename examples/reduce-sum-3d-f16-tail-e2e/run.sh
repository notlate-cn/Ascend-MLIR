#!/bin/bash
# 3D reduce-sum f16 — tail-peel via ragged-tail (DataCopyPad).  Re-enabled
# 2026-06-05 after the overlap->ragged redesign: the tail now processes the
# true remainder at honest offset/size and the GM store goes through
# DataCopyPad, so the unaligned f16 tail offset is legal and rows 48..49 are
# written.  Verifies session.validation=pass.

# ============================================================================
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

D0=5; D1=10; D2=8
ROWS=$(( D0 * D1 ))
# f16 DataCopy requires 32-byte alignment = 16 half elements per copy.
# XBLOCK_SUB must be a multiple of 16 for f16 output writes.
XBLOCK=16; XBLOCK_SUB=16
BLOCK_DIM=$(( (ROWS + XBLOCK - 1) / XBLOCK ))

echo "======================================================"
echo " reduce-sum-3D f16 E2E"
echo "======================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --d0 "$D0" --d1 "$D1" --d2 "$D2"

echo "[STAGE 1] MLIR codegen"
"$AFIR_OPT" "$DIR/reduce_sum_3d_f16.mlir" --auto-fuse-codegen \
  -o "$DIR/reduce_sum_3d_f16_kernel.mlir" 2>&1
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/reduce_sum_3d_f16_kernel.mlir" \
  -o "$DIR/reduce_sum_3d_f16_kernel.cpp" \
  --tiling-space-out "$DIR/tiling_space.json" 2>&1

"$PYTHON" - <<PYEOF
import json, pathlib
p = pathlib.Path("$DIR/tiling_space.json")
ts = json.loads(p.read_text())
for param in ts["tiling_params"]:
    if param["name"] == "XBLOCK":     param["values"] = [$XBLOCK]
    elif param["name"] == "XBLOCK_SUB": param["values"] = [$XBLOCK_SUB]
p.write_text(json.dumps(ts, indent=2))
PYEOF

echo "[STAGE 2] runtime-session compile"
BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output.npy"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

"$RUNTIME_SESSION" \
  --kernel "$DIR/reduce_sum_3d_f16_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name reduce_sum_3d_f16__v0 2>&1

TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TILING_PARAMS+=",dim_arg0_0=${D0},dim_arg0_1=${D1},dim_arg0_2=${D2}"
TILING_PARAMS+=",dim_arg1_0=${D0},dim_arg1_1=${D1}"

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
  "atol": 5e-3,
  "rtol": 5e-3
}
EOF

echo "[STAGE 3] sim run + verify"
"$RUNTIME_SESSION" --run-manifest "$RUN_MANIFEST" --run >"$VALIDATION_LOG" 2>&1
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]' "$VALIDATION_LOG" || true
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"
echo "  ✓ session.validation=pass"
