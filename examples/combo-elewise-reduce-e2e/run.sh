#!/bin/bash
# Combo (elementwise + reduce) end-to-end via --auto-fuse-codegen.
# Computation: out[d0,d1] = sum_{d2}( a[d0,d1,d2] + b[d0,d1,d2] )
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

D0=4; D1=8; D2=32
ROWS=$(( D0 * D1 ))
XBLOCK=8; XBLOCK_SUB=8
BLOCK_DIM=$(( (ROWS + XBLOCK - 1) / XBLOCK ))

echo "========================================================"
echo " combo (elewise + reduce) E2E"
echo "========================================================"

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --d0 "$D0" --d1 "$D1" --d2 "$D2"

echo "[STAGE 1] MLIR codegen"
"$AFIR_OPT" "$DIR/combo_elewise_reduce.mlir" --auto-fuse-codegen \
  -o "$DIR/combo_kernel.mlir" 2>&1
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/combo_kernel.mlir" \
  -o "$DIR/combo_kernel.cpp" \
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
  --kernel "$DIR/combo_kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name combo_elewise_reduce__v0 2>&1

# Tiling params (from kernel_id signature):
#   XBLOCK, XBLOCK_SUB, dim_arg0_0, dim_arg0_1, dim_arg0_2,
#   dim_arg1_2, dim_arg2_0, dim_arg2_1
TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TILING_PARAMS+=",dim_arg0_0=${D0},dim_arg0_1=${D1},dim_arg0_2=${D2}"
TILING_PARAMS+=",dim_arg1_2=${D2}"
TILING_PARAMS+=",dim_arg2_0=${D0},dim_arg2_1=${D1}"

cat > "$RUN_MANIFEST" <<EOF
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [
    { "name": "a", "path": "${DIR}/a.npy" },
    { "name": "b", "path": "${DIR}/b.npy" }
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
  "atol": 1e-5,
  "rtol": 1e-5
}
EOF

echo "[STAGE 3] sim run + verify"
"$RUNTIME_SESSION" --run-manifest "$RUN_MANIFEST" --run >"$VALIDATION_LOG" 2>&1
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]' "$VALIDATION_LOG" || true
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"
echo "  ✓ session.validation=pass"
