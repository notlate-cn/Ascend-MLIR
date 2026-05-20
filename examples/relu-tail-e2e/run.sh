#!/bin/bash
# Elementwise relu with sub-aligned tail.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"

N=1010
XBLOCK=128; XBLOCK_SUB=16
BLOCK_DIM=$(( (N + XBLOCK - 1) / XBLOCK ))

echo "===================================================="
echo " relu-tail E2E: N=$N, XBLOCK=$XBLOCK, XBLOCK_SUB=$XBLOCK_SUB"
echo "===================================================="

"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --n "$N"

"$AFIR_OPT" "$DIR/relu.mlir" --auto-fuse-codegen -o "$DIR/relu_kernel.mlir" 2>&1
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/relu_kernel.mlir" -o "$DIR/relu_kernel.cpp" \
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

BUILD_DIR="$DIR/build_e2e"
rm -fr "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT="$BUILD_DIR/output.npy"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"

"$RUNTIME_SESSION" --kernel "$DIR/relu_kernel.cpp" --kernel-kind vec \
  --output "$ARTIFACT_ROOT" --name relu__v0 2>&1

TILING_PARAMS="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TILING_PARAMS+=",dim_arg0_0=${N}"

cat > "$RUN_MANIFEST" <<E2
{
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "${ARTIFACT_ROOT}",
  "inputs": [ { "name": "input", "path": "${DIR}/input.npy" } ],
  "outputs": [ { "name": "out", "path": "${ACTUAL_OUTPUT}" } ],
  "expected_outputs": [ { "name": "out", "path": "${DIR}/expected.npy" } ],
  "tiling": {
    "schema": "${DIR}/tiling_space.json",
    "params": "${TILING_PARAMS}"
  },
  "block_dim": ${BLOCK_DIM},
  "workspace_size": 16777216,
  "atol": 1e-5,
  "rtol": 1e-5
}
E2

"$RUNTIME_SESSION" --run-manifest "$RUN_MANIFEST" --run >"$VALIDATION_LOG" 2>&1
grep -E "session\." "$VALIDATION_LOG" | head -3
grep -q '^session.validation=pass$' "$VALIDATION_LOG"
echo "  ✓ session.validation=pass"
