#!/bin/bash
# Leading-axis broadcast E2E: out[a,b,c] = x[a,b,c] + y[b,c]   (y broadcast over a)
# After collapse([1,2]): iteration space [a(=A, broadcast), bc(=B*C, parallel, block axis)].
# The broadcast axis a is an ordinary whole-dim parallel axis (no BCAST tunable);
# ascendc.broadcast_l2 replicates y on-chip.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"; AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"; PYTHON="${PYTHON:-python3}"
A=8; B=4; C=32
BC=$(( B * C ))
# Block axis is `bc`; pin XBLOCK >= BC so block_dim = 1.
XBLOCK=128; XBLOCK_SUB=128
BLOCK_DIM=$(( (BC + XBLOCK - 1) / XBLOCK ))
echo "=== bcast-leading E2E (out[a,b,c] = x[a,b,c] + y[b,c]) ==="
"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --a "$A" --b "$B" --c "$C"
"$AFIR_OPT" "$DIR/bcast_leading.mlir" --vector-plan-codegen -o "$DIR/bcast_leading_kernel.mlir"
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/bcast_leading_kernel.mlir" -o "$DIR/bcast_leading_kernel.cpp" --tiling-space-out "$DIR/tiling_space.json"
"$PYTHON" - <<PYEOF
import json, pathlib
p = pathlib.Path("$DIR/tiling_space.json"); ts = json.loads(p.read_text())
vals = {"XBLOCK": $XBLOCK, "XBLOCK_SUB": $XBLOCK_SUB}
for x in ts["tiling_params"]:
    if x["name"] in vals: x["values"] = [vals[x["name"]]]
p.write_text(json.dumps(ts, indent=2)); print("  tiling patched:", vals)
PYEOF
BUILD_DIR="$DIR/build_e2e"; rm -fr "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
ART="$BUILD_DIR/artifact"
"$RUNTIME_SESSION" --kernel "$DIR/bcast_leading_kernel.cpp" --kernel-kind vec --output "$ART" --name bcast_leading__v0
# tiling_params lists: XBLOCK, XBLOCK_SUB only (bcast static, no dim_arg fields).
# args: v1=x (8x4x32), v2=y (4x32), v3=out (8x4x32).
TP="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
cat > "$BUILD_DIR/run_manifest.json" <<MEOF
{
  "task_id": "main", "backend": "sim", "artifact_root": "${ART}",
  "inputs": [{"name":"x","path":"${DIR}/x.npy"},{"name":"y","path":"${DIR}/y.npy"}],
  "outputs": [{"name":"out","path":"${BUILD_DIR}/output.npy"}],
  "expected_outputs": [{"name":"out","path":"${DIR}/expected.npy"}],
  "tiling": {"schema":"${DIR}/tiling_space.json","params":"${TP}"},
  "block_dim": ${BLOCK_DIM}, "workspace_size": 16777216, "profiling": false, "atol": 1e-4, "rtol": 1e-4
}
MEOF
"$RUNTIME_SESSION" --run-manifest "$BUILD_DIR/run_manifest.json" --run > "$BUILD_DIR/runtime_session.log" 2>&1
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]\|^\[DRVSTUB' "$BUILD_DIR/runtime_session.log" || true
grep -q '^session.backend=sim$' "$BUILD_DIR/runtime_session.log"
grep -q '^session.result=success$' "$BUILD_DIR/runtime_session.log"
grep -q '^session.validation=pass$' "$BUILD_DIR/runtime_session.log"
echo "Done."
