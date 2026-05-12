#!/bin/bash
# Middle-axis broadcast E2E: out[a,b,c] = x[a,b,c] + y[a,c]   (y broadcast over b)
# Iteration space [a(parallel, block axis), b(broadcast), c(parallel, whole)] —
# a and c can't collapse (b sits between).  b is an ordinary whole-dim parallel
# axis (no BCAST tunable); ascendc.broadcast_l2 replicates y on-chip.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"; AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"; PYTHON="${PYTHON:-python3}"
A=8; B=16; C=32
# Block axis is `a`; pin XBLOCK = XBLOCK_SUB = A so block_dim = 1 and a single sub-tile.
XBLOCK=${A}; XBLOCK_SUB=${A}
BLOCK_DIM=$(( (A + XBLOCK - 1) / XBLOCK ))
echo "=== bcast-middle E2E (out[a,b,c] = x[a,b,c] + y[a,c]) ==="
"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --a "$A" --b "$B" --c "$C"
"$AFIR_OPT" "$DIR/bcast_middle.mlir" --vector-plan-codegen -o "$DIR/bcast_middle_kernel.mlir"
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/bcast_middle_kernel.mlir" -o "$DIR/bcast_middle_kernel.cpp" --tiling-space-out "$DIR/tiling_space.json"
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
"$RUNTIME_SESSION" --kernel "$DIR/bcast_middle_kernel.cpp" --kernel-kind vec --output "$ART" --name bcast_middle
# tiling_params lists: XBLOCK, XBLOCK_SUB, dim_arg0_1, dim_arg0_2, dim_arg1_1, dim_arg4_1, dim_arg4_2.
# args: v1=x (8x16x32), v2=y (8x32), v3=out (8x16x32); dim_arg0=x, dim_arg1=y, dim_arg4=out.
TP="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TP+=",dim_arg0_1=${B},dim_arg0_2=${C},dim_arg1_1=${C},dim_arg4_1=${B},dim_arg4_2=${C}"
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
