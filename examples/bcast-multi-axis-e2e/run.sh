#!/bin/bash
# Multi-axis broadcast E2E: out[d0,d1,d2] = b[d0,d1,d2] + a[d1]   (a broadcast on d0 & d2)
# Validates ascendc-decompose-multi-axis-broadcast end-to-end.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"; AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"; PYTHON="${PYTHON:-python3}"
D0=4; D1=8; D2=16
# d0 / d2 are broadcast axes → whole-dim in the tile (a replicated on-chip);
# d1 is block-distributed (XBLOCK / XBLOCK_SUB).
XBLOCK=${D1}; XBLOCK_SUB=${D1}
BLOCK_DIM=$(( (D1 + XBLOCK - 1) / XBLOCK ))
echo "=== reduce-multi-axis broadcast E2E ==="
"$PYTHON" "$DIR/gen_inputs.py" --outdir "$DIR" --d0 "$D0" --d1 "$D1" --d2 "$D2"
"$AFIR_OPT" "$DIR/bcast_multi_axis.mlir" --vector-plan-codegen -o "$DIR/bcast_multi_axis_kernel.mlir"
"$AFIR_TRANSLATE" -mlir-to-cann "$DIR/bcast_multi_axis_kernel.mlir" -o "$DIR/bcast_multi_axis_kernel.cpp" --tiling-space-out "$DIR/tiling_space.json"
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
"$RUNTIME_SESSION" --kernel "$DIR/bcast_multi_axis_kernel.cpp" --kernel-kind vec --output "$ART" --name bcast_multi_axis
TP="XBLOCK=${XBLOCK},XBLOCK_SUB=${XBLOCK_SUB}"
TP+=",dim_arg1_0=${D0},dim_arg1_1=${D1},dim_arg1_2=${D2},dim_arg0_0=${D1},dim_arg4_1=${D1},dim_arg4_2=${D2}"
cat > "$BUILD_DIR/run_manifest.json" <<MEOF
{
  "task_id": "main", "backend": "sim", "artifact_root": "${ART}",
  "inputs": [{"name":"a","path":"${DIR}/a.npy"},{"name":"b","path":"${DIR}/b.npy"}],
  "outputs": [{"name":"out","path":"${BUILD_DIR}/output.npy"}],
  "expected_outputs": [{"name":"out","path":"${DIR}/expected.npy"}],
  "tiling": {"schema":"${DIR}/tiling_space.json","params":"${TP}"},
  "block_dim": ${BLOCK_DIM}, "workspace_size": 16777216, "profiling": false, "atol": 1e-5, "rtol": 1e-5
}
MEOF
"$RUNTIME_SESSION" --run-manifest "$BUILD_DIR/run_manifest.json" --run > "$BUILD_DIR/runtime_session.log" 2>&1
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]\|^\[DRVSTUB' "$BUILD_DIR/runtime_session.log" || true
grep -q '^session.validation=pass$' "$BUILD_DIR/runtime_session.log" && echo "VALIDATION PASS" || echo "VALIDATION FAIL"
