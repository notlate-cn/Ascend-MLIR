#!/usr/bin/env bash
# examples/matmul-add-leakyrelu/run.sh
# End-to-end pipeline: linalg IR -> AscendC kernel -> runtime-session mix validation
#
# Usage (on xvm):
#   source examples/env.sh
#   bash examples/matmul-add-leakyrelu/run.sh [--log]
#
# Dependencies: afir-opt, afir-translate, clang++, llvm-config, python3, numpy
set -euo pipefail
export ASCEND_DAV_SIM_VERSION=dav_3002

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/scripts/resolve_llvm_env.sh"

AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"

SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
BOOTSTRAP_BUILD_DIR="${BOOTSTRAP_BUILD_DIR:-${REPO_ROOT}/build/runtime-mix-bootstrap}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/build/runtime-mix-matmul-add-leakyrelu}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/build/runtime-mix-matmul-add-leakyrelu-data}"

VERBOSE=false
for arg in "$@"; do [[ $arg == "--log" ]] && VERBOSE=true; done
log() { $VERBOSE && echo "$@" || true; }

LLVM_BUILD_DIR="$(resolve_llvm_build_dir || true)"

echo "========================================================"
echo " matmul + add(bias[N]) + leaky_relu AFIR pipeline"
echo "========================================================"

# ── STAGE 2: Transform Tiling ────────────────────────────────────────────────
echo ""
echo "=== [STAGE 2] Transform Tiling ==="
$AFIR_OPT --transform-interpreter \
  "$SCRIPT_DIR/step2_transform.mlir" \
  --canonicalize --cse \
  -o "$SCRIPT_DIR/step2_tiled.mlir"
log "  step2_tiled.mlir done"

# ── STAGE 3: Bufferize ───────────────────────────────────────────────────────
echo "=== [STAGE 3] Bufferize ==="
$AFIR_OPT \
  '--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map' \
  "$SCRIPT_DIR/step2_tiled.mlir" \
  --annotate-ascendc-kernel-kind \
  --cse \
  -o "$SCRIPT_DIR/step3_bufferized.mlir"
log "  step3_bufferized.mlir done (kernel_kind annotated)"

# ── STAGE 4: Buffer Placement ────────────────────────────────────────────────
echo "=== [STAGE 4] Buffer Placement ==="
$AFIR_OPT --ascendc-buffer-placement \
  "$SCRIPT_DIR/step3_bufferized.mlir" \
  -o "$SCRIPT_DIR/step4_buffer_placement.mlir"
log "  step4_buffer_placement.mlir done"

# ── STAGE 5: linalg -> AscendC ──────────────────────────────────────────────
echo "=== [STAGE 5] linalg-to-ascendc ==="
$AFIR_OPT --linalg-to-ascendc \
  "$SCRIPT_DIR/step4_buffer_placement.mlir" \
  --canonicalize --cse \
  -o "$SCRIPT_DIR/step5_ascendc.mlir"
log "  step5_ascendc.mlir done"

# ── STAGE 6: Parallelize ─────────────────────────────────────────────────────
echo "=== [STAGE 6] Parallelize ==="
$AFIR_OPT --ascendc-parallelize \
  "$SCRIPT_DIR/step5_ascendc.mlir" \
  --canonicalize --cse \
  -o "$SCRIPT_DIR/step6_parallelize.mlir"
log "  step6_parallelize.mlir done"

# ── STAGE 7: Prepare for emit ────────────────────────────────────────────────
echo "=== [STAGE 7] Prepare for emit ==="
$AFIR_OPT --ascendc-prepare-for-emit \
  "$SCRIPT_DIR/step6_parallelize.mlir" \
  --canonicalize --cse \
  -o "$SCRIPT_DIR/step7_kernel.mlir"
$AFIR_OPT --canonicalize-cann-signature \
  "$SCRIPT_DIR/step7_kernel.mlir" \
  -o "$SCRIPT_DIR/step7_cann.mlir"
log "  step7_kernel.mlir, step7_cann.mlir done"

# ── STAGE 8: Codegen ─────────────────────────────────────────────────────────
echo "=== [STAGE 8] Codegen ==="
$AFIR_TRANSLATE -mlir-to-cann \
  "$SCRIPT_DIR/step7_cann.mlir" \
  -o "$SCRIPT_DIR/step8_kernel.cpp"
log "  step8_kernel.cpp done"

# ── Bootstrap mix-compiler + runtime-session ────────────────────────────────
echo ""
echo "=== Bootstrap RuntimeMix tools ==="

if [[ -z "${LLVM_BUILD_DIR}" ]]; then
  echo "Set LLVM_BUILD_DIR or provide an LLVM build with bin/llvm-config under:" >&2
  echo "  ${REPO_ROOT}/externals/llvm-project/build" >&2
  echo "  ${REPO_ROOT}/../llvm-project/llvm/build" >&2
  exit 2
fi

if [[ ! -x "${LLVM_BUILD_DIR}/bin/llvm-config" ]]; then
  echo "LLVM build dir is invalid (missing bin/llvm-config): ${LLVM_BUILD_DIR}" >&2
  exit 2
fi

mkdir -p "${BOOTSTRAP_BUILD_DIR}/bin"
if [[ -f "${BOOTSTRAP_BUILD_DIR}/CMakeCache.txt" ]]; then
  CACHE_SOURCE_DIR="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' \
    "${BOOTSTRAP_BUILD_DIR}/CMakeCache.txt")"
  if [[ -n "${CACHE_SOURCE_DIR}" && "${CACHE_SOURCE_DIR}" != "${REPO_ROOT}" ]]; then
    rm -rf "${BOOTSTRAP_BUILD_DIR}"
    mkdir -p "${BOOTSTRAP_BUILD_DIR}/bin"
  fi
fi

cmake -S "${REPO_ROOT}" -B "${BOOTSTRAP_BUILD_DIR}" \
  -DLLVM_BUILD_DIR="${LLVM_BUILD_DIR}" >/dev/null
cmake --build "${BOOTSTRAP_BUILD_DIR}" \
  --target mix-compiler runtime-session -j2 >/dev/null

rm -rf "${ARTIFACT_DIR}"

# ── Generate test data ────────────────────────────────────────────────────────
echo "=== [STAGE 9] Generate test data ==="
mkdir -p "${DATA_DIR}/input" "${DATA_DIR}/output" "${DATA_DIR}/npy"
python3 "$SCRIPT_DIR/gen_data.py" \
  --M 128 --K 256 --N 128 --seed 42 \
  --out-dir "${DATA_DIR}/npy"

# ── RuntimeMix compile ────────────────────────────────────────────────────────
echo "=== [STAGE 10] RuntimeMix compile ==="
MIX_COMPILE_LOG="${ARTIFACT_DIR}.mix-compiler.log"
if ! "${BOOTSTRAP_BUILD_DIR}/bin/mix-compiler" \
  --kernel "$SCRIPT_DIR/step8_kernel.cpp" \
  --cann-mlir "$SCRIPT_DIR/step7_cann.mlir" \
  --npy-dir "${DATA_DIR}/npy" \
  --output "${ARTIFACT_DIR}" \
  --soc "${SOC_VERSION}" >"${MIX_COMPILE_LOG}" 2>&1; then
  echo "FAIL: RuntimeMix compile failed" >&2
  cat "${MIX_COMPILE_LOG}" >&2 || true
  echo "--- mix compile diagnostics ---" >&2
  echo "artifact_dir=${ARTIFACT_DIR}" >&2
  echo "bootstrap_build_dir=${BOOTSTRAP_BUILD_DIR}" >&2
  echo "working_dir=$(pwd)" >&2
  for dir in \
    "${ARTIFACT_DIR}" \
    "${ARTIFACT_DIR}/work" \
    "${ARTIFACT_DIR}/work/preprocess_probe" \
    "${ARTIFACT_DIR}/host_dir" \
    "${ARTIFACT_DIR}/out"; do
    echo "--- ls ${dir} ---" >&2
    ls -la "${dir}" >&2 || true
  done
  echo "--- recent files under artifact_dir ---" >&2
  find "${ARTIFACT_DIR}" -maxdepth 4 -type f 2>/dev/null | sort | tail -n 80 >&2 || true
  exit 2
fi
cat "${MIX_COMPILE_LOG}"

python3 - "${ARTIFACT_DIR}/out/manifest.txt" "${ARTIFACT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
artifact_dir = Path(sys.argv[2])
manifest = {}
for raw in manifest_path.read_text().splitlines():
    line = raw.strip()
    if not line or "=" not in line:
        continue
    key, value = line.split("=", 1)
    manifest[key] = value
metadata_file = Path(manifest["metadata_path"])
if not metadata_file.is_absolute():
    metadata_file = (artifact_dir / metadata_file).resolve()
metadata = json.loads(metadata_file.read_text())
host_launch = metadata.get("host_launch", {})
helper_inputs = host_launch.get("helper_inputs", {})
print("=== [STAGE 10A] Tiling metadata ===")
print(f"  host_launch.mode        : {host_launch.get('mode', '')}")
print(f"  host_launch.helper_kind : {host_launch.get('helper_kind', '')}")
print(f"  tiling_backend          : {helper_inputs.get('tiling_backend', '')}")
print(f"  tiling_strategy         : {helper_inputs.get('tiling_strategy', '')}")
if helper_inputs.get("tiling_debug_note"):
    print(f"  tiling_debug_note       : {helper_inputs['tiling_debug_note']}")
PY

mapfile -t RUN_PATHS < <(python3 - \
  "${DATA_DIR}" "${ARTIFACT_DIR}" "${ARTIFACT_DIR}/out/manifest.txt" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

data_dir = Path(sys.argv[1])
artifact_dir = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
npy_dir = data_dir / "npy"
out_dir = data_dir / "output"
out_dir.mkdir(parents=True, exist_ok=True)

def read_manifest(path):
    manifest = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        manifest[key] = value
    return manifest

def load_mix_abi(artifact_dir, manifest):
    metadata_path = manifest.get("metadata_path")
    if metadata_path:
        metadata_file = Path(metadata_path)
        if not metadata_file.is_absolute():
            metadata_file = (artifact_dir / metadata_path).resolve()
        metadata = json.loads(metadata_file.read_text())
        abi = metadata["abi"]
        return {
            "inputs": abi["inputs"],
            "outputs": abi["outputs"],
            "workspace_bytes": int(abi["workspace_bytes"]),
            "block_dim": int(
                Path(
                    metadata["artifacts"]["launch_info_file_path"]
                    if Path(metadata["artifacts"]["launch_info_file_path"]).is_absolute()
                    else artifact_dir / metadata["artifacts"]["launch_info_file_path"]
                ).read_text().split("block_dim=", 1)[1].splitlines()[0]
            ),
        }

    inputs = []
    for idx in range(int(manifest["abi_input_count"])):
        inputs.append({
            "name": manifest[f"abi_input{idx}_name"],
        })
    outputs = []
    for idx in range(int(manifest["abi_output_count"])):
        outputs.append({
            "name": manifest[f"abi_output{idx}_name"],
        })
    return {
        "inputs": inputs,
        "outputs": outputs,
        "workspace_bytes": int(manifest.get("abi_workspace_bytes", "16777216")),
        "block_dim": int(manifest.get("abi_block_dim", "1")),
    }

def runtime_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float16):
        return "f16"
    if dtype == np.dtype(np.float32):
        return "f32"
    if dtype == np.dtype(np.float64):
        return "f64"
    if dtype == np.dtype(np.int8):
        return "int8"
    if dtype == np.dtype(np.int32):
        return "int32"
    if dtype == np.dtype(np.int64):
        return "int64"
    raise SystemExit(f"unsupported runtime dtype: {dtype}")

manifest = read_manifest(manifest_path)
mix_abi = load_mix_abi(artifact_dir, manifest)
input_count = len(mix_abi["inputs"])
inputs = []
for idx in range(input_count):
    name = mix_abi["inputs"][idx]["name"]
    npy_path = npy_dir / f"{name}.npy"
    if not npy_path.exists():
        npy_path = npy_dir / f"input{idx}.npy"
    if not npy_path.exists():
        raise SystemExit(f"missing input npy file for input {idx}")
    inputs.append({
        "name": name,
        "path": str(npy_path.resolve()),
    })

output_count = len(mix_abi["outputs"])
if output_count != 1:
    raise SystemExit(f"expected one output, got {output_count}")
output_name = mix_abi["outputs"][0]["name"]
output_npy = npy_dir / f"{output_name}.npy"
if not output_npy.exists():
    output_npy = npy_dir / "output0.npy"
if not output_npy.exists():
    output_npy = npy_dir / "output.npy"
if not output_npy.exists():
    raise SystemExit("missing golden output npy file")
golden = np.load(output_npy)

actual_output = out_dir / "output.npy"
run_manifest_path = data_dir / "runtime-manifest.json"
run_manifest = {
    "task_id": "main",
    "backend": "sim",
    "artifact_root": str(artifact_dir.resolve()),
    "inputs": inputs,
    "outputs": [
        {
            "name": output_name,
            "path": str(actual_output.resolve()),
            "shape": list(golden.shape),
            "dtype": runtime_dtype(golden.dtype),
        }
    ],
    "expected_outputs": [
        {
            "name": output_name,
            "path": str(output_npy.resolve()),
            "shape": list(golden.shape),
            "dtype": runtime_dtype(golden.dtype),
        }
    ],
    "tiling": {
        "binary": str((artifact_dir / "out" / "tiling.bin").resolve()),
    },
    "block_dim": mix_abi["block_dim"],
    "workspace_size": mix_abi["workspace_bytes"],
    "profiling": True,
    "atol": 1.0,
    "rtol": 1e-2,
}

run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
print(run_manifest_path.resolve())
print(actual_output.resolve())
print(output_npy.resolve())
PY
)
if [[ "${#RUN_PATHS[@]}" -ne 3 ]]; then
  echo "FAIL: expected 3 generated runtime paths, got ${#RUN_PATHS[@]}" >&2
  exit 2
fi
RUN_MANIFEST_PATH="${RUN_PATHS[0]}"
ACTUAL_OUTPUT_PATH="${RUN_PATHS[1]}"
GOLDEN_OUTPUT_PATH="${RUN_PATHS[2]}"

# ── runtime-session simulation ────────────────────────────────────────────────
echo "=== [STAGE 11] runtime-session ==="
[[ -f "${ARTIFACT_DIR}/out/tiling.bin" ]]

CANN_ARCH="$(uname -m)"
if [[ "${CANN_ARCH}" == "x86_64" ]]; then
  CANN_ARCH="x86_64-linux"
else
  CANN_ARCH="aarch64-linux"
fi
ASCEND_LIB64="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64"
SOC_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${SOC_VERSION}/lib"
DAV_SIM_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/simulator/${ASCEND_DAV_SIM_VERSION}/lib"
DEVICE_LIB="${ASCEND_HOME_PATH}/${CANN_ARCH}/lib64/device/lib64"

ASCEND_DAV_SIM_VERSION="${ASCEND_DAV_SIM_VERSION}" \
LD_LIBRARY_PATH="${ARTIFACT_DIR}/out:${ASCEND_LIB64}:${SOC_SIM_LIB}:${DAV_SIM_LIB}:${DEVICE_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${BOOTSTRAP_BUILD_DIR}/bin/runtime-session" \
  --run-manifest "${RUN_MANIFEST_PATH}" \
  --run

python3 - "${GOLDEN_OUTPUT_PATH}" "${ACTUAL_OUTPUT_PATH}" <<'PY'
import sys
import numpy as np

golden = np.load(sys.argv[1])
actual = np.load(sys.argv[2])
if golden.shape != actual.shape:
    raise SystemExit(f"shape mismatch: {actual.shape} vs {golden.shape}")
if golden.dtype != actual.dtype:
    raise SystemExit(f"dtype mismatch: {actual.dtype} vs {golden.dtype}")
diff = np.abs(actual - golden)
print(f"max_abs_diff={diff.max():.6e}")
print(f"mean_abs_diff={diff.mean():.6e}")
if not np.allclose(actual, golden, atol=1.0, rtol=1e-2):
    raise SystemExit("FAIL: outputs differ beyond tolerance")
print("PASS")
PY

echo "artifact_dir=${ARTIFACT_DIR}"
echo "data_dir=${DATA_DIR}"

rm -fr *.dump
rm -fr *.toml
