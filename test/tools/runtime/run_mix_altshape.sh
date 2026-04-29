#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"
if [ -x /home/niu/code/llvm-project/llvm/build/bin/llvm-config ]; then
  export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
runtime_verify_setup_env
runtime_verify_prepare_build_dir
runtime_verify_build_runtime_core
runtime_verify_build_example_toolchain
runtime_verify_build_mix_compiler

EXAMPLE_DIR="${PROJECT_ROOT}/examples/matmul-add-leakyrelu"
RUNTIME_SESSION="${PROJECT_ROOT}/build/bin/runtime-session"
MIX_COMPILER="${PROJECT_ROOT}/build/bin/mix-compiler"

ARTIFACT_DIR="$(mktemp -d /tmp/runtime-mix-altshape-artifact.XXXXXX)"
DATA_DIR="$(mktemp -d /tmp/runtime-mix-altshape-data.XXXXXX)"

cleanup() {
  rm -rf "${ARTIFACT_DIR}" "${DATA_DIR}"
}
trap cleanup EXIT

mkdir -p "${DATA_DIR}/npy" "${DATA_DIR}/output"

python3 "${EXAMPLE_DIR}/gen_data.py" \
  --M 64 --K 128 --N 96 --seed 42 \
  --out-dir "${DATA_DIR}/npy" >/dev/null

"${MIX_COMPILER}" \
  --kernel "${EXAMPLE_DIR}/step8_kernel.cpp" \
  --cann-mlir "${EXAMPLE_DIR}/step7_cann.mlir" \
  --npy-dir "${DATA_DIR}/npy" \
  --output "${ARTIFACT_DIR}" \
  --soc "${SOC_VERSION}" >/tmp/runtime_mix_altshape_compile.log 2>&1

python3 - "${DATA_DIR}" "${ARTIFACT_DIR}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

data_dir = Path(sys.argv[1])
artifact_dir = Path(sys.argv[2])
manifest_path = artifact_dir / "out" / "manifest.txt"

manifest = {}
for raw in manifest_path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    manifest[key] = value

def load_mix_abi(artifact_dir, manifest):
    metadata_path = manifest.get("metadata_path")
    if metadata_path:
        metadata_file = Path(metadata_path)
        if not metadata_file.is_absolute():
            metadata_file = (artifact_dir / metadata_path).resolve()
        metadata = json.loads(metadata_file.read_text())
        abi = metadata["abi"]
        launch_info_path = Path(metadata["artifacts"]["launch_info_file_path"])
        if not launch_info_path.is_absolute():
            launch_info_path = (artifact_dir / launch_info_path).resolve()
        block_dim = int(
            launch_info_path.read_text().split("block_dim=", 1)[1].splitlines()[0]
        )
        return {
            "inputs": abi["inputs"],
            "outputs": abi["outputs"],
            "workspace_bytes": int(abi["workspace_bytes"]),
            "block_dim": block_dim,
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

npy_dir = data_dir / "npy"
out_dir = data_dir / "output"
out_dir.mkdir(parents=True, exist_ok=True)

mix_abi = load_mix_abi(artifact_dir, manifest)
output_name = mix_abi["outputs"][0]["name"]
golden_path = next(
    p for p in [
        npy_dir / f"{output_name}.npy",
        npy_dir / "output0.npy",
        npy_dir / "output.npy",
    ] if p.exists()
)
golden = np.load(golden_path)

def runtime_dtype(dtype):
    dtype = np.dtype(dtype)
    mapping = {
        np.dtype(np.float16): "f16",
        np.dtype(np.float32): "f32",
        np.dtype(np.float64): "f64",
        np.dtype(np.int8): "int8",
        np.dtype(np.int32): "int32",
        np.dtype(np.int64): "int64",
    }
    return mapping[dtype]

inputs = []
for idx in range(len(mix_abi["inputs"])):
    name = mix_abi["inputs"][idx]["name"]
    path = npy_dir / f"{name}.npy"
    if not path.exists():
        path = npy_dir / f"input{idx}.npy"
    inputs.append({"name": name, "path": str(path.resolve())})

run_manifest = {
    "task_id": "main",
    "backend": "sim",
    "artifact_root": str(artifact_dir.resolve()),
    "inputs": inputs,
    "outputs": [{
        "name": output_name,
        "path": str((out_dir / "output.npy").resolve()),
        "shape": list(golden.shape),
        "dtype": runtime_dtype(golden.dtype),
    }],
    "expected_outputs": [{
        "name": output_name,
        "path": str(golden_path.resolve()),
        "shape": list(golden.shape),
        "dtype": runtime_dtype(golden.dtype),
    }],
    "tiling": {
        "binary": str((artifact_dir / "out" / "tiling.bin").resolve()),
    },
    "block_dim": mix_abi["block_dim"],
    "workspace_size": mix_abi["workspace_bytes"],
    "profiling": True,
    "atol": 1.0,
    "rtol": 1e-2,
}

(data_dir / "runtime-manifest.json").write_text(
    json.dumps(run_manifest, indent=2) + "\n"
)
PY

ASCEND_DAV_SIM_VERSION="${DAV_SIM_VERSION}" \
LD_LIBRARY_PATH="$(runtime_verify_mix_ld_library_path "${ARTIFACT_DIR}")" \
  "${RUNTIME_SESSION}" --run-manifest "${DATA_DIR}/runtime-manifest.json" --run \
  >/tmp/runtime_mix_altshape_run.log 2>&1

python3 - "${DATA_DIR}/npy/output0.npy" "${DATA_DIR}/output/output.npy" <<'PY'
import sys
import numpy as np

golden = np.load(sys.argv[1])
actual = np.load(sys.argv[2])

if golden.shape != actual.shape:
    raise SystemExit(f"shape mismatch: {actual.shape} vs {golden.shape}")
if golden.dtype != actual.dtype:
    raise SystemExit(f"dtype mismatch: {actual.dtype} vs {golden.dtype}")
if not np.allclose(actual, golden, atol=1.0, rtol=1e-2):
    diff = np.abs(actual.astype(np.float64) - golden.astype(np.float64))
    raise SystemExit(
        f"allclose failed: max_abs_diff={diff.max():.6e} mean_abs_diff={diff.mean():.6e}"
    )
print("mix altshape pass")
PY
