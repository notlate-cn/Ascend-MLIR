#!/usr/bin/env python3
"""Build a runtime-session manifest for the matmul+relu fused mix kernel."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def runtime_dtype(dtype):
    dtype = np.dtype(dtype)
    return {
        np.dtype(np.float16): "f16",
        np.dtype(np.float32): "f32",
    }.get(dtype) or sys.exit(f"unsupported runtime dtype: {dtype}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--data-dir",     required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-npy",      required=True)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    data_dir = Path(args.data_dir).resolve()

    metadata = json.loads((artifact_dir / "out" / "mix_metadata.json").read_text())
    abi = metadata["abi"]

    launch_info = (artifact_dir / metadata["artifacts"]["launch_info_file_path"]
                   if not Path(metadata["artifacts"]["launch_info_file_path"]).is_absolute()
                   else Path(metadata["artifacts"]["launch_info_file_path"])).resolve()
    block_dim = int(launch_info.read_text().split("block_dim=", 1)[1].splitlines()[0])

    inputs = []
    for idx, desc in enumerate(abi["inputs"]):
        npy = data_dir / f"input{idx}.npy"
        if not npy.exists():
            sys.exit(f"missing {npy}")
        inputs.append({"name": desc["name"], "path": str(npy)})

    if len(abi["outputs"]) != 1:
        sys.exit("expected exactly one output")
    out_desc = abi["outputs"][0]
    golden = data_dir / "output0.npy"
    golden_arr = np.load(golden)

    manifest = {
        "task_id": "main",
        "backend": "sim",
        "artifact_root": str(artifact_dir),
        "inputs": inputs,
        "outputs": [{
            "name": out_desc["name"],
            "path": args.out_npy,
            "shape": list(golden_arr.shape),
            "dtype": runtime_dtype(golden_arr.dtype),
        }],
        "expected_outputs": [{
            "name": out_desc["name"],
            "path": str(golden),
            "shape": list(golden_arr.shape),
            "dtype": runtime_dtype(golden_arr.dtype),
        }],
        "tiling": {
            "binary": str(artifact_dir / "out" / "tiling.bin"),
        },
        "block_dim": block_dim,
        "workspace_size": int(abi["workspace_bytes"]),
        "profiling": True,
        "atol": 1.0,
        "rtol": 1e-2,
    }
    Path(args.out_manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.out_manifest)


if __name__ == "__main__":
    main()
