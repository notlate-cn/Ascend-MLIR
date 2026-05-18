#!/usr/bin/env python3
"""Generate deterministic transformer runtime-session bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


INPUT_SHAPES = [
    (128,),
    (128,),
    (128,),
    (128, 512),
    (512,),
    (512, 128),
    (128,),
    (128,),
    (128,),
    (128, 128),
    (384,),
    (384, 128),
]


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


def output_specs(batch: int, seq: int) -> list[tuple[str, str, tuple[int, ...]]]:
    b4 = batch * 4
    return [
        ("arg13", "f32", (seq, batch, 128)),
        ("arg14", "f32", (128, 384)),
        ("arg15", "f32", (seq, 128, 384)),
        ("arg16", "f32", (seq, batch, 384)),
        ("arg17", "int64", (4,)),
        ("arg18", "f32", (3, seq, batch, 1, 128)),
        ("arg19", "f32", (1, seq, batch, 128)),
        ("arg20", "f32", (1, seq, batch, 128)),
        ("arg21", "f32", (1, seq, batch, 128)),
        ("arg22", "int64", (3,)),
        ("arg23", "f32", (b4, seq, 32)),
        ("arg24", "f32", (b4, seq, 32)),
        ("arg25", "f32", (b4, seq, 32)),
        ("arg26", "int64", (4,)),
        ("arg27", "f32", (batch, 4, 32, seq)),
        ("arg28", "int64", (4,)),
        ("arg29", "f32", (b4, seq, seq)),
        ("arg30", "int64", (4,)),
        ("arg31", "f32", (batch, 4, seq, seq)),
        ("arg32", "int64", (batch, 4, seq)),
        ("arg33", "f32", (batch, 4, seq)),
        ("arg34", "f32", (batch, 4, seq, 1)),
        ("arg35", "f32", (b4, seq, 32)),
        ("arg36", "f32", (seq, 1, 4, 32)),
        ("arg37", "f32", (128, 128)),
        ("arg38", "f32", (seq, 128)),
        ("arg39", "int64", (3,)),
        ("arg40", "f32", (batch, seq, 128)),
        ("arg41", "f32", (batch, seq, 128)),
        ("arg42", "f32", (batch, seq, 1)),
        ("arg43", "f32", (batch, seq, 1)),
        ("arg44", "f32", (batch, seq, 1)),
        ("arg45", "f32", (batch, seq, 128)),
        ("arg46", "f32", (batch, seq, 128)),
        ("arg47", "f32", (batch, seq, 1)),
        ("arg48", "f32", (batch, seq, 128)),
        ("arg49", "f32", (128, 512)),
        ("arg50", "f32", (batch, 128, 512)),
        ("arg51", "f32", (batch, seq, 512)),
        ("arg52", "f32", (512, 128)),
        ("arg53", "f32", (batch, 512, 128)),
        ("arg54", "f32", (batch, seq, 1)),
        ("arg55", "f32", (batch, seq, 128)),
    ]


def dtype_name_to_numpy(dtype: str) -> np.dtype:
    if dtype == "f32":
        return np.dtype(np.float32)
    if dtype == "int64":
        return np.dtype(np.int64)
    raise ValueError(f"unsupported dtype: {dtype}")


def write_zero(path: Path, shape: tuple[int, ...], dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros(shape, dtype=dtype_name_to_numpy(dtype)))


def build_manifest(args: argparse.Namespace, output_dir: Path) -> dict:
    inputs = [
        {"name": "arg0", "path": str(args.out_dir / "input0.npy")},
    ]
    for index in range(len(INPUT_SHAPES)):
        inputs.append(
            {"name": f"arg{index + 1}", "path": str(args.out_dir / f"input{index + 1}.npy")}
        )

    outputs = []
    for name, dtype, shape in output_specs(args.batch, args.seq):
        outputs.append(
            {
                "name": name,
                "path": str(output_dir / f"{name}.npy"),
                "shape": list(shape),
                "dtype": dtype,
            }
        )

    return {
        "task_id": "main",
        "backend": "sim",
        "artifact_root": str(args.artifact_root),
        "inputs": inputs,
        "outputs": outputs,
        "expected_outputs": [
            {"name": "arg55", "path": str(args.out_dir / "expected_arg55.npy")}
        ],
        "tiling": {
            "schema": str(args.tiling_schema),
            "params": f"dim_arg0_0={args.batch},dim_arg0_1={args.seq}",
        },
        "block_dim": args.block_dim,
        "workspace_size": args.workspace_size,
        "profiling": True,
        "atol": 1.0e-3,
        "rtol": 1.0e-3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--seq", type=positive_int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--tiling-schema", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--actual-output-dir", type=Path)
    parser.add_argument("--block-dim", type=positive_int, default=1)
    parser.add_argument("--workspace-size", type=positive_int, default=16 * 1024 * 1024)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_zero(args.out_dir / "input0.npy", (args.batch, args.seq, 128), "f32")
    for index, shape in enumerate(INPUT_SHAPES, start=1):
        write_zero(args.out_dir / f"input{index}.npy", shape, "f32")
    write_zero(args.out_dir / "expected_arg55.npy", (args.batch, args.seq, 128), "f32")

    if args.run_manifest:
        if not args.artifact_root:
            raise SystemExit("--artifact-root is required with --run-manifest")
        if not args.tiling_schema:
            raise SystemExit("--tiling-schema is required with --run-manifest")
        output_dir = args.actual_output_dir or args.run_manifest.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(args, output_dir)
        args.run_manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"transformer_data.batch={args.batch}")
    print(f"transformer_data.seq={args.seq}")
    print(f"transformer_data.inputs=13")
    print(f"transformer_data.outputs={len(output_specs(args.batch, args.seq))}")


if __name__ == "__main__":
    main()
