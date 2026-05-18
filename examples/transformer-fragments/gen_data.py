#!/usr/bin/env python3
"""Generate deterministic runtime-session bindings for transformer fragments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LAYERNORM_HIDDEN = 128
QKV_HIDDEN = 16
QKV_WIDTH = 48


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


def load_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)


def build_shape_values(args: argparse.Namespace) -> dict[str, int]:
    values = {
        "arg0_dim0": args.m if args.fragment == "layernorm" else args.batch * args.seq,
        "arg0_dim1": LAYERNORM_HIDDEN if args.fragment == "layernorm" else QKV_HIDDEN,
        "arg1_dim0": LAYERNORM_HIDDEN if args.fragment == "layernorm" else QKV_HIDDEN,
        "arg1_dim1": QKV_WIDTH if args.fragment == "qkv" else LAYERNORM_HIDDEN,
        "arg2_dim0": LAYERNORM_HIDDEN if args.fragment == "layernorm" else QKV_WIDTH,
        "arg3_dim0": args.batch * args.seq,
        "arg3_dim1": QKV_WIDTH,
    }
    if args.fragment == "layernorm":
        values.update({
            "result0_dim0": args.m,
            "result0_dim1": LAYERNORM_HIDDEN,
        })
    else:
        values.update({
            "result0_dim0": args.batch * args.seq,
            "result0_dim1": QKV_WIDTH,
        })

    aliases = {}
    for key, value in values.items():
        parts = key.split("_dim")
        if len(parts) == 2 and parts[0].startswith("arg"):
            aliases[f"dim_{parts[0]}_{parts[1]}"] = value
        if len(parts) == 2 and parts[0].startswith("result"):
            aliases[f"dim_{parts[0]}_{parts[1]}"] = value
    values.update(aliases)
    return values


def build_tiling_params(schema_path: Path, args: argparse.Namespace) -> str:
    schema = load_schema(schema_path)
    shape_values = build_shape_values(args)
    params: list[str] = []
    for field in schema.get("tiling_params", []):
        name = field["name"]
        if field.get("fixed"):
            shape_key = field.get("shape_key")
            if shape_key not in shape_values:
                raise SystemExit(f"unsupported fixed shape key: {shape_key}")
            value = shape_values[shape_key]
        elif name in shape_values:
            value = shape_values[name]
        elif "fixed_value" in field:
            value = int(field["fixed_value"])
        elif field.get("values"):
            value = int(field["values"][0])
        else:
            raise SystemExit(f"unsupported free tiling field: {name}")
        params.append(f"{name}={value}")
    return ",".join(params)


def generate_layernorm(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rng = np.random.default_rng(args.seed)
    x = rng.normal(0.0, 0.2, size=(args.m, LAYERNORM_HIDDEN)).astype(np.float32)
    gamma = rng.normal(1.0, 0.05, size=(LAYERNORM_HIDDEN,)).astype(np.float32)
    beta = rng.normal(0.0, 0.02, size=(LAYERNORM_HIDDEN,)).astype(np.float32)

    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (x - mean) * np.reciprocal(np.sqrt(var + np.float32(1.0e-5)))
    expected = expected * gamma.reshape(1, LAYERNORM_HIDDEN) + beta.reshape(1, LAYERNORM_HIDDEN)

    save(args.out_dir / "input_x.npy", x)
    save(args.out_dir / "input_gamma.npy", gamma)
    save(args.out_dir / "input_beta.npy", beta)
    save(args.out_dir / "expected_out.npy", expected)

    inputs = [
        {"name": "x", "path": str(args.out_dir / "input_x.npy")},
        {"name": "gamma", "path": str(args.out_dir / "input_gamma.npy")},
        {"name": "beta", "path": str(args.out_dir / "input_beta.npy")},
    ]
    outputs = [
        {
            "name": "mean",
            "path": str(args.actual_output.parent / "mean_actual.npy"),
            "shape": [args.m],
            "dtype": "f32",
        },
        {
            "name": "var",
            "path": str(args.actual_output.parent / "var_actual.npy"),
            "shape": [args.m],
            "dtype": "f32",
        },
        {
            "name": "out",
            "path": str(args.actual_output),
            "shape": [args.m, LAYERNORM_HIDDEN],
            "dtype": "f32",
        },
    ]
    return inputs, outputs


def generate_qkv(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rng = np.random.default_rng(args.seed)
    tokens = args.batch * args.seq
    x = rng.integers(-4, 4, size=(tokens, QKV_HIDDEN)).astype(np.float16)
    weight = rng.integers(-4, 4, size=(QKV_HIDDEN, QKV_WIDTH)).astype(np.float16)
    bias = rng.normal(0.0, 0.01, size=(QKV_WIDTH,)).astype(np.float32)

    expected = x.astype(np.float32) @ weight.astype(np.float32) + bias.reshape(1, QKV_WIDTH)

    save(args.out_dir / "input_x.npy", x)
    save(args.out_dir / "input_weight.npy", weight)
    save(args.out_dir / "input_bias.npy", bias)
    save(args.out_dir / "expected_out.npy", expected)
    save(args.out_dir / "input0.npy", x)
    save(args.out_dir / "input1.npy", weight)
    save(args.out_dir / "input2.npy", bias)
    save(args.out_dir / "output0.npy", expected)

    inputs = [
        {"name": "x", "path": str(args.out_dir / "input_x.npy")},
        {"name": "weight", "path": str(args.out_dir / "input_weight.npy")},
        {"name": "bias", "path": str(args.out_dir / "input_bias.npy")},
    ]
    outputs = [
        {
            "name": "out",
            "path": str(args.actual_output),
            "shape": [tokens, QKV_WIDTH],
            "dtype": "f32",
        },
    ]
    return inputs, outputs


def write_manifest(args: argparse.Namespace, inputs: list[dict[str, str]], outputs: list[dict[str, Any]]) -> None:
    params = build_tiling_params(args.tiling_schema, args)
    manifest = {
        "task_id": "main",
        "backend": "sim",
        "artifact_root": str(args.artifact_root),
        "inputs": inputs,
        "outputs": outputs,
        "expected_outputs": [
            {"name": "out", "path": str(args.out_dir / "expected_out.npy")}
        ],
        "tiling": {
            "schema": str(args.tiling_schema),
            "params": params,
        },
        "block_dim": args.block_dim,
        "workspace_size": args.workspace_size,
        "profiling": True,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    args.run_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragment", choices=["layernorm", "qkv"], required=True)
    parser.add_argument("--m", type=positive_int, default=4)
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--seq", type=positive_int, default=1)
    parser.add_argument("--seed", type=positive_int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--tiling-schema", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--actual-output", type=Path, required=True)
    parser.add_argument("--block-dim", type=positive_int, default=1)
    parser.add_argument("--workspace-size", type=positive_int, default=16 * 1024 * 1024)
    parser.add_argument("--atol", type=float, default=1.0e-2)
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.actual_output.parent.mkdir(parents=True, exist_ok=True)

    if args.fragment == "layernorm":
        inputs, outputs = generate_layernorm(args)
    else:
        inputs, outputs = generate_qkv(args)
    write_manifest(args, inputs, outputs)

    print(f"transformer_fragment.{args.fragment}.data=pass")
    print(f"transformer_fragment.{args.fragment}.inputs={len(inputs)}")
    print("transformer_fragment.%s.outputs=%d" % (args.fragment, len(outputs)))


if __name__ == "__main__":
    main()
