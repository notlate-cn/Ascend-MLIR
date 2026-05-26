#!/usr/bin/env python3
"""Generate deterministic transformer runtime-session bindings."""

from __future__ import annotations

import argparse
import json
import re
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


def parse_memref_type(text: str) -> dict:
    match = re.search(r"memref<([^x>]+(?:x[^x>]+)*)x([a-z0-9]+)", text)
    if not match:
        raise ValueError(f"unsupported memref type: {text}")
    dims = []
    for dim in match.group(1).split("x"):
        dims.append(None if dim == "?" else int(dim))
    return {"shape": dims, "dtype": match.group(2)}


def split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
    if current:
        parts.append("".join(current).strip())
    return parts


def parse_cann_kernel_signatures(cann_mlir: Path, compiler_manifest: dict) -> dict[str, dict]:
    text = cann_mlir.read_text(encoding="utf-8")
    abi_by_kernel = {
        entry["kernel_id"]: entry["abi"]
        for entry in compiler_manifest.get("kernel_entries", [])
    }
    signatures: dict[str, dict] = {}
    pattern = re.compile(r"func\.func @(kernel_\d+)\((.*?)\) attributes", re.S)
    for kernel_id, args_text in pattern.findall(text):
        abi = abi_by_kernel[kernel_id]
        arg_parts = split_top_level_commas(args_text)
        args = []
        for part in arg_parts[: abi["workspaceArgIndex"]]:
            if ":" not in part:
                continue
            name, type_text = part.split(":", 1)
            memref = parse_memref_type(type_text.strip())
            memref["name"] = name.strip().lstrip("%")
            args.append(memref)
        workspace_index = abi["workspaceArgIndex"]
        signatures[kernel_id] = {
            "args": args,
            "inputs": args[: abi["numInputs"]],
            "outputs": args[abi["numInputs"] : workspace_index],
        }
    return signatures


def concrete_shape(memref: dict, batch: int, seq: int) -> list[int]:
    pattern = memref["shape"]
    rank = len(pattern)

    def fill(defaults: list[int]) -> list[int]:
        return [static if static is not None else defaults[i] for i, static in enumerate(pattern)]

    if rank == 1:
        return fill([seq])
    if rank == 2:
        if pattern[0] == 128 and pattern[1] == 384:
            return [128, 384]
        if pattern[0] == 128 and pattern[1] == 128:
            return [128, 128]
        if pattern[0] == 128 and pattern[1] == 512:
            return [128, 512]
        if pattern[0] == 512 and pattern[1] == 128:
            return [512, 128]
        if pattern[1] == 128:
            return fill([seq, 128])
        return fill([seq, batch])
    if rank == 3:
        if pattern[1] == 4:
            return fill([batch, 4, seq])
        if pattern[1] == 128 and pattern[2] in (384, 512):
            return fill([seq, pattern[1], pattern[2]])
        if pattern[1] == 512 and pattern[2] == 128:
            return fill([batch, 512, 128])
        if pattern[2] == 32:
            return fill([batch * 4, seq, 32])
        if pattern[2] == 1:
            return fill([batch, seq, 1])
        if pattern[2] == 128:
            return fill([batch, seq, 128])
        if pattern[2] == 384:
            return fill([seq, batch, 384])
        if pattern[2] == 512:
            return fill([batch, seq, 512])
        return fill([batch * 4, seq, seq])
    if rank == 4:
        if pattern[0] == 1 and pattern[3] == 128:
            return fill([1, seq, batch, 128])
        if pattern[1] == 4 and pattern[2] == 32:
            return fill([batch, 4, 32, seq])
        if pattern[1] == 4:
            return fill([batch, 4, seq, seq])
        if pattern[1] == 1 and pattern[2] == 4 and pattern[3] == 32:
            return fill([seq, 1, 4, 32])
        return fill([batch, seq, 1, 128])
    if rank == 5 and pattern[0] == 3:
        return fill([3, seq, batch, 1, 128])
    return fill([seq] * rank)


def dtype_from_memref(memref: dict) -> str:
    dtype = memref["dtype"]
    if dtype == "i64":
        return "int64"
    if dtype == "f32":
        return "f32"
    if dtype == "f16":
        return "f16"
    raise ValueError(f"unsupported memref dtype: {dtype}")


def shape_signature(memref: dict) -> tuple:
    return tuple(memref["shape"]), memref["dtype"]


def build_tiling_schema(entry: dict, schema_dir: Path, compiler_manifest: dict) -> Path:
    schema_dir.mkdir(parents=True, exist_ok=True)
    kernel_id = entry["kernel_id"]
    schema = {
        "schema_version": compiler_manifest.get("schema_version", "2.0"),
        "kernel": kernel_id,
        "kernel_file": compiler_manifest.get("kernel_file", ""),
        "soc": compiler_manifest.get("soc", ""),
        "tiling_params": entry.get("tilingSchema", []),
    }
    path = schema_dir / f"{kernel_id}_tiling_space.json"
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


def tiling_params_for_entry(entry: dict, arg_shapes: list[list[int]]) -> str:
    params: list[str] = []
    for field in entry.get("tilingSchema", []):
        name = field["name"]
        if field.get("fixed"):
            shape_key = field.get("shape_key", "")
            match = re.fullmatch(r"arg(\d+)_dim(\d+)", shape_key)
            if not match:
                raise ValueError(f"unsupported shape key: {shape_key}")
            arg_index = int(match.group(1))
            dim_index = int(match.group(2))
            value = arg_shapes[arg_index][dim_index]
        elif "fixed_value" in field:
            value = int(field["fixed_value"])
        elif field.get("values"):
            value = int(field["values"][0])
        else:
            value = 1
        params.append(f"{name}={value}")
    return ",".join(params)


def materialize_mix_compile_npy_dir(
    root: Path,
    kernel_id: str,
    sig: dict,
    arg_shapes: list[list[int]],
    batch: int,
    seq: int,
) -> None:
    kernel_dir = root / kernel_id
    kernel_dir.mkdir(parents=True, exist_ok=True)

    for index, memref in enumerate(sig["inputs"]):
        shape = tuple(arg_shapes[index])
        dtype = dtype_from_memref(memref)
        names = {memref.get("name", f"arg{index}"), f"input{index}"}
        for name in names:
            write_zero(kernel_dir / f"{name}.npy", shape, dtype)

    for index, memref in enumerate(sig["outputs"]):
        shape = tuple(concrete_shape(memref, batch, seq))
        dtype = dtype_from_memref(memref)
        names = {memref.get("name", f"output{index}"), f"output{index}"}
        for name in names:
            write_zero(kernel_dir / f"{name}.npy", shape, dtype)


def consumer_operand_index(buffer_name: str) -> int:
    match = re.search(r"_operand(\d+)$", buffer_name)
    if not match:
        raise ValueError(f"cannot parse consumer operand index from {buffer_name}")
    return int(match.group(1))


def build_multi_kernel_manifest(args: argparse.Namespace, output_dir: Path) -> dict:
    compiler_manifest = json.loads(args.compiler_artifact_manifest.read_text(encoding="utf-8"))
    signatures = parse_cann_kernel_signatures(args.cann_mlir, compiler_manifest)
    entries = compiler_manifest.get("kernel_entries", [])
    entries_by_id = {entry["kernel_id"]: entry for entry in entries}
    edges = compiler_manifest.get("kernelGraph", {}).get("edges", [])

    schema_dir = args.tiling_schema_dir or args.run_manifest.parent / "tiling_schemas"
    incoming: dict[str, dict[int, tuple[str, str]]] = {}
    dependencies: dict[str, set[str]] = {entry["kernel_id"]: set() for entry in entries}
    output_candidates: dict[str, list[tuple[str, dict]]] = {}

    for entry in entries:
        kernel_id = entry["kernel_id"]
        sig = signatures[kernel_id]
        candidates = [(f"out{idx}", memref) for idx, memref in enumerate(sig["outputs"])]
        for arg_index in entry["abi"].get("writesToInputArgs", []):
            candidates.append((f"inout{arg_index}", sig["inputs"][arg_index]))
        output_candidates[kernel_id] = candidates

    for edge in edges:
        producer = edge["from"]
        consumer = edge["to"]
        dependencies.setdefault(consumer, set()).add(producer)
        for carried in edge.get("carriedBuffers", []):
            assigned = incoming.setdefault(consumer, {})
            hint = consumer_operand_index(carried)
            consumer_inputs = signatures[consumer]["inputs"]
            candidate_by_input = []
            for input_index, consumer_type in enumerate(consumer_inputs):
                if input_index in assigned:
                    continue
                names = [
                    name for name, memref in output_candidates[producer]
                    if shape_signature(memref) == shape_signature(consumer_type)
                ]
                if names:
                    candidate_by_input.append((input_index, names[0]))
            exact = [item for item in candidate_by_input if item[0] == hint]
            if exact:
                operand, output_name = exact[0]
            elif candidate_by_input:
                operand, output_name = candidate_by_input[0]
            else:
                raise ValueError(
                    f"no upstream output from {producer} matches {consumer} carried buffer {carried}"
                )
            assigned[operand] = (producer, output_name)

    tasks = []
    for entry in entries:
        kernel_id = entry["kernel_id"]
        sig = signatures[kernel_id]
        arg_shapes = [concrete_shape(arg, args.batch, args.seq) for arg in sig["args"]]
        if args.mix_compile_npy_root:
            materialize_mix_compile_npy_dir(
                args.mix_compile_npy_root,
                kernel_id,
                sig,
                arg_shapes,
                args.batch,
                args.seq,
            )
        inputs = []
        for index, memref in enumerate(sig["inputs"]):
            if index in incoming.get(kernel_id, {}):
                producer, output_name = incoming[kernel_id][index]
                inputs.append({
                    "name": f"arg{index}",
                    "source": "task_output",
                    "upstream_task": producer,
                    "upstream_output": output_name,
                })
                continue
            path = args.out_dir / f"{kernel_id}_arg{index}.npy"
            write_zero(path, tuple(arg_shapes[index]), dtype_from_memref(memref))
            inputs.append({"name": f"arg{index}", "path": str(path)})

        outputs = []
        for index, memref in enumerate(sig["outputs"]):
            outputs.append({
                "name": f"out{index}",
                "shape": concrete_shape(memref, args.batch, args.seq),
                "dtype": dtype_from_memref(memref),
            })
        for arg_index in entry["abi"].get("writesToInputArgs", []):
            memref = sig["inputs"][arg_index]
            outputs.append({
                "name": f"inout{arg_index}",
                "source": "input_alias",
                "input": f"arg{arg_index}",
                "shape": concrete_shape(memref, args.batch, args.seq),
                "dtype": dtype_from_memref(memref),
            })

        if kernel_id == entries[-1]["kernel_id"] and outputs:
            outputs[0]["path"] = str(output_dir / "arg55.npy")

        schema_path = build_tiling_schema(entry, schema_dir, compiler_manifest)
        task = {
            "task_id": kernel_id,
            "artifact_root": str(args.artifact_root / kernel_id),
            "inputs": inputs,
            "outputs": outputs,
            "tiling": {
                "schema": str(schema_path),
                "params": tiling_params_for_entry(entry, arg_shapes),
            },
            "block_dim": args.block_dim,
            "workspace_size": max(8192, int(entry.get("workspaceSizeBytes", 0) or args.workspace_size)),
            "profiling": True,
            "atol": 1.0e-3,
            "rtol": 1.0e-3,
        }
        deps = sorted(dependencies.get(kernel_id, set()), key=lambda name: int(name.split("_")[1]))
        if deps:
            task["dependencies"] = deps
        if kernel_id == entries[-1]["kernel_id"]:
            task["expected_outputs"] = [
                {"name": "out0", "path": str(args.out_dir / "expected_arg55.npy")}
            ]
        tasks.append(task)

    return {"backend": "sim", "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--seq", type=positive_int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--tiling-schema", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--actual-output-dir", type=Path)
    parser.add_argument("--compiler-artifact-manifest", type=Path)
    parser.add_argument("--compiler-runtime-manifest", type=Path)
    parser.add_argument("--cann-mlir", type=Path)
    parser.add_argument("--tiling-schema-dir", type=Path)
    parser.add_argument("--mix-compile-npy-root", type=Path)
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
            if not args.compiler_artifact_manifest and not args.compiler_runtime_manifest:
                raise SystemExit("--tiling-schema is required with --run-manifest")
        output_dir = args.actual_output_dir or args.run_manifest.parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        if (
            args.compiler_artifact_manifest
            and args.compiler_runtime_manifest
            and args.compiler_artifact_manifest.resolve() != args.compiler_runtime_manifest.resolve()
        ):
            raise SystemExit(
                "cannot pass both --compiler-artifact-manifest and "
                "--compiler-runtime-manifest with different paths"
            )
        if not args.compiler_artifact_manifest:
            args.compiler_artifact_manifest = args.compiler_runtime_manifest
        if args.compiler_artifact_manifest:
            if not args.cann_mlir:
                raise SystemExit("--cann-mlir is required with --compiler-artifact-manifest")
            manifest = build_multi_kernel_manifest(args, output_dir)
        else:
            manifest = build_manifest(args, output_dir)
        args.run_manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"transformer_data.batch={args.batch}")
    print(f"transformer_data.seq={args.seq}")
    print(f"transformer_data.inputs=13")
    print(f"transformer_data.outputs={len(output_specs(args.batch, args.seq))}")


if __name__ == "__main__":
    main()
