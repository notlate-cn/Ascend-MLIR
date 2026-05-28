from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import struct
from typing import Any

from ascend_debug.runner import CommandError, find_tool, run_command


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CommandError(f"invalid JSON in case.json: {path}: {error}") from error
    if not isinstance(data, dict):
        raise CommandError(f"case.json root must be an object: {path}")
    return data


def _require_object(root: dict[str, Any], field: str) -> dict[str, Any]:
    value = root.get(field)
    if not isinstance(value, dict):
        raise CommandError(f"missing required object field: {field}")
    return value


def _require_string(root: dict[str, Any], field: str) -> str:
    value = root.get(field)
    if not isinstance(value, str) or not value:
        raise CommandError(f"missing required string field: {field}")
    return value


def _resolve_case_path(case_dir: pathlib.Path, path: str, *, label: str) -> pathlib.Path:
    resolved = pathlib.Path(path)
    if not resolved.is_absolute():
        resolved = case_dir / resolved
    return resolved.resolve()


def _normalize_dtype(value: str) -> str:
    aliases = {
        "f16": "f16",
        "float16": "f16",
        "fp16": "f16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "f32": "f32",
        "float32": "f32",
        "fp32": "f32",
        "int8": "int8",
        "i8": "int8",
        "int32": "int32",
        "i32": "int32",
        "int64": "int64",
        "i64": "int64",
    }
    normalized = aliases.get(value.lower())
    if not normalized:
        raise CommandError(f"unsupported tensor dtype: {value}")
    return normalized


def _dtype_from_npy_descr(descr: str) -> str:
    mapping = {
        "<f2": "f16",
        ">f2": "f16",
        "|f2": "f16",
        "<f4": "f32",
        ">f4": "f32",
        "|i1": "int8",
        "<i1": "int8",
        ">i1": "int8",
        "<i4": "int32",
        ">i4": "int32",
        "<i8": "int64",
        ">i8": "int64",
    }
    dtype = mapping.get(descr)
    if not dtype:
        raise CommandError(f"unsupported .npy dtype descriptor: {descr}")
    return dtype


def _read_npy_metadata(path: pathlib.Path) -> tuple[list[int], str]:
    try:
        with path.open("rb") as handle:
            if handle.read(6) != b"\x93NUMPY":
                raise CommandError(f"not a .npy file: {path}")
            major, minor = handle.read(2)
            if (major, minor) == (1, 0):
                header_len = struct.unpack("<H", handle.read(2))[0]
            elif major in (2, 3):
                header_len = struct.unpack("<I", handle.read(4))[0]
            else:
                raise CommandError(f"unsupported .npy version: {major}.{minor}: {path}")
            header = ast.literal_eval(handle.read(header_len).decode("latin1"))
    except OSError as error:
        raise CommandError(f"cannot read .npy file: {path}: {error}") from error
    if not isinstance(header, dict):
        raise CommandError(f"invalid .npy header: {path}")
    shape = header.get("shape")
    descr = header.get("descr")
    if not isinstance(shape, tuple) or not isinstance(descr, str):
        raise CommandError(f"invalid .npy shape/dtype header: {path}")
    return [int(dim) for dim in shape], _dtype_from_npy_descr(descr)


def _normalize_shape(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(dim, int) for dim in value):
        raise CommandError(f"{field} must be an integer array")
    return [int(dim) for dim in value]


def _binding_arg_name(name: str) -> str:
    return name.split(".", 1)[1] if "." in name else name


def _normalize_binding(
    raw: Any,
    *,
    case_dir: pathlib.Path,
    field: str,
    infer_existing_data: bool,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CommandError(f"{field} entries must be objects")
    name = _require_string(raw, "name")
    path_text = _require_string(raw, "path")
    path = _resolve_case_path(case_dir, path_text, label=f"{field}.{name}.path")

    binding: dict[str, Any] = {
        "name": name,
        "path": str(path),
    }
    if "shape" in raw:
        binding["shape"] = _normalize_shape(raw["shape"], field=f"{field}.{name}.shape")
    dtype_value = raw.get("dtype", raw.get("datatype"))
    if dtype_value is not None:
        if not isinstance(dtype_value, str):
            raise CommandError(f"{field}.{name}.dtype must be a string")
        binding["dtype"] = _normalize_dtype(dtype_value)

    if infer_existing_data and ("shape" not in binding or "dtype" not in binding):
        if path.suffix.lower() != ".npy":
            missing = []
            if "shape" not in binding:
                missing.append("shape")
            if "dtype" not in binding:
                missing.append("dtype")
            raise CommandError(
                f"{field} binding {name} requires .npy data or explicit "
                f"{'/'.join(missing)}"
            )
        shape, dtype = _read_npy_metadata(path)
        binding.setdefault("shape", shape)
        binding.setdefault("dtype", dtype)

    return binding


def _normalize_bindings(
    root: dict[str, Any],
    *,
    case_dir: pathlib.Path,
    field: str,
    required: bool,
    infer_existing_data: bool,
) -> list[dict[str, Any]]:
    value = root.get(field)
    if value is None:
        if required:
            raise CommandError(f"missing required array field: {field}")
        return []
    if not isinstance(value, list):
        raise CommandError(f"{field} must be an array")
    return [
        _normalize_binding(
            item,
            case_dir=case_dir,
            field=field,
            infer_existing_data=infer_existing_data,
        )
        for item in value
    ]


def _infer_shape_args(inputs: list[dict[str, Any]]) -> dict[str, list[int]]:
    shape_args: dict[str, list[int]] = {}
    for binding in inputs:
        shape = binding.get("shape")
        if isinstance(shape, list):
            shape_args[_binding_arg_name(str(binding["name"]))] = [int(dim) for dim in shape]
    return shape_args


def _infer_kernel_name(source_mlir: pathlib.Path, source: dict[str, Any]) -> str:
    explicit = source.get("kernel_name", source.get("name"))
    if isinstance(explicit, str) and explicit:
        return explicit
    text = source_mlir.read_text(encoding="utf-8")
    match = re.search(r"\bfunc\.func\s+@([A-Za-z_][A-Za-z0-9_]*)", text)
    if match:
        return match.group(1)
    raise CommandError("source.kernel_name is required when source.mlir has no func.func symbol")


def _resolve_cann_root() -> pathlib.Path:
    for name in ("CANN_ROOT", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(name)
        if value:
            return pathlib.Path(value).resolve()
    raise CommandError("source-mode case requires CANN_ROOT, ASCEND_HOME_PATH, or ASCEND_TOOLKIT_HOME")


def _tool_from_env(env_name: str, default: str) -> str:
    return find_tool(os.environ.get(env_name, default))


def _translate_tool() -> str:
    return find_tool(
        os.environ.get("ASCEND_MLIR_TRANSLATE")
        or "ascend-mlir-translate"
    )


def _run_artifact_case(case_path: pathlib.Path, run_dir: pathlib.Path) -> int:
    reports_dir = run_dir / "reports"
    run_manifest = run_dir / "run_manifest.json"
    runtime_session = find_tool("runtime-session")
    run_command(
        [
            runtime_session,
            "--case",
            str(case_path),
            "--emit-run-manifest",
            str(run_manifest),
        ],
        stdout_path=run_dir / "runtime-session.prepare.log",
        stderr_report_path=reports_dir / "runtime-session.prepare.stderr.txt",
    )
    run_command(
        [
            runtime_session,
            "--run-manifest",
            str(run_manifest),
            "--run",
        ],
        stdout_path=run_dir / "runtime-session.run.log",
        stderr_report_path=reports_dir / "runtime-session.run.stderr.txt",
    )
    return 0


def _compile_source_case(
    *,
    root: dict[str, Any],
    case_path: pathlib.Path,
    run_dir: pathlib.Path,
) -> pathlib.Path:
    if "artifact" in root:
        raise CommandError("case.json must not contain both source and artifact")

    case_dir = case_path.parent
    source = _require_object(root, "source")
    source_mlir = _resolve_case_path(case_dir, _require_string(source, "mlir"), label="source.mlir")
    if not source_mlir.exists():
        raise CommandError(f"source MLIR does not exist: {source_mlir}")

    kernel_name = _infer_kernel_name(source_mlir, source)
    kernel_kind = str(source.get("kernel_kind", source.get("kernelKind", "vec")))
    soc = str(source.get("soc", os.environ.get("SOC_VERSION", "Ascend910B1")))
    if "cann_root" in source:
        cann_root = _resolve_case_path(case_dir, str(source["cann_root"]), label="source.cann_root")
    else:
        cann_root = _resolve_cann_root()

    inputs = _normalize_bindings(
        root,
        case_dir=case_dir,
        field="inputs",
        required=True,
        infer_existing_data=True,
    )
    outputs = _normalize_bindings(
        root,
        case_dir=case_dir,
        field="outputs",
        required=False,
        infer_existing_data=False,
    )
    expected_outputs = _normalize_bindings(
        root,
        case_dir=case_dir,
        field="expected_outputs",
        required=False,
        infer_existing_data=True,
    )

    opt = _tool_from_env("ASCEND_MLIR_OPT", "ascend-mlir-opt")
    translate = _translate_tool()
    runtime_session = find_tool("runtime-session")
    cxx = _tool_from_env("CXX", "c++")

    reports_dir = run_dir / "reports"
    step1 = run_dir / "step1_fused.mlir"
    step2 = run_dir / "step2_normalized.mlir"
    step3 = run_dir / "step3_kernelized.mlir"
    step4 = run_dir / "step4_scheduled.mlir"
    step5 = run_dir / "step5_realized.mlir"
    step6 = run_dir / "step6_ascendc.mlir"
    step7 = run_dir / "step7_parallelized.mlir"
    step8 = run_dir / "step8_kernel_ir.mlir"
    step9 = run_dir / "step9_cann.mlir"
    kernel_cpp = run_dir / "step10_kernel.cpp"
    tiling_space = run_dir / "phase5_tiling_space.json"
    artifact_manifest = run_dir / "phase5_artifact_manifest.json"
    host_tiling = run_dir / "host_tiling.cpp"
    artifact_root = run_dir / "artifact"

    run_command(
        [
            opt,
            str(source_mlir),
            "--linalg-generalize-named-ops",
            "--linalg-fuse-elementwise-ops",
            "--canonicalize",
            "--cse",
        ],
        stdout_path=step1,
        stderr_report_path=reports_dir / "010-fuse.stderr.txt",
    )
    stages = [
        (step1, ["--ascend-normalize"], step2, "020-normalize"),
        (step2, ["--ascend-kernelize"], step3, "030-kernelize"),
        (
            step3,
            [
                f"--ascend-schedule=target-tile-policy=target-aware cann-root={cann_root} soc={soc}"
            ],
            step4,
            "040-schedule",
        ),
        (
            step4,
            ["--ascend-realize=materialization-mode=memory-space-annotate"],
            step5,
            "050-realize",
        ),
        (step5, ["--ascend-compute-lower"], step6, "060-compute-lower"),
        (step6, ["--ascend-parallelize"], step7, "070-parallelize"),
        (step7, ["--ascend-prepare-for-emit"], step8, "080-prepare-for-emit"),
        (
            step8,
            ["--ascend-canonicalize-cann-signature", "--canonicalize", "--cse"],
            step9,
            "090-cann-signature",
        ),
    ]
    for input_path, pass_args, output_path, report_name in stages:
        run_command(
            [opt, str(input_path), *pass_args],
            stdout_path=output_path,
            stderr_report_path=reports_dir / f"{report_name}.stderr.txt",
        )

    run_command(
        [
            translate,
            "-mlir-to-cann",
            str(step9),
            f"--tiling-space-out={tiling_space}",
            f"--artifact-manifest-out={artifact_manifest}",
            f"--host-tiling-out={host_tiling}",
            f"--cann-soc={soc}",
            "-o",
            str(kernel_cpp),
        ],
        stdout_path=run_dir / "ascend-mlir-translate.log",
        stderr_report_path=reports_dir / "100-translate.stderr.txt",
    )

    run_command(
        [
            runtime_session,
            "--kernel",
            str(kernel_cpp),
            "--kernel-kind",
            kernel_kind,
            "--soc",
            soc,
            "--output",
            str(artifact_root),
            "--name",
            kernel_name,
        ],
        stdout_path=run_dir / "runtime-session.compile.log",
        stderr_report_path=reports_dir / "runtime-session.compile.stderr.txt",
    )
    run_command(
        [
            cxx,
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(host_tiling),
            "-o",
            str(artifact_root / "host_tiling.so"),
        ],
        stdout_path=run_dir / "host-tiling.compile.log",
        stderr_report_path=reports_dir / "host-tiling.compile.stderr.txt",
    )

    artifact_case: dict[str, Any] = {
        "schema_version": 1,
        "artifact": {
            "root": str(artifact_root),
            "manifest": str(artifact_manifest),
        },
        "backend": root["backend"],
        "shape_args": root.get("shape_args", _infer_shape_args(inputs)),
        "inputs": inputs,
    }
    if outputs:
        artifact_case["outputs"] = outputs
    if expected_outputs:
        artifact_case["expected_outputs"] = expected_outputs
    if "validation" in root:
        artifact_case["validation"] = root["validation"]
    if "profiling" in root:
        artifact_case["profiling"] = root["profiling"]

    artifact_case_path = run_dir / "case.artifact.json"
    artifact_case_path.write_text(
        json.dumps(artifact_case, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return artifact_case_path


def run_case(args: argparse.Namespace) -> int:
    case_path = args.case.resolve()
    if not case_path.exists():
        raise CommandError(f"case.json does not exist: {case_path}")

    run_dir = args.out.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    root = _load_json(case_path)
    if root.get("schema_version") != 1:
        raise CommandError("case.json requires schema_version: 1")
    if "source" in root:
        artifact_case = _compile_source_case(root=root, case_path=case_path, run_dir=run_dir)
        return _run_artifact_case(artifact_case, run_dir)
    if "artifact" in root:
        return _run_artifact_case(case_path, run_dir)
    raise CommandError("case.json requires either source or artifact")
