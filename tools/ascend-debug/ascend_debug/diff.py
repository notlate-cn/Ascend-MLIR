from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
import struct
from dataclasses import dataclass
from typing import Any

from ascend_debug import layout
from ascend_debug.runner import CommandError


DEFAULT_ATOL = 1.0e-5
DEFAULT_RTOL = 1.0e-5

_DTYPE_FORMATS = {
    "f2": "e",
    "f4": "f",
    "f8": "d",
    "i1": "b",
    "i2": "h",
    "i4": "i",
    "i8": "q",
    "u1": "B",
    "u2": "H",
    "u4": "I",
    "u8": "Q",
}


@dataclass(frozen=True)
class NpyArray:
    dtype: str
    shape: tuple[int, ...]
    values: tuple[float, ...]


def _load_json(path: pathlib.Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CommandError(f"{label} not found: {path}") from error
    except (OSError, UnicodeDecodeError) as error:
        raise CommandError(f"could not read {label}: {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CommandError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise CommandError(f"{label} must be a JSON object: {path}")
    return value


def _validate_relative_path(value: Any, *, label: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str):
        raise CommandError(f"{label} path must be a string")
    path = pathlib.PurePosixPath(value)
    if value in ("", ".") or path.is_absolute() or ".." in path.parts:
        raise CommandError(f"{label} path must stay inside run dir: {value}")
    return path


def _read_npy(path: pathlib.Path) -> NpyArray:
    try:
        data = path.read_bytes()
    except FileNotFoundError as error:
        raise CommandError(f"tensor file not found: {path}") from error
    except OSError as error:
        raise CommandError(f"could not read tensor file: {path}: {error}") from error

    if len(data) < 10 or not data.startswith(b"\x93NUMPY"):
        raise CommandError(f"tensor file is not a NPY file: {path}")

    major = data[6]
    minor = data[7]
    if (major, minor) == (1, 0):
        header_size_len = 2
        header_len = struct.unpack_from("<H", data, 8)[0]
    elif (major, minor) in ((2, 0), (3, 0)):
        header_size_len = 4
        header_len = struct.unpack_from("<I", data, 8)[0]
    else:
        raise CommandError(f"unsupported NPY version {major}.{minor}: {path}")

    header_start = 8 + header_size_len
    header_end = header_start + header_len
    try:
        header = ast.literal_eval(data[header_start:header_end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise CommandError(f"invalid NPY header: {path}: {error}") from error
    if not isinstance(header, dict):
        raise CommandError(f"NPY header must be a dict: {path}")

    descr = header.get("descr")
    shape = header.get("shape")
    if not isinstance(descr, str):
        raise CommandError(f"NPY header missing dtype descriptor: {path}")
    if not isinstance(shape, tuple) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise CommandError(f"NPY header shape must be a non-negative integer tuple: {path}")
    if header.get("fortran_order") not in (False, True):
        raise CommandError(f"NPY header fortran_order must be a boolean: {path}")

    byte_order = descr[0] if descr[0] in "<>|=" else "|"
    dtype = descr[1:] if descr[0] in "<>|=" else descr
    if byte_order == ">":
        endian = ">"
    else:
        endian = "<"
    if dtype not in _DTYPE_FORMATS:
        raise CommandError(f"unsupported NPY dtype {descr}: {path}")

    count = math.prod(shape)
    format_code = _DTYPE_FORMATS[dtype]
    item_size = struct.calcsize(endian + format_code)
    payload = data[header_end:]
    expected_size = count * item_size
    if len(payload) < expected_size:
        raise CommandError(f"NPY payload is truncated: {path}")
    if count == 0:
        values: tuple[float, ...] = ()
    else:
        unpacked = struct.unpack_from(endian + format_code * count, payload, 0)
        values = tuple(float(item) for item in unpacked)
    return NpyArray(dtype=descr, shape=shape, values=values)


def _relative_error(abs_error: float, expected: float) -> float:
    denominator = abs(expected)
    if denominator == 0:
        return 0.0 if abs_error == 0 else math.inf
    return abs_error / denominator


def _compare_arrays(
    *,
    comparison_id: str,
    lhs_path: str,
    rhs_path: str,
    lhs: NpyArray,
    rhs: NpyArray,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if lhs.shape != rhs.shape:
        return {
            "id": comparison_id,
            "status": "fail",
            "reason": "shape-mismatch",
            "lhs": lhs_path,
            "rhs": rhs_path,
            "lhs_shape": list(lhs.shape),
            "rhs_shape": list(rhs.shape),
            "lhs_dtype": lhs.dtype,
            "rhs_dtype": rhs.dtype,
            "atol": atol,
            "rtol": rtol,
        }
    if len(lhs.values) != len(rhs.values):
        raise CommandError(f"internal tensor size mismatch for {comparison_id}")

    abs_errors = [abs(actual - expected) for actual, expected in zip(rhs.values, lhs.values)]
    rel_errors = [_relative_error(error, expected) for error, expected in zip(abs_errors, lhs.values)]
    max_abs = max(abs_errors, default=0.0)
    max_rel = max(rel_errors, default=0.0)
    mean_abs = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    passed = all(error <= atol + rtol * abs(expected) for error, expected in zip(abs_errors, lhs.values))
    return {
        "id": comparison_id,
        "status": "pass" if passed else "fail",
        "lhs": lhs_path,
        "rhs": rhs_path,
        "shape": list(lhs.shape),
        "lhs_dtype": lhs.dtype,
        "rhs_dtype": rhs.dtype,
        "element_count": len(lhs.values),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "mean_abs_error": mean_abs,
    }


def _comparison_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = manifest.get("comparisons")
    if not isinstance(comparisons, list):
        raise CommandError("tensor manifest comparisons must be a list")
    entries = []
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            raise CommandError(f"tensor comparison {index} must be an object")
        comparison_id = comparison.get("id", f"comparison_{index}")
        if not isinstance(comparison_id, str) or not comparison_id:
            raise CommandError(f"tensor comparison {index} id must be a non-empty string")
        lhs = _validate_relative_path(comparison.get("lhs"), label=f"tensor comparison {index} lhs")
        rhs = _validate_relative_path(comparison.get("rhs"), label=f"tensor comparison {index} rhs")
        atol = comparison.get("atol", DEFAULT_ATOL)
        rtol = comparison.get("rtol", DEFAULT_RTOL)
        if not isinstance(atol, (int, float)) or isinstance(atol, bool) or atol < 0:
            raise CommandError(f"tensor comparison {index} atol must be a non-negative number")
        if not isinstance(rtol, (int, float)) or isinstance(rtol, bool) or rtol < 0:
            raise CommandError(f"tensor comparison {index} rtol must be a non-negative number")
        entries.append(
            {
                "id": comparison_id,
                "lhs": str(lhs),
                "rhs": str(rhs),
                "atol": float(atol),
                "rtol": float(rtol),
            }
        )
    return entries


def diff_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "tensors" / "manifest.json"
    manifest = _load_json(manifest_path, label="tensor manifest")
    if manifest.get("schema_version") != 1:
        raise CommandError(f"tensor manifest schema_version must be 1: {manifest_path}")

    results = []
    for comparison in _comparison_entries(manifest):
        lhs = _read_npy(run_dir / comparison["lhs"])
        rhs = _read_npy(run_dir / comparison["rhs"])
        results.append(
            _compare_arrays(
                comparison_id=comparison["id"],
                lhs_path=comparison["lhs"],
                rhs_path=comparison["rhs"],
                lhs=lhs,
                rhs=rhs,
                atol=comparison["atol"],
                rtol=comparison["rtol"],
            )
        )

    failed = [result for result in results if result["status"] != "pass"]
    summary = {
        "schema_version": 1,
        "tool": "ascend-debug",
        "status": "pass" if not failed else "fail",
        "comparison_count": len(results),
        "failed_count": len(failed),
        "comparisons": results,
    }
    report_path = run_dir / "summaries" / "tensor_diff.json"
    layout.write_json(report_path, summary)

    print(f"ascend_debug.diff.comparisons={len(results)}")
    print(f"ascend_debug.diff.failed={len(failed)}")
    print(f"ascend_debug.diff.status={summary['status']}")
    print(f"ascend_debug.diff.report={report_path}")
    return 0 if not failed else 1
