from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from ascend_debug import layout
from ascend_debug.runner import CommandError


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


def _kernel_sort_key(kernel_id: str) -> tuple[int, str]:
    suffix = ""
    for char in reversed(kernel_id):
        if not char.isdigit():
            break
        suffix = char + suffix
    return (int(suffix), kernel_id) if suffix else (10**9, kernel_id)


def _kernel_depths(dag_summary: dict[str, Any]) -> dict[str, int]:
    nodes = dag_summary.get("nodes", {})
    if not isinstance(nodes, dict):
        raise CommandError("kernel DAG summary nodes must be an object")
    depths = {}
    for kernel_id, node in nodes.items():
        if not isinstance(kernel_id, str) or not isinstance(node, dict):
            continue
        depth = node.get("depth")
        if isinstance(depth, int) and not isinstance(depth, bool):
            depths[kernel_id] = depth
    return depths


def _failed_comparisons(tensor_diff: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = tensor_diff.get("comparisons")
    if not isinstance(comparisons, list):
        raise CommandError("tensor diff summary comparisons must be a list")
    failed = []
    for comparison in comparisons:
        if isinstance(comparison, dict) and comparison.get("status") != "pass":
            failed.append(comparison)
    return failed


def _comparison_sort_key(
    comparison: dict[str, Any],
    depths: dict[str, int],
    original_index: int,
) -> tuple[int, int, int, str]:
    kernel_id = comparison.get("kernel_id")
    if not isinstance(kernel_id, str) or not kernel_id:
        return (10**9, original_index, 10**9, "")
    depth = depths.get(kernel_id, 10**9)
    kernel_number, kernel_name = _kernel_sort_key(kernel_id)
    return (depth, original_index, kernel_number, kernel_name)


def _locate(tensor_diff: dict[str, Any], dag_summary: dict[str, Any]) -> dict[str, Any]:
    depths = _kernel_depths(dag_summary)
    failed = _failed_comparisons(tensor_diff)
    mapped_failed = [
        (index, comparison)
        for index, comparison in enumerate(failed)
        if isinstance(comparison.get("kernel_id"), str) and comparison.get("kernel_id")
    ]
    sorted_failed = sorted(
        mapped_failed,
        key=lambda item: _comparison_sort_key(item[1], depths, item[0]),
    )
    first_bad = sorted_failed[0][1] if sorted_failed else None

    failed_kernel_ids = []
    seen = set()
    for _, comparison in sorted_failed:
        kernel_id = comparison["kernel_id"]
        if kernel_id in seen:
            continue
        seen.add(kernel_id)
        failed_kernel_ids.append(kernel_id)

    if not failed:
        status = "pass"
    elif first_bad:
        status = "fail"
    else:
        status = "unknown"

    comparisons = tensor_diff.get("comparisons", [])
    return {
        "schema_version": 1,
        "tool": "ascend-debug",
        "status": status,
        "method": "dag-depth-first-failed-checkpoint",
        "comparison_count": tensor_diff.get("comparison_count", len(comparisons)),
        "failed_count": len(failed),
        "failed_kernel_count": len(failed_kernel_ids),
        "failed_kernel_ids": failed_kernel_ids,
        "unmapped_failed_count": len(failed) - len(mapped_failed),
        "first_bad_kernel": first_bad.get("kernel_id") if first_bad else None,
        "first_bad_task": first_bad.get("task_id") if first_bad else None,
        "first_bad_comparison": first_bad,
    }


def locate_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    tensor_diff = _load_json(run_dir / "summaries" / "tensor_diff.json", label="tensor diff summary")
    dag_summary = _load_json(run_dir / "graphs" / "kernel_dag.summary.json", label="kernel DAG summary")
    summary = _locate(tensor_diff, dag_summary)
    report_path = run_dir / "summaries" / "locate.json"
    layout.write_json(report_path, summary)

    first_bad_kernel = summary["first_bad_kernel"] or "none"
    first_bad_comparison = summary["first_bad_comparison"] or {}
    first_bad_comparison_id = first_bad_comparison.get("id", "none")
    print(f"ascend_debug.locate.status={summary['status']}")
    print(f"ascend_debug.locate.failed_kernels={summary['failed_kernel_count']}")
    print(f"ascend_debug.locate.first_bad_kernel={first_bad_kernel}")
    print(f"ascend_debug.locate.first_bad_comparison={first_bad_comparison_id}")
    print(f"ascend_debug.locate.report={report_path}")
    return 0
