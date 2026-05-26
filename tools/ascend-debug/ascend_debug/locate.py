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


def _sort_kernel_ids(kernel_ids: list[str] | set[str], depths: dict[str, int]) -> list[str]:
    return sorted(kernel_ids, key=lambda kernel_id: (depths.get(kernel_id, 10**9), *_kernel_sort_key(kernel_id)))


def _edge_maps(dag_summary: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    nodes = dag_summary.get("nodes", {})
    if not isinstance(nodes, dict):
        raise CommandError("kernel DAG summary nodes must be an object")
    pred: dict[str, list[str]] = {kernel_id: [] for kernel_id in nodes if isinstance(kernel_id, str)}
    succ: dict[str, list[str]] = {kernel_id: [] for kernel_id in nodes if isinstance(kernel_id, str)}
    edges = dag_summary.get("edges", [])
    if not isinstance(edges, list):
        raise CommandError("kernel DAG summary edges must be a list")
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        succ.setdefault(src, []).append(dst)
        pred.setdefault(dst, []).append(src)
        pred.setdefault(src, pred.get(src, []))
        succ.setdefault(dst, succ.get(dst, []))
    return pred, succ


def _reachable(start: str, edges: dict[str, list[str]], depths: dict[str, int]) -> list[str]:
    visited: set[str] = set()
    stack = list(edges.get(start, []))
    while stack:
        kernel_id = stack.pop()
        if kernel_id in visited:
            continue
        visited.add(kernel_id)
        stack.extend(edges.get(kernel_id, []))
    return _sort_kernel_ids(visited, depths)


def _failed_comparisons(tensor_diff: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = tensor_diff.get("comparisons")
    if not isinstance(comparisons, list):
        raise CommandError("tensor diff summary comparisons must be a list")
    failed = []
    for comparison in comparisons:
        if isinstance(comparison, dict) and comparison.get("status") != "pass":
            failed.append(comparison)
    return failed


def _mapped_kernel_ids(comparisons: list[Any], *, status: str, depths: dict[str, int]) -> list[str]:
    kernel_ids = []
    seen = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict) or comparison.get("status") != status:
            continue
        kernel_id = comparison.get("kernel_id")
        if not isinstance(kernel_id, str) or not kernel_id or kernel_id in seen:
            continue
        seen.add(kernel_id)
        kernel_ids.append(kernel_id)
    return _sort_kernel_ids(kernel_ids, depths)


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
    pred, succ = _edge_maps(dag_summary)
    comparisons = tensor_diff.get("comparisons", [])
    failed = _failed_comparisons(tensor_diff)
    passed_kernel_ids = _mapped_kernel_ids(comparisons, status="pass", depths=depths)
    passed_kernel_set = set(passed_kernel_ids)
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
    failed_kernel_set = set(failed_kernel_ids)

    if not failed:
        status = "pass"
    elif first_bad:
        status = "fail"
    else:
        status = "unknown"

    first_bad_kernel = first_bad.get("kernel_id") if first_bad else None
    first_bad_depth = depths.get(first_bad_kernel) if isinstance(first_bad_kernel, str) else None
    first_bad_context: dict[str, Any] = {}
    if isinstance(first_bad_kernel, str):
        direct_upstream = _sort_kernel_ids(pred.get(first_bad_kernel, []), depths)
        direct_downstream = _sort_kernel_ids(succ.get(first_bad_kernel, []), depths)
        upstream = _reachable(first_bad_kernel, pred, depths)
        downstream = _reachable(first_bad_kernel, succ, depths)
        first_bad_context = {
            "direct_upstream": direct_upstream,
            "direct_downstream": direct_downstream,
            "upstream_checked_passed": [kernel_id for kernel_id in upstream if kernel_id in passed_kernel_set],
            "downstream_failed": [kernel_id for kernel_id in downstream if kernel_id in failed_kernel_set],
            "unchecked_direct_upstream": [
                kernel_id
                for kernel_id in direct_upstream
                if kernel_id not in passed_kernel_set and kernel_id not in failed_kernel_set
            ],
        }

    return {
        "schema_version": 1,
        "tool": "ascend-debug",
        "status": status,
        "method": "dag-depth-first-failed-checkpoint",
        "comparison_count": tensor_diff.get("comparison_count", len(comparisons)),
        "failed_count": len(failed),
        "failed_kernel_count": len(failed_kernel_ids),
        "failed_kernel_ids": failed_kernel_ids,
        "passed_kernel_ids": passed_kernel_ids,
        "unmapped_failed_count": len(failed) - len(mapped_failed),
        "first_bad_kernel": first_bad_kernel,
        "first_bad_depth": first_bad_depth,
        "first_bad_task": first_bad.get("task_id") if first_bad else None,
        "first_bad_comparison": first_bad,
        "first_bad_context": first_bad_context,
    }


def locate_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    tensor_diff = _load_json(run_dir / "summaries" / "tensor_diff.json", label="tensor diff summary")
    dag_summary = _load_json(run_dir / "graphs" / "kernel_dag.summary.json", label="kernel DAG summary")
    summary = _locate(tensor_diff, dag_summary)
    report_path = run_dir / "summaries" / "locate.json"
    layout.write_json(report_path, summary)

    first_bad_kernel = summary["first_bad_kernel"] or "none"
    first_bad_depth = summary["first_bad_depth"] or "none"
    first_bad_comparison = summary["first_bad_comparison"] or {}
    first_bad_comparison_id = first_bad_comparison.get("id", "none")
    first_bad_context = summary.get("first_bad_context", {})
    upstream_checked_passed = first_bad_context.get("upstream_checked_passed", [])
    unchecked_direct_upstream = first_bad_context.get("unchecked_direct_upstream", [])
    downstream_failed = first_bad_context.get("downstream_failed", [])
    upstream_checked_passed_text = ",".join(upstream_checked_passed) if upstream_checked_passed else "none"
    unchecked_direct_upstream_text = ",".join(unchecked_direct_upstream) if unchecked_direct_upstream else "none"
    downstream_failed_text = ",".join(downstream_failed) if downstream_failed else "none"
    print(f"ascend_debug.locate.status={summary['status']}")
    print(f"ascend_debug.locate.failed_kernels={summary['failed_kernel_count']}")
    print(f"ascend_debug.locate.first_bad_kernel={first_bad_kernel}")
    print(f"ascend_debug.locate.first_bad_depth={first_bad_depth}")
    print(f"ascend_debug.locate.first_bad_comparison={first_bad_comparison_id}")
    print(f"ascend_debug.locate.upstream_checked_passed={upstream_checked_passed_text}")
    print(f"ascend_debug.locate.unchecked_direct_upstream={unchecked_direct_upstream_text}")
    print(f"ascend_debug.locate.downstream_failed={downstream_failed_text}")
    print(f"ascend_debug.locate.report={report_path}")
    return 0
