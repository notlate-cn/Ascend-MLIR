from __future__ import annotations

import pathlib
import re
import shlex
from typing import Any


SECTION_RE = re.compile(r"^(BufferizedKernelIR|PlacementPlan|StaticMemoryPlan|MovementPlan|MemoryRealizationPlan):$")
FIELD_RE = re.compile(r"^\s+(?P<key>[A-Za-z_][A-Za-z0-9_]*) = (?P<value>.*)$")
DETAIL_RE = re.compile(
    r"^\s+(?P<kind>live_interval|workspace_slot|movement_step)\[(?P<index>\d+)\] = (?P<fields>.*)$"
)


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _kernel_sort_key(kernel_id: str) -> tuple[int, str]:
    suffix = ""
    for char in reversed(kernel_id):
        if not char.isdigit():
            break
        suffix = char + suffix
    return (int(suffix), kernel_id) if suffix else (10**9, kernel_id)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "unknown":
        return None
    if value.isdigit():
        return int(value)
    return value


def _parse_detail_fields(fields: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for token in shlex.split(fields):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed[key] = _parse_scalar(value)
    return parsed


def _normalize_static_plan(raw: dict[str, Any]) -> dict[str, Any]:
    plan = dict(raw)
    plan["live_interval_count"] = _int_value(plan.get("live_intervals"))
    plan["workspace_slot_count"] = _int_value(plan.get("workspace_slots"))
    live_intervals = []
    for interval in plan.pop("_live_intervals", []):
        live_intervals.append(
            {
                "value_id": _int_value(interval.get("value_id")),
                "start": _int_value(interval.get("start")),
                "end": _int_value(interval.get("end")),
                "place": interval.get("place") or "unknown",
                "byte_size": interval.get("byte_size"),
            }
        )
    workspace_slots = []
    for slot in plan.pop("_workspace_slots", []):
        workspace_slots.append(
            {
                "slot_id": _int_value(slot.get("slot_id")),
                "value_id": _int_value(slot.get("value_id")),
                "offset": _int_value(slot.get("offset")),
                "place": slot.get("place") or "unknown",
                "byte_size": slot.get("byte_size"),
                "byte_size_expr": slot.get("byte_size_expr"),
            }
        )
    movement_edges = []
    for edge in plan.pop("_movement_edges", []):
        movement_edges.append(
            {
                "step_id": _int_value(edge.get("step_id")),
                "value_id": _int_value(edge.get("value_id")),
                "slot_id": _int_value(edge.get("slot_id")),
                "src": edge.get("src") or "unknown",
                "dst": edge.get("dst") or "unknown",
                "path_selected": bool(edge.get("path_selected")),
                "path_variant": _int_value(edge.get("path_variant")),
                "byte_size": edge.get("byte_size"),
            }
        )
    plan["live_intervals"] = sorted(live_intervals, key=lambda item: (item["start"], item["value_id"]))
    plan["workspace_slots"] = sorted(workspace_slots, key=lambda item: item["slot_id"])
    plan["movement_edges"] = sorted(movement_edges, key=lambda item: item["step_id"])
    return plan


def _parse_realize_report(report_path: pathlib.Path) -> list[dict[str, Any]]:
    if not report_path.exists():
        return []

    kernels: dict[str, dict[str, Any]] = {}
    section = ""
    current_kernel = ""
    pending_static: dict[str, Any] | None = None
    pending_movement: dict[str, Any] | None = None

    for line in report_path.read_text(encoding="utf-8", errors="replace").splitlines():
        section_match = SECTION_RE.match(line)
        if section_match:
            section = section_match.group(1)
            if section == "StaticMemoryPlan":
                pending_static = {"_live_intervals": [], "_workspace_slots": [], "_movement_edges": []}
            elif section == "MovementPlan":
                pending_movement = {}
            continue

        detail_match = DETAIL_RE.match(line)
        if detail_match:
            fields = _parse_detail_fields(detail_match.group("fields"))
            kind = detail_match.group("kind")
            if kind == "live_interval" and pending_static is not None:
                pending_static.setdefault("_live_intervals", []).append(fields)
            elif kind == "workspace_slot" and pending_static is not None:
                pending_static.setdefault("_workspace_slots", []).append(fields)
            elif kind == "movement_step":
                kernel = current_kernel
                if pending_movement and isinstance(pending_movement.get("kernel"), str):
                    kernel = pending_movement["kernel"]
                if kernel:
                    kernels.setdefault(kernel, {}).setdefault("_movement_edges", []).append(fields)
            continue

        field_match = FIELD_RE.match(line)
        if not field_match:
            continue
        key = field_match.group("key")
        value = _parse_scalar(field_match.group("value"))

        if key == "kernel" and isinstance(value, str):
            current_kernel = value
            kernels.setdefault(current_kernel, {})
            if section == "StaticMemoryPlan" and pending_static is not None:
                pending_static["kernel_id"] = current_kernel
            if section == "MovementPlan" and pending_movement is not None:
                pending_movement["kernel"] = current_kernel
            continue

        if section == "StaticMemoryPlan" and pending_static is not None:
            pending_static[key] = value
            if current_kernel:
                kernels.setdefault(current_kernel, {}).update(pending_static)
        elif section == "MovementPlan" and current_kernel:
            if key in ("mode", "movements", "movement_demands", "workspace_reuse_candidates"):
                kernels.setdefault(current_kernel, {})[f"movement_{key}"] = value
        elif section == "MemoryRealizationPlan" and current_kernel:
            if key == "mode":
                kernels.setdefault(current_kernel, {})["realization_mode"] = value

    return [
        _normalize_static_plan(plan)
        for _, plan in sorted(kernels.items(), key=lambda item: _kernel_sort_key(item[0]))
        if plan.get("mode") == "workspace_layout" or plan.get("workspace_slots") or plan.get("_workspace_slots")
    ]


def _find_interval(value_id: int, intervals: list[dict[str, Any]]) -> dict[str, Any] | None:
    for interval in intervals:
        if interval["value_id"] == value_id:
            return interval
    return None


def _slot_byte_size(slot: dict[str, Any]) -> int:
    return _int_value(slot.get("byte_size"))


def _decorate_slots(plan: dict[str, Any]) -> None:
    intervals = plan.get("live_intervals", [])
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for slot in plan.get("workspace_slots", []):
        interval = _find_interval(slot["value_id"], intervals)
        slot["live_range"] = (
            f"{interval['start']}..{interval['end']}" if interval is not None else "unknown"
        )
        key = (slot["place"], slot["offset"])
        groups.setdefault(key, []).append(slot)

    reuse_groups = []
    for (place, offset), slots in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        reused = len(slots) > 1
        for slot in slots:
            slot["reused"] = reused
        if reused:
            reuse_groups.append(
                {
                    "place": place,
                    "offset": offset,
                    "slot_ids": [slot["slot_id"] for slot in slots],
                    "value_ids": [slot["value_id"] for slot in slots],
                }
            )
    plan["reuse_groups"] = reuse_groups


def _build_peak_timeline(plan: dict[str, Any]) -> list[dict[str, Any]]:
    intervals = plan.get("live_intervals", [])
    slots = plan.get("workspace_slots", [])
    if not intervals or not slots:
        return []

    min_time = min(interval["start"] for interval in intervals)
    max_time = max(interval["end"] for interval in intervals)
    timeline = []
    for time in range(min_time, max_time):
        active_values = []
        active_slots: dict[tuple[str, int], int] = {}
        for interval in intervals:
            if not (interval["start"] <= time < interval["end"]):
                continue
            active_values.append(interval["value_id"])
            for slot in slots:
                if slot["value_id"] != interval["value_id"]:
                    continue
                key = (slot["place"], slot["offset"])
                active_slots[key] = max(active_slots.get(key, 0), _slot_byte_size(slot))

        physical_slots = [
            {"place": place, "offset": offset, "byte_size": byte_size}
            for (place, offset), byte_size in sorted(active_slots.items(), key=lambda item: (item[0][1], item[0][0]))
        ]
        timeline.append(
            {
                "time": time,
                "usage_bytes": sum(slot["byte_size"] for slot in physical_slots),
                "active_physical_slots": physical_slots,
                "active_values": sorted(active_values),
            }
        )
    return timeline


def _dag_kernels(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not summary:
        return []
    nodes = summary.get("nodes", {})
    if not isinstance(nodes, dict):
        return []

    kernels = []
    for kernel_id in sorted(nodes, key=_kernel_sort_key):
        node = nodes[kernel_id]
        if not isinstance(node, dict):
            continue
        kernels.append(
            {
                "kernel_id": kernel_id,
                "depth": _int_value(node.get("depth")),
                "kind": node.get("kind"),
                "workspace_size": _int_value(node.get("workspace_size")),
                "output_shape": node.get("output_shape"),
                "selected_tile_shape": node.get("selected_tile_shape"),
                "input_degree": node.get("input_degree"),
                "output_degree": node.get("output_degree"),
            }
        )
    return kernels


def _build_kernel_coverage(
    dag_kernels: list[dict[str, Any]],
    detailed_kernels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    detailed_by_id = {
        kernel["kernel_id"]: kernel
        for kernel in detailed_kernels
        if isinstance(kernel.get("kernel_id"), str)
    }
    dag_ids = {kernel["kernel_id"] for kernel in dag_kernels}
    coverage = []
    for kernel in dag_kernels:
        kernel_id = kernel["kernel_id"]
        detailed = detailed_by_id.get(kernel_id)
        if detailed is not None:
            status = "realize-slot-plan"
            reason = "covered by StaticMemoryPlan workspace slots"
            workspace_bytes = detailed.get("workspace_bytes", kernel.get("workspace_size", 0))
        elif kernel.get("workspace_size", 0) > 0:
            status = "dag-workspace-only"
            reason = "DAG reports workspace, but Realize slot detail is unavailable"
            workspace_bytes = kernel.get("workspace_size", 0)
        else:
            status = "no-workspace"
            reason = "no workspace reported by DAG and no Realize slot detail"
            workspace_bytes = 0
        coverage.append(
            {
                "kernel_id": kernel_id,
                "depth": kernel.get("depth"),
                "kind": kernel.get("kind"),
                "dag_workspace_size": kernel.get("workspace_size", 0),
                "workspace_bytes": workspace_bytes,
                "memory_plan_status": status,
                "reason": reason,
            }
        )

    for kernel in detailed_kernels:
        kernel_id = kernel.get("kernel_id")
        if not isinstance(kernel_id, str) or kernel_id in dag_ids:
            continue
        coverage.append(
            {
                "kernel_id": kernel_id,
                "depth": None,
                "kind": None,
                "dag_workspace_size": None,
                "workspace_bytes": kernel.get("workspace_bytes", 0),
                "memory_plan_status": "realize-slot-plan",
                "reason": "covered by StaticMemoryPlan workspace slots; missing from DAG summary",
            }
        )
    return coverage


def _summarize_realize_memory(
    report_path: pathlib.Path,
    dag_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    plans = _parse_realize_report(report_path)
    if not plans:
        return None

    kernels = []
    for plan in plans:
        _decorate_slots(plan)
        timeline = _build_peak_timeline(plan)
        computed_peak = max((item["usage_bytes"] for item in timeline), default=0)
        reported_peak = _int_value(plan.get("peak_usage_bytes"))
        workspace_bytes = _int_value(plan.get("workspace_bytes"))
        kernel = {
            "kernel_id": plan.get("kernel_id"),
            "mode": plan.get("mode"),
            "realization_mode": plan.get("realization_mode"),
            "tracked_places": _int_value(plan.get("tracked_places")),
            "local_buffers": _int_value(plan.get("local_buffers")),
            "live_interval_count": _int_value(plan.get("live_interval_count")),
            "workspace_slot_count": _int_value(plan.get("workspace_slot_count")),
            "peak_usage_known": bool(plan.get("peak_usage_known")),
            "peak_usage_bytes_known": bool(plan.get("peak_usage_bytes_known")),
            "local_buffer_bytes": _int_value(plan.get("local_buffer_bytes")),
            "workspace_bytes": workspace_bytes,
            "peak_usage_bytes": reported_peak or computed_peak,
            "computed_peak_usage_bytes": computed_peak,
            "capacity_check_deferred": bool(plan.get("capacity_check_deferred")),
            "live_intervals": plan.get("live_intervals", []),
            "workspace_slots": plan.get("workspace_slots", []),
            "reuse_groups": plan.get("reuse_groups", []),
            "peak_timeline": timeline,
            "movement_edges": plan.get("movement_edges", []),
        }
        kernels.append(kernel)

    dag_kernels = _dag_kernels(dag_summary)
    coverage = _build_kernel_coverage(dag_kernels, kernels)
    kernel_count = len(coverage) if coverage else len(kernels)
    return {
        "schema_version": 1,
        "tool": "ascend-debug",
        "source": "reports/040-realize.report.txt",
        "analysis_level": "realize-memory-plan",
        "note": "Realize memory plan derived from StaticMemoryPlan live intervals, workspace slots, and movement edges.",
        "kernel_count": kernel_count,
        "detailed_kernel_count": len(kernels),
        "unplanned_kernel_count": max(kernel_count - len(kernels), 0),
        "total_workspace_bytes": sum(kernel["workspace_bytes"] for kernel in kernels),
        "peak_workspace_bytes": max((kernel["peak_usage_bytes"] for kernel in kernels), default=0),
        "slot_reuse_group_count": sum(len(kernel["reuse_groups"]) for kernel in kernels),
        "movement_edge_count": sum(len(kernel["movement_edges"]) for kernel in kernels),
        "kernel_coverage": coverage,
        "kernels": kernels,
    }


def _summarize_dag_workspace(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    dag_kernels = _dag_kernels(summary)
    if not dag_kernels:
        return None

    kernels = []
    by_depth: dict[int, dict[str, Any]] = {}
    for kernel in dag_kernels:
        depth = _int_value(kernel.get("depth"))
        workspace_size = _int_value(kernel.get("workspace_size"))
        kernel_id = kernel["kernel_id"]
        kernels.append(kernel)
        bucket = by_depth.setdefault(
            depth,
            {
                "depth": depth,
                "kernel_count": 0,
                "workspace_bytes": 0,
                "peak_workspace_bytes": 0,
                "kernels": [],
            },
        )
        bucket["kernel_count"] += 1
        bucket["workspace_bytes"] += workspace_size
        bucket["peak_workspace_bytes"] = max(bucket["peak_workspace_bytes"], workspace_size)
        bucket["kernels"].append(kernel_id)

    total_workspace = sum(kernel["workspace_size"] for kernel in kernels)
    peak_workspace = max((kernel["workspace_size"] for kernel in kernels), default=0)
    return {
        "schema_version": 1,
        "tool": "ascend-debug",
        "source": "kernel_dag.summary.json",
        "analysis_level": "workspace-overview",
        "note": "Workspace overview derived from DAG artifacts; Realize slot lifetime is not available here.",
        "kernel_count": len(kernels),
        "workspace_kernel_count": sum(1 for kernel in kernels if kernel["workspace_size"] > 0),
        "total_workspace_bytes": total_workspace,
        "peak_workspace_bytes": peak_workspace,
        "workspace_by_depth": [by_depth[depth] for depth in sorted(by_depth)],
        "kernels": kernels,
    }


def summarize_memory(run_dir: pathlib.Path, dag_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    realize_summary = _summarize_realize_memory(
        run_dir / "reports/040-realize.report.txt",
        dag_summary,
    )
    if realize_summary:
        return realize_summary
    return _summarize_dag_workspace(dag_summary)
