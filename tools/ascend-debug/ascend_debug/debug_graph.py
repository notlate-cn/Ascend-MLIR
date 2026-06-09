from __future__ import annotations

import html
import hashlib
import json
import pathlib
import re
from typing import Any

from ascend_debug import layout, stage_graph, timeline_model

PHASE_ORDER = ("Normalize", "Kernelize", "Schedule", "Realize", "Translate")


def _cell(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _infer_stage_phase(name: Any) -> str:
    text = str(name or "").lower()
    if text == "source":
        return "Source"
    if "normalize" in text:
        return "Normalize"
    if "kernelize" in text:
        return "Kernelize"
    if "schedule" in text:
        return "Schedule"
    if "realize" in text:
        return "Realize"
    translate_markers = (
        "compute-lower",
        "parallelize",
        "prepare-for-emit",
        "cann-signature",
    )
    if any(marker in text for marker in translate_markers):
        return "Translate"
    if "kernel-dag" in text:
        return "Kernel DAG"
    return str(name or "Stage")


def _stage_display_contract(stage: dict[str, Any]) -> dict[str, str]:
    name = str(stage.get("name") or "")
    phase = stage.get("phase")
    step = stage.get("step")
    step_info = stage.get("step_info")
    if not isinstance(step_info, dict):
        step_info = {}

    stage_label = phase if isinstance(phase, str) and phase else _infer_stage_phase(name)
    step_id = step if isinstance(step, str) and step else (name or "stage")
    step_label = str(step_info.get("title") or step_id)
    return {
        "stage_label": stage_label,
        "step_id": step_id,
        "step_label": step_label,
    }


def _load_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _split_report_list(value: str) -> list[str]:
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return []
    body = value[1:-1].strip()
    if not body:
        return []
    items: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(body):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            item = body[start:index].strip()
            if item:
                items.append(item)
            start = index + 1
    item = body[start:].strip()
    if item:
        items.append(item)
    return items


def _parse_schedule_problem_report(text: str) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None

    def finish_current() -> None:
        if not current:
            return
        kernel = current.get("kernel")
        if isinstance(kernel, str) and kernel:
            current["source"] = "schedule_report"
            contracts[kernel] = dict(current)

    list_fields = {
        "template_tags",
        "shape_constraints",
        "structure_constraints",
        "tileable_axes",
        "required_reduction_axes",
    }
    int_fields = {"result_rank", "guard_budget"}
    field_re = re.compile(r"^\s{2}(?P<name>[A-Za-z_]+)\s*=\s*(?P<value>.*)$")

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "ScheduleProblem:":
            finish_current()
            current = {}
            continue
        if current is None:
            continue
        if line == stripped and stripped.endswith(":"):
            finish_current()
            current = None
            continue
        match = field_re.match(line)
        if not match:
            continue
        name = match.group("name")
        value = match.group("value").strip()
        if name in list_fields:
            current[name] = _split_report_list(value)
        elif name in int_fields and re.fullmatch(r"-?[0-9]+", value):
            current[name] = int(value)
        elif name in {"kernel", "role", "result_shape"}:
            current[name] = value
    finish_current()
    return contracts


def _load_schedule_axis_contracts(
    run_dir: pathlib.Path, manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    reports = manifest.get("reports")
    if not isinstance(reports, list):
        return contracts
    for report in reports:
        if not isinstance(report, dict):
            continue
        stage = report.get("stage")
        rel_path = report.get("path")
        if not isinstance(rel_path, str):
            continue
        if stage != "schedule" and "schedule.report" not in rel_path:
            continue
        try:
            text = (run_dir / rel_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        contracts.update(_parse_schedule_problem_report(text))
    return contracts


def _schedule_axis_contract_badge(contract: dict[str, Any]) -> str:
    fields = (
        "shape_constraints",
        "structure_constraints",
        "tileable_axes",
        "required_reduction_axes",
    )
    return "schedule axis contract" if any(contract.get(field) for field in fields) else ""


def _annotate_schedule_axis_contracts(
    stages: list[dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
) -> None:
    if not contracts:
        return
    for stage in stages:
        graph = stage.get("graph")
        if not isinstance(graph, dict):
            continue
        for node in graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            kernel_id = node.get("kernel_id")
            if not isinstance(kernel_id, str) or kernel_id not in contracts:
                continue
            semantic = node.setdefault("semantic_attrs", {})
            if not isinstance(semantic, dict):
                semantic = {}
                node["semantic_attrs"] = semantic
            schedule = semantic.setdefault("schedule", {})
            if not isinstance(schedule, dict):
                schedule = {}
                semantic["schedule"] = schedule
            schedule["axis_contract"] = contracts[kernel_id]
            badge = _schedule_axis_contract_badge(contracts[kernel_id])
            if badge:
                badges = node.setdefault("badges", [])
                if isinstance(badges, list) and badge not in badges:
                    badges.append(badge)


def _stage_record(
    *,
    run_dir: pathlib.Path,
    stage: dict[str, Any],
    graph_view: dict[str, Any],
) -> dict[str, Any] | None:
    graph_json_rel = graph_view.get("json_rel_path")
    if not isinstance(graph_json_rel, str):
        return None
    graph = _load_json(run_dir / graph_json_rel)
    if not graph:
        return None
    record = {
        "order": stage["order"],
        "name": stage["name"],
        "path": stage["path"],
        "stage_view_path": f"views/{stage['path']}.html",
        "graph_json_path": graph_json_rel,
        "node_count": graph.get("node_count", 0),
        "edge_count": graph.get("edge_count", 0),
        "kernel_count": graph.get("kernel_count", 0),
        "graph": graph,
    }
    record.update(_stage_display_contract(stage))
    for field in ("phase", "step"):
        value = stage.get(field)
        if isinstance(value, str) and value:
            record[field] = value
    step_info = stage.get("step_info")
    if isinstance(step_info, dict):
        record["step_info"] = step_info
    return record


def _select_primary_stage(stages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for preferred in (
        "kernelize-out",
        "030-kernelize-out",
        "schedule-out",
        "040-schedule-out",
        "realize-out",
        "050-realize-out",
    ):
        for item in stages:
            if item["name"] == preferred:
                return item
    with_kernels = [item for item in stages if item.get("kernel_count", 0) > 0]
    if with_kernels:
        return max(with_kernels, key=lambda item: item["order"])
    return stages[-1] if stages else None


def _stage_brief(stage: dict[str, Any], index: int) -> dict[str, Any]:
    brief = {
        "stage_index": index,
        "order": stage.get("order"),
        "name": stage.get("name"),
        "path": stage.get("path"),
        "stage_view_path": stage.get("stage_view_path"),
    }
    for field in ("phase", "step", "stage_label", "step_id", "step_label"):
        value = stage.get(field)
        if isinstance(value, str) and value:
            brief[field] = value
    step_info = stage.get("step_info")
    if isinstance(step_info, dict):
        brief["step_info"] = step_info
    same_as_previous = stage.get("same_as_previous")
    if isinstance(same_as_previous, dict):
        brief["same_as_previous"] = same_as_previous
    return brief


def _stage_brief_steps(
    entries: list[tuple[int | None, dict[str, Any] | None]],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for index, stage in entries:
        if index is None or stage is None:
            continue
        key = (index, stage.get("order"), stage.get("path"))
        if key in seen:
            continue
        seen.add(key)
        steps.append(_stage_brief(stage, index))
    return steps


def _stage_connectivity_record(stage: dict[str, Any]) -> dict[str, Any]:
    graph = stage.get("graph")
    connectivity = graph.get("connectivity") if isinstance(graph, dict) else {}
    if not isinstance(connectivity, dict):
        connectivity = {}
    return {
        "order": stage.get("order"),
        "name": stage.get("name"),
        "path": stage.get("path"),
        "node_count": stage.get("node_count", 0),
        "edge_count": stage.get("edge_count", 0),
        "component_count": connectivity.get("component_count", 0),
        "isolated_count": connectivity.get("isolated_count", 0),
        "suspicious_isolated_count": connectivity.get("suspicious_isolated_count", 0),
        "dangling_effect_count": connectivity.get("dangling_effect_count", 0),
        "allowed_terminal_count": connectivity.get("allowed_terminal_count", 0),
        "edge_kind_counts": connectivity.get("edge_kind_counts", {}),
    }


def _stage_digest(run_dir: pathlib.Path, stage: dict[str, Any] | None) -> str | None:
    if not isinstance(stage, dict):
        return None
    path = stage.get("path")
    if not isinstance(path, str):
        return None
    try:
        text = (run_dir / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    lines = text.splitlines()
    start = 0
    while start < len(lines):
        stripped = lines[start].strip()
        if stripped == "" or stripped.startswith("//"):
            start += 1
            continue
        break
    payload = "\n".join(lines[start:])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _annotate_stage_equivalence(run_dir: pathlib.Path, stages: list[dict[str, Any]]) -> None:
    previous_stage: dict[str, Any] | None = None
    previous_digest: str | None = None
    for stage in stages:
        digest = _stage_digest(run_dir, stage)
        if (
            previous_stage is not None
            and digest is not None
            and previous_digest is not None
            and digest == previous_digest
        ):
            stage["same_as_previous"] = {
                "stage": previous_stage.get("name"),
                "order": previous_stage.get("order"),
                "reason": "IR payload is identical to the previous dumped step after ignoring debug header comments.",
            }
        previous_stage = stage
        previous_digest = digest


def _build_stage_groups(run_dir: pathlib.Path, stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {stage.get("name"): (index, stage) for index, stage in enumerate(stages)}
    groups: list[dict[str, Any]] = []
    previous_output: tuple[int, dict[str, Any]] | None = None

    source_entry = by_name.get("source")
    if source_entry:
        source_index, source_stage = source_entry
        source_brief = _stage_brief(source_stage, source_index)
        groups.append(
            {
                "name": "source",
                "kind": "source",
                "label": "source",
                "default_stage": source_brief,
                "input_stage": None,
                "output_stage": source_brief,
                "previous_output_stage": None,
                "input_same_as_previous_output": False,
                "steps": [source_brief],
            }
        )
        previous_output = source_entry

    phase_entries: dict[str, list[tuple[int, dict[str, Any]]]] = {phase: [] for phase in PHASE_ORDER}
    for index, stage in enumerate(stages):
        phase = stage.get("phase")
        if isinstance(phase, str) and phase in phase_entries:
            phase_entries[phase].append((index, stage))

    if any(phase_entries.values()):
        for phase in PHASE_ORDER:
            entries = phase_entries[phase]
            if not entries:
                continue
            default_index, default_stage = entries[-1]
            previous_index, previous_stage = previous_output if previous_output else (None, None)
            groups.append(
                {
                    "name": phase.lower().replace(" ", "-"),
                    "kind": "phase",
                    "label": phase,
                    "default_stage": _stage_brief(default_stage, default_index),
                    "input_stage": None,
                    "output_stage": _stage_brief(default_stage, default_index),
                    "previous_output_stage": _stage_brief(previous_stage, previous_index)
                    if previous_stage is not None and previous_index is not None
                    else None,
                    "input_same_as_previous_output": False,
                    "steps": [
                        _stage_brief(stage, index)
                        for index, stage in entries
                    ],
                }
            )
            previous_output = entries[-1]
        return groups

    for name, input_name, output_name in (
        ("normalize", "normalize-in", "normalize-out"),
        ("kernelize", "kernelize-in", "kernelize-out"),
        ("schedule", "schedule-in", "schedule-out"),
        ("realize", "realize-in", "realize-out"),
    ):
        output_entry = by_name.get(output_name)
        if not output_entry:
            continue
        input_entry = by_name.get(input_name)
        output_index, output_stage = output_entry
        input_index, input_stage = input_entry if input_entry else (None, None)
        previous_index, previous_stage = previous_output if previous_output else (None, None)
        same_as_previous = (
            input_stage is not None
            and previous_stage is not None
            and _stage_digest(run_dir, input_stage) == _stage_digest(run_dir, previous_stage)
        )
        step_briefs = _stage_brief_steps(
            [
                (input_index, input_stage),
                (output_index, output_stage),
            ]
        )
        groups.append(
            {
                "name": name,
                "kind": "pass",
                "label": name.capitalize(),
                "default_stage": _stage_brief(output_stage, output_index),
                "input_stage": _stage_brief(input_stage, input_index) if input_stage is not None and input_index is not None else None,
                "output_stage": _stage_brief(output_stage, output_index),
                "previous_output_stage": _stage_brief(previous_stage, previous_index) if previous_stage is not None and previous_index is not None else None,
                "input_same_as_previous_output": same_as_previous,
                "steps": step_briefs,
            }
        )
        previous_output = output_entry

    return groups


def _stage_kernel_ids(stages: list[dict[str, Any]]) -> set[str]:
    kernel_ids: set[str] = set()
    for stage in stages:
        graph = stage.get("graph")
        if not isinstance(graph, dict):
            continue
        for node in graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            kernel_id = node.get("kernel_id")
            if isinstance(kernel_id, str) and kernel_id:
                kernel_ids.add(kernel_id)
    return kernel_ids


def _kernel_detail_views(
    stages: list[dict[str, Any]], kernel_summary: dict[str, Any] | None
) -> dict[str, str]:
    if not isinstance(kernel_summary, dict):
        return {}
    nodes = kernel_summary.get("nodes")
    if not isinstance(nodes, dict):
        return {}
    real_kernel_ids = sorted(
        kernel_id for kernel_id in nodes if isinstance(kernel_id, str) and kernel_id
    )
    views = {
        kernel_id: f"views/kernels/{kernel_id}.html" for kernel_id in real_kernel_ids
    }
    if len(real_kernel_ids) == 1:
        only_kernel = real_kernel_ids[0]
        only_view = views[only_kernel]
        for kernel_id in _stage_kernel_ids(stages):
            views.setdefault(kernel_id, only_view)
    return views


def _node_base_key(node: dict[str, Any]) -> str:
    op_name = str(node.get("op_name") or "")
    result_type = str(node.get("result_type") or "")
    input_count = len(node.get("input_values") or [])
    result_count = len(node.get("result_values") or [])
    if op_name == "func.arg":
        return f"{op_name}|{node.get('label') or ''}|{result_type}"
    if op_name == "func.return":
        return f"{op_name}|{input_count}"
    return f"{op_name}|{result_type}|inputs={input_count}|results={result_count}"


def _semantic_nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    counts: dict[str, int] = {}
    semantic: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        base_key = _node_base_key(node)
        ordinal = counts.get(base_key, 0)
        counts[base_key] = ordinal + 1
        semantic[f"{base_key}#{ordinal}"] = node
    return semantic


def _node_diff_facts(node: dict[str, Any]) -> dict[str, Any]:
    semantic = node.get("semantic_attrs")
    if not isinstance(semantic, dict):
        semantic = {}
    schedule = semantic.get("schedule")
    if not isinstance(schedule, dict):
        schedule = {}
    return {
        "op_name": node.get("op_name"),
        "result_type": node.get("result_type"),
        "input_count": len(node.get("input_values") or []),
        "result_count": len(node.get("result_values") or []),
        "kernel_id": node.get("kernel_id"),
        "op_role": node.get("op_role"),
        "schedule_decision_id": node.get("schedule_decision_id"),
        "schedule_axis_contract": schedule.get("axis_contract"),
        "workspace_size_bytes": node.get("workspace_size_bytes"),
    }


def _node_brief(node: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    return {
        "id": node.get("id"),
        "line": node.get("line"),
        "op_name": node.get("op_name"),
        "label": node.get("label"),
        "result_type": node.get("result_type"),
        "kernel_id": node.get("kernel_id"),
        "op_role": node.get("op_role"),
        "schedule_decision_id": node.get("schedule_decision_id"),
        "workspace_size_bytes": node.get("workspace_size_bytes"),
    }


def _changed_fields(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    changes = []
    before_facts = _node_diff_facts(before)
    after_facts = _node_diff_facts(after)
    for field in sorted(after_facts):
        if before_facts.get(field) != after_facts.get(field):
            changes.append(
                {
                    "field": field,
                    "before": before_facts.get(field),
                    "after": after_facts.get(field),
                }
            )
    return changes


def _semantic_functions(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    functions: dict[str, dict[str, Any]] = {}
    for function in graph.get("functions", []):
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            functions[name] = function
    return functions


def _function_diff_facts(function: dict[str, Any]) -> dict[str, Any]:
    semantic = function.get("semantic_attrs")
    if not isinstance(semantic, dict):
        semantic = {}
    normalize = semantic.get("normalize")
    if not isinstance(normalize, dict):
        normalize = {}
    return {
        "normalized": normalize.get("normalized"),
        "symbol_constraints": normalize.get("symbol_constraints")
        if "symbol_constraints" in normalize
        else None,
    }


def _changed_function_fields(
    before: dict[str, Any], after: dict[str, Any]
) -> list[dict[str, Any]]:
    changes = []
    before_facts = _function_diff_facts(before)
    after_facts = _function_diff_facts(after)
    for field in sorted(after_facts):
        if before_facts.get(field) != after_facts.get(field):
            changes.append(
                {
                    "field": field,
                    "before": before_facts.get(field),
                    "after": after_facts.get(field),
                }
            )
    return changes


def _compute_stage_diffs(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diffs = []
    for before_index, (before, after) in enumerate(zip(stages, stages[1:])):
        after_index = before_index + 1
        before_nodes = _semantic_nodes(before.get("graph", {}))
        after_nodes = _semantic_nodes(after.get("graph", {}))
        before_functions = _semantic_functions(before.get("graph", {}))
        after_functions = _semantic_functions(after.get("graph", {}))
        before_keys = set(before_nodes)
        after_keys = set(after_nodes)
        added_keys = sorted(after_keys - before_keys)
        removed_keys = sorted(before_keys - after_keys)
        common_keys = sorted(before_keys & after_keys)
        common_function_names = sorted(set(before_functions) & set(after_functions))
        changed = []
        changed_functions = []
        unchanged_count = 0
        node_status: dict[str, dict[str, Any]] = {}

        for key in added_keys:
            node = after_nodes[key]
            node_id = node.get("id")
            if isinstance(node_id, str):
                node_status[node_id] = {
                    "status": "added",
                    "semantic_key": key,
                    "changes": [],
                }

        for key in common_keys:
            before_node = before_nodes[key]
            after_node = after_nodes[key]
            changes = _changed_fields(before_node, after_node)
            after_node_id = after_node.get("id")
            if changes:
                changed.append(
                    {
                        "semantic_key": key,
                        "before": _node_brief(before_node),
                        "after": _node_brief(after_node),
                        "changes": changes,
                    }
                )
                if isinstance(after_node_id, str):
                    node_status[after_node_id] = {
                        "status": "changed",
                        "semantic_key": key,
                        "changes": changes,
                    }
            else:
                unchanged_count += 1
                if isinstance(after_node_id, str):
                    node_status[after_node_id] = {
                        "status": "unchanged",
                        "semantic_key": key,
                        "changes": [],
                    }

        for function_name in common_function_names:
            before_function = before_functions[function_name]
            after_function = after_functions[function_name]
            changes = _changed_function_fields(before_function, after_function)
            if changes:
                changed_functions.append(
                    {
                        "name": function_name,
                        "line": after_function.get("line"),
                        "changes": changes,
                    }
                )

        diffs.append(
            {
                "from_stage": _stage_brief(before, before_index),
                "to_stage": _stage_brief(after, after_index),
                "added_count": len(added_keys),
                "removed_count": len(removed_keys),
                "changed_count": len(changed) + len(changed_functions),
                "unchanged_count": unchanged_count,
                "added_nodes": [_node_brief(after_nodes[key]) for key in added_keys[:64]],
                "removed_nodes": [_node_brief(before_nodes[key]) for key in removed_keys[:64]],
                "changed_nodes": changed[:64],
                "changed_functions": changed_functions[:64],
                "node_status": node_status,
            }
        )
    return diffs


def _overlay_summary(
    *,
    tensor_diff: dict[str, Any] | None,
    locate_summary: dict[str, Any] | None,
    memory_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    overlays: dict[str, Any] = {}
    if tensor_diff:
        overlays["tensor_diff"] = {
            "status": tensor_diff.get("status"),
            "comparison_count": tensor_diff.get("comparison_count"),
            "failed_count": tensor_diff.get("failed_count"),
        }
    if locate_summary:
        overlays["locate"] = {
            "status": locate_summary.get("status"),
            "first_bad_kernel": locate_summary.get("first_bad_kernel"),
            "first_bad_depth": locate_summary.get("first_bad_depth"),
            "method": locate_summary.get("method"),
        }
    if memory_summary:
        overlays["memory"] = {
            "analysis_level": memory_summary.get("analysis_level"),
            "peak_workspace_bytes": memory_summary.get("peak_workspace_bytes"),
            "total_workspace_bytes": memory_summary.get("total_workspace_bytes"),
            "slot_reuse_group_count": memory_summary.get("slot_reuse_group_count"),
            "movement_edge_count": memory_summary.get("movement_edge_count"),
        }
    return overlays


def _artifact_link(path: str, label: str | None = None) -> str:
    return f'<a href="../{_cell(path)}">{_cell(label or path)}</a>'


def _artifact_rows(items: list[dict[str, Any]], path_key: str = "path") -> str:
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        path = item.get(path_key)
        path_cell = _artifact_link(path) if isinstance(path, str) else ""
        rows.append(
            "<tr>"
            f"<td>{_cell(item.get('stage') or item.get('kind') or item.get('tool'))}</td>"
            f"<td>{path_cell}</td>"
            f"<td>{_cell(item.get('status', ''))}</td>"
            "</tr>"
        )
    return "\n".join(rows) or '<tr><td colspan="3">没有记录产物。</td></tr>'


def _summary_paths(run_dir: pathlib.Path) -> list[dict[str, Any]]:
    summary_dir = run_dir / "summaries"
    if not summary_dir.exists():
        return []
    rows = []
    for path in sorted(item for item in summary_dir.rglob("*.json") if item.is_file()):
        rel_path = path.relative_to(run_dir).as_posix()
        rows.append({"kind": "summary", "path": rel_path, "status": "present"})
    return rows


def _overlay_cards(overlays: dict[str, Any]) -> str:
    cards = []
    tensor_diff = overlays.get("tensor_diff", {})
    cards.append(
        "<section class=\"metric-card\">"
        "<h3>Tensor Diff</h3>"
        f"<dl><dt>状态</dt><dd>{_cell(tensor_diff.get('status', 'none'))}</dd>"
        f"<dt>失败</dt><dd>{_cell(tensor_diff.get('failed_count', 0))}</dd></dl>"
        "</section>"
    )
    locate = overlays.get("locate", {})
    cards.append(
        "<section class=\"metric-card\">"
        "<h3>Locate</h3>"
        f"<dl><dt>状态</dt><dd>{_cell(locate.get('status', 'none'))}</dd>"
        f"<dt>首个异常</dt><dd>{_cell(locate.get('first_bad_kernel', 'none'))}</dd></dl>"
        "</section>"
    )
    memory = overlays.get("memory", {})
    cards.append(
        "<section class=\"metric-card\">"
        "<h3>Memory</h3>"
        f"<dl><dt>峰值</dt><dd>{_cell(memory.get('peak_workspace_bytes', 0))}</dd>"
        f"<dt>复用组</dt><dd>{_cell(memory.get('slot_reuse_group_count', 0))}</dd></dl>"
        "</section>"
    )
    return "\n".join(cards)


def _stage_group_children(group: dict[str, Any]) -> list[dict[str, Any]]:
    raw_children: list[dict[str, Any]] = []
    steps = group.get("steps")
    if isinstance(steps, list) and steps:
        raw_children.extend(child for child in steps if isinstance(child, dict))
    else:
        for key in ("input_stage", "output_stage", "default_stage"):
            child = group.get(key)
            if isinstance(child, dict):
                raw_children.append(child)
    children = []
    seen: set[Any] = set()
    for child in raw_children:
        stage_index = child.get("stage_index")
        if stage_index in seen:
            continue
        seen.add(stage_index)
        children.append(child)
    return children


def _debug_artifact_href(rel_path: str) -> str:
    if rel_path.endswith(".cpp") or rel_path in {"artifact_manifest.json", "tiling_space.json"}:
        return f"artifacts/{rel_path}.html"
    if rel_path.startswith("reports/"):
        return f"reports/{rel_path.removeprefix('reports/')}.html"
    if rel_path.startswith("summaries/"):
        return f"summaries/{rel_path.removeprefix('summaries/')}.html"
    if rel_path.startswith("graphs/") and rel_path.endswith(".mlir"):
        return f"graphs/{rel_path.removeprefix('graphs/')}.html"
    return f"../{rel_path}"


def _timeline_lane_debug_items(lane: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    seen_paths: set[str] = set()
    for bucket, bucket_label in (
        ("contracts", "Contract"),
        ("artifacts", "Artifact"),
        ("diagnostics", "Diagnostic"),
    ):
        raw_items = lane.get(bucket)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            rel_path = item.get("path")
            if isinstance(rel_path, str) and rel_path:
                if rel_path in seen_paths:
                    continue
                seen_paths.add(rel_path)
            label = item.get("label") or item.get("kind") or rel_path or bucket_label
            record = {
                "bucket": bucket_label,
                "label": str(label),
                "source": item.get("source"),
                "status": item.get("status"),
            }
            if isinstance(rel_path, str) and rel_path:
                record["path"] = rel_path
                record["href"] = _debug_artifact_href(rel_path)
                record["raw_href"] = f"../{rel_path}"
            items.append(record)
    return items


def _build_timeline_followup_lanes(
    run_dir: pathlib.Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    model = timeline_model.build_timeline_model(run_dir, manifest)
    lanes = []
    for lane in model.get("lanes", []):
        if not isinstance(lane, dict):
            continue
        lane_id = lane.get("id")
        if lane_id not in ("translate", "artifacts", "runtime"):
            continue
        items = _timeline_lane_debug_items(lane)
        if not items:
            continue
        title = lane.get("title") or str(lane_id).title()
        if lane_id == "translate":
            title = "Translate Artifacts"
        lanes.append(
            {
                "id": lane_id,
                "title": title,
                "source": lane.get("source"),
                "items": items,
            }
        )
    return lanes


def _stage_artifact_lanes(debug_graph: dict[str, Any]) -> str:
    lanes = debug_graph.get("timeline_lanes")
    if not isinstance(lanes, list):
        return ""
    lane_sections = []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("id") or "")
        title = str(lane.get("title") or lane_id or "Artifacts")
        items = lane.get("items")
        if not isinstance(items, list) or not items:
            continue
        links = []
        for item in items:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            href = item.get("href")
            label = path if isinstance(path, str) and path else item.get("label")
            if isinstance(href, str) and href:
                source = item.get("source")
                title_attr = (
                    f' title="source: {_cell(source)}"'
                    if isinstance(source, str) and source
                    else ""
                )
                item_link = (
                    f'<a class="stage-button stage-child-button stage-artifact-link" '
                    f'href="{_cell(href)}" data-workbench-view-href="{_cell(href)}" '
                    f'data-workbench-view-title="{_cell(label)}"{title_attr}>'
                    f'<span class="stage-child-title">{_cell(label)}</span>'
                    "</a>"
                )
            else:
                item_link = (
                    '<span class="stage-button stage-child-button stage-artifact-link">'
                    f'<span class="stage-child-title">{_cell(label)}</span>'
                    "</span>"
                )
            links.append(item_link)
        lane_sections.append(
            f'<div class="stage-tree-group stage-artifact-lane" data-lane-id="{_cell(lane_id)}">'
            '<div class="stage-button stage-group-parent stage-artifact-parent">'
            f'<span class="stage-group-main">{_cell(title)}</span>'
            f'<small class="stage-group-meta">{len(links)} 项</small>'
            "</div>"
            f'<div class="stage-child-list">{"".join(links)}</div>'
            "</div>"
        )
    if not lane_sections:
        return ""
    return f'<div class="stage-followup-lanes">{"".join(lane_sections)}</div>'


def _stage_step_id_label(stage: dict[str, Any]) -> str:
    step_id = stage.get("step_id")
    if isinstance(step_id, str) and step_id:
        return step_id
    step = stage.get("step")
    if isinstance(step, str) and step:
        return step
    name = stage.get("name")
    return str(name or "stage")


def _stage_buttons(debug_graph: dict[str, Any]) -> str:
    primary = debug_graph.get("primary_stage")
    active_name = primary.get("name") if isinstance(primary, dict) else None
    groups = debug_graph.get("stage_groups", [])
    buttons = []
    if isinstance(groups, list) and groups:
        for group_index, group in enumerate(groups):
            if not isinstance(group, dict):
                continue
            default_stage = group.get("default_stage")
            if not isinstance(default_stage, dict):
                continue
            children = _stage_group_children(group)
            if not children:
                continue
            first_child = children[0]
            child_indices = ",".join(str(child.get("stage_index")) for child in children)
            parent_active = " parent-active" if any(child.get("name") == active_name for child in children) else ""
            steps = group.get("steps") if isinstance(group.get("steps"), list) else []
            meta = f"{len(steps)} 个步骤 dump" if steps else f"{len(children)} 个边界 dump"
            previous = group.get("previous_output_stage")
            equivalence = ""
            if group.get("input_same_as_previous_output") and isinstance(previous, dict):
                equivalence = (
                    '<small class="stage-equivalence">'
                    f'输入同 {_cell(previous.get("order"))} {_cell(previous.get("name"))}'
                    "</small>"
                )
            child_buttons = []
            for child in children:
                child_active = " active" if child.get("name") == active_name else ""
                child_title = _stage_step_id_label(child)
                child_buttons.append(
                    f'<button class="stage-button stage-child-button{child_active}" '
                    f'data-stage-index="{_cell(child.get("stage_index"))}" type="button">'
                    f'<span class="stage-child-title">{_cell(child_title)}</span>'
                    "</button>"
                )
            buttons.append(
                f'<div class="stage-tree-group" data-stage-group-index="{group_index}">'
                f'<button class="stage-button stage-group-parent{parent_active}" '
                f'data-stage-index="{_cell(first_child.get("stage_index"))}" '
                f'data-stage-child-indices="{_cell(child_indices)}" type="button">'
                f'<span class="stage-group-main">{_cell(group.get("label"))}</span>'
                f'<small class="stage-group-meta">{_cell(meta)}</small>'
                f"{equivalence}"
                "</button>"
                f'<div class="stage-child-list">{"".join(child_buttons)}</div>'
                "</div>"
            )
        followups = _stage_artifact_lanes(debug_graph)
        if followups:
            buttons.append(followups)
        return "\n".join(buttons)
    for index, item in enumerate(debug_graph.get("stages", [])):
        active = " active" if item.get("name") == active_name else ""
        buttons.append(
            f'<button class="stage-button stage-child-button{active}" data-stage-index="{index}" type="button">'
            f'{_cell(item.get("name"))}'
            "</button>"
        )
    followups = _stage_artifact_lanes(debug_graph)
    if followups:
        buttons.append(followups)
    return "\n".join(buttons)


def _artifact_index(debug_graph: dict[str, Any]) -> str:
    artifacts = debug_graph.get("artifacts", {})
    return f"""
<section id="artifact-index" class="artifact-index">
<h2>原始产物索引</h2>
<div class="artifact-grid">
<section>
<h3>执行命令</h3>
<table><tbody>{_artifact_rows(artifacts.get('commands', []), path_key='stdout')}</tbody></table>
</section>
<section>
<h3>报告</h3>
<table><tbody>{_artifact_rows(artifacts.get('reports', []))}</tbody></table>
</section>
<section>
<h3>图数据</h3>
<table><tbody>{_artifact_rows(artifacts.get('graphs', []))}</tbody></table>
</section>
<section>
<h3>Kernel / Runtime</h3>
<table><tbody>{_artifact_rows(artifacts.get('runtime', []))}</tbody></table>
</section>
<section>
<h3>摘要</h3>
<table><tbody>{_artifact_rows(artifacts.get('summaries', []))}</tbody></table>
</section>
</div>
</section>
"""


def _render_html(
    *,
    run_dir: pathlib.Path,
    debug_graph: dict[str, Any],
    view_rel_path: str,
) -> None:
    primary_stage = debug_graph.get("primary_stage")
    overlays = debug_graph.get("overlays", {})
    kernel_dag = debug_graph.get("kernel_dag", {})
    summary_path = debug_graph.get("summary_path", "summaries/debug_graph.json")
    workspace_json = layout.json_script_payload(debug_graph)
    style = """
<style>
:root {
  color-scheme: light;
  --bg: #f5f7fa;
  --panel: #ffffff;
  --line: #d7dde7;
  --text: #17202a;
  --muted: #667085;
  --blue: #2563eb;
  --teal: #0f766e;
  --red: #dc2626;
  --amber: #d97706;
  --sidebar-width: 15rem;
  --inspector-width: 25rem;
  --resizer-width: 0.55rem;
  --header-height: 4.5rem;
}
* { box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); }
a { color: var(--blue); text-decoration: none; }
header { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; min-height: var(--header-height); padding: 0.9rem 1.1rem; background: var(--panel); border-bottom: 1px solid var(--line); }
h1 { margin: 0 0 0.35rem; font-size: 1.2rem; }
h2 { margin: 0 0 0.65rem; font-size: 0.98rem; }
h3 { margin: 0 0 0.45rem; font-size: 0.84rem; }
.toolbar { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; color: var(--muted); font-size: 0.86rem; }
.app-shell { display: grid; grid-template-columns: var(--sidebar-width) minmax(28rem, 1fr) var(--resizer-width) var(--inspector-width); gap: 0.55rem; padding: 0.85rem; height: calc(100vh - var(--header-height)); min-height: 0; overflow: hidden; }
.app-shell.sidebar-collapsed { --sidebar-width: 3.4rem; }
.app-shell.document-mode { grid-template-columns: var(--sidebar-width) minmax(28rem, 1fr); }
.app-shell.document-mode .graph-panel { grid-column: 2; }
.app-shell.document-mode .layout-resizer { display: none; }
.app-shell.document-mode .inspector-panel { display: none; }
.sidebar, .graph-panel, .inspector-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
.sidebar { padding: 0.65rem; min-height: 0; overflow: hidden; display: grid; grid-template-rows: auto auto minmax(0, 1fr); }
.sidebar-header { display: flex; align-items: center; justify-content: space-between; gap: 0.45rem; margin-bottom: 0.65rem; }
.sidebar-header h2 { margin: 0; }
.sidebar-toggle { border: 1px solid var(--line); border-radius: 6px; background: #ffffff; color: var(--text); padding: 0.32rem 0.45rem; cursor: pointer; font-size: 0.78rem; }
.sidebar-toggle:hover { border-color: #60a5fa; color: #1d4ed8; }
.mode-tabs { display: grid; grid-template-columns: 1fr; gap: 0.35rem; margin-bottom: 0.9rem; }
.mode-tab, .stage-button { border: 1px solid var(--line); border-radius: 6px; background: #ffffff; color: var(--text); padding: 0.45rem 0.55rem; text-align: left; cursor: pointer; }
.mode-tab.active, .stage-button.active { border-color: #60a5fa; background: #eaf2ff; color: #1d4ed8; font-weight: 700; }
.sidebar-collapsed .sidebar { padding: 0.45rem; }
.sidebar-collapsed .sidebar-header h2,
.sidebar-collapsed .sidebar-body { display: none; }
.sidebar-collapsed .sidebar-header { justify-content: center; margin-bottom: 0.45rem; }
.sidebar-collapsed .mode-tabs { margin: 0; }
.sidebar-collapsed .mode-tab { font-size: 0; text-align: center; padding: 0.5rem 0; }
.sidebar-collapsed .mode-tab::before { content: attr(data-short); font-size: 0.82rem; font-weight: 700; }
.sidebar-body.kernel-mode .stage-navigation,
.sidebar-body.kernel-mode #stage-diff-panel,
.sidebar-body.kernel-mode #graph-audit-panel { display: none; }
.sidebar-body { min-height: 0; overflow-y: auto; padding-right: 0.12rem; }
.stage-list { display: grid; gap: 0.45rem; overflow: visible; padding-right: 0.15rem; }
.stage-navigation h2 { margin-bottom: 0.55rem; }
.stage-tree-group { display: grid; gap: 0.22rem; }
.stage-button span { display: inline-block; min-width: 2.2rem; color: var(--muted); font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.stage-group-parent { display: grid; gap: 0.12rem; font-size: 0.86rem; font-weight: 700; }
.stage-group-parent.parent-active { border-color: #93c5fd; background: #eff6ff; color: #1d4ed8; }
.stage-group-main { display: block; color: inherit; font-family: inherit; }
.stage-group-meta, .stage-equivalence { color: var(--muted); font-size: 0.72rem; font-weight: 500; }
.stage-equivalence { color: #0f766e; }
.stage-child-list { display: grid; gap: 0.24rem; }
.stage-child-button { margin-left: 0.65rem; padding: 0.34rem 0.45rem; font-size: 0.75rem; }
.stage-child-title { display: block; color: inherit; font-weight: 700; line-height: 1.15; }
.stage-followup-lanes { display: grid; gap: 0.45rem; margin-top: 0.45rem; }
.stage-artifact-parent { cursor: default; }
.stage-artifact-link { display: block; text-decoration: none; overflow-wrap: anywhere; }
.stage-phase-controls { display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }
.stage-phase-controls:empty { display: none; }
.stage-phase-button { border: 1px solid var(--line); border-radius: 999px; background: #fff; color: var(--text); padding: 0.24rem 0.55rem; cursor: pointer; font-size: 0.76rem; }
.stage-phase-button.active { border-color: #60a5fa; background: #eaf2ff; color: #1d4ed8; font-weight: 700; }
.stage-phase-note { color: var(--muted); font-size: 0.76rem; }
.stage-graph-controls { order: 1; flex: 1 1 100%; display: flex; flex-wrap: wrap; gap: 0.38rem 0.45rem; align-items: center; justify-content: flex-end; min-width: 0; }
.stage-graph-controls[hidden] { display: none; }
.segmented-control { display: inline-flex; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; background: #ffffff; }
.segmented-control button { border: 0; border-right: 1px solid var(--line); background: transparent; color: var(--text); padding: 0.32rem 0.5rem; cursor: pointer; font-size: 0.76rem; }
.segmented-control button:last-child { border-right: 0; }
.segmented-control button.active { background: #eaf2ff; color: #1d4ed8; font-weight: 700; }
.control-label, .edge-filter-controls label { display: inline-flex; align-items: center; gap: 0.28rem; color: #344054; font-size: 0.76rem; }
.control-select { border: 1px solid var(--line); border-radius: 6px; background: #ffffff; color: var(--text); padding: 0.28rem 0.42rem; font: inherit; }
.edge-filter-controls { display: inline-flex; flex-wrap: wrap; gap: 0.42rem; align-items: center; }
.graph-tool-button.active { border-color: #60a5fa; background: #eaf2ff; color: #1d4ed8; font-weight: 700; }
.graph-tool-button[hidden] { display: none; }
.stage-diff-panel, .graph-audit-panel { margin-top: 0.9rem; border: 1px solid #e3e8ef; border-radius: 7px; padding: 0.65rem; background: #fbfcfe; }
.diff-counts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.35rem; margin: 0.45rem 0; }
.diff-pill { border-radius: 6px; border: 1px solid var(--line); padding: 0.35rem; background: #fff; font-size: 0.78rem; }
.diff-pill strong { display: block; font-size: 0.98rem; }
.diff-added-text { color: #047857; }
.diff-changed-text { color: #b45309; }
.diff-removed-text { color: #b91c1c; }
.diff-list { margin: 0.4rem 0 0; padding-left: 1rem; color: #344054; font-size: 0.78rem; max-height: 7.5rem; overflow: auto; }
.audit-summary-strip { display: grid; gap: 0.12rem; margin-top: 0.45rem; border: 1px solid #dbe3ee; border-radius: 6px; padding: 0.45rem 0.5rem; background: #ffffff; color: #334155; }
.audit-summary-strip strong { font-size: 0.82rem; color: #17202a; }
.audit-summary-strip span { color: var(--muted); font-size: 0.74rem; line-height: 1.25; }
.audit-summary-strip.warn { border-color: #fde68a; background: #fffbeb; }
.audit-summary-strip.ok { border-color: #bbf7d0; background: #f0fdf4; }
.audit-metrics { display: grid; grid-template-columns: 1fr; gap: 0.25rem; margin-top: 0.45rem; }
.audit-metric { display: grid; grid-template-columns: 1.8rem minmax(0, 1fr); align-items: center; gap: 0.4rem; border: 1px solid #e3e8ef; border-radius: 6px; background: #ffffff; padding: 0.28rem 0.42rem; }
.audit-metric-value { font-weight: 800; font-size: 0.9rem; line-height: 1; text-align: right; }
.audit-metric-label { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #475569; font-family: SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.7rem; }
.audit-detail-group { margin-top: 0.55rem; padding-top: 0.5rem; border-top: 1px solid #e3e8ef; }
.audit-detail-heading { display: flex; align-items: center; justify-content: space-between; gap: 0.4rem; margin-bottom: 0.28rem; color: #344054; font-size: 0.74rem; font-weight: 800; }
.audit-detail-list { display: grid; gap: 0.26rem; margin: 0; padding: 0; list-style: none; font-size: 0.74rem; }
.audit-detail-row { display: grid; grid-template-columns: minmax(0, 1fr); gap: 0.05rem; color: #344054; }
.audit-kind { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.68rem; }
.audit-node { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #17202a; }
.audit-meta { margin-top: 0.5rem; color: var(--muted); font-size: 0.72rem; line-height: 1.3; }
.overlay-stack { display: grid; gap: 0.55rem; margin-top: 0.9rem; }
.metric-card { border: 1px solid #e3e8ef; border-radius: 7px; padding: 0.6rem; background: #fbfcfe; }
.metric-card.fail { border-color: #fecaca; background: #fff7f7; }
.metric-card.warn { border-color: #fde68a; background: #fffbeb; }
dl { margin: 0; display: grid; grid-template-columns: 6.2rem 1fr; gap: 0.24rem 0.45rem; }
dt { color: var(--muted); font-weight: 700; }
dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.graph-panel { min-width: 0; min-height: 0; overflow: hidden; display: grid; grid-template-rows: auto minmax(0, 1fr); align-self: stretch; }
.panel-header { display: grid; grid-template-columns: minmax(0, 1fr) minmax(17rem, 24rem); gap: 0.6rem 1rem; align-items: start; padding: 0.72rem 0.85rem 0.65rem; border-bottom: 1px solid #e4e9f1; }
.panel-title-block { min-width: 0; display: grid; gap: 0.34rem; align-content: start; }
.panel-title-block h2 { margin: 0; line-height: 1.25; }
.panel-subtitle { color: var(--muted); font-size: 0.82rem; line-height: 1.25; }
.step-explanation { max-width: 54rem; border: 1px solid #e3e8ef; border-radius: 7px; background: #fbfcfe; padding: 0.5rem 0.6rem; font-size: 0.77rem; line-height: 1.28; color: #344054; }
.step-explanation[hidden] { display: none; }
.step-explanation-title { margin-bottom: 0.32rem; color: #17202a; font-weight: 700; }
.step-explanation-grid { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 0.22rem 0.5rem; align-items: start; }
.step-explanation-grid dt { color: var(--muted); font-weight: 700; white-space: nowrap; }
.step-explanation-grid dd { margin: 0; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.step-explanation-same { margin-top: 0.35rem; color: #0f766e; font-size: 0.74rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.panel-tools-column { align-self: stretch; justify-self: end; width: 100%; display: grid; grid-template-rows: auto minmax(0, 1fr); gap: 0.46rem; }
.panel-control-strip { min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); gap: 0.38rem; justify-items: end; align-items: start; min-width: 0; }
.panel-control-strip .stage-phase-controls { order: 2; align-self: end; margin-top: 0.1rem; justify-content: flex-end; }
.graph-tools { width: 100%; display: grid; grid-template-columns: minmax(0, 1fr) auto auto auto; gap: 0.42rem; align-items: center; }
.graph-search { border: 1px solid var(--line); border-radius: 6px; padding: 0.42rem 0.55rem; min-width: 0; font: inherit; }
.graph-tool-button { border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 0.42rem 0.6rem; cursor: pointer; color: var(--text); }
.graph-tool-button:hover { border-color: #60a5fa; color: #1d4ed8; }
.zoom-value, .search-status { color: var(--muted); font-size: 0.78rem; white-space: nowrap; }
.search-status { grid-column: 1 / -1; justify-self: end; }
.search-status:empty { display: none; }
.graph-canvas-wrap { overflow: auto; min-height: 0; height: auto; background: #ffffff; cursor: grab; }
.graph-canvas-wrap.panning { cursor: grabbing; user-select: none; }
.graph-canvas-wrap.document-mode { padding: 0; overflow: hidden; cursor: default; }
.workbench-view-frame { width: 100%; height: 100%; min-height: 70vh; border: 0; background: #ffffff; display: block; }
#unified-debug-graph-svg { display: block; min-width: 100%; }
.function-frame-box { fill: #f8fafc; fill-opacity: 0.72; stroke: #475569; stroke-width: 2; stroke-dasharray: 9 5; }
.function-frame-label-bg { fill: #ffffff; stroke: #94a3b8; stroke-width: 1; }
.function-frame-label { fill: #1f2937; font-size: 12px; font-weight: 700; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.function-frame-meta { fill: #475569; font-size: 11px; }
.graph-edge-path { fill: none; stroke: #8a95a6; stroke-width: 1.5; marker-end: url(#arrow-head); }
.graph-edge-path.value { stroke: #8a95a6; }
.graph-edge-path.memory-effect { stroke: #0f766e; stroke-dasharray: 6 4; }
.graph-edge-path.resource-effect { stroke: #7c3aed; stroke-dasharray: 6 4; }
.graph-edge-path.control { stroke: #ea580c; stroke-dasharray: 2 4; }
.graph-edge-path.region { stroke: #60a5fa; stroke-dasharray: 2 6; opacity: 0.72; }
.graph-edge-path.symbol { stroke: #475569; stroke-dasharray: 4 4; }
.graph-edge-path.critical { stroke: var(--red); stroke-width: 2.7; }
.graph-edge-path.fusion { stroke: #16a34a; stroke-width: 2.5; }
.graph-edge-label { fill: #586274; font-size: 11px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.graph-edge.dimmed { opacity: 0.12; }
.graph-edge.focus-hidden, .graph-edge.edge-filter-hidden, .graph-edge.helper-collapsed { display: none; }
.graph-edge.neighborhood-edge .graph-edge-path { stroke-width: 2.8; }
.graph-edge.neighborhood-edge .graph-edge-label { font-weight: 700; fill: #334155; }
.graph-node, .kernel-dag-node { cursor: pointer; outline: none; }
.graph-node rect, .kernel-dag-node rect { fill: #ffffff; stroke: #95a1b2; stroke-width: 1.4; }
.graph-node.kernel-node rect { fill: #eef6ff; stroke: var(--blue); }
.graph-node.stage-kernel-aggregate rect { fill: #eef6ff; stroke: #2563eb; stroke-width: 2; }
.graph-node.stage-kernel-boundary rect { fill: #f8fafc; stroke: #64748b; stroke-dasharray: 6 4; }
.kernel-dag-node.vec rect { fill: #ecfdf5; stroke: #10b981; }
.kernel-dag-node.cube rect { fill: #fff7ed; stroke: #f97316; }
.kernel-dag-node.mix rect { fill: #f5f3ff; stroke: #7c3aed; }
.graph-node.dimmed { opacity: 0.22; }
.graph-node.focus-hidden, .graph-node.helper-collapsed { display: none; }
.graph-node.neighborhood-node rect { stroke: #2563eb; stroke-width: 2.6; }
.graph-node.parent-node rect { fill: #eff6ff; }
.graph-node.child-node rect { fill: #ecfdf5; }
.graph-node.selected.neighborhood-node rect { stroke: var(--teal); stroke-width: 3; }
.graph-node.issue rect, .kernel-dag-node.issue rect { stroke: var(--red); stroke-width: 3; }
.graph-node.diff-added rect { fill: #ecfdf5; stroke: #059669; stroke-width: 2.3; }
.graph-node.diff-changed rect { fill: #fffbeb; stroke: #d97706; stroke-width: 2.3; }
.graph-node.search-match rect, .kernel-dag-node.search-match rect { filter: drop-shadow(0 0 0.35rem rgba(37, 99, 235, 0.45)); stroke: #2563eb; stroke-width: 3; }
.graph-node:hover rect, .graph-node.selected rect, .kernel-dag-node:hover rect, .kernel-dag-node.selected rect { stroke: var(--teal); stroke-width: 2.6; }
.node-op { fill: var(--text); font-size: 13px; font-weight: 700; }
.node-result { fill: #344054; font-size: 12px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.node-shape { fill: #0f766e; font-size: 11px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.node-inputs { fill: #667085; font-size: 11px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.node-kernel { fill: var(--blue); font-size: 10px; text-anchor: end; }
.empty-state-title { fill: #17202a; font-size: 15px; font-weight: 700; }
.empty-state-subtitle { fill: #64748b; font-size: 12px; }
.node-badge-bg { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1; }
.node-badge { fill: #344054; font-size: 10px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.semantic-group { border: 1px solid #e3e8ef; border-radius: 6px; background: #ffffff; padding: 0.55rem; margin-top: 0.55rem; }
.semantic-group h4 { margin: 0 0 0.42rem; font-size: 0.76rem; color: #344054; }
.semantic-group .detail-grid { gap: 0.34rem; }
.semantic-note { margin: 0.1rem 0 0.45rem; color: var(--muted); font-size: 0.72rem; line-height: 1.3; }
.tile-param-list { display: grid; gap: 0.45rem; }
.tile-param-card { border: 1px solid #dbe3ee; border-radius: 6px; background: #fbfcfe; padding: 0.45rem 0.5rem; }
.tile-param-card h5 { margin: 0 0 0.34rem; font-size: 0.76rem; color: #17202a; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.tile-param-grid { display: grid; grid-template-columns: 1fr; gap: 0.24rem; font-size: 0.76rem; }
.tile-param-field { min-width: 0; display: grid; grid-template-columns: 5.6rem minmax(0, 1fr); gap: 0.35rem; align-items: baseline; }
.tile-param-field span { color: var(--muted); font-weight: 700; font-size: 0.68rem; white-space: nowrap; }
.tile-param-field code { min-width: 0; overflow-wrap: anywhere; color: #17202a; }
.tile-chip-list { display: flex; flex-wrap: wrap; gap: 0.26rem; }
.tile-chip { display: inline-flex; align-items: center; max-width: 100%; border: 1px solid #dbe3ee; border-radius: 999px; padding: 0.13rem 0.4rem; background: #f8fafc; color: #344054; font-size: 0.72rem; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.symbol-constraint-list { display: grid; gap: 0.42rem; }
.symbol-constraint-card { border: 1px solid #dbe3ee; border-radius: 6px; background: #fbfcfe; padding: 0.45rem 0.5rem; }
.symbol-constraint-card h5 { margin: 0 0 0.34rem; font-size: 0.76rem; color: #17202a; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.symbol-member-list { display: flex; flex-wrap: wrap; gap: 0.26rem; }
.symbol-member { display: inline-flex; border: 1px solid #dbe3ee; border-radius: 999px; padding: 0.13rem 0.4rem; background: #ffffff; color: #344054; font-size: 0.7rem; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.phase-legend { display: grid; gap: 0.28rem; font-size: 0.74rem; line-height: 1.32; }
.phase-legend-row { display: grid; grid-template-columns: minmax(7.4rem, max-content) minmax(0, 1fr); gap: 0.35rem; align-items: start; }
.phase-legend-row code { color: #17202a; }
.phase-legend-row span { color: #475569; }
.provenance-grid { display: grid; gap: 0.55rem; }
.provenance-block { border: 1px solid #e3e8ef; border-radius: 6px; background: #ffffff; padding: 0.55rem; }
.provenance-block h4 { margin: 0 0 0.42rem; font-size: 0.76rem; color: #344054; }
.provenance-edge-list { display: flex; flex-wrap: wrap; gap: 0.32rem; }
.provenance-edge-chip { display: inline-flex; gap: 0.28rem; align-items: baseline; max-width: 100%; border: 1px solid #dbe3ee; border-radius: 999px; padding: 0.16rem 0.42rem; background: #f8fafc; color: #344054; font-size: 0.74rem; }
.provenance-edge-chip strong { color: #17202a; }
.provenance-edge-chip code { overflow-wrap: anywhere; white-space: normal; }
.layout-resizer { cursor: col-resize; align-self: stretch; border-radius: 999px; background: linear-gradient(90deg, transparent, #cbd5e1, transparent); min-height: 0; }
.layout-resizer:hover, .layout-resizer.active { background: #93c5fd; }
.inspector-panel { min-width: 0; padding: 0.8rem; overflow-y: auto; min-height: 0; }
.inspector-actions { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-bottom: 0.6rem; }
.chip { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 0.18rem 0.45rem; color: #344054; background: #ffffff; font-size: 0.78rem; }
.inspector-section { border: 1px solid #e3e8ef; border-radius: 7px; background: #fbfcfe; padding: 0.65rem; margin-bottom: 0.65rem; }
.inspector-section h3 { margin-bottom: 0.55rem; }
.source-section-header { display: flex; justify-content: space-between; align-items: center; gap: 0.55rem; margin-bottom: 0.55rem; }
.source-section-header h3 { margin: 0; }
.source-expand-button { border: 1px solid var(--line); border-radius: 6px; background: #ffffff; color: var(--text); padding: 0.28rem 0.45rem; cursor: pointer; font-size: 0.76rem; }
.source-expand-button:hover { border-color: #60a5fa; color: #1d4ed8; }
.detail-grid { display: grid; grid-template-columns: 1fr; gap: 0.38rem; font-size: 0.82rem; }
.detail-row { border: 1px solid #e3e8ef; border-radius: 6px; background: #ffffff; padding: 0.34rem 0.42rem; min-width: 0; }
.detail-label { color: #475569; font-weight: 800; font-size: 0.72rem; line-height: 1.2; min-width: 0; overflow-wrap: anywhere; margin-bottom: 0.12rem; }
.detail-value { color: #111827; min-width: 0; white-space: nowrap; overflow-x: auto; overflow-y: hidden; overflow-wrap: normal; font-family: SFMono-Regular, Menlo, Consolas, monospace; padding-bottom: 0.04rem; }
.body-op-list { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.45rem; }
pre { margin: 0; background: #0b1020; color: #dbeafe; border: 1px solid #1e293b; padding: 0.75rem; border-radius: 7px; font: 12px/1.45 SFMono-Regular, Menlo, Consolas, monospace; }
.source-code { margin: 0; white-space: pre; overflow: auto; overflow-wrap: normal; max-height: 48vh; tab-size: 2; }
.source-code code { display: block; min-width: max-content; }
.mlir-comment { color: #94a3b8; font-style: italic; }
.mlir-symbol { color: #93c5fd; }
.mlir-ssa { color: #fcd34d; }
.mlir-op { color: #c4b5fd; font-weight: 700; }
.mlir-type { color: #6ee7b7; }
.mlir-attr { color: #fca5a5; }
.source-reader-overlay[hidden] { display: none; }
.source-reader-overlay { position: fixed; inset: 0; z-index: 20; background: rgba(15, 23, 42, 0.62); display: grid; place-items: center; padding: 1.25rem; }
.source-reader-panel { width: min(94vw, 84rem); height: min(88vh, 58rem); display: grid; grid-template-rows: auto minmax(0, 1fr); background: var(--panel); border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 24px 70px rgba(15, 23, 42, 0.35); overflow: hidden; }
.source-reader-header { display: flex; justify-content: space-between; align-items: center; gap: 1rem; padding: 0.8rem 1rem; border-bottom: 1px solid var(--line); }
.source-reader-header h2 { margin: 0; }
.source-reader-close { border: 1px solid var(--line); border-radius: 6px; background: #ffffff; color: var(--text); padding: 0.38rem 0.62rem; cursor: pointer; }
.source-reader-code { border: 0; border-radius: 0; max-height: none; height: 100%; }
.artifact-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 0.75rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th, td { border: 1px solid #d7dde7; padding: 0.34rem 0.42rem; text-align: left; vertical-align: top; }
th { background: #f2f5f9; }
@media (max-width: 1180px) {
  html, body { overflow: auto; }
  .app-shell { grid-template-columns: 1fr; }
  .app-shell.document-mode { grid-template-columns: var(--sidebar-width) minmax(0, 1fr); }
  .app-shell.document-mode .graph-panel { grid-column: 2; }
  .sidebar, .inspector-panel { position: static; max-height: none; }
  .layout-resizer { display: none; }
  .panel-header { grid-template-columns: 1fr; }
  .panel-tools-column { justify-self: stretch; }
  .panel-control-strip, .stage-graph-controls, .panel-control-strip .stage-phase-controls { justify-content: flex-start; }
  .graph-canvas-wrap { height: 62vh; }
  .graph-tools { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 1500px) {
  .step-explanation-grid { grid-template-columns: max-content minmax(0, 1fr); }
}
</style>
"""
    script = """
<script>
const workspace = JSON.parse(document.getElementById("graph-workspace-data").textContent);
let activeMode = "stage";
const initialParams = new URLSearchParams(window.location.search);
const requestedStage = initialParams.get("stage");
const requestedNode = initialParams.get("node");
let requestedNodeConsumed = false;

function defaultStageIndex() {
  if (requestedStage) {
    const requested = String(requestedStage);
    const index = workspace.stages.findIndex((stage) =>
      String(stage.order) === requested || stage.name === requested || stage.path === requested);
    if (index >= 0) return index;
  }
  return Math.max(0, workspace.stages.findIndex((stage) => workspace.primary_stage && stage.name === workspace.primary_stage.name));
}

let activeStageIndex = defaultStageIndex();
let selectedKey = null;
let pendingStageSelection = null;
let canvasPanState = null;
let graphViewState = {scale: 1};
let activeSearchResults = [];
let activeSearchIndex = 0;
let stageNeighborhoodActive = Boolean(requestedNode) && !isSourceStage(workspace.stages[activeStageIndex]);
let currentStageDisplayGraph = null;
const MIN_GRAPH_SCALE = 0.2;
const MAX_GRAPH_SCALE = 3;
const GRAPH_CANVAS_PADDING = 160;
const GRAPH_DEFAULT_MARGIN = 32;
const NODE_OUTPUT_WRAP_LIMIT = 28;
const MIN_INSPECTOR_WIDTH = 320;
const MAX_INSPECTOR_WIDTH = 920;
let inspectorResizeState = null;
const ALL_EDGE_KINDS = ["value", "memory_effect", "resource_effect", "control", "region", "symbol"];
const VALID_HIGHLIGHT_MODES = new Set(["direct", "upstream", "downstream", "both"]);
const VALID_DEPTHS = new Set(["1", "2", "3", "all"]);
const VALID_STAGE_GRAPH_VIEWS = new Set(["ops", "kernel-aggregate", "kernel-local"]);
const HELPER_NODE_OPS = new Set([
  "affine.apply",
  "arith.addi",
  "arith.constant",
  "arith.index_cast",
  "arith.muli",
  "emitasc.member",
  "emitasc.reinterpret_cast",
  "memref.dim",
  "tensor.dim",
]);

function defaultStageGraphViewState() {
  return {
    highlightMode: "both",
    depth: "all",
    focusView: false,
    edgeKinds: new Set(ALL_EDGE_KINDS),
    foldHelpers: false,
    graphView: "ops",
    graphViewExplicit: false,
    kernelId: initialParams.get("kernel") || null,
  };
}

function parseStageGraphViewState() {
  const state = defaultStageGraphViewState();
  const focusParam = initialParams.get("focus");
  const depthParam = initialParams.get("depth");
  const edgeParam = initialParams.get("edges");
  const graphViewParam = initialParams.get("graph_view");
  state.highlightMode = VALID_HIGHLIGHT_MODES.has(focusParam) ? focusParam : state.highlightMode;
  state.depth = VALID_DEPTHS.has(depthParam) ? depthParam : state.depth;
  if (VALID_STAGE_GRAPH_VIEWS.has(graphViewParam)) {
    state.graphView = graphViewParam;
    state.graphViewExplicit = true;
  }
  state.focusView = initialParams.get("view") === "focus";
  if (edgeParam) {
    const edgeKinds = new Set(edgeParam.split(",").filter((kind) => ALL_EDGE_KINDS.includes(kind)));
    state.edgeKinds = edgeKinds;
  }
  if (!state.edgeKinds.size) {
    state.edgeKinds = new Set();
    for (const kind of ALL_EDGE_KINDS) state.edgeKinds.add(kind);
  }
  state.foldHelpers = initialParams.get("fold") === "helpers";
  return state;
}

let stageGraphViewState = parseStageGraphViewState();

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function truncate(value, limit) {
  const text = String(value ?? "");
  return text.length <= limit ? text : `${text.slice(0, Math.max(0, limit - 1))}...`;
}

function setInspector(title, links = [], detailHtml = "") {
  document.getElementById("inspector-title").textContent = title;
  const actions = document.getElementById("inspector-actions");
  actions.innerHTML = links.map((link) => {
    const viewAttrs = link.workbench === false
      ? ""
      : ` data-workbench-view-href="${escapeHtml(link.href)}" data-workbench-view-title="${escapeHtml(link.label)}"`;
    const targetAttrs = link.external ? ' target="_blank" rel="noopener noreferrer"' : "";
    return `<a class="chip" href="${escapeHtml(link.href)}"${viewAttrs}${targetAttrs}>${escapeHtml(link.label)}</a>`;
  }).join("");
  document.getElementById("inspector-detail").innerHTML = detailHtml;
  installWorkbenchViewLinks(actions);
  installSourceExpandActions();
}

function enterGraphMode() {
  document.querySelector(".app-shell")?.classList.remove("document-mode");
  const canvas = document.getElementById("graph-canvas");
  canvas.classList.remove("document-mode");
}

function openWorkbenchView(href, title = "文档视图") {
  if (!href) return;
  selectedKey = `document:${href}`;
  document.querySelector(".app-shell")?.classList.add("document-mode");
  const canvas = document.getElementById("graph-canvas");
  canvas.classList.add("document-mode");
  document.getElementById("stage-graph-controls").hidden = true;
  document.getElementById("graph-title").textContent = title || "文档视图";
  document.getElementById("graph-subtitle").textContent = href;
  updateStepExplanation(null);
  canvas.innerHTML = `<iframe class="workbench-view-frame" src="${escapeHtml(href)}" title="${escapeHtml(title || href)}"></iframe>`;
  setInspector(
    `文档视图：${title || href}`,
    [{label: "独立打开", href, workbench: false, external: true}],
    detailRows([
      ["路径", href],
      ["显示方式", "工作台内嵌视图"],
    ])
  );
}

function installWorkbenchViewLinks(root = document) {
  root.querySelectorAll("[data-workbench-view-href]").forEach((link) => {
    if (link.dataset.workbenchViewInstalled === "true") return;
    link.dataset.workbenchViewInstalled = "true";
    link.addEventListener("click", (event) => {
      event.preventDefault();
      const href = link.dataset.workbenchViewHref || link.getAttribute("href") || "";
      const title = link.dataset.workbenchViewTitle || link.textContent || href;
      openWorkbenchView(href, title);
    });
  });
}

function valueText(value) {
  if (Array.isArray(value)) return value.length ? value.join(", ") : "无";
  return value == null || value === "" ? "无" : String(value);
}

function workspaceRelativeHref(runRelativePath) {
  if (!runRelativePath) return null;
  return runRelativePath.startsWith("views/")
    ? runRelativePath.slice("views/".length)
    : `../${runRelativePath}`;
}

function kernelDetailHref(kernelId) {
  const views = workspace.kernel_detail_views || {};
  return workspaceRelativeHref(views[kernelId]);
}

function resolveKernelDagEntry(kernelId) {
  const nodes = workspace.kernel_dag && workspace.kernel_dag.nodes ? workspace.kernel_dag.nodes : {};
  if (nodes[kernelId]) return {id: kernelId, node: nodes[kernelId]};
  const views = workspace.kernel_detail_views || {};
  const targetView = views[kernelId];
  if (targetView) {
    for (const [candidateId, candidate] of Object.entries(nodes)) {
      const candidateView = views[candidateId] || `views/kernels/${candidateId}.html`;
      if (candidateView === targetView) return {id: candidateId, node: candidate};
    }
  }
  const ids = Object.keys(nodes);
  if (ids.length === 1) {
    const onlyId = ids[0];
    return {id: onlyId, node: nodes[onlyId]};
  }
  return null;
}

function detailRows(rows) {
  return `<div class="detail-grid">${rows.map(([label, value]) => `
<div class="detail-row">
<div class="detail-label">${escapeHtml(label)}</div>
<div class="detail-value">${escapeHtml(valueText(value))}</div>
</div>`).join("")}</div>`;
}

function classifyMlirToken(token) {
  if (/^\\/\\//.test(token)) return "mlir-comment";
  if (/^%/.test(token)) return "mlir-ssa";
  if (/^[@#]/.test(token)) return "mlir-symbol";
  if (/^(func\\.func|linalg\\.[A-Za-z0-9_.$-]+|arith\\.[A-Za-z0-9_.$-]+|tensor\\.[A-Za-z0-9_.$-]+|memref\\.[A-Za-z0-9_.$-]+|scf\\.[A-Za-z0-9_.$-]+|ascendc\\.[A-Za-z0-9_.$-]+|emitasc\\.[A-Za-z0-9_.$-]+)$/.test(token)) return "mlir-op";
  if (/^(tensor|memref|index|i1|i8|i16|i32|i64|ui8|f16|f32|f64|bf16)$/.test(token)) return "mlir-type";
  if (/^[A-Za-z_][A-Za-z0-9_.$-]*:$/.test(token)) return "mlir-attr";
  return "";
}

function highlightMlirLine(line) {
  const commentIndex = line.indexOf("//");
  const code = commentIndex >= 0 ? line.slice(0, commentIndex) : line;
  const comment = commentIndex >= 0 ? line.slice(commentIndex) : "";
  const tokenPattern = /(%[A-Za-z0-9_.$-]+|[@#][A-Za-z0-9_.$-]+|[A-Za-z_][A-Za-z0-9_.$-]*:?)/g;
  let output = "";
  let cursor = 0;
  for (const match of code.matchAll(tokenPattern)) {
    output += escapeHtml(code.slice(cursor, match.index));
    const token = match[0];
    const cssClass = classifyMlirToken(token);
    output += cssClass
      ? `<span class="${cssClass}">${escapeHtml(token)}</span>`
      : escapeHtml(token);
    cursor = match.index + token.length;
  }
  output += escapeHtml(code.slice(cursor));
  if (comment) {
    output += `<span class="mlir-comment">${escapeHtml(comment)}</span>`;
  }
  return output;
}

function highlightMlir(value) {
  return valueText(value).split("\\n").map(highlightMlirLine).join("\\n");
}

function codeBlock(value) {
  return `<pre class="source-code"><code class="mlir-code">${highlightMlir(value)}</code></pre>`;
}

function expandSourceReader(title, codeHtml) {
  const overlay = document.getElementById("source-reader-overlay");
  document.getElementById("source-reader-title").textContent = title || "源码阅读";
  document.getElementById("source-reader-code").innerHTML = codeHtml || "";
  overlay.hidden = false;
}

function closeSourceReader() {
  document.getElementById("source-reader-overlay").hidden = true;
}

function installSourceExpandActions() {
  document.querySelectorAll(".source-expand-button").forEach((button) => {
    button.addEventListener("click", () => {
      const section = button.closest(".inspector-section");
      const title = section ? section.querySelector("h3") : null;
      const code = section ? section.querySelector(".mlir-code") : null;
      expandSourceReader(title ? title.textContent : "源码阅读", code ? code.innerHTML : "");
    });
  });
}

function renderNodeIrSection(node) {
  const hasBody = Boolean(node.region_body);
  const localIr = hasBody ? node.region_body : (node.source_excerpt || "");
  const title = hasBody ? "Region Body" : "Op 源码";
  if (!localIr) return "";
  return `
<section class="inspector-section">
<div class="source-section-header">
<h3>${title}</h3>
<button class="source-expand-button" type="button">展开阅读</button>
</div>
${codeBlock(localIr)}
</section>`;
}

function semanticValueText(value) {
  if (Array.isArray(value)) {
    if (!value.length) return "无";
    if (value.some((item) => item && typeof item === "object")) {
      return value.map((item) => semanticValueText(item)).join("; ");
    }
    return value.join(", ");
  }
  if (value && typeof value === "object") return JSON.stringify(value);
  return valueText(value);
}

function chipList(values, className) {
  if (!Array.isArray(values) || !values.length) return "";
  return `<div class="${className}-list">${values.map((value) => `<span class="${className}">${escapeHtml(value)}</span>`).join("")}</div>`;
}

function tileParamField(label, value) {
  if (Array.isArray(value) && !value.length) return "";
  if (value === undefined || value === null || value === "" || value === false) return "";
  const displayValue = Array.isArray(value) ? value.join("/") : value;
  return `<div class="tile-param-field"><span>${escapeHtml(label)}</span><code>${escapeHtml(displayValue)}</code></div>`;
}

function tileParamsHtml(params) {
  if (!Array.isArray(params) || !params.length) return null;
  const cards = params.map((param) => {
    if (!param || typeof param !== "object") return semanticValueText(param);
    const name = param.name || "tile";
    const fields = [
      tileParamField("axis", param.axis),
      tileParamField("kind", param.axis_kind),
      tileParamField("binding", param.binding),
      tileParamField("default", param.default),
      tileParamField("upper_bound", param.upper_bound),
      tileParamField("extent", param.extent),
      tileParamField("roles", param.roles),
    ].filter(Boolean).join("");
    const uses = Array.isArray(param.primitive_uses) && param.primitive_uses.length
      ? `<div class="tile-param-field"><span>uses</span>${chipList(param.primitive_uses, "tile-chip")}</div>`
      : "";
    return `<div class="tile-param-card"><h5>${escapeHtml(name)}</h5><div class="tile-param-grid">${fields}${uses}</div></div>`;
  }).join("");
  return `<div class="tile-param-list">${cards}</div>`;
}

function structuredLoweringHtml(info) {
  if (!info || typeof info !== "object" || !Object.keys(info).length) return "";
  return semanticDetailRows([
    ["contract", info.contract],
    ["representation", info.representation],
    ["loop_axes", info.loop_axes],
    ["guard_marker_count", info.guard_marker_count],
    ["tail_marker_count", info.tail_marker_count],
    ["cache_read_marker", info.cache_read_marker],
    ["cache_write_marker", info.cache_write_marker],
    ["pipeline_marker", info.pipeline_marker],
    ["double_buffer_marker", info.double_buffer_marker],
  ]);
}

function functionInfoForNode(node) {
  if (!node || !node.function) return {};
  const stage = activeStage();
  const graph = stage && stage.graph ? stage.graph : {};
  const functions = Array.isArray(graph.functions) ? graph.functions : [];
  return functions.find((item) => item && item.name === node.function) || {};
}

function symbolMemberText(member) {
  if (!member || typeof member !== "object") return semanticValueText(member);
  const value = member.value !== undefined && member.value !== null ? `value ${member.value}` : "value ?";
  const dim = member.dim !== undefined && member.dim !== null ? `dim ${member.dim}` : "dim ?";
  return `${value} ${dim}`;
}

function symbolConstraintsHtml(constraints) {
  if (!Array.isArray(constraints) || !constraints.length) return "";
  const cards = constraints.map((constraint) => {
    if (!constraint || typeof constraint !== "object") return "";
    const name = constraint.sym_name || "symbol";
    const members = Array.isArray(constraint.members) ? constraint.members : [];
    const memberHtml = members.length
      ? `<div class="symbol-member-list">${members.map((member) => `<span class="symbol-member">${escapeHtml(symbolMemberText(member))}</span>`).join("")}</div>`
      : `<span class="panel-subtitle">无成员</span>`;
    return `<div class="symbol-constraint-card"><h5>${escapeHtml(name)}</h5>${memberHtml}</div>`;
  }).filter(Boolean).join("");
  return cards ? `<div class="symbol-constraint-list">${cards}</div>` : "";
}

function symbolConstraintsFieldValue(normalize) {
  if (!normalize || typeof normalize !== "object" || !Object.prototype.hasOwnProperty.call(normalize, "symbol_constraints")) return null;
  const constraints = normalize.symbol_constraints;
  if (!Array.isArray(constraints)) return semanticValueText(constraints);
  return constraints.length ? constraints.length : "[]";
}

function semanticDetailRows(rows) {
  const visible = rows.filter(([, value]) => {
    if (Array.isArray(value)) return value.length > 0;
    return value !== undefined && value !== null && value !== "" && value !== false;
  });
  if (!visible.length) return "";
  return detailRows(visible.map(([label, value]) => [label, semanticValueText(value)]));
}

function renderSemanticGroup(title, rows) {
  const body = semanticDetailRows(rows);
  if (!body) return "";
  return `<div class="semantic-group"><h4>${escapeHtml(title)}</h4>${body}</div>`;
}

function renderSemanticGroupBody(title, body) {
  if (!body) return "";
  return `<div class="semantic-group"><h4>${escapeHtml(title)}</h4>${body}</div>`;
}

function movementPhaseDescription(phase) {
  const descriptions = {
    data_copy: "GM/UB 间搬运输入、输出或中间值",
    vector_compute: "向量计算主体阶段",
    write_back: "把结果写回目标 buffer 或 GM",
  };
  return descriptions[phase] || "Schedule movement phase";
}

function movementPhasesHtml(phases) {
  if (!Array.isArray(phases) || !phases.length) return "";
  const legend = phases.map((phase) => `
<div class="phase-legend-row"><code>${escapeHtml(phase)}</code><span>${escapeHtml(movementPhaseDescription(phase))}</span></div>`).join("");
  return `<div class="phase-legend">${legend}</div>`;
}

function renderSemanticAttrSections(node) {
  const semantic = node.semantic_attrs || {};
  const badges = Array.isArray(node.badges) ? node.badges : [];
  const functionInfo = functionInfoForNode(node);
  const functionSemantic = functionInfo.semantic_attrs || {};
  if (!badges.length && !Object.keys(semantic).length && !Object.keys(functionSemantic).length) return "";
  const kernel = semantic.kernel || {};
  const dagKernel = resolveKernelDagEntry(kernel.id);
  const dagKernelId = dagKernel && dagKernel.id !== kernel.id ? dagKernel.id : null;
  const dagKernelNode = dagKernel ? (dagKernel.node || {}) : {};
  const schedule = semantic.schedule || {};
  const axisContract = schedule.axis_contract || {};
  const movement = semantic.movement || {};
  const memory = semantic.memory || {};
  const position = memory.position || {};
  const normalize = functionSemantic.normalize || {};
  const symbolConstraints = Array.isArray(normalize.symbol_constraints) ? normalize.symbol_constraints : [];
  return `
<section class="inspector-section">
<h3>属性分组</h3>
${renderSemanticGroup("Normalize", [
  ["function", functionInfo.name],
  ["normalized", normalize.normalized ? "true" : null],
  ["symbol_constraints", symbolConstraintsFieldValue(normalize)],
])}
${renderSemanticGroupBody("Symbol Constraints", symbolConstraintsHtml(symbolConstraints))}
${renderSemanticGroup("Kernel", [
  ["kernel_id", kernel.id],
  ["kernel_dag_id", dagKernelId],
  ["kind", dagKernelNode.kind],
  ["depth", dagKernelNode.depth],
  ["output_shape", dagKernelNode.output_shape],
  ["workspace_size", dagKernelNode.workspace_size],
  ["op_role", kernel.role],
  ["op_roles", kernel.roles],
  ["template_families", kernel.template_families],
  ["primary", kernel.primary ? "true" : null],
])}
${renderSemanticGroup("Schedule", [
  ["schedule_decision_id", schedule.decision_id],
  ["schedule_family", schedule.family],
  ["schedule_template", schedule.template],
  ["schedule_contract", schedule.schedule_contract],
  ["structured_lowering", schedule.structured_lowering ? "present" : null],
  ["tail_policies", schedule.tail_policies],
  ["target_tile_policy", schedule.target_tile_policy],
  ["runtime_top_k", schedule.runtime_top_k],
])}
${renderSemanticGroupBody("Structured Lowering", structuredLoweringHtml(schedule.structured_lowering))}
${renderSemanticGroup("Schedule Axis Contract", [
  ["source", axisContract.source],
  ["role", axisContract.role],
  ["result_rank", axisContract.result_rank],
  ["result_shape", axisContract.result_shape],
  ["guard_budget", axisContract.guard_budget],
  ["template_tags", axisContract.template_tags],
  ["shape_constraints", axisContract.shape_constraints],
  ["structure_constraints", axisContract.structure_constraints],
  ["tileable_axes", axisContract.tileable_axes],
  ["required_reduction_axes", axisContract.required_reduction_axes],
])}
${renderSemanticGroupBody("Tile", `
${semanticDetailRows([["tile_binding", schedule.tile_binding]])}
${tileParamsHtml(schedule.tile_params) || ""}
`)}
${renderSemanticGroupBody("Movement", movementPhasesHtml(movement.phases))}
${renderSemanticGroup("Memory / Buffer", [
  ["position.kind", position.kind],
  ["position.depth", position.depth],
  ["position.is_double_buffer", position.is_double_buffer ? "true" : null],
  ["position.raw", position.raw],
  ["tensor_id", memory.tensor_id],
  ["reuse_id", memory.reuse_id],
  ["position_id", memory.position_id],
  ["memory_space", memory.memory_space],
])}
</section>`;
}

function opSampleForNodeIds(graph, nodeIds) {
  const nodeById = Object.fromEntries((graph.nodes || []).map((node) => [node.id, node]));
  const seen = new Set();
  const ops = [];
  for (const nodeId of nodeIds) {
    const node = nodeById[nodeId];
    const opName = node && node.op_name;
    if (!opName || seen.has(opName)) continue;
    seen.add(opName);
    ops.push(opName);
    if (ops.length >= 8) break;
  }
  return ops.length ? ops.join(" -> ") : "none";
}

function renderPathSummary(graph, nodeId) {
  if (!graph || !nodeId) return "";
  const activeEdges = (graph.edges || []).filter(edgePassesKindFilter);
  const upstream = collectReachableNeighborhood(activeEdges, nodeId, "ancestors", Infinity);
  const downstream = collectReachableNeighborhood(activeEdges, nodeId, "descendants", Infinity);
  return `
<section class="inspector-section">
<h3>Path Summary</h3>
${detailRows([
  ["Upstream", `${upstream.nodeIds.size} nodes / ${upstream.edgeIds.size} edges`],
  ["Downstream", `${downstream.nodeIds.size} nodes / ${downstream.edgeIds.size} edges`],
  ["Upstream ops", opSampleForNodeIds(graph, upstream.nodeIds)],
  ["Downstream ops", opSampleForNodeIds(graph, downstream.nodeIds)],
])}
</section>`;
}

function kernelDagEntry(kernelId) {
  if (!kernelId) return null;
  const resolved = resolveKernelDagEntry(kernelId);
  if (!resolved) return null;
  return {id: resolved.id, node: resolved.node || {}};
}

function tensorDiffForKernel(kernelId) {
  if (!kernelId) return null;
  const tensorDiff = workspace.overlay_details && workspace.overlay_details.tensor_diff ? workspace.overlay_details.tensor_diff : {};
  const comparisons = Array.isArray(tensorDiff.comparisons) ? tensorDiff.comparisons : [];
  return comparisons.find((item) => item && (item.kernel_id === kernelId || item.task_id === kernelId)) || null;
}

function renderKernelDagScheduleEntries(entries) {
  if (!Array.isArray(entries) || !entries.length) return "<p>没有 schedule entry 诊断信息。</p>";
  const rows = entries.map((entry) => {
    const tileParams = tileParamsHtml(entry.tile_params) || "无";
    const structured = structuredLoweringHtml(entry.structured_lowering) || "无";
    return `<tr>
<td>${escapeHtml(entry.decision_id)}</td>
<td>${escapeHtml(entry.guard)}</td>
<td>${escapeHtml(entry.priority)}</td>
<td>${escapeHtml(entry.fallback)}</td>
<td>${escapeHtml(entry.shape_bucket_key)}</td>
<td>${escapeHtml(entry.host_tiling_id)}</td>
<td>${escapeHtml(entry.block_dim)}</td>
<td>${escapeHtml(entry.workspace_size)}</td>
<td>${tileParams}</td>
<td>${structured}</td>
</tr>`;
  }).join("");
  return `<table class="detail-table">
<thead><tr><th>Decision</th><th>Guard</th><th>Priority</th><th>Fallback</th><th>Shape Bucket</th><th>Host Tiling</th><th>Block Dim</th><th>Workspace</th><th>Tile Params</th><th>Structured Lowering</th></tr></thead>
<tbody>${rows}</tbody>
</table>`;
}

function renderKernelDagNodeDetail(node) {
  if (!node) return "<p>没有找到 Kernel DAG 节点。</p>";
  const ops = Array.isArray(node.ops) ? node.ops : [];
  const opRows = ops.length
    ? ops.map((op) => `<tr><td>${escapeHtml(op.line)}</td><td>${escapeHtml(op.op)}</td><td>${escapeHtml(op.label)}</td><td>${escapeHtml(op.role)}</td><td>${escapeHtml(op.result_type)}</td></tr>`).join("")
    : '<tr><td colspan="5">没有 MLIR op 摘要。</td></tr>';
  return `
<section class="inspector-section">
<h3>Kernel DAG</h3>
${detailRows([
  ["kind", node.kind],
  ["depth", node.depth],
  ["input_degree", node.input_degree],
  ["output_degree", node.output_degree],
  ["output_shape", node.output_shape],
  ["output_dtype", node.output_dtype],
  ["workspace_size", node.workspace_size],
  ["semantic_source", node.semantic_source],
  ["schedule_entry_count", node.schedule_entry_count],
  ["guarded_schedule_entry_count", node.guarded_schedule_entry_count],
  ["fallback_schedule_entry_count", node.fallback_schedule_entry_count],
  ["host_tiling_ids", node.host_tiling_ids],
  ["tile_param_names", node.tile_param_names],
])}
</section>
<section class="inspector-section">
<h3>Schedule Entries</h3>
${renderKernelDagScheduleEntries(node.schedule_entries)}
</section>
<section class="inspector-section">
<h3>MLIR Ops</h3>
<table class="detail-table"><thead><tr><th>Line</th><th>Operation</th><th>Label</th><th>Role</th><th>Result</th></tr></thead><tbody>${opRows}</tbody></table>
</section>`;
}

function kernelRuntimeStatus(kernelId) {
  if (!kernelId) return {status: "no-kernel"};
  const locate = workspace.overlay_details && workspace.overlay_details.locate ? workspace.overlay_details.locate : {};
  const comparison = tensorDiffForKernel(kernelId);
  const failedIds = Array.isArray(locate.failed_kernel_ids) ? locate.failed_kernel_ids : [];
  const passedIds = Array.isArray(locate.passed_kernel_ids) ? locate.passed_kernel_ids : [];
  let locateStatus = "unchecked";
  if (locate.first_bad_kernel === kernelId) locateStatus = "first-bad";
  else if (failedIds.includes(kernelId)) locateStatus = "failed";
  else if (passedIds.includes(kernelId)) locateStatus = "passed";
  return {
    status: locateStatus,
    tensorStatus: comparison ? comparison.status : "none",
    comparisonId: comparison ? comparison.id : null,
    maxAbsError: comparison ? comparison.max_abs_error : null,
    firstBadKernel: locate.first_bad_kernel,
    firstBadDepth: locate.first_bad_depth,
  };
}

function directGraphContext(graph, nodeId) {
  const nodeById = Object.fromEntries((graph.nodes || []).map((node) => [node.id, node]));
  const edges = graph.edges || [];
  const enrich = (edge, direction) => {
    const peerId = direction === "in" ? edge.from : edge.to;
    const peer = nodeById[peerId] || {};
    const peerOp = peer.op_name || peer.label || peerId;
    return {
      ...edge,
      direction,
      peerId,
      peerOp,
      peerLabel: peer.label || peerOp,
    };
  };
  return {
    incoming: edges.filter((edge) => edge.to === nodeId).map((edge) => enrich(edge, "in")),
    outgoing: edges.filter((edge) => edge.from === nodeId).map((edge) => enrich(edge, "out")),
  };
}

function renderProvenanceEdgeList(edges, emptyText) {
  if (!edges.length) return `<div class="panel-subtitle">${escapeHtml(emptyText)}</div>`;
  return `<div class="provenance-edge-list">${edges.slice(0, 12).map((edge) => `
<span class="provenance-edge-chip">
<strong>${escapeHtml(edge.direction === "in" ? "from" : "to")}</strong>
<code>${escapeHtml(edge.peerOp)}</code>
<span>${escapeHtml(edge.kind || "value")}</span>
<code>${escapeHtml(edge.label || edge.value || "")}</code>
</span>`).join("")}</div>`;
}

function renderKernelLineage(node) {
  const kernelId = node && node.kernel_id;
  const functionName = node && node.function;
  if (!kernelId && !functionName) return "";
  const entry = kernelId ? kernelDagEntry(kernelId) : null;
  const dagNode = entry ? entry.node : {};
  const dagId = entry ? entry.id : null;
  const edges = workspace.kernel_dag && Array.isArray(workspace.kernel_dag.edges) ? workspace.kernel_dag.edges : [];
  const upstream = edges.filter((edge) => edge.to === dagId).map((edge) => edge.from);
  const downstream = edges.filter((edge) => edge.from === dagId).map((edge) => edge.to);
  const ops = Array.isArray(dagNode.ops) ? dagNode.ops.map((op) => op.label || op.op).filter(Boolean).slice(0, 6) : [];
  return `
<div class="provenance-block">
<h4>Kernel Lineage</h4>
${detailRows([
  ["stage kernel", kernelId || "none"],
  ["function", functionName],
  ["dag kernel", dagId],
  ["kind/depth", dagId ? `${dagNode.kind || "unknown"} / ${dagNode.depth || "?"}` : "none"],
  ["upstream", upstream.length ? upstream.join(", ") : "none"],
  ["downstream", downstream.length ? downstream.join(", ") : "none"],
  ["ops", ops.length ? ops.join(" -> ") : "none"],
  ["shape", dagNode.output_shape],
  ["workspace", dagNode.workspace_size],
])}
</div>`;
}

function renderDataflowProvenance(graph, node) {
  if (!graph || !node) return "";
  const context = directGraphContext(graph, node.id);
  return `
<div class="provenance-block">
<h4>Stage Dataflow</h4>
${detailRows([
  ["node", `${node.id} ${node.op_name || ""}`],
  ["results", node.result_values || node.label],
  ["inputs", node.input_values],
])}
${renderProvenanceEdgeList(context.incoming, "没有直接输入边。")}
${renderProvenanceEdgeList(context.outgoing, "没有直接输出边。")}
</div>`;
}

function renderRuntimeProvenance(node) {
  const kernelId = node && node.kernel_id;
  if (!kernelId) return "";
  const runtime = kernelRuntimeStatus(kernelId);
  return `
<div class="provenance-block">
<h4>Runtime / Locate</h4>
${detailRows([
  ["locate", runtime.status],
  ["tensor_diff", runtime.tensorStatus],
  ["comparison", runtime.comparisonId],
  ["max_abs_error", runtime.maxAbsError],
  ["first_bad", runtime.firstBadKernel],
  ["first_bad_depth", runtime.firstBadDepth],
])}
</div>`;
}

function renderMovementProvenance(node) {
  if (!node) return "";
  const semantic = node.semantic_attrs || {};
  const movement = semantic.movement || {};
  const memory = semantic.memory || {};
  const position = memory.position || {};
  const rows = [
    ["movement", movement.phases],
    ["memory_space", memory.memory_space],
    ["position", position.raw],
    ["tensor_id", memory.tensor_id],
    ["reuse_id", memory.reuse_id],
    ["position_id", memory.position_id],
  ];
  if (!semanticDetailRows(rows)) return "";
  return `
<div class="provenance-block">
<h4>Movement / Memory</h4>
${semanticDetailRows(rows)}
</div>`;
}

function renderProvenanceSection(stage, graph, node) {
  if (!node) return "";
  return `
<section class="inspector-section">
<h3>Provenance</h3>
<div class="provenance-grid">
<div class="provenance-block">
<h4>Stage Origin</h4>
${detailRows([
  ["Stage", stageStageTitle(stage)],
  ["Artifact", stage.path],
  ["line", node.line_end && node.line_end !== node.line ? `${node.line}-${node.line_end}` : node.line],
  ["graph_source", graph.graph_source || graph.tool || "ascend-debug"],
])}
</div>
${renderKernelLineage(node)}
${renderDataflowProvenance(graph, node)}
${renderRuntimeProvenance(node)}
${renderMovementProvenance(node)}
</div>
</section>`;
}

function renderStageNodeDetail(stage, graph, node, diff) {
  if (!node) return '<section class="inspector-section"><h3>节点详情</h3><div class="panel-subtitle">未选中节点。</div></section>';
  const diffStatus = diff && diff.status ? diff.status : "无";
  const detailItems = [
    ["Op", node.op_name],
    ["结果", node.result_values || node.label],
    ["输入", node.input_values],
    ["Type", node.result_type],
    ["Kernel", node.kernel_id],
    ["角色", node.op_role],
    ["行号", node.line_end && node.line_end !== node.line ? `${node.line}-${node.line_end}` : node.line],
    ["Diff 状态", diffStatus],
  ];
  if (node.constant_value !== undefined && node.constant_value !== null && node.constant_value !== "") {
    detailItems.splice(2, 0, ["常量值", node.constant_value]);
  }
  return `
<section class="inspector-section">
<h3>节点详情</h3>
${detailRows(detailItems)}
</section>
${renderSemanticAttrSections(node)}
${renderProvenanceSection(stage, graph, node)}
${renderNodeIrSection(node)}`;
}

function svgHeader(width, height, padding = GRAPH_CANVAS_PADDING) {
  const outerWidth = width + padding * 2;
  const outerHeight = height + padding * 2;
  return `<svg id="unified-debug-graph-svg" data-base-width="${outerWidth}" data-base-height="${outerHeight}" data-canvas-padding="${padding}" width="${outerWidth}" height="${outerHeight}" viewBox="0 0 ${outerWidth} ${outerHeight}" xmlns="http://www.w3.org/2000/svg">
<defs>
<marker id="arrow-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"></path>
</marker>
</defs>`;
}

function graphCanvas() {
  return document.getElementById("graph-canvas");
}

function currentSvg() {
  return document.getElementById("unified-debug-graph-svg");
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function applyGraphScale() {
  const svg = currentSvg();
  if (!svg) return;
  const baseWidth = Number(svg.dataset.baseWidth || svg.getAttribute("width") || 720);
  const baseHeight = Number(svg.dataset.baseHeight || svg.getAttribute("height") || 420);
  const scale = clamp(graphViewState.scale || 1, MIN_GRAPH_SCALE, MAX_GRAPH_SCALE);
  graphViewState.scale = scale;
  svg.style.width = `${baseWidth * scale}px`;
  svg.style.height = `${baseHeight * scale}px`;
  const zoom = document.getElementById("graph-zoom-value");
  if (zoom) zoom.textContent = `${Math.round(scale * 100)}%`;
}

function currentSvgPadding() {
  const svg = currentSvg();
  return svg ? Number(svg.dataset.canvasPadding || 0) : 0;
}

function scrollGraphToDefaultOrigin() {
  const canvas = graphCanvas();
  const padding = currentSvgPadding();
  const scale = graphViewState.scale || 1;
  canvas.scrollLeft = Math.max(0, padding * scale - GRAPH_DEFAULT_MARGIN);
  canvas.scrollTop = Math.max(0, padding * scale - GRAPH_DEFAULT_MARGIN);
}

function resetGraphView() {
  graphViewState.scale = 1;
  applyGraphScale();
  scrollGraphToDefaultOrigin();
  resetStageGraphViewState();
}

function resetStageGraphViewState() {
  if (activeMode !== "stage") return;
  stageGraphViewState = defaultStageGraphViewState();
  stageNeighborhoodActive = false;
  refreshStageGraphEffects(false);
  const stage = activeStage();
  if (stage) updateGraphUrlState(stage, selectedKey);
}

function fitGraphToView() {
  const canvas = graphCanvas();
  const svg = currentSvg();
  if (!svg) return;
  const baseWidth = Number(svg.dataset.baseWidth || svg.getAttribute("width") || 720);
  const baseHeight = Number(svg.dataset.baseHeight || svg.getAttribute("height") || 420);
  const fitScale = Math.min(
    (canvas.clientWidth - 24) / baseWidth,
    (canvas.clientHeight - 24) / baseHeight
  );
  graphViewState.scale = clamp(fitScale, MIN_GRAPH_SCALE, 1.5);
  applyGraphScale();
  scrollGraphToDefaultOrigin();
}

function zoomGraphAt(event) {
  if (!(event.ctrlKey || event.metaKey)) return;
  const canvas = graphCanvas();
  const oldScale = graphViewState.scale || 1;
  const factor = event.deltaY < 0 ? 1.12 : 0.88;
  const newScale = clamp(oldScale * factor, MIN_GRAPH_SCALE, MAX_GRAPH_SCALE);
  if (newScale === oldScale) return;
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const pointerX = event.clientX - rect.left;
  const pointerY = event.clientY - rect.top;
  const graphX = (canvas.scrollLeft + pointerX) / oldScale;
  const graphY = (canvas.scrollTop + pointerY) / oldScale;
  graphViewState.scale = newScale;
  applyGraphScale();
  canvas.scrollLeft = graphX * newScale - pointerX;
  canvas.scrollTop = graphY * newScale - pointerY;
}

function parseTranslate(element) {
  const transform = element.getAttribute("transform") || "";
  const match = /translate\\(([-0-9.]+),([-0-9.]+)\\)/.exec(transform);
  return match ? {x: Number(match[1]), y: Number(match[2])} : {x: 0, y: 0};
}

function centerGraphElement(element) {
  if (!element) return;
  const canvas = graphCanvas();
  const position = parseTranslate(element);
  const width = Number(element.dataset.width || 220);
  const height = Number(element.dataset.height || 82);
  const scale = graphViewState.scale || 1;
  const padding = currentSvgPadding();
  canvas.scrollLeft = Math.max(0, (position.x + padding + width / 2) * scale - canvas.clientWidth / 2);
  canvas.scrollTop = Math.max(0, (position.y + padding + height / 2) * scale - canvas.clientHeight / 2);
}

function isBlankCanvasPanTarget(target) {
  if (!(target instanceof Element)) return true;
  if (target === graphCanvas() || target === currentSvg()) return true;
  return !target.closest(".graph-node, .kernel-dag-node, .graph-edge, .graph-edge-path, .graph-edge-label");
}

function beginCanvasPan(event) {
  if (event.button !== 0 && event.button !== 2) return;
  if (event.button === 0 && !isBlankCanvasPanTarget(event.target)) return;
  const canvas = graphCanvas();
  event.preventDefault();
  canvasPanState = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    scrollLeft: canvas.scrollLeft,
    scrollTop: canvas.scrollTop,
  };
  canvas.classList.add("panning");
  canvas.setPointerCapture(event.pointerId);
}

function updateCanvasPan(event) {
  if (!canvasPanState || event.pointerId !== canvasPanState.pointerId) return;
  const canvas = graphCanvas();
  event.preventDefault();
  canvas.scrollLeft = canvasPanState.scrollLeft - (event.clientX - canvasPanState.startX);
  canvas.scrollTop = canvasPanState.scrollTop - (event.clientY - canvasPanState.startY);
}

function endCanvasPan(event) {
  if (!canvasPanState || event.pointerId !== canvasPanState.pointerId) return;
  const canvas = graphCanvas();
  canvas.classList.remove("panning");
  try {
    canvas.releasePointerCapture(event.pointerId);
  } catch (error) {
    // Pointer capture can already be gone if the pointer left the window.
  }
  canvasPanState = null;
}

function installCanvasPan() {
  const canvas = graphCanvas();
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("pointerdown", beginCanvasPan);
  canvas.addEventListener("pointermove", updateCanvasPan);
  canvas.addEventListener("pointerup", endCanvasPan);
  canvas.addEventListener("pointercancel", endCanvasPan);
  window.addEventListener("blur", () => {
    if (!canvasPanState) return;
    canvas.classList.remove("panning");
    canvasPanState = null;
  });
}

function installGraphNavigation() {
  const canvas = graphCanvas();
  canvas.addEventListener("wheel", zoomGraphAt, {passive: false});
  document.getElementById("graph-fit").addEventListener("click", fitGraphToView);
  document.getElementById("graph-reset").addEventListener("click", resetGraphView);
  const search = document.getElementById("graph-search");
  search.addEventListener("input", () => {
    activeSearchIndex = 0;
    searchActiveGraph();
  });
  search.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || activeSearchResults.length === 0) return;
    event.preventDefault();
    const item = activeSearchResults[activeSearchIndex % activeSearchResults.length];
    activeSearchIndex += 1;
    if (item) item.select();
  });
}

function setInspectorWidth(width) {
  const clamped = clamp(Number(width) || 0, MIN_INSPECTOR_WIDTH, MAX_INSPECTOR_WIDTH);
  document.documentElement.style.setProperty("--inspector-width", `${clamped}px`);
  localStorage.setItem("ascendDebugInspectorWidth", String(clamped));
}

function restoreInspectorWidth() {
  const stored = Number(localStorage.getItem("ascendDebugInspectorWidth"));
  if (stored) setInspectorWidth(stored);
}

function installInspectorResize() {
  const resizer = document.getElementById("inspector-resizer");
  const shell = document.querySelector(".app-shell");
  if (!resizer || !shell) return;
  resizer.addEventListener("pointerdown", (event) => {
    if (window.matchMedia("(max-width: 1180px)").matches) return;
    event.preventDefault();
    inspectorResizeState = {pointerId: event.pointerId};
    resizer.classList.add("active");
    resizer.setPointerCapture(event.pointerId);
  });
  resizer.addEventListener("pointermove", (event) => {
    if (!inspectorResizeState || inspectorResizeState.pointerId !== event.pointerId) return;
    const rect = shell.getBoundingClientRect();
    setInspectorWidth(rect.right - event.clientX);
  });
  const finish = (event) => {
    if (!inspectorResizeState || inspectorResizeState.pointerId !== event.pointerId) return;
    inspectorResizeState = null;
    resizer.classList.remove("active");
    try {
      resizer.releasePointerCapture(event.pointerId);
    } catch (error) {
      // Pointer capture can already be gone if the pointer left the window.
    }
  };
  resizer.addEventListener("pointerup", finish);
  resizer.addEventListener("pointercancel", finish);
}

function setSidebarCollapsed(collapsed) {
  const shell = document.querySelector(".app-shell");
  const button = document.getElementById("sidebar-toggle");
  shell.classList.toggle("sidebar-collapsed", Boolean(collapsed));
  button.textContent = collapsed ? "展开" : "收起";
  button.setAttribute("aria-expanded", collapsed ? "false" : "true");
  localStorage.setItem("ascendDebugSidebarCollapsed", collapsed ? "1" : "0");
}

function installSidebarToggle() {
  const button = document.getElementById("sidebar-toggle");
  if (!button) return;
  setSidebarCollapsed(localStorage.getItem("ascendDebugSidebarCollapsed") === "1");
  button.addEventListener("click", () => {
    const shell = document.querySelector(".app-shell");
    setSidebarCollapsed(!shell.classList.contains("sidebar-collapsed"));
  });
}

function installSourceReader() {
  const overlay = document.getElementById("source-reader-overlay");
  const close = document.getElementById("source-reader-close");
  if (!overlay || !close) return;
  close.addEventListener("click", closeSourceReader);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeSourceReader();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !overlay.hidden) closeSourceReader();
  });
}

function activeStage() {
  return workspace.stages[activeStageIndex] || workspace.stages[0] || null;
}

function stageKernelRecords(graph) {
  if (!graph || !Array.isArray(graph.nodes)) return [];
  const nodeById = Object.fromEntries(graph.nodes.map((node) => [node.id, node]));
  const records = [];
  const seen = new Set();
  if (Array.isArray(graph.kernels) && graph.kernels.length) {
    for (const kernel of graph.kernels) {
      const kernelId = kernel && kernel.kernel_id;
      if (!kernelId || seen.has(kernelId)) continue;
      const nodeIds = (kernel.node_ids || []).filter((id) => nodeById[id]);
      if (!nodeIds.length) continue;
      seen.add(kernelId);
      records.push({kernel_id: kernelId, node_ids: nodeIds, nodes: nodeIds.map((id) => nodeById[id])});
    }
  }
  const byKernel = new Map();
  for (const node of graph.nodes) {
    if (!node.kernel_id || seen.has(node.kernel_id)) continue;
    if (!byKernel.has(node.kernel_id)) byKernel.set(node.kernel_id, []);
    byKernel.get(node.kernel_id).push(node);
  }
  for (const [kernelId, nodes] of byKernel.entries()) {
    records.push({kernel_id: kernelId, node_ids: nodes.map((node) => node.id), nodes});
  }
  return records.sort((a, b) => a.kernel_id.localeCompare(b.kernel_id, undefined, {numeric: true}));
}

function stageHasKernelGroups(stage) {
  return Boolean(stage && stage.graph && stageKernelRecords(stage.graph).length);
}

function stageKernelAggregateSource(stage = activeStage()) {
  if (!stage || isSourceStage(stage)) return null;
  const stageIndex = workspace.stages.indexOf(stage);
  const startIndex = stageIndex >= 0 ? stageIndex : activeStageIndex;
  for (let index = startIndex; index >= 0; index -= 1) {
    const candidate = workspace.stages[index];
    if (!candidate || isSourceStage(candidate)) continue;
    if (stageHasKernelGroups(candidate)) {
      return {
        stage: candidate,
        index,
        is_current_stage: candidate === stage,
      };
    }
  }
  return null;
}

function stageHasKernelAggregateView(stage) {
  return Boolean(stageKernelAggregateSource(stage));
}

function defaultStageGraphView(stage) {
  return stageHasKernelAggregateView(stage) ? "kernel-aggregate" : "ops";
}

function ensureStageGraphView(stage) {
  const aggregateSource = stageKernelAggregateSource(stage);
  const hasAggregate = Boolean(aggregateSource);
  if (!hasAggregate && stageGraphViewState.graphView !== "ops") {
    stageGraphViewState.graphView = "ops";
    stageGraphViewState.kernelId = null;
  }
  if (!stageGraphViewState.graphViewExplicit) {
    stageGraphViewState.graphView = defaultStageGraphView(stage);
  }
  if (stageGraphViewState.graphView === "kernel-local") {
    const sourceGraph = aggregateSource && aggregateSource.stage ? aggregateSource.stage.graph : null;
    const ids = stageKernelRecords(sourceGraph).map((kernel) => kernel.kernel_id);
    if (!ids.includes(stageGraphViewState.kernelId)) {
      stageGraphViewState.graphView = hasAggregate ? "kernel-aggregate" : "ops";
      stageGraphViewState.kernelId = null;
    }
  }
}

function isSourceStage(stage) {
  if (!stage) return false;
  return stage.name === "source" || stage.step === "source" || stage.stage_label === "Source";
}

function stageGroupForIndex(index) {
  const groups = Array.isArray(workspace.stage_groups) ? workspace.stage_groups : [];
  return groups.find((group) => {
    if (Array.isArray(group.steps) && group.steps.some((step) => Number(step.stage_index) === Number(index))) {
      return true;
    }
    const inputIndex = group.input_stage ? Number(group.input_stage.stage_index) : -1;
    const outputIndex = group.output_stage ? Number(group.output_stage.stage_index) : -1;
    const defaultIndex = group.default_stage ? Number(group.default_stage.stage_index) : -1;
    return Number(index) === inputIndex || Number(index) === outputIndex || Number(index) === defaultIndex;
  }) || null;
}

function stageGroupChildren(group) {
  const rawChildren = Array.isArray(group.steps) && group.steps.length
    ? group.steps
    : [group.input_stage, group.output_stage, group.default_stage].filter(Boolean);
  const seen = new Set();
  const children = [];
  for (const child of rawChildren) {
    const stageIndex = Number(child.stage_index);
    if (seen.has(stageIndex)) continue;
    seen.add(stageIndex);
    children.push(child);
  }
  return children;
}

function stageNavigationSequence() {
  const groups = Array.isArray(workspace.stage_groups) ? workspace.stage_groups : [];
  const seen = new Set();
  const sequence = [];
  for (const group of groups) {
    for (const child of stageGroupChildren(group)) {
      const stageIndex = Number(child.stage_index);
      if (seen.has(stageIndex)) continue;
      seen.add(stageIndex);
      sequence.push(child);
    }
  }
  if (sequence.length) return sequence;
  return workspace.stages.map((stage, index) => ({...stage, stage_index: index}));
}

function stageBriefLabel(stage) {
  if (!stage) return "无";
  const stageName = String(stage.name || "");
  const orderText = stage.order === undefined || stage.order === null ? "" : String(stage.order);
  const paddedOrder = orderText.padStart(3, "0");
  if (orderText && stageName.startsWith(`${paddedOrder}-`)) return stageName;
  return `${stage.order} ${stage.name}`;
}

function stageStepInfo(stage) {
  return stage && stage.step_info && typeof stage.step_info === "object" ? stage.step_info : {};
}

function stageStageTitle(stage) {
  if (!stage) return "无";
  if (stage.stage_label) return stage.stage_label;
  if (stage.phase) return stage.phase;
  return stage.name === "source" ? "Source" : (stage.name || "Stage");
}

function stageStepTitle(stage) {
  const info = stageStepInfo(stage);
  return (stage && stage.step_label) || info.title || (stage && stage.step) || (stage && stage.name) || "stage";
}

function stageStepId(stage) {
  if (!stage) return "stage";
  if (stage.step_id) return stage.step_id;
  if (stage.step) return stage.step;
  return stage.name || "stage";
}

function stageGraphHeaderTitle(stage) {
  return `${stageStageTitle(stage)} / ${stageStepId(stage)}`;
}

function stageDiffTitle(stage) {
  const info = stageStepInfo(stage);
  if (info.title || (stage && stage.step)) return stageStepTitle(stage);
  return `Dump: ${stageBriefLabel(stage)}`;
}

function stageSameAsPreviousText(stage) {
  const same = stage && stage.same_as_previous;
  if (!same || typeof same !== "object" || !same.stage) return "";
  const reason = same.reason || "IR payload is identical to previous dumped step.";
  return `${reason} Reference: ${same.order || ""} ${same.stage}.`;
}

function stepExplanationRows(stage) {
  const info = stageStepInfo(stage);
  const rows = [
    ["目的", info.purpose],
    ["输入", info.inputs],
    ["输出", info.outputs],
    ["检查重点", info.inspect_hint],
    ["常见问题", info.common_failures],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  const sameText = stageSameAsPreviousText(stage);
  return {info, rows, sameText};
}

function renderStepExplanation(stage) {
  if (!stage) return "";
  const {info, rows, sameText} = stepExplanationRows(stage);
  if (!Object.keys(info).length && !sameText) return "";
  const body = rows.length
    ? `<dl class="step-explanation-grid">${rows.map(([label, value]) => `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`).join("")}</dl>`
    : "";
  const same = sameText ? `<div class="step-explanation-same">${escapeHtml(sameText)}</div>` : "";
  return `<div class="step-explanation-title">${escapeHtml(stageStepTitle(stage))}</div>${body}${same}`;
}

function updateStepExplanation(stage) {
  const panel = document.getElementById("step-explanation");
  if (!panel) return;
  const html = renderStepExplanation(stage);
  panel.hidden = !html;
  panel.innerHTML = html;
}

function selectedStageNodeSignature(stage, nodeId) {
  const graph = stage ? stage.graph : null;
  const node = graph && nodeId ? (graph.nodes || []).find((item) => item.id === nodeId) : null;
  if (!node) return null;
  return {
    id: node.id,
    label: node.label || "",
    opName: node.op_name || "",
    resultValues: Array.isArray(node.result_values) ? [...node.result_values] : [],
    line: node.line || null,
  };
}

function resolveStageNodeSelection(graph, signature) {
  const nodes = graph && Array.isArray(graph.nodes) ? graph.nodes : [];
  if (!nodes.length) return null;
  if (!signature) return nodes[0].id;
  const byId = nodes.find((node) => node.id === signature.id);
  if (byId) return byId.id;
  const primaryResult = signature.resultValues && signature.resultValues[0];
  if (primaryResult) {
    const byResult = nodes.find((node) => Array.isArray(node.result_values) && node.result_values.includes(primaryResult));
    if (byResult) return byResult.id;
  }
  if (signature.label) {
    const byLabel = nodes.find((node) => node.label === signature.label && node.op_name === signature.opName);
    if (byLabel) return byLabel.id;
  }
  const byOpAndLine = nodes.find((node) => node.op_name === signature.opName && node.line === signature.line);
  if (byOpAndLine) return byOpAndLine.id;
  return nodes[0].id;
}

function activateStageIndex(index) {
  pendingStageSelection = selectedStageNodeSignature(activeStage(), selectedKey);
  activeStageIndex = Number(index);
  stageNeighborhoodActive = !isSourceStage(activeStage());
  stageGraphViewState.graphViewExplicit = false;
  stageGraphViewState.kernelId = null;
  setMode("stage");
}

function updateStageButtonState() {
  document.querySelectorAll(".stage-child-button").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.stageIndex) === activeStageIndex);
  });
  document.querySelectorAll(".stage-group-parent").forEach((button) => {
    const childIndices = String(button.dataset.stageChildIndices || "")
      .split(",")
      .filter(Boolean)
      .map((value) => Number(value));
    button.classList.toggle("parent-active", childIndices.includes(activeStageIndex));
  });
}

function renderStagePhaseControls(stage = activeStage()) {
  const controls = document.getElementById("stage-phase-controls");
  if (!controls) return;
  if (activeMode !== "stage" || !stage) {
    controls.innerHTML = "";
    return;
  }
  const sequence = stageNavigationSequence();
  const currentPosition = sequence.findIndex((item) => Number(item.stage_index) === activeStageIndex);
  if (currentPosition < 0) {
    controls.innerHTML = "";
    return;
  }
  const phaseButtons = [];
  const addNavButton = (label, item) => {
    if (!item) return;
    const stepTitle = stageGraphHeaderTitle(item);
    phaseButtons.push(`<button class="stage-phase-button" data-stage-index="${escapeHtml(item.stage_index)}" title="${label} ${escapeHtml(stepTitle)}" aria-label="${label} ${escapeHtml(stepTitle)}" type="button">${label}</button>`);
  };
  if (currentPosition > 0) addNavButton("上一页", sequence[currentPosition - 1]);
  if (currentPosition >= 0 && currentPosition < sequence.length - 1) addNavButton("下一页", sequence[currentPosition + 1]);
  controls.innerHTML = phaseButtons.join("");
  controls.querySelectorAll(".stage-phase-button").forEach((button) => {
    button.addEventListener("click", () => activateStageIndex(button.dataset.stageIndex));
  });
}

function syncStageGraphControls() {
  const controls = document.getElementById("stage-graph-controls");
  if (!controls) return;
  controls.hidden = activeMode !== "stage";
  const stage = activeStage();
  const hasAggregate = stageHasKernelAggregateView(stage);
  document.querySelectorAll("[data-stage-graph-view]").forEach((button) => {
    const view = button.dataset.stageGraphView;
    button.classList.toggle("active", view === stageGraphViewState.graphView);
    button.disabled = view !== "ops" && !hasAggregate;
  });
  const backButton = document.getElementById("kernel-local-back-button");
  if (backButton) backButton.hidden = stageGraphViewState.graphView !== "kernel-local";
  document.querySelectorAll("[data-highlight-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.highlightMode === stageGraphViewState.highlightMode);
  });
  const depth = document.getElementById("highlight-depth");
  if (depth) depth.value = stageGraphViewState.depth;
  const focusToggle = document.getElementById("focus-toggle");
  if (focusToggle) {
    focusToggle.classList.toggle("active", stageGraphViewState.focusView);
    focusToggle.setAttribute("aria-pressed", stageGraphViewState.focusView ? "true" : "false");
  }
  document.querySelectorAll("[data-edge-kind-filter]").forEach((input) => {
    input.checked = stageGraphViewState.edgeKinds.has(input.dataset.edgeKindFilter);
  });
  const foldToggle = document.getElementById("fold-helper-toggle");
  if (foldToggle) foldToggle.checked = stageGraphViewState.foldHelpers;
}

function updateGraphUrlState(stage, nodeId) {
  if (!stage) return;
  const params = new URLSearchParams(window.location.search);
  params.set("stage", String(stage.order));
  if (nodeId) params.set("node", nodeId);
  else params.delete("node");
  params.set("focus", stageGraphViewState.highlightMode);
  params.set("depth", stageGraphViewState.depth);
  params.set("graph_view", stageGraphViewState.graphView);
  if (stageGraphViewState.kernelId) params.set("kernel", stageGraphViewState.kernelId);
  else params.delete("kernel");
  if (stageGraphViewState.focusView) params.set("view", "focus");
  else params.delete("view");
  const activeEdgeKinds = ALL_EDGE_KINDS.filter((kind) => stageGraphViewState.edgeKinds.has(kind));
  if (activeEdgeKinds.length === ALL_EDGE_KINDS.length) params.delete("edges");
  else params.set("edges", activeEdgeKinds.join(","));
  if (stageGraphViewState.foldHelpers) params.set("fold", "helpers");
  else params.delete("fold");
  const query = params.toString();
  const nextUrl = `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash || ""}`;
  window.history.replaceState(null, "", nextUrl);
}

function refreshStageGraphEffects(forceNeighborhood = false) {
  if (activeMode !== "stage") return;
  const stage = activeStage();
  const graph = currentStageDisplayGraph || (stage ? stage.graph : null);
  if (!stage || !graph) return;
  if (forceNeighborhood && selectedKey) stageNeighborhoodActive = true;
  syncStageGraphControls();
  applyStageNeighborhood(graph, selectedKey, stageNeighborhoodActive && Boolean(selectedKey));
  if (selectedKey) updateGraphUrlState(stage, selectedKey);
}

function installStageGraphControls() {
  document.querySelectorAll("[data-stage-graph-view]").forEach((button) => {
    if (button.id === "kernel-local-back-button") return;
    button.addEventListener("click", () => {
      const view = button.dataset.stageGraphView;
      if (!VALID_STAGE_GRAPH_VIEWS.has(view)) return;
      stageGraphViewState.graphView = view;
      stageGraphViewState.graphViewExplicit = true;
      if (view !== "kernel-local") stageGraphViewState.kernelId = null;
      selectedKey = null;
      stageNeighborhoodActive = false;
      renderStageGraph();
    });
  });
  const backButton = document.getElementById("kernel-local-back-button");
  if (backButton) backButton.addEventListener("click", returnToStageKernelAggregateView);
  document.querySelectorAll("[data-highlight-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      stageGraphViewState.highlightMode = button.dataset.highlightMode;
      refreshStageGraphEffects(true);
    });
  });
  const depth = document.getElementById("highlight-depth");
  if (depth) {
    depth.addEventListener("change", () => {
      stageGraphViewState.depth = VALID_DEPTHS.has(depth.value) ? depth.value : "all";
      refreshStageGraphEffects(true);
    });
  }
  const focusToggle = document.getElementById("focus-toggle");
  if (focusToggle) {
    focusToggle.addEventListener("click", () => {
      stageGraphViewState.focusView = !stageGraphViewState.focusView;
      refreshStageGraphEffects(true);
    });
  }
  document.querySelectorAll("[data-edge-kind-filter]").forEach((input) => {
    input.addEventListener("change", () => {
      const kind = input.dataset.edgeKindFilter;
      if (input.checked) stageGraphViewState.edgeKinds.add(kind);
      else stageGraphViewState.edgeKinds.delete(kind);
      if (!stageGraphViewState.edgeKinds.size) {
        for (const fallbackKind of ALL_EDGE_KINDS) stageGraphViewState.edgeKinds.add(fallbackKind);
      }
      refreshStageGraphEffects(true);
    });
  });
  const foldToggle = document.getElementById("fold-helper-toggle");
  if (foldToggle) {
    foldToggle.addEventListener("change", () => {
      stageGraphViewState.foldHelpers = foldToggle.checked;
      refreshStageGraphEffects(false);
    });
  }
  syncStageGraphControls();
}

function activeStageDiff(stage = activeStage()) {
  if (!stage || !Array.isArray(workspace.stage_diffs)) return null;
  return workspace.stage_diffs.find((item) => item.to_stage && item.to_stage.path === stage.path) || null;
}

function renderStageDiff(stage = activeStage()) {
  const summary = document.getElementById("stage-diff-summary");
  const details = document.getElementById("stage-diff-details");
  if (!summary || !details) return;
  if (activeMode !== "stage") {
    summary.innerHTML = '<span class="panel-subtitle">Kernel DAG 模式没有相邻 Stage Diff。</span>';
    details.innerHTML = "";
    return;
  }
  const diff = activeStageDiff(stage);
  if (!diff) {
    summary.innerHTML = '<span class="panel-subtitle">source Stage 没有前序阶段。</span>';
    details.innerHTML = "";
    return;
  }
  summary.innerHTML = `
<div class="panel-subtitle">${escapeHtml(stageDiffTitle(diff.from_stage))} -> ${escapeHtml(stageDiffTitle(diff.to_stage))}</div>
<div class="diff-counts">
<div class="diff-pill diff-added-text"><strong>${escapeHtml(diff.added_count)}</strong>新增</div>
<div class="diff-pill diff-changed-text"><strong>${escapeHtml(diff.changed_count)}</strong>变化</div>
<div class="diff-pill diff-removed-text"><strong>${escapeHtml(diff.removed_count)}</strong>移除</div>
</div>`;
  const changed = (diff.changed_nodes || []).slice(0, 8).map((node) => {
    const after = node.after || {};
    const fields = (node.changes || []).map((change) => change.field).join(", ");
    return `<li><span class="diff-changed-text">${escapeHtml(after.op_name || after.label || "node")}</span>: ${escapeHtml(fields || "metadata")}</li>`;
  });
  const changedFunctions = (diff.changed_functions || []).slice(0, 5).map((func) => {
    const fields = (func.changes || []).map((change) => change.field).join(", ");
    return `<li><span class="diff-changed-text">func.func @${escapeHtml(func.name || "anonymous")}</span>: ${escapeHtml(fields || "function metadata")}</li>`;
  });
  const added = (diff.added_nodes || []).slice(0, 5).map((node) => (
    `<li><span class="diff-added-text">${escapeHtml(node.op_name || node.label || "node")}</span>: 新增</li>`
  ));
  const removed = (diff.removed_nodes || []).slice(0, 5).map((node) => (
    `<li><span class="diff-removed-text">${escapeHtml(node.op_name || node.label || "node")}</span>: 从当前 Stage 移除</li>`
  ));
  const rows = [...changedFunctions, ...changed, ...added, ...removed];
  details.innerHTML = rows.length ? `<ul class="diff-list">${rows.join("")}</ul>` : '<span class="panel-subtitle">没有检测到语义节点变化。</span>';
}

function renderGraphAudit(stage = activeStage()) {
  const summary = document.getElementById("graph-audit-summary");
  const details = document.getElementById("graph-audit-details");
  if (!summary || !details) return;
  if (activeMode !== "stage") {
    summary.innerHTML = "";
    details.innerHTML = "";
    return;
  }
  const audit = stage && stage.graph && stage.graph.connectivity ? stage.graph.connectivity : {};
  const suspicious = audit.suspicious_isolated_count || 0;
  const dangling = audit.dangling_effect_count || 0;
  const allowed = audit.allowed_terminal_count || 0;
  const issueCount = suspicious + dangling;
  const summaryClass = issueCount ? "warn" : "ok";
  const summaryTitle = issueCount ? `${issueCount} 个连接问题` : "连接检查通过";
  const summaryText = issueCount
    ? "存在 dangling 或 isolated 节点，下面列出需要检查的项。"
    : "没有检测到 suspicious_isolated / dangling_effect。";
  summary.innerHTML = `
<div class="audit-summary-strip ${summaryClass}">
<strong>${escapeHtml(summaryTitle)}</strong>
<span>${escapeHtml(summaryText)}</span>
</div>
<div class="audit-metrics">
<div class="audit-metric"><span class="audit-metric-value diff-removed-text">${escapeHtml(suspicious)}</span><span class="audit-metric-label" title="suspicious_isolated">suspicious_isolated</span></div>
<div class="audit-metric"><span class="audit-metric-value diff-changed-text">${escapeHtml(dangling)}</span><span class="audit-metric-label" title="dangling_effect">dangling_effect</span></div>
<div class="audit-metric"><span class="audit-metric-value diff-added-text">${escapeHtml(allowed)}</span><span class="audit-metric-label" title="allowed_terminal">allowed_terminal</span></div>
</div>`;
  const issueRows = [
    ...(audit.suspicious_isolated_nodes || []).map((node) => ({...node, kind: "suspicious_isolated"})),
    ...(audit.dangling_effect_nodes || []).map((node) => ({...node, kind: "dangling_effect"})),
  ].slice(0, 8).map((node) => (
    `<li class="audit-detail-row"><span class="audit-kind diff-removed-text" title="${escapeHtml(node.kind)}">${escapeHtml(node.kind)}</span><span class="audit-node">${escapeHtml(node.op_name || node.label || "node")} ${escapeHtml(node.line ? `line ${node.line}` : "")}</span></li>`
  ));
  const allowedRows = (audit.allowed_terminal_nodes || []).slice(0, 5).map((node) => (
    `<li class="audit-detail-row"><span class="audit-kind diff-added-text" title="allowed_terminal">allowed_terminal</span><span class="audit-node">${escapeHtml(node.op_name || node.label || "node")} ${escapeHtml(node.reason || "")}</span></li>`
  ));
  const edgeKindCounts = audit.edge_kind_counts || {};
  const edgeKinds = Object.entries(edgeKindCounts)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([kind, count]) => `${kind}:${count}`)
    .join(", ");
  details.innerHTML = `
<div class="audit-detail-group audit-issues">
<div class="audit-detail-heading"><span>Issues</span><span>${escapeHtml(issueRows.length)}</span></div>
${issueRows.length ? `<ul class="audit-detail-list">${issueRows.join("")}</ul>` : '<span class="panel-subtitle">没有检测到 suspicious_isolated / dangling_effect。</span>'}
</div>
${allowedRows.length ? `<div class="audit-detail-group audit-allowed"><div class="audit-detail-heading"><span>Allowed terminal</span><span>${escapeHtml(allowedRows.length)}</span></div><ul class="audit-detail-list">${allowedRows.join("")}</ul></div>` : ""}
<div class="audit-meta">components: ${escapeHtml(audit.component_count || 0)}<br>edge kinds: ${escapeHtml(edgeKinds || "none")}</div>`;
}

function wrapNodeOutputText(text, limit = NODE_OUTPUT_WRAP_LIMIT) {
  let remaining = String(text || "").trim();
  const lines = [];
  while (remaining.length > limit) {
    let cut = -1;
    for (const separator of [" ", ",", ">"]) {
      const candidate = remaining.lastIndexOf(separator, limit);
      if (candidate > cut) cut = candidate;
    }
    let chunk = "";
    if (cut <= 0) {
      chunk = remaining.slice(0, limit).trimEnd();
      remaining = remaining.slice(limit).trimStart();
    } else {
      chunk = remaining.slice(0, cut + 1).trimEnd();
      remaining = remaining.slice(cut + 1).trimStart();
    }
    if (chunk) lines.push(chunk);
  }
  if (remaining) lines.push(remaining);
  return lines.length ? lines : [""];
}

function nodeOutputValue(node, value) {
  const constantValue = node && node.op_name === "arith.constant" ? node.constant_value : null;
  if (constantValue === undefined || constantValue === null || constantValue === "") return value;
  return `${value} = ${constantValue}`;
}

function outputShapeLinesForNode(node) {
  const resultValues = Array.isArray(node.result_values) ? node.result_values.filter(Boolean) : [];
  if (!resultValues.length) return [];
  const resultTypes = Array.isArray(node.result_types) ? node.result_types : [];
  const fallbackType = node.result_type || "";
  const typeForIndex = (index) => resultTypes[index] || fallbackType;
  return resultValues.flatMap((value, index) => {
    const outputValue = nodeOutputValue(node, value);
    const type = typeForIndex(index);
    if (!type) return wrapNodeOutputText(outputValue);
    const combined = `${outputValue} ${type}`;
    if (combined.length <= NODE_OUTPUT_WRAP_LIMIT) return [combined];
    return [...wrapNodeOutputText(outputValue), ...wrapNodeOutputText(type)];
  });
}

function stageNodeDiffInfo(stage, nodeId) {
  const diff = activeStageDiff(stage);
  return diff && diff.node_status ? diff.node_status[nodeId] : null;
}

function functionFramesForGraph(graph, layout) {
  const layoutNodes = layout && layout.nodes ? layout.nodes : {};
  let groups = Array.isArray(graph.functions) ? graph.functions : [];
  if (!groups.length && graph.function && Array.isArray(graph.nodes)) {
    groups = [{name: graph.function, node_ids: graph.nodes.map((node) => node.id)}];
  }
  const frames = [];
  for (const group of groups) {
    const positions = (group.node_ids || []).map((id) => layoutNodes[id]).filter(Boolean);
    if (!positions.length) continue;
    const minX = Math.min(...positions.map((position) => position.x));
    const minY = Math.min(...positions.map((position) => position.y));
    const maxX = Math.max(...positions.map((position) => position.x + position.width));
    const maxY = Math.max(...positions.map((position) => position.y + position.height));
    const padX = 18;
    const padTop = 28;
    const padBottom = 18;
    frames.push({
      name: group.name || "anonymous",
      count: positions.length,
      x: Math.max(4, minX - padX),
      y: Math.max(4, minY - padTop),
      width: maxX - minX + padX * 2,
      height: maxY - minY + padTop + padBottom,
      labelWidth: Math.min(maxX - minX + padX * 2 - 24, Math.max(168, 86 + String(group.name || "anonymous").length * 7.2)),
    });
  }
  return frames;
}

function isStageKernelBoundaryNode(node, direction) {
  if (!node || node.kernel_id) return false;
  if (node.op_name === "func.arg" && direction === "input") return true;
  if (node.op_name === "func.return" && direction === "output") return true;
  if (node.op_name === "tensor.empty" || node.op_name === "memref.alloc") return true;
  if (isHelperNode(node)) return false;
  if (String(node.op_name || "").startsWith("arith.")) return false;
  return true;
}

function buildStageKernelAggregateGraph(stage, graph) {
  const records = stageKernelRecords(graph);
  const originalNodeById = Object.fromEntries((graph.nodes || []).map((node) => [node.id, node]));
  const nodeToKernel = {};
  for (const record of records) {
    for (const nodeId of record.node_ids) nodeToKernel[nodeId] = record.kernel_id;
  }
  const boundaryNodeIds = new Set();
  const boundaryKinds = {};
  for (const edge of graph.edges || []) {
    const fromKernel = nodeToKernel[edge.from];
    const toKernel = nodeToKernel[edge.to];
    if (!fromKernel && toKernel && isStageKernelBoundaryNode(originalNodeById[edge.from], "input")) {
      boundaryNodeIds.add(edge.from);
      boundaryKinds[edge.from] = boundaryKinds[edge.from] === "output" ? "input/output" : "input";
    }
    if (fromKernel && !toKernel && isStageKernelBoundaryNode(originalNodeById[edge.to], "output")) {
      boundaryNodeIds.add(edge.to);
      boundaryKinds[edge.to] = boundaryKinds[edge.to] === "input" ? "input/output" : "output";
    }
  }
  const aggregateIdForNode = (nodeId) => {
    const kernelId = nodeToKernel[nodeId];
    if (kernelId) return `kernel:${kernelId}`;
    return boundaryNodeIds.has(nodeId) ? `boundary:${nodeId}` : null;
  };
  const edgeMap = new Map();
  for (const edge of graph.edges || []) {
    const fromId = aggregateIdForNode(edge.from);
    const toId = aggregateIdForNode(edge.to);
    if (!fromId || !toId || fromId === toId) continue;
    const kind = edgeKind(edge);
    const key = `${fromId}->${toId}:${kind}:${edge.value || ""}`;
    if (!edgeMap.has(key)) {
      edgeMap.set(key, {
        id: `kernel_edge_${edgeMap.size}`,
        from: fromId,
        to: toId,
        kind,
        value: edge.value || kind,
        label: edge.value || kind,
        source_count: 0,
      });
    }
    edgeMap.get(key).source_count += 1;
  }
  const edges = Array.from(edgeMap.values());
  const aggregateNodes = [];
  for (const record of records) {
    const roles = [...new Set(record.nodes.map((node) => node.op_role).filter(Boolean))];
    aggregateNodes.push({
      id: `kernel:${record.kernel_id}`,
      op_name: "Kernel",
      label: record.kernel_id,
      kernel_id: record.kernel_id,
      aggregate: true,
      internal_node_ids: record.node_ids,
      node_count: record.node_ids.length,
      input_values: [],
      result_values: [],
      result_type: "",
      badges: roles.slice(0, 3),
    });
  }
  for (const nodeId of Array.from(boundaryNodeIds).sort((a, b) => {
    const left = originalNodeById[a] || {};
    const right = originalNodeById[b] || {};
    return (left.line || 0) - (right.line || 0) || a.localeCompare(b, undefined, {numeric: true});
  })) {
    const source = originalNodeById[nodeId] || {};
    const boundaryKind = boundaryKinds[nodeId] || "boundary";
    aggregateNodes.push({
      ...source,
      id: `boundary:${nodeId}`,
      op_name: boundaryKind === "output" ? "Output" : (boundaryKind === "input" ? "Input" : "Boundary"),
      label: source.label || source.op_name || nodeId,
      boundary: true,
      boundary_kind: boundaryKind,
      original_node_id: nodeId,
      badges: [boundaryKind],
    });
  }
  const aggregateIds = aggregateNodes.map((node) => node.id);
  const pred = new Map(aggregateIds.map((id) => [id, new Set()]));
  const succ = new Map(aggregateIds.map((id) => [id, new Set()]));
  for (const edge of edges) {
    pred.get(edge.to)?.add(edge.from);
    succ.get(edge.from)?.add(edge.to);
  }
  const depth = new Map(aggregateIds.map((id) => [id, 1]));
  const queue = aggregateIds.filter((id) => !pred.get(id)?.size);
  const pending = new Map(aggregateIds.map((id) => [id, pred.get(id)?.size || 0]));
  while (queue.length) {
    const current = queue.shift();
    for (const next of succ.get(current) || []) {
      depth.set(next, Math.max(depth.get(next) || 1, (depth.get(current) || 1) + 1));
      pending.set(next, Math.max(0, (pending.get(next) || 0) - 1));
      if (pending.get(next) === 0) queue.push(next);
    }
  }
  const levels = {};
  for (const node of aggregateNodes) {
    const level = depth.get(node.id) || 1;
    if (!levels[level]) levels[level] = [];
    levels[level].push(node);
  }
  const nodeW = 238, nodeH = 92, colW = 290, rowH = 132, left = 42, top = 42;
  const maxDepth = Math.max(1, ...Object.keys(levels).map(Number));
  const maxRows = Math.max(1, ...Object.values(levels).map((items) => items.length));
  const layoutNodes = {};
  const nodes = [];
  for (const [levelText, items] of Object.entries(levels)) {
    const level = Number(levelText);
    items.sort((a, b) => {
      const leftKind = a.boundary ? (a.boundary_kind === "input" ? 0 : 2) : 1;
      const rightKind = b.boundary ? (b.boundary_kind === "input" ? 0 : 2) : 1;
      return leftKind - rightKind || a.id.localeCompare(b.id, undefined, {numeric: true});
    });
    items.forEach((node, row) => {
      layoutNodes[node.id] = {x: left + row * colW, y: top + (level - 1) * rowH, width: nodeW, height: nodeH};
      nodes.push(node);
    });
  }
  const layoutEdges = edges.map((edge) => {
    const src = layoutNodes[edge.from];
    const dst = layoutNodes[edge.to];
    if (!src || !dst) return null;
    const x1 = src.x + src.width / 2, y1 = src.y + src.height;
    const x2 = dst.x + dst.width / 2, y2 = dst.y;
    const mid = (y1 + y2) / 2;
    return {
      ...edge,
      path: `M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`,
      label_x: (x1 + x2) / 2,
      label_y: mid - 6,
    };
  }).filter(Boolean);
  return {
    ...graph,
    graph_view: "kernel-aggregate",
    node_count: nodes.length,
    edge_count: edges.length,
    nodes,
    edges,
    functions: [],
    layout: {
      visual_kind: "stage-kernel-aggregate-top-to-bottom",
      width: left + maxRows * colW + 80,
      height: top + maxDepth * rowH + 88,
      nodes: layoutNodes,
      edges: layoutEdges,
    },
  };
}

function buildStageKernelLocalGraph(stage, graph, kernelId) {
  const record = stageKernelRecords(graph).find((kernel) => kernel.kernel_id === kernelId);
  if (!record || !graph.layout) return null;
  const ids = new Set(record.node_ids);
  const nodes = (graph.nodes || []).filter((node) => ids.has(node.id));
  const edges = (graph.edges || []).filter((edge) => ids.has(edge.from) && ids.has(edge.to));
  const sourceLayoutNodes = graph.layout.nodes || {};
  const sourcePositions = nodes.map((node) => sourceLayoutNodes[node.id]).filter(Boolean);
  if (!sourcePositions.length) return null;
  const minX = Math.min(...sourcePositions.map((position) => position.x));
  const minY = Math.min(...sourcePositions.map((position) => position.y));
  const maxX = Math.max(...sourcePositions.map((position) => position.x + position.width));
  const maxY = Math.max(...sourcePositions.map((position) => position.y + position.height));
  const offsetX = 42 - minX;
  const offsetY = 42 - minY;
  const layoutNodes = {};
  for (const node of nodes) {
    const position = sourceLayoutNodes[node.id];
    if (!position) continue;
    layoutNodes[node.id] = {...position, x: position.x + offsetX, y: position.y + offsetY};
  }
  const layoutEdges = (graph.layout.edges || [])
    .filter((edge) => ids.has(edge.from) && ids.has(edge.to))
    .map((edge) => {
      let coordinateIndex = 0;
      const shiftedPath = String(edge.path || "").replace(/-?\\d+(?:\\.\\d+)?/g, (value) => {
        const delta = coordinateIndex % 2 === 0 ? offsetX : offsetY;
        coordinateIndex += 1;
        return String(Number(value) + delta);
      });
      return {
        ...edge,
        path: shiftedPath,
        label_x: Number(edge.label_x || 0) + offsetX,
        label_y: Number(edge.label_y || 0) + offsetY,
      };
    });
  return {
    ...graph,
    graph_view: "kernel-local",
    local_kernel_id: kernelId,
    node_count: nodes.length,
    edge_count: edges.length,
    kernel_count: 1,
    nodes,
    edges,
    functions: (graph.functions || []).map((fn) => ({
      ...fn,
      node_ids: (fn.node_ids || []).filter((id) => ids.has(id)),
    })).filter((fn) => fn.node_ids && fn.node_ids.length),
    layout: {
      visual_kind: "stage-kernel-local",
      width: maxX - minX + 84,
      height: maxY - minY + 84,
      nodes: layoutNodes,
      edges: layoutEdges,
    },
  };
}

function graphSearchText(value) {
  if (value == null) return "";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value.map(graphSearchText).join(" ");
  }
  if (typeof value === "object") {
    return Object.values(value).map(graphSearchText).join(" ");
  }
  return "";
}

function findStageNodeElement(nodeId) {
  return Array.from(document.querySelectorAll(".graph-node")).find((element) => element.dataset.nodeId === nodeId) || null;
}

function edgeKind(edge) {
  return edge && edge.kind ? edge.kind : "value";
}

function edgePassesKindFilter(edge) {
  return stageGraphViewState.edgeKinds.has(edgeKind(edge));
}

function highlightDepthLimit() {
  if (stageGraphViewState.highlightMode === "direct") return 1;
  return stageGraphViewState.depth === "all" ? Infinity : Number(stageGraphViewState.depth);
}

function isHelperNode(node) {
  return node && HELPER_NODE_OPS.has(node.op_name);
}

function clearStageNeighborhood() {
  document.querySelectorAll(".graph-node").forEach((element) => {
    element.classList.remove("dimmed", "neighborhood-node", "parent-node", "child-node", "focus-hidden", "helper-collapsed");
  });
  document.querySelectorAll(".graph-edge").forEach((element) => {
    element.classList.remove("dimmed", "neighborhood-edge", "focus-hidden", "edge-filter-hidden", "helper-collapsed");
  });
}

function collectReachableNeighborhood(edges, nodeId, direction, maxDepth = Infinity) {
  const nodeIds = new Set();
  const edgeIds = new Set();
  const queue = [{id: nodeId, depth: 0}];
  const seen = new Set([nodeId]);
  while (queue.length) {
    const currentItem = queue.shift();
    const current = currentItem.id;
    if (currentItem.depth >= maxDepth) continue;
    for (const edge of edges) {
      const next = direction === "ancestors"
        ? (edge.to === current ? edge.from : null)
        : (edge.from === current ? edge.to : null);
      if (!next) continue;
      if (edge.id) edgeIds.add(edge.id);
      nodeIds.add(next);
      if (seen.has(next)) continue;
      seen.add(next);
      queue.push({id: next, depth: currentItem.depth + 1});
    }
  }
  return {nodeIds, edgeIds};
}

function applyEdgeAndHelperFilters(graph, selectedNodeId) {
  const edgeById = Object.fromEntries((graph.edges || []).map((edge) => [edge.id, edge]));
  const nodeById = Object.fromEntries((graph.nodes || []).map((node) => [node.id, node]));
  const helperIds = new Set(
    (graph.nodes || [])
      .filter((node) => isHelperNode(node) && node.id !== selectedNodeId)
      .map((node) => node.id)
  );
  document.querySelectorAll(".graph-node").forEach((element) => {
    const node = nodeById[element.dataset.nodeId];
    element.classList.toggle("helper-collapsed", stageGraphViewState.foldHelpers && helperIds.has(node && node.id));
  });
  document.querySelectorAll(".graph-edge").forEach((element) => {
    const edge = edgeById[element.dataset.edgeId];
    const helperLinked = edge && (helperIds.has(edge.from) || helperIds.has(edge.to));
    element.classList.toggle("edge-filter-hidden", edge ? !edgePassesKindFilter(edge) : false);
    element.classList.toggle("helper-collapsed", stageGraphViewState.foldHelpers && helperLinked);
  });
}

function applyStageNeighborhood(graph, nodeId, enabled = true) {
  clearStageNeighborhood();
  if (!graph) return;
  if (!enabled || !nodeId) {
    applyEdgeAndHelperFilters(graph, nodeId);
    return;
  }
  const activeEdges = (graph.edges || []).filter(edgePassesKindFilter);
  const highlightDepth = highlightDepthLimit();
  const ancestorNeighborhood = ["upstream", "both", "direct"].includes(stageGraphViewState.highlightMode)
    ? collectReachableNeighborhood(activeEdges, nodeId, "ancestors", highlightDepth)
    : {nodeIds: new Set(), edgeIds: new Set()};
  const descendantNeighborhood = ["downstream", "both", "direct"].includes(stageGraphViewState.highlightMode)
    ? collectReachableNeighborhood(activeEdges, nodeId, "descendants", highlightDepth)
    : {nodeIds: new Set(), edgeIds: new Set()};
  const parentIds = ancestorNeighborhood.nodeIds;
  const childIds = descendantNeighborhood.nodeIds;
  const connectedEdgeIds = new Set([
    ...ancestorNeighborhood.edgeIds,
    ...descendantNeighborhood.edgeIds,
  ]);
  const visibleIds = new Set([nodeId, ...parentIds, ...childIds]);
  document.querySelectorAll(".graph-node").forEach((element) => {
    const id = element.dataset.nodeId;
    const isVisible = visibleIds.has(id);
    element.classList.toggle("dimmed", !isVisible);
    element.classList.toggle("focus-hidden", stageGraphViewState.focusView && !isVisible);
    element.classList.toggle("neighborhood-node", isVisible);
    element.classList.toggle("parent-node", parentIds.has(id));
    element.classList.toggle("child-node", childIds.has(id));
  });
  document.querySelectorAll(".graph-edge").forEach((element) => {
    const isConnected = connectedEdgeIds.has(element.dataset.edgeId);
    element.classList.toggle("dimmed", !isConnected);
    element.classList.toggle("focus-hidden", stageGraphViewState.focusView && !isConnected);
    element.classList.toggle("neighborhood-edge", isConnected);
  });
  applyEdgeAndHelperFilters(graph, nodeId);
}

function findKernelNodeElement(kernelId) {
  return Array.from(document.querySelectorAll(".kernel-dag-node")).find((element) => element.dataset.kernelId === kernelId) || null;
}

function memoryHasKernel(kernelId) {
  const memory = workspace.overlay_details && workspace.overlay_details.memory ? workspace.overlay_details.memory : {};
  const detailed = Array.isArray(memory.kernels) ? memory.kernels : [];
  return detailed.some((kernel) => kernel && kernel.kernel_id === kernelId);
}

function memoryViewLink(kernelId) {
  return `summaries/memory.json.html#kernel-${encodeURIComponent(kernelId)}`;
}

function setSearchStatus(text) {
  const status = document.getElementById("graph-search-status");
  if (status) status.textContent = text;
}

function searchActiveGraph() {
  const query = document.getElementById("graph-search").value.trim().toLowerCase();
  activeSearchResults = [];
  document.querySelectorAll(".graph-node, .kernel-dag-node").forEach((element) => element.classList.remove("search-match"));
  if (!query) {
    setSearchStatus("");
    return;
  }
  if (activeMode === "stage") {
    const stage = activeStage();
    const graph = currentStageDisplayGraph || (stage ? stage.graph : null);
    if (!stage || !graph) return;
    for (const node of graph.nodes || []) {
      if (!graphSearchText(node).toLowerCase().includes(query)) continue;
      const element = findStageNodeElement(node.id);
      if (!element) continue;
      element.classList.add("search-match");
      activeSearchResults.push({
        select: () => selectStageNode(stage, graph, node.id, {center: true}),
      });
    }
  } else {
    const nodes = workspace.kernel_dag && workspace.kernel_dag.nodes ? workspace.kernel_dag.nodes : {};
    for (const [kernelId, node] of Object.entries(nodes)) {
      if (!`${kernelId} ${graphSearchText(node)}`.toLowerCase().includes(query)) continue;
      const element = findKernelNodeElement(kernelId);
      if (!element) continue;
      element.classList.add("search-match");
      activeSearchResults.push({
        select: () => selectKernel(kernelId, {center: true}),
      });
    }
  }
  setSearchStatus(activeSearchResults.length ? `${activeSearchResults.length} 个匹配，按 Enter 切换` : "无匹配");
  if (activeSearchResults.length) activeSearchResults[0].select();
}

function afterGraphRender() {
  applyGraphScale();
  renderStageDiff();
  renderGraphAudit();
  searchActiveGraph();
}

function enterStageKernelLocalView(kernelId) {
  stageGraphViewState.graphView = "kernel-local";
  stageGraphViewState.graphViewExplicit = true;
  stageGraphViewState.kernelId = kernelId;
  selectedKey = null;
  stageNeighborhoodActive = false;
  renderStageGraph();
}

function renderStageKernelAggregateGraph(stage, graph) {
  return buildStageKernelAggregateGraph(stage, graph);
}

function renderStageKernelLocalGraph(stage, graph, kernelId) {
  return buildStageKernelLocalGraph(stage, graph, kernelId);
}

function returnToStageKernelAggregateView() {
  stageGraphViewState.graphView = "kernel-aggregate";
  stageGraphViewState.graphViewExplicit = true;
  selectedKey = stageGraphViewState.kernelId ? `kernel:${stageGraphViewState.kernelId}` : null;
  stageGraphViewState.kernelId = null;
  stageNeighborhoodActive = false;
  renderStageGraph();
}

function renderStageGraph() {
  enterGraphMode();
  const stage = activeStage();
  const sourceGraph = stage ? stage.graph : null;
  ensureStageGraphView(stage);
  const aggregateSource = stageKernelAggregateSource(stage);
  const aggregateStage = aggregateSource ? aggregateSource.stage : null;
  const aggregateGraph = aggregateStage ? aggregateStage.graph : null;
  let graph = sourceGraph;
  if (stage && aggregateGraph && stageGraphViewState.graphView === "kernel-aggregate") {
    graph = renderStageKernelAggregateGraph(aggregateStage, aggregateGraph);
    graph.kernel_aggregate_source_stage = {
      order: aggregateStage.order,
      name: aggregateStage.name,
      stage_index: aggregateSource.index,
      title: stageGraphHeaderTitle(aggregateStage),
      is_current_stage: aggregateSource.is_current_stage,
    };
    graph.detail_stage = aggregateStage;
  } else if (stage && aggregateGraph && stageGraphViewState.graphView === "kernel-local") {
    graph = renderStageKernelLocalGraph(aggregateStage, aggregateGraph, stageGraphViewState.kernelId);
    if (!graph) {
      stageGraphViewState.graphView = aggregateSource ? "kernel-aggregate" : "ops";
      stageGraphViewState.kernelId = null;
      graph = stageGraphViewState.graphView === "kernel-aggregate"
        ? renderStageKernelAggregateGraph(aggregateStage, aggregateGraph)
        : sourceGraph;
    }
    if (graph && aggregateStage) {
      graph.kernel_aggregate_source_stage = {
        order: aggregateStage.order,
        name: aggregateStage.name,
        stage_index: aggregateSource.index,
        title: stageGraphHeaderTitle(aggregateStage),
        is_current_stage: aggregateSource.is_current_stage,
      };
      graph.detail_stage = aggregateStage;
    }
  }
  currentStageDisplayGraph = graph;
  const canvas = document.getElementById("graph-canvas");
  if (!graph || !graph.layout) {
    document.getElementById("graph-title").textContent = stage ? stageGraphHeaderTitle(stage) : "Stage Graph";
    document.getElementById("graph-subtitle").textContent = "";
    updateStepExplanation(stage);
    canvas.innerHTML = `${svgHeader(720, 420)}<text x="28" y="42">当前 Stage 没有可展示的图。</text></svg>`;
    afterGraphRender();
    return;
  }
  const viewTitleSuffix = graph.graph_view === "kernel-local"
    ? ` / ${stageGraphViewState.kernelId} 局部图`
    : (graph.graph_view === "kernel-aggregate" ? " / Kernel 主图" : "");
  document.getElementById("graph-title").textContent = `${stageGraphHeaderTitle(stage)}${viewTitleSuffix}`;
  const localSuffix = graph.graph_view === "kernel-local" ? "，Kernel 局部视图" : (graph.graph_view === "kernel-aggregate" ? "，Kernel 聚合视图" : "");
  const sourceSuffix = graph.kernel_aggregate_source_stage && !graph.kernel_aggregate_source_stage.is_current_stage
    ? `，Kernel 来源：${graph.kernel_aggregate_source_stage.title}`
    : "";
  document.getElementById("graph-subtitle").textContent = `${graph.node_count} 个节点，${graph.edge_count} 条边，${graph.kernel_count} 个 Kernel${localSuffix}${sourceSuffix}`;
  updateStepExplanation(stage);
  renderStagePhaseControls(stage);
  syncStageGraphControls();
  updateStageButtonState();
  const layout = graph.layout;
  const nodeById = Object.fromEntries(graph.nodes.map((node, index) => [node.id, {node, index}]));
  const diff = activeStageDiff(stage);
  const firstBad = workspace.overlays.locate && workspace.overlays.locate.first_bad_kernel;
  let svg = svgHeader(layout.width || 720, layout.height || 420);
  svg += `<g class="graph-content" transform="translate(${GRAPH_CANVAS_PADDING},${GRAPH_CANVAS_PADDING})">`;
  for (const frame of functionFramesForGraph(graph, layout)) {
    svg += `<g class="function-frame" data-function-name="${escapeHtml(frame.name)}">
<rect class="function-frame-box" x="${frame.x}" y="${frame.y}" width="${frame.width}" height="${frame.height}" rx="8"></rect>
<rect class="function-frame-label-bg" x="${frame.x + 8}" y="${frame.y + 6}" width="${frame.labelWidth}" height="22" rx="5"></rect>
<text class="function-frame-label" x="${frame.x + 16}" y="${frame.y + 21}">func.func @${escapeHtml(frame.name)}</text>
<text class="function-frame-meta" x="${frame.x + frame.width - 72}" y="${frame.y + 21}">${frame.count} 节点</text>
</g>`;
  }
  for (const edge of layout.edges || []) {
    const edgeClass = edge.kind ? ` ${String(edge.kind).replace(/[^A-Za-z0-9_-]+/g, "-").replace(/_/g, "-")}` : "";
    const edgeLabel = edge.label || edge.value;
    svg += `<g class="graph-edge${edgeClass}" data-edge-id="${escapeHtml(edge.id)}" data-edge-kind="${escapeHtml(edge.kind || "value")}" data-edge-from="${escapeHtml(edge.from)}" data-edge-to="${escapeHtml(edge.to)}"><path class="graph-edge-path${edgeClass}" d="${escapeHtml(edge.path)}"></path><text class="graph-edge-label" x="${edge.label_x}" y="${edge.label_y}">${escapeHtml(truncate(edgeLabel, 24))}</text></g>`;
  }
  for (const node of graph.nodes) {
    const position = layout.nodes[node.id];
    if (!position) continue;
    const outputShapeLines = outputShapeLinesForNode(node);
    const outputShapeElements = outputShapeLines.map((line, lineIndex) =>
      `<text class="node-shape" x="14" y="${47 + lineIndex * 14}">${escapeHtml(line)}</text>`
    ).join("");
    const inputsY = outputShapeLines.length ? 47 + outputShapeLines.length * 14 + 5 : 47;
    const inputs = node.input_values && node.input_values.length ? node.input_values.join(", ") : "root";
    const detail = node.aggregate
      ? `${node.node_count || 0} ops`
      : (node.boundary ? `${node.op_name || "boundary"} ${node.label || ""}` : (node.body_summary ? `body: ${node.body_summary}` : `in: ${inputs}`));
    const issueClass = firstBad && node.kernel_id === firstBad ? " issue" : "";
    const kernelClass = node.kernel_id ? " kernel-node" : "";
    const aggregateClass = node.aggregate ? " stage-kernel-aggregate" : "";
    const boundaryClass = node.boundary ? " stage-kernel-boundary" : "";
    const diffInfo = diff && diff.node_status ? diff.node_status[node.id] : null;
    const diffClass = diffInfo && diffInfo.status !== "unchanged" ? ` diff-${diffInfo.status}` : "";
    const kernelText = node.aggregate ? "点击进入" : (node.boundary ? String(node.boundary_kind || "boundary") : (node.kernel_id ? `kernel ${truncate(node.kernel_id, 22)}` : `line ${node.line}`));
    const badgeElements = (Array.isArray(node.badges) ? node.badges.slice(0, 3) : []).map((badge, badgeIndex) => {
      const label = truncate(badge, 30);
      const width = Math.min(position.width - 28, Math.max(46, label.length * 6.4 + 18));
      const y = inputsY + 22 + badgeIndex * 15;
      return `<rect class="node-badge-bg" x="14" y="${y}" width="${width}" height="12" rx="4"></rect><text class="node-badge" x="22" y="${y + 9}">${escapeHtml(label)}</text>`;
    }).join("");
    svg += `<g class="graph-node${kernelClass}${aggregateClass}${boundaryClass}${issueClass}${diffClass}" data-node-id="${escapeHtml(node.id)}" data-node-index="${nodeById[node.id].index}" data-width="${position.width}" data-height="${position.height}" tabindex="0" role="button" transform="translate(${position.x},${position.y})">
<title>${escapeHtml(node.op_name)}</title>
<rect width="${position.width}" height="${position.height}" rx="6"></rect>
<text class="node-op" x="14" y="24">${escapeHtml(truncate(node.op_name, 28))}</text>
${outputShapeElements}
<text class="node-inputs" x="14" y="${inputsY}">${escapeHtml(truncate(detail, 30))}</text>
<text class="node-kernel" x="${position.width - 14}" y="22">${escapeHtml(kernelText)}</text>
${badgeElements}
</g>`;
  }
  svg += "</g></svg>";
  canvas.innerHTML = svg;
  applyGraphScale();
  scrollGraphToDefaultOrigin();
  document.querySelectorAll(".graph-node").forEach((element) => {
    element.addEventListener("click", () => {
      const node = graph.nodes.find((item) => item.id === element.dataset.nodeId);
      if (node && node.aggregate) enterStageKernelLocalView(node.kernel_id);
      else selectStageNode(stage, graph, element.dataset.nodeId, {neighborhood: true});
    });
    element.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        const node = graph.nodes.find((item) => item.id === element.dataset.nodeId);
        if (node && node.aggregate) enterStageKernelLocalView(node.kernel_id);
        else selectStageNode(stage, graph, element.dataset.nodeId, {neighborhood: true});
      }
    });
  });
  const pendingMatch = pendingStageSelection ? resolveStageNodeSelection(graph, pendingStageSelection) : null;
  const requestedNodeMatch = !pendingStageSelection && !requestedNodeConsumed && requestedNode
    ? graph.nodes.find((node) => node.id === requestedNode || node.label === requestedNode)
    : null;
  const selectedMatch = !pendingMatch && selectedKey && graph.nodes.some((node) => node.id === selectedKey) ? selectedKey : null;
  const preferred = pendingMatch || (requestedNodeMatch && requestedNodeMatch.id) || selectedMatch || (graph.nodes[0] && graph.nodes[0].id);
  const pendingMatchShouldActivateNeighborhood = pendingMatch && !isSourceStage(stage);
  const requestedNodeShouldActivateNeighborhood = requestedNodeMatch && !isSourceStage(stage);
  const shouldRefreshNeighborhood = Boolean(pendingMatchShouldActivateNeighborhood || requestedNodeShouldActivateNeighborhood || stageNeighborhoodActive);
  if (requestedNodeMatch) requestedNodeConsumed = true;
  pendingStageSelection = null;
  const shouldUpdateRenderedViewUrl = stageGraphViewState.graphViewExplicit || graph.graph_view === "kernel-local";
  if (preferred && graph.graph_view !== "kernel-aggregate") {
    selectStageNode(stage, graph, preferred, {
      neighborhood: shouldRefreshNeighborhood,
      updateUrl: shouldUpdateRenderedViewUrl,
    });
  }
  else if (preferred) updateGraphUrlState(stage, preferred);
  afterGraphRender();
}

function selectStageNode(stage, graph, nodeId, options = {}) {
  selectedKey = nodeId;
  stageNeighborhoodActive = options.neighborhood !== false;
  document.querySelectorAll(".graph-node").forEach((element) => element.classList.toggle("selected", element.dataset.nodeId === nodeId));
  const node = graph.nodes.find((item) => item.id === nodeId);
  const detailStage = graph.detail_stage || stage;
  const links = [];
  if (detailStage.stage_view_path) links.push({label: "查看完整 MLIR", href: `../${detailStage.stage_view_path}${node && node.line ? `#L${node.line}` : ""}`});
  const kernelHref = node && node.kernel_id ? kernelDetailHref(node.kernel_id) : null;
  if (kernelHref) links.push({label: "Kernel 详情", href: kernelHref});
  if (node && node.kernel_id && memoryHasKernel(node.kernel_id)) links.push({label: "内存视图", href: memoryViewLink(node.kernel_id)});
  const diff = node ? stageNodeDiffInfo(detailStage, nodeId) : null;
  setInspector(
    node ? `节点详情：${node.op_name} ${node.label || ""}` : "节点详情",
    links,
    renderStageNodeDetail(detailStage, graph, node, diff) + renderPathSummary(graph, nodeId)
  );
  applyStageNeighborhood(graph, nodeId, stageNeighborhoodActive);
  if (options.updateUrl !== false) updateGraphUrlState(stage, nodeId);
  if (options.center) {
    centerGraphElement(findStageNodeElement(nodeId));
  }
}

function renderKernelDag() {
  enterGraphMode();
  const summary = workspace.kernel_dag || {};
  const nodes = summary.nodes || {};
  const edges = summary.edges || [];
  const ids = Object.keys(nodes).sort((a, b) => (nodes[a].depth || 0) - (nodes[b].depth || 0) || a.localeCompare(b));
  if (!ids.length) {
    document.getElementById("graph-title").textContent = "Kernel DAG";
    document.getElementById("graph-subtitle").textContent = "0 个 Kernel，0 条边，关键深度 0";
    updateStepExplanation(null);
    renderStagePhaseControls(null);
    syncStageGraphControls();
    document.getElementById("graph-canvas").innerHTML = `${svgHeader(720, 420)}<text x="188" y="234" class="empty-state-title">当前 run 未收集 Kernel DAG 产物。</text><text x="188" y="260" class="empty-state-subtitle">需要 artifact manifest / run manifest / kernelized IR 后才能构建 Kernel DAG。</text></svg>`;
    setInspector(
      "Kernel DAG 为空",
      [],
      "<p>当前 debug run 没有 graphs/kernel_dag.summary.json，或 summary 中没有 kernel nodes。</p><p>这通常表示 collect 只收集了 pass/stage dump，没有传入 artifact/run manifest。</p>"
    );
    afterGraphRender();
    return;
  }
  const levels = {};
  for (const id of ids) {
    const depth = nodes[id].depth || 1;
    if (!levels[depth]) levels[depth] = [];
    levels[depth].push(id);
  }
  const nodeW = 228, nodeH = 88, colW = 310, rowH = 122, left = 42, top = 42;
  const maxDepth = Math.max(1, ...Object.keys(levels).map(Number));
  const maxRows = Math.max(1, ...Object.values(levels).map((items) => items.length));
  const width = left + maxDepth * colW + 80;
  const height = top + maxRows * rowH + 88;
  const positions = {};
  for (const [depthText, items] of Object.entries(levels)) {
    const depth = Number(depthText);
    items.forEach((id, row) => {
      positions[id] = {x: left + (depth - 1) * colW, y: top + row * rowH, width: nodeW, height: nodeH};
    });
  }
  const criticalPairs = new Set((summary.critical_path || []).slice(0, -1).map((id, index) => `${id}->${summary.critical_path[index + 1]}`));
  const fusionPairs = new Set((summary.simple_fusion_edges || []).map((edge) => `${edge.from}->${edge.to}`));
  const firstBad = workspace.overlays.locate && workspace.overlays.locate.first_bad_kernel;
  let svg = svgHeader(width, height);
  svg += `<g class="graph-content" transform="translate(${GRAPH_CANVAS_PADDING},${GRAPH_CANVAS_PADDING})">`;
  for (const edge of edges) {
    const src = positions[edge.from], dst = positions[edge.to];
    if (!src || !dst) continue;
    const x1 = src.x + src.width, y1 = src.y + src.height / 2;
    const x2 = dst.x, y2 = dst.y + dst.height / 2;
    const mid = (x1 + x2) / 2;
    const key = `${edge.from}->${edge.to}`;
    const edgeClass = criticalPairs.has(key) ? " critical" : (fusionPairs.has(key) ? " fusion" : "");
    svg += `<path class="graph-edge-path${edgeClass}" d="M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}"></path>`;
  }
  for (const id of ids) {
    const node = nodes[id] || {};
    const pos = positions[id];
    const kind = node.kind || "vec";
    const issueClass = id === firstBad ? " issue" : "";
    const op = (node.ops && node.ops[0] && (node.ops[0].label || node.ops[0].op)) || "kernel";
    svg += `<g class="kernel-dag-node ${escapeHtml(kind)}${issueClass}" data-kernel-id="${escapeHtml(id)}" data-width="${nodeW}" data-height="${nodeH}" tabindex="0" role="button" transform="translate(${pos.x},${pos.y})">
<title>${escapeHtml(id)}</title>
<rect width="${nodeW}" height="${nodeH}" rx="7"></rect>
<text class="node-op" x="14" y="24">${escapeHtml(id)} [${escapeHtml(kind)}]</text>
<text class="node-result" x="14" y="48">${escapeHtml(truncate(op, 31))}</text>
<text class="node-inputs" x="14" y="68">depth ${escapeHtml(node.depth)} | ws ${escapeHtml(node.workspace_size)}</text>
</g>`;
  }
  svg += "</g></svg>";
  document.getElementById("graph-title").textContent = "Kernel DAG";
  document.getElementById("graph-subtitle").textContent = `${summary.kernel_count || 0} 个 Kernel，${summary.graph_edges || 0} 条边，关键深度 ${summary.critical_path_depth || 0}`;
  updateStepExplanation(null);
  renderStagePhaseControls(null);
  syncStageGraphControls();
  document.getElementById("graph-canvas").innerHTML = svg;
  applyGraphScale();
  scrollGraphToDefaultOrigin();
  document.querySelectorAll(".kernel-dag-node").forEach((element) => {
    element.addEventListener("click", () => selectKernel(element.dataset.kernelId));
    element.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectKernel(element.dataset.kernelId);
      }
    });
  });
  if (ids.length) selectKernel(selectedKey && nodes[selectedKey] ? selectedKey : ids[0]);
  afterGraphRender();
}

function selectKernel(kernelId, options = {}) {
  selectedKey = kernelId;
  document.querySelectorAll(".kernel-dag-node").forEach((element) => element.classList.toggle("selected", element.dataset.kernelId === kernelId));
  const node = workspace.kernel_dag && workspace.kernel_dag.nodes ? workspace.kernel_dag.nodes[kernelId] : null;
  const kernelHref = kernelDetailHref(kernelId);
  const links = kernelHref ? [{label: "Kernel 详情", href: kernelHref}] : [];
  if (memoryHasKernel(kernelId)) links.push({label: "内存视图", href: memoryViewLink(kernelId)});
  setInspector(`Kernel 详情：${kernelId}`, links, renderKernelDagNodeDetail(node));
  if (options.center) {
    centerGraphElement(findKernelNodeElement(kernelId));
  }
}

function setMode(mode) {
  activeMode = mode;
  selectedKey = null;
  document.querySelectorAll(".mode-tab").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  const sidebarBody = document.querySelector(".sidebar-body");
  if (sidebarBody) sidebarBody.classList.toggle("kernel-mode", mode === "kernel");
  if (mode === "kernel") renderKernelDag();
  else renderStageGraph();
}

document.querySelectorAll(".mode-tab").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
document.querySelectorAll(".stage-button[data-stage-index]").forEach((button) => button.addEventListener("click", () => {
  activateStageIndex(button.dataset.stageIndex);
}));
installWorkbenchViewLinks();
restoreInspectorWidth();
installInspectorResize();
installSidebarToggle();
installSourceReader();
installCanvasPan();
installGraphNavigation();
installStageGraphControls();
renderStageGraph();
</script>
"""
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Ascend Debug 调试工作台</title>
{style}
</head>
<body>
<header>
<div>
<h1>Ascend Debug 调试工作台</h1>
<div class="toolbar">
<a href="../index.html">调试首页</a>
<a href="../{_cell(summary_path)}">原始 debug_graph.json</a>
<span>主 Stage：{_cell(primary_stage.get('name') if primary_stage else 'none')}</span>
<span>Kernel DAG：{_cell(kernel_dag.get('kernel_count', 0))} 个 Kernel</span>
<span>Kernel DAG 来源：{_cell(kernel_dag.get('semantic_source'))}</span>
</div>
</div>
</header>
<main class="app-shell">
<aside class="sidebar">
<div class="sidebar-header">
<h2>视图模式</h2>
<button id="sidebar-toggle" class="sidebar-toggle" type="button" aria-expanded="true">收起</button>
</div>
<div class="mode-tabs">
<button class="mode-tab active" data-mode="stage" data-short="S" type="button">Stage 演进</button>
<button class="mode-tab" data-mode="kernel" data-short="K" type="button">Kernel DAG</button>
</div>
<div class="sidebar-body">
<div id="stage-navigation" class="stage-navigation">
<h2>Stage 列表</h2>
<div id="stage-list" class="stage-list">
{_stage_buttons(debug_graph)}
</div>
</div>
<section id="stage-diff-panel" class="stage-diff-panel">
<h2>Stage Diff</h2>
<div id="stage-diff-summary"></div>
<div id="stage-diff-details"></div>
</section>
<section id="graph-audit-panel" class="graph-audit-panel">
<h2>Graph Audit</h2>
<div id="graph-audit-summary"></div>
<div id="graph-audit-details"></div>
</section>
<div class="overlay-stack">
{_overlay_cards(overlays)}
</div>
</div>
</aside>
<section class="graph-panel">
<div class="panel-header">
<div class="panel-title-block">
<h2 id="graph-title">统一 Stage Graph</h2>
<div id="graph-subtitle" class="panel-subtitle"></div>
<div id="step-explanation" class="step-explanation" hidden></div>
</div>
<div class="panel-tools-column">
<div class="graph-tools">
<input id="graph-search" class="graph-search" type="search" placeholder="搜索 op、Kernel、位置">
<button id="graph-fit" class="graph-tool-button" type="button">适配</button>
<button id="graph-reset" class="graph-tool-button" type="button">Reset</button>
<span id="graph-zoom-value" class="zoom-value">100%</span>
<span id="graph-search-status" class="search-status"></span>
</div>
<div class="panel-control-strip">
<div id="stage-phase-controls" class="stage-phase-controls"></div>
<div id="stage-graph-controls" class="stage-graph-controls">
<div id="stage-graph-view-controls" class="segmented-control" aria-label="Stage graph view">
<button data-stage-graph-view="ops" type="button">Op 图</button>
<button data-stage-graph-view="kernel-aggregate" type="button">Kernel 主图</button>
</div>
<button id="kernel-local-back-button" class="graph-tool-button" data-stage-graph-view="kernel-local" type="button" hidden>返回主图</button>
<div id="highlight-mode-controls" class="segmented-control" aria-label="Highlight mode">
<button data-highlight-mode="direct" type="button">Direct</button>
<button data-highlight-mode="upstream" type="button">Upstream</button>
<button data-highlight-mode="downstream" type="button">Downstream</button>
<button data-highlight-mode="both" type="button">Both</button>
</div>
<label class="control-label">Depth
<select id="highlight-depth" class="control-select">
<option value="1">1</option>
<option value="2">2</option>
<option value="3">3</option>
<option value="all">All</option>
</select>
</label>
<button id="focus-toggle" class="graph-tool-button" type="button" aria-pressed="false">Focus</button>
<div id="edge-filter-controls" class="edge-filter-controls">
<label><input data-edge-kind-filter="value" type="checkbox" checked>value</label>
<label><input data-edge-kind-filter="memory_effect" type="checkbox" checked>memory</label>
<label><input data-edge-kind-filter="resource_effect" type="checkbox" checked>resource</label>
<label><input data-edge-kind-filter="control" type="checkbox" checked>control</label>
</div>
<label class="control-label"><input id="fold-helper-toggle" type="checkbox">Fold helpers</label>
</div>
</div>
</div>
</div>
<div id="graph-canvas" class="graph-canvas-wrap">
<svg id="unified-debug-graph-svg" width="720" height="420" viewBox="0 0 720 420" xmlns="http://www.w3.org/2000/svg"></svg>
</div>
</section>
<div id="inspector-resizer" class="layout-resizer" role="separator" aria-label="调整节点详情宽度" aria-orientation="vertical"></div>
<aside class="inspector-panel">
<h2 id="inspector-title">节点详情</h2>
<div id="inspector-actions" class="inspector-actions"></div>
<div id="inspector-detail"></div>
</aside>
</main>
<div id="source-reader-overlay" class="source-reader-overlay" hidden>
<section class="source-reader-panel" role="dialog" aria-modal="true" aria-labelledby="source-reader-title">
<div class="source-reader-header">
<h2 id="source-reader-title">源码阅读</h2>
<button id="source-reader-close" class="source-reader-close" type="button">关闭</button>
</div>
<pre class="source-code source-reader-code"><code id="source-reader-code" class="mlir-code"></code></pre>
</section>
</div>
<script type="application/json" id="graph-workspace-data">{workspace_json}</script>
{script}
</body>
</html>
"""
    layout.write_text(run_dir / view_rel_path, document)


def render_debug_graph(
    *,
    run_dir: pathlib.Path,
    manifest: dict[str, Any],
    stage_graph_views: dict[str, dict[str, Any]],
    kernel_summary: dict[str, Any] | None,
    tensor_diff: dict[str, Any] | None,
    locate_summary: dict[str, Any] | None,
    memory_summary: dict[str, Any] | None,
) -> dict[str, str]:
    stages = []
    for stage in sorted(manifest["stages"], key=lambda item: item["order"]):
        graph_view = stage_graph_views.get(stage["path"])
        if not graph_view:
            continue
        record = _stage_record(run_dir=run_dir, stage=stage, graph_view=graph_view)
        if record:
            stages.append(record)
    schedule_axis_contracts = _load_schedule_axis_contracts(run_dir, manifest)
    _annotate_schedule_axis_contracts(stages, schedule_axis_contracts)
    _annotate_stage_equivalence(run_dir, stages)
    primary_stage = _select_primary_stage(stages)
    stage_diffs = _compute_stage_diffs(stages)
    stage_groups = _build_stage_groups(run_dir, stages)
    stage_connectivity = [_stage_connectivity_record(stage) for stage in stages]
    summary = {
        "schema_version": 1,
        "tool": "ascend-debug",
        "visual_kind": "unified-debug-workspace",
        "stage_count": len(stages),
        "primary_stage": primary_stage,
        "stage_diffs": stage_diffs,
        "stage_connectivity": stage_connectivity,
        "stage_groups": stage_groups,
        "stages": stages,
        "timeline_lanes": _build_timeline_followup_lanes(run_dir, manifest),
        "schedule_axis_contracts": schedule_axis_contracts,
        "kernel_dag": kernel_summary or {},
        "kernel_detail_views": _kernel_detail_views(stages, kernel_summary),
        "overlays": _overlay_summary(
            tensor_diff=tensor_diff,
            locate_summary=locate_summary,
            memory_summary=memory_summary,
        ),
        "overlay_details": {
            "tensor_diff": tensor_diff or {},
            "locate": locate_summary or {},
            "memory": memory_summary or {},
        },
        "artifacts": {
            "commands": manifest.get("commands", []),
            "reports": manifest.get("reports", []),
            "graphs": manifest.get("graphs", []),
            "runtime": manifest.get("artifacts", []),
            "summaries": _summary_paths(run_dir),
        },
        "summary_path": "summaries/debug_graph.json",
        "view_path": "views/debug_graph.html",
    }
    layout.write_json(run_dir / "summaries/debug_graph.json", summary)
    _render_html(run_dir=run_dir, debug_graph=summary, view_rel_path="views/debug_graph.html")
    return {
        "summary_path": "summaries/debug_graph.json",
        "view_path": "views/debug_graph.html",
    }
