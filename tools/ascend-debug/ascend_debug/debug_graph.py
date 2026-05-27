from __future__ import annotations

import html
import json
import pathlib
from typing import Any

from ascend_debug import layout, stage_graph


def _cell(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _load_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _stage_record(
    *,
    run_dir: pathlib.Path,
    stage: dict[str, Any],
    graph_view: dict[str, Any],
) -> dict[str, Any] | None:
    graph_json_rel = graph_view.get("json_rel_path")
    graph_view_rel = graph_view.get("view_rel_path")
    if not isinstance(graph_json_rel, str) or not isinstance(graph_view_rel, str):
        return None
    graph = _load_json(run_dir / graph_json_rel)
    if not graph:
        return None
    return {
        "order": stage["order"],
        "name": stage["name"],
        "path": stage["path"],
        "graph_json_path": graph_json_rel,
        "graph_view_path": graph_view_rel,
        "node_count": graph.get("node_count", 0),
        "edge_count": graph.get("edge_count", 0),
        "kernel_count": graph.get("kernel_count", 0),
        "graph": graph,
    }


def _select_primary_stage(stages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for preferred in ("kernelize-out", "schedule-out", "realize-out"):
        for item in stages:
            if item["name"] == preferred:
                return item
    with_kernels = [item for item in stages if item.get("kernel_count", 0) > 0]
    if with_kernels:
        return max(with_kernels, key=lambda item: item["order"])
    return stages[-1] if stages else None


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
    return {
        "op_name": node.get("op_name"),
        "result_type": node.get("result_type"),
        "input_count": len(node.get("input_values") or []),
        "result_count": len(node.get("result_values") or []),
        "kernel_id": node.get("kernel_id"),
        "op_role": node.get("op_role"),
        "schedule_decision_id": node.get("schedule_decision_id"),
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


def _compute_stage_diffs(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diffs = []
    for before, after in zip(stages, stages[1:]):
        before_nodes = _semantic_nodes(before.get("graph", {}))
        after_nodes = _semantic_nodes(after.get("graph", {}))
        before_keys = set(before_nodes)
        after_keys = set(after_nodes)
        added_keys = sorted(after_keys - before_keys)
        removed_keys = sorted(before_keys - after_keys)
        common_keys = sorted(before_keys & after_keys)
        changed = []
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

        diffs.append(
            {
                "from_stage": {
                    "order": before.get("order"),
                    "name": before.get("name"),
                    "path": before.get("path"),
                },
                "to_stage": {
                    "order": after.get("order"),
                    "name": after.get("name"),
                    "path": after.get("path"),
                },
                "added_count": len(added_keys),
                "removed_count": len(removed_keys),
                "changed_count": len(changed),
                "unchanged_count": unchanged_count,
                "added_nodes": [_node_brief(after_nodes[key]) for key in added_keys[:64]],
                "removed_nodes": [_node_brief(before_nodes[key]) for key in removed_keys[:64]],
                "changed_nodes": changed[:64],
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


def _kernel_rows(kernel_dag: dict[str, Any]) -> str:
    nodes = kernel_dag.get("nodes") if isinstance(kernel_dag.get("nodes"), dict) else {}
    if not nodes:
        return '<tr><td colspan="6">No kernel DAG summary available.</td></tr>'
    rows = []
    for kernel_id in sorted(nodes):
        node = nodes[kernel_id] if isinstance(nodes[kernel_id], dict) else {}
        rows.append(
            "<tr>"
            f"<td>{_cell(kernel_id)}</td>"
            f"<td>{_cell(node.get('kind'))}</td>"
            f"<td>{_cell(node.get('depth'))}</td>"
            f"<td>{_cell(node.get('output_shape'))}</td>"
            f"<td>{_cell(node.get('selected_tile_shape'))}</td>"
            f"<td>{_cell(node.get('workspace_size'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


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
    return "\n".join(rows) or '<tr><td colspan="3">No artifacts recorded.</td></tr>'


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
        f"<dl><dt>Status</dt><dd>{_cell(tensor_diff.get('status', 'none'))}</dd>"
        f"<dt>Failed</dt><dd>{_cell(tensor_diff.get('failed_count', 0))}</dd></dl>"
        "</section>"
    )
    locate = overlays.get("locate", {})
    cards.append(
        "<section class=\"metric-card\">"
        "<h3>Locate</h3>"
        f"<dl><dt>Status</dt><dd>{_cell(locate.get('status', 'none'))}</dd>"
        f"<dt>First bad</dt><dd>{_cell(locate.get('first_bad_kernel', 'none'))}</dd></dl>"
        "</section>"
    )
    memory = overlays.get("memory", {})
    cards.append(
        "<section class=\"metric-card\">"
        "<h3>Memory</h3>"
        f"<dl><dt>Peak</dt><dd>{_cell(memory.get('peak_workspace_bytes', 0))}</dd>"
        f"<dt>Reuse groups</dt><dd>{_cell(memory.get('slot_reuse_group_count', 0))}</dd></dl>"
        "</section>"
    )
    return "\n".join(cards)


def _stage_buttons(debug_graph: dict[str, Any]) -> str:
    primary = debug_graph.get("primary_stage")
    active_name = primary.get("name") if isinstance(primary, dict) else None
    buttons = []
    for index, item in enumerate(debug_graph.get("stages", [])):
        active = " active" if item.get("name") == active_name else ""
        buttons.append(
            f'<button class="stage-button{active}" data-stage-index="{index}" type="button">'
            f'<span>{_cell(item.get("order"))}</span>{_cell(item.get("name"))}'
            "</button>"
        )
    return "\n".join(buttons)


def _artifact_index(debug_graph: dict[str, Any]) -> str:
    artifacts = debug_graph.get("artifacts", {})
    return f"""
<section id="artifact-index" class="artifact-index">
<h2>Artifact Index</h2>
<div class="artifact-grid">
<section>
<h3>Commands</h3>
<table><tbody>{_artifact_rows(artifacts.get('commands', []), path_key='stdout')}</tbody></table>
</section>
<section>
<h3>Reports</h3>
<table><tbody>{_artifact_rows(artifacts.get('reports', []))}</tbody></table>
</section>
<section>
<h3>Graphs</h3>
<table><tbody>{_artifact_rows(artifacts.get('graphs', []))}</tbody></table>
</section>
<section>
<h3>Summaries</h3>
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
}
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); }
a { color: var(--blue); text-decoration: none; }
header { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; padding: 0.9rem 1.1rem; background: var(--panel); border-bottom: 1px solid var(--line); }
h1 { margin: 0 0 0.35rem; font-size: 1.2rem; }
h2 { margin: 0 0 0.65rem; font-size: 0.98rem; }
h3 { margin: 0 0 0.45rem; font-size: 0.84rem; }
.toolbar { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; color: var(--muted); font-size: 0.86rem; }
.app-shell { display: grid; grid-template-columns: 15rem minmax(34rem, 1fr) 25rem; gap: 0.85rem; padding: 0.85rem; min-height: calc(100vh - 4.5rem); }
.sidebar, .graph-panel, .inspector-panel, .artifact-index { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
.sidebar { padding: 0.8rem; align-self: start; position: sticky; top: 0.85rem; }
.mode-tabs { display: grid; grid-template-columns: 1fr; gap: 0.35rem; margin-bottom: 0.9rem; }
.mode-tab, .stage-button { border: 1px solid var(--line); border-radius: 6px; background: #ffffff; color: var(--text); padding: 0.45rem 0.55rem; text-align: left; cursor: pointer; }
.mode-tab.active, .stage-button.active { border-color: #60a5fa; background: #eaf2ff; color: #1d4ed8; font-weight: 700; }
.stage-list { display: grid; gap: 0.32rem; max-height: 46vh; overflow: auto; }
.stage-button span { display: inline-block; min-width: 2.2rem; color: var(--muted); font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.stage-diff-panel { margin-top: 0.9rem; border: 1px solid #e3e8ef; border-radius: 7px; padding: 0.65rem; background: #fbfcfe; }
.diff-counts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.35rem; margin: 0.45rem 0; }
.diff-pill { border-radius: 6px; border: 1px solid var(--line); padding: 0.35rem; background: #fff; font-size: 0.78rem; }
.diff-pill strong { display: block; font-size: 0.98rem; }
.diff-added-text { color: #047857; }
.diff-changed-text { color: #b45309; }
.diff-removed-text { color: #b91c1c; }
.diff-list { margin: 0.4rem 0 0; padding-left: 1rem; color: #344054; font-size: 0.78rem; max-height: 7.5rem; overflow: auto; }
.overlay-stack { display: grid; gap: 0.55rem; margin-top: 0.9rem; }
.metric-card { border: 1px solid #e3e8ef; border-radius: 7px; padding: 0.6rem; background: #fbfcfe; }
.metric-card.fail { border-color: #fecaca; background: #fff7f7; }
.metric-card.warn { border-color: #fde68a; background: #fffbeb; }
dl { margin: 0; display: grid; grid-template-columns: 6.2rem 1fr; gap: 0.24rem 0.45rem; }
dt { color: var(--muted); font-weight: 700; }
dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.graph-panel { min-width: 0; overflow: hidden; }
.panel-header { display: flex; justify-content: space-between; gap: 1rem; align-items: center; padding: 0.7rem 0.85rem; border-bottom: 1px solid #e4e9f1; }
.panel-subtitle { color: var(--muted); font-size: 0.82rem; }
.graph-tools { display: grid; grid-template-columns: minmax(12rem, 20rem) auto auto auto minmax(7rem, auto); gap: 0.45rem; align-items: center; }
.graph-search { border: 1px solid var(--line); border-radius: 6px; padding: 0.42rem 0.55rem; min-width: 0; font: inherit; }
.graph-tool-button { border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 0.42rem 0.6rem; cursor: pointer; color: var(--text); }
.graph-tool-button:hover { border-color: #60a5fa; color: #1d4ed8; }
.zoom-value, .search-status { color: var(--muted); font-size: 0.78rem; white-space: nowrap; }
.graph-canvas-wrap { overflow: auto; height: calc(100vh - 11rem); background: #ffffff; cursor: grab; }
.graph-canvas-wrap.panning { cursor: grabbing; user-select: none; }
#unified-debug-graph-svg { display: block; min-width: 100%; }
.graph-edge-path { fill: none; stroke: #8a95a6; stroke-width: 1.5; marker-end: url(#arrow-head); }
.graph-edge-path.critical { stroke: var(--red); stroke-width: 2.7; }
.graph-edge-path.fusion { stroke: #16a34a; stroke-width: 2.5; }
.graph-edge-label { fill: #586274; font-size: 11px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.graph-node, .kernel-dag-node { cursor: pointer; outline: none; }
.graph-node rect, .kernel-dag-node rect { fill: #ffffff; stroke: #95a1b2; stroke-width: 1.4; }
.graph-node.kernel-node rect { fill: #eef6ff; stroke: var(--blue); }
.kernel-dag-node.vec rect { fill: #ecfdf5; stroke: #10b981; }
.kernel-dag-node.cube rect { fill: #fff7ed; stroke: #f97316; }
.kernel-dag-node.mix rect { fill: #f5f3ff; stroke: #7c3aed; }
.graph-node.issue rect, .kernel-dag-node.issue rect { stroke: var(--red); stroke-width: 3; }
.graph-node.diff-added rect { fill: #ecfdf5; stroke: #059669; stroke-width: 2.3; }
.graph-node.diff-changed rect { fill: #fffbeb; stroke: #d97706; stroke-width: 2.3; }
.graph-node.search-match rect, .kernel-dag-node.search-match rect { filter: drop-shadow(0 0 0.35rem rgba(37, 99, 235, 0.45)); stroke: #2563eb; stroke-width: 3; }
.graph-node:hover rect, .graph-node.selected rect, .kernel-dag-node:hover rect, .kernel-dag-node.selected rect { stroke: var(--teal); stroke-width: 2.6; }
.node-op { fill: var(--text); font-size: 13px; font-weight: 700; }
.node-result { fill: #344054; font-size: 12px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.node-inputs { fill: #667085; font-size: 11px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }
.node-kernel { fill: var(--blue); font-size: 10px; text-anchor: end; }
.inspector-panel { min-width: 0; padding: 0.8rem; align-self: start; position: sticky; top: 0.85rem; max-height: calc(100vh - 1.7rem); overflow: auto; }
.inspector-actions { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-bottom: 0.6rem; }
.chip { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 0.18rem 0.45rem; color: #344054; background: #ffffff; font-size: 0.78rem; }
pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; background: #0b1020; color: #dbeafe; border: 1px solid #1e293b; padding: 0.75rem; border-radius: 7px; font: 12px/1.45 SFMono-Regular, Menlo, Consolas, monospace; }
.artifact-index { grid-column: 1 / -1; padding: 0.85rem; }
.artifact-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 0.75rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th, td { border: 1px solid #d7dde7; padding: 0.34rem 0.42rem; text-align: left; vertical-align: top; }
th { background: #f2f5f9; }
@media (max-width: 1180px) {
  .app-shell { grid-template-columns: 1fr; }
  .sidebar, .inspector-panel { position: static; max-height: none; }
  .graph-canvas-wrap { height: 62vh; }
  .graph-tools { grid-template-columns: 1fr 1fr; }
}
</style>
"""
    script = """
<script>
const workspace = JSON.parse(document.getElementById("graph-workspace-data").textContent);
let activeMode = "stage";
let activeStageIndex = Math.max(0, workspace.stages.findIndex((stage) => workspace.primary_stage && stage.name === workspace.primary_stage.name));
let selectedKey = null;
let canvasPanState = null;
let graphViewState = {scale: 1};
let activeSearchResults = [];
let activeSearchIndex = 0;
const MIN_GRAPH_SCALE = 0.2;
const MAX_GRAPH_SCALE = 3;

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

function setInspector(title, payload, links = []) {
  document.getElementById("inspector-title").textContent = title;
  const actions = document.getElementById("inspector-actions");
  actions.innerHTML = links.map((link) => `<a class="chip" href="${escapeHtml(link.href)}">${escapeHtml(link.label)}</a>`).join("");
  document.getElementById("inspector-json").textContent = JSON.stringify(payload, null, 2);
}

function svgHeader(width, height) {
  return `<svg id="unified-debug-graph-svg" data-base-width="${width}" data-base-height="${height}" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
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

function resetGraphView() {
  const canvas = graphCanvas();
  graphViewState.scale = 1;
  applyGraphScale();
  canvas.scrollLeft = 0;
  canvas.scrollTop = 0;
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
  canvas.scrollLeft = 0;
  canvas.scrollTop = 0;
}

function zoomGraphAt(event) {
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
  canvas.scrollLeft = Math.max(0, (position.x + width / 2) * scale - canvas.clientWidth / 2);
  canvas.scrollTop = Math.max(0, (position.y + height / 2) * scale - canvas.clientHeight / 2);
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

function activeStage() {
  return workspace.stages[activeStageIndex] || workspace.stages[0] || null;
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
    summary.innerHTML = '<span class="panel-subtitle">Kernel DAG mode has no adjacent stage diff.</span>';
    details.innerHTML = "";
    return;
  }
  const diff = activeStageDiff(stage);
  if (!diff) {
    summary.innerHTML = '<span class="panel-subtitle">Source stage has no predecessor.</span>';
    details.innerHTML = "";
    return;
  }
  summary.innerHTML = `
<div class="panel-subtitle">${escapeHtml(diff.from_stage.order)} ${escapeHtml(diff.from_stage.name)} -> ${escapeHtml(diff.to_stage.order)} ${escapeHtml(diff.to_stage.name)}</div>
<div class="diff-counts">
<div class="diff-pill diff-added-text"><strong>${escapeHtml(diff.added_count)}</strong>added</div>
<div class="diff-pill diff-changed-text"><strong>${escapeHtml(diff.changed_count)}</strong>changed</div>
<div class="diff-pill diff-removed-text"><strong>${escapeHtml(diff.removed_count)}</strong>removed</div>
</div>`;
  const changed = (diff.changed_nodes || []).slice(0, 8).map((node) => {
    const after = node.after || {};
    const fields = (node.changes || []).map((change) => change.field).join(", ");
    return `<li><span class="diff-changed-text">${escapeHtml(after.op_name || after.label || "node")}</span>: ${escapeHtml(fields || "metadata")}</li>`;
  });
  const added = (diff.added_nodes || []).slice(0, 5).map((node) => (
    `<li><span class="diff-added-text">${escapeHtml(node.op_name || node.label || "node")}</span>: added</li>`
  ));
  const removed = (diff.removed_nodes || []).slice(0, 5).map((node) => (
    `<li><span class="diff-removed-text">${escapeHtml(node.op_name || node.label || "node")}</span>: removed from current stage</li>`
  ));
  const rows = [...changed, ...added, ...removed];
  details.innerHTML = rows.length ? `<ul class="diff-list">${rows.join("")}</ul>` : '<span class="panel-subtitle">No semantic node changes detected.</span>';
}

function stageNodeDiffInfo(stage, nodeId) {
  const diff = activeStageDiff(stage);
  return diff && diff.node_status ? diff.node_status[nodeId] : null;
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
    const graph = stage ? stage.graph : null;
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
  setSearchStatus(activeSearchResults.length ? `${activeSearchResults.length} match(es), Enter cycles` : "No match");
  if (activeSearchResults.length) activeSearchResults[0].select();
}

function afterGraphRender() {
  applyGraphScale();
  renderStageDiff();
  searchActiveGraph();
}

function renderStageGraph() {
  const stage = activeStage();
  const graph = stage ? stage.graph : null;
  const canvas = document.getElementById("graph-canvas");
  if (!graph || !graph.layout) {
    canvas.innerHTML = `${svgHeader(720, 420)}<text x="28" y="42">No stage graph available.</text></svg>`;
    afterGraphRender();
    return;
  }
  document.getElementById("graph-title").textContent = `Stage Graph: ${stage.order} ${stage.name}`;
  document.getElementById("graph-subtitle").textContent = `${graph.node_count} nodes, ${graph.edge_count} edges, ${graph.kernel_count} kernels`;
  const layout = graph.layout;
  const nodeById = Object.fromEntries(graph.nodes.map((node, index) => [node.id, {node, index}]));
  const diff = activeStageDiff(stage);
  const firstBad = workspace.overlays.locate && workspace.overlays.locate.first_bad_kernel;
  let svg = svgHeader(layout.width || 720, layout.height || 420);
  for (const edge of layout.edges || []) {
    svg += `<g class="graph-edge"><path class="graph-edge-path" d="${escapeHtml(edge.path)}"></path><text class="graph-edge-label" x="${edge.label_x}" y="${edge.label_y}">${escapeHtml(truncate(edge.value, 24))}</text></g>`;
  }
  for (const node of graph.nodes) {
    const position = layout.nodes[node.id];
    if (!position) continue;
    const result = node.result_values && node.result_values.length ? node.result_values.join(", ") : node.label;
    const inputs = node.input_values && node.input_values.length ? node.input_values.join(", ") : "root";
    const issueClass = firstBad && node.kernel_id === firstBad ? " issue" : "";
    const kernelClass = node.kernel_id ? " kernel-node" : "";
    const diffInfo = diff && diff.node_status ? diff.node_status[node.id] : null;
    const diffClass = diffInfo && diffInfo.status !== "unchanged" ? ` diff-${diffInfo.status}` : "";
    const kernelText = node.kernel_id ? `kernel ${truncate(node.kernel_id, 22)}` : `line ${node.line}`;
    svg += `<g class="graph-node${kernelClass}${issueClass}${diffClass}" data-node-id="${escapeHtml(node.id)}" data-node-index="${nodeById[node.id].index}" data-width="${position.width}" data-height="${position.height}" tabindex="0" role="button" transform="translate(${position.x},${position.y})">
<title>${escapeHtml(node.op_name)}</title>
<rect width="${position.width}" height="${position.height}" rx="6"></rect>
<text class="node-op" x="14" y="24">${escapeHtml(truncate(node.op_name, 28))}</text>
<text class="node-result" x="14" y="47">${escapeHtml(truncate(result, 30))}</text>
<text class="node-inputs" x="14" y="68">in: ${escapeHtml(truncate(inputs, 30))}</text>
<text class="node-kernel" x="${position.width - 14}" y="22">${escapeHtml(kernelText)}</text>
</g>`;
  }
  svg += "</svg>";
  canvas.innerHTML = svg;
  document.querySelectorAll(".graph-node").forEach((element) => {
    element.addEventListener("click", () => selectStageNode(stage, graph, element.dataset.nodeId));
    element.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectStageNode(stage, graph, element.dataset.nodeId);
      }
    });
  });
  const preferred = selectedKey && graph.nodes.some((node) => node.id === selectedKey) ? selectedKey : (graph.nodes[0] && graph.nodes[0].id);
  if (preferred) selectStageNode(stage, graph, preferred);
  afterGraphRender();
}

function selectStageNode(stage, graph, nodeId, options = {}) {
  selectedKey = nodeId;
  document.querySelectorAll(".graph-node").forEach((element) => element.classList.toggle("selected", element.dataset.nodeId === nodeId));
  const node = graph.nodes.find((item) => item.id === nodeId);
  const links = [{label: "Stage MLIR", href: `../${stage.path}`}];
  if (stage.graph_view_path) links.push({label: "Stage View", href: `../${stage.graph_view_path}`});
  if (node && node.kernel_id) links.push({label: "Kernel View", href: `kernels/${node.kernel_id}.html`});
  if (node && node.kernel_id && memoryHasKernel(node.kernel_id)) links.push({label: "Memory View", href: memoryViewLink(node.kernel_id)});
  const diff = node ? stageNodeDiffInfo(stage, nodeId) : null;
  setInspector(node ? `${node.op_name} ${node.label || ""}` : "Stage Node", {stage: {order: stage.order, name: stage.name}, diff, node}, links);
  if (options.center) {
    centerGraphElement(findStageNodeElement(nodeId));
  }
}

function renderKernelDag() {
  const summary = workspace.kernel_dag || {};
  const nodes = summary.nodes || {};
  const edges = summary.edges || [];
  const ids = Object.keys(nodes).sort((a, b) => (nodes[a].depth || 0) - (nodes[b].depth || 0) || a.localeCompare(b));
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
  svg += "</svg>";
  document.getElementById("graph-title").textContent = "Kernel DAG";
  document.getElementById("graph-subtitle").textContent = `${summary.kernel_count || 0} kernels, ${summary.graph_edges || 0} edges, critical depth ${summary.critical_path_depth || 0}`;
  document.getElementById("graph-canvas").innerHTML = svg;
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
  const links = [{label: "Kernel View", href: `kernels/${kernelId}.html`}];
  if (memoryHasKernel(kernelId)) links.push({label: "Memory View", href: memoryViewLink(kernelId)});
  setInspector(`Kernel ${kernelId}`, {kernel_id: kernelId, ...node}, links);
  if (options.center) {
    centerGraphElement(findKernelNodeElement(kernelId));
  }
}

function setMode(mode) {
  activeMode = mode;
  selectedKey = null;
  document.querySelectorAll(".mode-tab").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  if (mode === "kernel") renderKernelDag();
  else renderStageGraph();
}

document.querySelectorAll(".mode-tab").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
document.querySelectorAll(".stage-button").forEach((button) => button.addEventListener("click", () => {
  activeStageIndex = Number(button.dataset.stageIndex);
  document.querySelectorAll(".stage-button").forEach((item) => item.classList.toggle("active", item === button));
  setMode("stage");
}));
installCanvasPan();
installGraphNavigation();
renderStageGraph();
</script>
"""
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ascend Debug Graph</title>
{style}
</head>
<body>
<header>
<div>
<h1>Ascend Debug Graph</h1>
<div class="toolbar">
<a href="../index.html">Dashboard</a>
<a href="../{_cell(summary_path)}">Raw debug_graph.json</a>
<span>Primary stage: {_cell(primary_stage.get('name') if primary_stage else 'none')}</span>
<span>Kernel DAG: {_cell(kernel_dag.get('kernel_count', 0))} kernels</span>
</div>
</div>
</header>
<main class="app-shell">
<aside class="sidebar">
<h2>Graph Mode</h2>
<div class="mode-tabs">
<button class="mode-tab active" data-mode="stage" type="button">Stage Evolution</button>
<button class="mode-tab" data-mode="kernel" type="button">Kernel DAG</button>
</div>
<h2>Stages</h2>
<div id="stage-list" class="stage-list">
{_stage_buttons(debug_graph)}
</div>
<section id="stage-diff-panel" class="stage-diff-panel">
<h2>Stage Diff</h2>
<div id="stage-diff-summary"></div>
<div id="stage-diff-details"></div>
</section>
<div class="overlay-stack">
{_overlay_cards(overlays)}
</div>
</aside>
<section class="graph-panel">
<div class="panel-header">
<div>
<h2 id="graph-title">Unified Stage Graph</h2>
<div id="graph-subtitle" class="panel-subtitle"></div>
</div>
<div class="graph-tools">
<input id="graph-search" class="graph-search" type="search" placeholder="Search op, kernel, loc">
<button id="graph-fit" class="graph-tool-button" type="button">Fit</button>
<button id="graph-reset" class="graph-tool-button" type="button">Reset</button>
<span id="graph-zoom-value" class="zoom-value">100%</span>
<span id="graph-search-status" class="search-status"></span>
</div>
</div>
<div id="graph-canvas" class="graph-canvas-wrap">
<svg id="unified-debug-graph-svg" width="720" height="420" viewBox="0 0 720 420" xmlns="http://www.w3.org/2000/svg"></svg>
</div>
</section>
<aside class="inspector-panel">
<h2 id="inspector-title">Inspector</h2>
<div id="inspector-actions" class="inspector-actions"></div>
<pre id="inspector-json"></pre>
<h2>Overlays</h2>
{_overlay_cards(overlays)}
<h2>Kernel DAG</h2>
<table>
<thead><tr><th>Kernel</th><th>Kind</th><th>Depth</th><th>Shape</th><th>Tile</th><th>Workspace</th></tr></thead>
<tbody>{_kernel_rows(kernel_dag)}</tbody>
</table>
</aside>
{_artifact_index(debug_graph)}
</main>
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
    primary_stage = _select_primary_stage(stages)
    stage_diffs = _compute_stage_diffs(stages)
    summary = {
        "schema_version": 1,
        "tool": "ascend-debug",
        "visual_kind": "unified-debug-workspace",
        "stage_count": len(stages),
        "primary_stage": primary_stage,
        "stage_diffs": stage_diffs,
        "stages": stages,
        "kernel_dag": kernel_summary or {},
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
