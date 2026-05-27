from __future__ import annotations

import html
import pathlib
import re
from typing import Any

from ascend_debug import layout


SSA_VALUE_RE = re.compile(r"%[A-Za-z0-9_.$-]+")
FUNC_RE = re.compile(r"func\.func\s+@(?P<name>[A-Za-z0-9_.$-]+)\((?P<args>[^)]*)\)")
OP_RE = re.compile(
    r"^\s*(?P<results>%[A-Za-z0-9_.$-]+(?:\s*,\s*%[A-Za-z0-9_.$-]+)*)\s*=\s*(?P<op>[A-Za-z_][A-Za-z0-9_.]*)"
)
RETURN_RE = re.compile(r"^\s*return\b")
BODY_OP_RE = re.compile(
    r"^\s*(?:%[A-Za-z0-9_.$-]+(?:\s*,\s*%[A-Za-z0-9_.$-]+)*\s*=\s*)?"
    r"(?P<op>[A-Za-z_][A-Za-z0-9_.]*)\b"
)


def _cell(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _rel_no_ext(rel_path: str) -> str:
    path = pathlib.PurePosixPath(rel_path)
    return f"{path.stem}"


def _stage_graph_rel_paths(stage_rel_path: str) -> tuple[str, str]:
    stem = _rel_no_ext(stage_rel_path)
    return (
        f"graphs/stages/{stem}.graph.json",
        f"views/graphs/stages/{stem}.graph.html",
    )


def _href(from_rel_path: str, to_rel_path: str) -> str:
    import posixpath

    source_dir = pathlib.PurePosixPath(from_rel_path).parent
    return posixpath.relpath(to_rel_path, start=str(source_dir))


def _parse_func_args(line: str) -> list[dict[str, Any]]:
    match = FUNC_RE.search(line)
    if not match:
        return []
    args = []
    for arg_match in re.finditer(r"(?P<name>%[A-Za-z0-9_.$-]+)\s*:\s*(?P<type>[^,\)]+)", match.group("args")):
        args.append({"name": arg_match.group("name"), "type": arg_match.group("type").strip()})
    return args


def _extract_attr(text: str, name: str) -> str | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*\"([^\"]+)\"", text)
    return match.group(1) if match else None


def _extract_int_attr(text: str, name: str) -> int | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*([0-9]+)", text)
    return int(match.group(1)) if match else None


def _extract_result_type(text: str) -> str | None:
    arrow_match = re.search(r"->\s*([^\n{]+)", text)
    if arrow_match:
        return arrow_match.group(1).strip()
    colon_match = re.search(r":\s*([^\n]+)$", text.strip())
    return colon_match.group(1).strip() if colon_match else None


def _collect_op_text(lines: list[str], index: int, op_name: str) -> tuple[str, int]:
    collected = [lines[index]]
    if op_name.startswith("linalg.") and op_name != "linalg.yield":
        cursor = index + 1
        while cursor < len(lines):
            collected.append(lines[cursor])
            if "} ->" in lines[cursor]:
                cursor += 1
                break
            cursor += 1
        return "\n".join(collected), cursor
    return lines[index], index + 1


def _extract_region_body(op_text: str) -> str | None:
    body_lines: list[str] = []
    in_region = False
    for line in op_text.splitlines()[1:]:
        stripped = line.strip()
        if stripped.startswith("}"):
            break
        if stripped.startswith("^bb"):
            in_region = True
        if in_region and stripped:
            body_lines.append(stripped)
    return "\n".join(body_lines) if body_lines else None


def _extract_body_ops(region_body: str | None) -> list[str]:
    if not region_body:
        return []
    body_ops: list[str] = []
    for line in region_body.splitlines():
        if line.strip().startswith("^"):
            continue
        match = BODY_OP_RE.match(line)
        if match:
            body_ops.append(match.group("op"))
    return body_ops


def _summarize_body_ops(body_ops: list[str]) -> str | None:
    compute_ops = [op for op in body_ops if op != "linalg.yield"]
    if compute_ops:
        return " -> ".join(compute_ops)
    if body_ops:
        return " -> ".join(body_ops)
    return None


def parse_stage_mlir(stage: dict[str, Any], text: str) -> dict[str, Any]:
    lines = text.splitlines()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    producer_by_value: dict[str, str] = {}
    defined_values: set[str] = set()
    function_name = None

    for line_number, line in enumerate(lines, start=1):
        func_match = FUNC_RE.search(line)
        if not func_match:
            continue
        function_name = func_match.group("name")
        for arg in _parse_func_args(line):
            node_id = f"n{len(nodes)}"
            node = {
                "id": node_id,
                "line": line_number,
                "op_name": "func.arg",
                "label": arg["name"],
                "input_values": [],
                "result_values": [arg["name"]],
                "result_type": arg["type"],
                "kernel_id": None,
                "op_role": None,
                "schedule_decision_id": None,
                "workspace_size_bytes": None,
            }
            nodes.append(node)
            producer_by_value[arg["name"]] = node_id
            defined_values.add(arg["name"])

    index = 0
    while index < len(lines):
        line = lines[index]
        op_match = OP_RE.match(line)
        return_match = RETURN_RE.match(line)
        if not op_match and not return_match:
            index += 1
            continue

        if op_match:
            op_name = op_match.group("op")
            op_text, next_index = _collect_op_text(lines, index, op_name)
            result_values = [value.strip() for value in op_match.group("results").split(",")]
            raw_values = SSA_VALUE_RE.findall(op_text)
            input_values = []
            for value in raw_values:
                if value in result_values or value not in defined_values:
                    continue
                if value not in input_values:
                    input_values.append(value)
        else:
            op_name = "func.return"
            op_text = line
            next_index = index + 1
            result_values = []
            input_values = []
            for value in SSA_VALUE_RE.findall(line):
                if value in defined_values and value not in input_values:
                    input_values.append(value)

        node_id = f"n{len(nodes)}"
        region_body = _extract_region_body(op_text)
        body_ops = _extract_body_ops(region_body)
        node = {
            "id": node_id,
            "line": index + 1,
            "line_end": next_index,
            "op_name": op_name,
            "label": result_values[0] if result_values else op_name,
            "input_values": input_values,
            "result_values": result_values,
            "result_type": _extract_result_type(op_text),
            "source_excerpt": op_text,
            "region_body": region_body,
            "body_ops": body_ops,
            "body_summary": _summarize_body_ops(body_ops),
            "kernel_id": _extract_attr(op_text, "ascend.kernel"),
            "op_role": _extract_attr(op_text, "ascend.op_role"),
            "schedule_decision_id": _extract_attr(op_text, "ascend.schedule.decision_id"),
            "workspace_size_bytes": _extract_int_attr(op_text, "cann.workspace_size_bytes"),
        }
        nodes.append(node)

        for value in input_values:
            producer = producer_by_value.get(value)
            if producer:
                edges.append(
                    {
                        "id": f"e{len(edges)}",
                        "from": producer,
                        "to": node_id,
                        "value": value,
                    }
                )
        for value in result_values:
            producer_by_value[value] = node_id
            defined_values.add(value)
        index = next_index

    kernel_ids = sorted(
        {node["kernel_id"] for node in nodes if isinstance(node.get("kernel_id"), str)}
    )
    return {
        "schema_version": 1,
        "tool": "ascend-debug",
        "stage": {
            "order": stage["order"],
            "name": stage["name"],
            "path": stage["path"],
        },
        "function": function_name,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "kernel_count": len(kernel_ids),
        "kernels": [
            {
                "kernel_id": kernel_id,
                "node_ids": [node["id"] for node in nodes if node.get("kernel_id") == kernel_id],
            }
            for kernel_id in kernel_ids
        ],
        "nodes": nodes,
        "edges": edges,
    }


def _edge_rows(graph: dict[str, Any]) -> str:
    node_labels = {node["id"]: node.get("label") or node.get("op_name") for node in graph["nodes"]}
    rows = []
    for edge in graph["edges"]:
        rows.append(
            "<tr>"
            f"<td>{_cell(edge.get('value'))}</td>"
            f"<td>{_cell(node_labels.get(edge.get('from')))}</td>"
            f"<td>{_cell(node_labels.get(edge.get('to')))}</td>"
            "</tr>"
        )
    return "\n".join(rows) or '<tr><td colspan="3">No data-flow edges detected.</td></tr>'


def _kernel_rows(graph: dict[str, Any]) -> str:
    rows = []
    for kernel in graph.get("kernels", []):
        rows.append(
            "<tr>"
            f"<td>{_cell(kernel.get('kernel_id'))}</td>"
            f"<td>{_cell(len(kernel.get('node_ids', [])))}</td>"
            f"<td>{_cell(', '.join(kernel.get('node_ids', [])))}</td>"
            "</tr>"
        )
    return "\n".join(rows) or '<tr><td colspan="3">No kernel boundary in this stage.</td></tr>'


def _truncate(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def _compute_graph_layout(graph: dict[str, Any]) -> dict[str, Any]:
    node_width = 220
    node_height = 82
    column_gap = 72
    layer_gap = 86
    margin_x = 36
    margin_y = 36
    predecessor_ids: dict[str, list[str]] = {
        node["id"]: []
        for node in graph["nodes"]
    }
    for edge in graph["edges"]:
        predecessor_ids.setdefault(edge["to"], []).append(edge["from"])

    layer_by_id: dict[str, int] = {}
    layers: dict[int, list[str]] = {}
    for node in graph["nodes"]:
        node_id = node["id"]
        predecessors = [
            layer_by_id[pred] + 1
            for pred in predecessor_ids.get(node_id, [])
            if pred in layer_by_id
        ]
        layer = max(predecessors) if predecessors else 0
        layer_by_id[node_id] = layer
        layers.setdefault(layer, []).append(node_id)

    node_layout: dict[str, dict[str, Any]] = {}
    max_x = margin_x
    max_y = margin_y
    for layer in sorted(layers):
        for row, node_id in enumerate(layers[layer]):
            x = margin_x + row * (node_width + column_gap)
            y = margin_y + layer * (node_height + layer_gap)
            node_layout[node_id] = {
                "x": x,
                "y": y,
                "width": node_width,
                "height": node_height,
                "layer": layer,
                "row": row,
            }
            max_x = max(max_x, x + node_width)
            max_y = max(max_y, y + node_height)

    edge_layout = []
    for edge in graph["edges"]:
        source = node_layout.get(edge["from"])
        target = node_layout.get(edge["to"])
        if not source or not target:
            continue
        source_x = source["x"] + source["width"] / 2
        source_y = source["y"] + source["height"]
        target_x = target["x"] + target["width"] / 2
        target_y = target["y"]
        bend = max(42, abs(target_y - source_y) / 2)
        path = (
            f"M {source_x:.1f} {source_y:.1f} "
            f"C {source_x:.1f} {source_y + bend:.1f}, "
            f"{target_x:.1f} {target_y - bend:.1f}, "
            f"{target_x:.1f} {target_y:.1f}"
        )
        edge_layout.append(
            {
                "id": edge["id"],
                "from": edge["from"],
                "to": edge["to"],
                "value": edge["value"],
                "path": path,
                "label_x": (source_x + target_x) / 2,
                "label_y": (source_y + target_y) / 2 - 8,
            }
        )

    return {
        "visual_kind": "svg-dag",
        "direction": "top-to-bottom",
        "node_width": node_width,
        "node_height": node_height,
        "width": max(720, max_x + margin_x),
        "height": max(420, max_y + margin_y),
        "nodes": node_layout,
        "edges": edge_layout,
    }


def render_svg_graph(
    graph: dict[str, Any],
    *,
    svg_id: str = "stage-graph-svg",
    canvas_class: str = "graph-canvas",
) -> str:
    graph_layout = graph.get("layout", {})
    node_layout = graph_layout.get("nodes", {})
    edge_layout = graph_layout.get("edges", [])
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    node_index_by_id = {
        node["id"]: index
        for index, node in enumerate(graph["nodes"])
    }
    edge_elements = []
    for edge in edge_layout:
        edge_elements.append(
            '<g class="graph-edge">'
            f'<path class="graph-edge-path" d="{_cell(edge.get("path"))}" />'
            f'<text class="graph-edge-label" x="{_cell(edge.get("label_x"))}" '
            f'y="{_cell(edge.get("label_y"))}">{_cell(_truncate(edge.get("value"), 24))}</text>'
            "</g>"
        )

    node_elements = []
    for node_id, position in node_layout.items():
        node = node_by_id.get(node_id)
        if not node:
            continue
        index = node_index_by_id[node_id]
        result = ", ".join(node.get("result_values", [])) or node.get("label")
        inputs = ", ".join(node.get("input_values", [])) or "root"
        detail = f"body: {node.get('body_summary')}" if node.get("body_summary") else f"in: {inputs}"
        kernel = node.get("kernel_id")
        classes = "graph-node kernel-node" if kernel else "graph-node"
        kernel_text = f"kernel {_truncate(kernel, 22)}" if kernel else f"line {node.get('line')}"
        node_elements.append(
            f'<g class="{classes}" data-node-index="{index}" tabindex="0" role="button" '
            f'transform="translate({_cell(position.get("x"))},{_cell(position.get("y"))})">'
            f'<title>{_cell(node.get("op_name"))}</title>'
            f'<rect width="{_cell(position.get("width"))}" '
            f'height="{_cell(position.get("height"))}" rx="6" />'
            f'<text class="node-op" x="14" y="24">{_cell(_truncate(node.get("op_name"), 28))}</text>'
            f'<text class="node-result" x="14" y="47">{_cell(_truncate(result, 30))}</text>'
            f'<text class="node-inputs" x="14" y="68">{_cell(_truncate(detail, 30))}</text>'
            f'<text class="node-kernel" x="206" y="22">{_cell(kernel_text)}</text>'
            "</g>"
        )

    if not node_elements:
        node_elements.append('<text x="24" y="42">No graph nodes detected.</text>')

    return f"""
<div class="{_cell(canvas_class)}" aria-label="Stage data-flow graph">
<svg id="{_cell(svg_id)}" width="{_cell(graph_layout.get('width', 720))}"
     height="{_cell(graph_layout.get('height', 420))}"
     viewBox="0 0 {_cell(graph_layout.get('width', 720))} {_cell(graph_layout.get('height', 420))}"
     xmlns="http://www.w3.org/2000/svg">
<defs>
<marker id="arrow-head" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" />
</marker>
</defs>
{''.join(edge_elements)}
{''.join(node_elements)}
</svg>
</div>
"""


def _render_stage_graph_html(
    *,
    run_dir: pathlib.Path,
    graph: dict[str, Any],
    view_rel_path: str,
    raw_mlir_rel_path: str,
    stage_links: list[dict[str, str]],
) -> None:
    dashboard_href = html.escape(_href(view_rel_path, "index.html"), quote=True)
    raw_href = html.escape(_href(view_rel_path, raw_mlir_rel_path), quote=True)
    graph_json_rel, _ = _stage_graph_rel_paths(raw_mlir_rel_path)
    json_href = html.escape(_href(view_rel_path, graph_json_rel), quote=True)
    timeline_links = []
    current_stage = graph["stage"]["path"]
    for link in stage_links:
        active = " active" if link["stage_path"] == current_stage else ""
        href = html.escape(_href(view_rel_path, link["view_rel_path"]), quote=True)
        timeline_links.append(
            f'<a class="stage-link{active}" href="{href}">{_cell(link["label"])}</a>'
        )
    svg_graph = render_svg_graph(graph)
    nodes_json = layout.json_script_payload(graph["nodes"])
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_cell(graph['stage']['name'])} Stage Graph - ascend-debug</title>
<style>
body {{ margin: 0; font-family: sans-serif; color: #17202a; background: #f8fafc; }}
header {{ padding: 0.85rem 1rem; background: #ffffff; border-bottom: 1px solid #cbd5e1; position: sticky; top: 0; z-index: 2; }}
h1 {{ margin: 0 0 0.35rem; font-size: 1rem; }}
.toolbar {{ display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; }}
.shell {{ display: grid; grid-template-columns: 13rem minmax(28rem, 1fr) 22rem; min-height: calc(100vh - 4.25rem); }}
.timeline {{ border-right: 1px solid #cbd5e1; background: #ffffff; padding: 0.85rem; }}
.stage-link {{ display: block; padding: 0.45rem 0.55rem; border-radius: 5px; color: #334155; text-decoration: none; }}
.stage-link.active {{ background: #dbeafe; color: #1d4ed8; font-weight: 700; }}
.canvas {{ padding: 1rem; overflow: auto; }}
.summary {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; }}
.badge {{ display: inline-block; margin-left: 0.4rem; padding: 0.08rem 0.35rem; border: 1px solid #cbd5e1; border-radius: 999px; background: #f8fafc; color: #475569; font-size: 0.75rem; }}
.badge.kernel {{ border-color: #93c5fd; background: #eff6ff; color: #1d4ed8; }}
.graph-canvas {{ width: 100%; overflow: auto; border: 1px solid #cbd5e1; border-radius: 6px; background: #ffffff; }}
#stage-graph-svg {{ display: block; min-width: 100%; }}
.graph-edge-path {{ fill: none; stroke: #64748b; stroke-width: 1.5; marker-end: url(#arrow-head); }}
.graph-edge-label {{ fill: #475569; font-size: 11px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }}
.graph-node {{ cursor: pointer; outline: none; }}
.graph-node rect {{ fill: #ffffff; stroke: #94a3b8; stroke-width: 1.4; }}
.graph-node.kernel-node rect {{ fill: #eff6ff; stroke: #2563eb; }}
.graph-node:hover rect, .graph-node.selected rect {{ stroke: #0f766e; stroke-width: 2.4; }}
.node-op {{ fill: #17202a; font-size: 13px; font-weight: 700; }}
.node-result {{ fill: #334155; font-size: 12px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }}
.node-inputs {{ fill: #64748b; font-size: 11px; font-family: SFMono-Regular, Menlo, Consolas, monospace; }}
.node-kernel {{ fill: #1d4ed8; font-size: 10px; text-anchor: end; }}
.inspector {{ border-left: 1px solid #cbd5e1; background: #ffffff; padding: 0.85rem; overflow: auto; }}
table {{ width: 100%; border-collapse: collapse; background: #ffffff; margin-top: 0.75rem; }}
th, td {{ border: 1px solid #cbd5e1; padding: 0.35rem 0.45rem; text-align: left; vertical-align: top; }}
th {{ background: #f1f5f9; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #0b1020; color: #dbeafe; border: 1px solid #1e293b; padding: 0.75rem; border-radius: 6px; }}
</style>
</head>
<body>
<header>
<h1>Stage Graph: {_cell(graph['stage']['name'])}</h1>
<div class="toolbar">
<a href="{dashboard_href}">Dashboard</a>
<a href="{raw_href}">Raw MLIR</a>
<a href="{json_href}">Graph JSON</a>
</div>
</header>
<main class="shell">
<aside class="timeline">
<strong>Stages</strong>
{''.join(timeline_links)}
</aside>
<section class="canvas">
<h2>Stage Graph</h2>
<div class="summary">
<span class="badge">nodes {_cell(graph['node_count'])}</span>
<span class="badge">edges {_cell(graph['edge_count'])}</span>
<span class="badge">kernels {_cell(graph['kernel_count'])}</span>
</div>
{svg_graph}
<h2>Kernel boundary</h2>
<table>
<thead><tr><th>Kernel</th><th>Nodes</th><th>Node ids</th></tr></thead>
<tbody>{_kernel_rows(graph)}</tbody>
</table>
<h2>Data edges</h2>
<table>
<thead><tr><th>Value</th><th>Producer</th><th>Consumer</th></tr></thead>
<tbody>{_edge_rows(graph)}</tbody>
</table>
</section>
<aside class="inspector">
<h2>Inspector</h2>
<p>Select a graph node to inspect compiler-visible facts.</p>
<pre id="inspector-json"></pre>
</aside>
</main>
<script type="application/json" id="nodes-json">{nodes_json}</script>
<script>
const nodes = JSON.parse(document.getElementById("nodes-json").textContent);
const inspector = document.getElementById("inspector-json");
const graphNodes = Array.from(document.querySelectorAll(".graph-node"));
function selectNode(index) {{
  graphNodes.forEach((node) => {{
    node.classList.toggle("selected", Number(node.dataset.nodeIndex) === index);
  }});
  inspector.textContent = JSON.stringify(nodes[index], null, 2);
}}
graphNodes.forEach((node) => {{
  const index = Number(node.dataset.nodeIndex);
  node.addEventListener("click", () => selectNode(index));
  node.addEventListener("keydown", (event) => {{
    if (event.key === "Enter" || event.key === " ") {{
      event.preventDefault();
      selectNode(index);
    }}
  }});
}});
if (graphNodes.length) selectNode(Number(graphNodes[0].dataset.nodeIndex));
</script>
</body>
</html>
"""
    layout.write_text(run_dir / view_rel_path, document)


def render_stage_graphs(run_dir: pathlib.Path, stages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    graph_views: dict[str, dict[str, Any]] = {}
    for stage in sorted(stages, key=lambda item: item["order"]):
        rel_path = stage["path"]
        source_path = run_dir / rel_path
        if not source_path.exists():
            continue
        text = source_path.read_text(encoding="utf-8", errors="replace")
        graph = parse_stage_mlir(stage, text)
        graph["layout"] = _compute_graph_layout(graph)
        json_rel_path, _ = _stage_graph_rel_paths(rel_path)
        layout.write_json(run_dir / json_rel_path, graph)
        graph_views[rel_path] = {
            "json_rel_path": json_rel_path,
            "node_count": graph["node_count"],
            "edge_count": graph["edge_count"],
            "kernel_count": graph["kernel_count"],
        }
    return graph_views
