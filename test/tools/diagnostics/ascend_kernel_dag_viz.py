#!/usr/bin/env python3
"""Render Ascend artifact manifests as a readable kernel DAG SVG."""

from __future__ import annotations

import argparse
import collections
import html
import json
import re
from pathlib import Path
from typing import Any


VALID_KERNEL_KINDS = ("vec", "cube", "mix")


def kernel_sort_key(kernel_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", kernel_id)
    if match:
        return (int(match.group(1)), kernel_id)
    return (10**9, kernel_id)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        root = json.load(file)
    if not isinstance(root, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return root


def normalize_kind(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "vec"
    resources = entry.get("resources") if isinstance(entry.get("resources"), dict) else {}
    kind = entry.get("kernelKind") or resources.get("kernelKind") or "vec"
    return kind if kind in VALID_KERNEL_KINDS else "vec"


def compact_shape(shape: Any) -> str:
    if shape is None:
        return "?"
    if isinstance(shape, list):
        return "x".join(str(dim) for dim in shape) if shape else "scalar"
    return str(shape)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "..."


def as_task_map(run_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tasks = run_manifest.get("tasks")
    if isinstance(tasks, list):
        return {
            task["task_id"]: task
            for task in tasks
            if isinstance(task, dict) and isinstance(task.get("task_id"), str)
        }
    if isinstance(run_manifest.get("task_id"), str):
        return {run_manifest["task_id"]: run_manifest}
    return {}


def collect_kernel_ids(
    runtime_manifest: dict[str, Any],
    entries_by_id: dict[str, dict[str, Any]],
    tasks_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    ids: set[str] = set(entries_by_id)
    ids.update(tasks_by_id)
    graph = runtime_manifest.get("kernelGraph") if isinstance(runtime_manifest.get("kernelGraph"), dict) else {}
    for node in graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []:
        if isinstance(node, dict) and isinstance(node.get("name"), str):
            ids.add(node["name"])
    for edge in graph.get("edges", []) if isinstance(graph.get("edges"), list) else []:
        if isinstance(edge, dict):
            if isinstance(edge.get("from"), str):
                ids.add(edge["from"])
            if isinstance(edge.get("to"), str):
                ids.add(edge["to"])
    return sorted(ids, key=kernel_sort_key)


def collect_op_summaries(kernelized_ir: Path | None) -> dict[str, list[dict[str, Any]]]:
    if kernelized_ir is None:
        return {}
    if not kernelized_ir.exists():
        raise SystemExit(f"kernelized IR not found: {kernelized_ir}")

    lines = kernelized_ir.read_text(encoding="utf-8", errors="replace").splitlines()
    kernel_attr = re.compile(r'ascend\.kernel = "(kernel_\d+)"')
    op_name = re.compile(r"=\s+([A-Za-z_][\w.]*)\b")
    role_attr = re.compile(r'ascend\.op_role = "([^"]+)"')
    result_type = re.compile(r"->\s+([^\s{]+)")
    summaries: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for index, line in enumerate(lines):
        kernel_match = kernel_attr.search(line)
        if not kernel_match:
            continue
        kernel_id = kernel_match.group(1)
        op_match = op_name.search(line)
        full_op = op_match.group(1) if op_match else "op"
        label = full_op
        for prefix in ("linalg.", "tensor.", "arith.", "math."):
            label = label.replace(prefix, "")

        if full_op == "linalg.generic":
            body = "\n".join(lines[index : index + 18])
            body_ops = []
            for needle, short in (
                ("math.exp", "exp"),
                ("math.sqrt", "sqrt"),
                ("arith.divf", "divf"),
                ("arith.mulf", "mulf"),
                ("arith.addf", "addf"),
                ("arith.subf", "subf"),
                ("arith.maximumf", "maxf"),
                ("arith.minimumf", "minf"),
                ("arith.select", "select"),
                ("arith.cmpf", "cmpf"),
                ("arith.truncf", "truncf"),
            ):
                if needle in body:
                    body_ops.append(short)
            if body_ops:
                suffix = "+".join(body_ops[:4])
                if len(body_ops) > 4:
                    suffix += "+..."
                label = f"{label}({suffix})"

        summaries[kernel_id].append(
            {
                "line": index + 1,
                "op": full_op,
                "label": label,
                "role": (role_attr.search(line).group(1) if role_attr.search(line) else None),
                "result_type": (
                    result_type.search(line).group(1) if result_type.search(line) else None
                ),
            }
        )
    return summaries


def build_graph(
    runtime_manifest: dict[str, Any],
    kernel_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, set[str]]]:
    graph = runtime_manifest.get("kernelGraph") if isinstance(runtime_manifest.get("kernelGraph"), dict) else {}
    edges: list[dict[str, Any]] = []
    pred: dict[str, set[str]] = collections.defaultdict(set)
    succ: dict[str, set[str]] = collections.defaultdict(set)
    known = set(kernel_ids)

    for edge in graph.get("edges", []) if isinstance(graph.get("edges"), list) else []:
        if not isinstance(edge, dict):
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        if src not in known or dst not in known:
            continue
        carried = edge.get("carriedBuffers")
        edges.append(
            {
                "from": src,
                "to": dst,
                "carried_buffer_count": len(carried) if isinstance(carried, list) else 0,
            }
        )
        pred[dst].add(src)
        succ[src].add(dst)
    for kernel_id in kernel_ids:
        pred[kernel_id]
        succ[kernel_id]
    return edges, pred, succ


def topo_depth(
    kernel_ids: list[str],
    pred: dict[str, set[str]],
    succ: dict[str, set[str]],
) -> tuple[dict[str, int], list[str]]:
    in_degree = {kernel_id: len(pred[kernel_id]) for kernel_id in kernel_ids}
    queue = collections.deque(
        sorted(
            [kernel_id for kernel_id in kernel_ids if in_degree[kernel_id] == 0],
            key=kernel_sort_key,
        )
    )
    depth = {kernel_id: 1 for kernel_id in kernel_ids}
    critical_prev: dict[str, str] = {}
    visited = []
    while queue:
        kernel_id = queue.popleft()
        visited.append(kernel_id)
        for dst in sorted(succ[kernel_id], key=kernel_sort_key):
            next_depth = depth[kernel_id] + 1
            if next_depth > depth.get(dst, 1):
                depth[dst] = next_depth
                critical_prev[dst] = kernel_id
            in_degree[dst] -= 1
            if in_degree[dst] == 0:
                queue.append(dst)
    if len(visited) != len(kernel_ids):
        raise SystemExit("kernel DAG contains a cycle")

    if not kernel_ids:
        return {}, []
    critical_end = max(kernel_ids, key=lambda kernel_id: (depth[kernel_id], kernel_sort_key(kernel_id)[0] * -1))
    critical_path = []
    current: str | None = critical_end
    while current:
        critical_path.append(current)
        current = critical_prev.get(current)
    critical_path.reverse()
    return depth, critical_path


def first_output(task: dict[str, Any] | None) -> dict[str, Any]:
    if not task:
        return {}
    outputs = task.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        return outputs[0]
    return {}


def selected_tile(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "?"
    entries = entry.get("scheduleEntries")
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        return "?"
    tiling = entries[0].get("tilingParams")
    if not isinstance(tiling, dict):
        return "?"
    return compact_shape(tiling.get("selected_tile_shape"))


def default_runtime_input_roots(roots: list[str], tasks_by_id: dict[str, dict[str, Any]]) -> set[str]:
    runtime_roots: set[str] = set()
    if "kernel_0" in roots:
        runtime_roots.add("kernel_0")
    for root in roots:
        task = tasks_by_id.get(root, {})
        inputs = task.get("inputs")
        if not isinstance(inputs, list):
            continue
        for input_desc in inputs:
            if not isinstance(input_desc, dict):
                continue
            path = str(input_desc.get("path", ""))
            name = str(input_desc.get("name", ""))
            if re.search(r"(^|/)(input|data)\d*\.npy$", path) or name in {"input", "data0"}:
                runtime_roots.add(root)
    return runtime_roots


def analyze(
    runtime_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
    kernelized_ir: Path | None,
    runtime_input_roots_override: list[str] | None,
) -> dict[str, Any]:
    entries = runtime_manifest.get("kernel_entries", [])
    if not isinstance(entries, list):
        entries = []
    entries_by_id = {
        entry["kernel_id"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("kernel_id"), str)
    }
    tasks_by_id = as_task_map(run_manifest)
    kernel_ids = collect_kernel_ids(runtime_manifest, entries_by_id, tasks_by_id)
    edges, pred, succ = build_graph(runtime_manifest, kernel_ids)
    depth, critical_path = topo_depth(kernel_ids, pred, succ)

    roots = sorted([kernel_id for kernel_id in kernel_ids if not pred[kernel_id]], key=kernel_sort_key)
    leaves = sorted([kernel_id for kernel_id in kernel_ids if not succ[kernel_id]], key=kernel_sort_key)
    runtime_input_roots = (
        set(runtime_input_roots_override)
        if runtime_input_roots_override
        else default_runtime_input_roots(roots, tasks_by_id)
    )
    runtime_input_roots &= set(roots)
    prepack_roots = sorted(set(roots) - runtime_input_roots, key=kernel_sort_key)
    kind_counts = collections.Counter(normalize_kind(entries_by_id.get(kernel_id)) for kernel_id in kernel_ids)
    op_summaries = collect_op_summaries(kernelized_ir)
    fusion_edges = [
        {"from": edge["from"], "to": edge["to"]}
        for edge in edges
        if normalize_kind(entries_by_id.get(edge["from"]))
        == normalize_kind(entries_by_id.get(edge["to"]))
        == "vec"
        and len(succ[edge["from"]]) == 1
        and len(pred[edge["to"]]) == 1
    ]
    fusion_nodes = {item for edge in fusion_edges for item in (edge["from"], edge["to"])}

    nodes = {}
    for kernel_id in kernel_ids:
        task = tasks_by_id.get(kernel_id)
        output = first_output(task)
        nodes[kernel_id] = {
            "kind": normalize_kind(entries_by_id.get(kernel_id)),
            "depth": depth.get(kernel_id, 1),
            "input_degree": len(pred[kernel_id]),
            "output_degree": len(succ[kernel_id]),
            "output_shape": compact_shape(output.get("shape")),
            "output_dtype": output.get("dtype", ""),
            "selected_tile_shape": selected_tile(entries_by_id.get(kernel_id)),
            "workspace_size": (
                task.get("workspace_size")
                if task and "workspace_size" in task
                else entries_by_id.get(kernel_id, {}).get("workspaceSizeBytes", 0)
            ),
            "ops": op_summaries.get(kernel_id, []),
            "is_root": kernel_id in roots,
            "is_leaf": kernel_id in leaves,
            "is_runtime_input_root": kernel_id in runtime_input_roots,
            "is_prepack_candidate_root": kernel_id in prepack_roots,
            "touches_simple_fusion_edge": kernel_id in fusion_nodes,
        }

    return {
        "kernel_count": len(kernel_ids),
        "task_count": len(tasks_by_id),
        "graph_edges": len(edges),
        "kind_counts": {kind: kind_counts.get(kind, 0) for kind in VALID_KERNEL_KINDS},
        "root_tasks": len(roots),
        "root_task_ids": roots,
        "leaf_tasks": len(leaves),
        "leaf_task_ids": leaves,
        "runtime_input_roots": len(runtime_input_roots),
        "runtime_input_root_ids": sorted(runtime_input_roots, key=kernel_sort_key),
        "prepack_candidate_roots": len(prepack_roots),
        "prepack_candidate_root_ids": prepack_roots,
        "critical_path_depth": max(depth.values(), default=0),
        "critical_path": critical_path,
        "simple_fusion_edges": sorted(fusion_edges, key=lambda edge: (kernel_sort_key(edge["from"]), kernel_sort_key(edge["to"]))),
        "edges": edges,
        "nodes": nodes,
    }


def render_svg(summary: dict[str, Any], out: Path) -> None:
    nodes = summary["nodes"]
    kernel_ids = sorted(nodes, key=kernel_sort_key)
    edges = summary["edges"]
    levels: dict[int, list[str]] = collections.defaultdict(list)
    for kernel_id in kernel_ids:
        levels[nodes[kernel_id]["depth"]].append(kernel_id)
    for level_nodes in levels.values():
        level_nodes.sort(key=kernel_sort_key)

    max_depth = max(levels, default=1)
    max_rows = max((len(level_nodes) for level_nodes in levels.values()), default=1)
    node_w, node_h = 236, 84
    col_w, row_h = 300, 104
    left, top = 96, 248
    right_pad, bottom_pad = 310, 170
    width = left + (max_depth - 1) * col_w + node_w + right_pad
    height = top + max_rows * row_h + bottom_pad
    positions = {
        kernel_id: (left + (nodes[kernel_id]["depth"] - 1) * col_w, top + index * row_h)
        for depth, level_nodes in levels.items()
        for index, kernel_id in enumerate(level_nodes)
    }
    palette = {
        "vec": ("#dbeafe", "#2563eb"),
        "cube": ("#ffedd5", "#f97316"),
        "mix": ("#ede9fe", "#7c3aed"),
    }
    critical_edges = set(zip(summary["critical_path"], summary["critical_path"][1:]))
    fusion_edges = {(edge["from"], edge["to"]) for edge in summary["simple_fusion_edges"]}

    lines = []
    add = lines.append
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    add("<defs>")
    add(
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" '
        'fill="#94a3b8"/></marker>'
    )
    add(
        '<marker id="arrow-critical" markerWidth="10" markerHeight="10" refX="9" refY="3" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" '
        'fill="#dc2626"/></marker>'
    )
    add(
        '<marker id="arrow-fusion" markerWidth="10" markerHeight="10" refX="9" refY="3" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" '
        'fill="#16a34a"/></marker>'
    )
    add("<style>")
    add(
        'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#0f172a}'
        '.mono{font-family:"SFMono-Regular",Menlo,Consolas,monospace}'
        ".title{font-size:24px;font-weight:700}.small{font-size:12px}"
        ".tiny{font-size:10px;fill:#475569}.level{font-size:12px;font-weight:700;fill:#64748b}"
        ".edge{fill:none;stroke:#94a3b8;stroke-width:1.2;opacity:.62}"
        ".edge-critical{fill:none;stroke:#dc2626;stroke-width:3;opacity:.95}"
        ".edge-fusion{fill:none;stroke:#16a34a;stroke-width:3;opacity:.92}"
        ".node-id{font-size:14px;font-weight:800}.node-op{font-size:12px;font-weight:700;fill:#111827}"
        ".node-line{font-size:10px;fill:#334155}.badge{font-size:9px;font-weight:800;fill:#92400e}"
        ".fusion-badge{font-size:9px;font-weight:800;fill:#166534}"
    )
    add("</style>")
    add("</defs>")
    add('<rect width="100%" height="100%" fill="#f8fafc"/>')
    add('<text class="title" x="36" y="42">Ascend Kernel DAG</text>')
    add(
        f'<text class="small" x="36" y="69">kernels/tasks={summary["kernel_count"]}; '
        f'edges={summary["graph_edges"]}; vec={summary["kind_counts"]["vec"]}, '
        f'cube={summary["kind_counts"]["cube"]}, mix={summary["kind_counts"]["mix"]}; '
        f'roots={summary["root_tasks"]}; prepack roots={summary["prepack_candidate_roots"]}; '
        f'critical path depth={summary["critical_path_depth"]}; '
        f'simple vec-&gt;vec fusion candidates={len(summary["simple_fusion_edges"])}</text>'
    )
    add(
        '<text class="tiny" x="36" y="92">Columns are DAG depth. Hover a node in a browser for '
        'full op lines, shape, tile, workspace, and candidate markers.</text>'
    )
    add(
        '<text class="tiny" x="36" y="110">Green edges are topology-level hints only; '
        'fusion legality still needs semantic and backend checks.</text>'
    )

    legend_x, legend_y = 36, 146
    for index, kind in enumerate(VALID_KERNEL_KINDS):
        fill, stroke = palette[kind]
        x = legend_x + index * 112
        add(
            f'<rect x="{x}" y="{legend_y}" width="20" height="14" rx="3" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )
        add(f'<text class="small" x="{x + 28}" y="{legend_y + 12}">{kind}</text>')
    add(
        f'<rect x="{legend_x + 350}" y="{legend_y}" width="20" height="14" rx="3" '
        'fill="#fef3c7" stroke="#d97706" stroke-width="2" stroke-dasharray="4 3"/>'
    )
    add(f'<text class="small" x="{legend_x + 378}" y="{legend_y + 12}">prepack root</text>')
    add(
        f'<line x1="{legend_x + 530}" y1="{legend_y + 7}" x2="{legend_x + 580}" '
        f'y2="{legend_y + 7}" class="edge-fusion" marker-end="url(#arrow-fusion)"/>'
    )
    add(f'<text class="small" x="{legend_x + 590}" y="{legend_y + 12}">simple fusion</text>')
    add(
        f'<line x1="{legend_x + 735}" y1="{legend_y + 7}" x2="{legend_x + 785}" '
        f'y2="{legend_y + 7}" class="edge-critical" marker-end="url(#arrow-critical)"/>'
    )
    add(f'<text class="small" x="{legend_x + 795}" y="{legend_y + 12}">critical path</text>')

    for depth in range(1, max_depth + 1):
        x = left + (depth - 1) * col_w
        add(f'<text class="level" x="{x}" y="{top - 26}">depth {depth}</text>')
        add(
            f'<line x1="{x - 20}" y1="{top - 12}" x2="{x - 20}" '
            f'y2="{height - bottom_pad + 15}" stroke="#e2e8f0" stroke-width="1"/>'
        )

    for edge in edges:
        src, dst = edge["from"], edge["to"]
        if src not in positions or dst not in positions:
            continue
        sx, sy = positions[src]
        dx, dy = positions[dst]
        x1, y1 = sx + node_w, sy + node_h / 2
        x2, y2 = dx, dy + node_h / 2
        mid = (x1 + x2) / 2
        if (src, dst) in critical_edges:
            css_class, marker = "edge-critical", "arrow-critical"
        elif (src, dst) in fusion_edges:
            css_class, marker = "edge-fusion", "arrow-fusion"
        else:
            css_class, marker = "edge", "arrow"
        add(
            f'<path class="{css_class}" d="M{x1:.1f},{y1:.1f} C{mid:.1f},{y1:.1f} '
            f'{mid:.1f},{y2:.1f} {x2:.1f},{y2:.1f}" marker-end="url(#{marker})"/>'
        )
        if edge["carried_buffer_count"] > 1:
            add(
                f'<text class="tiny mono" x="{mid:.1f}" y="{((y1 + y2) / 2 - 3):.1f}" '
                f'text-anchor="middle">x{edge["carried_buffer_count"]}</text>'
            )

    critical_nodes = set(summary["critical_path"])
    for kernel_id in kernel_ids:
        x, y = positions[kernel_id]
        node = nodes[kernel_id]
        fill, stroke = palette[node["kind"]]
        stroke_width = 2.0
        dash = ""
        if node["is_prepack_candidate_root"]:
            fill, stroke, stroke_width, dash = "#fef3c7", "#d97706", 2.2, ' stroke-dasharray="5 3"'
        if node["is_runtime_input_root"]:
            stroke, stroke_width = "#059669", 2.2
        if node["touches_simple_fusion_edge"] and not node["is_prepack_candidate_root"]:
            stroke, stroke_width = "#16a34a", max(stroke_width, 2.5)
        if kernel_id in critical_nodes:
            stroke, stroke_width, dash = "#dc2626", 3.0, ""

        op_labels = [op["label"] for op in node["ops"]]
        op_line = " + ".join(op_labels[:3]) if op_labels else "no-op-summary"
        if len(op_labels) > 3:
            op_line += " + ..."
        io_line = (
            f'in:{node["input_degree"]} out:{node["output_degree"]}  '
            f'out:{node["output_shape"]} {node["output_dtype"]}'
        ).strip()
        tile_line = f'tile:{node["selected_tile_shape"]}'
        title_lines = [
            kernel_id,
            f'kind={node["kind"]}',
            io_line,
            tile_line,
            f'workspace_size={node["workspace_size"]}',
        ]
        for op in node["ops"]:
            detail = f'L{op["line"]}: {op["op"]}'
            if op.get("result_type"):
                detail += f' -> {op["result_type"]}'
            title_lines.append(detail)
        if node["is_prepack_candidate_root"]:
            title_lines.append("prepack_candidate_root=true")
        if node["touches_simple_fusion_edge"]:
            title_lines.append("touches_simple_fusion_edge=true")

        add(f'<g id="{html.escape(kernel_id)}">')
        add("<title>" + html.escape("\n".join(title_lines)) + "</title>")
        add(
            f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="7" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"{dash}/>'
        )
        add(
            f'<text class="node-id mono" x="{x + 10}" y="{y + 18}">'
            f'{html.escape(kernel_id)} [{node["kind"]}]</text>'
        )
        badge: tuple[str, str] | None = None
        if node["is_prepack_candidate_root"]:
            badge = ("prepack", "badge")
        elif node["is_runtime_input_root"]:
            badge = ("input", "badge")
        elif node["touches_simple_fusion_edge"]:
            badge = ("fuse?", "fusion-badge")
        elif node["is_leaf"]:
            badge = ("leaf", "tiny")
        if badge:
            text, css_class = badge
            add(f'<text class="{css_class}" x="{x + node_w - 58}" y="{y + 18}">{text}</text>')
        add(f'<text class="node-op" x="{x + 10}" y="{y + 39}">{html.escape(truncate(op_line, 32))}</text>')
        add(f'<text class="node-line mono" x="{x + 10}" y="{y + 56}">{html.escape(truncate(io_line, 38))}</text>')
        add(f'<text class="node-line mono" x="{x + 10}" y="{y + 72}">{html.escape(truncate(tile_line, 38))}</text>')
        add("</g>")

    critical_label = " -> ".join(summary["critical_path"])
    fusion_label = ", ".join(f'{edge["from"]}->{edge["to"]}' for edge in summary["simple_fusion_edges"])
    add(f'<text class="small mono" x="36" y="{height - 92}">critical_path: {html.escape(critical_label)}</text>')
    add(f'<text class="small mono" x="36" y="{height - 68}">simple_fusion_edges: {html.escape(fusion_label)}</text>')
    add(f'<text class="tiny" x="36" y="{height - 42}">Generated by ascend_kernel_dag_viz.py.</text>')
    add("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def write_summary(summary: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--kernelized-ir", type=Path)
    parser.add_argument("--svg-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument(
        "--runtime-input-root",
        action="append",
        help="Root kernel that should not be counted as a static prepack candidate.",
    )
    args = parser.parse_args()

    if not args.svg_out and not args.summary_out:
        raise SystemExit("at least one of --svg-out or --summary-out is required")
    if not args.artifact_manifest and not args.runtime_manifest:
        raise SystemExit("--artifact-manifest is required")
    if (
        args.artifact_manifest
        and args.runtime_manifest
        and args.artifact_manifest.resolve() != args.runtime_manifest.resolve()
    ):
        raise SystemExit("cannot pass both --artifact-manifest and --runtime-manifest with different paths")

    artifact_manifest_path = args.artifact_manifest or args.runtime_manifest
    runtime_manifest = load_json(artifact_manifest_path)
    run_manifest = load_json(args.run_manifest) if args.run_manifest else {}
    summary = analyze(
        runtime_manifest,
        run_manifest,
        args.kernelized_ir,
        args.runtime_input_root,
    )
    if args.svg_out:
        render_svg(summary, args.svg_out)
    if args.summary_out:
        write_summary(summary, args.summary_out)

    print(f"ascend_kernel_dag_viz.kernel_count={summary['kernel_count']}")
    print(f"ascend_kernel_dag_viz.graph_edges={summary['graph_edges']}")
    print(f"ascend_kernel_dag_viz.root_tasks={summary['root_tasks']}")
    print(f"ascend_kernel_dag_viz.prepack_candidate_roots={summary['prepack_candidate_roots']}")
    print(f"ascend_kernel_dag_viz.critical_path_depth={summary['critical_path_depth']}")
    print(f"ascend_kernel_dag_viz.simple_fusion_edges={len(summary['simple_fusion_edges'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
