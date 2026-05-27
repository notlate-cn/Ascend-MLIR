from __future__ import annotations

import argparse
import html
import json
import pathlib
import posixpath
import sys
import webbrowser
from typing import Any

from ascend_debug import debug_graph, layout, memory, stage_graph
from ascend_debug.runner import CommandError


def _validate_run_relative_path(value: Any, *, manifest_path: pathlib.Path, label: str) -> str:
    if not isinstance(value, str):
        raise CommandError(f"manifest {label} path must be a string: {manifest_path}")
    path = pathlib.PurePosixPath(value)
    if (
        value in ("", ".")
        or path.is_absolute()
        or ".." in path.parts
        or posixpath.normpath(value) != value
    ):
        raise CommandError(f"manifest {label} path must stay inside run dir: {manifest_path}")
    return value


def load_manifest(run_dir: pathlib.Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CommandError(f"manifest not found: {manifest_path}") from error
    except (OSError, UnicodeDecodeError) as error:
        raise CommandError(f"could not read manifest: {manifest_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CommandError(f"manifest is not valid JSON: {manifest_path}: {error}") from error

    if not isinstance(manifest, dict):
        raise CommandError(f"manifest must be a JSON object: {manifest_path}")
    if "schema_version" not in manifest:
        raise CommandError(f"manifest missing schema_version: {manifest_path}")
    stages = manifest.get("stages")
    if not isinstance(stages, list):
        raise CommandError(f"manifest stages must be a list: {manifest_path}")
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise CommandError(f"manifest stage {index} must be an object: {manifest_path}")
        for field in ("order", "name", "path"):
            if field not in stage:
                raise CommandError(f"manifest stage {index} missing {field}: {manifest_path}")
        if isinstance(stage["order"], bool) or not isinstance(stage["order"], int):
            raise CommandError(f"manifest stage {index} order must be an integer: {manifest_path}")
        if not isinstance(stage["name"], str):
            raise CommandError(f"manifest stage {index} name must be a string: {manifest_path}")
        _validate_run_relative_path(
            stage["path"],
            manifest_path=manifest_path,
            label=f"stage {index}",
        )

    commands = manifest.get("commands", [])
    if not isinstance(commands, list):
        raise CommandError(f"manifest commands must be a list: {manifest_path}")
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            raise CommandError(f"manifest command {index} must be an object: {manifest_path}")
        for field in ("stage", "tool", "status"):
            if field in command and not isinstance(command[field], str):
                raise CommandError(f"manifest command {index} {field} must be a string: {manifest_path}")
        if "args" in command and not (
            isinstance(command["args"], list)
            and all(isinstance(arg, str) for arg in command["args"])
        ):
            raise CommandError(f"manifest command {index} args must be a string list: {manifest_path}")
        for field in ("stdout", "stderr"):
            if field in command:
                _validate_run_relative_path(
                    command[field],
                    manifest_path=manifest_path,
                    label=f"command {index} {field}",
                )

    reports = manifest.get("reports", [])
    if not isinstance(reports, list):
        raise CommandError(f"manifest reports must be a list: {manifest_path}")
    for index, report in enumerate(reports):
        if not isinstance(report, dict):
            raise CommandError(f"manifest report {index} must be an object: {manifest_path}")
        if "stage" in report and not isinstance(report["stage"], str):
            raise CommandError(f"manifest report {index} stage must be a string: {manifest_path}")
        if "path" not in report:
            raise CommandError(f"manifest report {index} missing path: {manifest_path}")
        _validate_run_relative_path(
            report["path"],
            manifest_path=manifest_path,
            label=f"report {index}",
        )

    graphs = manifest.get("graphs", [])
    if not isinstance(graphs, list):
        raise CommandError(f"manifest graphs must be a list: {manifest_path}")
    for index, graph in enumerate(graphs):
        if not isinstance(graph, dict):
            raise CommandError(f"manifest graph {index} must be an object: {manifest_path}")
        if "kind" in graph and not isinstance(graph["kind"], str):
            raise CommandError(f"manifest graph {index} kind must be a string: {manifest_path}")
        if "path" not in graph:
            raise CommandError(f"manifest graph {index} missing path: {manifest_path}")
        _validate_run_relative_path(
            graph["path"],
            manifest_path=manifest_path,
            label=f"graph {index}",
        )
    return manifest


def _cell(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _path_link(rel_path: str, *, exists: bool = True) -> str:
    label = _cell(rel_path)
    if not exists:
        return label
    href = html.escape(rel_path, quote=True)
    return f'<a href="{href}">{label}</a>'


def _link(href: str, label: str) -> str:
    return f'<a href="{html.escape(href, quote=True)}">{_cell(label)}</a>'


def _stage_view_rel_path(stage_rel_path: str) -> str:
    return f"views/{stage_rel_path}.html"


def _relative_href(from_rel_path: str, to_rel_path: str) -> str:
    source_dir = pathlib.PurePosixPath(from_rel_path).parent
    return posixpath.relpath(to_rel_path, start=str(source_dir))


def _metadata_rows(manifest: dict[str, Any]) -> str:
    keys = ("schema_version", "tool", "preset", "pipeline", "backend", "device_id")
    rows = []
    for key in keys:
        if key in manifest:
            rows.append(f"<dt>{_cell(key)}</dt><dd>{_cell(manifest[key])}</dd>")
    return "\n".join(rows)


def _overview_cards(manifest: dict[str, Any]) -> str:
    items = [
        ("tool", manifest.get("tool")),
        ("preset", manifest.get("preset")),
        ("pipeline", manifest.get("pipeline")),
        ("stages", len(manifest.get("stages", []))),
        ("commands", len(manifest.get("commands", []))),
    ]
    if manifest.get("backend") is not None:
        items.append(("backend", manifest.get("backend")))
    if manifest.get("device_id") is not None:
        items.append(("device", manifest.get("device_id")))
    return "\n".join(
        f'<div class="overview-card"><span>{_cell(label)}</span><strong>{_cell(value)}</strong></div>'
        for label, value in items
    )


def _stage_rows(
    run_dir: pathlib.Path,
    manifest: dict[str, Any],
    stage_views: dict[str, str],
    debug_graph_view_path: str,
) -> str:
    rows = []
    stages = sorted(manifest["stages"], key=lambda stage: stage["order"])
    for stage in stages:
        rel_path = str(stage["path"])
        exists = (run_dir / rel_path).exists()
        status = "存在" if exists else "缺失"
        view_rel_path = stage_views.get(rel_path)
        view_cell = _link(view_rel_path, "查看 MLIR") if view_rel_path else ""
        debug_cell = _link(f"{debug_graph_view_path}?stage={stage['order']}", "在工作台查看")
        rows.append(
            "<tr>"
            f"<td>{_cell(stage['order'])}</td>"
            f"<td>{_cell(stage['name'])}</td>"
            f"<td>{view_cell}</td>"
            f"<td>{debug_cell}</td>"
            f"<td>{_cell(status)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _stage_graph_rows(
    run_dir: pathlib.Path,
    manifest: dict[str, Any],
    graph_views: dict[str, dict[str, Any]],
) -> str:
    rows = []
    stages = sorted(manifest["stages"], key=lambda stage: stage["order"])
    for stage in stages:
        rel_path = str(stage["path"])
        exists = (run_dir / rel_path).exists()
        graph = graph_views.get(rel_path)
        graph_cell = ""
        json_cell = ""
        node_count = ""
        edge_count = ""
        kernel_count = ""
        if graph:
            graph_cell = _link(graph["view_rel_path"], "图")
            json_cell = _path_link(graph["json_rel_path"], exists=(run_dir / graph["json_rel_path"]).exists())
            node_count = graph["node_count"]
            edge_count = graph["edge_count"]
            kernel_count = graph["kernel_count"]
        rows.append(
            "<tr>"
            f"<td>{_cell(stage['order'])}</td>"
            f"<td>{_cell(stage['name'])}</td>"
            f"<td>{graph_cell}</td>"
            f"<td>{json_cell}</td>"
            f"<td>{_cell(node_count)}</td>"
            f"<td>{_cell(edge_count)}</td>"
            f"<td>{_cell(kernel_count)}</td>"
            f"<td>{_cell('存在' if exists else '缺失')}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _command_rows(run_dir: pathlib.Path, manifest: dict[str, Any]) -> str:
    rows = []
    for command in manifest.get("commands", []):
        args = " ".join(command.get("args", []))
        report_path = command.get("stderr") or command.get("stdout")
        report_cell = ""
        if report_path:
            report_cell = _path_link(report_path, exists=(run_dir / report_path).exists())
        rows.append(
            "<tr>"
            f"<td>{_cell(command.get('stage'))}</td>"
            f"<td>{_cell(command.get('tool'))}</td>"
            f"<td>{_cell(args)}</td>"
            f"<td>{_cell(command.get('status'))}</td>"
            f"<td>{report_cell}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _report_rows(run_dir: pathlib.Path, manifest: dict[str, Any]) -> str:
    rows = []
    for report in manifest.get("reports", []):
        rel_path = report["path"]
        exists = (run_dir / rel_path).exists()
        status = "存在" if exists else "缺失"
        rows.append(
            "<tr>"
            f"<td>{_cell(report.get('stage'))}</td>"
            f"<td>{_path_link(rel_path, exists=exists)}</td>"
            f"<td>{_cell(status)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _graph_rows(
    run_dir: pathlib.Path,
    manifest: dict[str, Any],
    artifact_views: dict[str, str],
) -> str:
    rows = []
    for graph in manifest.get("graphs", []):
        rel_path = graph["path"]
        exists = (run_dir / rel_path).exists()
        status = "存在" if exists else "缺失"
        view_rel_path = artifact_views.get(rel_path)
        view_cell = _link(view_rel_path, "查看") if view_rel_path else ""
        rows.append(
            "<tr>"
            f"<td>{_cell(graph.get('kind'))}</td>"
            f"<td>{_path_link(rel_path, exists=exists)}</td>"
            f"<td>{view_cell}</td>"
            f"<td>{_cell(status)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _summary_rows(run_dir: pathlib.Path, json_views: dict[str, str]) -> str:
    summary_dir = run_dir / "summaries"
    if not summary_dir.exists():
        return ""
    rows = []
    for path in sorted(item for item in summary_dir.rglob("*") if item.is_file()):
        rel_path = path.relative_to(run_dir).as_posix()
        view_rel_path = json_views.get(rel_path)
        view_cell = _link(view_rel_path, "查看") if view_rel_path else ""
        rows.append(
            "<tr>"
            f"<td>{_path_link(rel_path)}</td>"
            f"<td>{view_cell}</td>"
            f"<td>{_cell(path.stat().st_size)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _load_tensor_diff(run_dir: pathlib.Path) -> dict[str, Any] | None:
    diff_path = run_dir / "summaries/tensor_diff.json"
    if not diff_path.exists():
        return None
    try:
        summary = json.loads(diff_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CommandError(f"could not read tensor diff summary: {diff_path}: {error}") from error
    if not isinstance(summary, dict):
        raise CommandError(f"tensor diff summary must be a JSON object: {diff_path}")
    comparisons = summary.get("comparisons")
    if not isinstance(comparisons, list):
        raise CommandError(f"tensor diff summary comparisons must be a list: {diff_path}")
    return summary


def _load_locate_summary(run_dir: pathlib.Path) -> dict[str, Any] | None:
    locate_path = run_dir / "summaries/locate.json"
    if not locate_path.exists():
        return None
    try:
        summary = json.loads(locate_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CommandError(f"could not read locate summary: {locate_path}: {error}") from error
    if not isinstance(summary, dict):
        raise CommandError(f"locate summary must be a JSON object: {locate_path}")
    return summary


def _tensor_diff_rows(
    summary: dict[str, Any] | None,
    kernel_views: dict[str, str],
) -> str:
    if not summary:
        return ""
    rows = []
    for comparison in summary.get("comparisons", []):
        if not isinstance(comparison, dict):
            continue
        kernel_id = comparison.get("kernel_id")
        kernel_cell = ""
        if isinstance(kernel_id, str) and kernel_id:
            kernel_cell = (
                _link(kernel_views[kernel_id], kernel_id)
                if kernel_id in kernel_views
                else _cell(kernel_id)
            )
        rows.append(
            "<tr>"
            f"<td>{_cell(comparison.get('status'))}</td>"
            f"<td>{_cell(comparison.get('id'))}</td>"
            f"<td>{kernel_cell}</td>"
            f"<td>{_cell(comparison.get('task_id'))}</td>"
            f"<td>{_cell(comparison.get('max_abs_error'))}</td>"
            f"<td>{_cell(comparison.get('max_rel_error'))}</td>"
            f"<td>{_cell(comparison.get('mean_abs_error'))}</td>"
            f"<td>{_cell(comparison.get('atol'))}</td>"
            f"<td>{_cell(comparison.get('rtol'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _kernel_list_cell(value: Any, kernel_views: dict[str, str]) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    links = []
    for item in value:
        if not isinstance(item, str) or not item:
            continue
        links.append(_link(kernel_views[item], item) if item in kernel_views else _cell(item))
    return ", ".join(links) if links else "none"


def _locate_section(summary: dict[str, Any] | None, kernel_views: dict[str, str]) -> str:
    if not summary:
        return ""
    kernel_id = summary.get("first_bad_kernel")
    kernel_cell = _cell(kernel_id or "none")
    if isinstance(kernel_id, str) and kernel_id in kernel_views:
        kernel_cell = _link(kernel_views[kernel_id], kernel_id)
    first_bad_comparison = summary.get("first_bad_comparison")
    comparison_id = "none"
    if isinstance(first_bad_comparison, dict):
        comparison_id = first_bad_comparison.get("id", "none")
    first_bad_context = summary.get("first_bad_context", {})
    if not isinstance(first_bad_context, dict):
        first_bad_context = {}
    upstream_passed = _kernel_list_cell(first_bad_context.get("upstream_checked_passed"), kernel_views)
    unchecked_upstream = _kernel_list_cell(first_bad_context.get("unchecked_direct_upstream"), kernel_views)
    downstream_failed = _kernel_list_cell(first_bad_context.get("downstream_failed"), kernel_views)
    direct_upstream = _kernel_list_cell(first_bad_context.get("direct_upstream"), kernel_views)
    direct_downstream = _kernel_list_cell(first_bad_context.get("direct_downstream"), kernel_views)
    return f"""
<section class="locate-section">
<h2>Locate</h2>
<p class="locate-note">按 DAG 顺序找到的最早失败 checkpoint。它是首个异常候选，不等同于根因证明，尤其是上游 Kernel 没有 checkpoint 时。</p>
<div class="locate-grid">
<div class="locate-panel">
<h3>首个异常候选</h3>
<dl>
<dt>状态</dt><dd>{_cell(summary.get('status'))}</dd>
<dt>Kernel</dt><dd>{kernel_cell}</dd>
<dt>DAG 深度</dt><dd>{_cell(summary.get('first_bad_depth'))}</dd>
<dt>失败对比</dt><dd>{_cell(comparison_id)}</dd>
</dl>
</div>
<div class="locate-panel">
<h3>证据</h3>
<dl>
<dt>失败 Kernel 数</dt><dd>{_cell(summary.get('failed_kernel_count'))}</dd>
<dt>已通过的上游 checkpoint</dt><dd>{upstream_passed}</dd>
<dt>失败的下游 checkpoint</dt><dd>{downstream_failed}</dd>
<dt>定位方法</dt><dd>{_cell(summary.get('method'))}</dd>
</dl>
</div>
<div class="locate-panel">
<h3>覆盖缺口</h3>
<dl>
<dt>直接上游</dt><dd>{direct_upstream}</dd>
<dt>直接下游</dt><dd>{direct_downstream}</dd>
<dt>未 checkpoint 的直接上游</dt><dd>{unchecked_upstream}</dd>
</dl>
</div>
</div>
</section>
"""


def _render_mlir_view(run_dir: pathlib.Path, rel_path: str) -> str | None:
    source_path = run_dir / rel_path
    if not source_path.exists():
        return None
    view_rel_path = _stage_view_rel_path(rel_path)
    view_path = run_dir / view_rel_path
    raw_href = html.escape(_relative_href(view_rel_path, rel_path), quote=True)
    try:
        source_text = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise CommandError(f"could not read MLIR view source: {source_path}: {error}") from error

    lines = source_text.splitlines() or [""]
    line_rows = []
    dashboard_href = html.escape(_relative_href(view_rel_path, "index.html"), quote=True)
    for line_number, line in enumerate(lines, start=1):
        escaped_line = html.escape(line)
        line_rows.append(
            '<tr class="line-row">'
            f'<td class="gutter"><a href="#L{line_number}" id="L{line_number}">'
            f'<span class="line-number">{line_number}</span></a></td>'
            f'<td class="code"><pre>{escaped_line}</pre></td>'
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{_cell(pathlib.PurePosixPath(rel_path).name)} - ascend-debug</title>
<style>
:root {{ color-scheme: light; }}
body {{ font-family: sans-serif; margin: 0; color: #17202a; background: #eef2f7; }}
header {{ position: sticky; top: 0; z-index: 1; padding: 0.75rem 1rem; background: #ffffff; border-bottom: 1px solid #cbd5e1; }}
h1 {{ font-size: 1rem; margin: 0 0 0.5rem 0; }}
.toolbar {{ display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }}
input {{ min-width: 18rem; padding: 0.35rem 0.5rem; border: 1px solid #94a3b8; border-radius: 4px; color: #17202a; background: #ffffff; }}
main {{ padding: 0.75rem 1rem 2rem; }}
table.code-table {{ border-collapse: collapse; width: 100%; background: #0b1020; color: #dbeafe; border: 1px solid #1e293b; }}
.code-table td {{ vertical-align: top; border-bottom: 1px solid #1e293b; }}
.gutter {{ width: 4.5rem; text-align: right; padding: 0 0.65rem; background: #111827; user-select: none; }}
.gutter a {{ color: #93a4bd; text-decoration: none; }}
.code {{ padding-left: 0.75rem; }}
.code pre {{ margin: 0; padding: 0.12rem 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.5 SFMono-Regular, Menlo, Consolas, monospace; color: #dbeafe; background: transparent; }}
.line-row:hover pre {{ background: #172033; }}
.hidden {{ display: none; }}
.match pre {{ background: #1d4ed8; color: #ffffff; }}
</style>
</head>
<body>
<header>
<h1>{_cell(rel_path)}</h1>
<div class="toolbar">
<input id="search" type="search" placeholder="搜索 MLIR">
<a href="{raw_href}">原始 MLIR</a>
<a href="{dashboard_href}">调试首页</a>
</div>
</header>
<main>
<table class="code-table text-code-table">
<tbody>
{''.join(line_rows)}
</tbody>
</table>
</main>
<script>
const input = document.getElementById("search");
const rows = Array.from(document.querySelectorAll(".line-row"));
input.addEventListener("input", () => {{
  const needle = input.value.toLowerCase();
  for (const row of rows) {{
    const text = row.innerText.toLowerCase();
    const matched = !needle || text.includes(needle);
    row.classList.toggle("hidden", !matched);
    row.classList.toggle("match", Boolean(needle && matched));
  }}
}});
</script>
</body>
</html>
"""
    layout.write_text(view_path, document)
    return view_rel_path


def _attr(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _as_int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _summary_kernel_link(view_rel_path: str, kernel_id: Any) -> str:
    if not isinstance(kernel_id, str) or not kernel_id:
        return _cell(kernel_id)
    href = _relative_href(view_rel_path, f"views/kernels/{kernel_id}.html")
    return _link(href, kernel_id)


def _summary_card(label: str, value: Any, css_class: str = "") -> str:
    class_attr = f' {css_class}' if css_class else ""
    return (
        f'<section class="summary-card{class_attr}">'
        f"<span>{_cell(label)}</span>"
        f"<strong>{_cell(value)}</strong>"
        "</section>"
    )


def _summary_page_document(
    *,
    view_rel_path: str,
    rel_path: str,
    title: str,
    body: str,
) -> str:
    raw_href = html.escape(_relative_href(view_rel_path, rel_path), quote=True)
    dashboard_href = html.escape(_relative_href(view_rel_path, "index.html"), quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_cell(title)} - ascend-debug</title>
<style>
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #eef2f7; }}
a {{ color: #1d4ed8; text-decoration: none; }}
header {{ position: sticky; top: 0; z-index: 1; padding: 0.85rem 1rem; background: #ffffff; border-bottom: 1px solid #cbd5e1; }}
h1 {{ font-size: 1.15rem; margin: 0 0 0.45rem; }}
h2 {{ font-size: 1rem; margin: 1.15rem 0 0.65rem; }}
h3 {{ font-size: 0.9rem; margin: 0 0 0.5rem; }}
main {{ padding: 1rem; }}
.toolbar {{ display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; color: #64748b; font-size: 0.86rem; }}
.summary-card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr)); gap: 0.65rem; margin-bottom: 1rem; }}
.summary-card {{ border: 1px solid #cbd5e1; border-radius: 8px; background: #ffffff; padding: 0.75rem; min-height: 4.8rem; }}
.summary-card span {{ display: block; color: #64748b; font-weight: 700; font-size: 0.78rem; margin-bottom: 0.35rem; }}
.summary-card strong {{ display: block; font-size: 1.45rem; line-height: 1.1; overflow-wrap: anywhere; }}
.summary-card.fail {{ border-color: #fecaca; background: #fff7f7; }}
.summary-card.pass {{ border-color: #bbf7d0; background: #f0fdf4; }}
.summary-card.warn {{ border-color: #fde68a; background: #fffbeb; }}
.panel {{ border: 1px solid #cbd5e1; border-radius: 8px; background: #ffffff; padding: 0.85rem; margin: 0 0 0.85rem; }}
.note {{ color: #475569; margin: 0 0 0.75rem; }}
.empty-state {{ border: 1px dashed #94a3b8; border-radius: 8px; padding: 0.75rem; background: #f8fafc; color: #475569; }}
table {{ width: 100%; border-collapse: collapse; background: #ffffff; margin: 0 0 0.85rem; font-size: 0.84rem; }}
th, td {{ border: 1px solid #cbd5e1; padding: 0.4rem 0.5rem; text-align: left; vertical-align: top; }}
th {{ background: #f1f5f9; color: #334155; }}
.status-fail {{ color: #b91c1c; font-weight: 700; }}
.status-pass {{ color: #047857; font-weight: 700; }}
.memory-kernel-viz {{ border: 1px solid #cbd5e1; border-radius: 8px; background: #ffffff; padding: 0.85rem; margin-bottom: 1rem; overflow: auto; }}
.memory-kernel-header {{ display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: baseline; justify-content: space-between; margin-bottom: 0.65rem; }}
.memory-kernel-header h2, .memory-kernel-header h3 {{ margin: 0; }}
.memory-facts {{ display: flex; gap: 0.45rem; flex-wrap: wrap; color: #475569; font-size: 0.82rem; }}
.chip {{ display: inline-block; border: 1px solid #cbd5e1; border-radius: 999px; padding: 0.16rem 0.45rem; background: #ffffff; color: #334155; margin: 0.12rem; font-size: 0.78rem; }}
.reuse-group {{ border-color: #86efac; background: #f0fdf4; color: #047857; }}
.movement-edge {{ border-color: #bfdbfe; background: #eff6ff; color: #1d4ed8; }}
.ub-allocation-svg {{ display: block; min-width: 640px; max-width: none; background: #fbfdff; border: 1px solid #dbe3ee; border-radius: 8px; }}
.ub-allocation-svg text {{ font-family: SFMono-Regular, Menlo, Consolas, monospace; font-size: 11px; fill: #334155; }}
.ub-allocation-svg .axis {{ stroke: #94a3b8; stroke-width: 1; }}
.ub-allocation-svg .grid {{ stroke: #dbe3ee; stroke-width: 1; }}
.ub-allocation-svg .row-bg {{ fill: #f8fafc; stroke: #e2e8f0; }}
.ub-allocation-svg .slot-block {{ fill: #60a5fa; stroke: #1d4ed8; stroke-width: 1.2; }}
.ub-allocation-svg .slot-block.reuse-group {{ fill: #86efac; stroke: #059669; }}
.ub-allocation-svg .movement-edge {{ fill: #f59e0b; stroke: #b45309; }}
pre.json-source {{ margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; background: #0b1020; color: #dbeafe; border: 1px solid #1e293b; border-radius: 8px; padding: 0.8rem; font: 12px/1.45 SFMono-Regular, Menlo, Consolas, monospace; }}
</style>
</head>
<body>
<header>
<h1>{_cell(title)}</h1>
<div class="toolbar">
<a href="{raw_href}">原始 JSON</a>
<a href="{dashboard_href}">调试工作台</a>
<span>{_cell(rel_path)}</span>
</div>
</header>
<main>
{body}
</main>
</body>
</html>
"""


def _value_list(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(_cell(item) for item in value)


def _memory_kernel_by_id(summary: dict[str, Any] | None, kernel_id: str) -> dict[str, Any] | None:
    if not summary:
        return None
    for kernel in summary.get("kernels", []):
        if isinstance(kernel, dict) and kernel.get("kernel_id") == kernel_id:
            return kernel
    return None


def _memory_has_kernel_anchor(summary: dict[str, Any] | None, kernel_id: str) -> bool:
    if not summary:
        return False
    if summary.get("analysis_level") == "realize-memory-plan":
        return _memory_kernel_by_id(summary, kernel_id) is not None
    return _memory_kernel_by_id(summary, kernel_id) is not None


def _memory_interval_by_value(kernel: dict[str, Any]) -> dict[int, dict[str, Any]]:
    intervals = {}
    for interval in kernel.get("live_intervals", []):
        if isinstance(interval, dict):
            intervals[_as_int(interval.get("value_id"))] = interval
    return intervals


def _memory_slot_groups(kernel: dict[str, Any]) -> list[tuple[tuple[str, int], list[dict[str, Any]]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for slot in kernel.get("workspace_slots", []):
        if not isinstance(slot, dict):
            continue
        key = (str(slot.get("place") or "unknown"), _as_int(slot.get("offset")))
        groups.setdefault(key, []).append(slot)
    return sorted(groups.items(), key=lambda item: (item[0][0], item[0][1]))


def _render_ub_allocation_svg(kernel: dict[str, Any]) -> str:
    groups = _memory_slot_groups(kernel)
    intervals = _memory_interval_by_value(kernel)
    if not groups or not intervals:
        return '<div class="empty-state">这个 Kernel 没有可用的 Realize StaticMemoryPlan workspace slot。</div>'

    starts = [_as_int(interval.get("start")) for interval in intervals.values()]
    ends = [_as_int(interval.get("end")) for interval in intervals.values()]
    min_time = min(starts or [0])
    max_time = max(ends or [min_time + 1])
    if max_time <= min_time:
        max_time = min_time + 1
    time_span = max_time - min_time
    left = 118
    top = 42
    step = 96
    row_h = 54
    lane_h = 28
    width = max(640, left + time_span * step + 70)
    height = top + len(groups) * row_h + 54

    parts = [
        f'<svg class="ub-allocation-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="UB 分配时间线">',
        f'<line class="axis" x1="{left}" y1="{top - 14}" x2="{left + time_span * step}" y2="{top - 14}"></line>',
    ]
    for tick in range(min_time, max_time + 1):
        x = left + (tick - min_time) * step
        parts.append(f'<line class="grid" x1="{x}" y1="{top - 22}" x2="{x}" y2="{height - 28}"></line>')
        parts.append(f'<text x="{x - 4}" y="{top - 26}">{_cell(tick)}</text>')

    for row_index, ((place, offset), slots) in enumerate(groups):
        y = top + row_index * row_h
        parts.append(f'<rect class="row-bg" x="12" y="{y - 18}" width="{width - 28}" height="{row_h - 8}" rx="6"></rect>')
        parts.append(f'<text x="20" y="{y + 4}">{_cell(place)}@{_cell(offset)}</text>')
        for slot in sorted(
            slots,
            key=lambda item: (
                _as_int(intervals.get(_as_int(item.get("value_id")), {}).get("start")),
                _as_int(item.get("slot_id")),
            ),
        ):
            value_id = _as_int(slot.get("value_id"))
            interval = intervals.get(value_id, {})
            start = _as_int(interval.get("start"), min_time)
            end = _as_int(interval.get("end"), start + 1)
            end = max(end, start + 1)
            x = left + (start - min_time) * step + 4
            block_w = max(28, (end - start) * step - 8)
            css_class = "slot-block reuse-group" if slot.get("reused") else "slot-block"
            label = f"slot {slot.get('slot_id')} value {value_id}"
            parts.append(
                f'<rect class="{css_class}" x="{x}" y="{y - 10}" width="{block_w}" height="{lane_h}" rx="5">'
                f"<title>{_cell(label)}; 生命周期 {start}..{end}; {slot.get('byte_size')} bytes</title>"
                "</rect>"
            )
            parts.append(f'<text x="{x + 7}" y="{y + 8}">{_cell(label)}</text>')

    for edge in kernel.get("movement_edges", []):
        if not isinstance(edge, dict):
            continue
        value_id = _as_int(edge.get("value_id"))
        interval = intervals.get(value_id, {})
        x = left + (_as_int(interval.get("start"), min_time) - min_time) * step + 10
        parts.append(
            f'<circle class="movement-edge" cx="{x}" cy="{height - 19}" r="5">'
            f"<title>{_cell(edge.get('src'))}->{_cell(edge.get('dst'))}; value {value_id}; slot {edge.get('slot_id')}</title>"
            "</circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _memory_reuse_group_chips(kernel: dict[str, Any]) -> str:
    chips = []
    for group in kernel.get("reuse_groups", []):
        if not isinstance(group, dict):
            continue
        chips.append(
            '<span class="chip reuse-group">'
            f"{_cell(group.get('place'))}@{_cell(group.get('offset'))}: "
            f"slots {_value_list(group.get('slot_ids'))}; values {_value_list(group.get('value_ids'))}"
            "</span>"
        )
    return "".join(chips) or '<span class="chip">无 slot 复用</span>'


def _memory_movement_chips(kernel: dict[str, Any]) -> str:
    chips = []
    for edge in kernel.get("movement_edges", []):
        if not isinstance(edge, dict):
            continue
        role = "缓存" if edge.get("dst") not in (None, "unknown", "GM") else "搬运"
        chips.append(
            '<span class="chip movement-edge">'
            f"step {_cell(edge.get('step_id'))}: value {_cell(edge.get('value_id'))} "
            f"slot {_cell(edge.get('slot_id'))} {_cell(edge.get('src'))}->{_cell(edge.get('dst'))} "
            f"{role} {_cell(edge.get('byte_size'))} bytes"
            "</span>"
        )
    return "".join(chips) or '<span class="chip">无 movement edge</span>'


def _render_memory_kernel_viz(kernel: dict[str, Any], *, heading_level: int, title: str) -> str:
    kernel_id = kernel.get("kernel_id")
    heading = f"h{heading_level}"
    svg = _render_ub_allocation_svg(kernel)
    return f"""
<section id="kernel-{_attr(kernel_id)}" class="memory-kernel-viz">
<div class="memory-kernel-header">
<{heading}>{_cell(title)}</{heading}>
<div class="memory-facts">
<span>workspace {_cell(kernel.get('workspace_bytes'))} B</span>
<span>峰值 {_cell(kernel.get('peak_usage_bytes'))} B</span>
<span>slot {_cell(kernel.get('workspace_slot_count'))}</span>
<span>生命周期 {_cell(kernel.get('live_interval_count'))}</span>
</div>
</div>
{svg}
<h3>UB 复用</h3>
<div>{_memory_reuse_group_chips(kernel)}</div>
<h3>搬运 / 缓存</h3>
<div>{_memory_movement_chips(kernel)}</div>
</section>
"""


def _memory_coverage_table(summary: dict[str, Any], view_rel_path: str) -> str:
    rows = []
    for kernel in summary.get("kernel_coverage", []):
        if not isinstance(kernel, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{_summary_kernel_link(view_rel_path, kernel.get('kernel_id'))}</td>"
            f"<td>{_cell(kernel.get('depth'))}</td>"
            f"<td>{_cell(kernel.get('dag_workspace_size'))}</td>"
            f"<td>{_cell(kernel.get('memory_plan_status'))}</td>"
            f"<td>{_cell(kernel.get('reason'))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="5">没有 Kernel 覆盖数据。</td></tr>')
    return f"""
<h2>Kernel 覆盖</h2>
<table>
<thead><tr><th>Kernel</th><th>DAG 深度</th><th>DAG workspace bytes</th><th>Memory plan</th><th>原因</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""


def _render_memory_summary_view(view_rel_path: str, rel_path: str, summary: dict[str, Any]) -> str:
    cards = "".join(
        [
            _summary_card("分析级别", summary.get("analysis_level")),
            _summary_card("Kernels", summary.get("kernel_count")),
            _summary_card("详细 Kernel", summary.get("detailed_kernel_count", summary.get("workspace_kernel_count", 0))),
            _summary_card("workspace 峰值", summary.get("peak_workspace_bytes")),
            _summary_card("复用组", summary.get("slot_reuse_group_count", 0)),
            _summary_card("搬运边", summary.get("movement_edge_count", 0)),
        ]
    )
    body = [f'<div class="summary-card-grid">{cards}</div>']
    note = summary.get("note")
    if note:
        body.append(f'<p class="note">{_cell(note)}</p>')

    if summary.get("analysis_level") == "realize-memory-plan":
        body.append(_memory_coverage_table(summary, view_rel_path))
        for kernel in summary.get("kernels", []):
            if isinstance(kernel, dict):
                title = f"Kernel {kernel.get('kernel_id')} UB 分配"
                body.append(_render_memory_kernel_viz(kernel, heading_level=2, title=title))
    else:
        body.append(
            '<div class="empty-state">没有找到 Realize StaticMemoryPlan，因此无法绘制 UB slot 复用和生命周期。'
            "当前页面仅展示 DAG workspace 概览。</div>"
        )
        rows = []
        for kernel in summary.get("kernels", []):
            if not isinstance(kernel, dict):
                continue
            rows.append(
                f'<tr id="kernel-{_attr(kernel.get("kernel_id"))}">'
                f"<td>{_summary_kernel_link(view_rel_path, kernel.get('kernel_id'))}</td>"
                f"<td>{_cell(kernel.get('depth'))}</td>"
                f"<td>{_cell(kernel.get('workspace_size'))}</td>"
                f"<td>{_cell(kernel.get('kind'))}</td>"
                f"<td>{_cell(kernel.get('selected_tile_shape'))}</td>"
                "</tr>"
            )
        workspace_rows = "".join(rows) or '<tr><td colspan="5">没有 workspace 数据。</td></tr>'
        body.append(
            "<h2>Workspace 概览</h2>"
            "<table><thead><tr><th>Kernel</th><th>深度</th><th>Workspace bytes</th><th>类型</th><th>Tile</th></tr></thead>"
            f"<tbody>{workspace_rows}</tbody></table>"
        )

    return _summary_page_document(
        view_rel_path=view_rel_path,
        rel_path=rel_path,
        title="Memory 摘要",
        body="\n".join(body),
    )


def _render_tensor_diff_summary_view(view_rel_path: str, rel_path: str, summary: dict[str, Any]) -> str:
    status = summary.get("status")
    cards = "".join(
        [
            _summary_card("状态", status, "fail" if status == "fail" else "pass"),
            _summary_card("对比项", summary.get("comparison_count")),
            _summary_card("失败项", summary.get("failed_count"), "fail" if _as_int(summary.get("failed_count")) else ""),
        ]
    )
    rows = []
    for item in summary.get("comparisons", []):
        if not isinstance(item, dict):
            continue
        status_class = "status-fail" if item.get("status") == "fail" else "status-pass"
        rows.append(
            "<tr>"
            f'<td class="{status_class}">{_cell(item.get("status"))}</td>'
            f"<td>{_cell(item.get('id'))}</td>"
            f"<td>{_summary_kernel_link(view_rel_path, item.get('kernel_id'))}</td>"
            f"<td>{_cell(item.get('task_id'))}</td>"
            f"<td>{_cell(item.get('max_abs_error'))}</td>"
            f"<td>{_cell(item.get('max_rel_error'))}</td>"
            f"<td>{_cell(item.get('mean_abs_error'))}</td>"
            "</tr>"
        )
    body = f"""
<div class="summary-card-grid">{cards}</div>
<h2>对比结果</h2>
<table>
<thead><tr><th>状态</th><th>ID</th><th>Kernel</th><th>Task</th><th>max_abs_error</th><th>max_rel_error</th><th>mean_abs_error</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan="7">没有对比结果。</td></tr>'}</tbody>
</table>
"""
    return _summary_page_document(
        view_rel_path=view_rel_path,
        rel_path=rel_path,
        title="Tensor Diff 摘要",
        body=body,
    )


def _render_locate_summary_view(view_rel_path: str, rel_path: str, summary: dict[str, Any]) -> str:
    context = summary.get("first_bad_context", {})
    if not isinstance(context, dict):
        context = {}
    cards = "".join(
        [
            _summary_card("状态", summary.get("status"), "fail" if summary.get("status") == "fail" else "pass"),
            _summary_card("首个异常 Kernel", summary.get("first_bad_kernel")),
            _summary_card("首个异常深度", summary.get("first_bad_depth")),
            _summary_card("失败 Kernel 数", summary.get("failed_kernel_count")),
        ]
    )
    body = f"""
<div class="summary-card-grid">{cards}</div>
<section class="panel">
<h2>首个异常候选</h2>
<table>
<tbody>
<tr><th>Kernel</th><td>{_summary_kernel_link(view_rel_path, summary.get('first_bad_kernel'))}</td></tr>
<tr><th>对比项</th><td>{_cell((summary.get('first_bad_comparison') or {}).get('id') if isinstance(summary.get('first_bad_comparison'), dict) else None)}</td></tr>
<tr><th>方法</th><td>{_cell(summary.get('method'))}</td></tr>
</tbody>
</table>
</section>
<section class="panel">
<h2>DAG 上下文</h2>
<table>
<tbody>
<tr><th>直接上游</th><td>{_value_list(context.get('direct_upstream'))}</td></tr>
<tr><th>已通过的上游 checkpoint</th><td>{_value_list(context.get('upstream_checked_passed'))}</td></tr>
<tr><th>未 checkpoint 的直接上游</th><td>{_value_list(context.get('unchecked_direct_upstream'))}</td></tr>
<tr><th>直接下游</th><td>{_value_list(context.get('direct_downstream'))}</td></tr>
<tr><th>失败的下游 checkpoint</th><td>{_value_list(context.get('downstream_failed'))}</td></tr>
</tbody>
</table>
</section>
"""
    return _summary_page_document(
        view_rel_path=view_rel_path,
        rel_path=rel_path,
        title="Locate 摘要",
        body=body,
    )


def _render_debug_graph_summary_view(view_rel_path: str, rel_path: str, summary: dict[str, Any]) -> str:
    primary = summary.get("primary_stage", {})
    if not isinstance(primary, dict):
        primary = {}
    kernel_dag = summary.get("kernel_dag", {})
    if not isinstance(kernel_dag, dict):
        kernel_dag = {}
    body = f"""
<div class="summary-card-grid">
{_summary_card("Stage 数", summary.get('stage_count'))}
{_summary_card("主 Stage", primary.get('name'))}
{_summary_card("Kernels", kernel_dag.get('kernel_count'))}
{_summary_card("Kernel 边", kernel_dag.get('graph_edges'))}
</div>
<section class="panel">
<h2>调试工作台</h2>
<p><a href="{html.escape(_relative_href(view_rel_path, 'views/debug_graph.html'), quote=True)}">打开调试工作台</a></p>
</section>
"""
    return _summary_page_document(
        view_rel_path=view_rel_path,
        rel_path=rel_path,
        title="Debug Graph 摘要",
        body=body,
    )


def _render_typed_json_view(view_rel_path: str, rel_path: str, parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return None
    if rel_path == "summaries/memory.json":
        return _render_memory_summary_view(view_rel_path, rel_path, parsed)
    if rel_path == "summaries/tensor_diff.json":
        return _render_tensor_diff_summary_view(view_rel_path, rel_path, parsed)
    if rel_path == "summaries/locate.json":
        return _render_locate_summary_view(view_rel_path, rel_path, parsed)
    if rel_path == "summaries/debug_graph.json":
        return _render_debug_graph_summary_view(view_rel_path, rel_path, parsed)
    return None


def _render_json_view(run_dir: pathlib.Path, rel_path: str) -> str | None:
    source_path = run_dir / rel_path
    if not source_path.exists():
        return None
    view_rel_path = _stage_view_rel_path(rel_path)
    view_path = run_dir / view_rel_path
    raw_href = html.escape(_relative_href(view_rel_path, rel_path), quote=True)
    dashboard_href = html.escape(_relative_href(view_rel_path, "index.html"), quote=True)
    try:
        source_text = source_path.read_text(encoding="utf-8")
        parsed = json.loads(source_text)
        typed_document = _render_typed_json_view(view_rel_path, rel_path, parsed)
        if typed_document:
            layout.write_text(view_path, typed_document)
            return view_rel_path
        source_text = json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as error:
        raise CommandError(f"JSON view source is not valid JSON: {source_path}: {error}") from error
    except (OSError, UnicodeDecodeError) as error:
        raise CommandError(f"could not read JSON view source: {source_path}: {error}") from error

    lines = source_text.splitlines() or [""]
    line_rows = []
    for line_number, line in enumerate(lines, start=1):
        escaped_line = html.escape(line)
        line_rows.append(
            '<tr class="line-row">'
            f'<td class="gutter"><a href="#L{line_number}" id="L{line_number}">'
            f'<span class="line-number">{line_number}</span></a></td>'
            f'<td class="code"><pre>{escaped_line}</pre></td>'
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_cell(pathlib.PurePosixPath(rel_path).name)} - ascend-debug</title>
<style>
:root {{ color-scheme: light; }}
body {{ font-family: sans-serif; margin: 0; color: #17202a; background: #eef2f7; }}
header {{ position: sticky; top: 0; z-index: 1; padding: 0.75rem 1rem; background: #ffffff; border-bottom: 1px solid #cbd5e1; }}
h1 {{ font-size: 1rem; margin: 0 0 0.5rem 0; }}
.toolbar {{ display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }}
input {{ min-width: 18rem; padding: 0.35rem 0.5rem; border: 1px solid #94a3b8; border-radius: 4px; color: #17202a; background: #ffffff; }}
main {{ padding: 0.75rem 1rem 2rem; }}
table.code-table {{ border-collapse: collapse; width: 100%; background: #0b1020; color: #dbeafe; border: 1px solid #1e293b; }}
.code-table td {{ vertical-align: top; border-bottom: 1px solid #1e293b; }}
.gutter {{ width: 4.5rem; text-align: right; padding: 0 0.65rem; background: #111827; user-select: none; }}
.gutter a {{ color: #93a4bd; text-decoration: none; }}
.code {{ padding-left: 0.75rem; }}
.code pre {{ margin: 0; padding: 0.12rem 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.5 SFMono-Regular, Menlo, Consolas, monospace; color: #dbeafe; background: transparent; }}
.line-row:hover pre {{ background: #172033; }}
.hidden {{ display: none; }}
.match pre {{ background: #1d4ed8; color: #ffffff; }}
</style>
</head>
<body>
<header>
<h1>{_cell(rel_path)}</h1>
<div class="toolbar">
<input id="search" type="search" placeholder="搜索 JSON">
<a href="{raw_href}">原始 JSON</a>
<a href="{dashboard_href}">调试首页</a>
</div>
</header>
<main>
<table class="code-table text-code-table">
<tbody>
{''.join(line_rows)}
</tbody>
</table>
</main>
<script>
const input = document.getElementById("search");
const rows = Array.from(document.querySelectorAll(".line-row"));
input.addEventListener("input", () => {{
  const needle = input.value.toLowerCase();
  for (const row of rows) {{
    const text = row.innerText.toLowerCase();
    const matched = !needle || text.includes(needle);
    row.classList.toggle("hidden", !matched);
    row.classList.toggle("match", Boolean(needle && matched));
  }}
}});
</script>
</body>
</html>
"""
    layout.write_text(view_path, document)
    return view_rel_path


def _render_stage_views(run_dir: pathlib.Path, manifest: dict[str, Any]) -> dict[str, str]:
    stage_views = {}
    for stage in sorted(manifest["stages"], key=lambda item: item["order"]):
        rel_path = stage["path"]
        view_rel_path = _render_mlir_view(run_dir, rel_path)
        if view_rel_path:
            stage_views[rel_path] = view_rel_path
    return stage_views


def _render_graph_mlir_views(run_dir: pathlib.Path, manifest: dict[str, Any]) -> dict[str, str]:
    graph_views = {}
    for graph in manifest.get("graphs", []):
        rel_path = graph.get("path")
        if isinstance(rel_path, str) and rel_path.endswith(".mlir"):
            view_rel_path = _render_mlir_view(run_dir, rel_path)
            if view_rel_path:
                graph_views[rel_path] = view_rel_path
    return graph_views


def _render_json_views(run_dir: pathlib.Path, manifest: dict[str, Any]) -> dict[str, str]:
    json_paths = {
        rel_path
        for rel_path in ("summaries/memory.json", "summaries/tensor_diff.json", "summaries/locate.json")
        if (run_dir / rel_path).exists()
    }
    json_views = {}
    for rel_path in sorted(json_paths):
        view_rel_path = _render_json_view(run_dir, rel_path)
        if view_rel_path:
            json_views[rel_path] = view_rel_path
    return json_views


def _load_kernel_summary(run_dir: pathlib.Path) -> dict[str, Any] | None:
    summary_path = run_dir / "graphs/kernel_dag.summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CommandError(f"could not read kernel DAG summary: {summary_path}: {error}") from error
    if not isinstance(summary, dict):
        raise CommandError(f"kernel DAG summary must be a JSON object: {summary_path}")
    nodes = summary.get("nodes")
    if not isinstance(nodes, dict):
        raise CommandError(f"kernel DAG summary nodes must be an object: {summary_path}")
    return summary


def _write_memory_summary(run_dir: pathlib.Path, summary: dict[str, Any] | None) -> dict[str, Any] | None:
    memory_summary = memory.summarize_memory(run_dir, summary)
    if memory_summary:
        layout.write_json(run_dir / "summaries/memory.json", memory_summary)
    return memory_summary


def _kernel_sort_key(kernel_id: str) -> tuple[int, str]:
    suffix = ""
    for char in reversed(kernel_id):
        if not char.isdigit():
            break
        suffix = char + suffix
    return (int(suffix), kernel_id) if suffix else (10**9, kernel_id)


def _kernel_edge_maps(summary: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    pred: dict[str, list[str]] = {}
    succ: dict[str, list[str]] = {}
    for kernel_id in summary.get("nodes", {}):
        pred[kernel_id] = []
        succ[kernel_id] = []
    for edge in summary.get("edges", []):
        if not isinstance(edge, dict):
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if isinstance(src, str) and isinstance(dst, str):
            succ.setdefault(src, []).append(dst)
            pred.setdefault(dst, []).append(src)
    for item in pred.values():
        item.sort(key=_kernel_sort_key)
    for item in succ.values():
        item.sort(key=_kernel_sort_key)
    return pred, succ


def _kernel_link_list(kernel_ids: list[str], current_view: str) -> str:
    if not kernel_ids:
        return ""
    links = []
    for kernel_id in kernel_ids:
        href = _relative_href(current_view, f"views/kernels/{kernel_id}.html")
        links.append(_link(href, kernel_id))
    return ", ".join(links)


def _render_kernel_views(
    run_dir: pathlib.Path,
    summary: dict[str, Any] | None,
    graph_views: dict[str, str],
    json_views: dict[str, str],
    memory_summary: dict[str, Any] | None,
) -> dict[str, str]:
    if not summary:
        return {}
    pred, succ = _kernel_edge_maps(summary)
    kernel_views = {}
    kernelized_view = graph_views.get("graphs/kernelized.mlir")
    for kernel_id in sorted(summary["nodes"], key=_kernel_sort_key):
        node = summary["nodes"][kernel_id]
        if not isinstance(node, dict):
            continue
        view_rel_path = f"views/kernels/{kernel_id}.html"
        dashboard_href = html.escape(_relative_href(view_rel_path, "index.html"), quote=True)
        dag_href = html.escape(_relative_href(view_rel_path, "graphs/kernel_dag.svg"), quote=True)
        memory_kernel = _memory_kernel_by_id(memory_summary, kernel_id)
        memory_href = ""
        memory_link = ""
        memory_view_path = json_views.get("summaries/memory.json")
        if memory_view_path and _memory_has_kernel_anchor(memory_summary, kernel_id):
            memory_href = html.escape(
                f"{_relative_href(view_rel_path, memory_view_path)}#kernel-{kernel_id}",
                quote=True,
            )
            memory_link = f'<a href="{memory_href}">内存视图</a>'
        fact_rows = []
        for label, value in (
            ("kind", node.get("kind")),
            ("depth", node.get("depth")),
            ("input_degree", node.get("input_degree")),
            ("output_degree", node.get("output_degree")),
            ("output_shape", node.get("output_shape")),
            ("output_dtype", node.get("output_dtype")),
            ("selected_tile_shape", node.get("selected_tile_shape")),
            ("workspace_size", node.get("workspace_size")),
            ("is_root", node.get("is_root")),
            ("is_leaf", node.get("is_leaf")),
            ("is_prepack_candidate_root", node.get("is_prepack_candidate_root")),
            ("touches_simple_fusion_edge", node.get("touches_simple_fusion_edge")),
        ):
            fact_rows.append(f"<tr><th>{_cell(label)}</th><td>{_cell(value)}</td></tr>")

        op_rows = []
        for op in node.get("ops", []):
            if not isinstance(op, dict):
                continue
            line = op.get("line")
            line_cell = _cell(line)
            if isinstance(line, int) and kernelized_view:
                href = f"{_relative_href(view_rel_path, kernelized_view)}#L{line}"
                line_cell = _link(href, str(line))
            op_rows.append(
                "<tr>"
                f"<td>{line_cell}</td>"
                f"<td>{_cell(op.get('op'))}</td>"
                f"<td>{_cell(op.get('label'))}</td>"
                f"<td>{_cell(op.get('role'))}</td>"
                f"<td>{_cell(op.get('result_type'))}</td>"
                "</tr>"
            )
        if not op_rows:
            op_rows.append('<tr><td colspan="5">没有 MLIR op 摘要。</td></tr>')

        upstream = _kernel_link_list(pred.get(kernel_id, []), view_rel_path)
        downstream = _kernel_link_list(succ.get(kernel_id, []), view_rel_path)
        memory_section = ""
        if memory_kernel:
            memory_section = _render_memory_kernel_viz(
                memory_kernel,
                heading_level=2,
                title="UB 分配",
            )
        document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{_cell(kernel_id)} - ascend-debug</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #17202a; background: #f8fafc; }}
table {{ border-collapse: collapse; width: 100%; background: #ffffff; margin: 0 0 1rem; }}
th, td {{ border: 1px solid #cbd5e1; padding: 0.4rem 0.55rem; text-align: left; vertical-align: top; }}
th {{ background: #f1f5f9; }}
.toolbar {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 1rem; }}
.mono {{ font-family: SFMono-Regular, Menlo, Consolas, monospace; }}
.memory-kernel-viz {{ border: 1px solid #cbd5e1; border-radius: 8px; background: #ffffff; padding: 0.85rem; margin: 1rem 0; overflow: auto; }}
.memory-kernel-header {{ display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: baseline; justify-content: space-between; margin-bottom: 0.65rem; }}
.memory-kernel-header h2 {{ margin: 0; }}
.memory-facts {{ display: flex; gap: 0.45rem; flex-wrap: wrap; color: #475569; font-size: 0.82rem; }}
.chip {{ display: inline-block; border: 1px solid #cbd5e1; border-radius: 999px; padding: 0.16rem 0.45rem; background: #ffffff; color: #334155; margin: 0.12rem; font-size: 0.78rem; }}
.reuse-group {{ border-color: #86efac; background: #f0fdf4; color: #047857; }}
.movement-edge {{ border-color: #bfdbfe; background: #eff6ff; color: #1d4ed8; }}
.ub-allocation-svg {{ display: block; min-width: 640px; max-width: none; background: #fbfdff; border: 1px solid #dbe3ee; border-radius: 8px; }}
.ub-allocation-svg text {{ font-family: SFMono-Regular, Menlo, Consolas, monospace; font-size: 11px; fill: #334155; }}
.ub-allocation-svg .axis {{ stroke: #94a3b8; stroke-width: 1; }}
.ub-allocation-svg .grid {{ stroke: #dbe3ee; stroke-width: 1; }}
.ub-allocation-svg .row-bg {{ fill: #f8fafc; stroke: #e2e8f0; }}
.ub-allocation-svg .slot-block {{ fill: #60a5fa; stroke: #1d4ed8; stroke-width: 1.2; }}
.ub-allocation-svg .slot-block.reuse-group {{ fill: #86efac; stroke: #059669; }}
.ub-allocation-svg .movement-edge {{ fill: #f59e0b; stroke: #b45309; }}
</style>
</head>
<body>
<h1>{_cell(kernel_id)}</h1>
<div class="toolbar">
<a href="{dashboard_href}">调试首页</a>
<a href="{dag_href}">Kernel DAG</a>
{memory_link}
</div>
<h2>Kernel 基本信息</h2>
<table><tbody>{''.join(fact_rows)}</tbody></table>
<h2>DAG 上下文</h2>
<table><tbody>
<tr><th>上游</th><td>{upstream}</td></tr>
<tr><th>下游</th><td>{downstream}</td></tr>
</tbody></table>
<h2>MLIR Ops</h2>
<table>
<thead><tr><th>行号</th><th>Operation</th><th>标签</th><th>角色</th><th>结果类型</th></tr></thead>
<tbody>{''.join(op_rows)}</tbody>
</table>
{memory_section}
</body>
</html>
"""
        layout.write_text(run_dir / view_rel_path, document)
        kernel_views[kernel_id] = view_rel_path
    return kernel_views


def _kernel_rows(summary: dict[str, Any] | None, kernel_views: dict[str, str]) -> str:
    if not summary or not kernel_views:
        return ""
    rows = []
    for kernel_id in sorted(kernel_views, key=_kernel_sort_key):
        node = summary["nodes"].get(kernel_id, {})
        rows.append(
            "<tr>"
            f"<td>{_link(kernel_views[kernel_id], kernel_id)}</td>"
            f"<td>{_cell(node.get('kind'))}</td>"
            f"<td>{_cell(node.get('depth'))}</td>"
            f"<td>{_cell(node.get('output_shape'))}</td>"
            f"<td>{_cell(node.get('selected_tile_shape'))}</td>"
            f"<td>{_cell(node.get('workspace_size'))}</td>"
            f"<td>{_cell(len(node.get('ops', [])))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _id_list_cell(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ", ".join(_cell(item) for item in value)


def _physical_slot_cell(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    labels = []
    for slot in value:
        if not isinstance(slot, dict):
            continue
        labels.append(f"{_cell(slot.get('offset'))}@{_cell(slot.get('place'))}")
    return ", ".join(labels) if labels else "none"


def _memory_overview_rows(summary: dict[str, Any] | None, kernel_views: dict[str, str]) -> str:
    if not summary:
        return ""
    rows = []
    for kernel in summary.get("kernels", []):
        if not isinstance(kernel, dict):
            continue
        kernel_id = kernel.get("kernel_id")
        kernel_cell = _cell(kernel_id)
        if isinstance(kernel_id, str) and kernel_id in kernel_views:
            kernel_cell = _link(kernel_views[kernel_id], kernel_id)
        rows.append(
            "<tr>"
            f"<td>{kernel_cell}</td>"
            f"<td>{_cell(kernel.get('depth'))}</td>"
            f"<td>{_cell(kernel.get('workspace_size'))}</td>"
            f"<td>{_cell(kernel.get('kind'))}</td>"
            f"<td>{_cell(kernel.get('input_degree'))}</td>"
            f"<td>{_cell(kernel.get('output_degree'))}</td>"
            f"<td>{_cell(kernel.get('selected_tile_shape'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _memory_timeline_rows(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    rows = []
    for kernel in summary.get("kernels", []):
        if not isinstance(kernel, dict):
            continue
        kernel_id = kernel.get("kernel_id")
        for item in kernel.get("peak_timeline", []):
            if not isinstance(item, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{_cell(kernel_id)}</td>"
                f"<td>{_cell(item.get('time'))}</td>"
                f"<td>{_cell(item.get('usage_bytes'))}</td>"
                f"<td>{_physical_slot_cell(item.get('active_physical_slots'))}</td>"
                f"<td>{_id_list_cell(item.get('active_values'))}</td>"
                "</tr>"
            )
    return "\n".join(rows)


def _memory_slot_rows(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    rows = []
    for kernel in summary.get("kernels", []):
        if not isinstance(kernel, dict):
            continue
        kernel_id = kernel.get("kernel_id")
        for slot in kernel.get("workspace_slots", []):
            if not isinstance(slot, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{_cell(kernel_id)}</td>"
                f"<td>{_cell(slot.get('slot_id'))}</td>"
                f"<td>{_cell(slot.get('place'))}</td>"
                f"<td>{_cell(slot.get('offset'))}</td>"
                f"<td>{_cell(slot.get('byte_size'))}</td>"
                f"<td>{_cell(slot.get('value_id'))}</td>"
                f"<td>{_cell(slot.get('live_range'))}</td>"
                f"<td>{'yes' if slot.get('reused') else 'no'}</td>"
                "</tr>"
            )
    return "\n".join(rows)


def _memory_coverage_rows(summary: dict[str, Any] | None, kernel_views: dict[str, str]) -> str:
    if not summary:
        return ""
    rows = []
    for kernel in summary.get("kernel_coverage", []):
        if not isinstance(kernel, dict):
            continue
        kernel_id = kernel.get("kernel_id")
        kernel_cell = _cell(kernel_id)
        if isinstance(kernel_id, str) and kernel_id in kernel_views:
            kernel_cell = _link(kernel_views[kernel_id], kernel_id)
        rows.append(
            "<tr>"
            f"<td>{kernel_cell}</td>"
            f"<td>{_cell(kernel.get('depth'))}</td>"
            f"<td>{_cell(kernel.get('dag_workspace_size'))}</td>"
            f"<td>{_cell(kernel.get('memory_plan_status'))}</td>"
            f"<td>{_cell(kernel.get('reason'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _memory_section(summary: dict[str, Any] | None, kernel_views: dict[str, str]) -> str:
    if not summary:
        return ""
    if summary.get("analysis_level") == "realize-memory-plan":
        coverage_rows = _memory_coverage_rows(summary, kernel_views)
        timeline_rows = _memory_timeline_rows(summary)
        slot_rows = _memory_slot_rows(summary)
        if not coverage_rows and not timeline_rows and not slot_rows:
            return ""
        return f"""
<section>
<h2>Memory</h2>
<p class="memory-note"><strong>Realize memory plan</strong>. {_cell(summary.get('note'))}</p>
<dl>
<dt>workspace 峰值 bytes</dt><dd>{_cell(summary.get('peak_workspace_bytes'))}</dd>
<dt>workspace 总量 bytes</dt><dd>{_cell(summary.get('total_workspace_bytes'))}</dd>
<dt>workspace slot 复用组</dt><dd>{_cell(summary.get('slot_reuse_group_count'))}</dd>
<dt>搬运边</dt><dd>{_cell(summary.get('movement_edge_count'))}</dd>
</dl>
<h3>Kernel 覆盖</h3>
<table>
<thead><tr><th>Kernel</th><th>DAG 深度</th><th>DAG workspace bytes</th><th>Memory plan</th><th>原因</th></tr></thead>
<tbody>
{coverage_rows}
</tbody>
</table>
<h3>峰值时间线</h3>
<table>
<thead><tr><th>Kernel</th><th>时间</th><th>占用 bytes</th><th>活跃 physical slot</th><th>活跃 value</th></tr></thead>
<tbody>
{timeline_rows}
</tbody>
</table>
<h3>Workspace Slot</h3>
<table>
<thead><tr><th>Kernel</th><th>Slot</th><th>Place</th><th>Offset</th><th>Bytes</th><th>Value</th><th>生命周期</th><th>复用</th></tr></thead>
<tbody>
{slot_rows}
</tbody>
</table>
</section>
"""

    rows = _memory_overview_rows(summary, kernel_views)
    if not rows:
        return ""
    return f"""
<section>
<h2>Memory</h2>
<p class="memory-note">{_cell(summary.get('note'))}</p>
<dl>
<dt>workspace 峰值 bytes</dt><dd>{_cell(summary.get('peak_workspace_bytes'))}</dd>
<dt>workspace 总量 bytes</dt><dd>{_cell(summary.get('total_workspace_bytes'))}</dd>
<dt>有 workspace 的 Kernel</dt><dd>{_cell(summary.get('workspace_kernel_count'))}</dd>
</dl>
<table>
<thead><tr><th>Kernel</th><th>深度</th><th>Workspace bytes</th><th>类型</th><th>输入</th><th>输出</th><th>Tile</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</section>
"""


def render_index(run_dir: pathlib.Path, manifest: dict[str, Any]) -> pathlib.Path:
    index_path = run_dir / "index.html"
    stage_views = _render_stage_views(run_dir, manifest)
    stage_graph_views = stage_graph.render_stage_graphs(run_dir, manifest["stages"])
    graph_views = _render_graph_mlir_views(run_dir, manifest)
    kernel_summary = _load_kernel_summary(run_dir)
    memory_summary = _write_memory_summary(run_dir, kernel_summary)
    tensor_diff = _load_tensor_diff(run_dir)
    locate_summary = _load_locate_summary(run_dir)
    debug_graph_view = debug_graph.render_debug_graph(
        run_dir=run_dir,
        manifest=manifest,
        stage_graph_views=stage_graph_views,
        kernel_summary=kernel_summary,
        tensor_diff=tensor_diff,
        locate_summary=locate_summary,
        memory_summary=memory_summary,
    )
    json_views = _render_json_views(run_dir, manifest)
    kernel_views = _render_kernel_views(
        run_dir,
        kernel_summary,
        graph_views,
        json_views,
        memory_summary,
    )
    command_section = ""
    if manifest.get("commands"):
        command_section = f"""
<details class="advanced-section">
<summary>高级信息：执行命令</summary>
<table>
<thead><tr><th>Stage</th><th>Tool</th><th>Args</th><th>状态</th><th>报告</th></tr></thead>
<tbody>
{_command_rows(run_dir, manifest)}
</tbody>
</table>
</details>
"""
    debug_graph_section = f"""
<section class="primary-debug-section">
<a class="primary-debug-link" href="{_cell(debug_graph_view['view_path'])}">打开调试工作台</a>
<span>统一查看 Stage 演进、Kernel DAG、Tensor Diff、Locate 和 Memory。</span>
</section>
"""
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>ascend-debug</title>
<style>
body {{ font-family: sans-serif; margin: 0; color: #1f2933; background: #f8fafc; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 2rem; }}
header {{ background: #ffffff; border-bottom: 1px solid #dbe3ee; }}
header .inner {{ max-width: 1120px; margin: 0 auto; padding: 1.2rem 2rem; }}
h1 {{ margin: 0; font-size: 1.55rem; }}
section, details {{ margin: 1rem 0; }}
table {{ border-collapse: collapse; width: 100%; background: #ffffff; }}
th, td {{ border: 1px solid #cbd5e1; padding: 0.45rem 0.6rem; text-align: left; }}
th {{ background: #f1f5f9; }}
.overview-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: 0.75rem; }}
.overview-card {{ border: 1px solid #dbe3ee; background: #ffffff; border-radius: 8px; padding: 0.7rem; }}
.overview-card span {{ display: block; color: #64748b; font-size: 0.78rem; font-weight: 700; margin-bottom: 0.25rem; }}
.overview-card strong {{ display: block; font-size: 1.1rem; overflow-wrap: anywhere; }}
.primary-debug-section {{ display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; border: 1px solid #bfdbfe; background: #eff6ff; border-radius: 8px; padding: 0.9rem; margin: 1rem 0; }}
.primary-debug-link {{ display: inline-block; padding: 0.45rem 0.7rem; background: #1d4ed8; color: #ffffff; border-radius: 6px; text-decoration: none; font-weight: 700; }}
.advanced-section {{ border: 1px solid #dbe3ee; background: #ffffff; border-radius: 8px; padding: 0.75rem; }}
.advanced-section summary {{ cursor: pointer; font-weight: 700; }}
dt {{ font-weight: 700; float: left; clear: left; margin-right: 0.4rem; }}
dd {{ margin: 0 0 0.35rem 0; }}
.locate-note {{ margin: 0.25rem 0 0.75rem; color: #475569; }}
.memory-note {{ margin: 0.25rem 0 0.75rem; color: #475569; }}
.locate-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); gap: 0.75rem; }}
.locate-panel {{ border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; padding: 0.75rem; }}
.locate-panel h3 {{ margin: 0 0 0.65rem 0; font-size: 1rem; }}
.locate-panel dl {{ margin: 0; }}
.locate-panel dt {{ float: none; clear: none; margin: 0 0 0.15rem 0; color: #475569; font-weight: 700; }}
.locate-panel dd {{ margin: 0 0 0.6rem 0; }}
</style>
</head>
<body>
<header><div class="inner"><h1>Ascend Debug</h1></div></header>
<main>
{debug_graph_section}
<section>
<h2>运行概览</h2>
<div class="overview-grid">
{_overview_cards(manifest)}
</div>
</section>
<section>
<h2>Stage Timeline</h2>
<table>
<thead><tr><th>顺序</th><th>Stage</th><th>MLIR</th><th>调试图</th><th>状态</th></tr></thead>
<tbody>
{_stage_rows(run_dir, manifest, stage_views, debug_graph_view['view_path'])}
</tbody>
</table>
</section>
{command_section}
</main>
</body>
</html>
"""
    layout.write_text(index_path, document)
    return index_path


def open_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    index_path = render_index(run_dir, manifest)
    print(f"ascend-debug.open.index={index_path}")
    if not args.no_browser:
        try:
            webbrowser.open(index_path.resolve().as_uri())
        except Exception as error:  # pragma: no cover - depends on host browser setup.
            print(f"ascend-debug.open.browser=unavailable: {error}", file=sys.stderr)
    return 0
