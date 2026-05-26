from __future__ import annotations

import argparse
import html
import json
import pathlib
import posixpath
import sys
import webbrowser
from typing import Any

from ascend_debug import layout
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


def _stage_rows(
    run_dir: pathlib.Path,
    manifest: dict[str, Any],
    stage_views: dict[str, str],
) -> str:
    rows = []
    stages = sorted(manifest["stages"], key=lambda stage: stage["order"])
    for stage in stages:
        rel_path = str(stage["path"])
        exists = (run_dir / rel_path).exists()
        status = "present" if exists else "missing"
        view_rel_path = stage_views.get(rel_path)
        view_cell = _link(view_rel_path, "View") if view_rel_path else ""
        rows.append(
            "<tr>"
            f"<td>{_cell(stage['order'])}</td>"
            f"<td>{_cell(stage['name'])}</td>"
            f"<td>{_path_link(rel_path, exists=exists)}</td>"
            f"<td>{view_cell}</td>"
            f"<td>{_cell(status)}</td>"
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
        status = "present" if exists else "missing"
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
        status = "present" if exists else "missing"
        view_rel_path = artifact_views.get(rel_path)
        view_cell = _link(view_rel_path, "View") if view_rel_path else ""
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
        view_cell = _link(view_rel_path, "View") if view_rel_path else ""
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
<p class="locate-note">Earliest failed checkpoint in DAG order. This is a first-bad candidate, not root-cause proof when upstream kernels are not checked.</p>
<div class="locate-grid">
<div class="locate-panel">
<h3>First Bad Candidate</h3>
<dl>
<dt>Status</dt><dd>{_cell(summary.get('status'))}</dd>
<dt>Kernel</dt><dd>{kernel_cell}</dd>
<dt>DAG depth</dt><dd>{_cell(summary.get('first_bad_depth'))}</dd>
<dt>Failed comparison</dt><dd>{_cell(comparison_id)}</dd>
</dl>
</div>
<div class="locate-panel">
<h3>Evidence</h3>
<dl>
<dt>Failed kernels</dt><dd>{_cell(summary.get('failed_kernel_count'))}</dd>
<dt>Upstream passed checkpoints</dt><dd>{upstream_passed}</dd>
<dt>Downstream failed checkpoints</dt><dd>{downstream_failed}</dd>
<dt>Method</dt><dd>{_cell(summary.get('method'))}</dd>
</dl>
</div>
<div class="locate-panel">
<h3>Coverage Gap</h3>
<dl>
<dt>Direct upstream</dt><dd>{direct_upstream}</dd>
<dt>Direct downstream</dt><dd>{direct_downstream}</dd>
<dt>Direct upstream without checkpoint</dt><dd>{unchecked_upstream}</dd>
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
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_cell(pathlib.PurePosixPath(rel_path).name)} - ascend-debug</title>
<style>
body {{ font-family: sans-serif; margin: 0; color: #17202a; background: #f8fafc; }}
header {{ position: sticky; top: 0; z-index: 1; padding: 0.75rem 1rem; background: #ffffff; border-bottom: 1px solid #cbd5e1; }}
h1 {{ font-size: 1rem; margin: 0 0 0.5rem 0; }}
.toolbar {{ display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }}
input {{ min-width: 18rem; padding: 0.35rem 0.5rem; border: 1px solid #94a3b8; border-radius: 4px; }}
main {{ padding: 0.75rem 1rem 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #ffffff; }}
td {{ vertical-align: top; border-bottom: 1px solid #e2e8f0; }}
.gutter {{ width: 4.5rem; text-align: right; padding: 0 0.65rem; background: #f1f5f9; user-select: none; }}
.gutter a {{ color: #64748b; text-decoration: none; }}
.code {{ padding-left: 0.75rem; }}
pre {{ margin: 0; padding: 0.12rem 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.5 SFMono-Regular, Menlo, Consolas, monospace; }}
.hidden {{ display: none; }}
.match pre {{ background: #fef9c3; }}
</style>
</head>
<body>
<header>
<h1>{_cell(rel_path)}</h1>
<div class="toolbar">
<input id="search" type="search" placeholder="Search MLIR">
<a href="{raw_href}">Raw MLIR</a>
<a href="{dashboard_href}">Dashboard</a>
</div>
</header>
<main>
<table>
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
body {{ font-family: sans-serif; margin: 0; color: #17202a; background: #f8fafc; }}
header {{ position: sticky; top: 0; z-index: 1; padding: 0.75rem 1rem; background: #ffffff; border-bottom: 1px solid #cbd5e1; }}
h1 {{ font-size: 1rem; margin: 0 0 0.5rem 0; }}
.toolbar {{ display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }}
input {{ min-width: 18rem; padding: 0.35rem 0.5rem; border: 1px solid #94a3b8; border-radius: 4px; color: #17202a; background: #ffffff; }}
main {{ padding: 0.75rem 1rem 2rem; }}
table {{ border-collapse: collapse; width: 100%; background: #0f172a; color: #e2e8f0; }}
td {{ vertical-align: top; border-bottom: 1px solid #1e293b; }}
.gutter {{ width: 4.5rem; text-align: right; padding: 0 0.65rem; background: #111827; user-select: none; }}
.gutter a {{ color: #94a3b8; text-decoration: none; }}
.code {{ padding-left: 0.75rem; }}
pre {{ margin: 0; padding: 0.12rem 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.5 SFMono-Regular, Menlo, Consolas, monospace; color: #e2e8f0; }}
.hidden {{ display: none; }}
.match pre {{ background: #334155; color: #ffffff; }}
</style>
</head>
<body>
<header>
<h1>{_cell(rel_path)}</h1>
<div class="toolbar">
<input id="search" type="search" placeholder="Search JSON">
<a href="{raw_href}">Raw JSON</a>
<a href="{dashboard_href}">Dashboard</a>
</div>
</header>
<main>
<table>
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
    json_paths = set()
    for graph in manifest.get("graphs", []):
        rel_path = graph.get("path")
        if isinstance(rel_path, str) and rel_path.endswith(".json"):
            json_paths.add(rel_path)

    summary_dir = run_dir / "summaries"
    if summary_dir.exists():
        for path in summary_dir.rglob("*.json"):
            if path.is_file():
                json_paths.add(path.relative_to(run_dir).as_posix())

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
        summary_rel_path = json_views.get("graphs/kernel_dag.summary.json", "graphs/kernel_dag.summary.json")
        summary_href = html.escape(_relative_href(view_rel_path, summary_rel_path), quote=True)
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
            op_rows.append('<tr><td colspan="5">No MLIR op summary available.</td></tr>')

        upstream = _kernel_link_list(pred.get(kernel_id, []), view_rel_path)
        downstream = _kernel_link_list(succ.get(kernel_id, []), view_rel_path)
        document = f"""<!doctype html>
<html lang="en">
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
</style>
</head>
<body>
<h1>{_cell(kernel_id)}</h1>
<div class="toolbar">
<a href="{dashboard_href}">Dashboard</a>
<a href="{dag_href}">Kernel DAG</a>
<a href="{summary_href}">DAG summary JSON</a>
</div>
<h2>Kernel Facts</h2>
<table><tbody>{''.join(fact_rows)}</tbody></table>
<h2>DAG Context</h2>
<table><tbody>
<tr><th>upstream</th><td>{upstream}</td></tr>
<tr><th>downstream</th><td>{downstream}</td></tr>
</tbody></table>
<h2>MLIR Ops</h2>
<table>
<thead><tr><th>Line</th><th>Operation</th><th>Label</th><th>Role</th><th>Result Type</th></tr></thead>
<tbody>{''.join(op_rows)}</tbody>
</table>
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


def render_index(run_dir: pathlib.Path, manifest: dict[str, Any]) -> pathlib.Path:
    index_path = run_dir / "index.html"
    stage_views = _render_stage_views(run_dir, manifest)
    graph_views = _render_graph_mlir_views(run_dir, manifest)
    json_views = _render_json_views(run_dir, manifest)
    kernel_summary = _load_kernel_summary(run_dir)
    kernel_views = _render_kernel_views(run_dir, kernel_summary, graph_views, json_views)
    tensor_diff = _load_tensor_diff(run_dir)
    locate_summary = _load_locate_summary(run_dir)
    command_section = ""
    if manifest.get("commands"):
        command_section = f"""
<section>
<h2>Commands</h2>
<table>
<thead><tr><th>Stage</th><th>Tool</th><th>Args</th><th>Status</th><th>Report</th></tr></thead>
<tbody>
{_command_rows(run_dir, manifest)}
</tbody>
</table>
</section>
"""
    report_section = ""
    if manifest.get("reports"):
        report_section = f"""
<section>
<h2>Reports</h2>
<table>
<thead><tr><th>Stage</th><th>Path</th><th>Status</th></tr></thead>
<tbody>
{_report_rows(run_dir, manifest)}
</tbody>
</table>
</section>
"""
    graph_section = ""
    if manifest.get("graphs"):
        graph_section = f"""
<section>
<h2>Graphs</h2>
<table>
<thead><tr><th>Kind</th><th>Path</th><th>View</th><th>Status</th></tr></thead>
<tbody>
{_graph_rows(run_dir, manifest, {**graph_views, **json_views})}
</tbody>
</table>
</section>
"""
    kernel_rows = _kernel_rows(kernel_summary, kernel_views)
    kernel_section = ""
    if kernel_rows:
        kernel_section = f"""
<section>
<h2>Kernels</h2>
<table>
<thead><tr><th>Kernel</th><th>Kind</th><th>Depth</th><th>Output Shape</th><th>Tile</th><th>Workspace</th><th>Ops</th></tr></thead>
<tbody>
{kernel_rows}
</tbody>
</table>
</section>
"""
    tensor_diff_rows = _tensor_diff_rows(tensor_diff, kernel_views)
    tensor_diff_section = ""
    if tensor_diff_rows:
        tensor_diff_section = f"""
<section>
<h2>Tensor Diff</h2>
<p>status={_cell(tensor_diff.get('status'))}; comparisons={_cell(tensor_diff.get('comparison_count'))}; failed={_cell(tensor_diff.get('failed_count'))}</p>
<table>
<thead><tr><th>Status</th><th>ID</th><th>Kernel</th><th>Task</th><th>max_abs_error</th><th>max_rel_error</th><th>mean_abs_error</th><th>atol</th><th>rtol</th></tr></thead>
<tbody>
{tensor_diff_rows}
</tbody>
</table>
</section>
"""
    locate_section = _locate_section(locate_summary, kernel_views)
    summary_rows = _summary_rows(run_dir, json_views)
    summary_section = ""
    if summary_rows:
        summary_section = f"""
<section>
<h2>Summaries</h2>
<table>
<thead><tr><th>Path</th><th>View</th><th>Bytes</th></tr></thead>
<tbody>
{summary_rows}
</tbody>
</table>
</section>
"""
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ascend-debug</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #1f2933; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #cbd5e1; padding: 0.4rem 0.55rem; text-align: left; }}
th {{ background: #f1f5f9; }}
dt {{ font-weight: 700; float: left; clear: left; margin-right: 0.4rem; }}
dd {{ margin: 0 0 0.35rem 0; }}
.locate-note {{ margin: 0.25rem 0 0.75rem; color: #475569; }}
.locate-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); gap: 0.75rem; }}
.locate-panel {{ border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; padding: 0.75rem; }}
.locate-panel h3 {{ margin: 0 0 0.65rem 0; font-size: 1rem; }}
.locate-panel dl {{ margin: 0; }}
.locate-panel dt {{ float: none; clear: none; margin: 0 0 0.15rem 0; color: #475569; font-weight: 700; }}
.locate-panel dd {{ margin: 0 0 0.6rem 0; }}
</style>
</head>
<body>
<h1>ascend-debug</h1>
<section>
<h2>Run</h2>
<dl>
{_metadata_rows(manifest)}
</dl>
</section>
<section>
<h2>Stages</h2>
<table>
<thead><tr><th>Order</th><th>Stage</th><th>Path</th><th>View</th><th>Status</th></tr></thead>
<tbody>
{_stage_rows(run_dir, manifest, stage_views)}
</tbody>
</table>
</section>
{command_section}
{report_section}
{graph_section}
{kernel_section}
{tensor_diff_section}
{locate_section}
{summary_section}
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
