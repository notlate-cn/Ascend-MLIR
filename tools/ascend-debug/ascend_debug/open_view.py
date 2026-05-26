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


def _metadata_rows(manifest: dict[str, Any]) -> str:
    keys = ("schema_version", "tool", "preset", "pipeline", "backend", "device_id")
    rows = []
    for key in keys:
        if key in manifest:
            rows.append(f"<dt>{_cell(key)}</dt><dd>{_cell(manifest[key])}</dd>")
    return "\n".join(rows)


def _stage_rows(run_dir: pathlib.Path, manifest: dict[str, Any]) -> str:
    rows = []
    stages = sorted(manifest["stages"], key=lambda stage: stage["order"])
    for stage in stages:
        rel_path = str(stage["path"])
        status = "present" if (run_dir / rel_path).exists() else "missing"
        rows.append(
            "<tr>"
            f"<td>{_cell(stage['order'])}</td>"
            f"<td>{_cell(stage['name'])}</td>"
            f"<td>{_cell(rel_path)}</td>"
            f"<td>{_cell(status)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _command_rows(manifest: dict[str, Any]) -> str:
    rows = []
    for command in manifest.get("commands", []):
        args = " ".join(command.get("args", []))
        report_path = command.get("stderr") or command.get("stdout")
        rows.append(
            "<tr>"
            f"<td>{_cell(command.get('stage'))}</td>"
            f"<td>{_cell(command.get('tool'))}</td>"
            f"<td>{_cell(args)}</td>"
            f"<td>{_cell(command.get('status'))}</td>"
            f"<td>{_cell(report_path)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _report_rows(run_dir: pathlib.Path, manifest: dict[str, Any]) -> str:
    rows = []
    for report in manifest.get("reports", []):
        rel_path = report["path"]
        status = "present" if (run_dir / rel_path).exists() else "missing"
        rows.append(
            "<tr>"
            f"<td>{_cell(report.get('stage'))}</td>"
            f"<td>{_cell(rel_path)}</td>"
            f"<td>{_cell(status)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _graph_rows(run_dir: pathlib.Path, manifest: dict[str, Any]) -> str:
    rows = []
    for graph in manifest.get("graphs", []):
        rel_path = graph["path"]
        status = "present" if (run_dir / rel_path).exists() else "missing"
        rows.append(
            "<tr>"
            f"<td>{_cell(graph.get('kind'))}</td>"
            f"<td>{_cell(rel_path)}</td>"
            f"<td>{_cell(status)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_index(run_dir: pathlib.Path, manifest: dict[str, Any]) -> pathlib.Path:
    index_path = run_dir / "index.html"
    command_section = ""
    if manifest.get("commands"):
        command_section = f"""
<section>
<h2>Commands</h2>
<table>
<thead><tr><th>Stage</th><th>Tool</th><th>Args</th><th>Status</th><th>Report</th></tr></thead>
<tbody>
{_command_rows(manifest)}
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
<thead><tr><th>Kind</th><th>Path</th><th>Status</th></tr></thead>
<tbody>
{_graph_rows(run_dir, manifest)}
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
<thead><tr><th>Order</th><th>Stage</th><th>Path</th><th>Status</th></tr></thead>
<tbody>
{_stage_rows(run_dir, manifest)}
</tbody>
</table>
</section>
{command_section}
{report_section}
{graph_section}
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
