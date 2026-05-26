from __future__ import annotations

import argparse
import html
import json
import pathlib
import sys
import webbrowser
from typing import Any

from ascend_debug import layout
from ascend_debug.runner import CommandError


def load_manifest(run_dir: pathlib.Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CommandError(f"manifest not found: {manifest_path}") from error
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


def render_index(run_dir: pathlib.Path, manifest: dict[str, Any]) -> pathlib.Path:
    index_path = run_dir / "index.html"
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
