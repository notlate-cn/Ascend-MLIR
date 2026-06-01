from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
from typing import Any

from ascend_debug.runner import CommandError, find_tool


def _load_summary(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CommandError(f"tensor diff summary not found: {path}") from error
    except (OSError, UnicodeDecodeError) as error:
        raise CommandError(f"could not read tensor diff summary: {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CommandError(f"tensor diff summary is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise CommandError(f"tensor diff summary must be a JSON object: {path}")
    return value


def diff_run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "tensors" / "manifest.json"
    report_path = run_dir / "summaries" / "tensor_diff.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    runtime_session = find_tool("runtime-session")
    argv = [
        runtime_session,
        "--compare-tensors",
        str(manifest_path),
        "--emit-validation-summary",
        str(report_path),
    ]
    completed = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise CommandError(
            f"command failed with exit code {completed.returncode}: {shlex.join(argv)}\n{completed.stderr}",
            argv=tuple(argv),
            returncode=completed.returncode,
            stderr=completed.stderr,
        )

    summary = _load_summary(report_path)
    status = summary.get("status")
    failed_count = summary.get("failed_count", 0)
    comparison_count = summary.get("comparison_count", 0)
    if status not in ("pass", "fail"):
        raise CommandError(f"tensor diff summary has invalid status: {report_path}")

    print(f"ascend_debug.diff.comparisons={comparison_count}")
    print(f"ascend_debug.diff.failed={failed_count}")
    print(f"ascend_debug.diff.status={status}")
    print(f"ascend_debug.diff.report={report_path}")
    return 0 if status == "pass" else 1
