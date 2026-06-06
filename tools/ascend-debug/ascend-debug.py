#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

from ascend_debug import __version__
from ascend_debug.collect import collect_run
from ascend_debug.diff import diff_run
from ascend_debug.locate import locate_run
from ascend_debug.open_view import open_run
from ascend_debug.runner import CommandError
from ascend_debug.run_case import run_case
from ascend_debug.serve import serve_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ascend-debug",
        description="Collect and inspect Ascend-MLIR debug artifacts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect a debug run")
    collect.add_argument("input", type=pathlib.Path)
    collect.add_argument("--out", type=pathlib.Path, required=True)
    collect.add_argument(
        "--mode",
        choices=["quick", "deep"],
        default=None,
        help="Collection mode: quick emits coarse phase dumps; deep emits pass-level dumps.",
    )
    collect.add_argument(
        "--preset",
        choices=["quick", "deep"],
        default=None,
        help=argparse.SUPPRESS,
    )
    collect.add_argument(
        "--pipeline",
        choices=["normalize-kernelize", "full-codegen"],
        default=None,
        help=argparse.SUPPRESS,
    )
    collect.add_argument("--artifact-manifest", type=pathlib.Path)
    collect.add_argument("--run-manifest", type=pathlib.Path)
    collect.add_argument("--kernelized-ir", type=pathlib.Path)
    collect.add_argument(
        "--debug-contract-dir",
        type=pathlib.Path,
        help="Directory containing versioned Ascend Debug Contract JSON files.",
    )
    collect.add_argument(
        "--memory-detail",
        action="store_true",
        help="Use target-aware Realize options so memory.json can include UB/workspace slot lifetimes when available.",
    )
    collect.add_argument(
        "--cann-root",
        type=pathlib.Path,
        help=(
            "CANN root for --mode deep or --memory-detail. Defaults to "
            "ASCEND_HOME_PATH, ASCEND_HOME, CANN_ROOT, or ASCEND_TOOLKIT_HOME."
        ),
    )
    collect.add_argument(
        "--soc",
        help="SoC name for --mode deep or --memory-detail.",
    )
    collect.add_argument(
        "--realize-options",
        default="",
        help="Advanced extra option string appended inside --ascend-realize=...",
    )
    collect.set_defaults(handler=collect_run)

    open_cmd = subparsers.add_parser("open", help="Generate or open the debug dashboard")
    open_cmd.add_argument("run_dir", type=pathlib.Path)
    open_cmd.add_argument("--no-browser", action="store_true")
    open_cmd.set_defaults(handler=open_run)

    serve = subparsers.add_parser("serve", help="Serve a debug dashboard and refresh HTML views on request")
    serve.add_argument("run_dir", type=pathlib.Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-browser", action="store_true")
    serve.set_defaults(handler=serve_run)

    diff = subparsers.add_parser("diff", help="Compare collected tensors")
    diff.add_argument("run_dir", type=pathlib.Path)
    diff.set_defaults(handler=diff_run)

    locate = subparsers.add_parser("locate", help="Locate first bad kernel")
    locate.add_argument("run_dir", type=pathlib.Path)
    locate.set_defaults(handler=locate_run)

    run = subparsers.add_parser("run", help="Prepare and run a case.json")
    run.add_argument("case", type=pathlib.Path)
    run.add_argument("--out", type=pathlib.Path, required=True)
    run.add_argument(
        "--prepare-runtime-artifacts",
        action="store_true",
        help="Prepare runtime artifacts and run_manifest.json without executing the runtime session.",
    )
    run.set_defaults(handler=run_case)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except CommandError as error:
        print(f"ascend-debug: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
