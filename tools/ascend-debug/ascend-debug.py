#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

from ascend_debug import __version__
from ascend_debug.collect import collect_run
from ascend_debug.diff import diff_run
from ascend_debug.open_view import open_run
from ascend_debug.runner import CommandError


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
    collect.add_argument("--preset", choices=["quick", "deep"], default="quick")
    collect.add_argument("--pipeline", choices=["normalize-kernelize"], default="normalize-kernelize")
    collect.add_argument("--runtime-manifest", type=pathlib.Path)
    collect.add_argument("--run-manifest", type=pathlib.Path)
    collect.add_argument("--kernelized-ir", type=pathlib.Path)
    collect.add_argument("--dag-viz", type=pathlib.Path)
    collect.set_defaults(handler=collect_run)

    open_cmd = subparsers.add_parser("open", help="Generate or open the debug dashboard")
    open_cmd.add_argument("run_dir", type=pathlib.Path)
    open_cmd.add_argument("--no-browser", action="store_true")
    open_cmd.set_defaults(handler=open_run)

    diff = subparsers.add_parser("diff", help="Compare collected tensors")
    diff.add_argument("run_dir", type=pathlib.Path)
    diff.set_defaults(handler=diff_run)

    locate = subparsers.add_parser("locate", help="Locate first bad kernel")
    locate.add_argument("run_dir", type=pathlib.Path)
    locate.set_defaults(handler=lambda args: parser.exit(2, "ascend-debug locate is implemented in a later slice\n"))

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
