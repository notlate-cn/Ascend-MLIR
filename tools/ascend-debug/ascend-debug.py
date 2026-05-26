#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

from ascend_debug import __version__


STAGES = (
    (0, "000-source.mlir"),
    (10, "010-normalize-in.mlir"),
    (19, "019-normalize-out.mlir"),
    (20, "020-kernelize-in.mlir"),
    (29, "029-kernelize-out.mlir"),
)


def collect_debug_run(args: argparse.Namespace) -> int:
    run_dir = args.out
    stages_dir = run_dir / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    for _, name in STAGES:
        shutil.copyfile(args.input, stages_dir / name)

    manifest = {
        "tool": "ascend-debug",
        "preset": args.preset,
        "pipeline": args.pipeline,
        "device_scope": "single_run_single_device",
        "stages": [
            {
                "order": order,
                "path": str(stages_dir / name),
            }
            for order, name in STAGES
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "version": __version__,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def open_debug_run(args: argparse.Namespace) -> int:
    index = args.run_dir / "index.html"
    index.write_text("<!doctype html><title>ascend-debug</title>\n")
    print(index)
    return 0


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
    collect.add_argument("--preset", choices=["quick"], default="quick")
    collect.add_argument("--pipeline", choices=["normalize-kernelize"], default="normalize-kernelize")
    collect.set_defaults(handler=collect_debug_run)

    open_cmd = subparsers.add_parser("open", help="Generate or open the debug dashboard")
    open_cmd.add_argument("run_dir", type=pathlib.Path)
    open_cmd.add_argument("--no-browser", action="store_true")
    open_cmd.set_defaults(handler=open_debug_run)

    diff = subparsers.add_parser("diff", help="Compare collected tensors")
    diff.add_argument("run_dir", type=pathlib.Path)
    diff.set_defaults(handler=lambda args: parser.exit(2, "ascend-debug diff is implemented in a later slice\n"))

    locate = subparsers.add_parser("locate", help="Locate first bad kernel")
    locate.add_argument("run_dir", type=pathlib.Path)
    locate.set_defaults(handler=lambda args: parser.exit(2, "ascend-debug locate is implemented in a later slice\n"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
