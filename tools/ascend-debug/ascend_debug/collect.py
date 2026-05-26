from __future__ import annotations

import argparse

from ascend_debug import __version__, layout
from ascend_debug.runner import CommandError, find_tool, run_command


def _clear_quick_artifacts(run_dir) -> None:
    for stage in layout.QUICK_NORMALIZE_KERNELIZE_STAGES:
        (run_dir / stage.path).unlink(missing_ok=True)
    (run_dir / "manifest.json").unlink(missing_ok=True)
    (run_dir / "provenance.json").unlink(missing_ok=True)


def collect_quick(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    run_dir = args.out.resolve()

    stages = layout.QUICK_NORMALIZE_KERNELIZE_STAGES
    layout.prepare_run_dir(run_dir)
    _clear_quick_artifacts(run_dir)

    if not input_path.exists():
        raise CommandError(f"input MLIR does not exist: {input_path}")

    source = run_dir / stages[0].path
    normalize_in = run_dir / stages[1].path
    normalize_out = run_dir / stages[2].path
    kernelize_in = run_dir / stages[3].path
    kernelize_out = run_dir / stages[4].path

    layout.copy_stage(input_path, source)
    layout.copy_stage(source, normalize_in)

    opt = find_tool("ascend-mlir-opt")
    run_command([opt, str(normalize_in), "--ascend-normalize"], stdout_path=normalize_out)
    layout.copy_stage(normalize_out, kernelize_in)
    run_command([opt, str(kernelize_in), "--ascend-kernelize"], stdout_path=kernelize_out)

    layout.write_manifest(
        run_dir,
        preset=args.preset,
        pipeline=args.pipeline,
        stages=stages,
    )
    layout.write_provenance_skeleton(
        run_dir,
        original_input=args.input,
        version=__version__,
    )
    print(f"ascend-debug.collect.out={run_dir}")
    return 0
