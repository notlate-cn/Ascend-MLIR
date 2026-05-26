from __future__ import annotations

import argparse

from ascend_debug import __version__, layout
from ascend_debug.runner import CommandError, find_tool, run_command


DEEP_REPORTS = (
    ("normalize", "reports/010-normalize.report.txt"),
    ("kernelize", "reports/020-kernelize.report.txt"),
    ("schedule", "reports/030-schedule.report.txt"),
    ("realize", "reports/040-realize.report.txt"),
)


def _clear_artifacts(run_dir, stages, reports=()) -> None:
    for stage in stages:
        (run_dir / stage.path).unlink(missing_ok=True)
    for _, report in reports:
        (run_dir / report).unlink(missing_ok=True)
    (run_dir / "manifest.json").unlink(missing_ok=True)
    (run_dir / "provenance.json").unlink(missing_ok=True)
    (run_dir / "index.html").unlink(missing_ok=True)


def _record_command(stage: str, args: list[str], stdout_path: str, report_path: str) -> dict:
    return {
        "stage": stage,
        "tool": "ascend-mlir-opt",
        "args": args,
        "stdout": stdout_path,
        "stderr": report_path,
        "status": "success",
    }


def _run_opt_stage(
    *,
    opt: str,
    stage: str,
    input_path,
    input_rel: str,
    output_path,
    output_rel: str,
    report_path,
    report_rel: str,
    pass_arg: str,
) -> dict:
    run_command(
        [opt, str(input_path), pass_arg],
        stdout_path=output_path,
        stderr_report_path=report_path,
    )
    return _record_command(stage, [input_rel, pass_arg], output_rel, report_rel)


def collect_run(args: argparse.Namespace) -> int:
    if args.preset == "quick":
        return collect_quick(args)
    if args.preset == "deep":
        return collect_deep(args)
    raise CommandError(f"unsupported preset: {args.preset}")


def collect_quick(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    run_dir = args.out.resolve()

    stages = layout.QUICK_NORMALIZE_KERNELIZE_STAGES
    layout.prepare_run_dir(run_dir)
    _clear_artifacts(run_dir, stages)

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
        version=__version__,
    )
    layout.write_provenance_skeleton(
        run_dir,
        original_input=args.input,
        version=__version__,
    )
    print(f"ascend-debug.collect.out={run_dir}")
    return 0


def collect_deep(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    run_dir = args.out.resolve()

    stages = layout.DEEP_NORMALIZE_KERNELIZE_STAGES
    layout.prepare_run_dir(run_dir)
    _clear_artifacts(run_dir, stages, DEEP_REPORTS)

    if not input_path.exists():
        raise CommandError(f"input MLIR does not exist: {input_path}")

    stage_paths = {stage.name: run_dir / stage.path for stage in stages}
    report_paths = {name: run_dir / path for name, path in DEEP_REPORTS}
    report_rels = {name: path for name, path in DEEP_REPORTS}

    layout.copy_stage(input_path, stage_paths["source"])
    layout.copy_stage(stage_paths["source"], stage_paths["normalize-in"])

    opt = find_tool("ascend-mlir-opt")
    commands = [
        _run_opt_stage(
            opt=opt,
            stage="normalize",
            input_path=stage_paths["normalize-in"],
            input_rel="stages/010-normalize-in.mlir",
            output_path=stage_paths["normalize-out"],
            output_rel="stages/019-normalize-out.mlir",
            report_path=report_paths["normalize"],
            report_rel=report_rels["normalize"],
            pass_arg="--ascend-normalize",
        )
    ]

    layout.copy_stage(stage_paths["normalize-out"], stage_paths["kernelize-in"])
    commands.append(
        _run_opt_stage(
            opt=opt,
            stage="kernelize",
            input_path=stage_paths["kernelize-in"],
            input_rel="stages/020-kernelize-in.mlir",
            output_path=stage_paths["kernelize-out"],
            output_rel="stages/029-kernelize-out.mlir",
            report_path=report_paths["kernelize"],
            report_rel=report_rels["kernelize"],
            pass_arg="--ascend-kernelize",
        )
    )

    layout.copy_stage(stage_paths["kernelize-out"], stage_paths["schedule-in"])
    commands.append(
        _run_opt_stage(
            opt=opt,
            stage="schedule",
            input_path=stage_paths["schedule-in"],
            input_rel="stages/030-schedule-in.mlir",
            output_path=stage_paths["schedule-out"],
            output_rel="stages/039-schedule-out.mlir",
            report_path=report_paths["schedule"],
            report_rel=report_rels["schedule"],
            pass_arg="--ascend-schedule=target-tile-policy=legacy-default dump-report=true debug-stage=schedule",
        )
    )

    layout.copy_stage(stage_paths["schedule-out"], stage_paths["realize-in"])
    commands.append(
        _run_opt_stage(
            opt=opt,
            stage="realize",
            input_path=stage_paths["realize-in"],
            input_rel="stages/040-realize-in.mlir",
            output_path=stage_paths["realize-out"],
            output_rel="stages/049-realize-out.mlir",
            report_path=report_paths["realize"],
            report_rel=report_rels["realize"],
            pass_arg="--ascend-realize=dump-report=true debug-stage=realize",
        )
    )

    reports = [{"stage": name, "path": path} for name, path in DEEP_REPORTS]
    layout.write_manifest(
        run_dir,
        preset=args.preset,
        pipeline=args.pipeline,
        stages=stages,
        version=__version__,
        commands=commands,
        reports=reports,
    )
    layout.write_provenance_skeleton(
        run_dir,
        original_input=args.input,
        version=__version__,
    )
    print(f"ascend-debug.collect.out={run_dir}")
    return 0
