from __future__ import annotations

import argparse
import os
import pathlib
import shlex

from ascend_debug import __version__, kernel_dag, layout
from ascend_debug.runner import CommandError, find_tool, run_command


DEEP_REPORTS = (
    ("normalize", "reports/010-normalize.report.txt"),
    ("kernelize", "reports/020-kernelize.report.txt"),
    ("schedule", "reports/030-schedule.report.txt"),
    ("realize", "reports/040-realize.report.txt"),
)

KERNEL_DAG_REPORT = ("kernel-dag", "reports/050-kernel-dag.report.txt")


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


def _env_path(*names: str) -> pathlib.Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return pathlib.Path(value)
    return None


def _memory_detail_cann_root(args: argparse.Namespace) -> pathlib.Path:
    cann_root = args.cann_root or _env_path("ASCEND_HOME_PATH", "ASCEND_HOME", "CANN_ROOT")
    if not cann_root:
        raise CommandError(
            "--memory-detail requires --cann-root or ASCEND_HOME_PATH/ASCEND_HOME/CANN_ROOT"
        )
    return cann_root.resolve()


def _realize_pass_arg(args: argparse.Namespace) -> str:
    options = ["dump-report=true", "debug-stage=realize"]
    if args.memory_detail:
        cann_root = _memory_detail_cann_root(args)
        soc = args.soc or os.environ.get("ASCEND_SOC_VERSION") or "Ascend910B2"
        options.extend(
            [
                "placement-mode=target-aware",
                "materialization-mode=plan-only",
                f"cann-root={cann_root}",
                f"soc={soc}",
            ]
        )
    if args.realize_options:
        try:
            options.extend(shlex.split(args.realize_options))
        except ValueError as error:
            raise CommandError(f"invalid --realize-options: {error}") from error
    return "--ascend-realize=" + " ".join(options)


def _graph_requested(args: argparse.Namespace) -> bool:
    return bool(
        args.artifact_manifest
        or args.runtime_manifest
        or args.run_manifest
        or args.kernelized_ir
    )


def _resolve_existing_path(path: pathlib.Path, *, label: str) -> pathlib.Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise CommandError(f"{label} does not exist: {resolved}")
    return resolved


def _copy_graph_artifact(
    *,
    src: pathlib.Path,
    run_dir: pathlib.Path,
    rel_path: str,
    kind: str,
) -> dict:
    dst = run_dir / rel_path
    layout.copy_stage(src, dst)
    return {"kind": kind, "path": rel_path}


def _collect_graph_artifacts(
    *,
    args: argparse.Namespace,
    run_dir: pathlib.Path,
    default_kernelized_ir: pathlib.Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not _graph_requested(args):
        return [], [], []
    if not args.artifact_manifest and not args.runtime_manifest:
        raise CommandError("--artifact-manifest is required when collecting graph artifacts")
    if (
        args.artifact_manifest
        and args.runtime_manifest
        and args.artifact_manifest.resolve() != args.runtime_manifest.resolve()
    ):
        raise CommandError(
            "cannot pass both --artifact-manifest and --runtime-manifest with different paths"
        )

    graphs: list[dict] = []
    commands: list[dict] = []
    reports: list[dict] = []

    artifact_manifest_arg = args.artifact_manifest or args.runtime_manifest
    manifest_label = "artifact manifest" if args.artifact_manifest else "runtime manifest"
    artifact_manifest = _resolve_existing_path(artifact_manifest_arg, label=manifest_label)
    manifest_rel = (
        "graphs/artifact_manifest.json"
        if args.artifact_manifest
        else "graphs/runtime_manifest.json"
    )
    manifest_kind = "artifact-manifest" if args.artifact_manifest else "runtime-manifest"
    graphs.append(
        _copy_graph_artifact(
            src=artifact_manifest,
            run_dir=run_dir,
            rel_path=manifest_rel,
            kind=manifest_kind,
        )
    )
    artifact_manifest_dst = run_dir / manifest_rel

    run_manifest_dst = None
    if args.run_manifest:
        run_manifest = _resolve_existing_path(args.run_manifest, label="run manifest")
        graphs.append(
            _copy_graph_artifact(
                src=run_manifest,
                run_dir=run_dir,
                rel_path="graphs/run_manifest.json",
                kind="run-manifest",
            )
        )
        run_manifest_dst = run_dir / "graphs/run_manifest.json"

    kernelized_ir = (
        _resolve_existing_path(args.kernelized_ir, label="kernelized IR")
        if args.kernelized_ir
        else default_kernelized_ir
    )
    graphs.append(
        _copy_graph_artifact(
            src=kernelized_ir,
            run_dir=run_dir,
            rel_path="graphs/kernelized.mlir",
            kind="kernelized-ir",
        )
    )
    kernelized_ir_dst = run_dir / "graphs/kernelized.mlir"

    svg_rel = "graphs/kernel_dag.svg"
    summary_rel = "graphs/kernel_dag.summary.json"
    report_stage, report_rel = KERNEL_DAG_REPORT
    tool_args = [
        "--artifact-manifest" if args.artifact_manifest else "--runtime-manifest",
        manifest_rel,
        "--kernelized-ir",
        "graphs/kernelized.mlir",
        "--svg-out",
        svg_rel,
        "--summary-out",
        summary_rel,
        "--kernel-view-base",
        "../views/kernels",
    ]
    if run_manifest_dst:
        tool_args[2:2] = ["--run-manifest", "graphs/run_manifest.json"]

    summary = kernel_dag.analyze_paths(
        artifact_manifest_path=artifact_manifest_dst,
        run_manifest_path=run_manifest_dst,
        kernelized_ir=kernelized_ir_dst,
    )
    kernel_dag.render_svg(summary, run_dir / svg_rel, "../views/kernels")
    kernel_dag.write_summary(summary, run_dir / summary_rel)
    kernel_dag.write_report(summary, run_dir / report_rel)
    commands.append(
        {
            "stage": report_stage,
            "tool": "ascend-debug",
            "args": tool_args,
            "stdout": report_rel,
            "status": "success",
        }
    )
    reports.append({"stage": report_stage, "path": report_rel})
    graphs.extend(
        [
            {"kind": "kernel-dag-svg", "path": svg_rel},
            {"kind": "kernel-dag-summary", "path": summary_rel},
        ]
    )
    return commands, reports, graphs


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
    if args.preset != "deep" and (
        args.memory_detail or args.realize_options or args.cann_root or args.soc
    ):
        raise CommandError("Realize memory options require --preset deep")
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

    graph_commands, graph_reports, graphs = _collect_graph_artifacts(
        args=args,
        run_dir=run_dir,
        default_kernelized_ir=kernelize_out,
    )
    layout.write_manifest(
        run_dir,
        preset=args.preset,
        pipeline=args.pipeline,
        stages=stages,
        version=__version__,
        commands=graph_commands,
        reports=graph_reports,
        graphs=graphs,
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
            pass_arg=_realize_pass_arg(args),
        )
    )

    graph_commands, graph_reports, graphs = _collect_graph_artifacts(
        args=args,
        run_dir=run_dir,
        default_kernelized_ir=stage_paths["kernelize-out"],
    )
    commands.extend(graph_commands)
    reports = [{"stage": name, "path": path} for name, path in DEEP_REPORTS]
    reports.extend(graph_reports)
    layout.write_manifest(
        run_dir,
        preset=args.preset,
        pipeline=args.pipeline,
        stages=stages,
        version=__version__,
        commands=commands,
        reports=reports,
        graphs=graphs,
    )
    layout.write_provenance_skeleton(
        run_dir,
        original_input=args.input,
        version=__version__,
    )
    print(f"ascend-debug.collect.out={run_dir}")
    return 0
