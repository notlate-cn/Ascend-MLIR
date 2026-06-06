from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import shlex

from ascend_debug import __version__, contracts, failure, kernel_dag, layout
from ascend_debug.runner import CommandError, find_tool, run_command


DEEP_REPORTS = (
    ("normalize", "reports/010-normalize.report.txt"),
    ("kernelize", "reports/020-kernelize.report.txt"),
    ("schedule", "reports/030-schedule.report.txt"),
    ("realize", "reports/040-realize.report.txt"),
)

FULL_CODEGEN_REPORTS = (
    ("normalize-prep", "reports/010-normalize-prep.report.txt"),
    ("normalize", "reports/020-normalize.report.txt"),
    ("kernelize", "reports/030-kernelize.report.txt"),
    ("schedule", "reports/040-schedule.report.txt"),
    ("realize", "reports/050-realize.report.txt"),
    ("compute-lower", "reports/060-compute-lower.report.txt"),
    ("parallelize", "reports/070-parallelize.report.txt"),
    ("prepare-for-emit", "reports/080-prepare-for-emit.report.txt"),
    ("cann-signature", "reports/090-cann-signature.report.txt"),
)

KERNEL_DAG_REPORT = ("kernel-dag", "reports/050-kernel-dag.report.txt")


def _clear_artifacts(run_dir, stages, reports=()) -> None:
    for stage in stages:
        (run_dir / stage.path).unlink(missing_ok=True)
    for _, report in reports:
        (run_dir / report).unlink(missing_ok=True)
    (run_dir / "manifest.json").unlink(missing_ok=True)
    (run_dir / "run_status.json").unlink(missing_ok=True)
    (run_dir / "provenance.json").unlink(missing_ok=True)
    (run_dir / "index.html").unlink(missing_ok=True)
    shutil.rmtree(run_dir / "debug_contract", ignore_errors=True)


def _record_command(stage: str, args: list[str], stdout_path: str, report_path: str) -> dict:
    return failure.command_record(
        stage=stage,
        tool="ascend-mlir-opt",
        args=args,
        stdout=stdout_path,
        stderr=report_path,
    )


def _write_collect_failure_manifest(
    *,
    args: argparse.Namespace,
    run_dir: pathlib.Path,
    stages: tuple[layout.StageArtifact, ...],
    commands: list[dict],
    reports: list[dict] | None,
    graphs: list[dict] | None,
    error: CommandError,
) -> None:
    failed_command = failure.command_from_error(error)
    manifest_commands = [*commands]
    if failed_command:
        manifest_commands.append(failed_command)
    failed_stage = str((failed_command or {}).get("stage") or "collect")
    status_rel = failure.write_run_status(
        run_dir,
        stage=failed_stage,
        phase="compile",
        command=failed_command,
        error=error,
    )
    report_records = reports if reports is not None else failure.report_records_from_commands(manifest_commands)
    layout.write_manifest(
        run_dir,
        mode=args.mode,
        preset=args.preset,
        pipeline=args.pipeline,
        stages=failure.existing_stages(run_dir, stages),
        version=__version__,
        status="failed",
        failed_stage=failed_stage,
        failed_phase="compile",
        failure_status=status_rel,
        commands=manifest_commands,
        reports=report_records,
        graphs=graphs or [],
    )
    layout.write_provenance_skeleton(
        run_dir,
        original_input=args.input,
        version=__version__,
    )


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


def _full_codegen_cann_root(args: argparse.Namespace) -> pathlib.Path:
    cann_root = args.cann_root or _env_path(
        "ASCEND_HOME_PATH", "ASCEND_HOME", "CANN_ROOT", "ASCEND_TOOLKIT_HOME"
    )
    if not cann_root:
        raise CommandError(
            "--mode deep requires --cann-root or "
            "ASCEND_HOME_PATH/ASCEND_HOME/CANN_ROOT/ASCEND_TOOLKIT_HOME"
        )
    return cann_root.resolve()


def _full_codegen_soc(args: argparse.Namespace) -> str:
    return (
        args.soc
        or os.environ.get("ASCEND_SOC_VERSION")
        or os.environ.get("SOC_VERSION")
        or "Ascend910B1"
    )


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
        or args.run_manifest
        or args.kernelized_ir
    )


def _contract_graph_requested(
    contract_bundle: contracts.ContractBundle | None,
) -> bool:
    return bool(
        contract_bundle
        and contract_bundle.has(kernel_dag.KERNEL_DAG_CONTRACT_SCHEMA)
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


def _collect_debug_contracts(
    args: argparse.Namespace,
    run_dir: pathlib.Path,
) -> tuple[contracts.ContractBundle | None, list[dict]]:
    contract_dir = getattr(args, "debug_contract_dir", None)
    if not contract_dir:
        return None, []
    bundle = contracts.load_contract_bundle(contract_dir)
    out_dir = run_dir / "debug_contract"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_records: list[dict] = []
    for path in sorted(bundle.root.glob("*.json")):
        contract = contracts.load_contract_file(path)
        rel_path = f"debug_contract/{path.name}"
        layout.copy_stage(path, run_dir / rel_path)
        graph_records.append(
            {
                "kind": "debug-contract",
                "schema": contract["schema"],
                "path": rel_path,
            }
        )
    return bundle, graph_records


def _collect_graph_artifacts(
    *,
    args: argparse.Namespace,
    run_dir: pathlib.Path,
    default_kernelized_ir: pathlib.Path,
    contract_bundle: contracts.ContractBundle | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not _graph_requested(args) and not _contract_graph_requested(contract_bundle):
        return [], [], []
    if not args.artifact_manifest and not _contract_graph_requested(contract_bundle):
        raise CommandError("--artifact-manifest is required when collecting graph artifacts")

    graphs: list[dict] = []
    commands: list[dict] = []
    reports: list[dict] = []
    svg_rel = "graphs/kernel_dag.svg"
    summary_rel = "graphs/kernel_dag.summary.json"
    report_stage, report_rel = KERNEL_DAG_REPORT

    if contract_bundle and contract_bundle.has(kernel_dag.KERNEL_DAG_CONTRACT_SCHEMA):
        if args.artifact_manifest:
            artifact_manifest = _resolve_existing_path(
                args.artifact_manifest, label="artifact manifest"
            )
            graphs.append(
                _copy_graph_artifact(
                    src=artifact_manifest,
                    run_dir=run_dir,
                    rel_path="graphs/artifact_manifest.json",
                    kind="artifact-manifest",
                )
            )
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
        if args.kernelized_ir:
            kernelized_ir = _resolve_existing_path(
                args.kernelized_ir, label="kernelized IR"
            )
            graphs.append(
                _copy_graph_artifact(
                    src=kernelized_ir,
                    run_dir=run_dir,
                    rel_path="graphs/kernelized.mlir",
                    kind="kernelized-ir",
                )
            )
        contract = contract_bundle.get(kernel_dag.KERNEL_DAG_CONTRACT_SCHEMA)
        summary = kernel_dag.summary_from_contract(contract)
        kernel_dag.render_svg(summary, run_dir / svg_rel, "../views/kernels")
        kernel_dag.write_summary(summary, run_dir / summary_rel)
        kernel_dag.write_report(summary, run_dir / report_rel)
        commands.append(
            {
                "stage": report_stage,
                "tool": "ascend-debug",
                "args": [
                    "--debug-contract",
                    kernel_dag.KERNEL_DAG_CONTRACT_SCHEMA,
                    "--svg-out",
                    svg_rel,
                    "--summary-out",
                    summary_rel,
                ],
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

    artifact_manifest = _resolve_existing_path(args.artifact_manifest, label="artifact manifest")
    manifest_rel = "graphs/artifact_manifest.json"
    graphs.append(
        _copy_graph_artifact(
            src=artifact_manifest,
            run_dir=run_dir,
            rel_path=manifest_rel,
            kind="artifact-manifest",
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

    tool_args = [
        "--artifact-manifest",
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
    return _run_opt_stage_args(
        opt=opt,
        stage=stage,
        input_path=input_path,
        input_rel=input_rel,
        output_path=output_path,
        output_rel=output_rel,
        report_path=report_path,
        report_rel=report_rel,
        pass_args=[pass_arg],
    )


def _run_opt_stage_args(
    *,
    opt: str,
    stage: str,
    input_path,
    input_rel: str,
    output_path,
    output_rel: str,
    report_path,
    report_rel: str,
    pass_args: list[str],
) -> dict:
    try:
        run_command(
            [opt, str(input_path), *pass_args],
            stdout_path=output_path,
            stderr_report_path=report_path,
        )
    except CommandError as error:
        error.debug_command = failure.failed_command_record(
            stage=stage,
            tool="ascend-mlir-opt",
            args=[input_rel, *pass_args],
            stdout=output_rel,
            stderr=report_rel,
            error=error,
        )
        raise
    return _record_command(stage, [input_rel, *pass_args], output_rel, report_rel)


def _normalize_collect_selection(args: argparse.Namespace) -> None:
    requested_mode = args.mode
    legacy_preset = args.preset
    legacy_pipeline = args.pipeline
    if requested_mode and (legacy_preset or legacy_pipeline):
        raise CommandError("--mode cannot be combined with legacy --preset/--pipeline")

    if requested_mode == "quick":
        args.mode = "quick"
        args.preset = "deep"
        args.pipeline = "normalize-kernelize"
        return
    if requested_mode == "deep":
        args.mode = "deep"
        args.preset = "deep"
        args.pipeline = "full-codegen"
        return

    if legacy_preset or legacy_pipeline:
        args.preset = legacy_preset or (
            "deep" if legacy_pipeline == "full-codegen" else "quick"
        )
        args.pipeline = legacy_pipeline or "normalize-kernelize"
        args.mode = "deep" if args.pipeline == "full-codegen" else "quick"
        return

    args.mode = "quick"
    args.preset = "deep"
    args.pipeline = "normalize-kernelize"


def collect_run(args: argparse.Namespace) -> int:
    _normalize_collect_selection(args)
    if args.pipeline == "full-codegen" and args.preset != "deep":
        raise CommandError("--pipeline full-codegen requires --preset deep")
    if args.pipeline == "full-codegen" and args.memory_detail:
        raise CommandError("--memory-detail is only supported by --mode quick")
    if args.preset != "deep" and (
        args.memory_detail or args.realize_options or args.cann_root or args.soc
    ):
        raise CommandError("Realize memory options require --mode quick")
    if args.pipeline == "full-codegen":
        return collect_full_codegen(args)
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
    contract_bundle, contract_graphs = _collect_debug_contracts(args, run_dir)

    source = run_dir / stages[0].path
    normalize_in = run_dir / stages[1].path
    normalize_out = run_dir / stages[2].path
    kernelize_in = run_dir / stages[3].path
    kernelize_out = run_dir / stages[4].path

    layout.copy_stage(input_path, source)
    layout.copy_stage(source, normalize_in)

    opt = find_tool("ascend-mlir-opt")
    commands: list[dict] = []
    try:
        commands.append(
            _run_opt_stage_args(
                opt=opt,
                stage="normalize",
                input_path=normalize_in,
                input_rel="stages/010-normalize-in.mlir",
                output_path=normalize_out,
                output_rel="stages/019-normalize-out.mlir",
                report_path=run_dir / "reports/010-normalize.report.txt",
                report_rel="reports/010-normalize.report.txt",
                pass_args=["--ascend-normalize"],
            )
        )
        layout.copy_stage(normalize_out, kernelize_in)
        commands.append(
            _run_opt_stage_args(
                opt=opt,
                stage="kernelize",
                input_path=kernelize_in,
                input_rel="stages/020-kernelize-in.mlir",
                output_path=kernelize_out,
                output_rel="stages/029-kernelize-out.mlir",
                report_path=run_dir / "reports/020-kernelize.report.txt",
                report_rel="reports/020-kernelize.report.txt",
                pass_args=["--ascend-kernelize"],
            )
        )

        graph_commands, graph_reports, graphs = _collect_graph_artifacts(
            args=args,
            run_dir=run_dir,
            default_kernelized_ir=kernelize_out,
            contract_bundle=contract_bundle,
        )
    except CommandError as error:
        _write_collect_failure_manifest(
            args=args,
            run_dir=run_dir,
            stages=stages,
            commands=commands,
            reports=None,
            graphs=None,
            error=error,
        )
        raise
    layout.write_manifest(
        run_dir,
        mode=args.mode,
        preset=args.preset,
        pipeline=args.pipeline,
        stages=stages,
        version=__version__,
        commands=[*commands, *graph_commands],
        reports=graph_reports,
        graphs=[*contract_graphs, *graphs],
    )
    layout.write_provenance_skeleton(
        run_dir,
        original_input=args.input,
        version=__version__,
    )
    print(f"ascend-debug.collect.out={run_dir}")
    return 0


def collect_full_codegen(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    run_dir = args.out.resolve()

    stages = layout.FULL_CODEGEN_STAGES
    layout.prepare_run_dir(run_dir)
    _clear_artifacts(run_dir, stages, FULL_CODEGEN_REPORTS)

    if not input_path.exists():
        raise CommandError(f"input MLIR does not exist: {input_path}")
    contract_bundle, contract_graphs = _collect_debug_contracts(args, run_dir)

    stage_paths = {stage.name: run_dir / stage.path for stage in stages}
    stage_rels = {stage.name: stage.path for stage in stages}
    report_paths = {name: run_dir / path for name, path in FULL_CODEGEN_REPORTS}
    report_rels = {name: path for name, path in FULL_CODEGEN_REPORTS}

    layout.copy_stage(input_path, stage_paths["source"])

    opt = find_tool("ascend-mlir-opt")
    cann_root = _full_codegen_cann_root(args)
    soc = _full_codegen_soc(args)
    checkpoint_dump_dir = run_dir / "stages"

    realize_options = [
        "materialization-mode=memory-space-annotate",
        "dump-report=true",
        "debug-stage=realize",
        f"debug-dump-dir={checkpoint_dump_dir}",
    ]
    if args.realize_options:
        try:
            realize_options.extend(shlex.split(args.realize_options))
        except ValueError as error:
            raise CommandError(f"invalid --realize-options: {error}") from error

    pass_steps = [
        (
            "normalize-prep",
            "source",
            "010-normalize-prep-out",
            [
                "--linalg-generalize-named-ops",
                "--linalg-fuse-elementwise-ops",
                "--canonicalize",
                "--cse",
            ],
        ),
        (
            "normalize",
            "010-normalize-prep-out",
            "020-normalize-out",
            ["--ascend-normalize=dump-report=true debug-stage=normalize"],
        ),
        (
            "kernelize",
            "020-normalize-out",
            "030-kernelize-out",
            [
                "--ascend-kernelize="
                "dump-report=true "
                "debug-stage=kernelize "
                f"debug-dump-dir={checkpoint_dump_dir}"
            ],
        ),
        (
            "schedule",
            "030-kernelize-out",
            "040-schedule-out",
            [
                "--ascend-schedule="
                f"target-tile-policy=target-aware cann-root={cann_root} soc={soc} "
                "dump-report=true "
                "debug-stage=schedule "
                f"debug-dump-dir={checkpoint_dump_dir}"
            ],
        ),
        (
            "realize",
            "040-schedule-out",
            "050-realize-out",
            ["--ascend-realize=" + " ".join(realize_options)],
        ),
        ("compute-lower", "050-realize-out", "060-compute-lower-out", ["--ascend-compute-lower"]),
        ("parallelize", "060-compute-lower-out", "070-parallelize-out", ["--ascend-parallelize"]),
        (
            "prepare-for-emit",
            "070-parallelize-out",
            "080-prepare-for-emit-out",
            ["--ascend-prepare-for-emit"],
        ),
        (
            "cann-signature",
            "080-prepare-for-emit-out",
            "090-cann-signature-out",
            ["--ascend-canonicalize-cann-signature", "--canonicalize", "--cse"],
        ),
    ]

    commands = []
    try:
        for stage, input_stage, output_stage, pass_args in pass_steps:
            commands.append(
                _run_opt_stage_args(
                    opt=opt,
                    stage=stage,
                    input_path=stage_paths[input_stage],
                    input_rel=stage_rels[input_stage],
                    output_path=stage_paths[output_stage],
                    output_rel=stage_rels[output_stage],
                    report_path=report_paths[stage],
                    report_rel=report_rels[stage],
                    pass_args=pass_args,
                )
            )

        graph_commands, graph_reports, graphs = _collect_graph_artifacts(
            args=args,
            run_dir=run_dir,
            default_kernelized_ir=stage_paths["030-kernelize-out"],
            contract_bundle=contract_bundle,
        )
    except CommandError as error:
        _write_collect_failure_manifest(
            args=args,
            run_dir=run_dir,
            stages=stages,
            commands=commands,
            reports=None,
            graphs=None,
            error=error,
        )
        raise
    commands.extend(graph_commands)
    reports = [{"stage": name, "path": path} for name, path in FULL_CODEGEN_REPORTS]
    reports.extend(graph_reports)
    manifest_stages = tuple(
        stage for stage in stages if (run_dir / stage.path).exists()
    )
    layout.write_manifest(
        run_dir,
        mode=args.mode,
        preset=args.preset,
        pipeline=args.pipeline,
        stages=manifest_stages,
        version=__version__,
        commands=commands,
        reports=reports,
        graphs=[*contract_graphs, *graphs],
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
    contract_bundle, contract_graphs = _collect_debug_contracts(args, run_dir)

    stage_paths = {stage.name: run_dir / stage.path for stage in stages}
    report_paths = {name: run_dir / path for name, path in DEEP_REPORTS}
    report_rels = {name: path for name, path in DEEP_REPORTS}

    layout.copy_stage(input_path, stage_paths["source"])
    layout.copy_stage(stage_paths["source"], stage_paths["normalize-in"])

    opt = find_tool("ascend-mlir-opt")
    commands = []
    try:
        commands.append(
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
        )

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
            contract_bundle=contract_bundle,
        )
    except CommandError as error:
        _write_collect_failure_manifest(
            args=args,
            run_dir=run_dir,
            stages=stages,
            commands=commands,
            reports=None,
            graphs=None,
            error=error,
        )
        raise
    commands.extend(graph_commands)
    reports = [{"stage": name, "path": path} for name, path in DEEP_REPORTS]
    reports.extend(graph_reports)
    layout.write_manifest(
        run_dir,
        mode=args.mode,
        preset=args.preset,
        pipeline=args.pipeline,
        stages=stages,
        version=__version__,
        commands=commands,
        reports=reports,
        graphs=[*contract_graphs, *graphs],
    )
    layout.write_provenance_skeleton(
        run_dir,
        original_input=args.input,
        version=__version__,
    )
    print(f"ascend-debug.collect.out={run_dir}")
    return 0
