from __future__ import annotations

import pathlib

from ascend_debug import __version__, layout
from ascend_debug.runner import CommandError, find_tool, run_command

# (stage_name, input_stage_key, output_stage_key, [afir-opt pass flags])
# Stages are a logical 6-way split of develop's `--auto-fuse-codegen`
# pipeline (lib/Conversion/AutoFuse/Pipeline.cpp). Order is preserved
# exactly; only dump points are inserted. Flags are afir-opt CLI names
# from include/Conversion/Passes.td. NO C++ dump options.
PASS_STEPS: list[tuple[str, str, str, list[str]]] = [
    (
        "normalize", "source", "010-normalize-out",
        [
            "--linalg-generalize-named-ops",
            "--linalg-fuse-elementwise-ops",
            "--linalg-fold-unit-extent-dims",
            "--canonicalize",
        ],
    ),
    (
        "kernelize", "010-normalize-out", "020-kernelize-out",
        [
            "--auto-fuse-group-analysis",
            "--auto-fuse-group-outline",
        ],
    ),
    (
        "schedule", "020-kernelize-out", "030-schedule-out",
        [
            "--auto-fuse-restore-matmul",
            "--auto-fuse-isolate-kernel-outputs",
            "--afir-symbolize-shapes",
            "--auto-fuse-tile-fuse",
            "--canonicalize",
        ],
    ),
    (
        "realize", "030-schedule-out", "040-realize-out",
        [
            "--annotate-ascendc-kernel-kind",
            "--auto-fuse-fold-shadow-alloc",
            "--auto-fuse-insert-tile-buffers",
            "--ascendc-buffer-placement",
            "--linalg-to-ascendc",
        ],
    ),
    (
        "parallelize", "040-realize-out", "050-parallelize-out",
        [
            "--ascendc-decompose-multi-axis-broadcast",
            "--ascendc-parallelize",
            "--ascendc-flatten-gm-ptr",
            "--canonicalize",
            "--cse",
        ],
    ),
    (
        "finalize", "050-parallelize-out", "060-finalize-out",
        [
            "--auto-fuse-verify-tiling-info-schema",
            "--ascendc-pack-tiling-data",
            "--ascendc-finalize-kernel",
            "--canonicalize-cann-signature",
            "--ascendc-rcore-combine",
        ],
    ),
]

_DEFAULT_TOOL = "afir-opt"


def _stage_rel(output_key: str) -> str:
    return f"stages/{output_key}.mlir"


def collect_run(args) -> int:
    run_dir: pathlib.Path = args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stages").mkdir(exist_ok=True)

    opt = find_tool(_DEFAULT_TOOL)

    # stage key -> absolute artifact path
    paths: dict[str, pathlib.Path] = {"source": run_dir / "stages" / "000-source.mlir"}
    layout.copy_stage(args.input, paths["source"])

    commands: list[dict] = []
    stage_records: list[dict] = []
    for stage, in_key, out_key, flags in PASS_STEPS:
        out_path = run_dir / _stage_rel(out_key)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [opt, str(paths[in_key]), *flags, "-o", str(out_path)]
        run_command(argv, stderr_report_path=run_dir / "stages" / f"{out_key}.report.txt")
        paths[out_key] = out_path
        commands.append({"stage": stage, "tool": _DEFAULT_TOOL, "argv": argv})
        stage_records.append({"name": stage, "path": _stage_rel(out_key)})

    # StageArtifact(order:int, name, path, phase, step). `step` feeds
    # layout.STEP_INFO_BY_STEP for per-stage explanation text; develop's
    # stage names may have no entry (dashboard then shows none — fine for MVP).
    stages = tuple(
        layout.StageArtifact(order=i, name=r["name"], path=r["path"], step=r["name"])
        for i, r in enumerate(stage_records, start=1)
    )
    layout.write_manifest(
        run_dir,
        mode="develop-codegen",
        preset="",
        pipeline="auto-fuse-codegen",
        stages=stages,
        version=__version__,
        commands=commands,
        reports=[],
        graphs=[],
    )
    layout.write_provenance_skeleton(
        run_dir, original_input=args.input, version=__version__
    )
    print(f"ascend-debug.collect.out={run_dir}")
    return 0
