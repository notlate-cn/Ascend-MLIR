from __future__ import annotations

import json
import pathlib
import shutil
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageArtifact:
    order: int
    name: str
    path: str
    phase: str | None = None
    step: str | None = None


QUICK_NORMALIZE_KERNELIZE_STAGES: tuple[StageArtifact, ...] = (
    StageArtifact(0, "source", "stages/000-source.mlir"),
    StageArtifact(10, "normalize-in", "stages/010-normalize-in.mlir"),
    StageArtifact(19, "normalize-out", "stages/019-normalize-out.mlir"),
    StageArtifact(20, "kernelize-in", "stages/020-kernelize-in.mlir"),
    StageArtifact(29, "kernelize-out", "stages/029-kernelize-out.mlir"),
)

DEEP_NORMALIZE_KERNELIZE_STAGES: tuple[StageArtifact, ...] = (
    StageArtifact(0, "source", "stages/000-source.mlir"),
    StageArtifact(10, "normalize-in", "stages/010-normalize-in.mlir"),
    StageArtifact(19, "normalize-out", "stages/019-normalize-out.mlir"),
    StageArtifact(20, "kernelize-in", "stages/020-kernelize-in.mlir"),
    StageArtifact(29, "kernelize-out", "stages/029-kernelize-out.mlir"),
    StageArtifact(30, "schedule-in", "stages/030-schedule-in.mlir"),
    StageArtifact(39, "schedule-out", "stages/039-schedule-out.mlir"),
    StageArtifact(40, "realize-in", "stages/040-realize-in.mlir"),
    StageArtifact(49, "realize-out", "stages/049-realize-out.mlir"),
)

FULL_CODEGEN_STAGES: tuple[StageArtifact, ...] = (
    StageArtifact(0, "source", "stages/000-source.mlir"),
    StageArtifact(10, "010-normalize-prep-out", "stages/010-normalize-prep-out.mlir", "Normalize", "linalg-cleanup"),
    StageArtifact(20, "020-normalize-out", "stages/020-normalize-out.mlir", "Normalize", "ascend-normalize"),
    StageArtifact(21, "021-kernelize-structured-ops", "stages/021-kernelize-structured-ops.mlir", "Kernelize", "structured-ops"),
    StageArtifact(22, "022-kernelize-structural-marking", "stages/022-kernelize-structural-marking.mlir", "Kernelize", "structural-marking"),
    StageArtifact(23, "023-kernelize-role-classification", "stages/023-kernelize-role-classification.mlir", "Kernelize", "role-classification"),
    StageArtifact(24, "024-kernelize-final-patterns", "stages/024-kernelize-final-patterns.mlir", "Kernelize", "final-patterns"),
    StageArtifact(30, "030-kernelize-out", "stages/030-kernelize-out.mlir", "Kernelize", "ascend-kernelize"),
    StageArtifact(31, "031-schedule-cleared", "stages/031-schedule-cleared.mlir", "Schedule", "cleared"),
    StageArtifact(32, "032-schedule-decisions", "stages/032-schedule-decisions.mlir", "Schedule", "decisions"),
    StageArtifact(33, "033-schedule-final", "stages/033-schedule-final.mlir", "Schedule", "final"),
    StageArtifact(40, "040-schedule-out", "stages/040-schedule-out.mlir", "Schedule", "ascend-schedule"),
    StageArtifact(41, "041-realize-planned", "stages/041-realize-planned.mlir", "Realize", "planned"),
    StageArtifact(42, "042-realize-bufferized", "stages/042-realize-bufferized.mlir", "Realize", "bufferized"),
    StageArtifact(43, "043-realize-memory-space-annotated", "stages/043-realize-memory-space-annotated.mlir", "Realize", "memory-space-annotated"),
    StageArtifact(50, "050-realize-out", "stages/050-realize-out.mlir", "Realize", "ascend-realize"),
    StageArtifact(60, "060-compute-lower-out", "stages/060-compute-lower-out.mlir", "Translate", "ascend-compute-lower"),
    StageArtifact(70, "070-parallelize-out", "stages/070-parallelize-out.mlir", "Translate", "ascend-parallelize"),
    StageArtifact(80, "080-prepare-for-emit-out", "stages/080-prepare-for-emit-out.mlir", "Translate", "ascend-prepare-for-emit"),
    StageArtifact(90, "090-cann-signature-out", "stages/090-cann-signature-out.mlir", "Translate", "ascend-canonicalize-cann-signature"),
)


def prepare_run_dir(run_dir: pathlib.Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ["stages", "reports", "graphs", "tensors/final", "tensors/checkpoints", "profiles", "summaries"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_stage(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_script_payload(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def write_manifest(
    run_dir: pathlib.Path,
    *,
    mode: str,
    preset: str,
    pipeline: str,
    stages: tuple[StageArtifact, ...],
    version: str,
    commands: list[dict[str, Any]] | None = None,
    reports: list[dict[str, Any]] | None = None,
    graphs: list[dict[str, Any]] | None = None,
) -> None:
    def stage_record(stage: StageArtifact) -> dict[str, Any]:
        record: dict[str, Any] = {"order": stage.order, "name": stage.name, "path": stage.path}
        if stage.phase:
            record["phase"] = stage.phase
        if stage.step:
            record["step"] = stage.step
        return record

    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "tool": "ascend-debug",
            "version": version,
            "input": stages[0].path,
            "mode": mode,
            "preset": preset,
            "pipeline": pipeline,
            "backend": "compile",
            "device_id": None,
            "device_scope": "single_run_single_device",
            "stages": [stage_record(stage) for stage in stages],
            "commands": commands or [],
            "reports": reports or [],
            "graphs": graphs or [],
        },
    )


def write_provenance_skeleton(
    run_dir: pathlib.Path,
    *,
    original_input: pathlib.Path,
    version: str,
) -> None:
    write_json(
        run_dir / "provenance.json",
        {
            "schema_version": 1,
            "tool": "ascend-debug",
            "version": version,
            "original_input": str(original_input),
            "boundaries": [],
            "kernels": [],
            "runtime_tasks": [],
        },
    )
