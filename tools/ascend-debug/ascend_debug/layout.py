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


def write_manifest(
    run_dir: pathlib.Path,
    *,
    preset: str,
    pipeline: str,
    stages: tuple[StageArtifact, ...],
    version: str,
    commands: list[dict[str, Any]] | None = None,
    reports: list[dict[str, Any]] | None = None,
    graphs: list[dict[str, Any]] | None = None,
) -> None:
    write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "tool": "ascend-debug",
            "version": version,
            "input": stages[0].path,
            "preset": preset,
            "pipeline": pipeline,
            "backend": "compile",
            "device_id": None,
            "device_scope": "single_run_single_device",
            "stages": [
                {"order": stage.order, "name": stage.name, "path": stage.path}
                for stage in stages
            ],
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
