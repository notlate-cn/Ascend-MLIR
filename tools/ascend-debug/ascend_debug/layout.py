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


STEP_INFO_BY_STEP: dict[str, dict[str, str]] = {
    "source": {
        "title": "Source input",
        "purpose": "Original MLIR before Ascend debug collection starts.",
        "inputs": "user-provided MLIR file",
        "outputs": "baseline IR for the first compiler phase",
        "inspect_hint": "Use this to confirm the test case shape, operands, and original op sequence.",
        "common_failures": "Wrong test input or stale run directory.",
    },
    "linalg-cleanup": {
        "title": "Canonicalize tensor/linalg input",
        "purpose": "Run generic cleanup before Ascend-specific normalization.",
        "inputs": "source MLIR",
        "outputs": "canonical linalg/tensor IR",
        "inspect_hint": "Check whether named ops were generalized and obvious CSE/canonicalize noise disappeared.",
        "common_failures": "Unsupported high-level op remains before Kernelize.",
    },
    "ascend-normalize": {
        "title": "Normalize output boundary",
        "purpose": "Provide the stable normalized IR consumed by Kernelize.",
        "inputs": "canonical linalg/tensor IR",
        "outputs": "ascend.normalized function IR",
        "inspect_hint": "Check function attrs and linalg.generic bodies before kernel grouping.",
        "common_failures": "Missing ascend.normalized marker or unsupported dialect survives.",
    },
    "structured-ops": {
        "title": "Identify kernelizable ops",
        "purpose": "Find supported structured/tensor/arith ops that can participate in kernel formation.",
        "inputs": "normalized tensor/linalg IR",
        "outputs": "semantic participation facts for dependency analysis",
        "inspect_hint": "Check which ops are analyzed, transparent, ignored, or rejected.",
        "common_failures": "An expected producer is unsupported or disappeared from dependency analysis.",
    },
    "structural-marking": {
        "title": "Mark structural relationships",
        "purpose": "Mark roots, transparent view chains, branch/merge groups, and co-location constraints.",
        "inputs": "dependency graph",
        "outputs": "structural markers on participating ops",
        "inspect_hint": "Check view-like ops, branch groups, and boundary roots before role classification.",
        "common_failures": "A view chain or shared input creates a wrong boundary or missing edge.",
    },
    "role-classification": {
        "title": "Classify op roles",
        "purpose": "Assign Vector, Cube, Reduction, Memory, and Primary roles for candidate construction.",
        "inputs": "structurally marked dependency graph",
        "outputs": "ascend.op_roles and primary role markers",
        "inspect_hint": "Check whether the intended compute root became Primary and got the expected role.",
        "common_failures": "Role mismatch prevents fusion or sends the op to the wrong template family.",
    },
    "final-patterns": {
        "title": "Build final kernel patterns",
        "purpose": "Merge candidates, choose kernel partitions, and attach kernel ids.",
        "inputs": "role-classified candidate graph",
        "outputs": "ascend.kernel, ascend.primary, template family metadata, kernel graph edges",
        "inspect_hint": "Check kernel ids, primary ops, template families, and cross-kernel edges.",
        "common_failures": "Unexpected split/merge, missing primary op, or incorrect kernel graph edge.",
    },
    "ascend-kernelize": {
        "title": "Kernelize output boundary",
        "purpose": "Expose the stable kernelized IR consumed by Schedule.",
        "inputs": "final kernel patterns",
        "outputs": "phase boundary with kernel ids and pattern metadata",
        "inspect_hint": "This may be identical to the final internal Kernelize step when no extra cleanup runs.",
        "common_failures": "Boundary differs unexpectedly from final kernel patterns.",
    },
    "cleared": {
        "title": "Clear stale schedule metadata",
        "purpose": "Remove previous schedule attrs so the current run cannot reuse stale decision data.",
        "inputs": "kernelized IR with optional old schedule attrs",
        "outputs": "kernelized IR without owned schedule attrs",
        "inspect_hint": "Check that old decision_id, tile_params, tail_plan, and target policy attrs are gone.",
        "common_failures": "Old schedule attrs survive and make later steps look successful for the wrong reason.",
    },
    "decisions": {
        "title": "Choose tile and tail plan",
        "purpose": "Build schedule problems, match templates, search candidates, and select the runtime contract.",
        "inputs": "kernel patterns and target model",
        "outputs": "schedule candidates, selected decision, tile_params, tail_plan",
        "inspect_hint": "Check ScheduleDecisionSet, tile_params, tail_policies, and tuning cache keys.",
        "common_failures": "No template, empty candidates, guard budget pruning, or missing target model data.",
    },
    "final": {
        "title": "Attach schedule contract",
        "purpose": "Attach the chosen schedule decision as the downstream symbolic runtime contract.",
        "inputs": "selected schedule decision",
        "outputs": "decision_id, tile_params, tail_plan, tail_policies, target_tile_policy, structured lowering marker",
        "inspect_hint": "Primary ops and func attrs should expose the same decision_id, tile_params, tail_policies, and target_tile_policy.",
        "common_failures": "Kernel metadata mismatch, missing tile_params, or incomplete tail plan.",
    },
    "ascend-schedule": {
        "title": "Schedule output boundary",
        "purpose": "Expose the stable scheduled IR consumed by Realize.",
        "inputs": "schedule-attached IR",
        "outputs": "phase boundary with symbolic tile and tail contracts",
        "inspect_hint": "This is often identical to Attach schedule contract; check the same-as marker.",
        "common_failures": "Boundary differs unexpectedly from the attached schedule contract.",
    },
    "planned": {
        "title": "Build memory realization plan",
        "purpose": "Plan buffer values, placement, movement, workspace, and realization without mutating tensor semantics.",
        "inputs": "scheduled tensor IR",
        "outputs": "placement plan, movement plan, static memory plan",
        "inspect_hint": "Check planned GM/on-chip places, movement demands, workspace slots, and deferred decisions.",
        "common_failures": "Missing schedule contract, impossible placement, or unsupported movement path.",
    },
    "bufferized": {
        "title": "Bufferize tensor IR",
        "purpose": "Convert tensor values to memrefs while preserving producer/consumer and view-chain relationships.",
        "inputs": "planned tensor IR",
        "outputs": "memref IR with explicit buffers and view chains",
        "inspect_hint": "Check new allocs, subviews, copies, and whether linalg inputs/outs still match the planned values.",
        "common_failures": "Bufferization failure, lost view-chain relation, or unexpected temporary buffer.",
    },
    "memory-space-annotated": {
        "title": "Annotate memory spaces",
        "purpose": "Materialize the memory plan by assigning GM/VECIN/VECCALC/VECOUT spaces and required copies/views.",
        "inputs": "bufferized memref IR",
        "outputs": "memory_space attrs, workspace layout, materialized movement",
        "inspect_hint": "Check alloc/copy nodes, memory_space attrs, workspace slots, and deferred movement counts.",
        "common_failures": "Half-applied bridge, missing output copy, invalid dynamic view rewrite, or workspace reuse error.",
    },
    "ascend-realize": {
        "title": "Realize output boundary",
        "purpose": "Expose the stable memref/memory-space IR consumed by backend lowering.",
        "inputs": "memory-space annotated IR",
        "outputs": "phase boundary with realized buffers and movement",
        "inspect_hint": "This may be identical to the last Realize internal step when no extra cleanup runs.",
        "common_failures": "Boundary differs unexpectedly from annotated memory-space IR.",
    },
    "ascend-compute-lower": {
        "title": "Lower compute to AscendC IR",
        "purpose": "Replace linalg/memref compute and movement with AscendC dialect operations.",
        "inputs": "realized memref IR",
        "outputs": "AscendC compute, copy, local tensor, and queue operations",
        "inspect_hint": "Check whether primary compute ops became AscendC loops/copies and whether tile args are used.",
        "common_failures": "Unsupported op, missing tile_arg, invalid memory space, or unsupported view chain.",
    },
    "ascend-parallelize": {
        "title": "Map parallel loops",
        "purpose": "Map compiler loops and symbolic tile parameters to block-level parallel execution.",
        "inputs": "AscendC compute IR with tile args",
        "outputs": "block index usage and parallelized loop structure",
        "inspect_hint": "Check ascendc.get_block_idx and loop bounds for TB_M/TB_N usage.",
        "common_failures": "Loop not parallelized, wrong block dimension, or tile arg no longer dominates use.",
    },
    "ascend-prepare-for-emit": {
        "title": "Prepare for CANN emission",
        "purpose": "Normalize AscendC IR into the form accepted by the CANN printer and host tiling ABI.",
        "inputs": "parallelized AscendC IR",
        "outputs": "emit-ready AscendC global function and tiling struct usage",
        "inspect_hint": "Check function signature, py_struct tiling data, and global/local tensor setup.",
        "common_failures": "ABI shape mismatch, missing tiling field, or unsupported emit construct.",
    },
    "ascend-canonicalize-cann-signature": {
        "title": "Canonicalize CANN signature",
        "purpose": "Rewrite the function ABI to the final CANN kernel signature.",
        "inputs": "emit-ready AscendC function",
        "outputs": "CANN-compatible kernel arguments and tiling struct",
        "inspect_hint": "Check input/output pointer order, workspace/tiling args, and final kernel attrs.",
        "common_failures": "Wrong argument order, missing output buffer, or tiling struct mismatch.",
    },
}


def step_info_for(stage: StageArtifact) -> dict[str, str] | None:
    if stage.step and stage.step in STEP_INFO_BY_STEP:
        return STEP_INFO_BY_STEP[stage.step]
    if stage.name == "source":
        return STEP_INFO_BY_STEP["source"]
    return None


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
    backend: str = "compile",
    input_path: str | None = None,
    status: str = "success",
    failed_stage: str | None = None,
    failed_phase: str | None = None,
    failure_status: str | None = None,
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
        step_info = step_info_for(stage)
        if step_info:
            record["step_info"] = step_info
        return record

    manifest = {
        "schema_version": 1,
        "tool": "ascend-debug",
        "version": version,
        "mode": mode,
        "preset": preset,
        "pipeline": pipeline,
        "backend": backend,
        "status": status,
        "device_id": None,
        "device_scope": "single_run_single_device",
        "stages": [stage_record(stage) for stage in stages],
        "commands": commands or [],
        "reports": reports or [],
        "graphs": graphs or [],
    }
    if input_path is not None:
        manifest["input"] = input_path
    elif stages:
        manifest["input"] = stages[0].path
    if failed_stage:
        manifest["failed_stage"] = failed_stage
    if failed_phase:
        manifest["failed_phase"] = failed_phase
    if failure_status:
        manifest["failure_status"] = failure_status
    write_json(run_dir / "manifest.json", manifest)


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
