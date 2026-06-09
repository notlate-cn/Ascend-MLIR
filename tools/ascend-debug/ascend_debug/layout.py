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
        "title": "原始输入",
        "purpose": "展示 Ascend debug 收集开始前的初始 IR。",
        "inputs": "用户提供的 MLIR 文件",
        "outputs": "后续编译阶段的基准 IR",
        "inspect_hint": "确认用例 shape、操作数和原始 op 顺序是否符合预期。",
        "common_failures": "输入用例不对，或复用了过期的 run 目录。",
    },
    "linalg-cleanup": {
        "title": "规范化 tensor/linalg 输入",
        "purpose": "在 Ascend 专用 normalize 前，先执行通用 linalg 清理。",
        "inputs": "source MLIR",
        "outputs": "规范化后的 linalg/tensor IR",
        "inspect_hint": "检查 named op 是否已泛化，明显的 CSE/canonicalize 噪声是否消失。",
        "common_failures": "Kernelize 前仍残留不支持的高层 op。",
    },
    "ascend-normalize": {
        "title": "Normalize 输出边界",
        "purpose": "给 Kernelize 提供稳定的 normalized IR。",
        "inputs": "规范化后的 linalg/tensor IR",
        "outputs": "带 ascend.normalized 和 symbol constraints 的函数 IR",
        "inspect_hint": "检查 kernel 分组前的函数属性、symbol constraints 和 linalg.generic body。",
        "common_failures": "缺少 ascend.normalized、symbol constraints 非法，或仍有不支持的 dialect。",
    },
    "structured-ops": {
        "title": "识别可 Kernelize 的算子",
        "purpose": "找出能参与 kernel 形成的 structured/tensor/arith 算子。",
        "inputs": "normalized tensor/linalg IR",
        "outputs": "供依赖分析使用的算子参与信息",
        "inspect_hint": "检查哪些 op 被分析、透传、忽略或拒绝。",
        "common_failures": "预期的 producer 不受支持，或没有进入依赖分析。",
    },
    "structural-marking": {
        "title": "标记结构关系",
        "purpose": "标记 root、透明 view 链、分支/合并组和 co-location 约束。",
        "inputs": "依赖图",
        "outputs": "参与 op 上的结构标记",
        "inspect_hint": "在角色分类前检查 view-like op、分支组和边界 root。",
        "common_failures": "view 链或共享输入导致边界错误，或依赖边缺失。",
    },
    "role-classification": {
        "title": "分类算子角色",
        "purpose": "为候选 kernel 构造分配 Vector、Cube、Reduction、Memory 和 Primary 等角色。",
        "inputs": "已标记结构关系的依赖图",
        "outputs": "ascend.op_roles 和 primary 角色标记",
        "inspect_hint": "检查预期的计算 root 是否成为 Primary，并拿到正确角色。",
        "common_failures": "角色不匹配导致无法融合，或被分派到错误模板族。",
    },
    "final-patterns": {
        "title": "生成最终 Kernel Pattern",
        "purpose": "合并候选、选择 kernel 分区，并附加 kernel id。",
        "inputs": "已完成角色分类的候选图",
        "outputs": "ascend.kernel、ascend.primary、模板族元数据和 kernel 图边",
        "inspect_hint": "检查 kernel id、primary op、模板族和跨 kernel 依赖边。",
        "common_failures": "非预期拆分/合并、缺少 primary op，或 kernel 图边错误。",
    },
    "ascend-kernelize": {
        "title": "Kernelize 输出边界",
        "purpose": "向 Schedule 暴露稳定的 kernelized IR。",
        "inputs": "最终 Kernel Pattern",
        "outputs": "带 kernel id 和 pattern 元数据的阶段边界 IR",
        "inspect_hint": "如果没有额外 cleanup，这一步可能和 Kernelize 最后一个内部 step 完全一致。",
        "common_failures": "边界 IR 与最终 Kernel Pattern 非预期不一致。",
    },
    "cleared": {
        "title": "清理旧 Schedule 元数据",
        "purpose": "移除上一次 schedule 属性，避免当前运行复用过期决策。",
        "inputs": "可能带旧 schedule 属性的 kernelized IR",
        "outputs": "已清理本 pass 拥有的 schedule 属性的 kernelized IR",
        "inspect_hint": "检查旧 decision_id、tile_params、tail_plan 和 target policy 是否已移除。",
        "common_failures": "旧 schedule 属性残留，导致后续 step 看起来成功但基于错误数据。",
    },
    "decisions": {
        "title": "选择 tile 和 tail 方案",
        "purpose": "构造 schedule problem，匹配模板，搜索候选，并选择运行时契约。",
        "inputs": "Kernel Pattern 和目标模型",
        "outputs": "schedule problem、symbol-axis contract、选中的 decision、tile_params、tail_plan",
        "inspect_hint": "检查 ScheduleProblem 的 shape/structure 约束、tileable_axes、required_reduction_axes、ScheduleDecisionSet 和 tuning cache key。",
        "common_failures": "没有可用模板、候选为空、guard budget 剪枝过度，或缺少目标模型数据。",
    },
    "final": {
        "title": "挂载 Schedule 契约",
        "purpose": "把选中的 schedule decision 挂到 IR 上，作为后续符号化运行时契约。",
        "inputs": "选中的 schedule decision",
        "outputs": "decision_id、tile_params、tail_plan、tail_policies、target_tile_policy、schedule_contract、symbol-axis contract 摘要",
        "inspect_hint": "Primary op 和 func 属性应暴露一致的 decision_id、tile_params、tail_policies、target_tile_policy、schedule_contract，并可回查 schedule report 中的轴约束。",
        "common_failures": "Kernel 元数据不一致、缺少 tile_params/schedule_contract，symbol-axis contract 缺失，或 tail plan 不完整。",
    },
    "ascend-schedule": {
        "title": "Schedule 输出边界",
        "purpose": "向 Realize 暴露稳定的 scheduled IR。",
        "inputs": "已挂载 schedule 契约的 IR",
        "outputs": "带符号化 tile 和 tail 契约的阶段边界 IR",
        "inspect_hint": "这一步通常与“挂载 Schedule 契约”一致，可查看 same-as 标记。",
        "common_failures": "边界 IR 与已挂载的 schedule 契约非预期不一致。",
    },
    "ascend-kernel-split": {
        "title": "拆分物理 Kernel",
        "purpose": "按 ascend.kernel 逻辑分组 outline 成独立 func.func，形成后续 CANN artifact 的物理 kernel 边界。",
        "inputs": "已挂载 schedule 契约和 kernel metadata 的 scheduled IR",
        "outputs": "一个或多个 global kernel function，以及跨 kernel carried buffer ABI",
        "inspect_hint": "检查 func.func @kernel_N 数量是否与 logical kernel 分组一致，cross-kernel value 是否变成函数参数/结果。",
        "common_failures": "logical kernel 分组不闭合、跨 kernel 值无法合法 ABI 化，或 split 后 schedule metadata 丢失。",
    },
    "planned": {
        "title": "构造内存实现计划",
        "purpose": "在不改变 tensor 语义的前提下规划 buffer、放置、搬运、workspace 和实现方式。",
        "inputs": "scheduled tensor IR",
        "outputs": "placement plan、movement plan、static memory plan",
        "inspect_hint": "检查计划中的 GM/on-chip 位置、搬运需求、workspace slot 和延迟决策。",
        "common_failures": "缺少 schedule 契约、放置不可行，或搬运路径不受支持。",
    },
    "bufferized": {
        "title": "Bufferize tensor IR",
        "purpose": "把 tensor value 转成 memref，同时保留 producer/consumer 和 view 链关系。",
        "inputs": "已规划的 tensor IR",
        "outputs": "带显式 buffer 和 view 链的 memref IR",
        "inspect_hint": "检查新增 alloc、subview、copy，以及 linalg inputs/outs 是否仍匹配计划值。",
        "common_failures": "Bufferization 失败、view 链关系丢失，或产生非预期临时 buffer。",
    },
    "memory-space-annotated": {
        "title": "标注内存空间",
        "purpose": "通过分配 GM/VECIN/VECCALC/VECOUT 空间以及必要 copy/view 来落地内存计划。",
        "inputs": "bufferized memref IR",
        "outputs": "memory_space 属性、workspace 布局、已物化的搬运",
        "inspect_hint": "检查 alloc/copy 节点、memory_space 属性、workspace slot 和延迟搬运计数。",
        "common_failures": "bridge 只应用了一半、缺少输出 copy、动态 view rewrite 无效，或 workspace 复用错误。",
    },
    "ascend-realize": {
        "title": "Realize 输出边界",
        "purpose": "向后端 lowering 暴露稳定的 memref/memory-space IR。",
        "inputs": "已标注 memory-space 的 IR",
        "outputs": "带 realized buffer 和搬运信息的阶段边界 IR",
        "inspect_hint": "如果没有额外 cleanup，这一步可能和 Realize 最后一个内部 step 完全一致。",
        "common_failures": "边界 IR 与 memory-space annotated IR 非预期不一致。",
    },
    "ascend-compute-lower": {
        "title": "Lower 到 AscendC IR",
        "purpose": "用 AscendC dialect op 替换 linalg/memref 计算和搬运。",
        "inputs": "realized memref IR",
        "outputs": "AscendC 计算、copy、local tensor 和 queue 操作",
        "inspect_hint": "检查 primary compute op 是否变成 AscendC loop/copy，tile arg 是否被正确使用。",
        "common_failures": "op 不受支持、缺少 tile_arg、内存空间无效，或 view 链不受支持。",
    },
    "ascend-parallelize": {
        "title": "映射并行循环",
        "purpose": "把编译器循环和符号化 tile 参数映射到 block 级并行执行。",
        "inputs": "带 tile arg 的 AscendC compute IR",
        "outputs": "block index 使用和并行化后的循环结构",
        "inspect_hint": "检查 ascendc.get_block_idx 和循环边界是否使用 TB_M/TB_N。",
        "common_failures": "循环未并行化、block 维度错误，或 tile arg 不再支配使用点。",
    },
    "ascend-prepare-for-emit": {
        "title": "准备 CANN 发射",
        "purpose": "把 AscendC IR 规范化为 CANN printer 和 host tiling ABI 可接受的形式。",
        "inputs": "已并行化的 AscendC IR",
        "outputs": "可发射的 AscendC global function 和 tiling struct 使用",
        "inspect_hint": "检查函数签名、py_struct tiling 数据，以及 global/local tensor 设置。",
        "common_failures": "ABI 形状不匹配、缺少 tiling 字段，或存在不支持的 emit 构造。",
    },
    "ascend-canonicalize-cann-signature": {
        "title": "规范化 CANN 签名",
        "purpose": "把函数 ABI 改写成最终 CANN kernel 签名。",
        "inputs": "可发射的 AscendC function",
        "outputs": "兼容 CANN 的 kernel 参数和 tiling struct",
        "inspect_hint": "检查输入/输出指针顺序、workspace/tiling 参数和最终 kernel 属性。",
        "common_failures": "参数顺序错误、缺少输出 buffer，或 tiling struct 不匹配。",
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
    StageArtifact(41, "045-kernel-split-out", "stages/045-kernel-split-out.mlir", "Kernel Split", "ascend-kernel-split"),
    StageArtifact(42, "041-realize-planned", "stages/041-realize-planned.mlir", "Realize", "planned"),
    StageArtifact(43, "042-realize-bufferized", "stages/042-realize-bufferized.mlir", "Realize", "bufferized"),
    StageArtifact(44, "043-realize-memory-space-annotated", "stages/043-realize-memory-space-annotated.mlir", "Realize", "memory-space-annotated"),
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
    artifacts: list[dict[str, Any]] | None = None,
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
        "artifacts": artifacts or [],
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
