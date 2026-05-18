# Ascend-MLIR V2 实现跟踪看板

本文档用于跟踪 `Ascend-MLIR-Detailed-Implementation-V2` 的工程落地进展。

- 详细规格入口：`docs/Ascend-MLIR-Detailed-Implementation-V2.zh.md`
- 拆分规格：`docs/Ascend-MLIR-Detailed-Implementation-V2-1.zh.md` 到 `V2-9.zh.md`
- 第一轮 MVP 计划：`docs/superpowers/plans/2026-05-07-ascend-mlir-v2-mvp.md`
- Phase 1 Kernelize 计划：`docs/superpowers/plans/2026-05-08-ascend-mlir-v2-kernelize-candidates.md`
- Phase 2 Schedule 计划：`docs/superpowers/plans/2026-05-08-ascend-mlir-v2-schedule-full-search.md`
- 代码命名去版本化计划：`docs/superpowers/plans/2026-05-09-ascend-mlir-remove-v2-code-naming.md`
- Phase 3B target-aware placement 计划：`docs/superpowers/plans/2026-05-11-ascend-realize-target-aware-placement-mvp.md`
- Phase 3B workspace layout/lifetime 计划：`docs/superpowers/plans/2026-05-11-ascend-realize-workspace-layout-lifetime-mvp.md`
- Phase 3B explicit data movement 计划：`docs/superpowers/plans/2026-05-11-ascend-realize-data-movement-plan-mvp.md`
- Phase 3B memory-space annotation 计划：`docs/superpowers/plans/2026-05-11-ascend-realize-memory-space-annotate-mvp.md`
- Phase 5 backend integration 设计：`docs/superpowers/specs/2026-05-11-ascend-phase5-backend-integration-design.md`
- Phase 5 backend integration 计划：`docs/superpowers/plans/2026-05-11-ascend-phase5-backend-integration.md`
- Phase 5C full pipeline gap 记录：`docs/superpowers/plans/2026-05-11-ascend-phase5c-full-pipeline-gap.md`
- Phase 5C Realize-to-Phase5 bridge 计划：`docs/superpowers/plans/2026-05-11-ascend-realize-phase5-bridge.md`
- Phase 5C broadcast-add-reduce mainline E2E 计划：`docs/superpowers/plans/2026-05-11-broadcast-add-reduce-mainline-e2e.md`
- Axis schedule contract / coalescing 泛化计划：`docs/superpowers/plans/2026-05-12-axis-schedule-contract-coalescing.md`
- Phase 5C+ dynamic shape mainline completion 计划：`docs/superpowers/plans/2026-05-14-phase5c-dynamic-shape-mainline-completion.md`
- Runtime graph / queue lifetime hardening 计划：`docs/superpowers/plans/2026-05-18-ascend-runtime-graph-and-queue-lifetime.md`
- Phase 5C full pipeline bridge 报告：`docs/Ascend-MLIR-Phase5C-Full-Pipeline-Bridge-Report.zh.md`

说明：`V2` 在本文档中只表示方案版本。当前代码目录、namespace、CMake target、IR attrs、测试 target 使用版本无关 `Ascend` 命名。

## 状态约定

| 状态 | 含义 |
|---|---|
| `Done` | 已实现、已 review、已在 xvm/docker 中完成必要验证 |
| `In Review` | 已实现，等待或正在 code review / spec review |
| `In Progress` | 正在开发 |
| `Planned` | 已明确范围，尚未开始 |
| `Blocked` | 受外部依赖或设计问题阻塞 |
| `Deferred` | 有意延后，不属于当前批次 |

## 当前总览

| 阶段 | 范围 | 状态 | 当前结论 |
|---|---|---|---|
| Phase 0 | V2 MVP 编译主干 | `Done` | Normalize -> Kernelize -> Schedule 纵向链路已打通 |
| Phase 1 | Kernelize 完整候选分析 | `Done` | Kernelize Phase 1 候选分析与 pattern partition 路径已完成并验证 |
| Phase 2 | Schedule 完整搜索与 guard/cache | `Done` | Phase 2 final review 与 xvm/docker 验证已完成 |
| Phase 3 | Realize plan objects | `Done` | `--ascend-realize` plan-object MVP、review follow-up 与代码命名去版本化已完成 |
| Phase 4 | Target model 完整化 | `Done` | `TargetMemoryModel`、`TargetIntrinsicModel`、`TargetCostModel`、`TargetModelVerifier` MVP 已完成；多 SoC 覆盖后续增强 |
| Phase 3B | Realize materialization 增强 | `Done` | One-Shot Bufferize、target-aware placement plan、workspace layout/lifetime、explicit data movement plan、memory-space annotation MVP、普通 vector output Phase 5 bridge 已完成；selected movement 已支持 local alloc/copy、同形 workspace subview 物化、跨形状 flat workspace packing，以及 `tensor.extract_slice` / `tensor.cast` / `tensor.expand_shape` / `tensor.collapse_shape` / `tensor.reshape` 对应的 memref view-chain consumer 重写；dynamic `extract_slice` / `reshape` view-chain 会在目标 memory space 上重建并回写实际 rewrite 计数；StaticMemoryPlanner 已支持非重叠 vector temporary 复用物理 slot offset |
| Phase 5 | Translate / runtime artifact 对接 | `Done` | 官方 Ascend backend 入口、support matrix、ABI wrapper、runtime artifact emitters 与 transformer smoke 已完成 |
| Phase 5C | Full pipeline ordinary acceptance | `Done` | 最小普通 tensor/linalg 用例已通过完整 Phase 0 -> Phase 5 positive smoke；`broadcast-add-reduce`、`relu-broadcast-transpose`、`add-broadcast-concat`、`gather-elementwise-fusion`、`split-relu-brc-add-mul`、`matmul-add-leakyrelu`、`two-kernel-dag`、`two-kernel-rank-mix-dag`、`three-kernel-dag`、`rmsnorm-reduction-core` 已有新主线 demo 并通过 runtime-session sim；example suite 已接入 queue lifetime checker |
| Phase 5C+ | Shape-general Schedule 泛化 | `In Progress` | `axisScheduleConstraints` / `axisCoalescingHints`、role-driven reduction/vector/cube tiling、tail-policy guards、schedule tile metadata 已落地；cube schedule tile 已覆盖 matmul `[M,N,K]` logical axes，K 轴进入 tail plan；Schedule 搜索已扩展到 axis-product / coalescing-hint 候选并加入 cost ranking；`ScheduleDecision` guard ownership 已收敛到 `ScheduleInstance`；StructuredLowering 已发射 guard/tail marker attrs、target tile policy metadata 和 per-kernel schedule metadata；runtime manifest 已支持 compiler-generated multi-kernel DAG、per-kernel schedule metadata 以及 shape/workspace/resource schema，且 workspace size 可由 Realize static planner 回填到 CANN artifact ABI；target-aware tile policy 已能消费 CANN target model，gather semantic alignment 已从 `TargetMemoryModel::AlignmentRule` 推导；Schedule tuning signature 已支持 module-level、line-based file cache 与 schema-versioned tuning DB；主线 examples 已迁移为显式 `target-aware` helper；Kernelize 已提供 registry + generated OpInterface 双路径语义契约，linalg/tensor/arith external models 已真实注册；attention_sdpa handwritten contract 已进入 Schedule template；transpose Kernelize 已去 rank/permutation 定制，rank2 swap transpose backend lowering 保持闭环，transformer dynamic mainline full codegen / backend / translate smoke 已通过 |
| Phase 6 | 架构文档与 demo 重写 | `Deferred` | 待 V2 主链路稳定后启动 |

## Design vs Implementation Gap Board

| Area | Design Target | Current Implementation | Status | Next Closure Step |
|---|---|---|---|---|
| ScheduleDecision ownership | `ScheduleDecision` refines `ScheduleInstance` without copying instance-owned fields | `ScheduleDecision` owns `instance`; guard counts are read through `instance`; static guard forbids duplicate fields | `Closed` | Keep static contract guard in regression |
| Structured markers | `TailPlanMarker` / guard markers / cache markers are explicit IR carriers | Schedule emits selected tile, guard marker, tail marker, tail policy and target tile policy attrs | `MVP Closed` | Materialize richer marker objects when cache/runtime consumers need them |
| Target-driven tile | Tile selection consumes target memory / intrinsic / cost model | `--ascend-schedule='target-tile-policy=target-aware ...'` loads CANN target profile, verifies memory / intrinsic / cost models, and derives tile size from UB capacity plus data-movement path cost availability；tail-policy preference 与 vector buffer count 已进入 `TargetTilePolicy`，`ScheduleDecision` 读取策略对象而不是静态优先序；静态可估 footprint 的 kernel 按 UB 容量推导 tile，dynamic inner / dynamic reduction footprint 保守落到后端已验证的默认 tile，并用 `target_dynamic_*_32` policy id 显式标记；stock `--ascend-schedule` 默认改为 `require-explicit`，兼容调用必须显式写 `target-tile-policy=legacy-default`，商用调用必须显式写 `target-aware`；ScheduleSearch 已生成 role-driven、result、half/split、axis-product 与 coalescing-hint 候选，并通过 cost ranking 保持 bounded tile 优先；module-level `ascend.schedule.tuning_cache`、`tuning-cache-in/out` line-based 文件 cache 与 `tuning-db-in/out` schema-versioned tuning DB 已支持跨编译进程复用 selected tuning signatures；主线 examples 通过 `examples/mainline-target-env.sh` 统一显式选择 `target-aware` | `MVP Closed` | 后续让 tuning DB 承接实测 score / negative records 的完整 auto-tuning 数据 |
| Multi-kernel manifest | Runtime manifest can describe multiple kernel entries | Runtime manifest preserves legacy single-kernel root fields and emits explicit multi-kernel `kernel_entries` plus `kernelGraph.nodes/edges`；Kernelize 已从最终 kernel partition 生成 `ascend.kernel_graph.edges`，manifest 不再依赖手写 module attrs；translator 已优先消费 `ascend.schedule.kernel_metadata` per-kernel schedule schema，并把内部 kernel id 映射为最终 CANN entry id，同一 entry 内部边会被丢弃，避免 `kernel_1` 这类 partition id 泄漏到 manifest；per-kernel 与 root schema 已补充 `shape`、`workspace`、`resources` 字段；Realize 会把静态 workspace bytes 写入 `cann.workspace_size_bytes`，CANN tiling space、runtime manifest、host tiling helper 统一消费该 ABI attr | `Closed` | 后续扩展动态 shape workspace expression，而不是回退到常量 0 |
| Transformer dynamic | Full transformer graph compiles through new mainline | `examples/transformer/run-mainline.sh` proves Normalize + Kernelize prefix, verifies rank-agnostic transpose Kernelize, and now passes full codegen / Phase5 backend / translate / runtime artifact smoke；自动 SDPA-like matcher 已 fail-closed，避免把 MLP 双 matmul 误合成一个 handwritten kernel | `Closed` | 后续补专用 transformer performance lowering |
| Cross-stage contract hardening | 上层主干不应靠裸字符串 / 重复 enum / 隐式 dialect 注册扩展 | `ascendc.unit`、`AiCore.*`、`ascendc.kernel_kind`、op-role、gather attrs 已集中到 `Common/Attributes.h`；Realize `MemoryPlace` 与 Backend `MemorySpace` 均已统一 alias 到 target profile 枚举；Normalize / Kernelize / Schedule / Realize 已补齐 dependent dialects；Kernelize 已新增 public `KernelizeOpInterface` 语义契约与 public `KernelizeOpModelRegistry`，linalg / tensor-view / arith-constant 语义 helper 已抽到共享实现，默认 registry 只作为 fallback；generated `KernelizeSemanticOpInterface` 已有真实 external model registration，`afir-opt` 对 linalg/tensor/arith 优先走 external model，AFIR 自有 op 可通过 MLIR native interface 提供 semantics；`DependencyAnalysis` 消费 `Analyze` / `Transparent` / `Unsupported` participation，不再靠私有 target-op 匹配，unsupported tensor producer 已 fail-closed 并给出 producer/consumer diagnostic；native / external model populate failure 也在 `DependencyAnalysis` 入口 fail-closed，避免失败 interface op 从 producer graph 静默消失；Kernelize access pattern 不再按 `linalg.matmul` / `batch_matmul` / `transpose` / `fill` 名字分支，generic contraction 由 indexing maps + iterator kinds 识别；Kernelize iterator type 分析改为 `IteratorKind` enum，不再对 attr printed form 做字符串 contains；Kernelize fallback、Schedule pattern view、Realize memory-space annotate / Phase5 unit annotation 已优先消费 `ascend.op_roles` array 再兼容 legacy `ascend.op_role`；`KernelizeConfig` 已删除未消费的 `maxBranchesPerCandidate` / `localTopKPerPrimaryOpNeighborhood` / `maxPrimitivePerOp` 假配置；Fusion primitive 名称已从裸字符串收敛为 `KernelizePrimitiveKind` enum；`KernelPatternEdgeKind::MustCoLocate` / `MustSeparate` 已进入 partition 决策；`ascend.kernelize.handwritten_group`、`ascend.kernelize.handwritten_kind`、`ascend.kernelize.template_families` 与自动 SDPA-like matcher 已可注入 `HandwrittenPattern` 候选，由 closure / contract / partition / Schedule template 主链路统一校验；Phase5 body classifier 已收敛为 `Backend/LinalgBodyClassifier` 单一来源，Realize 与 ComputeLower 共用；Schedule gather semantic alignment 已由 target-aware `AlignmentRule` 覆写；Kernelize branch/merge group 已沿允许的链传播到中间结构节点；Schedule `Memory` role 已有 `memory_copy` 模板与 full-axis tile 兜底 | `P0 Slice Closed` | 后续把更多 AFIR 自有 op 迁移到 native interface，逐步减少 fallback registry 体量 |
| Kernelize template family merge | primitive merge 不应依赖 `(lhs,rhs)` 有序对，`vector+cube` 与 `cube+vector` 等组合必须对称解析 | `CandidateMergeAnalysis` 的 table-family 解析已迁入 `KernelizeFamilyResolver`；candidate 构造优先消费 `preferredTemplateFamilies` trait，再回退 role-derived families；解析器保留 `vector+reduction -> reduction`、`cube+vector -> cube`、同 family intersection 优先级，并输出 `family_resolver = "kernelize_trait_resolver"` report；reverse vector->cube 回归已覆盖 | `Closed` | 后续让更多 primitive/family 由 op interface 或 backend trait 提供 |
| Kernelize reduction fusion closure | 非种子 reduction 应能被融合进父 vector kernel，长链 merge 应迭代收敛，horizontal dependency 应沿分析图而非 raw SSA | reduction role 不再无条件标记 `Primary`；`KernelizeSeedPolicy` 显式表达 `MaySeed` / `NonSeedWhenFused` / `NeverSeed`，linalg reduction 默认 `NonSeedWhenFused` 并进入 `DependencyAnalysis` report；`ConsumerIntoPrimary` 按 seed policy 而非 role-only heuristic 选择 vector consumer 作为 primary，softmax-like row reduction 可进入 vector kernel；`CandidateMergeAnalyzer` 改为 fixed-point 合并，3+ 长链可继续喂回；horizontal reachability 改为消费 `DependencyAnalysisResult.index.consumers`，并要求 sibling primary result shape 兼容，避免仅因共享常量把不同 shape 的 fill 合成一个 kernel；DependencyAnalysis 只沿 tensor-typed operands 追溯无 region 中间 op，支持 tensor view 链同时避免把 `tensor.empty(%dim)` shape 依赖误当 data dependency；AxisCoalescer 的 axis carrier 与 primary output carrier 解耦，非 primary reduction 仍可承载 fused reduction+vector epilogue 的 reduction axes | `Closed` | 后续按 op/interface 细化 seed policy，而不是恢复 role-only 判断 |
| Realize plan / IR mutation | `MemoryRealizationPlan` 必须由 materialize 层驱动真实 IR mutation 与计数回写 | `MemoryRealizationDriver::materialize(module, bundles, MemorySpaceAnnotate)` 已统一执行 plan 校验、memory-space annotation、Phase5 bridge 和 per-kernel alloc/copy 计数回写；`RealizePass` 不再直接绕过 driver 调用 annotate / bridge helpers；`BufferizationDriver` 已按 `ascend.op_roles` array 识别 vector temporary，并把 `tensor.extract_slice` / `tensor.cast` / `tensor.expand_shape` / `tensor.collapse_shape` / `tensor.reshape` 作为透明 tensor view 追溯 producer/consumer 关系；target-aware workspace slot 落到 `VECIN`，生产 plan 可选择 `GM -> VECIN` direct path，不再生成不可达 `GM -> VECCALC` movement；Phase5 concat/subview output bridge 已增加 dim-use preflight，失败路径不再留下半插入 `VECOUT` alloc；cube->vector bridge 使用 dominance 判定安全 consumer，并通过 `LinalgBodyClassifier` 支持 `linalg.matmul` / `linalg.batch_matmul`；selected value-level movement step 已在同一入口物化为 local alloc/copy，静态同形同 block 输入可合并到 workspace base + rank-reduced subview；跨形状 selected movement 已用 flat workspace + `memref.reinterpret_cast` view 打包；consumer 是 `memref.subview` / `memref.cast` / `memref.expand_shape` / `memref.collapse_shape` / `memref.reshape` view-chain 时会在目标 memory space 上重建 view 后改写 linalg input；dynamic subview / reshape 链按实际 rewrite 数更新 `dynamic_view_chain_rewrites` report；movement view-chain 物化前已增加 group/item preflight，无法重建的 dynamic view-chain 会计入 `deferred_view_chain_rewrites` 并避免半改 IR；StaticMemoryPlanner 已复用非重叠 vector temporary 的物理 slot offset并回写真实 peak/workspace bytes | `MVP Closed` | 后续补 region 跨 block / 更多 view-like op 的 selected movement materialization |
| Schedule cube K axis | cube schedule 必须覆盖 matmul 的 `[M,N,K]` logical axes，不得只按 result `[M,N]` 生成 tile/tail plan | cube role-driven tile 与 full tile 已改为基于 `CoalescedAxisInfo.logicalAxes`，selected tile shape 保留 K 轴；rank2 vector/reduction selected-tile lowering 允许消费 kernel-level tile 前缀，避免 cube+vector epilogue 因 K 轴元数据失败 | `Closed` | 后续补 batch_matmul rank3 schedule/lowering |
| Schedule function metadata scope | 单 `func.func` 内多个 kernel 不得把第一个 kernel 的 selected tile / tail plan 静默当成全函数事实 | `StructuredLoweringDriver` 已写入 `ascend.schedule.kernel_metadata` per-kernel array；单 kernel / 同 metadata 情况保留 legacy func-level attrs，多 kernel metadata 不一致时清理 legacy attrs，避免下游读取错误整函数事实；`CannRuntimeArtifacts` 已按 kernel 名消费 per-kernel selected tile / tail plan / decision id，legacy func-level attrs 只作为单 kernel fallback | `Closed` | 后续 runtime manifest 扩展 workspace / resource 字段时继续沿用 per-kernel schema |
| Axis static extent consistency | 同一 logical axis 不能在同一 kernel pattern 内静默合并不同静态 extent | `AxisCoalescer` 在同轴静态 extent 冲突时记录 barrier 并发出 op 级错误，避免把第一个 extent 写成错误 guard / selected tile | `Closed` | 后续将所有 coalescing barrier 的消费策略统一到 ScheduleProblem |
| Public header boundary | 商用 API 只暴露 pass 入口与 Common/Backend 稳定契约，阶段内部 analyzer / planner / types 不进入 public include | `Kernelize` / `Schedule` / `Realize` public include 目录只保留 `*Pass.h`、public semantic contract 与必要 external-model registration declaration；内部头已移到 `lib/Conversion/Ascend/...`，内部单测通过私有 include dir 访问；`ascend-public-header-boundary.mlir` 与 `check_ascend_public_headers.sh` 守住边界，当前允许 `KernelizePass.h`、`KernelizeOpInterface.h`、`KernelizeExternalModels.h` | `Closed` | 后续若新增内部头，默认放在 `lib` 私有目录；新增公共头必须同步 guard |

## Phase 5C+ ROI Closure Batch

本批次承接专家二轮 review 后的 9 个 ROI 项，目标是把已关闭的设计 gap 接到生产路径或测试守门，而不是扩大到完整商用优化器。

| 项 | 状态 | 主要产物 | 验证 |
|---|---|---|---|
| Schedule 搜索空间扩展 | `Done` | `ScheduleSearch` 增加 axis-product 与 coalescing-hint tile candidates，`generated_candidates` 覆盖扩大；cost ranking 已纳入排序并保持 bounded vector/reduction tile 优先 | `ascend-schedule-search.mlir`；`ascend-schedule-vector-bounded-tile.mlir` |
| Schedule cache / report 正确化 | `Done` | `selectedDecisionEntries`、`guardBudgetPruned`、`negativeCacheHits`、`persistentTuningHits` 分离；只统计选中 decision；module-level `ascend.schedule.tuning_cache` 支持同一 pipeline 内重复 schedule pass 复用 tuning signature | `ascend-schedule-cache.mlir` |
| 主线 examples 显式 target-aware | `Done` | `examples/mainline-target-env.sh` 统一解析 CANN root；主线 demo scripts 显式传 `target-tile-policy=target-aware` | `bash -n examples/*/run-mainline.sh examples/transformer/run-mainline.sh` |
| compiler-generated kernel DAG metadata | `Done` | `KernelPattern` 根据最终 partition 写 `ascend.kernel_graph.edges` | `ascend-kernelize-kernel-graph.mlir` |
| Runtime / manifest 消费 per-kernel schedule metadata | `Done` | `CannRuntimeArtifacts` 按 kernel 名读取 `ascend.schedule.kernel_metadata` 的 tile/tail/decision id，并在 root / per-kernel entry 输出 shape/workspace/resource schema；workspace size 不再硬写 0，统一读取 `cann.workspace_size_bytes`，该 attr 可由 Realize static planner 回填 | `cann-translate-runtime-artifacts.mlir`；`cann-translate-runtime-artifacts-multi.mlir`；`ascend-realize-workspace-layout.mlir` |
| Realize selected movement IR materialization | `Done` | `materializeMovementSteps` 在 memory-space materialize 生产路径插入 `VECIN` local alloc/copy；movement 先于 annotation，避免 plan source 被 VECCALC 重写；StaticMemoryPlanner 已复用非重叠 vector temporary 的物理 slot offset；consumer view-chain 已覆盖 `tensor.extract_slice` / `tensor.cast` / `tensor.expand_shape` / `tensor.collapse_shape` / `tensor.reshape`；dynamic subview / reshape 链按实际 rewrite 数更新 movement report | `ascend-realize-workspace-layout.mlir`；`ascend-realize-view-chain-movement.mlir`；`ascend-realize-dynamic-view-chain-movement.mlir`；`ascend-realize-reshape-view-chain-movement.mlir`；`AscendRealizePlannerTest.StaticMemoryPlannerReusesOffsetsForNonOverlappingVectorTemporaries` |
| Kernelize reduction seed policy | `Done` | `KernelizeSeedPolicy` 将 reduction seed 行为显式化；linalg reduction 默认 `NonSeedWhenFused`，`DependencyAnalysis` report 输出 `seed_policy`，`ConsumerIntoPrimary` 按 seed policy 融合 reduction->vector，而不是继续依赖 role-only heuristic | `ascend-kernelize-reduction-seed-policy.mlir`；`ascend-kernelize-reduction-fusion.mlir`；`AscendKernelizeOpInterfaceTest` |
| Kernelize primitive / family trait resolver | `Done` | `KernelizeFamilyResolver` 统一解析 template families；candidate 构造优先消费 `preferredTemplateFamilies`，再回退 role-derived families；merge report 输出 `family_resolver`；handwritten / must-colocate / must-separate attrs 集中到 `Common/Attributes.h` | `ascend-kernelize-template-family-traits.mlir`；`ascend-kernelize-merge-horizontal.mlir`；`AscendKernelPatternTest` |
| Schedule persistent tuning cache IO | `Done` | `--ascend-schedule` 保留 `tuning-cache-in` / `tuning-cache-out` line-based 兼容路径，并新增 `tuning-db-in` / `tuning-db-out` schema-versioned tuning DB；DB record 带 `schema=1`、target、policy、signature、family、template、result/tile shape，按 target/policy 过滤后 seed `ScheduleCacheModel`，输出稳定排序 | `ascend-schedule-persistent-cache.mlir`；`ascend-schedule-cache.mlir`；`ascend-schedule-tuning-db.mlir`；`ascend-schedule-tuning-db-invalid.mlir` |
| KernelizeOpInterface / trait model | `Done` | public `KernelizeOpSemanticInfo` contract added；linalg / tensor view / arith constant semantic helpers 已抽到共享实现；`KernelizeOpModelRegistry` 先查 native/external `KernelizeSemanticOpInterface`，再走 fallback registry；`registerKernelizeExternalModels` 已真实 attach linalg/tensor/arith external models，`afir-opt` 默认注册；`DependencyAnalysis` consumes participation (`Analyze` / `Transparent` / `Unsupported`) instead of private target-op matching；unsupported tensor producers now fail closed with producer/consumer diagnostic；generated `KernelizeSemanticOpInterface` 支持 AFIR-owned ops 通过 MLIR native interface 提供 semantics，registry fallback 保持兼容 | `ascend-kernelize-op-interface-*.mlir`；`ascend-kernelize-op-interface-native.mlir`；`AscendKernelizeOpInterfaceTest` |
| HandwrittenPattern 注入器 | `Done` | `ascend.kernelize.handwritten_group` 和自动 SDPA-like matcher 生成 `HandwrittenPattern` candidate，统一走 closure / contract / partition；matcher fail-closed 到 seed cube -> reduction -> vector chain -> second cube，避免误收 MLP 双 matmul；SDPA-like matcher 写入 `handwritten_kind = attention_sdpa` 与 `template_families = [attention_sdpa, cube]`，Schedule 选择 `attention_sdpa/grouped_tile_per_block`，attention-like supported-body smoke 可通过 ComputeLower 且不残留 `linalg.`；真实 softmax `subf/exp/div` 后端 lowering 仍以明确 unsupported diagnostic fail-closed | `ascend-kernelize-handwritten-pattern.mlir`；`ascend-kernelize-attention-handwritten-pattern.mlir`；`ascend-schedule-attention-handwritten-pattern.mlir`；`ascend-full-pipeline-attention-handwritten-smoke.mlir` |
| 文档收敛 | `Done` | 本节与 Gap Board 更新，记录本批次边界和验证项 | 本文档 |

本批次后仍保留的非目标：

- Realize selected movement 已支持跨形状 flat workspace packing 与 `memref.subview` / `memref.cast` / `memref.expand_shape` / `memref.collapse_shape` / `memref.reshape` consumer view-chain 重写；dynamic view-chain 当前覆盖 dynamic subview / reshape，region 跨 block 与更多 view-like op 仍是后续增强。
- Schedule 已有 module-level tuning signature 复用和 line-based file cache；完整 auto-tuning result database / cost-history schema 仍是后续增强。
- Public semantic contract、registry fallback、generated TableGen OpInterface 与 external model registration hook 已完成；后续按 AFIR 自有 op 增量接入具体 external/native models。
- HandwrittenPattern 已有 fail-closed SDPA-like 自动 matcher；FlashAttention 专用 schedule/lowering 不在本批次。

## Phase 0：V2 MVP 编译主干

| 任务 | 对应规格 | 状态 | 主要产物 | 验证 |
|---|---|---|---|---|
| V2 pass skeleton | V2-1 / V2-9 | `Done` | `--ascend-normalize`、`--ascend-kernelize`、`--ascend-schedule` | `check-afir` 覆盖 |
| Target Profile MVP | V2-8 | `Done` | `TargetProfile`、`CannTargetProfileLoader`、`--ascend-print-target-profile` | `test/Target/ascend-target-profile.mlir` |
| Normalize MVP | V2-2 | `Done` | dialect 白名单、`cf.assert` shape-guard op 级例外、`ascend.normalized` | `test/Conversion/ascend-normalize.mlir` |
| Kernelize MVP | V2-3 | `Done` | `ascend.op_role`、`ascend.kernel`、`ascend.primary` | `test/Conversion/ascend-kernelize-mvp.mlir` |
| Schedule MVP | V2-4 | `Done` | fixed schedule family/template/decision attrs | `test/Conversion/ascend-schedule-mvp.mlir` |
| Vertical MVP pipeline | V2-9 | `Done` | Normalize -> Kernelize -> Schedule smoke test | `test/Conversion/ascend-pipeline-mvp.mlir` |

### Phase 0 验证记录

验证环境：

- 主机：只做代码开发
- xvm/docker：编译与测试
- 同步路径：`/home/niu/code/Ascend-MLIR`
- 实际可用 LLVM build：`/home/niu/code/llvm-project/llvm/build`

已执行：

```bash
cmake -S . -B build-v2-verify -G Ninja \
  -DLLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build \
  -DAFIR_ENABLE_BINDING_PYTHON=false

cmake --build build-v2-verify --target afir-opt -j10

/home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v \
  build-v2-verify/test/Conversion/ascend-normalize.mlir \
  build-v2-verify/test/Conversion/ascend-kernelize-mvp.mlir \
  build-v2-verify/test/Conversion/ascend-schedule-mvp.mlir \
  build-v2-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir

ASCEND_TOOLKIT_HOME=/home/niu/Ascend/20260323_newest/cann \
/home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v \
  build-v2-verify/test/Target/ascend-target-profile.mlir

cmake --build build-v2-verify --target check-afir -j10
```

结果：

| 命令 | 结果 |
|---|---|
| `afir-opt` build | passed |
| V2 focused lit | 4/4 passed |
| target profile lit | 1/1 passed |
| `check-afir` | 30 discovered, 27 passed, 3 unsupported |

## Phase 1：Kernelize 完整候选分析

目标：把当前 MVP 的单 op 标记升级为 V2-3 定义的可开发级候选分析。

| 任务 | 状态 | 说明 | 验收 |
|---|---|---|---|
| `DependencyAnalyzer` | `Done` | 构建 producer/consumer/use-def 索引 | 覆盖简单链、分支、共享输入 |
| `OpSemanticSummary` | `Done` | 统一 op 语义摘要 cache | debug report 可输出每个 op 摘要 |
| `StructuralMarker` | `Done` | 标记 gather/broadcast/reduction/matmul 等结构 | lit 覆盖结构标记 |
| `OpRoleClassifier` 完整化 | `Done` | 从 MVP role 扩展到主/辅角色分类 | vector/reduction/cube/gather 等覆盖 |
| primitive seed / expand | `Done` | primitive 驱动候选生成 | 候选含 primaryOps/primitives |
| legality / profitability | `Done` | 三阶段早剪枝 | report 输出过滤原因 |
| `CandidateClosure` | `Done` | 闭包计算与合法性判断 | 覆盖跨 producer/consumer 场景 |
| candidate merge | `Done` | 单主角色候选合并 | 稳定排序和 tie-break |
| horizontal fusion | `Done` | 水平融合候选 | 独立 lit 覆盖 |
| `KernelPatternGraph` | `Done` | 构建 kernel pattern graph | 输出 carried value / barrier 等边 |
| `KernelPartitioner` 完整化 | `Done` | 最终 kernel 划分 | 多 kernel pipeline smoke |

计划文件：

- `docs/superpowers/plans/2026-05-08-ascend-mlir-v2-kernelize-candidates.md`

### Phase 1 执行记录

| 计划任务 | 状态 | 提交 | 验证 |
|---|---|---|---|
| Task 1: Shared Kernelize Data Model | `Done` | `bb7a9c0` | `git diff --check` passed；xvm `afir-opt` build passed；`ascend-kernelize-mvp.mlir` 1/1 passed |
| Task 2: Dependency Analysis and Semantic Summary | `Done` | `8fea70c` / `25ac438` / `762162c` / `b03c8d1` | `git diff --check` passed；xvm `ascend-kernelize-dependency.mlir` 与 `ascend-kernelize-mvp.mlir` 2/2 passed |
| Task 3: Structural Marking | `Done` | `18e0a2d` / `1fb4c2b` | `git diff --check` passed；xvm `ascend-kernelize-roles.mlir`、`ascend-kernelize-dependency.mlir`、`ascend-kernelize-mvp.mlir` 3/3 passed |
| Task 4: Full OpRole Classification | `Done` | `af8a78d` / `61b63d6` | `git diff --check` passed；xvm `ascend-kernelize-roles.mlir`、`ascend-kernelize-dependency.mlir`、`ascend-kernelize-mvp.mlir`、`ascend-v2-pipeline-mvp.mlir` 4/4 passed |
| Task 5: Primitive Candidate Analysis and CandidateClosure | `Done` | `28970ee` / `cf95af4` / `d25f4dc` | `git diff --check` passed；xvm `ascend-kernelize-candidates.mlir`、`ascend-kernelize-roles.mlir`、`ascend-kernelize-dependency.mlir`、`ascend-kernelize-mvp.mlir`、`ascend-v2-pipeline-mvp.mlir` 5/5 passed |
| Task 6: Candidate Merge Analysis | `Done` | `4f8d11c` / `c0879fe` / `64762fe` | `git diff --check` passed；xvm `ascend-kernelize-merge-horizontal.mlir`、`ascend-kernelize-candidates.mlir`、`ascend-kernelize-roles.mlir`、`ascend-kernelize-dependency.mlir`、`ascend-kernelize-mvp.mlir`、`ascend-v2-pipeline-mvp.mlir` 6/6 passed |
| Task 7: Horizontal Fusion Analysis | `Done` | `29e4c92` / `fe55c2b` | `git diff --check` passed；xvm `ascend-kernelize-merge-horizontal.mlir`、`ascend-kernelize-candidates.mlir`、`ascend-kernelize-roles.mlir`、`ascend-kernelize-dependency.mlir`、`ascend-kernelize-mvp.mlir`、`ascend-v2-pipeline-mvp.mlir` 6/6 passed；本轮 horizontal source 允许 closed single-primary fallback candidates；共享输入按 DPS inputs ∩ closure external inputs，互不可达使用 raw SSA reachability 保守检查 |
| Task 8: KernelPattern Graph and Partitioner | `Done` | `279040d` | `git diff --check` passed；xvm focused lit 7/7 passed |

### Phase 1 验证记录

验证环境同 Phase 0：

- 主机：只做代码开发与 static check
- xvm/docker：编译与测试
- 同步路径：`/home/niu/code/Ascend-MLIR`
- 实际可用 LLVM build：`/home/niu/code/llvm-project/llvm/build`

已执行：

```bash
git diff --check

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-verify/test/Conversion/ascend-kernelize-dependency.mlir build-v2-verify/test/Conversion/ascend-kernelize-roles.mlir build-v2-verify/test/Conversion/ascend-kernelize-candidates.mlir build-v2-verify/test/Conversion/ascend-kernelize-merge-horizontal.mlir build-v2-verify/test/Conversion/ascend-kernelize-patterns.mlir build-v2-verify/test/Conversion/ascend-kernelize-mvp.mlir build-v2-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target check-afir -j10'
```

结果：

| 命令 | 结果 |
|---|---|
| `git diff --check` | passed |
| `afir-opt` build + Phase 1 focused lit | 7 discovered, 7 passed |
| `check-afir` | 35 discovered, 32 passed, 3 unsupported |

提交范围：

- Phase 1 实现：`bb7a9c0` through `279040d`
- Phase 1 跟踪文档：`8450631`、`8fc44d1`、`44ad7ec`、`d5ae92a`、`0c6e243`、`3c7968b`、`91b6e89`，以及本验证提交

## Phase 2：Schedule 完整搜索

目标：把 fixed schedule MVP 升级为 V2-4 的可搜索、可 guard、可缓存调度系统。

| 任务 | 状态 | 说明 | 验收 |
|---|---|---|---|
| `KernelPatternView` | `Done` | 从 `ascend.kernel` / `ascend.primary` / `ascend.op_role` 重建 pattern-level schedule view | 新增 pattern-view lit；同一 kernel 内 ops 共享 schedule decision |
| `AxisCoalescer` | `Done` | 轴合并与 coalesced axis info | rank-2/reduction/broadcast/matmul/multi-primary lit |
| `ScheduleProblemBuilder` | `Done` | 从 `KernelPatternView` + axis info 构建调度问题 | report 输出 shape/axis/constraint |
| `TemplateRegistry` | `Done` | 注册 schedule family/template | `vector_generic` 不再按 rank 拆 template；vector/reduction/cube family 可查询 |
| `ScheduleSearch` | `Done` | 搜索 `ScheduleInstance` | compileTimeTopK 生效 |
| guard 生成 | `Done` | `candidateGuards` / `decisionGuards` | static/dynamic shape 与 guard budget prune lit |
| cache 建模 | `Done` | `ShapeBucketCache` / `TuningResultCache` | key、negative cache、同 pass cache hit lit |
| `ScheduleDecisionSet` | `Done` | 输出多个候选决策 | runtimeTopK 可导出 |
| structured lowering | `Done` | 生成稳定 loop skeleton marker | report 与 IR marker lit |

计划文件：

- `docs/superpowers/plans/2026-05-08-ascend-mlir-v2-schedule-full-search.md`

### Phase 2 执行记录

| 计划任务 | 状态 | 提交 | 验证 |
|---|---|---|---|
| Task 0: Schedule full search 计划 | `Done` | `047d04f` | 计划覆盖开发、spec review、code review、xvm/docker 验证、跟踪更新 |
| Task 1: Shared Schedule Types and KernelPatternView | `Done` | `df41143` | `git diff --check` passed；spec review passed；code quality review approved；xvm `afir-opt` build passed；focused lit 3/3 passed；`check-afir` 36 discovered, 33 passed, 3 unsupported |
| Task 2: AxisCoalescer MVP | `Done` | `0d0d058` | TDD RED/GREEN completed；spec review passed；code quality review approved；xvm `afir-opt` build passed；focused lit 4/4 passed；`check-afir` 37 discovered, 34 passed, 3 unsupported |
| Task 3: ScheduleProblemBuilder MVP | `Done` | `3d7ba87` | TDD RED/GREEN completed；spec review passed；code quality review approved；xvm `afir-opt` build passed；focused lit 5/5 passed；`check-afir` 38 discovered, 35 passed, 3 unsupported |
| Task 4: TemplateRegistry MVP | `Done` | `36d0d34` | TDD RED/GREEN completed；spec review passed；code quality review approved；xvm clean `afir-opt` build passed；focused lit 6/6 passed；`check-afir` 39 discovered, 36 passed, 3 unsupported |
| Task 5: ScheduleSearch And compileTimeTopK | `Done` | `c0a0dae` | TDD RED/GREEN completed；spec review passed；code quality review approved；xvm clean `afir-opt` build passed；focused lit 7/7 passed；`check-afir` 40 discovered, 37 passed, 3 unsupported |
| Task 6: Guard Generation And Guard Budget | `Done` | `37226a3` | TDD RED/GREEN completed；spec review passed；code quality review approved；xvm clean `afir-opt` build passed；focused lit 8/8 passed；`check-afir` 41 discovered, 38 passed, 3 unsupported |
| Task 7: ScheduleDecisionSet Builder | `Done` | `7cf1348` | TDD RED/GREEN completed；spec review passed；code quality review approved；xvm clean `afir-opt` build passed；focused lit 9/9 passed；`check-afir` 42 discovered, 39 passed, 3 unsupported |
| Task 8: Schedule Cache Model | `Done` | `20c5a45` | spec review passed；code quality review approved；xvm clean `afir-opt` build passed；focused lit 10/10 passed；`check-afir` 43 discovered, 40 passed, 3 unsupported |
| Task 9: StructuredLoweringDriver MVP | `Done` | `a1ae3be` | TDD RED/GREEN completed；spec review passed；code quality re-review approved；xvm clean `afir-opt` build passed；focused lit 11/11 passed；`check-afir` 44 discovered, 41 passed, 3 unsupported |
| Task 10: Full Phase 2 Verification, Review, And Tracking | `Done` | 本文档提交 | final spec review approved；final code/test review approved；schedule focused 10/10 passed；pipeline smoke 1/1 passed；`check-afir` 44 discovered, 41 passed, 3 unsupported |

### Phase 2 Expert Review Follow-up

| 项 | 状态 | 处理结论 | 验证 |
|---|---|---|---|
| `KernelPattern` edge dedup key | `Done` | 移除重叠 bit-pack，改为 `KernelPatternEdgeKey` + `DenseSet` | 新增 C++ 单测覆盖旧碰撞样例；xvm `AscendKernelPatternTest` passed |
| `runtimeTopK` hardcode | `Done` | 新增 `--runtime-top-k`，接入 `ScheduleSearchOptions`，非空 decision set clamp 到 `[1, decisions.size()]` | lit 覆盖默认、`runtime-top-k=2`、`runtime-top-k=0`；focused lit passed |
| `AxisCoalescer` broadcast dead branch | `Done` | 删除不会命中的 `AxisKind::Broadcast` switch 分支，保留 `broadcastAxisMask` 处理 | `afir-opt` focused build/lit passed |
| `resolveTableFamily` asymmetric table | `Deferred` | 当前实现按 producer -> consumer 方向使用 MVP 表，不属于本轮 bugfix | 后续复合候选能力扩展时再处理 |
| `SubsumedCandidate` size equality | `No Action` | review 判定为误报：当前判断基于 union size，不会把等长不相交集合误判为包含 | 无代码改动 |
| CMake dialect deps | `No Action` | 当前直接使用的 func/linalg deps 已在 `AscendConversion` 中链接，无新增 arith/tensor/math C++ symbol 证据 | 无代码改动 |

Review / verification:

| 命令 | 结果 |
|---|---|
| `git diff --check` | passed |
| TDD RED: `ascend-schedule-decision-set.mlir` 增加 `runtime-top-k=0` 期望 | failed as expected：旧实现输出 `runtime_top_k = 0` |
| xvm focused build/test | `ninja -C build afir-opt AscendKernelPatternTest` passed；`AscendKernelPatternTest` 1/1 passed；`ascend-schedule-decision-set.mlir` 1/1 passed；`ctest -R AscendKernelPatternTest` passed |
| xvm `check-afir` | 45 discovered, 44 passed, 1 failed：`tools/examples/example-pipelines.mlir` 缺少既有 example `run_manifest.json`，与本轮改动无关 |
| xvm `check-unittests` after `source examples/env.sh` | 10 discovered, 8 passed, 2 failed：既有 runtime `MatmulTilingDispatcherTest` abort、`MixDirectTilingArtifactsTest` 缺 `libascend_hal.so`；新增 `AscendKernelPatternTest` passed |

### Phase 2 收口摘要

| 项 | 结果 |
|---|---|
| 代码范围 | 新增 `include/Conversion/Ascend/Schedule/*.h` 9 个、`lib/Conversion/Ascend/Schedule/*.cpp` 8 个；局部更新 `SchedulePass.cpp` 与 `lib/Conversion/Ascend/CMakeLists.txt` |
| 测试范围 | 当前 `test/Conversion/ascend-schedule-*.mlir` 共 10 个；Phase 2 新增 9 个 focused schedule lit |
| 行为覆盖 | pattern view、axis coalescing、problem builder、template registry、search、guards、decision set、cache、structured lowering marker |
| 最终验证 | schedule focused 10/10 passed；pipeline smoke 1/1 passed；`check-afir` 44 discovered, 41 passed, 3 unsupported |
| final review | V2-4 spec review approved；code/test review approved |
| 残余风险 | 当前 `AxisCoalescer` 已覆盖分类/report、broadcast/reduction/multi-primary；若后续要求真正把相邻轴折叠成更少 logical axes，需要在 Phase 3/后续 Schedule 增量中补更强 collapse 测试 |
| Phase 3 交接 | `StructuredLowering` 当前只写 `loop_skeleton_v0` marker，不做内存物化；Phase 3 从 `PlacementPlan`、`StaticMemoryPlan`、`MovementPlan`、`MemoryRealizationPlan` 接续 |

### Phase 2 验证记录

验证环境同 Phase 0：

- 主机：只做代码开发与 static check
- xvm/docker：编译与测试
- 同步路径：`/home/niu/code/Ascend-MLIR`
- 实际可用 LLVM build：`/home/niu/code/llvm-project/llvm/build`

Task 1 已执行：

```bash
git diff --check

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-verify/test/Conversion/ascend-schedule-mvp.mlir build-v2-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir build-v2-verify/test/Conversion/ascend-schedule-pattern-view.mlir'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target check-afir -j10'
```

结果：

| 命令 | 结果 |
|---|---|
| `git diff --check` | passed |
| `afir-opt` build + Phase 2 Task 1 focused lit | 3 discovered, 3 passed |
| `check-afir` | 36 discovered, 33 passed, 3 unsupported |

Task 2 已执行：

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-verify/test/Conversion/ascend-schedule-axis-coalescing.mlir build-v2-verify/test/Conversion/ascend-schedule-pattern-view.mlir build-v2-verify/test/Conversion/ascend-schedule-mvp.mlir build-v2-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target check-afir -j10'
```

结果：

| 命令 | 结果 |
|---|---|
| `git diff --check --cached` | passed |
| `afir-opt` build + Phase 2 Task 2 focused lit | 4 discovered, 4 passed |
| `check-afir` | 37 discovered, 34 passed, 3 unsupported |

Task 3 已执行：

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-verify/test/Conversion/ascend-schedule-problem.mlir build-v2-verify/test/Conversion/ascend-schedule-axis-coalescing.mlir build-v2-verify/test/Conversion/ascend-schedule-pattern-view.mlir build-v2-verify/test/Conversion/ascend-schedule-mvp.mlir build-v2-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-verify --target check-afir -j10'
```

结果：

| 命令 | 结果 |
|---|---|
| `git diff --check --cached` | passed |
| `afir-opt` build + Phase 2 Task 3 focused lit | 5 discovered, 5 passed |
| `check-afir` | 38 discovered, 35 passed, 3 unsupported |

Task 4 已执行：

```bash
ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake -S . -B build-v2-task4-verify -G Ninja -DLLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build -DAFIR_ENABLE_BINDING_PYTHON=false && cmake --build build-v2-task4-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-task4-verify/test/Conversion/ascend-schedule-template-registry.mlir build-v2-task4-verify/test/Conversion/ascend-schedule-problem.mlir build-v2-task4-verify/test/Conversion/ascend-schedule-axis-coalescing.mlir build-v2-task4-verify/test/Conversion/ascend-schedule-pattern-view.mlir build-v2-task4-verify/test/Conversion/ascend-schedule-mvp.mlir build-v2-task4-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-task4-verify --target check-afir -j10'
```

结果：

| 命令 | 结果 |
|---|---|
| `git diff --check --cached` | passed |
| clean `afir-opt` build + Phase 2 Task 4 focused lit | 6 discovered, 6 passed |
| `check-afir` | 39 discovered, 36 passed, 3 unsupported |

Task 5 已执行：

```bash
rsync -av --relative include/Conversion/AscendV2/Schedule/ScheduleTypes.h include/Conversion/AscendV2/Schedule/ScheduleSearch.h lib/Conversion/AscendV2/Schedule/ScheduleSearch.cpp lib/Conversion/AscendV2/Schedule/SchedulePass.cpp lib/Conversion/AscendV2/CMakeLists.txt test/Conversion/ascend-schedule-search.mlir xvm@orb:/home/niu/code/Ascend-MLIR/

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake -S . -B build-v2-task5-verify -G Ninja -DLLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build -DAFIR_ENABLE_BINDING_PYTHON=false && cmake --build build-v2-task5-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-task5-verify/test/Conversion/ascend-schedule-search.mlir build-v2-task5-verify/test/Conversion/ascend-schedule-template-registry.mlir build-v2-task5-verify/test/Conversion/ascend-schedule-problem.mlir build-v2-task5-verify/test/Conversion/ascend-schedule-axis-coalescing.mlir build-v2-task5-verify/test/Conversion/ascend-schedule-pattern-view.mlir build-v2-task5-verify/test/Conversion/ascend-schedule-mvp.mlir build-v2-task5-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-task5-verify --target check-afir -j10'
```

结果：

| 命令 | 结果 |
|---|---|
| `git diff --check` | passed |
| clean `afir-opt` build + Phase 2 Task 5 focused lit | 7 discovered, 7 passed |
| `check-afir` | 40 discovered, 37 passed, 3 unsupported |

Task 6 已执行：

```bash
rsync -av --relative include/Conversion/AscendV2/Schedule/ScheduleTypes.h include/Conversion/AscendV2/Schedule/ScheduleSearch.h lib/Conversion/AscendV2/Schedule/ScheduleSearch.cpp lib/Conversion/AscendV2/Schedule/SchedulePass.cpp test/Conversion/ascend-schedule-guards.mlir xvm@orb:/home/niu/code/Ascend-MLIR/

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake -S . -B build-v2-task6-verify -G Ninja -DLLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build -DAFIR_ENABLE_BINDING_PYTHON=false && cmake --build build-v2-task6-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-task6-verify/test/Conversion/ascend-schedule-guards.mlir build-v2-task6-verify/test/Conversion/ascend-schedule-search.mlir build-v2-task6-verify/test/Conversion/ascend-schedule-template-registry.mlir build-v2-task6-verify/test/Conversion/ascend-schedule-problem.mlir build-v2-task6-verify/test/Conversion/ascend-schedule-axis-coalescing.mlir build-v2-task6-verify/test/Conversion/ascend-schedule-pattern-view.mlir build-v2-task6-verify/test/Conversion/ascend-schedule-mvp.mlir build-v2-task6-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-task6-verify --target check-afir -j10'
```

结果：

| 命令 | 结果 |
|---|---|
| `git diff --check` | passed |
| clean `afir-opt` build + Phase 2 Task 6 focused lit | 8 discovered, 8 passed |
| `check-afir` | 41 discovered, 38 passed, 3 unsupported |

Task 7 已执行：

```bash
rsync -av --relative include/Conversion/AscendV2/Schedule/ScheduleTypes.h include/Conversion/AscendV2/Schedule/ScheduleDecision.h lib/Conversion/AscendV2/Schedule/ScheduleDecision.cpp lib/Conversion/AscendV2/Schedule/SchedulePass.cpp lib/Conversion/AscendV2/CMakeLists.txt test/Conversion/ascend-schedule-decision-set.mlir test/Conversion/ascend-schedule-pattern-view.mlir xvm@orb:/home/niu/code/Ascend-MLIR/

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake -S . -B build-v2-task7-verify -G Ninja -DLLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build -DAFIR_ENABLE_BINDING_PYTHON=false && cmake --build build-v2-task7-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-task7-verify/test/Conversion/ascend-schedule-decision-set.mlir build-v2-task7-verify/test/Conversion/ascend-schedule-guards.mlir build-v2-task7-verify/test/Conversion/ascend-schedule-search.mlir build-v2-task7-verify/test/Conversion/ascend-schedule-template-registry.mlir build-v2-task7-verify/test/Conversion/ascend-schedule-problem.mlir build-v2-task7-verify/test/Conversion/ascend-schedule-axis-coalescing.mlir build-v2-task7-verify/test/Conversion/ascend-schedule-pattern-view.mlir build-v2-task7-verify/test/Conversion/ascend-schedule-mvp.mlir build-v2-task7-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-task7-verify --target check-afir -j10'
```

结果：

| 命令 | 结果 |
|---|---|
| `git diff --check` | passed |
| clean `afir-opt` build + Phase 2 Task 7 focused lit | 9 discovered, 9 passed |
| `check-afir` | 42 discovered, 39 passed, 3 unsupported |

Task 8 已执行：

```bash
rsync -av --relative include/Conversion/AscendV2/Schedule/ScheduleTypes.h include/Conversion/AscendV2/Schedule/ScheduleCache.h lib/Conversion/AscendV2/Schedule/ScheduleCache.cpp lib/Conversion/AscendV2/Schedule/SchedulePass.cpp lib/Conversion/AscendV2/CMakeLists.txt test/Conversion/ascend-schedule-cache.mlir xvm@orb:/home/niu/code/Ascend-MLIR/

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake -S . -B build-v2-task8-verify -G Ninja -DLLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build -DAFIR_ENABLE_BINDING_PYTHON=false && cmake --build build-v2-task8-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-task8-verify/test/Conversion/ascend-schedule-cache.mlir build-v2-task8-verify/test/Conversion/ascend-schedule-decision-set.mlir build-v2-task8-verify/test/Conversion/ascend-schedule-guards.mlir build-v2-task8-verify/test/Conversion/ascend-schedule-search.mlir build-v2-task8-verify/test/Conversion/ascend-schedule-template-registry.mlir build-v2-task8-verify/test/Conversion/ascend-schedule-problem.mlir build-v2-task8-verify/test/Conversion/ascend-schedule-axis-coalescing.mlir build-v2-task8-verify/test/Conversion/ascend-schedule-pattern-view.mlir build-v2-task8-verify/test/Conversion/ascend-schedule-mvp.mlir build-v2-task8-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-task8-verify --target check-afir -j10'
```

结果：

| 命令 | 结果 |
|---|---|
| `git diff --check` | passed |
| clean `afir-opt` build + Phase 2 Task 8 focused lit | 10 discovered, 10 passed |
| `check-afir` | 43 discovered, 40 passed, 3 unsupported |

Task 9 已执行：

```bash
rsync -av --relative test/Conversion/ascend-schedule-structured-lowering.mlir xvm@orb:/home/niu/code/Ascend-MLIR/

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && set +e; build-v2-task8-verify/bin/afir-opt test/Conversion/ascend-schedule-structured-lowering.mlir --ascend-normalize --ascend-kernelize --ascend-schedule="dump-report=true debug-stage=schedule" 2>&1 | /home/niu/code/llvm-project/llvm/build/bin/FileCheck test/Conversion/ascend-schedule-structured-lowering.mlir; status=$?; echo TASK9_RED_FILECHECK_STATUS=$status; exit 0'

rsync -av --relative include/Conversion/AscendV2/Schedule/StructuredLoweringDriver.h lib/Conversion/AscendV2/Schedule/StructuredLoweringDriver.cpp lib/Conversion/AscendV2/Schedule/SchedulePass.cpp lib/Conversion/AscendV2/CMakeLists.txt test/Conversion/ascend-schedule-structured-lowering.mlir xvm@orb:/home/niu/code/Ascend-MLIR/

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake -S . -B build-v2-task9-verify -G Ninja -DLLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build -DAFIR_ENABLE_BINDING_PYTHON=false && cmake --build build-v2-task9-verify --target afir-opt -j10 && /home/niu/code/llvm-project/llvm/build/bin/llvm-lit -v build-v2-task9-verify/test/Conversion/ascend-schedule-structured-lowering.mlir build-v2-task9-verify/test/Conversion/ascend-schedule-cache.mlir build-v2-task9-verify/test/Conversion/ascend-schedule-decision-set.mlir build-v2-task9-verify/test/Conversion/ascend-schedule-guards.mlir build-v2-task9-verify/test/Conversion/ascend-schedule-search.mlir build-v2-task9-verify/test/Conversion/ascend-schedule-template-registry.mlir build-v2-task9-verify/test/Conversion/ascend-schedule-problem.mlir build-v2-task9-verify/test/Conversion/ascend-schedule-axis-coalescing.mlir build-v2-task9-verify/test/Conversion/ascend-schedule-pattern-view.mlir build-v2-task9-verify/test/Conversion/ascend-schedule-mvp.mlir build-v2-task9-verify/test/Conversion/ascend-v2-pipeline-mvp.mlir'

ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && cmake --build build-v2-task9-verify --target check-afir -j10'
```

结果：

| 命令 | 结果 |
|---|---|
| RED: Task 8 binary + new structured lowering lit | failed as expected；`TASK9_RED_FILECHECK_STATUS=1` |
| `git diff --check --cached` | passed |
| clean `afir-opt` build + Phase 2 Task 9 focused lit | 11 discovered, 11 passed |
| `check-afir` | 44 discovered, 41 passed, 3 unsupported |

## Phase 3：Realize plan objects

目标：建立第四层 `--ascend-realize` pass、稳定 plan object 和 MVP debug report，后续再接入真实 bufferization、placement、static memory、movement 和 materialization。

| 任务 | 状态 | 说明 | 验收 |
|---|---|---|---|
| Task 0: Realize MVP 计划 | `Done` | 拆分第一批 Realize 实现范围 | 计划文件已提交 |
| Task 1: Realize pass skeleton and plan reports | `Done` | `--ascend-realize`、`debug-stage=realize`、MVP plan objects | focused Realize lit 4/4 passed；pipeline smoke passed；review follow-up passed |
| `BufferizationDriver` facts MVP | `Done` | 新增只读 `BufferizationDriver`，按 kernel 收集 tensor input / output / temporary facts；完整 One-Shot Bufferize 接入后续继续推进 | focused Realize facts lit passed；Ascend Conversion lit 23/23 passed |
| `PlacementPlan` GM-default MVP | `Done` | 新增只读 `PlacementPlanner`，将已收集 buffer facts 保守映射到 `GM`，并报告 deferred local count；target-aware placement 后续继续推进 | focused placement lit passed；Ascend Conversion lit passed |
| `StaticMemoryPlan` read-only MVP | `Done` | 新增 `StaticMemoryPlanner`，报告 empty workspace 与 tracked place count；真实 workspace layout / lifetime 后续推进 | focused completion lit passed；planner unit passed |
| `MovementPlan` read-only MVP | `Done` | 新增 `MovementPlanner`，GM-only noop movement，并校验 static memory MVP 不变量；显式 data movement 后续推进 | focused completion lit passed；planner unit passed |
| `MemoryRealizationPlan` read-only MVP | `Done` | 新增 `MemoryRealizationDriver`，冻结 read-only plan，报告 `plan_identity_only` 验证范围；真实 materialization 后续推进 | focused completion lit passed；planner unit passed |

### Phase 3 验证记录

| 命令 | 结果 |
|---|---|
| TDD RED: `ascend-realize-mvp.mlir` / `ascend-realize-rejects-unscheduled.mlir` | failed as expected：`--ascend-realize` 未注册 |
| TDD RED: `ascend-realize-rejects-partial-attrs.mlir` | failed as expected：半标记 op 被静默忽略 |
| TDD RED: `ascend-realize-rejects-inconsistent-attrs.mlir` | failed as expected：同 kernel 不一致 schedule attrs 被接受 |
| xvm focused build/test | `ninja -C build afir-opt` passed；Realize focused lit 4/4 passed；pipeline smoke passed |
| spec review | passed：未越界实现真实 bufferization / placement / movement / materialization |
| code quality review | approved after re-review：同 kernel schedule attr 一致性已补充 |
| xvm `check-afir` | not completed：broader run 长时间停在既有 `externals/pyasc/.../Translation.cpp` 编译单元，已中断；本轮以 focused Realize + pipeline smoke 作为验证依据 |
| TDD RED: Realize bufferization facts | failed as expected：旧实现仍输出 `mode = "gm_only"` / `buffer_values = 0`；mixed-use 回归中旧逻辑会把 `%mid` 同时计为 output 和 temporary |
| TDD GREEN: Realize bufferization facts | `ninja -C build afir-opt` passed；`llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir build/test/Conversion/ascend-realize-bufferization-facts.mlir` 2/2 passed |
| spec review: Bufferization facts MVP | passed：实现符合计划，只做只读 tensor fact collection，无 One-Shot Bufferize / IR mutation / placement / movement 越界 |
| code quality review: Bufferization facts MVP | approved after re-review：角色互斥计数修复，dead result 不再误计为 output |
| xvm focused build/unit/ctest | `ninja -C build afir-opt AscendCommonAttributesTest AscendKernelPatternTest` passed；`AscendCommonAttributesTest` 1/1 passed；`AscendKernelPatternTest` 1/1 passed；`ctest -R "Ascend(CommonAttributes\|KernelPattern)Test"` 2/2 passed |
| xvm Conversion lit | `llvm-lit -v build/test/Conversion` 30/30 passed；`llvm-lit -v build/test/Conversion --filter="ascend-"` 23/23 passed |
| code naming guard | `test/tools/check_ascend_no_v2_code_naming.sh` passed |
| TDD RED: Realize placement plan | failed as expected：旧实现缺少 `mode = "gm_default"`，且仍输出 `selected_places = 0` |
| TDD GREEN: Realize placement plan | `ninja -C build afir-opt` passed；`llvm-lit -v build/test/Conversion/ascend-realize-mvp.mlir build/test/Conversion/ascend-realize-placement-plan.mlir` 2/2 passed |
| spec review: PlacementPlan GM-default MVP | passed：实现符合计划，只做只读 GM-default placement counters，无 TargetMemoryModel / IR mutation / memory materialization 越界 |
| code quality review: PlacementPlan GM-default MVP | approved：GM-default 计数不变量一致；已按建议为 split-input LIT 增加 report anchors |
| TDD RED: Phase 3 completion MVP | failed as expected：旧实现缺少 `StaticMemoryPlan` / `MovementPlan` / `MemoryRealizationPlan` 新 report 字段；新增 planner unit test 在生产修复前失败 |
| TDD GREEN: Phase 3 completion MVP | `ninja -C build afir-opt AscendRealizePlannerTest` passed；`AscendRealizePlannerTest` 8/8 passed；focused completion lit 2/2 passed |
| spec review: Phase 3 completion MVP | passed：五个 Realize plan objects 均有 builder/driver 和 report；无 One-Shot Bufferize / TargetMemoryModel / IR mutation / memory materialization 越界 |
| code quality review: Phase 3 completion MVP | approved after re-review：`verification_scope = "plan_identity_only"`；movement/static/realization plan id 与 MVP shape 不变量均有 unit 覆盖 |

### Phase 3 Expert Review Follow-up

| 项 | 状态 | 处理结论 | 验证 |
|---|---|---|---|
| `AxisKind::Broadcast` enum 残留 | `Done` | 删除 enum value，broadcast 继续通过 `broadcastAxes` metadata 表达 | `ascend-schedule-axis-coalescing.mlir` / `ascend-schedule-problem.mlir` passed |
| `AscendRealizePass` phantom dependent dialects | `Done` | 移除 `FuncDialect` / `LinalgDialect` / `MemRefDialect` 依赖声明和死 include | `ninja -C build afir-opt` passed；Realize focused lit passed |
| `RealizeTypes.h` dead include | `Done` | 删除未使用 `SmallVector.h` / `StringRef.h` / `LLVM.h` include | `ninja -C build AscendCommonAttributesTest` passed |
| Realize `MemoryPlace` 与 TargetProfile 命名边界 | `Done` | 增加注释说明 Realize placement enum 与 target hardware memory hierarchy 不同 | `afir-opt` build passed |
| unscheduled Realize 测试命名 | `Done` | `ascend-realize-requires-schedule.mlir` 重命名为 `ascend-realize-rejects-unscheduled.mlir` | renamed lit passed |
| Ascend shared attributes | `Done` | 新增 `Conversion/Ascend/Common/Attributes.h`，Kernelize / Schedule / Realize 使用同源常量 | 新增 `AscendCommonAttributesTest` passed |
| 代码命名去版本化 | `Done` | 源码目录、namespace、CMake target、IR attrs、测试名迁移为版本无关 `Ascend` 命名；方案/文档版本名保留 | guard、xvm build、unit、ctest、Ascend lit passed |

### Phase 3/4 Expert Review Follow-up Round 2

| 项 | 状态 | 处理结论 | 验证 |
|---|---|---|---|
| `buildFallbackPattern` 丢弃 closure 计算结果 | `Done` | 删除无效 `computeCandidateClosure` 调用；candidate / merged / horizontal candidate 的真实 closure 计算保留 | `AscendKernelPatternTest` passed；Ascend Conversion lit passed |
| `MovementPlanner` 硬耦合 `empty_workspace` | `Done` | 仅保留 kernel id 与 tracked place 不变量；不再拒绝 future static-memory mode、workspace slots、known peak usage | `AscendRealizePlannerTest` 覆盖 future static plan cases |
| `AscendRealizePass` 空模块诊断 | `Done` | empty/no scheduled op 路径直接 emit 明确 module diagnostic，避免依赖通用 fallback 错误 | `ascend-realize-rejects-unscheduled.mlir` passed |
| `ScheduleContract::templateFamilies` 生命周期 | `Done` | 改为 owning `SmallVector<std::string, 2>`，merge / report helpers 同步使用 owning strings | xvm `afir-opt` build、Kernelize/Realize focused lit passed |

Review / verification:

| 命令 | 结果 |
|---|---|
| TDD RED: `AscendCommonAttributesTest` | failed as expected：`Conversion/Ascend/Common/Attributes.h` 不存在 |
| TDD RED: code naming guard | failed as expected：旧代码中存在 `AscendV2` / `ascend.v2` / `ascend-v2-pipeline` |
| TDD RED: guard fallback probe | failed as expected：无 `rg` 环境下，临时 `AscendV2Probe` 同样被 `grep/find` fallback 捕获 |
| xvm build | `ninja -C build afir-opt AscendCommonAttributesTest AscendKernelPatternTest` passed |
| xvm unit tests | `AscendCommonAttributesTest` 1/1 passed；`AscendKernelPatternTest` 1/1 passed；`ctest -R "Ascend(CommonAttributes\|KernelPattern)Test"` 2/2 passed |
| xvm focused lit | Realize / Schedule / pipeline smoke 7/7 passed |
| xvm Ascend Conversion lit | `ascend-*.mlir` 22/22 passed |
| code naming guard | host `rg` path passed；host no-`rg` fallback passed；xvm `command -v rg` returned missing and guard passed through `find`/`grep` fallback |

## Phase 4：Target Model 完整化

目标：把 Target MVP 升级为 V2-8 完整跨层 target 查询模型。

| 任务 | 状态 | 说明 | 验收 |
|---|---|---|---|
| `TargetMemoryModel` | `Done` | logical places、capacity、alignment、visibility、direct path graph；multi-hop routing 延后到 TargetRouting，intrinsic-backed path validation 由 `TargetModelVerifier` MVP 覆盖 | `AscendTargetMemoryModelTest` + target profile lit |
| `TargetIntrinsicModel MVP` | `Done` | query-only intrinsic table、unit map、movement map、compute map、dtype-pattern token lookup；path constraints / memory-model integration 后续继续推进，path-kind 闭合由 `TargetModelVerifier` MVP 覆盖 | `AscendTargetIntrinsicModelTest` + target profile lit |
| `TargetCostModel MVP` | `Done` | query-only memory-rate lookup、direct-path cost lookup、transfer-cycle estimate；Schedule / Realize consumption 后续继续推进 | `AscendTargetCostModelTest` + target profile lit |
| `TargetModelVerifier MVP` | `Done` | profile/memory/intrinsic/cost 闭合检查；`QueueTransfer` 在当前 MVP 中作为执行单元 handoff，不要求 movement intrinsic | `AscendTargetModelVerifierTest` |
| 多 SoC 支持 | `Planned` | 910B2 之外的 ini | 参数化 lit 或 unit tests |

## Phase 3B：Realize Materialization 增强

目标：在 Phase 4 target 查询模型补齐后，把 Phase 3 的 read-only plan objects 升级为真实内存实现，产出 Phase 5 可消费的 materialized IR。

| 任务 | 状态 | 说明 | 依赖 |
|---|---|---|---|
| One-Shot Bufferize opt-in MVP | `Done` | `ascend-realize` 新增 `materialization-mode=one-shot-bufferize`，在显式开启时调用 upstream One-Shot Bufferize 将 tensor IR 改写为 memref IR；默认 `plan-only` 保持原 read-only plan/report 行为 | `ascend-realize-one-shot-bufferize.mlir` |
| target-aware placement plan MVP | `Done` | 新增 `placement-mode=target-aware`，加载 CANN target profile 并构建 `TargetMemoryModel`；输入/输出保守保持 `GM`，vector temporary 只有在 `VECIN` 合法、有容量且存在 `GM -> VECIN` direct path 时计为 on-chip place；默认 `gm-default` 行为保持不变。`TargetCostModel` 排序、value-level place map、`memory_space` 写入和 copy/materialization 后续推进 | `ascend-realize-target-aware-placement.mlir`；`AscendRealizePlannerTest` |
| workspace layout / lifetime MVP | `Done` | `StaticMemoryPlanner` 对 on-chip placement 生成 `workspace_layout`：vector temporary 有 live interval / workspace slot，slot place 为 `VECIN`；静态 tensor byte facts 与稳定 value-level facts 已从 `BufferizationDriver` 进入 `BufferizedKernelIR`，可报告 local/workspace/peak byte usage；target-aware 模式已用 `TargetMemoryModel` 校验 `VECIN` peak byte capacity；非重叠 vector temporary 可复用物理 slot offset，peak/workspace bytes 按物理复用后容量回写 | `ascend-realize-workspace-layout.mlir`；`AscendRealizePlannerTest` |
| explicit data movement plan MVP | `Done` | `MovementPlanner` 对 on-chip workspace 场景生成 `movement_planning`：记录 value-level movement step、movement demand、workspace reuse candidate；target-aware 模式用 `TargetMemoryModel` 选择 `GM -> VECIN` direct path，缺 direct path 时 fail-closed 保持 deferred；selected direct input movement 已可物化为 local alloc + copy，静态同形同 block 输入可合并到一个 workspace base；跨形状 selected movement 用 flat workspace + `memref.reinterpret_cast` view 打包，并在各自 consumer 前插入 copy；producer/consumer 经过 `tensor.extract_slice` / `tensor.cast` / `tensor.expand_shape` / `tensor.collapse_shape` view-chain 时，materialize 层会在目标 memory space 上重建 view 后改写 consumer | `ascend-realize-data-movement-plan.mlir`；`ascend-realize-cross-shape-workspace-packing.mlir`；`ascend-realize-view-chain-movement.mlir`；`ascend-realize-reshape-view-chain-movement.mlir`；`AscendRealizePlannerTest` |
| memory-space annotation materialization MVP | `Done` | 新增 `materialization-mode=memory-space-annotate`：先运行 One-Shot Bufferize，再由 `MemoryRealizationDriver::materialize(module, bundles, MemorySpaceAnnotate)` 把已证明的 vector temporary `memref.alloc` 标为 `VECCALC` memory space；report 按 kernel 记录 `memory_space_annotations`；跨 kernel temporary 保守不标注；movement step 与 Phase5 bridge 的 alloc/copy 物化由同一 driver 汇总计数 | `ascend-realize-memory-space-annotate.mlir`；`AscendRealizePlannerTest` |
| Phase 5 bridge for ordinary vector / reduction output | `Done` | `memory-space-annotate` 模式下由同一个 `MemoryRealizationDriver::materialize` 入口为 Phase 5 支持的最终 vector output 与保守 reduction output 生成 `VECOUT` alloc，并插入 `VECOUT -> GM` epilogue `memref.copy`；保留 GM result 作为 ABI/return buffer，alloc/copy 计数从实际 IR mutation 回写到 `MemoryRealizationPlan` | `ascend-full-pipeline-ordinary-smoke.mlir`；`ascend-full-pipeline-broadcast-add-reduce.mlir`；`AscendRealizePlannerTest.MemoryRealizationMaterializeMutatesIRAndPlan` |
| selected value-level workspace/copy materialization enhancement | `Done` | selected movement 的同形静态 workspace alloc/subview、跨形状 flat workspace packing、`memref.subview` / `memref.cast` / `memref.expand_shape` / `memref.collapse_shape` consumer view-chain 重写已落地；静态 planner 已支持非重叠 vector temporary slot reuse；更广义 dynamic view-chain copy materialization 继续作为后续增强 | `ascend-realize-cross-shape-workspace-packing.mlir`；`ascend-realize-view-chain-movement.mlir`；`ascend-realize-reshape-view-chain-movement.mlir` |
| Realize commercial blocker closure | `Done` | `BufferizationDriver` 的 vector temporary 识别已从 legacy `ascend.op_role` 扩展到 `ascend.op_roles` array；target-aware static workspace 从 `VECCALC` 改为可路由的 `VECIN`，production planner 链路可选中并物化 `GM -> VECIN`；cube bridge 不再 walk `linalg::MatmulOp` 专名，而是走 `LinalgBodyClassifier` 后端能力分类，覆盖 `matmul` / `batch_matmul` | `AscendRealizePlannerTest.BufferizationDriverReadsOpRolesArrayForVectorTemporary`；`TargetAwareMovementPlannerSelectsPathFromProductionWorkspacePlan`；`Phase5CubeBridgeSupportsBatchMatmul` |

### Phase 3B 验证记录

| 命令 | 结果 |
|---|---|
| TDD RED: `ascend-realize-memory-space-annotate.mlir` multi-kernel count | failed as expected：旧实现把 module-wide annotation count 写入每个 kernel，`kernel_1` 期望 0 实际为 1 |
| xvm focused build/test | `ninja -C build afir-opt AscendRealizePlannerTest` passed；`ctest -R "AscendRealizePlannerTest"` 1/1 passed |
| xvm focused lit | Realize memory-space / one-shot / movement / workspace / MVP / completion 6/6 passed |
| spec review | passed：`memory-space-annotate` 先 One-Shot Bufferize，只标注已证明 vector temporary，不插入 workspace/subview/copy |
| code quality review | approved after re-review：per-kernel count、cross-kernel temporary negative case、alloc dynamic sizes / symbol operands / alignment 均已覆盖 |
| xvm unit regression | `ctest -R "Ascend(CommonAttributes\|KernelPattern\|RealizePlanner\|TargetMemoryModel\|TargetIntrinsicModel\|TargetCostModel\|TargetModelVerifier)Test"` 7/7 passed |
| xvm Ascend Conversion lit | `llvm-lit -v build/test/Conversion --filter="ascend-"` 30/30 passed |
| xvm target profile lit | `llvm-lit -v build/test/Target/ascend-target-profile.mlir` 1/1 passed |
| code naming guard | host passed；xvm passed |
| whitespace | `git diff --check -- . ':!AGENTS.md'` passed |

## Phase 5：Translate / Runtime Artifact

目标：对齐 V2-6 / V2-9，把上游决策转成 backend/runtime 可消费产物。

| 任务 | 状态 | 说明 | 验收 |
|---|---|---|---|
| `ComputeLoweringDriver` 对齐 | `Done` | 新增 `--ascend-compute-lower` 正式入口，复用现有 LinalgToAscendC lowering，并通过 support matrix 对 unsupported op / movement path fail-closed；external `func.func` declaration 保守 no-op | `ascend-compute-lower.mlir`；`ascend-compute-lower-unsupported*.mlir`；`ascend-compute-lower-external.mlir`；`AscendBackendSupportMatrixTest` |
| ABI lowering 对齐 | `Done` | 新增 `--ascend-parallelize`、`--ascend-prepare-for-emit`、`--ascend-canonicalize-cann-signature` 正式入口，旧原型入口保留 | `ascend-backend-abi-wrappers.mlir` |
| `HostTilingEmitter` | `Done` | `afir-translate --host-tiling-out` 输出静态 shape / 单 kernel C ABI source | `cann-translate-runtime-artifacts.mlir` |
| `RuntimeManifestBuilder` | `Done` | `afir-translate --runtime-manifest-out` 输出静态 shape manifest；单 kernel root 字段保持兼容，多 global kernel 生成 `kernel_entries` 与 `kernelGraph.nodes/edges`，并校验未知端点 / 非 DAG 边 | `cann-translate-runtime-artifacts.mlir`；`cann-translate-runtime-artifacts-multi.mlir`；`cann-translate-runtime-artifacts-unsupported.mlir` |
| `tiling_space.json` export | `Done` | `--tiling-space-out` 升级为 `schema_version = "2.0"`，包含 workspace/block dim/schema fields；兼容旧多 global module 选择首个 global kernel 的行为 | `cann-translate-runtime-artifacts.mlir`；`cann-translate-runtime-artifacts-unsupported.mlir` |
| transformer dynamic smoke | `Done` | `examples/transformer/transformer_dynamic.mlir` 已纳入 Phase 5 验收 smoke；Kernelize 按 `LayoutTransform` 语义接收高 rank transpose；Schedule 已用 `ascend.schedule.kernel_metadata` 按 kernel 持久化 func 级 metadata，避免 selected tile / tail plan 静默覆盖；ComputeLower 对 GM copy / transpose / matmul / batch_matmul / scalar generic fallback 闭环，完整 transformer prefix full codegen 通过 | `ascend-phase5-transformer-dynamic-smoke.mlir`；`ascend-kernelize-transpose.mlir`；`ascend-compute-lower-transpose*.mlir`；`ascend-compute-lower-*-gm.mlir`；`ascend-schedule-multi-kernel-func-metadata.mlir` |
| pre-lowered ordinary example acceptance | `Done` | 旧式前处理路径已保留为 `examples/relu-broadcast-transpose/run-legacy.sh`；默认 `run.sh` 已切到 `run-mainline.sh`，生成 `phase5_tiling_space.json`、`phase5_runtime_manifest.json`、`host_tiling.cpp` 后跑通 runtime-session sim 验证 | xvm mainline run passed：`session.result=success`、`session.validation=pass` |
| full Phase 0 -> Phase 5 ordinary positive smoke | `Done` | 最小普通 tensor/linalg 用例通过 Normalize / Kernelize / Schedule / Realize / ComputeLower / ABI wrappers；Realize bridge 物化 Phase 5 可消费的 `VECOUT` output 和 GM epilogue copy | `ascend-full-pipeline-ordinary-smoke.mlir`；bridge report |
| broadcast-add-reduce mainline E2E | `Done` | `examples/broadcast-add-reduce/run-mainline.sh` 从 `step0_input.mlir` 出发，逐步生成 fused / normalized / kernelized / scheduled / realized / AscendC / CANN ABI IR、`step10_kernel.cpp`、Phase 5 artifacts，并通过 runtime-session sim；默认 `M=640,N=15000,BLOCK_DIM=20` | xvm `run-mainline.sh --log` passed：`session.result=success`、`session.validation=pass` |
| relu-broadcast-transpose mainline E2E | `Done` | `examples/relu-broadcast-transpose/run-mainline.sh` 从 `step0_input.mlir` 出发，逐步生成 fused / normalized / kernelized / scheduled / realized / AscendC / CANN ABI IR、`step10_kernel.cpp`、Phase 5 artifacts，并通过 runtime-session sim；覆盖 `(d0,d1)->(d1,0)` 常量投影 broadcast/transpose、relu/max 与 add 融合 | `ascend-full-pipeline-relu-broadcast-transpose.mlir`；xvm `run-mainline.sh --log` passed：`session.result=success`、`session.validation=pass` |
| Phase5 body classifier single source | `Done` | 新增 `Backend/LinalgBodyClassifier`，把 Realize bridge 与 ComputeLower 原本各自维护的 linalg body 支持判断收敛到同一处；当前由 `AscendBackendSupportMatrix` 查询 compute kind 支持，保留 rank2 swap transpose backend 闭环 | `AscendLinalgBodyClassifierTest`；`ascend-compute-lower*` / `ascend-realize*` / `ascend-full-pipeline*` focused lit |
| add-broadcast-concat mainline E2E | `Done` | `examples/add-broadcast-concat/run-mainline.sh` 从 `step0_input.mlir` 出发，覆盖两个独立 broadcast elementwise producer + `tensor.concat dim(0)`；Realize 将 concat bufferize 产生的 GM->GM copy 收敛为 `VECOUT -> GM subview`，ComputeLower selected-tile loop 可插入到 writeback 前并保持支配关系 | `ascend-full-pipeline-add-broadcast-concat.mlir`；xvm `run.sh --log` passed：`session.result=success`、`session.validation=pass` |
| gather-elementwise-fusion mainline E2E | `Done` | `examples/gather-elementwise-fusion/run-mainline.sh` 从 `step0_input.mlir` 出发，覆盖 mark/fuse/normalize/kernelize/schedule/realize/compute/parallelize/emit 到 CANN kernel；支持 index-select gather + relu + bias add 融合；默认 `run.sh` 已委托新主线，旧 transform-interpreter 路径保留为 `run-legacy.sh`；已移除 N/K 16 对齐脚本前置拒绝，非 32B GM↔Local `DataCopyL2Op` tail 走 scalar fallback，aligned path 保留 `AscendC::DataCopy` | `ascend-full-pipeline-gather-elementwise-fusion.mlir`；`cann-translate-gather.mlir`；xvm default run、N/K tail matrix 与 runtime tool passed：`session.result=success`、`session.validation=pass` |
| axis schedule contract / coalescing 泛化 | `Done` | Schedule 侧新增 axis contract 数据模型、AxisCoalescer 推导 `axisScheduleConstraints` / `axisCoalescingHints` 并输出 debug report；`ScheduleProblem` 消费 axis contract；tail-policy-aware guard generation 已接入；role-driven reduction/vector tile search 选择 `[32,N]` bounded parallel tile；selected tile shape / tail policy metadata 持久化到 scheduled ops、父 `func.func` 和 runtime manifest；Phase 5 对 selected tile materialization、32B runtime buffer 对齐、rows<16 broadcast tail fallback 已闭环 | `docs/superpowers/plans/2026-05-12-axis-schedule-contract-coalescing.md`；xvm 54 个 Ascend/CANN source LIT 54/54 passed；sim matrix `M=65/70/72,N=128`、`M=70,N=123`、`M=128,N=123`、默认 `M=640,N=15000` 全部 validation passed；static checks passed |
| vector template 泛化 | `Done` | `TemplateRegistry` 将 `vector_static_1d` / `vector_static_2d` 合并为 `vector_generic`，template 数量不随 rank 增长；tiling 仍由 `ScheduleSearch` 基于实际 shape / axis contract 生成 `ScheduleInstance` | `ascend-schedule-template-registry.mlir` 新增 rank3 vector 覆盖；schedule cache/search/pattern-view/mvp LIT 期望同步 |
| all-parallel vector bounded tile | `Done` | `ScheduleSearch` 对 `vector_generic` 生成 role-driven bounded parallel tile，rank2 dynamic/static 大 shape 优先选择 `[32,N]`；`ascend-compute-lower` 对 rank2 identity 与 rank1 row/col projection all-parallel selected tile 物化 `scf.for` / subview / tiled writeback，随后 `--ascend-parallelize` 将外层 tile loop 映射为 `ascendc.get_block_idx`；broadcast-transpose 等 unsupported map 保守回退旧整块 lowering | `ascend-schedule-search.mlir`；`ascend-schedule-vector-bounded-tile.mlir`；`ascend-compute-lower-selected-all-parallel-tile-materializes-loop.mlir`；`ascend-compute-lower-selected-all-parallel-tile-fallback.mlir`；`ascend-full-pipeline-rank2-elementwise-add.mlir` |
| Phase 5C+ metadata bridge | `Done` | `ScheduleDecision` 不再复制 instance guard 字段；StructuredLowering 持久化 guard/tail marker attrs 与 target tile policy；target tile 默认策略集中到 `TargetTilePolicy` hook；runtime manifest 保留旧字段并增加 `kernel_entries[0].tilingParams` | `check_ascend_schedule_decision_contract.sh`；`ascend-normalize.mlir`；`ascend-schedule-structured-lowering.mlir`；`ascend-schedule-search.mlir`；`cann-translate-runtime-artifacts.mlir` |
| multi-kernel DAG runtime manifest | `Done` | `RuntimeManifestBuilder` 可为多个 `ascendc.global` kernel 生成 `kernel_entries`，并从 `ascend.kernel_graph.edges` 生成 `kernelGraph`；host tiling 仍保持单 kernel MVP 边界 | `cann-translate-runtime-artifacts-multi.mlir`；`cann-translate-runtime-artifacts-unsupported.mlir` |
| target-aware schedule tile policy | `Done` | `--ascend-schedule='target-tile-policy=target-aware cann-root=... soc=...'` 加载 CANN target profile，构建并验证 memory / intrinsic / cost model，再按 UB 容量和 GM↔VEC path cost availability 派生 `target_ub_<tile>` | `ascend-schedule-target-tile-policy.mlir` |
| transformer_dynamic transpose / GM fallback closure | `Done` | Kernelize 对 `LayoutTransform` 不再按 rank/permutation 判 supported；backend lowering 通过统一 transpose plan lower on-chip rank2 transpose，并对 GM 输出 transpose / matmul / batch_matmul / scalar generic 走 fail-closed scalar loop fallback；`run-mainline.sh` 当前 full codegen 通过 | `ascend-kernelize-transpose.mlir`；`ascend-compute-lower-transpose*.mlir`；`ascend-compute-lower-matmul-gm.mlir`；`ascend-compute-lower-batch-matmul-gm.mlir`；`ascend-compute-lower-generic-*-scalar-loop.mlir`；`ascend-phase5-transformer-dynamic-smoke.mlir` |
| split-relu-brc-add-mul mainline E2E | `Done` | `examples/split-relu-brc-add-mul/run-mainline.sh` 从 `step0_input.mlir` 出发，覆盖 split 两路 relu + row/col broadcast + mul + concat；selected all-parallel tile 支持 `(d0)`/`(d1)` rank1 projection，允许共享只读 input root 的两路 producer 插到各自 writeback 前；默认 `run.sh` 已委托新主线，旧 transform-interpreter 路径保留为 `run-legacy.sh` | xvm default run passed；`run_simbackend_examples.sh split-relu-brc-add-mul` passed；5-example SimBackend smoke passed |
| matmul-add-leakyrelu mainline E2E | `Done` | `examples/matmul-add-leakyrelu/run-mainline.sh` 从 `step0_input.mlir` 出发，覆盖 dynamic `linalg.matmul` + bias add + leaky_relu；Schedule 为 cube 角色生成覆盖 `[M,N,K]` logical axes 的 tile/tail plan，并对 dynamic M 生成 bounded M-axis tile；Realize 物化 GM->A1->A2、GM->B1->B2、CO1->VECIN cube/vector bridge，并标注 `AiCore.Cube` / `AiCore.Vector` 供 Phase 5 mix lowering 消费；cube bridge 仅在 matmul 输出后续用途均为同 block/same-kernel/supported vector DPS input 时触发，fanout 到 return/copy/non-vector 时保守拒绝；默认 `run.sh` 已委托新主线，旧 transform-interpreter 路径保留为 `run-legacy.sh` | `ascend-realize-cube-phase5-bridge.mlir`；`ascend-full-pipeline-matmul-add-leakyrelu.mlir`；`ascend-schedule-search.mlir` K-axis regression；xvm default run、altshape、repeat/output reuse、6-example SimBackend smoke passed |
| supported example default entry migration | `Done` | `broadcast-add-reduce`、`relu-broadcast-transpose`、`add-broadcast-concat`、`gather-elementwise-fusion`、`split-relu-brc-add-mul`、`matmul-add-leakyrelu` 的默认 `run.sh` 已委托新主线 `run-mainline.sh`；旧 transform-interpreter 入口保留为 `run-legacy.sh` 便于对比 | 6-example SimBackend smoke passed |
| runtime graph / queue lifetime hardening | `Done` | 新增 `check_ascend_queue_lifetime.py` 并接入 full-pipeline / example suite；generic all-parallel 与 named elementwise lowering 在释放 dequeued queue tensors 前插入 `pipe_all` barrier；`CannRuntimeArtifacts` 将 `ascend.kernel_graph.edges` 内部 kernel id 映射到最终 CANN entry id；新增 `two-kernel-rank-mix-dag`、`three-kernel-dag`、`rmsnorm-reduction-core`，覆盖 rank-mixed task-output、3 kernel chain、RMSNorm square->reduce->scale DAG；N=64 aligned DataCopy fast path 已用 runtime-session 数值回归覆盖，不再用 lane0 scalar patch | `check_ascend_queue_lifetime.py`；`ascend-queue-lifetime-full-pipeline.mlir`；`ascend-full-pipeline-rank2-elementwise-add.mlir`；`cann-translate-runtime-artifacts-multi.mlir`；`example-pipelines.mlir` 10/10 passed |

### Phase 5 验证记录

| 命令 | 结果 |
|---|---|
| host static checks | `git diff --check -- . ':!AGENTS.md'` passed；`test/tools/check_ascend_no_v2_code_naming.sh` passed |
| xvm Phase 5 focused verification | `AscendBackendSupportMatrixTest` 1/1 passed；Phase 5 focused LIT 8/8 passed |
| xvm Task 4 focused translation | `ninja -C build afir-translate` passed；Target runtime artifact lit 3/3 passed |
| xvm runtime artifact write failure | `--host-tiling-out=/dev/full` 返回 status 1，并输出 `failed to flush runtime artifact '/dev/full': No space left on device` MLIR diagnostic；无 LLVM fatal |
| xvm Phase 5 final regression | xvm code naming guard passed；`ctest -R "Ascend(...)"` 8/8 passed；`llvm-lit -v build/test/Conversion --filter="ascend-"` 36/36 passed；`llvm-lit -v build/test/Target` 11/11 passed |
| Task 4 code review | approved：runtime artifact 文件错误处理、`--tiling-space-out` 多 global 兼容、single-kernel manifest/host tiling 校验均通过 |
| final review | spec review passed；code quality re-review approved：`--ascend-compute-lower` external declaration crash 改为 no-op；runtime manifest `shapeArgOrder.abiPosition` 改为 dense `shape_args` ABI 顺序 |
| xvm transformer dynamic smoke | `ascend-phase5-transformer-dynamic-smoke.mlir` 1/1 passed；当前完整 transformer 图明确报 unsupported |
| xvm ordinary Phase 5 example | `examples/relu-broadcast-transpose` copied to `/tmp` and run with build tools passed；new Phase 5 entries used for compute lower / ABI lowering / CANN signature；runtime-session sim reported `session.result=success` and `session.validation=pass` |
| xvm full-pipeline ordinary smoke | `ascend-full-pipeline-ordinary-smoke.mlir` 1/1 passed；one-shot/memory-space annotate 链路进入 `ascend-compute-lower` 后生成 `ascendc.add_l2` 和 `ascendc.data_copy_l2`；target-aware 链路继续通过 ABI wrappers |
| xvm Phase 5C Ascend Conversion regression | `llvm-lit -v test/Conversion --filter="ascend-"` 51/51 passed |
| xvm broadcast-add-reduce mainline E2E | 默认 `M=640,N=15000,BLOCK_DIM=20` passed；输出 `step10_kernel.cpp` / `phase5_tiling_space.json` / runtime artifact；`session.result=success`、`session.validation=pass` |
| xvm relu-broadcast-transpose mainline E2E | 默认 `M=640,N=500,BLOCK_DIM=20` passed；输出 `step10_kernel.cpp` / `phase5_tiling_space.json` / runtime artifact；`session.result=success`、`session.validation=pass`；常量投影 broadcast/transpose 当前保守走整块 lowering |
| xvm add-broadcast-concat mainline E2E | 默认 `M=640,N=500,BLOCK_DIM=20` passed；输出 `step10_kernel.cpp` / `phase5_tiling_space.json` / runtime artifact；`session.result=success`、`session.validation=pass`；覆盖 concat dim0 output subview writeback |
| xvm gather-elementwise-fusion mainline E2E | `run.sh --log` 默认 `M=64,N=64,K=16,BLOCK_DIM=20` passed；`run_simbackend_examples.sh gather-elementwise-fusion` passed；focused LIT 3/3 passed；runtime-session sim `session.result=success`、`session.validation=pass` |
| xvm gather-elementwise-fusion M-tail baseline matrix | `M=65,N=80,K=16`、`M=96,N=128,K=16`、`M=160,N=128,K=16`、`M=96,N=128,K=32` passed；早期 matrix 覆盖 M 维动态 shape 与 16 对齐 N/K |
| xvm gather-elementwise-fusion N/K tail matrix | `M=65,N=127,K=31,BLOCK_DIM=20`、`M=96,N=128,K=31,BLOCK_DIM=20`、`M=96,N=127,K=32,BLOCK_DIM=20` passed；`run_simbackend_examples.sh gather-elementwise-fusion` 已纳入三组 matrix 并 passed；覆盖 K tail、N tail、N/K 同时 tail |
| xvm gather N/K tail focused regression | `ninja -C build afir-opt afir-translate AscendCommonAttributesTest` passed；`AscendCommonAttributesTest` 1/1 passed；focused LIT 9/9 passed：schedule axis/problem/guards/search、full-pipeline gather、realize gather bridge、CANN gather/runtime artifacts；host `git diff --check -- . ':!AGENTS.md'` passed；`check_ascend_no_v2_code_naming.sh` passed |
| xvm ordinary example regression after gather tail | `examples/broadcast-add-reduce/run.sh --log`、`examples/relu-broadcast-transpose/run.sh --log`、`examples/add-broadcast-concat/run.sh --log` passed；均报告 `session.result=success`、`session.validation=pass` |
| xvm broadcast-add-reduce shape matrix | `M=65,N=128,BLOCK_DIM=2`、`M=70,N=128,BLOCK_DIM=2`、`M=72,N=128,BLOCK_DIM=2`、`M=70,N=123,BLOCK_DIM=2`、`M=128,N=123,BLOCK_DIM=2` passed；覆盖 rows<16 tail、tail=16/32、N 非 128、无 tail |
| xvm bounded vector tile focused LIT | `ascend-schedule-search.mlir`、`ascend-schedule-vector-bounded-tile.mlir`、`ascend-compute-lower-selected-all-parallel-tile-materializes-loop.mlir`、`ascend-compute-lower-selected-all-parallel-tile-fallback.mlir`、`ascend-full-pipeline-rank2-elementwise-add.mlir`、selected reduction tile、两个 demo full-pipeline LIT passed |
| xvm split-relu-brc-add-mul mainline E2E | 默认 `M=640,N=512,BLOCK_DIM=20` passed；输出 `step10_kernel.cpp` / `phase5_tiling_space.json` / runtime artifact；`run_simbackend_examples.sh split-relu-brc-add-mul` passed；runtime-session sim `session.result=success`、`session.validation=pass` |
| xvm 5-example mainline SimBackend smoke | `run_simbackend_examples.sh broadcast-add-reduce relu-broadcast-transpose add-broadcast-concat gather-elementwise-fusion split-relu-brc-add-mul` passed；覆盖 relu/add/broadcast 二次 manifest 验证、gather 三组 N/K tail matrix、split 默认 mainline |
| xvm matmul-add-leakyrelu mainline E2E | 默认 `M=128,K=256,N=128,BLOCK_DIM=1` passed；输出 `step10_kernel.cpp` / `phase5_tiling_space.json` / runtime artifact；`session.result=success`、`session.validation=pass`；NumPy `max_abs_diff=0.000000e+00`、`mean_abs_diff=0.000000e+00` |
| xvm matmul mix runtime regression | `run_simbackend_examples.sh matmul-add-leakyrelu` passed；`run_mix_repeat.sh` passed；`run_mix_output_reuse.sh` passed；`run_mix_altshape.sh` passed，覆盖 `M=64,K=128,N=96` |
| xvm 6-example mainline SimBackend smoke | `run_simbackend_examples.sh broadcast-add-reduce relu-broadcast-transpose add-broadcast-concat gather-elementwise-fusion split-relu-brc-add-mul matmul-add-leakyrelu` passed；覆盖普通 vector 示例、gather 三组 N/K tail matrix、split 默认 mainline、matmul mix mainline |
| xvm runtime tool regression after matmul mainline | `bash test/tools/runtime/run_runtime.sh` passed；`test_taskgraph_runtime` 973/973 passed；C API runtime 15/15 passed；runtime unit 115/115 passed；SimBackend baseline 与 mix repeat passed |
| xvm Ascend Conversion regression after matmul mix | `llvm-lit -v build/test/Conversion --filter="ascend-"` 55/55 passed；新增 matmul full-pipeline 与 cube bridge LIT passed；code naming guard passed |
| xvm supported example default entries | `examples/broadcast-add-reduce/run.sh --log` 与 `examples/relu-broadcast-transpose/run.sh --log` 均委托新主线并通过 runtime-session sim：`session.result=success`、`session.validation=pass`；已删除 relu 根目录旧生成物后重跑，确认不依赖 stale artifacts |
| xvm runtime tool regression | `bash test/tools/runtime/run_runtime.sh` passed；覆盖 runtime-session CLI / vec example / DAG sim / C API / TaskGraph / SimBackend baseline / mix repeat |
| xvm example smoke follow-up | 删除 broadcast/relu 根目录旧生成物后，`run_simbackend_examples.sh broadcast-add-reduce` passed；`run_simbackend_examples.sh add-broadcast-concat` passed；`run_simbackend_examples.sh gather-elementwise-fusion` passed；`add-broadcast-concat/build_mainline` + `broadcast-add-reduce/build_mainline` cross-session smoke passed |
| xvm Phase 5C+ focused verification | `ninja -C build afir-opt afir-translate AscendBackendSupportMatrixTest` passed；`AscendBackendSupportMatrixTest` 4/4 passed；`check_ascend_schedule_decision_contract.sh` passed；xvm code naming guard passed；`llvm-lit -v build/test/Conversion --filter="ascend-"` 59/59 passed；`llvm-lit -v build/test/Target` 15/15 passed；`examples/transformer/run-mainline.sh` passed：`mainline_prefix=pass`、`rank2_transpose_closure=pass`、`next_gap=rank3_transpose_semantics`；transpose 泛化后 `llvm-lit -v build/test/Conversion --filter="ascend-"` 59/59 passed，`examples/transformer/run-mainline.sh` passed：`transpose_kernelize_generalization=pass`、`next_gap=batch_matmul_schedule`；contract hardening slice 通过 `AscendCommonAttributesTest` 2/2、`AscendRealizePlannerTest` 15/15、`ascend-dependent-dialects.mlir`、`ascend-kernelize-linalg-interface.mlir`、Target lit 18/18 |
| xvm Phase5 body classifier consolidation | `AscendLinalgBodyClassifierTest` 4/4 passed；`AscendBackendSupportMatrixTest` 4/4 passed；`AscendRealizePlannerTest` 15/15 passed；focused ComputeLower/Realize/Phase5/transpose lit 37/37 passed；`llvm-lit -v build/test/Conversion --filter="ascend-"` 61/61 passed；`llvm-lit -v build/test/Target` 15/15 passed；`examples/transformer/run-mainline.sh` passed：`transpose_kernelize_generalization=pass`、`next_gap=batch_matmul_schedule` |
| xvm Realize materialize driver consolidation | `AscendRealizePlannerTest` 16/16 passed，新增 `MemoryRealizationMaterializeMutatesIRAndPlan` 覆盖 driver 入口直接驱动 IR mutation 与 plan 计数回写；Realize / ComputeLower / Phase5 focused lit 36/36 passed；Ascend Conversion lit 61/61 passed；Target lit 15/15 passed；transformer mainline smoke passed，`next_gap=batch_matmul_schedule`；code naming guard passed |
| xvm Schedule cube K-axis consolidation | `ascend-schedule-search.mlir` 新增 matmul `[M,N,K]` selected tile / tail plan 回归 passed；Schedule / ComputeLower / Phase5 / matmul focused lit 29/29 passed；`examples/matmul-add-leakyrelu/run-mainline.sh --log` passed，`session.result=success`、`session.validation=pass`、`max_abs_diff=0`；Ascend Conversion lit 61/61 passed；Target lit 15/15 passed；transformer mainline smoke passed，`next_gap=batch_matmul_schedule`；code naming guard passed |
| xvm Schedule multi-kernel func metadata consolidation | `ascend-schedule-multi-kernel-func-metadata.mlir` RED confirmed old schedule still rejected conflicting func-level metadata；per-kernel metadata fix 后 focused Schedule / Realize / ComputeLower / full-pipeline / transformer lit 50/50 passed；Ascend Conversion lit 62/62 passed；Target lit 15/15 passed；Ascend ctest 9/9 passed；`examples/transformer/run-mainline.sh` passed：`multi_kernel_func_metadata=per_kernel`、`next_gap=batch_matmul_schedule`；host static checks 与 xvm code naming guard passed |
| xvm Backend MemorySpace enum consolidation | `AscendBackendSupportMatrixTest` static_assert RED confirmed backend `MemorySpace` was a distinct enum；fix 后 `MemorySpace` alias 到 `mlir::ascend::MemoryPlace`，`parseMemorySpace(22)` 覆盖 `GMFlat`，unknown 走 `kUnknownMemorySpace` sentinel；Backend / ComputeLower / Realize focused lit 35/35 passed；Ascend Conversion lit 62/62 passed；Target lit 15/15 passed；Ascend ctest 9/9 passed；xvm code naming guard 与 host static checks passed |
| xvm op_roles array schedule contract | `ascend-schedule-op-roles-array.mlir` RED confirmed Schedule saw role `unknown` when only `ascend.op_roles` existed；fix 后 Schedule derives cube / reduction / vector / memory from full role array, Kernelize fallback contract also reads full role array before legacy scalar attr；Kernelize / Schedule focused lit 23/23 passed；Ascend Conversion lit 63/63 passed；Target lit 15/15 passed；Ascend ctest 9/9 passed；xvm code naming guard 与 host static checks passed |
| Kernelize dead config cleanup | `KernelizeConfig` 删除未被任何 analyzer/driver 消费的 `maxBranchesPerCandidate` 与 `localTopKPerPrimaryOpNeighborhood`，避免暴露无效配置契约；`rg` confirmed no remaining references；Kernelize focused lit 8/8 passed；Ascend Conversion lit 63/63 passed；Target lit 15/15 passed；Ascend ctest 9/9 passed；xvm code naming guard 与 host static checks passed |
| xvm Kernelize template-family merge symmetry | `ascend-kernelize-merge-horizontal.mlir` REVERSE case RED confirmed old code did not merge vector->cube；fix 后 Kernelize / pipeline / schedule focused lit 11/11 passed；Ascend Conversion lit 62/62 passed；Target lit 15/15 passed；transformer smoke passed；code naming guard passed |
| xvm Schedule axis static extent conflict | `ascend-schedule-axis-coalescing.mlir` CONFLICT case RED confirmed old schedule returned status 0 and reused `d0 == 4` for a second `d0 == 8` op；fix 后 Schedule / Realize / full-pipeline focused lit 41/41 passed；Ascend Conversion lit 62/62 passed；Target lit 15/15 passed；transformer smoke passed；code naming guard passed |
| xvm transformer full-codegen closure | `batch_matmul` schedule rank3、post-reduction singleton carry、GM->GM copy、GM transpose、GM copy/broadcast generic、GM scalar generic reduction、GM matmul / batch_matmul fallback 均补 LIT；`examples/transformer/run-mainline.sh` 输出 `full_codegen=pass`；Ascend Conversion lit 70/70 passed；Target lit 15/15 passed；Ascend ctest 9/9 passed；code naming guard 与 host `git diff --check` passed |
| xvm KernelPattern hard constraints | `AscendKernelPatternTest` 新增 `MustCoLocate` / `MustSeparate` partition 覆盖并 3/3 passed；`KernelPartitioner` 现在按 co-location 连通分量原子选择候选，并在贪心选择时排除 must-separate 冲突；Kernelize lit 8/8 passed；Ascend Conversion lit 70/70 passed；Ascend ctest 9/9 passed；xvm code naming guard 与 host `git diff --check` passed |
| xvm Kernelize primitive enum | `FusionCandidate.primitive` 与 `MergedCandidate.primitiveCombo` 改为 `KernelizePrimitiveKind`，report 通过集中 stringify 保持输出兼容；schedule family 字符串改用 `Common/Attributes.h` 常量，`op_roles` array 判断改用 `OpRole` stringify；`rg` confirmed Kernelize primitive 赋值不再使用裸字符串；`ninja -C build afir-opt AscendKernelPatternTest` passed；Kernelize lit 8/8 passed；Ascend Conversion lit 70/70 passed；Ascend ctest 9/9 passed |
| xvm Kernelize iterator enum | `OpSemanticSummary.iteratorTypes` 改为 `IteratorKind` enum；`DependencyAnalysis` 删除 `attr.print()` + `contains("parallel/reduction")` 回退，只接受 linalg typed iterator 或精确 string attr；Kernelize lit 8/8 passed；Ascend Conversion lit 70/70 passed；Ascend ctest 9/9 passed |
| xvm Phase5 bridge failure preflight | `AscendRealizePlannerTest.Phase5BridgeFailureDoesNotLeavePartialVecOutAlloc` RED confirmed concat/subview bridge failure left a `VECOUT` alloc；fix 后 concat output bridge 先校验 all dim uses 再做任何 IR mutation，并推迟 `ascendc.unit` 标注到 preflight 之后；Realize focused lit 15/15 passed；Ascend Conversion lit 70/70 passed；Ascend ctest 9/9 passed |
| xvm Realize op_roles array consumption | `ascend-realize-op-roles-array.mlir` RED confirmed only `ascend.op_roles=["Primary","Vector","Injective"]` produced `memory_space_annotations = 0`；fix 后 Realize role checks consume `ascend.op_roles` array before legacy scalar `ascend.op_role`，并集中 `Vector/Cube` array contract strings to `Common/Attributes.h`；focused lit passed；Ascend Conversion lit 71/71 passed；Ascend ctest 9/9 passed |
| xvm Schedule target semantic alignment | `ascend-schedule-axis-coalescing.mlir` TARGETGATHER case RED confirmed target-aware f32 gather 仍输出 legacy `semantic_align=16`；fix 后 gather 轴由 `TargetTilePolicy` 回填 semantic alignment，target-aware 模式按 `TargetMemoryModel::AlignmentRule` 与 result element width 推导为 `8`；focused lit passed；Ascend Conversion lit 71/71 passed；Ascend ctest 9/9 passed |
| xvm Kernelize structural group propagation | `ascend-kernelize-roles.mlir` RED confirmed branch/merge 中间节点没有 group，且 role 仍是普通 vector；fix 后 branch/merge group 沿 elementwise/broadcast/gather/layout-transform 链传播，遇到 merge-root / branch-root 形态停止，中间节点 role 包含 `Branch` / `Merge`；focused lit passed；Kernelize/Schedule/Realize/full-pipeline focused lit 46/46 passed；Ascend Conversion lit 71/71 passed；Ascend ctest 9/9 passed；xvm code naming guard passed |
| xvm Schedule memory role fallback | `ascend-schedule-memory-role.mlir` RED confirmed `OpRole::Memory` kernel 可被解析但没有 template，Schedule 报 `no schedule template`；fix 后新增共享 `kOpRoleMemory`、`memory_copy` template 与 full logical-axis tile；focused lit passed；`AscendCommonAttributesTest` 2/2 passed；Ascend Conversion lit 72/72 passed；Ascend ctest 9/9 passed；xvm code naming guard passed |
| xvm Schedule target policy hooks | `AscendScheduleDecisionTest.TailPolicyPreferenceComesFromTargetPolicy` RED confirmed `TargetTilePolicy` 没有 tail-policy preference API；fix 后 `TargetTilePolicy` 持有 tail preference 与 vector buffer count，`ScheduleDecision` 按策略选择 tail，target-aware tile 推导使用 policy buffer count；focused unit passed；Schedule lit 16/16 passed |
| xvm Realize nested cube bridge dominance | `AscendRealizePlannerTest.Phase5CubeBridgeDominatesNestedVectorUse` RED confirmed matmul 后 `scf.if` region 内 vector consumer 未桥接，alloc/copy 计数为 0；fix 后 `collectSafeCubeVectorUses` 使用 `DominanceInfo` 判断安全 consumer，focused unit passed；Ascend ctest 10/10 passed；Realize/Schedule/ComputeLower/full-pipeline focused lit 59/59 passed；xvm code naming guard passed |
| xvm LinalgToAscendC GM matmul test sync | 完整 Conversion lit 暴露 `linalg-to-ascendc.mlir` 旧 case 仍期望 GM `linalg.matmul` 不转换；当前 `ComputeConversion` 已支持 GM matmul scalar-loop fallback，且 `ascend-compute-lower-matmul-gm.mlir` 已覆盖同语义；测试期望改为检查 `scf.for` / `arith.mulf` / `arith.addf` / `memref.store`，Ascend Conversion lit 79/79 passed |
| xvm Schedule target-aware rank fallback removal | `ascend-schedule-target-tile-policy.mlir` RED confirmed rank-1 reduction 在 target-aware 下仍输出 `[32,?]`；fix 后 dynamic-inner rank2 保留后端安全默认 tile `[32,32]` / `target_dynamic_inner_32`，dynamic reduction 轴在无法静态估算 input footprint 时保守输出 `[32,?]` / `target_dynamic_reduction_32`，避免 target-aware mainline examples 退化到 `1x1` 或过大 reduction tile；focused lit passed |
| xvm Kernelize generic contraction traiting | `ascend-kernelize-generic-contraction.mlir` RED confirmed matmul-like `linalg.generic` 仍被名字无关 fallback 识别为 Reduction；fix 后 contraction 由 linalg indexing maps + reduction iterator + output map 不含 reduction dim 推导，`rg` confirmed Kernelize 不再对具体 `linalg.*` 名字做 access-pattern 分支；focused Kernelize lit 4/4 passed |
| xvm Kernelize reduction fusion closure | `ascend-kernelize-reduction-fusion.mlir` RED confirmed reduction seed 被强制标为 `Primary`，导致 vector consumer 无法成为父 kernel；`AscendCandidateMergeAnalyzerTest.IteratesMergedCandidatesToFixpoint` RED confirmed pairwise merge 不收敛；`AscendKernelPatternBuilderTest.ProducesExplicitPlacementEdgesFromAttrs` RED confirmed hard constraints 只有测试注入、无自然生产者；`ascend-schedule-axis-coalescing.mlir` POST-REDUCE RED confirmed非 primary reduction + primary vector epilogue 会触发 singleton extent 冲突；`ascend-kernelize-dependency.mlir` 覆盖 dynamic empty shape 依赖不计为 data dependency；`ascend-kernelize-merge-horizontal.mlir` 覆盖共享常量但 result shape 不兼容的 fills 不做 horizontal fusion；fix 后 reduction 只标 `Reduction`，reduction->vector candidate 以 vector consumer 为 primary，merge fixed-point 收敛，horizontal dependency 使用分析图，`MustCoLocate` / `MustSeparate` 可由 kernelize group attrs 生产，AxisCoalescer 用 dominant-role op 承载 reduction axes；focused unit 2/2 passed；Kernelize lit 10/10 passed；`ascend-phase5-transformer-dynamic-smoke.mlir` passed；Ascend ctest 10/10 passed；Conversion ascend lit 75/75 passed；Target lit 15/15 passed；xvm code naming guard passed |
| xvm Ascend public header boundary | `ascend-public-header-boundary.mlir` RED confirmed `include/Conversion/Ascend/Kernelize/CandidateClosure.h` 等内部头仍公开；fix 后 Kernelize / Schedule / Realize public include 仅保留 `KernelizePass.h` / `SchedulePass.h` / `RealizePass.h`，内部头移到 `lib/Conversion/Ascend/...`；focused header lit passed；key internal unit targets rebuilt；Ascend ctest 10/10 passed；Ascend Conversion lit 81/81 passed；xvm code naming guard passed |
| xvm StaticMemoryPlan byte peak accounting | `AscendRealizePlannerTest.StaticMemoryPlannerComputesStaticBytePeakForVectorTemporary` RED confirmed planner 缺 byte-level peak API；`AscendRealizePlannerTest.BufferizationDriverCollectsStaticByteFacts` RED confirmed tensor facts 不计算 byte size；fix 后 static `tensor<64xf16>` input/output/temp bytes 进入 `BufferizedKernelIR`，target-aware vector temporary 报告 `local_buffer_bytes=128` / `workspace_bytes=128` / `peak_usage_bytes=128`；focused Realize unit passed；Realize focused lit 43/43 passed；Ascend ctest 10/10 passed；Ascend Conversion lit 81/81 passed |
| xvm Realize commercial blocker closure | `BufferizationDriverReadsOpRolesArrayForVectorTemporary` RED confirmed plan tier 只读 legacy scalar role；`TargetAwareMovementPlannerSelectsPathFromProductionWorkspacePlan` RED confirmed production workspace 仍落 `VECCALC` 且 selected path 为 0；`Phase5CubeBridgeSupportsBatchMatmul` RED confirmed cube bridge 只处理 `linalg.matmul`；fix 后 focused 6 tests passed；`AscendRealizePlannerTest` 30/30 passed；Ascend ctest 10/10 passed；Conversion ascend lit 75/75 passed；Target lit 15/15 passed；xvm code naming guard passed；host `git diff --check` passed |
| xvm Schedule explicit policy and idempotency | `ascend-schedule-requires-target-policy.mlir` RED confirmed stock `--ascend-schedule` still silently emitted legacy schedule；`ascend-schedule-idempotency.mlir` RED confirmed stale `ascend.schedule.kernel_metadata` caused conflicting metadata failure on rerun；fix 后 default policy is `require-explicit`，Schedule pass pre-clears owned schedule attrs before recomputing, existing tests/examples explicitly request `target-tile-policy=legacy-default`；focused 5 lit passed；Ascend ctest 10/10 passed；Conversion ascend lit 77/77 passed；Target lit 15/15 passed；xvm code naming guard passed；host `git diff --check` passed |
| xvm StaticMemoryPlan target capacity check | `AscendRealizePlannerTest.StaticMemoryPlannerRejectsPeakUsageOverTargetCapacity` RED confirmed planner 没有 target capacity overload；fix 后 target-aware static memory plan 用 `TargetMemoryModel::getCapacity(VECIN)` 校验 `peakUsageByteCount`，超容量返回 failure，容量通过时 `capacity_check_deferred=false`；focused Realize unit passed；workspace/data-movement focused lit passed |
| xvm StaticMemoryPlan value-level workspace slots | `AscendRealizePlannerTest.BufferizationDriverBuildsStableValueFacts` / `StaticMemoryPlannerBuildsValueLevelWorkspaceSlots` RED confirmed planner 缺 value-level facts 与 workspace slot API；fix 后 `BufferizationDriver` 输出稳定 `valueFacts`，`StaticMemoryPlanner` 为 vector temporary 生成 `liveIntervals` / `workspaceSlots`；focused unit passed；Ascend ctest 10/10 passed；Realize/ComputeLower/full-pipeline focused lit 43/43 passed；Ascend Conversion lit 81/81 passed；Target lit 15/15 passed；code naming guard passed |
| xvm MovementPlan value-level movement steps | `AscendRealizePlannerTest.MovementPlannerBuildsValueLevelStepsForWorkspaceSlots` / `MovementPlannerSelectsDirectTargetPathForMovementSteps` RED confirmed planner 缺 movement step API 与 target-aware path overload；fix 后 `MovementPlan` 保留 per-slot movement step，target-aware build 用 `TargetMemoryModel::findDirectPaths` 选择 `GM -> VECIN` direct path；缺 direct path 的手工 fixture 保持 deferred；focused unit passed；Ascend ctest 10/10 passed；Realize/ComputeLower/full-pipeline focused lit 43/43 passed；Ascend Conversion lit 81/81 passed；Target lit 15/15 passed；code naming guard passed |
| xvm MovementStep selected path materialization | `AscendRealizePlannerTest.MemoryRealizationMaterializesSelectedMovementSteps` RED confirmed selected movement step 不改 IR、不回写 alloc/copy 计数；fix 后 `MemoryRealizationDriver` 为 selected GM->local movement 创建目标 memory-space alloc、插入 `memref.copy`、改写同 block/same-kernel linalg input，并把 movement 与 Phase5 bridge materialization counts 合并；focused unit passed；Ascend ctest 10/10 passed；Realize/ComputeLower/full-pipeline focused lit 43/43 passed；Ascend Conversion lit 81/81 passed；Target lit 15/15 passed；code naming guard passed |
| xvm MovementStep workspace subview materialization | `AscendRealizePlannerTest.MemoryRealizationMaterializesMovementStepsThroughWorkspaceSubviews` RED confirmed two selected same-shape GM inputs created two `VECIN` allocs and zero subviews；fix 后同 kernel / 同 block / 同目标 memory space / 静态 identity 同形 movement steps 合并为一个 workspace alloc，按 slot 生成 rank-reduced `memref.subview`，并保留每 step 一个 copy；focused unit 2/2 passed；Ascend ctest 10/10 passed；Realize/ComputeLower/full-pipeline focused lit 43/43 passed；Ascend Conversion lit 81/81 passed；Target lit 15/15 passed；code naming guard passed |
| xvm Realize selected movement view-chain | `ascend-realize-view-chain-movement.mlir` RED confirmed `tensor.extract_slice` consumer chain made `temporary_values = 0` and selected movement materialization stayed at 0 alloc/copy；`ascend-realize-reshape-view-chain-movement.mlir` RED confirmed reshape view-chain 不能 materialize；fix 后 Bufferization facts 透明追溯 tensor view producer/consumer，materialize 层在 `VECIN` 上重建 `memref.subview` / `memref.cast` / `memref.expand_shape` / `memref.collapse_shape` view-chain 并改写 consumer；`AscendRealizePlannerTest` passed；Realize/full-pipeline lit 26 discovered passed；Ascend Conversion + CANN translate lit 105 discovered passed；Ascend ctest 10/10 passed；all `examples/*/run-mainline.sh` passed；code naming guard passed |
| xvm commercial readiness residual closure | Realize dynamic view-chain、Kernelize seed policy、Kernelize family resolver、Schedule file-backed tuning cache、generated `KernelizeSemanticOpInterface` 五项完成并逐项通过 spec/code review；final review 追加修复 native interface resolution failure fail-open 风险；final host static checks `git diff --check`、`check_ascend_public_headers.sh`、`check_ascend_no_v2_code_naming.sh` passed；xvm `ninja -C build afir-opt afir-translate AscendKernelizeOpInterfaceTest AscendKernelPatternTest AscendRealizePlannerTest AscendScheduleDecisionTest` passed；ctest focused 4/4 passed；focused lit 13/13 passed；Conversion `ascend-` lit 92/92 passed；Target lit 23/23 passed；`examples/transformer/run-mainline.sh` 输出 `transformer_dynamic.full_codegen=pass`；`examples/relu-broadcast-transpose/run-mainline.sh` 与 `examples/matmul-add-leakyrelu/run-mainline.sh --log` 均输出 `session.validation=pass`；matmul mainline 单独复跑也输出 `max_abs_diff=0.000000e+00` |
| xvm runtime graph / queue lifetime hardening | `ninja -C build -j6 afir-opt afir-translate` passed；focused LIT 6/6 passed（queue lifetime checker、full-pipeline queue lifetime、multi `TilingData`、rank2 elementwise barrier、CANN graph alias、10-example suite）；Ascend/Target filtered LIT 104/104 passed；full Target LIT 23/23 passed；Ascend ctest 14/14 passed；standalone runtime-session regressions passed for `two-kernel-dag --n 64`、`two-kernel-rank-mix-dag --n 64 --k 32`、`three-kernel-dag --n 64`、`rmsnorm-reduction-core`；generated queue lifetime audit over 40 MLIR files reported `deque=228 free=228`；source/generated audit found no lane0 `SetValue(0)` scalar patch, only ReduceSum scalar-result `GetValue(0)` reads |
| xvm CANN workspace ABI closure | RED confirmed `ascend-realize-workspace-layout.mlir` report had `workspace_bytes = 128` but output IR lacked `cann.workspace_size_bytes`；fix 后 Realize stamps static planner bytes onto scheduled func, multi-kernel same func conservatively aggregates unique kernel workspace bytes；`CannRuntimeArtifacts` consumes `cann.workspace_size_bytes` for tiling space `workspace_size_expr`、runtime manifest root/per-entry `workspaceSizeBytes` / `workspace.sizeBytes` and host tiling `_GetWorkspaceSize`；focused LIT 3/3 passed；Realize + CANN translate filtered LIT 44/44 passed；Ascend ctest 14/14 passed；Conversion + Target LIT 133/133 passed；examples LIT 1/1 passed；full `build/test` only hits existing Python checker discovery unresolved outside this change |

## Phase 6：文档与 Demo 收敛

| 任务 | 状态 | 说明 |
|---|---|---|
| 合并单节 review 文档回 V2 主文档 | `Deferred` | 待各节实现稳定 |
| 重写 Architecture Design V1 | `Deferred` | 基于稳定 V2 |
| 重写 demo 文档 | `Deferred` | 基于稳定 V2 案例结构 |

## Review 与验证规则

每个实现任务默认需要以下状态项：

| 项 | 要求 |
|---|---|
| spec review | 检查是否满足对应 V2 章节和计划范围 |
| code review | 检查编译风险、API 风险、测试脆弱性 |
| focused lit | 覆盖本任务新增行为 |
| docker build | 在 xvm/docker 中构建相关 target |
| regression | 至少跑 `check-afir` 或说明不可跑原因 |
| commit | 每个可 review 单元单独提交 |

## 当前下一步

Phase 5 Translate / Runtime Artifact 首轮已完成：

```text
Phase 5: ComputeLoweringDriver -> ABI lowering -> HostTilingEmitter -> RuntimeManifestBuilder -> transformer smoke
```

执行入口：

- `docs/Ascend-MLIR-Detailed-Implementation-V2.zh.md`
- `docs/Ascend-MLIR-Detailed-Implementation-V2-6.zh.md`
- `docs/Ascend-MLIR-Detailed-Implementation-V2-9.zh.md`

后续切分：

1. Phase 4 剩余 target 查询模型已完成：`TargetMemoryModel`、`TargetIntrinsicModel`、`TargetCostModel`、`TargetModelVerifier`
2. Phase 3B MVP 链路已闭环：One-Shot Bufferize、target-aware placement、workspace layout/lifetime、explicit data movement plan、memory-space annotation 均已实现和验证
3. Phase 5C 已完成最小普通 tensor/linalg 用例的完整 Phase 0 -> Phase 5 positive smoke
4. `broadcast-add-reduce` 已完成 axis contract、tail-policy guards、bounded M-axis tiling、Phase 5 tile metadata 和 rows<16 tail fallback；默认 `M=640,N=15000` 与动态 tail shape matrix 均已走新主线并通过 runtime-session sim
5. `relu-broadcast-transpose` 已新增新主线脚本和 full-pipeline LIT，覆盖 `(d0,d1)->(d1,0)` 常量投影 broadcast/transpose 到 CANN codegen/runtime sim；rank2 identity all-parallel ordinary path 已补充 selected tile -> `ascendc.get_block_idx` full-pipeline LIT
6. `gather-elementwise-fusion` 默认入口已迁移到新主线，覆盖 index-select gather + relu + bias add 到 CANN codegen/runtime sim；M 维 shape matrix 与 N/K 非 16 tail matrix 已通过，脚本/data generator 不再要求 N/K 16 对齐
7. `matmul-add-leakyrelu` 默认入口已迁移到新主线，覆盖 dynamic matmul + bias add + leaky_relu 到 CANN mix codegen/runtime sim
8. Phase 5C+ 已完成显式 multi-kernel DAG runtime manifest、target-aware schedule tile policy、rank-agnostic transpose Kernelize、rank2 swap transpose backend lowering，以及 multi-kernel function schedule metadata fail-closed；下一步先把 schedule metadata 下沉到 per-kernel schema，再推进 transformer attention rank3 batch matmul 与 compiler-generated kernel DAG metadata
