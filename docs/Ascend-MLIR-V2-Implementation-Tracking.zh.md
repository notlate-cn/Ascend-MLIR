# Ascend-MLIR V2 实现跟踪看板

本文档用于跟踪 `Ascend-MLIR-Detailed-Implementation-V2` 的工程落地进展。

- 详细规格入口：`docs/Ascend-MLIR-Detailed-Implementation-V2.zh.md`
- 拆分规格：`docs/Ascend-MLIR-Detailed-Implementation-V2-1.zh.md` 到 `V2-9.zh.md`
- 第一轮 MVP 计划：`docs/superpowers/plans/2026-05-07-ascend-mlir-v2-mvp.md`
- Phase 1 Kernelize 计划：`docs/superpowers/plans/2026-05-08-ascend-mlir-v2-kernelize-candidates.md`
- Phase 2 Schedule 计划：`docs/superpowers/plans/2026-05-08-ascend-mlir-v2-schedule-full-search.md`

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
| Phase 2 | Schedule 完整搜索与 guard/cache | `In Progress` | Task 2 AxisCoalescer 已完成；下一步进入 ScheduleProblemBuilder |
| Phase 3 | Realize plan objects | `Planned` | 依赖稳定 `ScheduleDecisionSet` |
| Phase 4 | Target model 完整化 | `Planned` | 与 Phase 2/3 并行推进 |
| Phase 5 | Translate / runtime artifact 对接 | `Planned` | 依赖 Realize 和 ABI 设计稳定 |
| Phase 6 | 架构文档与 demo 重写 | `Deferred` | 待 V2 主链路稳定后启动 |

## Phase 0：V2 MVP 编译主干

| 任务 | 对应规格 | 状态 | 主要产物 | 验证 |
|---|---|---|---|---|
| V2 pass skeleton | V2-1 / V2-9 | `Done` | `--ascend-normalize`、`--ascend-kernelize`、`--ascend-schedule` | `check-afir` 覆盖 |
| Target Profile MVP | V2-8 | `Done` | `TargetProfile`、`CannTargetProfileLoader`、`--ascend-print-target-profile` | `test/Target/ascend-target-profile.mlir` |
| Normalize MVP | V2-2 | `Done` | dialect 白名单、`ascend.v2.normalized` | `test/Conversion/ascend-normalize.mlir` |
| Kernelize MVP | V2-3 | `Done` | `ascend.v2.op_role`、`ascend.v2.kernel`、`ascend.v2.primary` | `test/Conversion/ascend-kernelize-mvp.mlir` |
| Schedule MVP | V2-4 | `Done` | fixed schedule family/template/decision attrs | `test/Conversion/ascend-schedule-mvp.mlir` |
| Vertical MVP pipeline | V2-9 | `Done` | Normalize -> Kernelize -> Schedule smoke test | `test/Conversion/ascend-v2-pipeline-mvp.mlir` |

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
| `KernelPatternView` | `Done` | 从 `ascend.v2.kernel` / `ascend.v2.primary` / `ascend.v2.op_role` 重建 pattern-level schedule view | 新增 pattern-view lit；同一 kernel 内 ops 共享 schedule decision |
| `AxisCoalescer` | `Done` | 轴合并与 coalesced axis info | rank-2/reduction/broadcast/matmul/multi-primary lit |
| `ScheduleProblemBuilder` | `Planned` | 从 `KernelPattern` 构建调度问题 | report 输出 shape/axis/constraint |
| `TemplateRegistry` | `Planned` | 注册 schedule family/template | vector/reduction/cube family 可查询 |
| `ScheduleSearch` | `Planned` | 搜索 `ScheduleInstance` | compileTimeTopK 生效 |
| guard 生成 | `Planned` | `candidateGuards` / `decisionGuards` | 动态 shape bucket lit |
| cache 建模 | `Planned` | `ShapeBucketCache` / `TuningResultCache` | key、negative cache、LRU 测试 |
| `ScheduleDecisionSet` | `Planned` | 输出多个候选决策 | runtimeTopK 可导出 |
| structured lowering | `Planned` | 生成稳定 loop skeleton | 替代手写 transform |

计划文件：

- `docs/superpowers/plans/2026-05-08-ascend-mlir-v2-schedule-full-search.md`

### Phase 2 执行记录

| 计划任务 | 状态 | 提交 | 验证 |
|---|---|---|---|
| Task 0: Schedule full search 计划 | `Done` | `047d04f` | 计划覆盖开发、spec review、code review、xvm/docker 验证、跟踪更新 |
| Task 1: Shared Schedule Types and KernelPatternView | `Done` | `df41143` | `git diff --check` passed；spec review passed；code quality review approved；xvm `afir-opt` build passed；focused lit 3/3 passed；`check-afir` 36 discovered, 33 passed, 3 unsupported |
| Task 2: AxisCoalescer MVP | `Done` | `0d0d058` | TDD RED/GREEN completed；spec review passed；code quality review approved；xvm `afir-opt` build passed；focused lit 4/4 passed；`check-afir` 37 discovered, 34 passed, 3 unsupported |

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

## Phase 3：Realize Plan Objects

目标：把现有 buffer placement 原型升级为 V2-5 的显式 plan 闭环。

| 任务 | 状态 | 说明 | 验收 |
|---|---|---|---|
| `BufferizationDriver` 对接 | `Planned` | 继续复用 One-Shot Bufferize | bufferized IR smoke |
| `PlacementPlan` | `Planned` | resolved placement 规划 | 与 target memory place 对齐 |
| `StaticMemoryPlan` | `Planned` | workspace layout / lifetime | peak workspace 可验证 |
| `MovementPlan` | `Planned` | 显式 data movement 路径 | IR 与 plan 双向一致 |
| `MemoryRealizationPlan` | `Planned` | 汇总 realization 结果 | verifier 通过 |

## Phase 4：Target Model 完整化

目标：把 Target MVP 升级为 V2-8 完整跨层 target 查询模型。

| 任务 | 状态 | 说明 | 验收 |
|---|---|---|---|
| `TargetMemoryModel` | `Planned` | memory places、capacity、alignment、path graph | 必需路径 verifier |
| `TargetIntrinsicModel` | `Planned` | compute/movement intrinsic dtype map | intrinsic lookup 测试 |
| `TargetCostModel` | `Planned` | memory rates / path cost | cost query 测试 |
| profile verifier | `Planned` | profile/memory/intrinsic 闭合检查 | 缺字段 fail-fast |
| 多 SoC 支持 | `Planned` | 910B2 之外的 ini | 参数化 lit 或 unit tests |

## Phase 5：Translate / Runtime Artifact

目标：对齐 V2-6 / V2-9，把上游决策转成 backend/runtime 可消费产物。

| 任务 | 状态 | 说明 | 验收 |
|---|---|---|---|
| `ComputeLoweringDriver` 对齐 | `Planned` | 消费 realization/schedule plan | backend op 覆盖 |
| ABI lowering 对齐 | `Planned` | CANN kernel signature | ABI verifier |
| `HostTilingEmitter` | `Planned` | host tiling codegen | tiling field 顺序一致 |
| `RuntimeManifestBuilder` | `Planned` | schedule entries / guard set / DAG | manifest verifier |
| `tiling_space.json` export | `Planned` | 自动导出调优空间 | JSON schema 测试 |

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

下一步继续执行 Phase 2 计划：

```text
Task 3: ScheduleProblemBuilder MVP
```

执行入口：

- `docs/superpowers/plans/2026-05-08-ascend-mlir-v2-schedule-full-search.md`

后续切分：

1. `AxisCoalescer`
2. `ScheduleProblemBuilder`
3. `TemplateRegistry`
4. `ScheduleSearch`
5. guard 生成
6. cache 建模
7. `ScheduleDecisionSet`
8. structured lowering
