# Ascend-MLIR V2 实现跟踪看板

本文档用于跟踪 `Ascend-MLIR-Detailed-Implementation-V2` 的工程落地进展。

- 详细规格入口：`docs/Ascend-MLIR-Detailed-Implementation-V2.zh.md`
- 拆分规格：`docs/Ascend-MLIR-Detailed-Implementation-V2-1.zh.md` 到 `V2-9.zh.md`
- 第一轮 MVP 计划：`docs/superpowers/plans/2026-05-07-ascend-mlir-v2-mvp.md`
- Phase 1 Kernelize 计划：`docs/superpowers/plans/2026-05-08-ascend-mlir-v2-kernelize-candidates.md`
- Phase 2 Schedule 计划：`docs/superpowers/plans/2026-05-08-ascend-mlir-v2-schedule-full-search.md`
- 代码命名去版本化计划：`docs/superpowers/plans/2026-05-09-ascend-mlir-remove-v2-code-naming.md`

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
| Phase 3 | Realize plan objects | `In Progress` | `--ascend-realize` MVP 与代码命名去版本化已完成，下一步接入 bufferization / placement |
| Phase 4 | Target model 完整化 | `Planned` | 与 Phase 2/3 并行推进 |
| Phase 5 | Translate / runtime artifact 对接 | `Planned` | 依赖 Realize 和 ABI 设计稳定 |
| Phase 6 | 架构文档与 demo 重写 | `Deferred` | 待 V2 主链路稳定后启动 |

## Phase 0：V2 MVP 编译主干

| 任务 | 对应规格 | 状态 | 主要产物 | 验证 |
|---|---|---|---|---|
| V2 pass skeleton | V2-1 / V2-9 | `Done` | `--ascend-normalize`、`--ascend-kernelize`、`--ascend-schedule` | `check-afir` 覆盖 |
| Target Profile MVP | V2-8 | `Done` | `TargetProfile`、`CannTargetProfileLoader`、`--ascend-print-target-profile` | `test/Target/ascend-target-profile.mlir` |
| Normalize MVP | V2-2 | `Done` | dialect 白名单、`ascend.normalized` | `test/Conversion/ascend-normalize.mlir` |
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
| `TemplateRegistry` | `Done` | 注册 schedule family/template | vector/reduction/cube family 可查询 |
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
| `PlacementPlan` | `Planned` | resolved placement 规划 | 与 target memory place 对齐 |
| `StaticMemoryPlan` | `Planned` | workspace layout / lifetime | peak workspace 可验证 |
| `MovementPlan` | `Planned` | 显式 data movement 路径 | IR 与 plan 双向一致 |
| `MemoryRealizationPlan` | `Planned` | 汇总 realization 结果 | verifier 通过 |

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

下一步进入 Phase 3 计划拆解：

```text
Phase 3: Realize Plan Objects
```

执行入口：

- `docs/Ascend-MLIR-Detailed-Implementation-V2-5.zh.md`
- `docs/Ascend-MLIR-Detailed-Implementation-V2.zh.md`

后续切分：

1. 基于 V2-5 生成 Phase 3 implementation plan
2. 先落 `MemoryRealizationPlan` / `PlacementPlan` 数据模型
3. 再接入 bufferization、static memory、movement plan 与 verifier
