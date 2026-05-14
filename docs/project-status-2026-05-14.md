# Ascend-MLIR 项目状态总览（2026-05-14）

> 编制日期：2026-05-14（branch `llm-net`，HEAD `190c814`）
> 范围：全栈架构总览 + 与 `docs/vector-plan/` 设计文档的偏移分析 + 当前开发进度
> 读者：希望快速接手 / 评估当前位置的工程师

---

## 1. 项目目标与定位

Ascend-MLIR 把 PyTorch 上的算子图一路 lower 到 Ascend NPU 的
AscendC kernel + aclnn 调用，并自动搜索 tiling 参数。整体技术线由两条互
相支撑的轴构成：

1. **Compiler 轴**：`torch.export → linalg → afir → vector-plan tiled →
   ascendc → CANN signature → AscendC C++ 源码`，配合 SymExpr 一套
   符号 shape 系统贯穿全程。
2. **Runtime 轴**：`ExecutionSession → GlobalScheduler → SimBackend /
   NpuBackend / AclnnBackend → NativeExecutionRunner`，并由 autotuner +
   network-runner 组合起来做"多 kernel 混合编译 + 搜参 + 验证"。

当前分支 `llm-net` 的主线工作集中在两件事：

- **vector-plan 子系统**：从最初 03 文档草拟的 Pass2 框架，演进为已落地
  的 `Collapse → TilePlanGen → GroupEmitter → LoopNestBuilder` 流水线，
  并把 AutoFuse 调度器（AF）逐步移植进来。
- **Mixed AscendC + aclnn 端到端**：Python 侧 `network_runner.py` 把
  outline / codegen / compile / autotune / verify 5 个 phase 全部串起，
  在 CPU camodel 上跑通 `mixed-attn-e2e`。

---

## 2. 顶层目录速查

```
include/                # public 头文件，与 lib/ 镜像
lib/
  Analysis/SymbolicShape/         # SymExpr + DimSymbolTable
  Conversion/
    AFIRToASCIR / AFIRToASCIRText # AFIR <-> ASCIR
    AscendCBufferPlacement        # GM/UB 层级
    AscendCFoldConcatAlloc
    AscendCParallelize            # 多核并行
    AscendCPrepareForEmit         # 最终 IR 清理（Phase B 落点）
    CanonicalizeCannSignature     # TilingData struct 规范化
    EliminateCfAssert
    FuseGatherElementwise
    LinalgToAscendC               # generic.body → ascendc.* ops
    LowerNonLinalgOps
    MarkStructuredOps
    TorchFrontend                 # torch dialect → linalg
    VectorPlan/                   # 本项目核心：5 大子目录
      GroupAnalysis/  GroupOutline/  TileFuse/  TileInfo/
  Dialect/AFIR/                   # AFIR dialect & Transforms
  Dialect/TmTensor/
  Runtime/
    AclnnBackend/  Artifact/  Execution/  Mix/  Profile/  Support/
  Target/CannKernel/              # MLIR -> AscendC C++ 翻译
tools/
  afir-opt/  afir-translate/      # 编译器 CLI
  aclnn-backend/                  # 生成 host 端 aclnn + AscendC 调用
  autotuner/                      # tiling 参数搜索
  mix-compiler/  mix-tiling-helper/
  runtime-session/                # 单一统一入口，对应 single-runtime-cli
python/
  network_runner.py               # 5-phase 混合 runner
  torch/torch2linalg/             # torch.export → linalg 桥
  torch/framework/                # E2E 测试框架
examples/                         # ~30 个 e2e demo
docs/
  vector-plan/                    # ★ 本文对照基线
  vector-plan.bak/                # 旧设计快照
  superpowers/plans/              # 滚动 plan / 补充设计
```

---

## 3. 完整 lowering pipeline

```
PyTorch nn.Module
  │  torch.export + torch-mlir
  ▼
Linalg IR （tensor<?xf32>, dynamic dims 标注）
  │  fold-unit-extent-dims
  │  linalg-generalize-named-ops          (mul/add/... → linalg.generic)
  │  linalg-elementwise-op-fusion
  ▼
afir-symbolize-shapes                     [P6 已落地, c862fbe]
  │  ─ 给每个 ? 维分配 SymId
  │  ─ 通过 indexing map 做 unification 传播
  │  ─ attrs: afir.dim_symbols (func) / afir.symbolic_shapes (op)
  ▼
vector-plan-group-analysis                [Pass1, 标 group attr]
  │  ─ MarkStructuredOps + LinalgInferShape 等准备
  ▼
vector-plan-group-outline                 [拆 .mlir / .json]
  │  ─ 输出 group_*.mlir + network.json
  ▼
vector-plan-tile-fuse                     [Pass2, 核心]
  │  ─ Collapse  ─ Case A/B2/C 分类 + 广播吸收
  │  ─ TilePlanGen ─ 枚举 / 构建 / 评分 / 选 (P3-P6)
  │  ─ GroupEmitter ─ op clone + 广播轴 LICM
  │  ─ LoopNestBuilder + SliceComputer
  ▼
one-shot-bufferize                        (tensor → memref)
  ▼
ascendc-buffer-placement                  (GM/UB 层级)
  ▼
linalg-to-ascendc                         (generic.body → ascendc.*)
  ▼
ascendc-parallelize                       (get_block_idx, 多核划分)
  ▼
ascendc-pack-tiling-data                  (Phase B 第二段)
ascendc-prepare-for-emit                  (Phase B 第三段)
canonicalize-cann-signature
  ▼
afir-translate -mlir-to-cann
  ▼
AscendC C++ kernel + tiling_space.json
  ▼
autotuner + runtime-session  →  best_config.json  →  最终 .bin
```

每个阶段对应的目录 / pass 名见第 2 节，关键文件在第 4 节细化。

---

## 4. 各子系统现状

### 4.1 Torch frontend & E2E 框架

- **torch.export → linalg**：`python/torch/torch2linalg/convert.py`
  借 torch-mlir bridge。
- **E2E framework**：`python/torch/framework/{pipeline.py,tensor_spec.py}`，
  `examples/torch_e2e/test_fused.py` 是当前 canonical demo。
- **可跑用例**：
  - `test_fused_elementwise`：`(a*b + a) * b`，2D dynamic
  - `test_fused_relu`：`relu(a*b + a)`，select+cmpf 模式（998edb7 修复）
  - `examples/add-mul-relu-e2e`：3D dynamic `max(a+b*c, 0)`，对应
    plan `2026-05-06-add-mul-relu-fusion-e2e.md`

### 4.2 AFIR dialect & SymExpr

- **AFIR ops / dialect**：`lib/Dialect/AFIR/`，含 transforms 目录
  - `AFIRSymbolizeShapes.cpp`（392 行）：linalg 级符号化
  - `LinalgAddBroadcast.cpp`、`LinalgMark.cpp` 等准备 pass
  - `EmitNetworkJsonPass.cpp`：手写 network.mlir 也能产出 network.json
    （commit 9c68556）
- **SymExpr 库**：`lib/Analysis/SymbolicShape/`
  - `SymExpr.cpp`（414 行 9 op：Sym/Const/Add/Sub/Mul/CeilDiv/Mod/Min/Max）
  - `DimSymbolTable.cpp`
- **覆盖范围（档 C: C1–C4 + P6 全部已合入 llm-net）**：
  | 阶段 | commit |
  |------|--------|
  | C1：afir.iter_extents 携带 SymExpr | 8e07d98 |
  | C1c：tile-fuse 透传 | 6d720ba |
  | C2：emit block_dim_expr 到 tiling_space | dd4593c |
  | C3：用 afir.dim_symbols 去重 TilingData | 6d81d9c |
  | C4：autotuner evalBlockDimExpr 支持完整语法 | 44f509f |
  | P6：UB-peak penalty 进入 costEstimate | b81c9dc |

### 4.3 vector-plan 子系统（与设计文档对照详见 §5）

实际落地的目录结构：

```
lib/Conversion/VectorPlan/
  GroupAnalysis/  ─ Pass1
  GroupOutline/   ─ GroupOutlinePass.cpp (414L)
                    NetworkJsonEmitter.cpp (199L)
                    IsolateKernelOutputs.cpp ★ 新（28c8ea6 修 R3）
  TileFuse/       ─ Collapse.cpp (599L)        Case A/B2/C
                    TilePlanGen.cpp (593L)     P2–P6
                    GroupEmitter.cpp (556L)    P3 + LICM
                    LoopNestBuilder.cpp (135L)
                    SliceComputer.cpp (105L)
                    BroadcastAbsorb.cpp / FoldShadowAlloc.cpp
                    InsertTileBuffers.cpp / TileFuseUtils.cpp
                    TileFusePass.cpp 主驱动
  TileInfo/       ─ TilePlanToTileInfo.*
include/Conversion/VectorPlan/
  GroupInfo.h     ★ AxisKind / AxisClass / AxisGrouping 在此
  TilePlan.h      ★ ReduceTemplate enum
  TileInfo.h      ValueExpr / TileFieldSpec
```

### 4.4 LinalgToAscendC + AscendC* passes

- `lib/Conversion/LinalgToAscendC/` 把 `linalg.generic.body` 翻译成
  `ascendc.*` ops。已支持算子集：
  - elementwise: add/mul/sub/div/max/min/neg/abs/sin/cos/tanh/sigmoid/
    exp/log/sqrt/rsqrt
  - reduce: sum/max/min（带 init）
  - data move: Broadcast（单轴）、Gather、Transpose（memref 重排）
  - select+cmpf 重写为 maximumf/minimumf（998edb7）
- **关键修复**：
  - 22f06cf：遇到不支持的 body op 直接报错，不再静默丢弃（此前是潜在
    数值错误源）
  - bb26416：reduce 结果注册成 live tensor，VECCALC 输出修复
- **AscendCParallelize / BufferPlacement / PrepareForEmit / Canonicalize-
  CannSignature** 各自独立 pass，目录与 §2 同名，逻辑相对薄。
- **AscendCFoldConcatAlloc**：消除 concat 引入的多余 alloc。

### 4.5 CannKernel target（AscendC C++ 输出）

- `lib/Target/CannKernel/CannTranslation.cpp`（约 120 KB / 数千行），
  通过 `ascir::CodeEmitter` 走 op-by-op 输出。
- TilingData struct：含 `XBLOCK / XBLOCK_SUB + dim_arg*_*`，对应
  `afir-translate` 的产物。
- block_dim_expr：`ceil(arg0_dim0 / XBLOCK)` 之类的 SymExpr，序列化
  到 `tiling_space.json`，autotuner 在 C4 之后能完整求值。

### 4.6 Runtime 子系统

`lib/Runtime/Execution/` 形成的现状（无 top-level `runtime/` 目录，全部
在 `lib/Runtime/`）：

| 组件 | 角色 |
|------|------|
| **ExecutionSession** | 应用入口，提交任务图、跑执行计划 |
| **GlobalScheduler** | 进程级调度，3 档优先级、跨会话资源预约 |
| **NativeExecutionRunner** | 拆 RealDevice/Simulation；dlopen `libruntime[_camodel].so` |
| **SimBackend** | 单线程模拟器调度队列（SimulatorDispatchQueue） |
| **NpuBackend** | 真实设备路径（910B 等） |
| **AclnnBackend** | host 端 aclnn 调用 + AscendC kernel launch 发射 |
| **HostLaunchHelper** | 把 AscendC kernel launch 包成 host 调用 |
| **OutputComparator** | 数值校验（atol/rtol） |

历史上经过 2026-04-13 ~ 04-24 一轮大规模重构（参见 plan 列表）：
统一到 single-runtime CLI、删除 legacy CompatRuntime / Compiler /
Executor / SimValidator、引入 GlobalScheduler 资源模型、stream resource
model、quota / fairness baseline。**这一轮 cleanup 已基本收口**。

### 4.7 AclnnBackend (host 生成侧)

- `lib/Runtime/AclnnBackend/AclnnBackend.cpp`：发射 `network_host.cpp`
- 当前只对 `tm_tensor.attention → PromptFlashAttention` 这条 op 走 aclnn
  fallback；其他 op 直接 fail-fast（按 plan `2026-04-20-aclnn-fallback`
  的范围）。
- 最近一波修复（c8cefcc → 4c166bc → 74a2af9 → af90ff0）：
  - 通过 HostLaunchHelper 发射 AscendC launch（c8cefcc）
  - 修 host 链接用 .bin 而非 .o（4c166bc）
  - schema-order tilings + dtype guard（74a2af9）
  - 路由通过 ExecutionSession::run 保证确定性（af90ff0）
  - workspace_size = 16 MiB（d1b31a0）
- 失败兜底：aclInit 失败回落 host-mode（CPU 参考）；
  `ACL_ERROR_REPEAT_INITIALIZE` 不再误判（6f4155c）。

### 4.8 Autotuner

- `tools/autotuner/autotuner_main.cpp`（924 行，单文件 CLI）
- 工作流：tiling_space.json → 解析 SymExpr → 枚举候选 → ArtifactCompiler
  编译 → ExecutionSession 跑 → OutputComparator 对 .npy → 出
  `best_config.json`
- 最近升级：6b07b7b 把 tunable 范围从 afir-translate 取来（不再硬编码），
  并做 combo pruning；889a0ab 让动态 shape kernel 全程能跑。
- **已知小坑**（python 侧 workaround 中）：
  - `block_dim_expr` 偶发空字符串 → `pipeline._patch_tiling_space()` 兜底
    （SymExpr 已经能处理，但发射端在某些路径仍会漏写）
  - tiling 搜索范围 `[16,32,64]` 在 `python/torch/framework/pipeline.py`
    硬编码

### 4.9 Network Runner（5-phase Python orchestration）

`python/network_runner.py`（491 行），按 plan
`2026-05-13-network-runner-mixed-cpu-sim.md` **5 个 phase 全部落地**：

| Phase | 内容 | commit |
|-------|------|--------|
| 1 | Outline + emit network.json（自动或手写） | 034e935 |
| 2 | Per-kernel codegen + translate + g++ link | 01d9f84 |
| 3 | Default tilings build + 中间结果 dump | 44c970a |
| 4 | Per-kernel autotune（调既有 autotuner） | f9dd6cb |
| 5 | Best-tilings rebuild + run + numpy verify | c0cb484 |

代表性 e2e：

- **mixed-attn-e2e**（commit 2f9d400）：`pre-norm linalg → flash attention
  via aclnn → post-proj linalg`，CPU sim 通过。
- **two-elewise-e2e**（commit 99a5128）：演示自动 outline 把一个 module
  拆成 2 个独立 AscendC kernel。
- 其它纯 AscendC：`relu-e2e / reduce-*-e2e / bcast-*-e2e / transpose-*-e2e`
  等约 25 个；纯 aclnn：`aclnn-attn-e2e`。

---

## 5. 与 `docs/vector-plan/` 设计文档的偏移分析

基线：`docs/vector-plan/{00..06}-*.md`（共 4084 行，README + 8 章）。
对照点：当前 `lib/Conversion/VectorPlan/` 实现 + `include/Conversion/
VectorPlan/*.h`，以及补丁性 plan：

- `docs/superpowers/plans/2026-04-25-vector-plan-tile-fuse-collapse.md`
- `2026-04-27-tile-plan-gen-loopnest-groupemitter.md`
- `2026-04-28-vector-plan-codegen-phase-b.md`
- `2026-04-30-prepare-for-emit-refactor.md`
- `2026-05-11-port-af-scheduler-to-vector-plan.zh.md`（AF 调度器移植，
  P1 已落地未提交）
- `2026-05-12-broadcast-as-parallel-axis.zh.md`

### 5.1 章节级对照表

| 文档 | 范围 | 实现状态 | 偏移量 |
|------|------|---------|--------|
| 00-architecture | 流水线总纲 (Pass1/Outline/Pass2) | ✅ 实现已超越 | 文档不缺 |
| 00-data-model | 各阶段数据结构 | ✅ 实现+扩展 | **缺 AxisKind/Class/Grouping** |
| 01-group-analysis | Pass1 attribute 标注 | ✅ 已完成 | 文档与实现基本一致 |
| 02-group-outline | 文件分离 + JSON 导出 | ✅ 已完成且超越 | **NetworkJsonEmitter 在文档外** |
| 03-tile-fuse | Pass2 P1–P7 | ⚠️ 实现远超文档 | **P4–P7 文档仍是 TODO 占位** |
| 04-tile-info | TileInfo 数据流 | 🟡 部分实现 | Phase B 待补 |
| 05-codegen-design | 编译流水线 | 🟡 部分实现 | Phase B passes 框架已注册 |
| 06-autotuner-design | 参数搜索框架 | 🟡 设计完整、实现 ~30–40% | 已被 tools/autotuner 实物部分超过 |

### 5.2 明显偏移点（按重要性）

**(D1) 03-tile-fuse Phase 4–7 未写**
- 设计文档 Phase 4 起就是 TODO 占位。
- 实现 `TilePlanGen.cpp`（593 行）已涵盖 P3–P6，
  含 reduce 三模板 (Common/FullLoad/RCore)、transpose 两模板、
  block-axis swap、UB-peak penalty。
- 影响：新人只看文档会以为 P4+ 还没开始。
- 关键 commit：`ed8ae3d / 1beaa2d / 17962e3 / aad7ab2 / b81c9dc`。

**(D2) AxisKind / AxisClass / AxisGrouping 未在 00-data-model 中体现**
- AF scheduler 移植引入这套轴分类（Y/R/X/N 四类，后续广播轴归并为
  普通并行轴）。
- 实现：`include/Conversion/VectorPlan/GroupInfo.h`，由 `Collapse.cpp`
  计算并存进 GroupInfo。
- 关键 commit：`1064913`（搬入 GroupInfo.h）+ `54a81fd`（Collapse 拥有
  分类）+ `704475d`（broadcast 归一）+ `037c0e7`（e2e gates + lit）。

**(D3) Collapse 的 Case A/B2/C 分类**
- 03 文档只有抽象的"广播轴剪枝 + BAII L1/L2"概念。
- 实现按 plan `2026-04-25-vector-plan-tile-fuse-collapse.md` 精化为
  Case A / B2 / C 分类，`Collapse.cpp` 内有具体伪代码对应。

**(D4) GroupEmitter 的广播轴 LICM 实现细节**
- 03 §"BAII L2"只说"GroupEmitter 主动把缺轴 Load 上浮"。
- 实际 `GroupEmitter::emitOperand()` 的判定/上浮逻辑跟文档抽象描述
  存在微妙差异，需要补充。

**(D5) Phase B 落点：PrepareForEmit 拆 3 个 pass**
- 04 / 05 文档规划"通过 tiling.infos 走新路径"。
- 补充设计 `2026-04-30-prepare-for-emit-refactor.md` 拆为三段：
  - `AscendCFlattenGMPtrPass`
  - `PackTilingDataPass`
  - `FinalizeKernelPass`
- 实现：pass 框架已注册，**实现细节未完成，仍走旧 path**。

**(D6) 02 文档外的 NetworkJsonEmitter**
- 02-group-outline 没提 network.json，但实现已稳定输出（commit
  `62ad017 / f6b9121 / 5cdf4c8`），且作为 mixed runner 的契约。

**(D7) 06-autotuner-design 描述仍偏理论**
- 实际 `tools/autotuner/autotuner_main.cpp` 已是可用产物，且
  network-runner 把它当 sub-process 调用；文档的 Solver/Runner 抽象
  没在代码里完全建立对应。

### 5.3 文档已过时但代码已稳定的清单

需要在下一轮文档维护时回写：

1. 03 文档 Phase 4–7 章节（用 P3/P4/P5/P6 commit 串重写）
2. 00-data-model 增章节："AxisKind / AxisClass / AxisGrouping"
3. 02 文档增节："network.json schema"
4. 04/05 文档明确标注：Phase B = 三 pass 拆分（FlattenGMPtr /
   PackTilingData / FinalizeKernel）框架已就位，实现待推进
5. 06 文档对照 `tools/autotuner/autotuner_main.cpp` 重新校准

### 5.4 工程债务（实现"半成"）

| 项 | 状态 | 备注 |
|---|---|---|
| `AscendCFlattenGMPtrPass` 实现 | 框架在、逻辑 stub | Phase B 第一段 |
| `PackTilingDataPass` 实现 | 框架在、逻辑 stub | Phase B 第二段 |
| `FinalizeKernelPass` 实现 | 框架在、逻辑 stub | Phase B 第三段 |
| AF scheduler 移植 P2+ | P1 done 未提交 | 见记忆 [[project_af_scheduler_port]] |
| AutoTuner Solver/Runner 与 06 文档对齐 | tools/ 实物已超 | 文档需重写 |

---

## 6. 最近里程碑（按 commit 串）

按时间倒序，截到 `190c814`（2026-05-14）：

```
190c814 docs(notes): R3 reduce-codegen-status 标记为 fixed
28c8ea6 fix(vector-plan): IsolateKernelOutputs 修 R3（DPS init aliasing）
1f12c5b docs(notes): reduce-path 通过 network runner 的状态登记
99a5128 feat: two-elewise-e2e exercises auto-outline → 2 kernels
af90ff0 fix(host-launch): 路由通过 ExecutionSession::run 保确定性
d1b31a0 fix(host-launch): workspace_size = 16 MiB
79cc793 test: phase3-5 测试保持 small static shape (camodel flake)
889a0ab fix(host-launch+autotuner): dynamic-shape kernel e2e
2f9d400 feat: mixed-attn-e2e end-to-end on CPU sim ★
c0cb484 feat(network-runner): phase 5 best-tilings build/run/verify
f9dd6cb feat(network-runner): phase 4 per-kernel autotune
44c970a feat(network-runner): phase 3 default-tilings build + dump
01d9f84 feat(network-runner): phase 2 codegen+translate+compile
034e935 feat(network-runner): phase 1 outline / emit-network-json
22f06cf fix(linalg-to-ascendc): 不再静默丢未支持 op
998edb7 fix(linalg-to-ascendc): select+cmpf → maximumf/minimumf
b81c9dc feat(vector-plan): P6 — UB-peak penalty
44f509f feat(autotuner): C4 — evalBlockDimExpr 完整 SymExpr
6d81d9c feat(ascendc): C3 — TilingData dim 字段去重
dd4593c feat(vector-plan): C2 — block_dim_expr SymExpr 落 json
6d720ba feat(vector-plan): C1c — symbolic axis 透传 tile-fuse
8e07d98 feat(afir): C1 — afir.iter_extents
c862fbe feat(afir): linalg-level shape symbolization
704475d feat(vector-plan): B-1 — broadcast = ordinary parallel
17962e3 feat(vector-plan): P5c — ubY 枚举 + PruneTilingCase
1beaa2d feat(vector-plan): P5b — reduce-axis tiling-case 枚举
ed8ae3d refactor(vector-plan): P5a — enumerate/build/cost/pick 骨架
1064913 refactor(vector-plan): AxisKind 等迁入 GroupInfo.h
0e617c3 feat(vector-plan): P4 — strided GM store for transpose
```

### 6.1 当前已稳定的能力

- 静态 + 动态 shape 都能从 torch.export 走到 .bin，自动 outline 多 kernel
- aclnn op 与 AscendC kernel 在同一 host program 中编排
- SymExpr 一套贯穿 vector-plan、tiling space、autotuner 求值
- 5-phase network-runner 在 CPU camodel 上端到端
- relu (select+cmpf) / reduce / broadcast / transpose 等都有 e2e 例子

### 6.2 最近修掉的 bug

- relu select-pattern 静默丢失（998edb7 + 22f06cf）
- block_dim_expr 空表达式 crash（44f509f）
- DPS init aliasing 导致 reduce 输出脏（28c8ea6，对应 R3）
- host-launch workspace 太小（d1b31a0）
- aclnn 调用与 AscendC launch 顺序 / 链接错误一连串（c8cefcc → af90ff0）

---

## 7. 已知未解决的坑（latent）

| # | 现象 | 范围 | 优先级 |
|---|------|------|--------|
| 1 | **kg1-hang as 2nd launch**：mixed runner 第二次 kernel launch 偶发挂死（[[project_network_runner_v1]]） | runtime / camodel | 高 |
| 2 | **camodel 在大 tensor 下非确定性**（79cc793 workaround 里说明） | 测试稳定性 | 中 |
| 3 | **Reduce path R1 / R2 / R4 / R5** 仍未修（[[project_reduce_path_r3]]） | LinalgToAscendC + vector-plan | 高 |
| 4 | **`block_dim_expr` 偶发空字符串**：python 侧 `_patch_tiling_space()` 兜底；根因在某条 emit 路径 | afir-translate | 中 |
| 5 | **Tiling 搜索范围 `[16,32,64]` 硬编码**（python/torch/framework/pipeline.py） | autotuner 入参 | 低 |
| 6 | **Multi-output 验证缺陷**：autotuner 仅验 `expected_0.npy` | autotuner | 低 |
| 7 | **Reduce 维度 tiling 未实现**：当前仅 parallel 维度 tile，沿 reduce 轴 split 还没做 | vector-plan TilePlanGen | 中 |
| 8 | **`cf.assert` 不能 codegen**：靠 EliminateCfAssert 提前删除，动态 broadcast 检查会丢 | EliminateCfAssert / target | 低 |
| 9 | **AclnnLoweringPass 仅支持 attention**：其他 op fail-fast | Runtime/AclnnBackend | 中 |
| 10 | **AF scheduler P2+ 未推**：P1 已落地未提交 | vector-plan | 高（影响调度质量） |

---

## 8. 当前开发进度小结

按"compiler"和"runtime"两条主线分别看：

### Compiler 线（vector-plan + AscendC）

- ✅ 完成：Pass1 / Outline / TileFuse 骨架 / SymExpr 全链路 / network.json
- ✅ 完成：reduce 三模板 / transpose 两模板 / broadcast 归一 / UB-peak 剪枝
- 🟡 推进中：Phase B（FlattenGMPtr + PackTilingData + FinalizeKernel）
- 🟡 推进中：AF scheduler 移植（P1 done，P2+ 排队）
- ⏳ 未开始：Reduce 维度 tiling、aclnn 模式识别 (deferred)

### Runtime / Tooling 线

- ✅ 完成：runtime 大重构 + single-runtime CLI + GlobalScheduler
- ✅ 完成：autotuner 走 ExecutionSession + dynamic shape
- ✅ 完成：network-runner 1–5 phase 全跑通
- ✅ 完成：mixed-attn-e2e on CPU sim
- 🟡 推进中：AclnnBackend op 白名单扩展（目前只有 attention）
- ⏳ 未开始：真实 NPU 设备验证（NpuBackend 路径有，端到端尚未跑）

### 文档线

- ✅ 完成：plan / 滚动设计文档非常充分（`docs/superpowers/plans/`
  下 60+ 篇）
- ❌ 落后：`docs/vector-plan/00-06` 未追平实现（详见 §5）
- ❌ 落后：`docs/torch-e2e-pipeline.md` 未反映 SymExpr 通路

---

## 9. 建议下一步

按"低风险 / 高 ROI" 排序：

1. **修 kg1-hang 与 reduce R1/R2/R4/R5**：当前阻塞 mixed AscendC + aclnn
   网络真正向上 stack 的最大风险点。
2. **推 Phase B 三 pass 的实现**：去掉旧 `prepare-for-emit` 的特例分支，
   把 04/05 文档真正"装上"。
3. **AF scheduler P2 提交并推进 P3**：现在 P1 在 worktree，长期不提交容易
   bit-rot。
4. **回写 `docs/vector-plan/03-tile-fuse.md` Phase 4-7 章节**：以现有
   commit 串和 `TilePlanGen.cpp` 注释为蓝本。
5. **AclnnBackend 白名单扩展**：至少把 softmax/layernorm 走起来（与
   `[[project_aclnn_pattern_matching]]` 计划合并）。
6. **NPU 真机一次完整跑通**：NativeExecutionRunner 路径完备，缺一次端到端
   验证 + 文档。

---

*本文档为 2026-05-14 时点快照；后续若 vector-plan 03 文档被回写或 Phase B
推进，请同步更新 §5 / §8。引用 commit / 文件路径在写作时已校验存在；
若分支演进，请以最新 `git log` 为准。*
