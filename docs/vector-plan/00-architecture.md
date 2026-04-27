# Vector Plan Generation — 架构总纲

**Date:** 2026-04-23  
**Status:** Architecture Overview (精简自 00-unified-design.md)  
**Scope:** 流水线架构、融合规则概要、接口边界、迁移路径

> [!NOTE]
> 本文档是架构级概述，不包含 C++ 实现代码。各阶段的具体实现详见 `01~06` 模块文档。

---

## 1. 总体目标

在 Ascend-MLIR 中落地一个自洽、可验证、可扩展的 plan-generation pass 流水线：
把经过预处理的 linalg-on-tensor 程序自动生成 tiled+fused 的 Ascend NPU kernel，
无需手写 transform 脚本。

Pass 1 的分析产物是**受限 DAG 子图（Group）**，而非"单根链"。Group 描述一组可以共享
同一 loop nest 执行的 linalg op 子图；"chain" 仅在非正式叙述中保留，作为最常见形态
（生产者-消费者垂直链）的简称。

支持的 group 类型：
- **VectorGroup**：由纯 linalg Vector op 组成的融合子图，覆盖 reduce→pointwise、
  reduce→reduce 链式、broadcast/keepdim、同输入并行 sibling（horizontal fusion）、
  LayerNorm、Softmax 等场景
- **CubeGroup**：以 linalg.matmul 为锚点，吸收 epilogue VectorGroup 和 prologue op，
  覆盖 QKV projection、FFN linear 等 matmul+epilogue 场景

**Phase 1 约束**：
- VectorGroup 支持 Vertical + Horizontal fusion（含 reduction op 的 group 可参与水平融合）
- CubeGroup epilogue 只允许吸收已成型的全并行 VectorGroup（保守）；
  支持吸收含 horizontal 成员的复杂 VectorGroup 留待 Phase 3

---

## 2. 三段流水线架构

整个 plan generation 拆成三段，Pass 1 与 Outline Pass 通过 IR attribute 通信，
Outline Pass 与 Pass 2 通过 func 边界（标准 MLIR func pass）通信：

**Pass 1: vector-plan-group-analysis** (func-level)
- 输入：linalg-on-tensor func
- 算法：迭代融合（§3）
- 输出：每个 linalg op 上的 group annotation（`group_id` / `topo_index`）
- 详细实现 → [01-group-analysis.md](./01-group-analysis.md)

**Outline Pass: vector-plan-group-outline** (module-level)
- 输入：带 group annotation 的 func
- 算法：分桶 + 两级拓扑排序 + func outlining + 文件分离
- 输出：`network.mlir`（coordinator）+ `kernel_group{N}.mlir`（每个 group 一个文件）
- 详细实现 → [02-group-outline.md](./02-group-outline.md)

**Pass 2: vector-plan-tile-fuse** (func-level, per kernel file)
- 输入：`kernel_group{N}.mlir`
- 算法：Collapse → TilePlan 生成 → Loop Nest 生成
- 输出：tiled+fused kernel func + `tiling.infos`
- 详细实现 → [03-tile-fuse.md](./03-tile-fuse.md), [04-tile-info.md](./04-tile-info.md)

**三段分离的原因**：
- Pass 1（图算法）与 Pass 2（IR 变换）解耦，各自可独立测试
- Outline Pass 把"哪些 op 属于哪个 kernel"物化为 func 边界，Pass 2 无需感知 group_id
- 每个 kernel func 是独立编译单元，Pass 2 可并行处理

**通信方式选择**：Pass 1 只通过 IR attribute 通信，不用 MLIR Analysis 框架。
原因：Analysis 在 IR 被修改时自动 invalidate，不适合"Pass 1 写、Outline Pass 读"的
生产者-消费者模式；attribute 存活于序列化/反序列化，中间 IR 可直接 FileCheck。

### 2.1 Pass 1 → Outline Pass：attribute schema

`vector_plan.*` attribute 的生命周期完全局限于 **Pass 1 内部**：
服务于迭代融合算法，由 Outline Pass 消费后全部 strip，不进入任何 kernel 文件。

Pass 1 只写两个 per-op attribute：

| attribute | 类型 | 消费方 | 作用 |
|-----------|------|--------|------|
| `vector_plan.group_id` | `i32` | Outline Pass | 决定分桶 |
| `vector_plan.topo_index` | `i32` | Outline Pass | 全局拓扑位置序号 |

### 2.2 Outline Pass 产出示例

```mlir
// network.mlir
module {
  func.func private @kernel_group0(%arg0: tensor<?xf16>, ...) -> tensor<?x?xf16>
  func.func @network(%in0, %in1) -> tensor<?x?xf16> {
    %r0 = call @kernel_group0(%in0, %in1)
    return %r0
  }
}

// kernel_group0.mlir — 干净的 linalg-on-tensor，无 vector_plan.* 属性
module {
  func.func @kernel_group0(%arg0: tensor<?xf16>, ...) -> tensor<?x?xf16> {
    %0 = linalg.generic { ... }
    return %0
  }
}
```

### 2.3 Pass 2 内部三阶段

| 阶段 | 内容 | VectorGroup | CubeGroup |
|------|------|-------------|-----------|
| Collapse | 合并 G-axes | 广播轴剪枝 + A/B2/C 分类 | 跳过（恒等映射）|
| TilePlan | 推导切分参数 | XBLOCK/XBLOCK_SUB/RBLOCK | BM/BN/Tb_M/Tb_N/t_K |
| Loop Nest | 建 scf.for + 发射 | 统一 collapsed loop + 拓扑序发射 | 层次化 BM→BN→Tb_M→Tb_N→t_K |

---

## 3. Group 识别：融合规则概要

详细实现 → [01-group-analysis.md](./01-group-analysis.md)

### 3.1 迭代融合框架

与 Inductor 的 `fuse_nodes` 对齐，采用**多轮迭代 pairwise 融合**。
每个 linalg op 初始为独立的单成员 Group；
非 linalg op 不进入任何 Group，天然成为融合边界。

**Priority**：
- VectorGroup + VectorGroup：DEFAULT（先处理）
- CubeGroup + VectorGroup：LOW（所有 Vector fusion 收敛后才运行）

**Score**：两 group 之间共享 tensor 的字节数。

### 3.2 轴类上确界规则

```
AxisClass(axis a) = ⊔{member m ∈ G} class_of(m, a)

join(Parallel,  Parallel)  = Parallel
join(Parallel,  Reduction) = Reduction    
join(Parallel,  Absent)    = Parallel     
join(Reduction, Absent)    = Reduction
```

- **Tileable 轴** = 所有 class == Parallel 的轴
- **Reduction 轴** = 所有 class == Reduction 的轴

### 3.3 VectorGroup can_fuse 规则清单

| # | 规则 | 类型 |
|---|------|------|
| 1 | FusionKind 确定（Vertical / Horizontal / None） | 公共 |
| 2 | 合并后无环（cycle check） | 公共 |
| 3 | 轴类相容（merged_Tileable 非空） | 公共 |
| 4 | 链接 tensor 的所有 Vector-track 消费者都在 g1 或 g2 内 | Vertical |
| 5 | DPS init 透明（linalg.fill 按普通成员纳入） | 公共 |
| 6 | Epilogue 不依赖 reduction partial sum | 含 reduction 时 |
| H1 | Tileable 轴并集非空 | Horizontal |
| H2 | 新增 boundary input ≤ maxHorizontalExtraInputs | Horizontal |

### 3.4 CubeGroup can_fuse 规则清单

**Epilogue（CubeGroup ← VectorGroup）**：全 parallel、无 alias、affine projective map、
无 view node、单消费者、非纯 shape op。

**Prologue（VectorGroup → CubeGroup）**：全 parallel、无 alias、单消费者、
affine projective for A/B。

### 3.5 Reduction Splitting（可选）

`enableReductionSplit=true` 时，RBLOCK 从 `Full` 变为 `Inner`（串行 split）。
epilogue 成员数量上限由 `maxReduceEpilogueOps`（默认 3）控制。

---

## 4. Collapse 概要

详细实现 → [03-tile-fuse.md](./03-tile-fuse.md) Phase 1

**目的**：把 group 内部的多根维度合并成单根，让生成的 loop nest 尽量扁平。
仅作用于 **VectorGroup**（CubeGroup 跳过）。

### 4.1 Per-input 分类

| 情况 | 条件 | 含义 |
|------|------|------|
| **A** | `G ∩ result(map) = ∅` | 整组轴被 broadcast 掉（纯外层） |
| **C** | `G ⊆ result(map)` 且连续同序 | 可直接折叠 |
| **B2** | `G ⊆ result(map)` 但乱序/不连续 | 硬边界，生成两个 Variant |

> **注**：原 B1 分类（部分轴缺失=隐式广播）已被 BAII（Broadcast Axis Independence Invariant）取代。在分类前，广播轴会从 G 中被剪枝并置于外层循环，不参与 collapse。详见 [03-tile-fuse.md](./03-tile-fuse.md)。

### 4.2 处理流程

1. **Pre-check**：提取广播轴 BCast(G)；对 G 剪枝得到非广播子组 G'；对 G' 做 A/B2/C 分类
2. **B2**：生成 Variant 1（插 `linalg.transpose`，no_collapse barrier）和 Variant 2（改 consumer map + `load_with_transpose`），由 Autotuner 选优
3. 对所有成员调 `linalg::collapseOpIterationDims`（仅对 G'）

### 4.3 典型场景

| 场景 | 输入 Shape | Collapse 结果 |
|------|-----------|---------------|
| LayerNorm | `[B, S, H]` | `[B*S, H]`，scale/bias → Case A |
| Softmax | `[B, H, S, S]` | `[B*H*S, S]` |
| Transpose+PW | `[B, S, H]` + input `[H,S]` | Case B2 → 2 Variants |
| Horizontal sibling | `add1[B,S,H]` + `add2[B,S]` | BAII: d_H 提取为广播轴，不 collapse |

---

## 5. Loop Nest 生成策略

详细实现 → [03-tile-fuse.md](./03-tile-fuse.md) Phase 3–7

Pass 2 **不复用** `linalg::tileUsingForOp` / `tileAndFuseProducerOfSlice`，
而是自行建 loop nest，按拓扑序**统一发射所有成员 op**。

原因：tile-and-fuse API 无法处理 horizontal sibling（无直接 SSA 边但共享迭代空间）。

### 5.1 三个核心组件

| 组件 | 职责 |
|------|------|
| **LoopNestBuilder** | 从 TilePlan 建 scf.for 嵌套，产出 `loop_ivs` 映射 |
| **SliceComputer** | 从 indexing_map + loop_ivs 推导 extract_slice offset/sizes |
| **GroupEmitter** | 按 topoMembers 拓扑序发射：boundary → extract_slice，interior → tiled op，output → insert_slice |

### 5.2 VectorGroup vs CubeGroup 对比

| | VectorGroup | CubeGroup |
|---|---|---|
| loop nest | 统一 collapsed scf.for | BM/BN → Tb_M/Tb_N → t_K |
| horizontal fusion | 共享 loop body，天然支持 | 不涉及 |
| reduction | `enableReductionSplit` 建 RBLOCK inner loop | t_K 始终 Inner |
| Accumulator Pattern | RBLOCK loop 前 init，loop 内累加，loop 后 epilogue | t_K loop 前 init，loop 内 matmul，loop 后 epilogue |

---

## 6. TileInfo 数据模型

详细实现 → [04-tile-info.md](./04-tile-info.md)

TileInfo 是 TilePlan 与 TilingData / AutoTuner 之间的**稳定可序列化中间层**：

```
TilePlan（含 MLIR Value/OpFoldResult）
  → TileInfo（稳定序列化对象，无 MLIR 内部指针）
      → tiling.infos（module-level attribute，权威表示）
      → PrepareForEmit → TilingData
      → AutoTuner → tiling_func.cpp
```

### 6.1 TileLevel × AxisRole 矩阵

```
                  AxisRole=Parallel          AxisRole=Reduction
TileLevel=Outer   XBLOCK / BM / BN          —
TileLevel=Inner   XBLOCK_SUB / Tb_M / Tb_N  BK / RBLOCK_sub
TileLevel=Full    —                          RBLOCK (v1 不切)
```

### 6.2 TileInfo 不变量

1. 每个 `TileInfo` 只对应一个 `(groupId, planId)`
2. `abiIndex` 是 packable 字段的唯一顺序权威；Derived 字段 `abiIndex` 必须为 nullopt
3. `ShapeDim` 只引用原始 tensor 参数的 primitive 维度
4. TileInfo 只持有 `ValueExpr`，不持有 SSA Value
5. AutoTuner 只搜索 `TunableTile`
6. `blockDimExprs` 必须显式表达（不反推）

---

## 7. Codegen 流水线

详细实现 → [05-codegen-design.md](./05-codegen-design.md)

### 7.1 接口契约：Pass 2 → Codegen

| 接口 | 来源 | 用途 |
|------|------|------|
| `tiling.infos` | module attribute | 唯一结构化接口，含 fields/axes/block_dim |
| `ascendc.parallel` | scf.for attribute | 标记 dispatch 循环（句柄，不决定维度数） |
| `ascendc.prologue` | scf.for attribute | CubeGroup K loop data-move 标注 |
| `ascendc.unit` | linalg op attribute | 计算单元（Cube / Vector） |

### 7.2 N-D Dispatch

dispatch 维度由 `tiling.infos.block_dim` 决定：
- VectorGroup：N=1，退化为现有行为
- CubeGroup：N=2，生成 `block_idx / grid_n` + `block_idx % grid_n`

---

## 8. 接口边界

### 8.1 TilePlan → TileInfo

语义降维：把含 MLIR Value 的编译器内部对象转成稳定可序列化对象。

### 8.2 TileInfo → TilingData

ABI 物化：PrepareForEmit 按 `abiIndex` 构造 `TilingData` struct，不从 func args 反推。

### 8.3 TileInfo → AutoTuner

搜索投影：只暴露 `TunableTile` 的 candidates，AutoTuner 不从 loop 结构反推 block_dim。

---

## 9. MLIR 表示

### 9.1 Pass Options

```tablegen
def VectorPlanGroupAnalysis : Pass<"vector-plan-group-analysis", "func::FuncOp"> {
  let options = [
    Option<"enableReductionSplit", "enable-reduction-split", "bool", "false",
           "Make RBLOCK tunable (Inner) instead of fixed full">,
    Option<"maxReduceEpilogueOps", "max-reduce-epilogue-ops", "int32_t", "3",
           "Max pointwise ops after a reduction when enableReductionSplit=true">,
    Option<"maxHorizontalExtraInputs", "max-horizontal-extra-inputs", "int32_t", "4",
           "Max additional boundary inputs allowed when fusing horizontal pairs">,
  ];
}

def VectorPlanGroupOutline : Pass<"vector-plan-group-outline", "ModuleOp"> {
  let options = [
    Option<"kernelFuncPrefix", "kernel-func-prefix", "std::string",
           "\"kernel_group\"", "Prefix for outlined kernel func names">,
  ];
}

def VectorPlanTileFuse : Pass<"vector-plan-tile-fuse", "func::FuncOp"> {
  let options = [
    Option<"enableCollapse", "enable-collapse", "bool", "true", "">,
    Option<"enforceCompletePlan", "enforce-complete-plan", "bool", "true", "">,
    Option<"hwMatmulMAlign", "hw-matmul-m-align", "int64_t", "16", "">,
    Option<"hwMatmulNAlign", "hw-matmul-n-align", "int64_t", "16", "">,
    Option<"hwMatmulKAlign", "hw-matmul-k-align", "int64_t", "16", "">,
    Option<"l0aCapacityBytes", "l0a-capacity-bytes", "int64_t", "65536", "">,
  ];
}
```

### 9.2 tiling.infos 序列化示例

**VectorGroup**（LayerNorm `[B*S, H]`）：

```mlir
module attributes {
  tiling.infos = [{
    kernel = "group0_plan0",
    group = 0 : i64, plan = 0 : i64,
    axes = [
      {axis = 0 : i64, name = "BS", role = "parallel",
       extent = {op = "mul",
                 lhs = {op = "shape_dim", arg = 0, dim = 0},
                 rhs = {op = "shape_dim", arg = 0, dim = 1}}},
      {axis = 1 : i64, name = "H", role = "reduction",
       extent = {op = "shape_dim", arg = 0, dim = 2}}
    ],
    fields = [
      {id = "tile.xblock", abi_name = "XBLOCK", abi_index = 0, kind = "tunable",
       axis = 0, level = "outer",
       default = {op = "const", value = 256},
       search = {candidates = [64, 128, 256]}},
      ...
    ],
    block_dim = [{op = "ceildiv",
                  lhs = {op = "mul", ...},
                  rhs = {op = "field_ref", id = "tile.xblock"}}]
  }]
}
```

**CubeGroup**（matmul + epilogue）：block_dim 有 2 个元素（grid_y, grid_x）。
完整示例参见 [04-tile-info.md](./04-tile-info.md)。

---

## 10. 迁移路径

| 阶段 | 内容 | 兼容性 |
|------|------|--------|
| Phase A | 引入 TileInfo，Pass 2 写 `tiling.infos` | runtime ABI 不变 |
| Phase B | PrepareForEmit 改为消费 TileInfo | 不再猜字段顺序 |
| Phase C | AutoTuner 改为消费 TileInfo；hard-fail 若缺失 | 删除旧反推路径 |
| Phase D | `tiling.tiles` / `tiling.shapes` 降级为兼容视图 | — |

**现有测试兼容性**：
- Phase 1 VectorGroup 测试完全不变
- `tiling.tiles` / `tiling.shapes` 不变
- AutoTuner 读 `search.candidates` 逻辑不变；新增 `alignment` 过滤
- PrepareForEmit 通过 `TileLevel=Outer` 个数判断 1D/2D dispatch

---

## 11. 数据流总览

```
Source linalg IR
  │
  ├─ [Pass 1] vector-plan-group-analysis
  │   → group_id / topo_index annotations
  │
  ├─ [Outline Pass] vector-plan-group-outline
  │   → network.mlir + kernel_group{N}.mlir
  │
  ├─ [Pass 2] vector-plan-tile-fuse (per kernel, 可并行)
  │   ├─ Collapse → CollapsedGroupInfo
  │   ├─ TilePlan Generation → TilePlan
  │   ├─ Loop Nest 生成
  │   └─ Module Metadata → tiling.infos
  │
  └─ Codegen Pipeline
       → bufferize → buffer-placement → linalg-to-ascendc
       → ascendc-parallelize → ascendc-prepare-for-emit
       → afir-translate → kernel_group{N}.cpp
```
