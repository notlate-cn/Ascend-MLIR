# Vector Plan Generation — 统一设计方案

**Date:** 2026-04-14  
**Status:** Draft for review  
**Supersedes:**
- `docs/superpowers/plans/2026-04-10-vector-plan-generation-final.md`
- `docs/superpowers/specs/2026-04-14-cube-chain-reduction-split-design.md`
- `docs/superpowers/specs/2026-04-14-tile-info-design.md`

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

**Pass 1: vector-plan-group-analysis**

- 输入：linalg-on-tensor func
- 算法：迭代融合（§3）
- 输出：每个 linalg op 上的 group annotation（group_id / is_anchor / topo_index）

[中间 IR 可 dump、可 FileCheck group annotation]

**Outline Pass: vector-plan-group-outline**
- 输入：带 group annotation 的 func
- 算法：分桶 + 两级拓扑排序 + func outlining + 文件分离（§2.2）
- 输出：
  - network.mlir（coordinator func + kernel func 声明） 
  - kernel_group{N}.mlir（每个 group 一个文件，含 kernel func 定义）

[每个文件独立可 dump、可 FileCheck]

**Pass 2: vector-plan-tile-fuse**

func-level pass，对每个 kernel_group{N}.mlir 独立运行
- 输入：kernel_group{N}.mlir
- 算法：
  1. Collapse（§4）：将 G-axes 合并为更少维度，产出 CollapsedGroupInfo
  2. TilePlan 生成（§6.5）：从 CollapsedGroupInfo 推导切分参数
  3. Loop Nest 生成（§5）：自行建 scf.for loop nest，按 topoMembers 拓扑序，统一发射所有成员 op（不复用 tile-and-fuse 基础设施）
- 输出：tiled+fused kernel func + tiling.infos

**三段分离的原因**：
- Pass 1（图算法）与 Pass 2（IR 变换）解耦，各自可独立测试
- Outline Pass 把"哪些 op 属于哪个 kernel"物化为 func 边界，Pass 2 无需感知 group_id
- 每个 kernel func 是独立编译单元，Pass 2 可并行处理，最终自然对应一个 `AscendC.cpp`

**通信方式选择**：Pass 1 只通过 IR attribute 通信，不用 MLIR Analysis 框架。
原因：Analysis 在 IR 被修改时自动 invalidate，不适合"Pass 1 写、Outline Pass 读"的
生产者-消费者模式；attribute 存活于序列化/反序列化，中间 IR 可直接 FileCheck。

### 2.1 Pass 1 → Outline Pass：attribute schema

`vector_plan.*` attribute 的生命周期完全局限于 **Pass 1 内部**：
服务于迭代融合算法（多轮 pairwise 融合、group 归属跟踪），
由 Outline Pass 消费后全部 strip，不进入任何 kernel 文件，对 Pass 2 不可见。

Pass 1 只写三个 per-op attribute，不写 module-level 元数据：

```mlir
%0 = linalg.reduce { ... }
     {vector_plan.group_id = 0 : i32,
      vector_plan.topo_index = 2 : i32} ins(...) outs(...)

%1 = linalg.generic { ... }
     {vector_plan.group_id = 0 : i32,
      vector_plan.topo_index = 3 : i32} ins(...) outs(...)
```

| attribute | 类型 | 消费方 | 作用 |
|-----------|------|--------|------|
| `vector_plan.group_id` | `i32` | Outline Pass | 决定分桶，游离 op 分配唯一 singleton group_id |
| `vector_plan.topo_index` | `i32` | Outline Pass | 全局拓扑位置序号，用于排序 |

**两个 attribute 均由 Outline Pass 消费，写入 kernel 文件时全部 strip。**
Pass 2 的输入（`kernel_group{N}.mlir`）不含任何 `vector_plan.*` attribute。

GroupInfo 由 Outline Pass 在写文件前从 IR 重建：
- `kind`：检查组内是否含 matmul op
- `canonicalAxes`：对组内所有 op 的 `iterator_types` 按上确界规则（§3.2）合并推导
- `boundaryIn / boundaryOut`：遍历组内 op 的 def-use

**Pass 2 的轴分析**：kernel 文件里只有一个 group 的 op，Pass 2 直接对 func 内全部
linalg op 重跑上确界推导，得到相同的 canonical axes，无需读取任何 attribute。

### 2.2 Outline Pass：分桶、拓扑排序与 func outlining

Outline Pass 是 module-level pass，完整步骤：

```
Step 1：分桶
  for op in func（按程序序扫描）:
      groups[op.group_id].push(op)
  // 未标注 group_id 的非 linalg op 不进任何 group

Step 2：组内拓扑序
  for g in groups:
      g.topoMembers = sort(g.ops, by topo_index)
  // topo_index 直接排序，无需重新遍历 SSA 边

Step 3：组间拓扑序
  representative(g) = min(topo_index of g.members)
  sorted_groups = sort(groups, by representative)
  // 全局拓扑序保证：若 Group A 的某 op 是 Group B 的 producer，
  // 则 A 的所有 topo_index < B 的该 op 的 topo_index，
  // 因此 representative(A) < representative(B)，顺序正确

Step 4：重建 GroupInfo
  for g in sorted_groups:
      g.kind    = hasMatmul(g.topoMembers) ? Cube : Vector
      g.canonicalAxes = ⊔{m ∈ g.topoMembers} m.iterator_types   // 上确界规则（§3.2）
      g.boundaryIn  = {values used by g but defined outside g}
      g.boundaryOut = {values defined by g and used outside g}
  // canonical axes 由上确界推导，Pass 2 直接从 func 内所有 op 重跑，无需任何 attribute

Step 5：Outline 每个 group 为独立 kernel func
  for g in sorted_groups:
      @kernel_group{g.id}(g.boundaryIn...) -> (g.boundaryOut...)
      // 把 g.topoMembers 从原 func 中提取，SSA 值→ func 参数/返回值

Step 6：构建 coordinator func
  // 原 func 保留，清空 group 内 op，改为顺序调用
  // 每个 kernel func 以 func.func private 声明（外部引用）
  for g in sorted_groups:
      results = call @kernel_group{g.id}(g.boundaryIn...)
  return final_result

Step 7：文件分离 + attribute strip（split pass）
  // kernel_group{N}.mlir 写出时，strip 所有 vector_plan.* attribute
  // Pass 2 的输入是干净的 linalg-on-tensor func，无任何 annotation
  //
  // network.mlir：coordinator func + 所有 kernel func 的 private 声明
  // kernel_group{N}.mlir：每个 group 一个文件，只含该 kernel func 定义
  // 两步分离：Outline Pass 先完成纯 IR 变换（Steps 1-6），
  //           split pass 再写文件，中间态仍可 dump/FileCheck
```

Outline 产出示例：

```mlir
// network.mlir：coordinator，持有 kernel func 声明
module {
  func.func private @kernel_group0(%arg0: tensor<?xf16>, ...) -> tensor<?x?xf16>
  func.func private @kernel_group1(%arg0: tensor<?x?xf16>, ...) -> tensor<?x?xf16>

  func.func @network(%in0, %in1, %in2) -> tensor<?x?xf16> {
    %r0 = call @kernel_group0(%in0, %in1)   // group 0
    %r1 = call @kernel_group1(%in2, %r0)    // group 1，依赖 group 0 输出
    return %r1
  }
}
```

```mlir
// kernel_group0.mlir：独立编译单元，走 Pass 2 tile-fuse
module {
  func.func @kernel_group0(%arg0: tensor<?xf16>, ...) -> tensor<?x?xf16> {
    %0 = linalg.generic { ... }
    return %0
  }
}
```

```mlir
// kernel_group1.mlir
module {
  func.func @kernel_group1(%arg0: tensor<?x?xf16>, ...) -> tensor<?x?xf16> {
    %0 = linalg.matmul { ... }
    return %0
  }
}
```

### 2.3 Pass 2：Tile-Fuse 内部流程

Pass 2 是 func-level pass，对每个 `kernel_group{N}.mlir` 独立运行，内部分三个阶段：

**Stage 1：Collapse（§4）**
- 对 func 内所有 linalg op 重跑上确界推导，得到 GroupInfo
- 对 VectorGroup：执行 per-input 分类（Case A/B1/B2/C），合并 G-axes → CollapsedGroupInfo（含 collapsedAxes、B2 variant 标记）
- 对 CubeGroup：跳过（恒等映射）

**Stage 2：TilePlan 生成（§6.5）**
- 从 CollapsedGroupInfo 推导切分参数（tile sizes、loop order）
- VectorGroup：生成 XBLOCK / XBLOCK_SUB / RBLOCK
- CubeGroup：生成 BM / BN / Tb_M / Tb_N / t_K
- 若 B2 存在：对同一 group 生成两份 TilePlan（Variant 1 / Variant 2）

**Stage 3：Loop Nest 生成（§5）**
- 对每个 (CollapsedGroupInfo, TilePlan) 对：
  - LoopNestBuilder 建 scf.for loop nest
  - GroupEmitter 按 topoMembers 拓扑序统一发射所有成员 op
    - VectorGroup：普通 collapsed loop，或含 transpose 时走原始 G-axes loop
    - CubeGroup：epilogue → matmul 层次化 loop nest
- 写出 tiling.infos（含 axes / fields / blockDimExprs）

**VectorGroup 与 CubeGroup 的 Loop Nest 生成路径不同**：

| | VectorGroup | CubeGroup |
|---|---|---|
| loop nest | 统一 collapsed scf.for，按拓扑序发射所有 op | 层次化：BM/BN → Tb_M/Tb_N → t_K loop 包裹 matmul |
| horizontal fusion | 共享同一 loop body，天然支持 | 不涉及 |
| reduction | 视 `enableReductionSplit` 建 RBLOCK inner loop | t_K 始终为 Inner |
| transpose | B2 Variant 1/2 分别处理（§5.4） | 不涉及 |

---

## 3. Group 识别：迭代融合算法

### 3.1 总体框架

与 Inductor 的 `fuse_nodes` 对齐，采用**多轮迭代 pairwise 融合**。
每个 linalg op 初始是独立的单成员 Group（对应 Inductor 的 `SchedulerNode`）；
非 linalg op（`linalg.matmul`、`tensor.expand_shape`、`tensor.extract_slice` 等）
不进入任何 Group，天然成为融合边界。

```
// 对应 Inductor fuse_nodes()
repeat until no progress:
    pairs = get_fusable_pairs(all_groups)        // 对应 get_possible_fusions()
    if pairs is empty: break

    top_pairs = pairs 中最高 priority 的子集     // 对应 GetPossibleFusionsWithPrioritySort()
    for pair in top_pairs (按 score 降序):
        if pair.g1 和 pair.g2 都还存在:
            merge(pair.g1, pair.g2)
```

**Priority**（对应 AutoFuse FusionPriority）：
- VectorGroup + VectorGroup：DEFAULT
- CubeGroup + VectorGroup：LOW（所有 Vector fusion 收敛后才运行）

**Score**：两 group 之间共享 tensor 的字节数（融合后省去这部分 global memory 读写）。

**为什么需要多轮**：迭代融合能正确处理图中存在 skip connection 的情况。
以 LayerNorm 为例：

```
input → mean → sub → square → mean2 → sqrt → div → output
                └─────────────────────────────────↗
```

`sub` 有两个消费者（`square` 和 `div`）。

- 单次反向扫描：访问顺序决定结果，可能漏融 `sub`
- 多轮迭代：Round 1 先把 `{sqrt, mean2, square, div}` 融合成 G；
  Round 2 再融合 `sub` 进 G（此时 `sub` 的两个消费者均已在 G 中）✓

### 3.2 轴类上确界规则

```
AxisClass(axis a) = ⊔{member m ∈ G} class_of(m, a)

class_of(m, a) =
  Parallel   if a is parallel in m
  Reduction  if a is reduction in m
  Absent     if a does not appear in m's iter space

上确界规则：
  Parallel  ⊔ Parallel  = Parallel
  Parallel  ⊔ Reduction = Reduction    
  Parallel  ⊔ Absent    = Parallel     
  Reduction ⊔ Absent    = Reduction
```

- **Tileable 轴** = 所有 class == Parallel 的轴（可分核，可 tile）
- **Reduction 轴** = 所有 class == Reduction 的轴（不做并行切分；串行分块由 enableReductionSplit 控制）

### 3.3 VectorGroup can_fuse 规则

融合关系分为两类：

- `FusionKind::Vertical`   — g1 和 g2 之间存在直接 SSA producer-consumer 关系
- `FusionKind::Horizontal` — g1 和 g2 之间无直接 SSA 边，但共享至少一个 boundary input tensor

**公共规则**：`can_fuse(g1, g2) for VectorGroup + VectorGroup:`

  1. FusionKind 确定：

       - 若存在直接 SSA producer-consumer → Vertical
       - 否则，若存在共享 boundary input tensor → Horizontal
       - 否则 → 不融合

  2. 合并后无环（cycle check）

  3. 轴类相容（对 g1 ∪ g2 的所有成员按上确界规则重新推导）：

       ```
       merged_Tileable  = {a | ⊔ class(a) == Parallel}   非空
       merged_Reduction = {a | ⊔ class(a) == Reduction}
       merged_Tileable ∩ merged_Reduction = ∅            （由上确界定义保证，显式写出作为不变量）
       ```

  4. 链接 tensor 的所有 Vector-track 消费者都在 g1 或 g2 内：

     - 仅对 Vertical pair 有实质约束：g1 产出的 tensor T 若被 g2 外的 op 消费，则 T 无论如何都要物化，fusion 意义不大； 
     - Horizontal pair 无 producer-consumer 链接 tensor，此条件自动成立。
     - 替代旧设计"多消费者是硬边界"的保守规则：只要 fan-out 的目标都在同一融合 group 内，producer 的结果可以留在 UB 寄存器中，无需物化到 global memory

  5. DPS init 透明：linalg.fill 作为 linalg.reduce 的 outs 产生者，按普通成员纳入 group，tensor.empty 源头不进 group

  6. Epilogue 的 reduction 依赖约束（仅当 merged group 含 reduction op 时）：

     epilogue 侧的所有输入，要么来自 group 外的 tensor，要么来自 reduction 的最终输出（post-reduction result tensor），不能依赖 reduction 迭代过程中产生的中间 partial sum, 比如CumSum类。

     这保证 loop nest 生成时 epilogue 可以放在 RBLOCK loop 外，是 accumulator pattern（§8.4）能成立的前提。

  **Horizontal 额外规则**

  H1. 合并后 Tileable 轴集合取并集，merged_Tileable 非空且与 merged_Reduction 不相交
      （轴集合不要求完全一致；缺失轴的成员在 loop nest 生成时补插 broadcast，对应 Collapse 的 B1 处理）
  H2. 合并后新增 boundary input 数量 ≤ maxHorizontalExtraInputs（默认 = 4），
      防止 horizontal fusion 把过多独立 tensor 拉进同一 loop，增加 UB 压力

### 3.4 CubeGroup can_fuse 规则

**matmul 锚点识别**：

- op 是 `linalg.matmul` / `linalg.batch_matmul`

上游必须保证 matmul 以命名 op 形式进入 Pass 1，不接受 `linalg.generic` 形态的 matmul。
若上游输出为 `linalg.generic`，需在 Pass 1 前通过 `linalg-specialize-generic-ops` 或等价 pass 完成规范化。

**Epilogue（CubeGroup ← VectorGroup）**：

1. VectorGroup 内每个 op 的 iterator_types 全为 "parallel"

2. 无 in-place 写或 aliasing

3. 每个 op 的 indexing_map 对 matmul 输出轴是 affine projective （identity / broadcast / rank-drop；禁止 permutation 使 M/N 轴混合）
4. matmul 输出到 VectorGroup anchor 的路径上不能存在 `tensor.expand_shape / linalg.broadcast` 等显式 view node；side-input 不能有 batch 轴 broadcast

5. 所有输入要么来自 group 内成员，要么是 group 外的常量 / 纯标量

- Fan-out 检查：matmul 输出只能被一个 VectorGroup 消费；若被多个 VectorGroup 引用 → score = 0，不融合

- 不融合场景：VectorGroup 内全为纯 shape/view op → 不融合

**Prologue（VectorGroup → CubeGroup）**：

1. VectorGroup 内每个 op 的 iterator_types 全为 "parallel"

2. 无 in-place 写或 aliasing

3. 单消费者约束：VectorGroup 的每个输出只被 matmul 或 group 内其他成员消费，
不能被 group 外任何 op 引用

4. indexing_map 对 A/B 的输入轴是 affine projective


**启发式过滤**（v1 可跳过，后续有 profiling 数据后开启）：

1. 内存放大：fuse 后额外读取字节数 ≤ 原写入字节数 × 1.1
2. 不允许 constant_pad_nd 等引入非对齐读取的 op
3. 低精度模板（f16/bf16）不引入需要 f32 中间结果的 upcast

**CubeGroup 不做 Collapse**：M/N 轴始终独立，Collapse 对 CubeGroup 返回恒等映射。

### 3.5 Reduction Splitting（可选）

当 `enableReductionSplit=true` 时，Reduction 轴的 `RBLOCK_j` 从 `FixedTile/Full` 变为
`TunableTile/Inner`（串行 split，不做 parallel reduction）。

epilogue 数量上限（对应 AutoFuse `max_reduce_can_fuse_elementwise_nums`）：

```
reduction 之后的 pointwise epilogue 成员数量不超过 maxReduceEpilogueOps（默认 = 3）
超出上限时 can_fuse 返回 false，切断 epilogue 生长
```

---

## 4. Collapse——简化 canonical 迭代空间

**目的**：把 group 内部的多根维度合并成单根，让生成的 loop nest 尽量扁平。

**时机**：Group Analysis 之后、TilePlan 生成之前，仅作用于 **VectorGroup**（CubeGroup 跳过）。

**本质**：Collapse 相当于 AutoFuse "轴合并"（`UnifySubgraphAxis`）在 linalg-on-tensor 中的对应——
先确认 group 内所有成员对同一候选组 G 的解释一致（可行性预检），再把这种一致性物化为单根轴。
在任何 IR 修改之前完成可行性预检，类比 AutoFuse 的 `CanAxisMap` 先判断再 flush。

### 4.1 候选 collapse 组

从 group 的 canonical 轴序（对所有成员按上确界推导所得）出发，找"连续的同类型轴"（连续 Tileable 或连续 Reduction），
记为 `G = {d_i, ..., d_{i+k}}`。

### 4.2 Per-input 分类

对 group 内每个 op 的每条 indexing_map，判断 G 中的轴如何出现在该 map 的 result dims：

| 情况 | 条件 | 含义 |
|------|------|------|
| **A** | `G ∩ result(map) = ∅` | 整组轴被 broadcast 掉，该 input 不依赖 G |
| **C** | `G ⊆ result(map)` 且 G 内轴在 result 中**连续同序**出现 | 可直接折叠 |
| **B1** | `G ∩ result(map) ≠ ∅` 且 `G ⊄ result(map)` | 部分轴缺失，需先补全 |
| **B2** | `G ⊆ result(map)` 但轴在 result 中**乱序或不连续** | 轴顺序/位置不对齐，硬边界 |

B1 和 B2 可以组合出现（缺失且乱序），按先 B1 后 B2 顺序检测。

### 4.3 可行性预检（Pre-Check）

**在任何 IR 修改之前**，对候选组 G 扫描所有成员的所有 map 完成分类。

**跨成员视角**（含 Horizontal fusion 的 DAG group）：

同一个 boundary input tensor T 可能被多个 sibling member 分别使用，各自独立分类处理。

```
// Pre-Check 判定流程
for each member m in topoMembers:
    for each input map of m:
        classify → A / B1 / B2 / C

// 全为 A/C → 直接 collapse
// 存在 B1 → 各使用点独立插 linalg.broadcast，再 collapse
// 存在 B2 → 生成两个 Collapse Variant（§4.4 Step 2），后续步骤对两个 Variant 分别执行
```

### 4.4 规整流程

**Step 1：补全缺失维（处理 B1）**

对每个 B1 input，在其 consumer 之前独立插入 `linalg.broadcast`，
把缺失的 G 内轴补展开，使 `G ⊆ result(map)` 成立：

```
// 示例：G = {d0, d1}，input 只有 d1（shape [S]）
// 插入 linalg.broadcast：[S] -> [B, S]，map 变成 (d0,d1) -> (d0,d1)
```

插入的 broadcast op 打标记 `vector_plan.no_collapse = true`，collapse 遍历时跳过自身折叠，
并加入 group.topoMembers（collapse 阶段允许 group 增长）。

对含 Horizontal sibling 的 group：若 sibling A 对 T 是 B1、sibling B 对 T 是 C，
则只在 A 的使用点前插 broadcast，B 不受影响。

**Step 2：处理 B2（生成两个 Collapse Variant）**

B2 input 存在时，生成两个 Collapse Variant，各自产出独立的 CollapsedGroupInfo 和 TilePlan，
由 Autotuner 选优。

**Variant 1（Preserve）——计算侧重排**

对每个 B2 input，在其 consumer 之前插入 `linalg.transpose`，把乱序轴重排为连续同序：

```
// 示例：G = {d0, d1}，input [S, B]，consumer map: (d0,d1)→(d1,d0)（B2）
// 插入 linalg.transpose：[S, B] → [B, S]
// consumer map 变为 (d0,d1)→(d0,d1) → Case C
```

插入的 transpose op 标记 `vector_plan.no_collapse = true`，加入 group.topoMembers。

**约束**：该 transpose 是 collapse 边界——G 中的轴只在 transpose 输出侧参与 collapse，
transpose 本身及其输入侧保持展开。原因：transpose 自身的 input map 就是 B2，
若强行 collapse 则触发新的 B2 检测，引发无限递归。

GroupEmitter 发射时：`linalg.transpose` 映射为 `ConfusionTranspose`（UB 内重排，计算侧）。

**Variant 2（Eliminate）——Load 侧重排**

对每个 B2 input，不插 transpose 节点，而是：
1. 把该 consumer 的 indexing_map 改为 canonical（B2 → Case C）
2. 在对应 boundary input tensor 上打标记 `vector_plan.load_with_transpose = true`

```
// 示例：boundary input T [S, B]，consumer map: (d0,d1)→(d1,d0)（B2）
// 改 consumer map 为 (d0,d1)→(d0,d1)，T 打标 load_with_transpose=true
// T 对所有 consumer 均变为 Case C → 无 collapse barrier
```

无 collapse barrier，G 在整个 group 内连续参与 collapse。

GroupEmitter 发射 boundary input 时：`load_with_transpose=true` 的 tensor 映射为
`ConfusionTranspose`（GM→UB 搬运时完成重排，Load 侧）。

**Horizontal sibling**：
- Variant 1：各 sibling 的 B2 使用点独立插 transpose
- Variant 2：各 sibling 的 consumer map 独立改写，同一 boundary input 只打一次标

### 4.5 完整步骤

1. 按 canonical 轴序（上确界推导所得）找连续同类型候选组 G
2. **Pre-check**：对 G × 所有成员 × 所有 map 做 A/B1/B2/C 分类（含 sibling 跨成员视角）
3. **规整 B1**：有 B1 → 各使用点独立插 `linalg.broadcast`（补缺失维）
4. **规整 B2**：有 B2 → 生成 Variant 1（插 `linalg.transpose`）和 Variant 2（改 consumer map + 打 `load_with_transpose`），步骤 5–6 对两个 Variant 分别执行
5. 对所有成员（全部 A 或 C）调 `linalg::collapseOpIterationDims`
6. 在 group 边界 + 规整 op 输出处插 `collapse_shape` / `expand_shape`

### 4.6 典型场景

```
LayerNorm  [B, S, H] → [B*S, H]
  scale/bias [H]      → Case A（G={d_B,d_S} 完全缺失）
  其余 ops             → Case C

Softmax    [B, H, S, S] → [B*H*S, S]
  所有 ops             → Case C

Transpose+Pointwise  [B, S, H] → input 以 (H,S) 顺序出现
  该 input            → Case B2，生成两个 Variant：
    Variant 1：插 linalg.transpose [H,S]→[S,H]（no_collapse barrier）
               → collapse 只在 transpose 输出侧进行，发射时映射为 ConfusionTranspose（计算侧）
    Variant 2：consumer map 改为 canonical，boundary input 打 load_with_transpose
               → 无 barrier，整体 collapse，发射时映射为 ConfusionTranspose（Load 侧）

Horizontal sibling  add1[B,S,H] 和 add2[B,S] 共享 boundary input x[B,S]
  G = {d_B, d_S}（候选 collapse 组）
  add1 的 indexing_map 对 x：(d0,d1,d2) → (d0,d1)，G ⊆ result，连续同序 → Case C
  add2 的 side_input y[S]：G ∩ result = {d_S}，G ⊄ result（d_B 缺失）→ Case B1
  → 只在 add2 使用 y 的地方插 linalg.broadcast [S]->[B,S]，add1 不受影响，统一 collapse
```

---

## 5. Pass 2 实现算法

### 5.1 总体策略

Pass 2 **不复用** `linalg::tileUsingForOp` / `tileAndFuseProducerOfSlice` 等 tile-and-fuse
基础设施，而是自行建 loop nest，再按拓扑序**统一发射所有成员 op**。

原因：tile-and-fuse API 沿 SSA producer 链单向拉入 op，无法处理 horizontal sibling
（无直接 SSA 边但共享迭代空间的多个 sink）。统一发射在同一 loop body 内按 topoMembers
拓扑序处理所有 op，SSA 正确性由拓扑序保证，天然支持水平融合。

### 5.2 三个核心组件

**LoopNestBuilder**

从 TilePlan 建 scf.for 嵌套。按 `TilePlan.loopOrder` 依次建层，每层对应一个 G-axis，
IV 为该轴的 tile 偏移。产出 `loop_ivs`（G-axis → IV Value 的映射）和 innermost
insertion point。

**SliceComputer**

给定 `loop_ivs` 与某 op 的 `indexing_map`，推导该 op 每个 operand/result 对应 tensor 维度
的 `extract_slice` offset 和 sizes：

```
for dim_i in tensor.dims:
    g = indexing_map⁻¹(dim_i)     // 该 dim 对应哪个 G-axis
    offset[dim_i] = loop_ivs[g]   // 对应 G-axis 的 loop IV
    size[dim_i]   = tile_sizes[g] // 对应 tile size
```

对不参与当前 loop 层迭代的维度（如 reduction 轴在 parallel loop 层），offset=0，
size=full_dim。具体算法（affine map 求逆/投影）在 §5.5 展开。

**GroupEmitter**

按 topoMembers 拓扑序遍历，对每个 op：

1. **boundary input**（来自 group 外部）：用 SliceComputer 计算 offset/sizes，
   emit `tensor.extract_slice`
2. **interior op**（operands 是前序 emission 的结果）：operands 已是 tiled value，
   直接 emit tiled op，形状与 tile size 一致
3. **group boundary output**：emit `tensor.insert_slice`，结果通过 scf.for iter_arg 传出

### 5.3 整体流程

```
输入：CollapsedGroupInfo（topoMembers, TilePlan, B2 variant 标记）

1. 确定 loop 路径（§5.4）→ is_original_axes
2. LoopNestBuilder(TilePlan, is_original_axes) → loop_ivs, insertion_point
3. 若 enableReductionSplit=true：
   a. 在 parallel loop body 内 emit linalg.fill（init acc）
   b. 建 RBLOCK scf.for
   c. pre-reduction topoMembers → emit 进 RBLOCK loop body
   d. epilogue topoMembers → emit 在 RBLOCK loop 之后
4. GroupEmitter 在 innermost insertion_point 按 topoMembers 拓扑序 emit：
   for op in topoMembers (topo order):
       operand_slices = SliceComputer(loop_ivs, op.indexing_maps, tile_sizes)
       emit tiled op with operand_slices
5. 对所有 group boundary outputs：emit insert_slice + scf.yield
```

### 5.4 两条 Loop 路径

含 Variant 1 transpose（`no_collapse=true`）的 group 与普通 group 走不同的 loop 建立路径：

| 场景 | Loop 建立基准 | SliceComputer 输入 |
|------|-------------|--------------------|
| 普通 group（无 no_collapse op） | Collapsed G-axes（单一 flat IV） | flat IV 直接使用 |
| 含 Variant 1 transpose 的 group | 原始 G-axes（各轴独立 IV） | 各 G-axis IV 分别使用 |

Variant 1 路径下 transpose op 在 loop body 内的形态：

```
scf.for %iv_d0 = 0 to dim(d0) step t_d0 {
  scf.for %iv_d1 = 0 to dim(d1) step t_d1 {
    // input：B2 indexing_map → SliceComputer 得 offset=[iv_d1, iv_d0], sizes=[t_d1, t_d0]
    %in_tile = tensor.extract_slice %T[%iv_d1, %iv_d0][t_d1, t_d0]
    // tiled transpose：perm 不变，shape [t_d1, t_d0] → [t_d0, t_d1]
    %out_tile = linalg.transpose %in_tile perm=[1, 0]
    // output：canonical indexing_map → offset=[iv_d0, iv_d1]（由 GroupEmitter 处理后续 consumer）
  }
}
```

transpose op 本身由 GroupEmitter 按普通 op 处理；SliceComputer 对 input 用 B2 map、
对 output 用 canonical map，无需特殊分支。

**Variant 2**（`load_with_transpose=true` 标记）：
走普通 collapsed loop 路径。GroupEmitter 遇到打了该标记的 boundary input 时，
将对应 `tensor.extract_slice` 替换为带重排语义的占位 op，发射时落地为
`ConfusionTranspose`（GM→UB 搬运时完成重排）。

### 5.5 SliceComputer 算法（待细化）

SliceComputer 的核心是从 op 的 `indexing_map`（`AffineMap`）中恢复 tensor 各维度与
G-axis 的对应关系：

1. 遍历 `indexing_map` 的每个结果表达式
2. 若结果表达式是单个维度 `d_i`（AffineDimExpr），则该 tensor dim 对应 G-axis `d_i`；
   offset = `loop_ivs[d_i]`，size = `tile_sizes[d_i]`
3. 若结果表达式是常量 0 或该维度不在当前 loop 层（如 reduction 轴在 parallel loop），
   offset = 0，size = full_dim
4. 其他表达式形式（stride ≠ 1、仿射组合等）→ 当前版本暂不支持，pre-check 阶段应已排除

---

## 6. 核心数据结构

### 6.1 AxisRole / TileLevel / TileFieldKind

```cpp
enum class AxisRole : uint8_t {
  Parallel,
  Reduction,
};

enum class TileLevel : uint8_t {
  Outer,   // 外层分核：XBLOCK / BM / BN
  Inner,   // 内层 tile：XBLOCK_SUB / BK / RBLOCK_sub
  Full,    // 默认 full：RBLOCK_0 v1 不切
};

enum class TileFieldKind : uint8_t {
  TunableTile, // autotuner 搜索的 tile 参数
  FixedTile,   // 固定 tile 值或默认 full 的字段
  ShapeDim,    // 运行时 shape 透传
  Derived,     // 由其他字段/shape 推导；通常不进 TilingData bytes
};
```

### 6.2 TileLevel × AxisRole 完整矩阵

```
                  AxisRole=Parallel          AxisRole=Reduction
TileLevel=Outer   XBLOCK（vector 分核）       —
                  BM / BN（cube 2D 分核）
TileLevel=Inner   XBLOCK_SUB（vector UB 批）  BK（cube L0 K tile）
                                             RBLOCK_sub（vector reduction split）
TileLevel=Full    —                           RBLOCK（vector v1，不切）
```

Cube BK 和 Vector RBLOCK_sub 落在同一格（`Inner + Reduction`），统一由
`TileFieldSpec` 表达，只是默认值和约束不同。

### 6.3 GroupKind / GroupInfo

```cpp
enum class GroupKind : uint8_t {
  Vector,
  Cube,
};

// FusionKind（Vertical / Horizontal）是 can_fuse 内的局部分类，不持久化为字段。

struct AxisInfo {
  StringRef name;        // "B", "S", "H"；可为空
  int64_t   staticSize;  // ShapedType::kDynamic 表示动态
  AxisRole  kind;        // Parallel (Tileable) 或 Reduction (Full)
};

struct GroupInfo {
  GroupKind                     kind;
  SmallVector<linalg::LinalgOp> topoMembers;   // 拓扑序（确定性遍历顺序）
  SmallVector<linalg::LinalgOp> sinks;         // group 末端（无 group 内消费者的成员）
  SmallVector<AxisInfo>         canonicalAxes; // 对所有成员 iterator_types 按上确界推导所得
  SmallVector<Value>            boundaryIn;    // group 外输入 tensor
  SmallVector<Value>            boundaryOut;   // group 外输出 tensor
};

struct CollapsedGroupInfo : GroupInfo {
  SmallVector<AxisInfo>     collapsedAxes;
  SmallVector<int>          axisMap;       // 原轴 idx → 新轴 idx
};

// CubeGroup 专用扩展
struct CubeGroupInfo : GroupInfo {
  linalg::LinalgOp          matmul;
  linalg::LinalgOp          epilogueAnchor; // 最后 epilogue op；无 epilogue 时 == matmul
};
```

### 6.4 TileParam 与 TilePlan

```cpp
struct TileParam {
  StringRef    name;         // "XBLOCK" / "XBLOCK_SUB" / "RBLOCK_0" / "BM" 等
  Value        ssa;          // pass 插入的 func index arg
  OpFoldResult defaultValue; // 静态 shape → IntegerAttr；动态 → Value
  int32_t      axisIdx;      // 对应 collapsedAxes 的下标
  TileLevel    level;        // Outer / Inner / Full
  AxisRole     role;         // Parallel / Reduction
};

struct TilePlan {
  const CollapsedGroupInfo* group;

  // VectorGroup：
  //   tileable[0] = {XBLOCK(Outer), XBLOCK_SUB(Inner)}
  //   tileable[i>0] = {XBLOCK_SUB_i(Inner)}
  //   full[j] = {RBLOCK_j(Full 或 Inner，取决于 enableReductionSplit)}
  // CubeGroup：
  //   tileable[M] = {BM(Outer), Tb_M(Inner)}
  //   tileable[N] = {BN(Outer), Tb_N(Inner)}
  //   full[K] = {t_K(Inner)}
  //   tileable[batch_i] = {XBLOCK_i(Outer), XBLOCK_SUB_i(Inner)}（若有）
  SmallVector<SmallVector<TileParam>> tileable;
  SmallVector<TileParam>              full;

  // block_dim 表达式：
  //   VectorGroup：blockDimExprs[0] = ceildiv(tileable[0]_extent, XBLOCK)
  //   CubeGroup：blockDimExprs[0] = ceildiv(M, BM)，blockDimExprs[1] = ceildiv(N, BN)
  SmallVector<OpFoldResult, 2>        blockDimExprs;
};
```

### 6.5 TilePlan 生成规则

**VectorGroup（默认策略）**：

```
input: CollapsedVectorGroupInfo，canonical 轴序 axes
output: TilePlan

let Tileable = [a for a in axes if class(a) == Parallel]
let Full     = [a for a in axes if class(a) == Reduction]

assert len(Tileable) ≥ 1

// 外层 parallel：两级 split
tileable[0] = {XBLOCK(Outer,Parallel), XBLOCK_SUB(Inner,Parallel)}

// 其余 parallel：一级 tile，默认 full
for i in 1..len(Tileable):
  tileable[i] = {XBLOCK_SUB_i(Inner,Parallel), defaultValue=dim_size}

// 所有 reduction：
if not enableReductionSplit:
  full[j] = {RBLOCK_j, kind=FixedTile, level=Full, defaultValue=dim_size}
else:
  full[j] = {RBLOCK_j, kind=TunableTile, level=Inner,
             defaultValue=min(dim_size, UB_CAPACITY/elem_size),
             search={enabled=true, candidates=[...2的幂次...],
                     upperBound=UB_CAPACITY/elem_size}}
```

**CubeGroup（5 参数结构）**：

```
轴识别：
  M 轴 = matmul lhs 的第 -2 维
  N 轴 = matmul rhs 的第 -1 维
  K 轴 = matmul lhs 的第 -1 维（= rhs 的第 -2 维）
  Batch 轴 = batch_matmul 的前缀维（若有）

// M 轴：两级
tileable[M] = {
  BM(Outer,Parallel):   default=128, candidates=[64,128,256], alignment=HW_M_ALIGN,
  Tb_M(Inner,Parallel): default=64,  candidates=[32,64,128],  alignment=HW_M_ALIGN
}

// N 轴：两级
tileable[N] = {
  BN(Outer,Parallel):   default=128, candidates=[64,128,256], alignment=HW_N_ALIGN,
  Tb_N(Inner,Parallel): default=128, candidates=[64,128,256], alignment=HW_N_ALIGN
}

// K 轴：一级 Inner（必须切，非 Full）
full[K] = {
  t_K(Inner,Reduction): default=64, candidates=[64,128,256],
                         alignment=HW_K_ALIGN, upperBound=L0A_CAPACITY/elem_size
}

// Batch 轴（若有）：与 VectorGroup 相同
batch[i] = {XBLOCK_i(Outer,Parallel), XBLOCK_SUB_i(Inner,Parallel)}

// blockDimExprs：2D grid
blockDimExprs[0] = ceildiv(M_extent, BM)   // grid_y
blockDimExprs[1] = ceildiv(N_extent, BN)   // grid_x
```

硬件常量来源（从 target attr 或 pass option 读取）：

```cpp
Option<"hwMatmulMAlign", "hw-matmul-m-align", "int64_t", "16", "">
Option<"hwMatmulNAlign", "hw-matmul-n-align", "int64_t", "16", "">
Option<"hwMatmulKAlign", "hw-matmul-k-align", "int64_t", "16", "">
Option<"l0aCapacityBytes", "l0a-capacity-bytes", "int64_t", "65536", "">
```

### 6.6 SearchSpace

```cpp
struct SearchSpace {
  bool enabled = false;
  SmallVector<int64_t> candidates;

  // 硬件/buffer 对齐约束。
  // 所有 candidates 必须满足 value % alignment == 0。
  // vector RBLOCK_sub：无约束（留空）。
  // cube BK：alignment = 矩阵指令的 K 粒度（FP16=16, INT8=32）。
  std::optional<int64_t> alignment;

  // 上界约束，由 buffer 容量决定。
  // vector RBLOCK_sub：可选（UB 大小限制）。
  // cube BK：必填（L0A/L0B buffer 容量，单位 element 数）。
  std::optional<ValueExpr> upperBound;
};
```

---

## 7. TileInfo 数据模型（TilePlan → TileInfo 连接层）

### 7.1 定位

TileInfo 是 TilePlan 与 TilingData / AutoTuner 之间的**稳定可序列化中间层**：

```
TilePlan（编译器内部，含 MLIR Value/OpFoldResult）
  → TileInfo（稳定序列化对象，无 MLIR 内部指针）
      → PrepareForEmit 生成 TilingData
      → AutoTuner 搜索视图
```

### 7.2 ValueExpr

```cpp
struct ValueExpr {
  enum Kind {
    Const,    // int64_t 常量
    ShapeDim, // tensor arg[argIndex] 的 dimIndex 维
    FieldRef, // 引用其他 TileFieldSpec.fieldId
    Mul,      // lhs * rhs
    Add,      // lhs + rhs    (预留)
    CeilDiv,  // ceildiv(lhs, rhs)
    Min,      // min(lhs, rhs) (预留)
    Max,      // max(lhs, rhs) (预留)
  } kind;

  int64_t constValue;
  ShapeRef shape;          // {argIndex, dimIndex}
  std::string fieldId;
  // shared_ptr 使 ValueExpr 可拷贝，TileFieldSpec / TileInfo 保持可拷贝语义
  std::shared_ptr<ValueExpr> lhs;
  std::shared_ptr<ValueExpr> rhs;
};
```

v1 必须支持的节点：`Const`、`ShapeDim`、`FieldRef`、`Mul`、`CeilDiv`。

### 7.3 TileAxisInfo / TileFieldSpec / TileInfo

```cpp
struct ShapeRef {
  int32_t argIndex;
  int32_t dimIndex;
};

struct TileAxisInfo {
  int32_t    axisIndex;     // collapsed axis 的下标
  std::string axisName;     // "BS" / "H" / "M" / "K"
  AxisRole   role;          // Parallel / Reduction
  ValueExpr  extentExpr;    // 逻辑大小（可以是 B*S 这样的复合表达式）
};

struct TileFieldSpec {
  std::string           fieldId;       // 稳定语义 ID，如 "tile.xblock"
  std::string           abiName;       // TilingData 字段名，如 "XBLOCK"
  std::string           abiType;       // v1 统一 "i64"
  std::optional<int32_t> abiIndex;     // TilingData struct 中的顺序；
                                       // Derived 字段为 nullopt

  TileFieldKind         kind;          // TunableTile / FixedTile / ShapeDim / Derived
  std::optional<int32_t>     axisIndex;
  std::optional<TileLevel>   level;
  std::optional<ShapeRef>    shapeBinding;    // kind=ShapeDim 时必填
  std::optional<ValueExpr>   defaultExpr;     // TunableTile / FixedTile 时填写
  std::optional<SearchSpace> search;          // TunableTile 时填写
};

struct TileInfo {
  std::string kernelId;                // "group0_plan0"
  int32_t     groupId;
  int32_t     planId;

  SmallVector<TileAxisInfo>  axes;
  SmallVector<TileFieldSpec> fields;

  // VectorGroup：1 个元素；CubeGroup：2 个元素（grid_y, grid_x）
  SmallVector<ValueExpr, 2>  blockDimExprs;
};
```

### 7.4 TileInfo 不变量

1. **单一归属**：每个 `TileInfo` 只对应一个 `(groupId, planId)`
2. **稳定 ID**：`fieldId` 用于语义对齐；`abiName` 仅用于 ABI 可读性
3. **ABI 顺序权威**：`abiIndex` 是 packable 字段在 TilingData struct 中的唯一顺序权威；
   `Derived` 字段的 `abiIndex` 必须为 `nullopt`
4. **Shape 只绑定 primitive dim**：`ShapeDim` 只引用原始 tensor 参数的 primitive 维度
5. **无 MLIR Value 外流**：TileInfo 只持有 `ValueExpr`，不持有 SSA Value
6. **AutoTuner 只搜索 `TunableTile`**：`FixedTile` 不搜索，`Derived` 不直接 pack
7. **blockDimExprs 不靠反推**：必须在 TileInfo 里显式表达；
   CubeGroup 的 `blockDimExprs` 必须有两个元素（grid_y, grid_x）

---

## 8. 各阶段数据流

```
Source linalg IR
  │
  ├─ [Pass 1] vector-plan-group-analysis
  │   迭代融合算法（§3）
  │   → 每个 linalg op 打上 vector_plan.group_id / topo_index
  │
  │   [中间 IR 可 dump、可 FileCheck group annotation]
  │
  ├─ [Outline Pass] vector-plan-group-outline
  │   Step 1-3: 分桶 + 两级拓扑排序（by topo_index）
  │   Step 4: 重建 GroupInfo（kind / canonicalAxes / boundaryIn / boundaryOut）
  │   Step 5: Outline 每个 group → @kernel_groupN func
  │   Step 6: 原 func → coordinator func（private 声明 + 顺序 call）
  │   Step 7: 文件分离（split pass）
  │       → network.mlir（coordinator + kernel func private 声明）
  │       → kernel_group{N}.mlir（每个 group 一个文件）
  │
  │   [每个文件独立可 dump、可 FileCheck]
  │
  ├─ [Pass 2] vector-plan-tile-fuse（per kernel_group{N}.mlir，可并行）
  │   │
  │   ├─ Group Collapse（§4）
  │   │   → CollapsedGroupInfo per group
  │   │   VectorGroup：走 Cases A/B1/B2/C（含 sibling 跨成员预检）
  │   │   CubeGroup：恒等映射，跳过
  │   │
  │   ├─ TilePlan Generation（§5.5）
  │   │   → TilePlan per group
  │   │   VectorGroup：XBLOCK/XBLOCK_SUB/RBLOCK
  │   │   CubeGroup：BM/BN/Tb_M/Tb_N/t_K
  │   │
  │   ├─ Loop Nest 生成（§8.2–8.4）
  │   │   VectorGroup：以 anchor 建 loop nest，按 topoMembers 拓扑序发射
  │   │   CubeGroup：以 epilogueAnchor 建 loop nest，反向拉入 matmul
  │   │   外层 scf.for 打 ascendc.parallel
  │   │
  │   └─ Module Metadata
  │       → tiling.infos（TileInfo 数组，权威表示）
  │       → tiling.tiles / tiling.shapes（兼容投影）
  │
  └─ (unchanged Phase 2)
       network.mlir（coordinator）→ host launch codegen
       kernel_group{N}.mlir → bufferize → buffer-placement → linalg-to-ascendc → kernel_group{N}.cpp
```

### 8.1 Local Buffer 降级不变量

tile+fuse 的副产物应自动满足：

1. **Tile 尺寸收敛**：group 内所有 reduction 产出的 tensor，loop nest 生成后静态 shape
   等于 tile size（`XBLOCK_SUB` × `RBLOCK_0`），而不是全长
2. **Init 内移**：reduction 产出对应的 `linalg.fill` / `tensor.empty` 出现在最内层
   loop body 内，不在外层 loop 之前
3. **无全长临时**：group 边界 out 之外没有"全长 reduction 中间 buffer"

若不满足，按以下降级路径执行：
- 方案 A：Pass 2 loop nest 生成后增加轻量 post-pass，显式把 `linalg.fill` 搬进最内层 loop body
- 方案 B：独立 `--local-buffer-downgrade` pass，直接改写 tensor 类型与 fill 位置

### 8.2 VectorGroup Loop Nest 生成

VectorGroup 的 loop nest 生成以 canonical axes 为基准，在公共 loop body 内
按 `topoMembers` 拓扑序发射所有成员 op（而非"tile root + 反向 fuse producers"）：

```
// 阶段 1：以 canonical Tileable 轴建外层分核 loop
build for_XBLOCK    from canonicalAxes[Tileable[0]] {ascendc.parallel}
build for_XBLOCK_SUB from canonicalAxes[Tileable[0]]（在 for_XBLOCK 内）

// 阶段 2：若含 reduction（enableReductionSplit=true）
//   acc init 在 RBLOCK loop 之前；acc 累加在 RBLOCK loop 内；epilogue 在之后

// 阶段 3：在 loop body 内按 topoMembers 拓扑序发射所有成员
for m in topoMembers:
    emit m（已 tile 或作为 producer/sibling fuse 进 loop nest）
```

对 Horizontal sibling（无直接 SSA 边的两个 member）：
- 共享同一外层 loop nest（由 canonical axes 建立）
- 在 loop body 内按拓扑序排列，共享的 boundary input 同一次迭代内只读取一次

目标产出 IR 结构（以双 sibling add1/add2 共享输入 x 为例）：

```
scf.for %xb = 0 to BS step XBLOCK {       // ascendc.parallel
  scf.for %xs = 0 to XBLOCK step XBLOCK_SUB {
    %s1 = linalg.generic(x[xb+xs], side1)  // sibling1，按 topoMembers 顺序发射
    %s2 = linalg.generic(x[xb+xs], side2)  // sibling2
    insert_slice %s1 → out1[xb+xs]
    insert_slice %s2 → out2[xb+xs]
  }
}
```

### 8.3 CubeGroup Loop Nest 生成

CubeGroup 的 loop nest 生成以 `epilogueAnchor` 为 tile 起点，向后（producer 方向）fuse，
matmul 作为 epilogue 的 producer 被拉入，epilogue op 按拓扑序在 loop body 内发射：

```
// 轮 1：BM/BN 层，产生分核 loop
tile epilogueAnchor [BM, BN]
按拓扑序 fuse epilogue ops 进 for_BN
fuse matmul 进 for_BN          ← matmul 作为 producer 被反向 fuse
标注 for_BM / for_BN: ascendc.parallel = true

// 轮 2：Tb_M/Tb_N 层，在 BM/BN 结果上再 tile
tile tiled_epilogueAnchor [Tb_M, Tb_N]
按拓扑序 fuse epilogue ops 进 for_Tb_N
fuse matmul_BN 进 for_Tb_N

// 轮 3：K 轴 [t_K]，单独 tile matmul
tile matmul_Tb [0, 0, t_K]
标注 for_K: prologue = "lhs:A1->A2, rhs:B1->B2"
标注 matmul_final: ascendc.unit = "AiCore.Cube"
标注 epilogue ops:  ascendc.unit = "AiCore.Vector"
```

目标产出 IR 结构：

```
scf.for %BM {ascendc.parallel}
  scf.for %BN {ascendc.parallel}
    scf.for %Tb_M
      scf.for %Tb_N
        scf.for %t_K {prologue=A1->A2/B1->B2}
          linalg.matmul  [AiCore.Cube]
        linalg.bias_add  [AiCore.Vector]
        linalg.relu      [AiCore.Vector]
```

### 8.4 Accumulator Pattern（非全载 reduction 与 K 轴切分共用）

非全载 reduction（VectorGroup `enableReductionSplit=true`）和 CubeGroup K 轴切分（`t_K`）
共享同一个 accumulator 模式：在 reduction/K loop **之前**初始化 accumulator，
loop 内累加，loop **之后**对 accumulator 执行 epilogue。

**VectorGroup（RBLOCK split）**：

```
scf.for %xb = 0 to BS step XBLOCK {       // ascendc.parallel
  scf.for %xs = 0 to XBLOCK step XBLOCK_SUB {
    acc = linalg.fill(0, tensor.empty)     // ← RBLOCK loop 之前
    scf.for %rb = 0 to H step RBLOCK_sub { // reduction loop
      acc = linalg.reduce(x[xb+xs, rb:rb+RBLOCK_sub], acc)
    }
    epilogue(acc) → y[xb+xs]              // ← RBLOCK loop 之后
  }
}
```

**CubeGroup（t_K）**：

```
scf.for %BM { ascendc.parallel }
  scf.for %BN { ascendc.parallel }
    scf.for %Tb_M {
      scf.for %Tb_N {
        acc = linalg.fill(0, tensor.empty) // ← t_K loop 之前
        scf.for %tk { prologue=A1->A2/B1->B2 }  // K loop
          acc = linalg.matmul(A_tile, B_tile, acc)
        epilogue(acc) → output_tile        // ← t_K loop 之后，AiCore.Vector
      }
    }
```

两者的共同约束（对应 §3.3 规则 6 / §3.4 E1）：
- epilogue 的输入只依赖 accumulator（post-reduction/post-K 结果），不依赖 loop 内 partial sum
- accumulator 的 `linalg.fill` 必须在 reduction/K loop 内侧、epilogue loop 外侧（§7.1 Init 内移不变量）
- CubeGroup 的 t_K 始终为 `Inner`（非 Full）；VectorGroup 的 RBLOCK 在 `enableReductionSplit=false` 时为 `Full`（全载，无此模式），`=true` 时为 `Inner`（启用此模式）

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

// Module-level pass：把 group annotation 物化为独立 kernel func
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

标准 pipeline 写法：

```
vector-plan-group-analysis,
vector-plan-group-outline,
vector-plan-tile-fuse
```

### 9.2 tiling.infos 序列化

`TileInfo` 作为 module-level attribute（`tiling.infos`）挂在 IR 上，
是权威表示；`tiling.tiles` / `tiling.shapes` 保留为兼容投影。

**VectorGroup 示例**（LayerNorm `[B*S, H]`，`enableReductionSplit=false`）：

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
      {id = "tile.xblock",    abi_name = "XBLOCK",    abi_index = 0, kind = "tunable",
       axis = 0, level = "outer",
       default = {op = "const", value = 256},
       search = {candidates = [64, 128, 256]}},
      {id = "tile.xblock_sub", abi_name = "XBLOCK_SUB", abi_index = 1, kind = "tunable",
       axis = 0, level = "inner",
       default = {op = "const", value = 64},
       search = {candidates = [32, 64]}},
      {id = "tile.rblock0",   abi_name = "RBLOCK_0",  abi_index = 2, kind = "fixed",
       axis = 1, level = "full",
       default = {op = "shape_dim", arg = 0, dim = 2}},
      {id = "shape.0.0", abi_name = "dim_B",  abi_index = 3, kind = "shape_dim",
       from_arg = 0, dim = 0},
      {id = "shape.0.1", abi_name = "dim_S",  abi_index = 4, kind = "shape_dim",
       from_arg = 0, dim = 1},
      {id = "shape.0.2", abi_name = "dim_H",  abi_index = 5, kind = "shape_dim",
       from_arg = 0, dim = 2}
    ],
    block_dim = [{op = "ceildiv",
                  lhs = {op = "mul",
                         lhs = {op = "shape_dim", arg = 0, dim = 0},
                         rhs = {op = "shape_dim", arg = 0, dim = 1}},
                  rhs = {op = "field_ref", id = "tile.xblock"}}]
  }]
}
```

**CubeGroup 示例**（matmul `[M, K, N]` + bias_add + relu）：

```mlir
tiling.infos = [{
  kernel = "group1_plan0",
  axes = [
    {axis = 0, name = "M", role = "parallel",  extent = {op = "shape_dim", arg = 0, dim = 0}},
    {axis = 1, name = "N", role = "parallel",  extent = {op = "shape_dim", arg = 1, dim = 1}},
    {axis = 2, name = "K", role = "reduction", extent = {op = "shape_dim", arg = 0, dim = 1}}
  ],
  fields = [
    {id = "tile.bm",   abi_name = "BM",   abi_index = 0, kind = "tunable",
     axis = 0, level = "outer", default = {op = "const", value = 128},
     search = {candidates = [64, 128, 256], alignment = 16}},
    {id = "tile.bn",   abi_name = "BN",   abi_index = 1, kind = "tunable",
     axis = 1, level = "outer", default = {op = "const", value = 128},
     search = {candidates = [64, 128, 256], alignment = 16}},
    {id = "tile.tb_m", abi_name = "Tb_M", abi_index = 2, kind = "tunable",
     axis = 0, level = "inner", default = {op = "const", value = 64},
     search = {candidates = [32, 64, 128], alignment = 16}},
    {id = "tile.tb_n", abi_name = "Tb_N", abi_index = 3, kind = "tunable",
     axis = 1, level = "inner", default = {op = "const", value = 128},
     search = {candidates = [64, 128, 256], alignment = 16}},
    {id = "tile.tk",   abi_name = "t_K",  abi_index = 4, kind = "tunable",
     axis = 2, level = "inner", default = {op = "const", value = 64},
     search = {candidates = [64, 128, 256], alignment = 16,
               upper_bound = {op = "const", value = 4096}}}
  ],
  block_dim = [
    {op = "ceildiv", lhs = {op = "shape_dim", arg = 0, dim = 0},
                     rhs = {op = "field_ref", id = "tile.bm"}},
    {op = "ceildiv", lhs = {op = "shape_dim", arg = 1, dim = 1},
                     rhs = {op = "field_ref", id = "tile.bn"}}
  ]
}]
```

---

## 10. 接口边界

### 10.1 TilePlan → TileInfo

语义降维：把含 MLIR Value/OpFoldResult 的编译器内部对象转成稳定可序列化对象。

1. 每个 collapsed axis 变成一条 `TileAxisInfo`
2. 每个 tile param 变成一个 `TileFieldSpec`
3. 原始 tensor 参数的 primitive dim 绑定变成 `ShapeDim` 字段
4. `blockDimExprs` 由 TilePlan 生成阶段显式给出（不靠反推）
5. `TilePlan.defaultValue` 若是 `OpFoldResult`，此处转成 `ValueExpr`

### 10.2 TileInfo → TilingData

ABI 物化：PrepareForEmit 直接读取 `TileInfo.fields`，按 `abiIndex` 构造
`emitasc.py_struct<"TilingData", ...>`，不再通过扫描 i64 args 反推字段。

- `TunableTile` / `FixedTile` / `ShapeDim` → 进入 TilingData
- `Derived` → 默认不进入 TilingData bytes

### 10.3 TileInfo → AutoTuner

搜索投影：

```cpp
struct AutotuneParam {
  std::string fieldId;
  std::string abiName;
  SmallVector<int64_t> candidates;
};

struct AutotuneSpec {
  std::string kernelId;
  SmallVector<AutotuneParam> params;        // TunableTile 字段
  SmallVector<TileFieldSpec> fixedFields;   // FixedTile 字段
  SmallVector<TileFieldSpec> shapeFields;   // ShapeDim 字段
  SmallVector<ValueExpr, 2>  blockDimExprs;
};
```

AutoTuner **不从 loop 结构反推 `block_dim`**，不从字段名前缀猜字段角色。

### 10.4 ValueExpr 求值

```cpp
int64_t evalExpr(const ValueExpr& expr,
                 ArrayRef<SmallVector<int64_t>> shapeValues,  // [argIndex][dimIndex]
                 const TunedTileValues& tuned);               // fieldId → chosen value
```

仅在 host tiling（CPU 侧）调用，不进入 kernel。

---

## 11. 迁移路径

| 阶段 | 内容 | 兼容性 |
|------|------|--------|
| Phase A | 引入 TileInfo，Pass 2 写 `tiling.infos` | runtime ABI 不变 |
| Phase B | PrepareForEmit 改为消费 TileInfo | 不再猜字段顺序 |
| Phase C | AutoTuner 改为消费 TileInfo；hard-fail 若缺失 | 删除旧反推路径 |
| Phase D | `tiling.tiles` / `tiling.shapes` 降级为兼容视图 | — |

**现有测试兼容性**：
- Phase 1 VectorGroup 测试：完全不变（`enableReductionSplit` 默认 false）
- Phase 1 TileInfo 格式：完全兼容；SearchSpace 新字段对 v1 vector 为空
- `tiling.tiles` / `tiling.shapes`：不变；compat 投影逻辑不涉及 GroupKind
- AutoTuner：读 `search.candidates` 逻辑不变；新增 `alignment` 过滤步骤
- PrepareForEmit：通过 `TileLevel=Outer` 的个数判断 1D/2D dispatch，不需要读 GroupKind

---