# Vector Plan Generation — Phase 1 实施方案（定稿）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标:** 在 Ascend-MLIR 中落地一个自洽、可验证、可扩展的 Vector Phase 1 实现：把经过常规预处理的 linalg 程序自动编译为 tiled + fused 的 Ascend NPU Vector kernel，无需手写 transform 脚本。

**Spec:** `docs/superpowers/specs/2026-04-09-mlir-ai-compiler-generalization-design.md`

**技术栈:** MLIR C++（upstream linalg / SCF / tensor / affine），TableGen，lit + FileCheck

---

## 1. 核心心智模型

本方案不是"从 spec 抄代码"，而是围绕三个清晰正交的概念组织：**Chain / Collapse / TilePlan**。任何一个阶段的产出都是下一阶段的输入，互不重叠。

### 1.1 Chain —— tile+fuse 的最大闭包

**定义**：一个 Chain 是一组 linalg Vector op，它们能被 `tileConsumerAndFuseProducersUsingSCF` 塞进同一个 tile loop nest、成为一个 kernel。

**构造方式**：根驱动反向生长。

- **Root** = Vector op，其输出流出 Vector track 边界（被 matmul / reshape / func.return / 另一 chain 消费）。反向扫描 block，第一个满足这个条件的 op 就是 root。
- **生长** = 从 root 出发，按 SSA 逆序尝试吸收 producer。
- **硬边界**（见到即切）：
  - 非 linalg op（`linalg.matmul`、`tensor.expand_shape`、`tensor.extract_slice` 等）
  - 动态 offset slice / gather / scatter（非 affine indexing）
  - 跨 block / 跨 region

**Outer-loop fusion 支持**（等价于 Inductor 的 `OuterLoopFusedSchedulerNode`，以下简称 OLFSN）：

本 pass 的 chain 机制就是 OLFSN 在 MLIR 侧的对应形态。轴类上确界（`Parallel ⊔ Reduction = Reduction`）直接对应 OLFSN 的"外层共享、内层独立"语义——整根轴被归到 Full 时，chain 内所有成员都必须跑完整根轴，但**内层循环结构**（tiling / vectorization / 是否 reduction / 是否有内层 transpose）允许各成员独立。

**覆盖的融合问题**（v1 目标）：

1. **Reduce → pointwise 消费**：row-mean、softmax 末段的 div、LayerNorm 的 affine。
2. **Reduce → reduce 链式**：LayerNorm 的 mean/var、softmax 的 max/sum。两个 reduction 内层维度可以完全不同，只要外层并行轴能对齐。
3. **内层异构的多 kernel**：不同 vector 宽度、不同 reduce 维度、不同 tile split，只要外层轴同构就能共享外层 loop nest。
4. **Broadcast / keepdim 带来的 rank 不齐**：由 `Absent` 分类自动吸收，不要求显式 rank 对齐。

**等价性对照表**（Inductor OLFSN ↔ 本 pass）：

| Inductor OLFSN 能力 | 本 pass 对应机制 | 位置 |
|---|---|---|
| `can_fuse_vertical_outer_loop`：允许 reduction 作为生产者进入融合 | 轴类上确界 `Parallel ⊔ Reduction = Reduction` | §1.1 本节 |
| `_get_outer_loop_fusion_depth`：计算外层 vars 前缀匹配深度 | canonical 轴序 + Tileable 集合 | §1.1 本节 |
| `check_outer_fusion_loop_level_attr`：校验外层 tile 属性一致 | `tileConsumerAndFuseProducersUsingSCF` 的 tiling contract 自动保证 | Task 5 |
| `try_outer_loop_fusion_with_local_buf`：把中间 tensor 降级为栈上局部 buffer | tile+fuse 的 destination rewrite + Phase 2 bufferize（副产物） | §1.5 |
| `try_share_local_buffer`：生命周期复用 local buffer slot | Phase 2 buffer-placement 的常规 liveness 分析 | §1.5 |
| OLFSN 的 fallback：属性校验失败时退回单 kernel codegen | chain 不合法时切开、每个成员独立成链 | §1.1 可吸收判据 (3)(4) |

**显式不覆盖的场景**（需要由前置 pass 归一后再进本 pass，详见 §5 延期项）：

> **【待明确】** 外层 transpose / 外层 slice / 外层 concat 等"外层迭代空间扰动"的前置归一方案先挂起，待另行明确。本 pass 当前只处理前置已归一的输入，遇到未归一的外层扰动走硬边界切 chain。

- 具体做法：把 chain 的 canonical 迭代空间每一根轴做一次分类：

  ```
  AxisClass(axis a) =
    ⨆{member m ∈ C} class_of(m, a)  
    
  class_of(m, a) =
    Parallel       if a is parallel in m
    Reduction      if a is reduction in m
    Absent         if a does not appear in m's iter space
    
  上确界（least upper bound）规则：
    Parallel  ⊔ Parallel  = Parallel
    Parallel  ⊔ Reduction = Reduction    // 整根轴必须整块（full）
    Parallel  ⊔ Absent    = Parallel     // broadcast 不强制 full
    Reduction ⊔ Absent    = Reduction
  ```

- Chain 的**可 tile 轴集**（Tileable）= 所有 class == Parallel 的轴；
- Chain 的**必须 full 轴集**（Full）= 所有 class == Reduction 的轴；
- Absent 的成员在 tile 时由 upstream tile+fuse 按 projective map 自动切片，无需特殊处理。

**可吸收判据**（producer P 能否并入 chain C）：

```
P 可吸收 iff:
  1. P 是 linalg Vector op
  2. P 的 output → C 某成员的 indexing_map 是 affine projective
     （identity / permutation / broadcast / rank-drop 都 OK）
  3. 吸收 P 之后，Tileable 集合非空（至少保留 1 根轴可用于多核分派）
  4. 吸收 P 之后，Tileable 与 Full 交集依旧为空（无轴冲突）
  5. DPS init 的透明处理：若 P 是下一个成员的 outs 产生者
     （典型如 linalg.fill 作为 linalg.reduce 的 init），按普通成员
     纳入 chain，轴类按 P 的 iter space 参与上确界计算。P 的
     tensor.empty 源头仍属硬边界（非 linalg op）不进 chain，
     但会被 tile+fuse 在 loop body 内重生为 tile 大小的局部 init。
```

条件 (3) 保证 chain 还能用；条件 (4) 是正确性前提；条件 (5) 保证 reduction 初始化算子不会被甩出 chain 外导致多余的全长 init buffer。任何冲突 → 切开 chain，P 成为新 chain 的 root。

**覆盖的典型模式**：
- 纯 pointwise chain（bias+relu、QKV bias）：Tileable = 所有维，Full = ∅
- LayerNorm：Tileable = {BS}，Full = {H}，root = 最后一个 affine 的 pointwise
- Softmax：Tileable = {B, 4, S_row}，Full = {S_col}，root = 最后的 div
- Broadcast / transpose 作 producer：仅通过 indexing_map 进入 chain，不单独处理

### 1.2 Collapse —— 简化 canonical 迭代空间

**目的**：把 chain 内部的多根维度合并成单根，让生成的 loop nest 尽量扁平。

**时机**：chain 分析之后、tile 之前，作用于 **chain 的每一个成员**。

**候选 collapse 组**：从 chain 的 canonical 轴序出发，找"连续的同类型轴"（连续 Tileable 或连续 Full），记为 `G = {d_i, ..., d_{i+k}}`。

**Per-input 分类**：对 chain 每个成员的每条 `indexing_map`，计算 `G ∩ result(map)` 并归类：

| 情况 | 条件 | collapse 可否直接应用 |
|---|---|---|
| **A** | `G ∩ result(map) = ∅`（整组被 broadcast 掉） | ✓ 直接可 collapse，map 原样保留（组里每根轴本来就不出现） |
| **C** | `G ⊆ result(map)` 且 G 里所有轴在 result 中**连续同序**出现 | ✓ 直接可 collapse，对应位置压成单根 `e` |
| **B** | 其余所有情况（部分保留、乱序、跨 permute） | ✗ 需要先物化（见下），然后其他成员按 A/C 处理 |

**Case B 的处理：物化 broadcast**

在 consumer 之前插入一个 `linalg.broadcast`，把原始 Case B input 扩到与 consumer iter space 同秩的 identity 形态。关键是**物化出来的 broadcast op 本身不参与这次 collapse**，它保留原秩，靠 `tensor.collapse_shape` 做形状桥接：

```mlir
// 原始：
%x : tensor<Mxf32>                                          ← Case B source
%out = linalg.generic {
  indexing_maps = [(m,n) -> (m), (m,n) -> (m,n)],           ← Case B for G={m,n}
  ...
} ins(%x) outs(%init)

// 物化后（还未 collapse）：
%x_big = linalg.broadcast ins(%x : tensor<Mxf32>)
                          outs(%empty : tensor<MxNxf32>)    ← 保留 2D，标记 NoCollapse
                          dimensions = [1]
%out = linalg.generic {
  indexing_maps = [(m,n) -> (m,n), (m,n) -> (m,n)],         ← 现在是 Case C
  ...
} ins(%x_big) outs(%init)

// 再 collapse consumer：
%x_flat = tensor.collapse_shape %x_big [[0, 1]]             ← boundary reshape
            : tensor<MxNxf32> into tensor<MNxf32>
%out = linalg.generic {
  indexing_maps = [(e) -> (e), (e) -> (e)],
  iterator_types = ["parallel"]
} ins(%x_flat) outs(%init_flat)
```

物化规则：

1. `linalg.broadcast` 打标记 `vector_plan.no_collapse = true`，collapse 遍历成员时跳过它
2. 在物化 op 和下游 consumer 之间插 `tensor.collapse_shape`（G 对应的 reassociation）
3. 物化 op 加入 chain.members（chain 在 collapse 阶段允许增长，不是分析阶段冻结）
4. tile+fuse 把它作为 rank-2 producer fuse 进来，按 m 切片，Phase 2 bufferize 若识别成 stride-0 view 则零拷贝；v1 不做这层优化但允许真拷贝

**这不是"轴归一化"**：物化只影响单个 Case B input、不改其他成员的 layout；它把隐式 broadcast 提前成显式 op，使 `linalg::collapseOpIterationDims` 能统一走 A/C 路径。

**完整 collapse 步骤**：
1. 按 canonical 轴序找连续同类型候选组
2. 对每个组 × 每个成员 × 每条 map 做 A/B/C 分类
3. 收集所有 Case B，先物化（插 `linalg.broadcast` + 打 `no_collapse` 标记），chain 成员随之增长
4. 对剩余成员（所有都是 A 或 C）调 `linalg::collapseOpIterationDims`
5. 在 chain 边界 + 物化 op 输出处插 `collapse_shape` / `expand_shape`

**期望**（v1 主要目标网络，Case B 计数应为 0）：
- LayerNorm `[B, S, H]` → `[B*S, H]`：scale / bias 是 Case A，其余 Case C，**无 Case B**
- Softmax `[B, 4, S, S]` → `[B*4*S, S]`：全部 Case C，**无 Case B**
- QKV bias chain：bias 张量 Case A，**无 Case B**
- `relu-broadcast-transpose`：单 op chain，无需 collapse
- 所有 chain 压到 **1 Tileable + 0 或 1 Full** 的 2D 形态
- 不能 collapse 或 Case B 过多时保留原始形态，走完整多轴 TilePlan

### 1.3 TilePlan —— 完整的 tile 参数契约

**原则**：canonical 轴里每一根都要有一个 symbolic tile 参数。没有隐含 default，没有 "忽略此轴"。

**结构**：

```cpp
struct TilePlan {
  // collapse 之后 root 的 canonical 轴序
  SmallVector<AxisKind> axes;             // Tileable 或 Full
  
  // 每根 Tileable 轴的 tile 参数
  // - 第一根（outermost）：两级 split
  //     {XBLOCK, XBLOCK_SUB}   外层多核分派 + 内层 UB 批
  // - 其余 Tileable 轴：一级 tile
  //     {XBLOCK_SUB_i}         内层 UB 批
  SmallVector<TileParam> tileable;
  
  // 每根 Full 轴的 tile 参数
  // - 每根：一级 tile，默认值 = 完整维长（full reduction）
  //     {RBLOCK_j}
  SmallVector<TileParam> full;
};

struct TileParam {
  StringRef    name;         // "XBLOCK" / "XBLOCK_SUB" / "XBLOCK_SUB_1" / "RBLOCK_0" ...
  Value        ssa;           // 作为 func index arg 注入的 SSA 值
  OpFoldResult defaultValue;  // 写入 tiling.tiles，供 autotune 种子
                              // 静态 shape → IntegerAttr
                              // 动态 shape → Value（DimExpr / tensor.dim 结果）
};
```

**动态 shape 适配**：`defaultValue` 字段使用 `OpFoldResult`（或等价的 `Attribute | Value` 变体），允许承载 DimExpr。典型入口：Full 轴的 default 值 = 完整维长，在动态 shape 下是 `tensor.dim` 的 SSA 结果，而不是整数常量。v1 不做 e2e 动态 shape 测试，但数据结构必须兼容——否则后续加动态 shape 支持时要动 `TileParam` schema，影响面大。`tiling.tiles` 的序列化层在动态 shape 下写入一个符号占位（例如 `"dyn"`），由 autotune 运行时解析。

**生成规则**（v1 默认策略）：

```
input: 一个 collapsed chain，其 canonical 轴序 axes
output: TilePlan

let Tileable = [a for a in axes if class(a) == Parallel]
let Full     = [a for a in axes if class(a) == Reduction]

assert len(Tileable) ≥ 1           // 否则 chain 在分析阶段就被拒绝

// 外层 parallel：两级 split
tileable[0] = { name="XBLOCK",     default=autotune,        arg_idx=n+0 }
            + { name="XBLOCK_SUB", default=autotune,        arg_idx=n+1 }

// 其余 parallel：一级 tile，默认 full（即不切）
for i in 1..len(Tileable):
  tileable[i] = { name=f"XBLOCK_SUB_{i}", default=dim_size, arg_idx=... }

// 所有 reduction：一级 tile，默认 full
for j in 0..len(Full):
  full[j] = { name=f"RBLOCK_{j}", default=dim_size, arg_idx=... }
```

**关键不变量**：

1. **完整性**：canonical iter space 中每一根轴都出现在 TilePlan 里，不存在"隐含 full"。
2. **对称性**：Tileable[0] 永远是两级 split（多核分派强制要求），其余轴都是一级 tile。
3. **可 autotune**：所有参数都有默认值（写入 `tiling.tiles`），且都可被 Phase 3 独立替换。
4. **可 collapse 还原**：如果 Task 3 collapse 成功把一条链压到 1 Tileable + 1 Full，TilePlan 只包含 `{XBLOCK, XBLOCK_SUB, RBLOCK_0}`，loop nest 3 深（含 `ascendc.parallel`），是最简单的形态。如果 collapse 失败（Task 3 决定不 collapse），TilePlan 里会多出 `XBLOCK_SUB_i` / `RBLOCK_j`，loop nest 更深但逻辑不变。

### 1.4 流水线总览

```
                      source linalg IR
                           │
                  [ 预处理 —— 假设已完成 ]
                  (torch-mlir 后处理、unit-extent、fuse-elementwise、
                   canonicalize、cse —— 不由本 pass 负责)
                           │
                           ▼
          ┌────────────── Task 1 ──────────────┐
          │   vector-plan-generation pass 骨架  │
          │   注册 + Option + Passes.td         │
          └────────────────┬───────────────────┘
                           │
                           ▼
          ┌────────────── Task 2 ──────────────┐
          │           Chain Analysis            │
          │   produce: List<ChainInfo>          │
          │   每个 ChainInfo 含 root、members、  │
          │   canonical axes、axis class、       │
          │   boundary in/out                   │
          └────────────────┬───────────────────┘
                           │
                           ▼
          ┌────────────── Task 3 ──────────────┐
          │           Chain Collapse            │
          │   per-chain collapse 分析，          │
          │   apply collapseOpIterationDims，    │
          │   boundary reshape                   │
          │   produce: CollapsedChainInfo       │
          └────────────────┬───────────────────┘
                           │
                           ▼
          ┌────────────── Task 4 ──────────────┐
          │          TilePlan Generation        │
          │   完整 TilePlan per chain            │
          │   symbolic index args 注入 func     │
          │   produce: List<TilePlan>           │
          └────────────────┬───────────────────┘
                           │
                           ▼
          ┌────────────── Task 5 ──────────────┐
          │      TilePlan Realization           │
          │   tileConsumerAndFuseProducersUsingSCF│
          │   以 TilePlan 的 symbolic 值填 tile  │
          │   sizes，fusionControlFn 限 chain    │
          │   外层 scf.for 打 ascendc.parallel   │
          │   后置 LICM                          │
          └────────────────┬───────────────────┘
                           │
                           ▼
          ┌────────────── Task 6 ──────────────┐
          │          Module Metadata            │
          │   tiling.tiles / tiling.shapes      │
          │   attached on module                 │
          └────────────────┬───────────────────┘
                           │
                           ▼
                    (unchanged Phase 2)
             bufferize → buffer-placement →
             linalg-to-ascendc → parallelize →
                      codegen
```

### 1.5 Local Buffer 降级不变量

Inductor OLFSN 需要一个独立的 `try_outer_loop_fusion_with_local_buf` pass，显式把 reduction 中间结果从全局大 tensor 降级到只活在外层迭代内的 local buffer。**在本方案的 MLIR 路线上这一步是免费的**：

- `tileConsumerAndFuseProducersUsingSCF` 在 fuse producer 时会把 producer 的结果 tensor 改写为"消费者 tile 大小"的局部 tensor；
- Phase 2 的 bufferize + buffer-placement 把这个局部 tensor 分配到 UB / stack；
- 结果是：reduction 产物的生命周期自动收敛到外层 tile 一轮迭代内，**无需本 pass 额外实现 OLFSN 意义上的 local buffer pass**。

**不变量**（Task 5 realize 产物必须满足，Task 7 的 FileCheck 必须显式断言）：

1. **Tile 尺寸收敛**：chain 内所有 reduction 产出的 tensor，在 realize 之后其静态 shape 必须等于 "tile size"（`XBLOCK_SUB` × `RBLOCK_0` 等），而不是原始全长。
2. **Init 内移**：这些 reduction 产物对应的 `linalg.fill` / `tensor.empty` 出现在**最内层 loop body** 内，不在外层 loop 之前。等价于 §1.1 判据 (5) 的 DPS init 透明处理在 Task 5 的可见后果。
3. **无跨 iteration 生存的全长临时**：chain 边界 out 之外没有任何 "全长 reduction 中间 buffer"。如果看到 `tensor.empty` 出现在 chain 外层并被 insert_slice 多次写入，视为违反不变量。

**Fallback 路径**（v1 spike 必须回答）：

如果 Task 0 spike 发现 `tileConsumerAndFuseProducersUsingSCF` 在 `linalg.reduce` 这类带 DPS init 的生产者上**无法自动达成上述不变量**（典型的失败模式：reduce 的 init fill 被留在外层 loop 之前、中间 tensor 保持全长、或 TilingInterface 行为与 `linalg.generic + reduction iterator` 不一致），按下列降级路径执行：

1. **降级方案 A —— 手动内移 init**：在本 pass 的 Task 5 之后增加一个轻量 post-pass，显式把 chain 内部的 `linalg.fill` + `tensor.empty` 搬进最内层 loop body，并把对应 tensor 类型改写为 tile-size 的静态 shape。这是最小改动，仅补齐 tile+fuse 的缺口。
2. **降级方案 B —— 独立 local buffer pass**：若方案 A 对 reduction 的 DPS chain 仍处理不干净（多次写入、生命周期交叉等），新增一个独立 pass `--local-buffer-downgrade`，专门处理 chain 内 reduction 中间 buffer 的 "全长 → tile-size" 降级。此 pass 不复用 `tileConsumerAndFuseProducersUsingSCF`，而是直接基于 chain 元数据改写 tensor 类型与 `linalg.fill` 位置，完全对齐 Inductor `try_outer_loop_fusion_with_local_buf` 的语义。
3. **触发条件**：Task 0.0 (row_mean spike) 或 Task 0.1 (softmax spike) 的任一个 FileCheck 失败即触发降级决策；优先尝试方案 A，A 失败再上 B。

**判定流程**：

```
Task 0 spike
   ├── row_mean 不变量达成? ── 是 ──┐
   │                                 ▼
   └── softmax 不变量达成? ── 是 ── 走默认路径（无 local buffer pass）
                  │
                  └── 否 ── 方案 A 能补齐? ── 是 ── 走方案 A
                                    │
                                    └── 否 ── 走方案 B（新增独立 pass）
```

无论走哪条路径，**§1.5 不变量始终不变**；变的只是"谁负责保证它"。这条降级链必须在 Task 0 结束时给出确定的答案并写入 plan 附录（更新本节顶部的"默认路径"指向）。

---

## 2. Scope

### 2.1 本版要做

- vector-plan-generation pass 的完整骨架与注册
- Chain / Collapse / TilePlan / Realize 四步核心
- **Chain 机制等价于 Inductor OLFSN 的 MLIR 形态**（详见 §1.1），覆盖四类融合场景：reduce→pointwise、reduce→reduce 链式、内层异构多 kernel、broadcast/keepdim rank 不齐
- **Local buffer 降级由 tile+fuse 副产物保证**（详见 §1.5），默认路径不实现独立 pass；若 Task 0 spike 失败则按 §1.5 的 fallback 方案 A 或 B 执行
- 支持 reduce-后-pointwise 的 chain（softmax、LayerNorm、row_mean 模式）
- **Case B broadcast 物化**：collapse 遇到部分保留的 broadcast input 时自动插入 `linalg.broadcast` + 边界 reshape，绕开 linalg 代数对 projective permutation 的硬约束
- 完整 TilePlan（每根轴都有 symbolic tile 参数，`defaultValue` 采用 `OpFoldResult` 以兼容动态 shape）
- **`tiling.infos` 元数据写入**：Task 6 负责从 `TilePlan` 构建 `TileInfo` 并挂在 module 上，作为 PrepareForEmit / AutoTuner 的单一权威源头（见 `docs/superpowers/specs/2026-04-14-tile-info-design.md`）
- `tiling.tiles` / `tiling.shapes` 作为兼容投影保留，不破坏当前 autotuner 可用性
- 四条集成测试：bias+relu（纯 pointwise）、LayerNorm（reduce 在中间）、softmax（双 reduction outer-fuse）、relu-broadcast-transpose（broadcast 压进 indexing_map）

### 2.2 本版不做

1. **torch-mlir 后处理相关 pass**（EliminateCfAssert / CanonicalizeExtractBroadcast）—— 由外部 pipeline 负责，本 pass 假设输入已规整。
2. **Kernel function outline** —— realization 就地改写，不拆新 func。Phase 2 直接消费。
3. **多 plan 枚举** —— 每条 chain 只生成一个 default TilePlan。备选 plan（换外层轴、split reduction）留到后续。
4. **动态 shape 端到端测试** —— 代码保留 `DimExpr` + `tiling.shapes` 机制，集成测试只跑静态 shape。
5. **Reduction splitting**（RBLOCK < full）—— 默认 RBLOCK = 完整维长，UB 放不下的 case 留到后续。

### 2.3 成功标准

- `afir-opt --vector-plan-generation` 注册可用
- Task 2 的 chain 分析在 3 个测试 case 上输出的 chain 数量 / 成员 / axis class 全部符合手算答案
- Task 3 的 collapse 在 LayerNorm / softmax case 上压到 2D（1 Tileable + 1 Full）
- Task 5 生成的 IR 通过 FileCheck：外层 `scf.for` 带 `ascendc.parallel`、body 内所有 chain 成员都被 fuse（没有留在 loop 外的 linalg op）
- Task 6 写入的 `tiling.tiles` 仍能被现有 Phase 3 autotuner（`tools/agentic_autotuner`）读取（兼容投影不破坏）
- Task 6 写入的 `tiling.infos` 包含完整的 `TileInfo`：每个字段有 `abiIndex`、`kind`、`defaultExpr`；`blockDimExprs` 非空；`TilePlan` 里的 `OpFoldResult` 全部转成 `ValueExpr`

---

## 3. 关键数据结构

```cpp
// include/Conversion/VectorPlanGeneration/Analysis.h

namespace mlir::afir::vectorplan {

enum class AxisKind : uint8_t { Parallel, Reduction };

struct AxisInfo {
  StringRef name;              // "B", "S", "H"; 可为空
  int64_t   staticSize;        // ShapedType::kDynamic 表示动态
  AxisKind  kind;              // Parallel (Tileable) 或 Reduction (Full)
};

struct ChainInfo {
  linalg::LinalgOp          root;
  SmallVector<linalg::LinalgOp> members;    // 程序序，root 在最后
  SmallVector<AxisInfo>     canonicalAxes;  // root 的 iter space
  
  // boundary：chain 外部
  SmallVector<Value>        boundaryIn;     // chain 之外流入的 tensor
  SmallVector<Value>        boundaryOut;    // chain 流出的 tensor
  
  // 诊断
  std::string               cutReason;      // 若本 chain 是因 cut 形成
};

struct CollapsedChainInfo : ChainInfo {
  // collapse 之后的 canonical 轴（可能比原 canonicalAxes 短）
  SmallVector<AxisInfo>     collapsedAxes;
  // 每根原始轴落在哪个 collapsed 轴（原轴 idx → 新轴 idx）
  SmallVector<int>          axisMap;
};

struct TileParam {
  StringRef    name;
  Value        ssa;           // 由 pass 插入的 func index arg
  // 写入 tiling.tiles / TileInfo.defaultExpr；
  // 静态 shape → IntegerAttr；动态 shape → Value（tensor.dim / DimExpr）
  OpFoldResult defaultValue;
  int32_t      axisIdx;       // 对应 collapsedAxes 的下标
};

struct TilePlan {
  const CollapsedChainInfo* chain;
  
  // tileable[0] 永远是 {XBLOCK, XBLOCK_SUB} 两个参数
  // tileable[i>0] 永远是 {XBLOCK_SUB_i} 一个参数
  SmallVector<SmallVector<TileParam>> tileable;
  // full[j] 永远是 {RBLOCK_j} 一个参数
  SmallVector<TileParam>              full;

  // block_dim 表达式，由 Task 4 在生成 TilePlan 时显式计算并存储：
  //   blockDimExprs[0] = ceildiv(extent(tileable[0] axis), XBLOCK)
  // v1 只填一个元素；将来支持 2D grid 时追加 blockDimExprs[1]。
  // 此字段用于 Task 6 构建 TileInfo，不直接参与 realize。
  SmallVector<OpFoldResult, 2>        blockDimExprs;
};

// 主分析入口
FailureOr<SmallVector<ChainInfo>>          analyzeChains(func::FuncOp);
FailureOr<SmallVector<CollapsedChainInfo>> collapseChains(
    MutableArrayRef<ChainInfo>, IRRewriter&);
FailureOr<SmallVector<TilePlan>>           generatePlans(
    ArrayRef<CollapsedChainInfo>, func::FuncOp, IRRewriter&);
LogicalResult                              realizePlans(
    ArrayRef<TilePlan>, IRRewriter&);

} // namespace
```

---

## 4. Tasks

### Task 0: API 可行性 spike（前置硬门禁）

**目的**：在写具体 pass 之前，确认两个 upstream API 能真正支撑整条流水线。否则整个方案需要降级。

**Files**:
- Create: `tools/afir-opt/spike/tile_fuse_spike.cpp`
- Create: `test/VectorPlan/spike_row_mean.mlir`
- Create: `test/VectorPlan/spike_softmax.mlir`
- Create: `test/VectorPlan/spike_collapse.mlir`

- [ ] **Step 0.0**: 写最小 `spike_row_mean.mlir` —— 静态 shape `16x128` 的 row-mean：`linalg.fill → linalg.reduce(addf, dim=1) → linalg.generic(div)`。用 driver 对最后的 div 调 `tileConsumerAndFuseProducersUsingSCF`，`tileSizes = [XBLOCK_SUB_val=4]`，`fusionControlFn = FuseOnly`：
  - **§1.5 不变量断言**（FileCheck）：
    1. fuse 后 IR 中 `linalg.fill` 与 `linalg.reduce` 都出现在**内层** `scf.for` body 内；
    2. reduce 结果 tensor 的静态类型是 `tensor<4xf32>`（= `XBLOCK_SUB`），**不是** `tensor<16xf32>`；
    3. 外层 loop 之前只剩 `tensor.empty`（chain 边界 init），不存在对 reduce 中间 buffer 的全长 `insert_slice`。
  - **触发降级**：若任一断言失败，按 §1.5 的 fallback 判定流程走方案 A（手动内移 init post-pass）。方案 A 也失败则走方案 B（独立 `--local-buffer-downgrade` pass）。
  - **这一步通过，再做 Step 0.1 softmax spike**；row_mean 是单 reduction 的最小形态，bug 定位半径最小。

- [ ] **Step 0.1**: 写一个 transient driver，手工构造一个最小 softmax IR（`max → sub → exp → sum → div`），用 `tileConsumerAndFuseProducersUsingSCF` 对 div 做 tile：
  - `SCFTileAndFuseOptions.tilingOptions.tileSizes = [XBLOCK_SUB_val, 0]`（外层 parallel 切，内层 S_col 不切）
  - `SCFTileAndFuseOptions.fusionControlFn` 返回 `ControlFnResult::FuseOnly` 允许吸收所有 producer
  - 断言生成 IR 中 5 个 linalg op 全部落在 `scf.for` body 内，loop 外只剩 `tensor.empty`

- [ ] **Step 0.2**: 在 Step 0.1 的 driver 里把 `XBLOCK_SUB_val` 换成 **func 的 index arg**（SSA 值），重跑。确认 upstream API 接受 dynamic tile size 而不是强制 IntegerAttr。
  - 若不接受 → spike 记录失败，Task 5 需要退路：在 tile size 位置先填常量 `1`，然后后置重写 `scf.for` 的 upperBound / step。

- [ ] **Step 0.3**: 写第二个 driver，对一个 `linalg.generic` 显式调用 `linalg::collapseOpIterationDims({(0,1)})`，验证：
  - 返回的新 op 的 indexing_maps / iterator_types 正确
  - 上下游的 tensor shape 改变，需要我们在 boundary 插 `collapse_shape` / `expand_shape`
  - 若 upstream 报错（比如 indexing_map 在组内不连续）→ 记录该形态为"不可 collapse"

- [ ] **Step 0.4**: 写 scope 决定文档 `docs/superpowers/specs/spike-outcomes.md`，列出：
  - **§1.5 local buffer 不变量**在 row_mean 和 softmax 上是否被默认路径自动满足；若未满足，写明将走 §1.5 的方案 A 还是方案 B，并更新 §1.5 顶部的"默认路径"指向
  - API 是否支持 symbolic tile size
  - `collapseOpIterationDims` 的适用条件
  - 若任一 API 不支持 → 给出 Task 5 / Task 3 的降级方案

**验证**:
```bash
cmake --build build --target vector-plan-spike
./build/bin/vector-plan-spike test/VectorPlan/spike_row_mean.mlir
./build/bin/vector-plan-spike test/VectorPlan/spike_softmax.mlir
./build/bin/vector-plan-spike test/VectorPlan/spike_collapse.mlir
```

Spike 产出必须落盘（提交），Task 1 才能开始。

---

### Task 1: vector-plan-generation pass 骨架

**Files**:
- Create: `include/Conversion/VectorPlanGeneration/VectorPlanGeneration.h`
- Create: `include/Conversion/VectorPlanGeneration/Analysis.h`（见第 3 节数据结构）
- Create: `lib/Conversion/VectorPlanGeneration/VectorPlanGenerationPass.cpp`
- Create: `lib/Conversion/VectorPlanGeneration/CMakeLists.txt`
- Modify: `include/Conversion/Passes.td`（追加 TableGen 定义）
- Modify: `include/Conversion/Passes.h`（GEN_PASS_DECL / REGISTRATION 自动生效）
- Modify: `lib/Conversion/CMakeLists.txt`（加 subdirectory）
- Modify: `tools/afir-opt/CMakeLists.txt`（link 新 lib）

- [ ] **Step 1.1**: Passes.td 添加：

```tablegen
def VectorPlanGeneration : Pass<"vector-plan-generation", "func::FuncOp"> {
  let summary = "Analyze vector chains, collapse axes, and generate tiled kernels";
  let description = [{
    Transforms a linalg-on-tensor function into a Vector Phase 1 tiled form:
    chain analysis, axis collapse, complete TilePlan generation, and
    tileConsumerAndFuseProducersUsingSCF realization.
  }];
  let constructor = "mlir::afir::createVectorPlanGenerationPass()";
  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "affine::AffineDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
  ];
  let options = [
    Option<"enableCollapse", "enable-collapse", "bool", "true",
           "Whether to run axis collapse before tiling">,
    Option<"enforceCompletePlan", "enforce-complete-plan", "bool", "true",
           "Require every canonical axis to have a tile parameter">,
  ];
}
```

- [ ] **Step 1.2**: 骨架实现，runOnOperation 里按流水线串 4 个 helper：
```cpp
void VectorPlanGenerationPass::runOnOperation() {
  auto func = getOperation();
  IRRewriter rewriter(&getContext());
  
  FailureOr<SmallVector<ChainInfo>> chains = analyzeChains(func);
  if (failed(chains)) return signalPassFailure();
  
  FailureOr<SmallVector<CollapsedChainInfo>> collapsed;
  if (enableCollapse)
    collapsed = collapseChains(*chains, rewriter);
  else
    collapsed = liftWithoutCollapse(*chains);
  if (failed(collapsed)) return signalPassFailure();
  
  auto plans = generatePlans(*collapsed, func, rewriter);
  if (failed(plans)) return signalPassFailure();
  
  if (failed(realizePlans(*plans, rewriter)))
    return signalPassFailure();
}
```

- [ ] **Step 1.3**: 每个 helper 先实现为 stub，返回空 vector / success，只做日志打印：
```cpp
FailureOr<SmallVector<ChainInfo>> analyzeChains(func::FuncOp func) {
  llvm::errs() << "[vector-plan] analyzeChains on " << func.getName() << "\n";
  return SmallVector<ChainInfo>{};
}
// ... 其余 3 个类似
```

- [ ] **Step 1.4**: lit smoke test：
```mlir
// test/VectorPlan/pass_smoke.mlir
// RUN: afir-opt --vector-plan-generation %s -o - | FileCheck %s
// CHECK: func.func @empty
func.func @empty() { return }
```
跑通，确认 pass 能注册、能跑、不崩。

- [ ] **Step 1.5**: commit
```
feat(vector-plan): scaffold vector-plan-generation pass
```

---

### Task 2: Chain Analysis

**Files**:
- Create: `lib/Conversion/VectorPlanGeneration/ChainAnalysis.cpp`
- Create: `test/VectorPlan/chain_pointwise.mlir`
- Create: `test/VectorPlan/chain_layernorm.mlir`
- Create: `test/VectorPlan/chain_softmax.mlir`
- Create: `test/VectorPlan/chain_cut.mlir`

- [ ] **Step 2.1**: op 分类 helper
```cpp
enum class OpTrack { Vector, Cube, NotTileable };

OpTrack classify(Operation* op) {
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) return OpTrack::Cube;
  if (auto lin = dyn_cast<linalg::LinalgOp>(op)) {
    // 所有 indexing_map 必须是 affine projective（不能有 mod / floordiv / symbol）
    for (AffineMap m : lin.getIndexingMapsArray())
      if (!m.isProjectedPermutation(/*allowZeroInResults=*/true))
        return OpTrack::NotTileable;
    return OpTrack::Vector;
  }
  // tensor.extract_slice/insert_slice/expand/collapse → NotTileable 边界
  return OpTrack::NotTileable;
}
```

- [ ] **Step 2.2**: Root 识别 + 反向生长（block 级扫描，不用 walk）
```cpp
FailureOr<SmallVector<ChainInfo>> analyzeChains(func::FuncOp func) {
  DenseSet<Operation*> claimed;
  SmallVector<ChainInfo> chains;
  
  Block& body = func.getBody().front();
  for (Operation& op : llvm::reverse(body)) {
    if (claimed.count(&op)) continue;
    if (classify(&op) != OpTrack::Vector) continue;
    if (!isRootCandidate(&op, claimed)) continue;  // 输出是否流出 Vector
    
    ChainInfo chain;
    chain.root = cast<linalg::LinalgOp>(&op);
    growChainBackward(chain, claimed);
    chains.push_back(std::move(chain));
  }
  std::reverse(chains.begin(), chains.end());
  return chains;
}
```

- [ ] **Step 2.3**: `growChainBackward` 核心 —— 维护 `Tileable` / `Full` 轴集合的动态更新

```cpp
void growChainBackward(ChainInfo& chain, DenseSet<Operation*>& claimed) {
  // 初始 canonical 轴 = root 的 iteration space
  initCanonicalAxes(chain);
  claimed.insert(chain.root.getOperation());
  chain.members.push_back(chain.root);
  
  // 候选 producer 的宽度优先队列（按 SSA 逆序）
  SmallVector<linalg::LinalgOp> worklist = collectDirectProducers(chain.root);
  
  while (!worklist.empty()) {
    linalg::LinalgOp p = worklist.pop_back_val();
    if (claimed.count(p)) continue;
    if (classify(p) != OpTrack::Vector) {
      recordBoundaryInput(chain, p->getResult(0));
      continue;
    }
    
    // 计算 p 吸收后的 Tileable / Full 变化
    auto delta = computeAxisDelta(chain, p);
    if (!delta.has_value() || 
        !delta->tileableNonEmpty() || 
        delta->hasConflict()) {
      recordBoundaryInput(chain, p->getResult(0));
      chain.cutReason = formatCutReason(p, *delta);
      continue;
    }
    applyAxisDelta(chain, *delta);
    chain.members.push_back(p);
    claimed.insert(p);
    appendProducersToWorklist(p, worklist);
  }
  // 记录 boundary output
  recordBoundaryOutputs(chain);
  // 按 program order 排 members（root 在末尾）
  std::sort(chain.members.begin(), chain.members.end(), cmpByProgramOrder);
}
```

- [ ] **Step 2.4**: 四条 lit 测试覆盖判据边界

  **chain_pointwise.mlir**: QKV bias + relu 级 chain，期望 1 条 chain，members=2，axes 全 Parallel
  ```mlir
  // CHECK: chain 0: members=2, tileable={d0,d1}, full={}
  ```

  **chain_layernorm.mlir**: 简化的 LayerNorm chain（mean → sub → var → sqrt → div → affine）
  ```mlir
  // CHECK: chain 0: members=6, tileable={d0}, full={d1}
  ```

  **chain_softmax.mlir**: `max → sub → exp → sum → div`，双 reduction
  ```mlir
  // CHECK: chain 0: members=5, tileable={d0,d1,d2}, full={d3}
  ```

  **chain_cut.mlir**: 故意构造一个 Tileable / Full 冲突 —— chain 1 reduction over d0，chain 2 parallel over d0 but chain 2 reads chain 1's output via 冲突 map
  ```mlir
  // CHECK: chain 0: members=1, cut_reason="axis d0: Tileable/Full conflict introduced by ..."
  // CHECK: chain 1: members=1
  ```

- [ ] **Step 2.5**: 打印 helper + `--mlir-print-local-scope` + `--debug-only=vector-plan` 方便人工核对

- [ ] **Step 2.6**: commit
```
feat(vector-plan): chain analysis with outer-loop fusion support
```

---

### Task 3: Chain Collapse

**Files**:
- Create: `lib/Conversion/VectorPlanGeneration/ChainCollapse.cpp`
- Create: `test/VectorPlan/collapse_layernorm.mlir`
- Create: `test/VectorPlan/collapse_softmax.mlir`
- Create: `test/VectorPlan/collapse_refuse.mlir`

- [ ] **Step 3.1**: 候选组 + per-input 分类

```cpp
// 第一步：找连续同类型轴作为候选，不考虑 per-input 限制
SmallVector<std::pair<int,int>> findCandidateGroups(const ChainInfo& chain);

// 第二步：对 (map, group) pair 做 A/B/C 分类
enum class GroupPresence { AbsentA, PresentC, PartialB };
GroupPresence classifyMapAgainstGroup(
    AffineMap map, std::pair<int,int> group);
```

- [ ] **Step 3.2**: Case B 物化

```cpp
// 对 chain 中所有触发 Case B 的 (member, operand) 对：
//   1. 在 member 之前插入 linalg.broadcast，expand 到 member iter rank
//   2. member 的对应 operand 改为 broadcast 结果
//   3. 物化 op 打 attribute `vector_plan.no_collapse = true`
//   4. chain.members 追加物化 op（位置在原 member 之前）
// 返回新增的物化 op 列表，供后续 collapse 跳过
SmallVector<linalg::LinalgOp> materializePartialBroadcasts(
    ChainInfo& chain, ArrayRef<std::pair<int,int>> groups, IRRewriter& rw);
```

**约束**：物化 op 的 iter rank 必须 = consumer iter rank；dimensions 参数从原始 Case B map 反推（原 map 中缺失的 iter dim 就是要 broadcast 的 dim）。

- [ ] **Step 3.3**: 应用 collapse

```cpp
FailureOr<SmallVector<CollapsedChainInfo>>
collapseChains(MutableArrayRef<ChainInfo> chains, IRRewriter& rw) {
  SmallVector<CollapsedChainInfo> out;
  for (auto& chain : chains) {
    auto groups = findCandidateGroups(chain);
    
    CollapsedChainInfo cc;
    static_cast<ChainInfo&>(cc) = chain;
    
    if (groups.empty()) {
      // 不 collapse，canonical axes 原样
      cc.collapsedAxes.assign(chain.canonicalAxes.begin(),
                              chain.canonicalAxes.end());
      cc.axisMap.resize(chain.canonicalAxes.size());
      std::iota(cc.axisMap.begin(), cc.axisMap.end(), 0);
      out.push_back(std::move(cc));
      continue;
    }
    
    // 先物化所有 Case B
    auto materialized = materializePartialBroadcasts(cc, groups, rw);
    
    // 对非物化的 chain 成员应用 collapseOpIterationDims
    for (auto& m : cc.members) {
      if (m->hasAttr("vector_plan.no_collapse")) continue;  // 物化产物，保留原秩
      auto res = linalg::collapseOpIterationDims(m, groups, rw);
      if (failed(res)) return failure();
      // 更新 chain 里的 op 引用为 res->op
    }
    
    // 边界 reshape：
    //   - chain IO 边界（外部 producer/consumer 和 chain 之间）
    //   - 物化 op 输出和下游 collapsed consumer 之间
    insertBoundaryReshapes(cc, groups, materialized, rw);
    
    buildCollapsedAxes(cc, groups);
    out.push_back(std::move(cc));
  }
  return out;
}
```

- [ ] **Step 3.4**: 4 条 lit 测试

  **collapse_layernorm.mlir**: `[B, S, H]` → `[B*S, H]`，scale/bias Case A，consumer Case C，断言无 `linalg.broadcast` 被物化
  ```
  // CHECK: collapsed chain 0: tileable={d0'}, full={d1'}
  // CHECK: tensor.collapse_shape %{{.*}} {{\[\[0, 1\], \[2\]\]}}
  // CHECK-NOT: vector_plan.no_collapse
  ```

  **collapse_softmax.mlir**: `[B, 4, S, S]` → `[B*4*S, S]`，全 Case C
  ```
  // CHECK: collapsed chain 0: tileable={d0'}, full={d1'}
  // CHECK-NOT: vector_plan.no_collapse
  ```

  **collapse_case_b_materialize.mlir**: 故意构造一个 `(m, n) -> (m)` 的 input、collapse 组 `{m, n}`，断言物化路径触发
  ```mlir
  // 输入: linalg.generic 读 [M] 用 (m,n) -> (m)，输出 [M,N]
  // CHECK: linalg.broadcast
  // CHECK-SAME: vector_plan.no_collapse
  // CHECK: tensor.collapse_shape
  // CHECK: linalg.generic {{.*}} iterator_types = ["parallel"]
  // CHECK-SAME: (e) -> (e)
  ```

  **collapse_refuse.mlir**: 故意构造一个 producer 的 reduction 落在候选组中间，使整组无候选可 collapse，断言 chain 保留原始形态
  ```
  // CHECK: collapsed chain 0: tileable={d0,d1,d2}, full={d3}
  // CHECK-NOT: tensor.collapse_shape
  ```

- [ ] **Step 3.5**: commit
```
feat(vector-plan): chain collapse with Case B broadcast materialization
```

---

### Task 4: TilePlan Generation

**Files**:
- Create: `lib/Conversion/VectorPlanGeneration/TilePlanGen.cpp`
- Create: `test/VectorPlan/plan_layernorm.mlir`
- Create: `test/VectorPlan/plan_softmax.mlir`
- Create: `test/VectorPlan/plan_multi_axis.mlir`

- [ ] **Step 4.1**: TilePlan 构建函数

```cpp
FailureOr<SmallVector<TilePlan>> generatePlans(
    ArrayRef<CollapsedChainInfo> chains, func::FuncOp func, IRRewriter& rw) {
  // 统计需要注入的 index arg 总数
  int totalParams = 0;
  for (auto& cc : chains) totalParams += countTileParams(cc);
  
  // 一次性追加到 func signature
  SmallVector<Value> newArgs = appendIndexArgs(func, totalParams, rw);
  
  SmallVector<TilePlan> plans;
  int cursor = 0;
  for (auto& cc : chains) {
    TilePlan plan = buildPlanForChain(cc, newArgs, cursor);
    plans.push_back(std::move(plan));
  }
  return plans;
}
```

- [ ] **Step 4.2**: `buildPlanForChain` 按 1.3 节规则写：第一个 Parallel 两级 split，其余 Parallel 每个一级 tile，所有 Reduction 每个一级 tile。名字规则固定。

  **同时计算 `blockDimExprs`**：
  ```cpp
  // v1 只填一个元素：ceildiv(extent of tileable[0] axis, XBLOCK)
  // extent 优先从 collapsedAxes[tileable0Idx].staticSize 取；
  // 动态 shape 下用 tensor.dim（返回 Value，包装成 OpFoldResult）。
  Value xblock = plan.tileable[0][0].ssa;  // XBLOCK 的 SSA
  OpFoldResult extent = getAxisExtent(cc, tileableAxes[0], rw);
  Value blockDim = rw.create<arith::CeilDivUIOp>(loc, extent, xblock);
  plan.blockDimExprs.push_back(blockDim);
  ```

- [ ] **Step 4.3**: lit 测试

  **plan_layernorm.mlir**（collapse 后 2D）：
  ```
  // CHECK: func.func @f({{.*}}, %XBLOCK: index, %XBLOCK_SUB: index, %RBLOCK_0: index)
  ```

  **plan_softmax.mlir**（collapse 后 2D）：同上

  **plan_multi_axis.mlir**（故意禁用 collapse，`--enable-collapse=false`）：
  ```
  // CHECK: func.func @f({{.*}}, %XBLOCK: index, %XBLOCK_SUB: index,
  // CHECK-SAME: %XBLOCK_SUB_1: index, %XBLOCK_SUB_2: index, %RBLOCK_0: index)
  ```

- [ ] **Step 4.4**: commit
```
feat(vector-plan): complete TilePlan generation with symbolic tile args
```

---

### Task 5: TilePlan Realization

**Files**:
- Create: `lib/Conversion/VectorPlanGeneration/TilePlanRealize.cpp`
- Create: `test/VectorPlan/realize_pointwise.mlir`
- Create: `test/VectorPlan/realize_layernorm.mlir`
- Create: `test/VectorPlan/realize_softmax.mlir`

- [ ] **Step 5.1**: 把 TilePlan 翻译成 tile sizes SmallVector

```cpp
SmallVector<OpFoldResult> makeTileSizes(const TilePlan& plan) {
  SmallVector<OpFoldResult> sizes(plan.chain->collapsedAxes.size(),
                                  rw.getIndexAttr(0));  // 0 = 不 tile
  // outer Tileable：两级 split → 外层填 XBLOCK_SUB，先 tile 成最内
  // （upstream API 要求一次 tile；两级 split 用两次 tile 串联）
  // 详见 Step 5.2
  return sizes;
}
```

- [ ] **Step 5.2**: **两级 split 的实现方式**。Upstream `tileConsumerAndFuseProducersUsingSCF` 一次只产一层 scf.for。两级 split 的做法：
  1. 第一次 tile（outer）：tile sizes = `{XBLOCK, 0, 0, ..., 0}`（其余轴不 tile）
  2. 第二次 tile（inner，在 outer 结果上再 tile）：tile sizes = `{XBLOCK_SUB, 0, ..., 0}`
  3. 第三次 tile（reduction，如果 RBLOCK 不是 full）：tile sizes = `{0, ..., RBLOCK_j, ...}`
  4. 如果其余 Tileable 轴 / Full 轴默认 full，就不进入 tile 调用（tile size = 0 即可）

  每一次 tile 都要带 fusionControlFn 限制在 chain 成员之内：
  ```cpp
  opts.tilingOptions.setTileSizes(sizes);
  opts.fusionControlFn = [&](OpOperand* operand, OpResult producer,
                             bool isDestinationOperand) {
    auto op = producer.getDefiningOp();
    if (llvm::is_contained(chain.members, cast<linalg::LinalgOp>(op)))
      return SCFTileAndFuseOptions::ControlFnResult{/*yieldProducerReplacement=*/false};
    return std::nullopt;
  };
  ```

- [ ] **Step 5.3**: 外层 loop 打 `ascendc.parallel = true` 属性

- [ ] **Step 5.4**: 后置跑 LICM（针对 broadcast producer 的 loop-invariant 情况）
```cpp
// pass 末尾
OpPassManager nested("func.func");
nested.addPass(createLoopInvariantCodeMotionPass());
if (failed(runPipeline(nested, func))) return signalPassFailure();
```

- [ ] **Step 5.5**: lit 测试

  **realize_pointwise.mlir**（纯 elementwise）：
  ```
  // CHECK: scf.for %[[i_outer:.*]] = %c0 to %{{.*}} step %XBLOCK {
  // CHECK: } {ascendc.parallel = true}
  // CHECK:   scf.for %[[i_inner:.*]] = {{.*}} step %XBLOCK_SUB {
  // CHECK:     linalg.generic
  ```

  **realize_layernorm.mlir**（LayerNorm）：
  ```
  // CHECK: scf.for %{{.*}} step %XBLOCK
  // CHECK: } {ascendc.parallel = true}
  // CHECK:   scf.for %{{.*}} step %XBLOCK_SUB
  // CHECK:     linalg.generic {{.*}} iterator_types = ["parallel", "reduction"]
  // CHECK:     linalg.generic {{.*}} iterator_types = ["parallel"]
  ```
  body 内应有至少 2 个 linalg.generic（reduction + pointwise），外部不剩 chain 成员。

  **realize_softmax.mlir**（softmax）：
  ```
  // CHECK: scf.for %{{.*}} step %XBLOCK
  // CHECK: } {ascendc.parallel = true}
  // CHECK-COUNT-5: linalg.generic
  ```

- [ ] **Step 5.6**: commit
```
feat(vector-plan): realize TilePlan via tileConsumerAndFuseProducersUsingSCF
```

---

### Task 6: Module Metadata

**Files**:
- Create: `lib/Conversion/VectorPlanGeneration/ModuleMetadata.cpp`
- Create: `test/VectorPlan/metadata_layernorm.mlir`
- Create: `test/VectorPlan/metadata_tileinfo.mlir`

- [ ] **Step 6.1**: 收集 tile 参数 → 写 `tiling.tiles`（兼容投影，保持对当前 autotuner 的兼容性）

```
module attributes {
  tiling.tiles = [
    {name = "XBLOCK",     default = 256 : i64, axis = 0 : i64},
    {name = "XBLOCK_SUB", default = 64  : i64, axis = 0 : i64},
    {name = "RBLOCK_0",   default = 128 : i64, axis = 1 : i64, kind = "full"}
  ]
}
```

- [ ] **Step 6.2**: 收集 `func.func` 的原始 tensor 参数 → 写 `tiling.shapes`（从 `func.getArgument(i).getType()` 取，而非从 collapsed axes 反推）

- [ ] **Step 6.3**: **TilePlan → TileInfo 转换，写 `tiling.infos`**（权威表示）

  转换规则（对应 `docs/superpowers/specs/2026-04-14-tile-info-design.md` §8.1）：

  ```cpp
  TileInfo buildTileInfo(const TilePlan& plan, int chainId, int planId,
                         func::FuncOp func) {
    TileInfo info;
    info.kernelId = "chain" + str(chainId) + "_plan" + str(planId);
    info.chainId = chainId;
    info.planId  = planId;

    // 1. axes: 每根 collapsedAxis → TileAxisInfo
    for (auto [i, ax] : enumerate(plan.chain->collapsedAxes)) {
      TileAxisInfo axInfo;
      axInfo.axisIndex  = i;
      axInfo.axisName   = ax.name;
      axInfo.role       = ax.kind == AxisKind::Parallel
                            ? AxisRole::Parallel : AxisRole::Reduction;
      axInfo.extentExpr = opFoldResultToValueExpr(getAxisExtent(*plan.chain, i));
      info.axes.push_back(std::move(axInfo));
    }

    // 2. fields: tile params → TileFieldSpec（TunableTile / FixedTile）
    int abiIdx = 0;
    for (auto& params : plan.tileable)
      for (auto& p : params)
        info.fields.push_back(makeTunableField(p, abiIdx++));
    for (auto& p : plan.full)
      info.fields.push_back(makeFixedField(p, abiIdx++));

    // 3. shape fields: func 原始 tensor 参数的 primitive dim → ShapeDim
    for (auto [argIdx, arg] : enumerate(func.getArguments())) {
      auto shaped = dyn_cast<ShapedType>(arg.getType());
      if (!shaped) continue;
      for (int dimIdx = 0; dimIdx < shaped.getRank(); ++dimIdx)
        info.fields.push_back(makeShapeField(argIdx, dimIdx, shaped, abiIdx++));
    }

    // 4. blockDimExprs: 直接从 TilePlan 转成 ValueExpr
    for (auto& e : plan.blockDimExprs)
      info.blockDimExprs.push_back(opFoldResultToValueExpr(e));

    return info;
  }
  ```

  关键子函数：
  - `opFoldResultToValueExpr`：`IntegerAttr` → `ValueExpr{Const}`；`tensor.dim` result → `ValueExpr{ShapeDim}`；`arith.ceildivui(x, y)` → `ValueExpr{CeilDiv, lhs, rhs}`；递归处理 `arith.muli`。
  - `makeTunableField`：`kind=TunableTile`，`abiIndex=abiIdx`，`defaultExpr=opFoldResultToValueExpr(p.defaultValue)`，`search.candidates` v1 填空（由 autotuner 外部注入候选值）。
  - `makeFixedField`：`kind=FixedTile`，`level=Full`，`defaultExpr` 转换同上。

  最后把所有 `TileInfo` 序列化为 `ArrayAttr` 挂在 module 上：
  ```
  tiling.infos = [{kernel = "chain0_plan0", ...}]
  ```

- [ ] **Step 6.4**: lit 测试

  **metadata_layernorm.mlir**：验证 `tiling.tiles` + `tiling.shapes` 结构（兼容性）

  **metadata_tileinfo.mlir**：验证 `tiling.infos` 关键字段：
  ```
  // CHECK: tiling.infos
  // CHECK-SAME: kernel = "chain0_plan0"
  // CHECK-SAME: abi_index = 0
  // CHECK-SAME: kind = "tunable"
  // CHECK-SAME: abi_index = 2
  // CHECK-SAME: kind = "fixed"
  // CHECK-SAME: block_dim
  // CHECK-SAME: ceildiv
  ```

- [ ] **Step 6.5**: commit
```
feat(vector-plan): emit tiling.infos as authoritative TileInfo; keep tiling.tiles/shapes as compat views
```

---

### Task 7: 集成测试

**Files**:
- Create: `test/VectorPlan/e2e_bias_relu.mlir`
- Create: `test/VectorPlan/e2e_layernorm.mlir`
- Create: `test/VectorPlan/e2e_softmax.mlir`
- Create: `test/VectorPlan/e2e_relu_broadcast_transpose.mlir`（复用 examples/ 现成 case）

- [ ] **Step 7.1**: bias+relu —— 纯 pointwise smoke test。linalg 输入 → 跑完整 pipeline → IR 含 `scf.for` + `ascendc.parallel`，chain 数 = 1

- [ ] **Step 7.2**: 简化 LayerNorm（静态 shape `16x128`）—— 验证 reduce-in-middle 的 chain 不被切开，collapse 后 2D

- [ ] **Step 7.3**: 简化 softmax（静态 shape `4x8x8`）—— 验证双 reduction outer-fuse，chain 数 = 1

- [ ] **Step 7.4**: `relu-broadcast-transpose` —— 复用 `examples/relu-broadcast-transpose/step0_input_out.mlir`，验证单 op chain、broadcast+transpose 完全由 indexing_map 承载

- [ ] **Step 7.5**: 跑一次完整 Phase 2 管线验证（只在 LayerNorm case 上）—— 接上 `bufferize → buffer-placement → linalg-to-ascendc → parallelize → codegen`，确认新 pass 的输出能被下游消费。如果某一步崩了，根因修回本 pass（而非改下游）。

- [ ] **Step 7.6**: commit
```
test(vector-plan): e2e integration tests for pointwise/LN/softmax/broadcast
```

---

## 5. 延期事项清单

| 项 | 原因 | 何时做 |
|---|---|---|
| Torch-MLIR 后处理集成（EliminateCfAssert、ExtractBroadcast canonicalize） | 独立 pass，不属本方案职责，本 pass 假设输入已规整 | 另发 plan |
| 多 plan 枚举（换外层轴、split reduction、flat 形态） | v1 只需要 1 个 default plan 能跑通 | v1.1 |
| Reduction splitting（RBLOCK < full） | UB 不够大的 case，v1 目标网络无此需求 | v1.2 |
| Kernel function outline（每 chain 拆独立 func） | Phase 2 能直接吃就地改写的 IR | 有 perf / 调度需求时 |
| 硬归一化（自动插入 transpose/broadcast 统一 layout） | 预处理 pass 已覆盖绝大多数 case；真正的硬归一化是独立复杂子问题 | 有反例网络时 |
| 动态 shape 端到端测试 | 本版基础设施保留 DimExpr，但不做 e2e 验证 | 动态 shape 支持整包做时 |
| TilePlan cost model / 启发式 | v1 用固定 default 值填 `tiling.tiles`，autotune 自己搜 | 有 benchmark 压力时 |
| AutoTuner 迁移到消费 `tiling.infos`（TileInfo 投影） | 对应 spec §11 Phase C；v1 autotuner 仍读 `tiling.tiles`，`tiling.infos` 作为并行输出 | Phase C 完成后旧路径 hard-fail 删除 |
| `TileInfo` AttrDef 类型化（TableGen AttrDef + verifier） | v1 用裸 DictionaryAttr 先走通，AttrDef 重构为独立任务 | TileInfo 稳定后 |
| `search.candidates` 填充策略 | v1 `TunableTile` 字段的候选值由 autotuner 外部注入，TileInfo 本身不存储 | AutoTuner migration 时同步 |

---

## 6. 自审清单

在 Task 0 spike 完成之前，本 plan 的任何 Task 都不要动。重读本文时检查：

- [ ] Task 2–5 每一步都有具体代码 / 测试 / 验证？
- [ ] Chain / Collapse / TilePlan 三个数据结构的字段与 Task 中用法一致？
- [ ] 测试 case 覆盖 bias+relu / LayerNorm / softmax / broadcast-transpose 四种形态？
- [ ] 没有"类似 Task N"的偷懒引用？
- [ ] Scope 里延期的事没有在 Task 里偷偷做？
- [ ] Task 0 的失败路径有明确降级方案？
- [ ] §1.1 是否写清楚 OLFSN 等价性、覆盖面、以及显式不覆盖的场景（外层扰动类）？
- [ ] §1.5 的 local buffer 降级不变量在 Task 5 / Task 7 测例里有显式 FileCheck 断言？
- [ ] Task 0 spike 先跑 row_mean 再跑 softmax？Step 0.4 的 spike-outcomes 文档是否回答了"走默认路径 / 方案 A / 方案 B"这道选择题？
- [ ] `TileParam.defaultValue` 使用 `OpFoldResult`，对动态 shape 开放？（§3 结构体已与 §1.3 文字对齐）
- [ ] `TilePlan.blockDimExprs` 在 Task 4 Step 4.2 里显式计算，不由 Task 6 反推？
- [ ] Task 6 写出的 `tiling.infos` 包含完整 TileInfo（axes + fields + blockDimExprs），对照 spec §6.3 数据结构？
