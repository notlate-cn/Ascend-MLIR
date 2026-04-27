# impl-03: Pass 2 — Tile Fuse

**设计依据**: [00-architecture.md](./00-architecture.md) §4–5, [00-data-model.md](./00-data-model.md) §5–6  
**前置**: impl-00（数据结构）；Pass 2 独立于 impl-01/02，可并行开发，但需要 kernel_group{N}.mlir 作为输入

---

## 定位

`vector-plan-tile-fuse` 是 func-level pass，对每个 `kernel_group{N}.mlir` 独立运行。
**不复用** `linalg::tileUsingForOp` / `tileAndFuseProducerOfSlice`，自行建 loop nest。
统一发射策略让 horizontal fusion（无 SSA 边的 sibling）天然工作。

内部三阶段：
```
Collapse  →  TilePlanGen  →  LoopNest + Emit
```

**关键约束**：
- Pass 2 不感知 `group_id`；func 内所有 linalg op 属于同一 group
- CubeGroup 跳过 Collapse（恒等映射）
- **BAII**（Broadcast Axis Independence Invariant）对所有 VectorGroup 硬性成立；详见下节

---

## 核心不变量：Broadcast Axis Independence Invariant (BAII)

对一个 VectorGroup `G` 与其 canonical iteration axes `A = {a_0, ..., a_{n-1}}`，定义**广播轴集合** `BCast(G)`：

```
BCast(G) = { a_i ∈ A | ∃ boundary input I ∈ inputs(G),
                       indexingMap(I) 在 a_i 上缺失 }
```

即：只要 group 内任一 boundary input 在某轴上是广播（affine map results 中不出现该 dim），该轴即为广播轴。

**BAII L1（Collapse 独立性）**：

> `BCast(G)` 中的任何轴不得出现在 `CollapsedGroupInfo::collapsedAxes` 内。
> 等价地：`axisMap[i] == -1` 对所有 `a_i ∈ BCast(G)` 成立。

**BAII L2（Loop-nest 外层性）**：

> 在 `TilePlan` 对应的 loop nest 中，对任一广播轴 `a_b ∈ BCast(G)` 与任一消费 `a_b` 的 tile 循环轴 `a_t`（`a_t ∉ BCast(G)` 且 `map(I).results` 中包含 `a_b` 或 `a_t`，I 为 boundary input），`a_b` 的循环层级必须在 `a_t` 之外。

**范围**：BAII 仅对 `GroupInfo::Kind::Vector` 生效。`Kind::Cube` 跳过 Collapse（恒等映射）且 matmul 的广播（如 bias）由硬件直接吸收，不覆盖。

**强度**：硬不变量。TileFuse 产出 `CollapsedGroupInfo` 与 `TilePlan` 时**必须**保证 BAII 成立；任何违反由 pass verifier 报错而非 warn。

**判据**：BAII 的判据是 **affine map 形态**（results 是否缺该 dim），**不是 size**。`indexingMap` 语法上缺轴即视为广播；即便运行时某轴 size=1 但 map 中出现，不作为广播处理（由硬件/运行时的 stride=0 机制消化）。

**工程依据**：AF（AutoFuse，CANN 图编译器）在三年以上工程实践中对 broadcast 轴的处理核心不是"怎么切"，而是"哪些轴不能按普通方式搜"（ATT schedule `ShouldSkipAxis` + `IsFromReduceSplit`、Reorder broadcast 轴优先级高于普通轴）。BAII 把这条经验固化为 MLIR 层的硬不变量。

**TileFuse 对 BAII 的落地路径**：

1. 上游保留 `linalg.broadcast`；TileFuse 入口的 **BroadcastAbsorb** 前置 pass 把广播语义吸收进 `linalg.generic::indexing_maps`（见下节）。
2. Phase 1 Collapse 的"广播轴剪枝"子步骤把 `BCast(G)` 从 candidate collapse 组 G 中切开，使 B1 分类（部分缺失）在设计层被消除——剩余只有 A/B2/C 三类。
3. Phase 2 TilePlanGen 把广播轴放入 `TilePlan::full`（step=1 逐值迭代），遵循 `kMaxFullLoopIters` escape。
4. Phase 3 LoopNest 的 loop order 规则 `[分核] → [BCast full] → [tile 内层] → [reduction]` 自然满足 L2；**GroupEmitter 主动把缺轴 Load 上浮到对应广播轴循环之外**（不假设后端 LICM）。
5. TileFuse 结束时 verifier 校验 L1/L2。

---

## 术语与索引空间约定

全 pass 内 `axisIdx` **统一指 post-collapse iteration dim 的新编号**。语义锁定如下：

**Collapse 对 iteration domain 的改写**：Phase 1 Collapse 重写 `linalg.generic` 的 iterator types
与 indexing maps，把每个保留下来的 collapse 子组合成单个新 iter dim。广播轴**不参与 collapse
但保留为独立 iter dim**。Collapse 后的 iteration domain 按以下顺序编号：

```
post-collapse iter dim = [collapsed_or_kept_axes ...] + [broadcast_axes ...]
                         └──────────── 0..|collapsedAxes|-1 ─────────────┘
                                                              └── |collapsedAxes|..n-1 ──┘
```

- `CollapsedGroupInfo::collapsedAxes[i]` 下标 `i` == post-collapse idx。
- `CollapsedGroupInfo::broadcastAxes` 存 **post-collapse idx**（不是原始 canonical idx），
  值范围 `[|collapsedAxes|, n)`，按 canonical 原始序（升序）排列。
- `CollapsedGroupInfo::axisMap[original_canonical_idx] -> post-collapse idx; -1 if absorbed into a collapsed group by mapping`
  BAII L1 的表述等价重写为：对 `a_i ∈ BCast(G)`，`axisMap[a_i]` 指向 `broadcastAxes` 中的项，
  而不是被合并进某个 collapsed 子组。
- `TileParam::axisIdx`、`TilePlan` 各字段、`LoopNestResult::loopIVs` 的 key 全部使用
  post-collapse idx。
- `SliceComputer` 里 `dimExpr.getPosition()` 返回的也是 post-collapse idx（Collapse pass
  重写 indexing map 时已经改编号）。

Debug / verifier 需要原始 canonical idx 时，通过 `canonicalAxisOf(postCollapseIdx)` 反查，
不混入主流程。

---

## 前置 Pass: BroadcastAbsorb

**位置**：TileFuse 入口前（逻辑上也可视作 TileFuse 的 Phase 0）。

**输入**：含 `linalg.broadcast` + `linalg.generic` 的 `kernel_group{N}.mlir`。

**输出**：等价语义，但 `linalg.broadcast` 被吸收进下游 `linalg.generic` 的 `indexing_maps`；无法吸收的 `linalg.broadcast` 保留为普通 op（此时该 broadcast 会物化）。

**吸收规则**（canonicalization 级别的 pattern）：

```
%b = linalg.broadcast ins(%x) dimensions=[d1]
%r = linalg.generic { indexing_maps = [map_b, ...] } ins(%b, ...)
    ↓
%r = linalg.generic { indexing_maps = [map_b ∘ drop_broadcast_dim, ...] }
     ins(%x, ...)
```

即：将下游 generic 对 broadcast 结果的访问 map 转换为对原始 input 的缺轴 map。

**吸收失败条件**（保留 `linalg.broadcast` 作为普通 op）：
- 消费者不是 `linalg.generic`（如被 `tensor.reshape` 等非仿射 op 截断）
- 下游 generic 的 iteration domain 与 broadcast 目标形状不匹配（罕见）

**失败后的语义**：保留的 `linalg.broadcast` 在 TileFuse 眼中是普通 elementwise op（与 `linalg.add` 同层），其输出是物化后的完整形状，下游 consumer 的 indexing_map 不缺轴 → 不产生 `BCast` 贡献。代价是一次物化搬运，但不破坏 BAII。v1 接受此开销；后续可扩展 pattern 覆盖更多 case。

**设计权衡**：不新增 `ViewOpAttrInfo` 一类的 Load 侧属性结构——MLIR 的 `indexing_maps` 已承载"哪些轴对哪个 operand 是广播"的信息，无需二次建模。

---

## Phase 1: Collapse — 简化迭代空间

文件：`lib/Conversion/VectorPlan/TileFuse/Collapse.cpp`

### 候选 collapse 组 G

从 canonical axes（上确界推导）中找**连续同类型**轴（连续 Parallel 或连续 Reduction）：

```cpp
SmallVector<SmallVector<int>> findCandidateGroups(ArrayRef<AxisInfo> canonicalAxes) {
  SmallVector<SmallVector<int>> groups;
  SmallVector<int> current;
  for (auto [i, ax] : enumerate(canonicalAxes)) {
    if (!current.empty() && ax.role != canonicalAxes[current.back()].role) {
      if (current.size() > 1) groups.push_back(current);
      current.clear();
    }
    current.push_back(i);
  }
  if (current.size() > 1) groups.push_back(current);
  return groups;
}
```

### 广播轴剪枝（BAII L1 落地）

在 `findCandidateGroups` 产出 candidate 之后、Per-input 分类之前，对每个 candidate G 按
`BCast(G)` 切开，丢弃 size < 2 的碎片。剪枝后 G 内不再包含任何广播轴 → B1 分类在设计层消失，
后续分类只需处理 A/B2/C 三类。

```cpp
// 输入：candidate group G，广播轴集合 BCast
// 输出：剪枝后若干连续非广播轴子组（size ≥ 2 才保留）
SmallVector<SmallVector<int>> pruneBroadcastAxes(
    ArrayRef<int> G, const DenseSet<int>& BCast) {
  SmallVector<SmallVector<int>> out;
  SmallVector<int> current;
  for (int d : G) {
    if (BCast.contains(d)) {
      if (current.size() >= 2) out.push_back(current);
      current.clear();
    } else {
      current.push_back(d);
    }
  }
  if (current.size() >= 2) out.push_back(current);
  return out;
}
```

剪枝示例：
- `G = [d0, d1, d2]`，`d1` 广播 → 剪出 `[d0]`（丢弃）+ `[d2]`（丢弃），整体不 collapse
- `G = [d0, d1, d2]`，`d2` 广播 → 剪出 `[d0, d1]` 保留
- `G = [d0, d1, d2, d3]`，`d1` 广播 → 剪出 `[d0]`（丢弃）+ `[d2, d3]` 保留

### Per-input 分类（A / B2 / C）

**前置条件**：此处的 G 是**广播轴剪枝后的**子组，已不含任何广播轴。因此 `resultG.size() < G.size()`
（原 B1）在此阶段不可能发生；若仍命中说明前置剪枝有 bug，fall through 到断言。

```cpp
enum class InputClass { A, B2, C };

InputClass classifyInput(AffineMap map, ArrayRef<int> G) {
  // 找 map results 中出现的 G 内轴
  DenseSet<int> resultG;
  for (auto expr : map.getResults())
    if (auto dim = dyn_cast<AffineDimExpr>(expr))
      if (llvm::is_contained(G, (int)dim.getPosition()))
        resultG.insert(dim.getPosition());

  if (resultG.empty()) return InputClass::A;    // G 完全缺失（纯外层轴）
  assert(resultG.size() == G.size() &&
         "broadcast axes should have been pruned from G; B1 must not occur here");

  // 所有 G 轴均在 result 中，检查是否连续同序
  SmallVector<int> positions;
  for (int g : G)
    for (auto [i, expr] : enumerate(map.getResults()))
      if (auto dim = dyn_cast<AffineDimExpr>(expr))
        if ((int)dim.getPosition() == g) { positions.push_back(i); break; }

  for (int i = 1; i < (int)positions.size(); ++i)
    if (positions[i] != positions[i - 1] + 1) return InputClass::B2;

  return InputClass::C;
}
```

#### 场景说明

分类依据：boundary input 的 `indexing_map` 中，候选 collapse 组 G 内各轴的出现方式。

| 分类 | 判定条件 | 直觉含义 |
|:---:|---------|---------|
| **A** | G 内轴**全部不出现**在 map results 中 | 该 input 完全不索引 G 方向，collapse 不影响它 |
| **B2** | G 内轴**全部出现但顺序不连续/乱序** | 该 input 对 G 轴有转置访问，collapse 会破坏语义 |
| **C** | G 内轴**全部出现且连续同序** | 最理想情况，可直接 collapse |

> **B1 已在设计层消除**：原 B1（G 内轴部分出现 = 该 input 在某轴广播）在 BAII 下走"广播轴剪枝"
> 路径——BCast 轴从 G 切开，不进入 collapse。剪枝后的 G 不可能再出现 B1。
> 决策理由见下文"决策档案：为何不走 B1（broadcast+collapse）路径"。

**典型场景举例**（以 LayerNorm `[B, S, H]`、候选 `G = {d_B, d_S}` 为例）：

- **Case A** — `scale[H]`：map 为 `(d_B, d_S, d_H) → (d_H)`，G 中的 `d_B`、`d_S` 完全未出现。
  collapse `G` 对该 input 的访问模式无任何影响，无需处理。

- **Case C** — `input[B, S, H]`：map 为 `(d_B, d_S, d_H) → (d_B, d_S, d_H)`，G 中
  `d_B` 在 result 位置 0、`d_S` 在位置 1，连续递增。可安全 collapse 为 `[B*S, H]`。

- **广播场景** — 若某 input `X[B, 1, H]` 对 `d_S` 广播（map `(d_B, d_S, d_H) → (d_B, d_H)`，
  d_S 缺失），`d_S` 会被归入 `BCast(G)` 由剪枝阶段从 G 中切开；candidate G 变为 `{d_B}`
  （单轴，不 collapse）。`d_S` 在 LoopNest 中走 full-loop 外层（BAII L2），Load `X` 在
  `d_S` 循环内 loop-invariant 由 LICM hoist。

- **Case B2** — input `T[S, B]`，consumer map 为 `(d_B, d_S) → (d_S, d_B)`，`G = {d_B, d_S}`：
  `d_B` 在 result 位置 1、`d_S` 在位置 0，不是连续递增——本质是转置访问。
  **处理**：生成两个 Variant 由 Autotuner 选优：
  - Variant 1：插 `linalg.transpose` 重排数据 → 打 `no_collapse` barrier，loop nest 按原始独立轴建立
  - Variant 2：修改 consumer 的 indexing map 使 B2→C，在 boundary input 打
    `load_with_transpose` 标记，由 GM→UB 搬运时通过硬件 ConfusionTranspose 完成重排

**严格排序 `A < C < B2`**：当同一 boundary input 在不同 op 中被分到不同 class 时，
取最严格的（B2 > C > A）。B2 对整个 group 的 collapse 策略影响最大——一旦存在 B2，
整个 group 需走双 Variant 路径（Phase 5 / Phase 6）。


#### 决策档案：为何不走 B1（broadcast + collapse）路径

以下为 BAII 设计的**核心证据**：把 broadcast 轴补齐后与其它轴一起 collapse 会产生实际
搬运量膨胀，而唯一救回路径（强制 tile size 为 reassoc 内层 size 的倍数 + strip-mine +
hoist）等价于反解 collapse，即退回"不 collapse broadcast 轴"的方案。AF 的 ATT schedule
在三年以上实践中也是硬跳过 broadcast split 轴。两条证据合流，确立 BAII。

此档案不反映当前实现（当前实现按剪枝路径走"方案 2"），仅作为 BAII 决策依据存档。

```
// ═══════════════════════════════════════════════════════
// B1 场景示例（具体数值）— 决策档案，非当前实现
// ═══════════════════════════════════════════════════════
//
// add + mul 融合 group
// x1 shape: [4, 1, 8]   ← B1 input（d_B 缺失）
// x2 shape: [4, 3, 8]
// x3 shape: [4, 3, 8]
// 候选 collapse 组 G = {d_A, d_B}，x1 的 map (d_A,d_B,d_C)→(d_A,d_C) 缺少 d_B

// ─────────────────────────────────────────────
// 方案 1：显式 broadcast + collapse + 切分
// ─────────────────────────────────────────────

// Step 1: 插入 linalg.broadcast，把 x1[4,1,8] 展开为 brc[4,3,8]
//
// brc 的内存布局（行优先）：
//   brc[0,0,:] = x1[0,0,:]   即 [x1_00 x1_01 ... x1_07]
//   brc[0,1,:] = x1[0,0,:]   ← 重复！和 brc[0,0,:] 完全相同
//   brc[0,2,:] = x1[0,0,:]   ← 重复！
//   brc[1,0,:] = x1[1,0,:]
//   brc[1,1,:] = x1[1,0,:]   ← 重复！
//   ...共 4×3=12 行，但只有 4 行独立数据（膨胀 3 倍）

// Step 2: collapse G={d_A, d_B} → d_AB，所有 tensor 变成 2D
//   brc[4,3,8]  → brc_flat[12, 8]
//   x2[4,3,8]   → x2_flat[12, 8]
//   x3[4,3,8]   → x3_flat[12, 8]

// Step 3: 切分 d_AB 轴 (XBLOCK=6, XBLOCK_SUB=4)
//   核数 = ceildiv(12, 6) = 2
//
// 核 0 处理 d_AB ∈ [0, 6)：
//   tile 0: d_AB=[0,3]  → brc_flat[0:4, :] = [x1[0,:], x1[0,:], x1[0,:], x1[1,:]]
//                                               ↑ 3 份相同数据，只有 2 行独立
//   tile 1: d_AB=[4,5]  → brc_flat[4:6, :] = [x1[1,:], x1[1,:]]
//                                               ↑ 2 份相同数据
//
// 核 1 处理 d_AB ∈ [6, 12)：
//   tile 2: d_AB=[6,9]  → brc_flat[6:10, :] = [x1[2,:], x1[2,:], x1[2,:], x1[3,:]]
//   tile 3: d_AB=[10,11] → brc_flat[10:12, :] = [x1[3,:], x1[3,:]]
//
// 问题：12 行数据搬了 12 次 DataCopy，但独立数据只有 4 行 → 搬运效率 33%

// ─────────────────────────────────────────────
// 方案 2：不显式 broadcast，缩小 collapse 组
// ─────────────────────────────────────────────

// Step 1: 发现 x1 在 d_B 缺失 → 从 G 剔除 d_B
//   G' = {d_A}（单轴，collapse 无实际效果）→ 保留原始三维

// Step 2: 对 d_A 做分核 tile，d_B 保留为独立循环，d_C 做内层 tile
//   XBLOCK=2 (分核粒度), XBLOCK_SUB=4 (C 方向 tile)
//   核数 = ceildiv(4, 2) = 2
//
// 核 0 处理 a ∈ [0, 2)：
for a = 0 to 4 step 2:                // 分核：核0 a∈[0,2), 核1 a∈[2,4)
  for b = 0 to 3:                     // d_B 完整遍历，不 collapse
    for c = 0 to 8 step 4:            // d_C tile

      // x1 slice: x1[a:a+2, 0, c:c+4]  ← 连续 2×4=8 个元素
      // 核 0, b=0, c=0: x1[0:2, 0, 0:4] = [[x1_00..x1_03], [x1_10..x1_13]]
      // 核 0, b=1, c=0: x1[0:2, 0, 0:4] = [[x1_00..x1_03], [x1_10..x1_13]]  ← 同上！
      // 核 0, b=2, c=0: x1[0:2, 0, 0:4] = [[x1_00..x1_03], [x1_10..x1_13]]  ← 同上！
      //
      // 关键：x1 不索引 b，所以 3 次迭代读的是完全相同的 x1 slice
      // → Codegen 识别 PureBroadcast map → 只 DataCopy 一次 → broadcast_l2 展开

      x1_slice = x1[a:a+2, 0, c:c+4]              // 8 元素，连续内存
      x2_slice = x2[a:a+2, b, c:c+4]              // 8 元素
      x3_slice = x3[a:a+2, b, c:c+4]              // 8 元素

      // Codegen 发射：
      //   DataCopy x1_slice (8 elem) GM → UB     ← 只搬独立数据
      //   broadcast_l2 展开到 [2, 1, 4]           ← 硬件零开销
      //   add_l2(x1_bcast, x2_slice)
      //   mul_l2(add_result, x3_slice)
      add_slice = broadcast_add(x1_slice, x2_slice)
      out_slice = mul(add_slice, x3_slice)

// ─────────────────────────────────────────────
// 数据搬运对比（以核 0 为例）
// ─────────────────────────────────────────────
//
// 方案 1 (broadcast+collapse):
//   x1 搬运：brc_flat[0:6, 0:8] = 48 elem（实际独立数据只有 16 elem）→ 效率 33%
//
// 方案 2 (保留 b 循环):
//   x1 搬运：x1[0:2, 0, 0:4] × 2(c tile) = 16 elem（可复用：b=0,1,2 读同一份）
//   如果 Codegen 识别到 b 循环内 x1 不变 → hoist 到 b 循环外 → 只搬 1 次
//   最终搬运：16 elem → 效率 100%
//
// | 指标        | 方案 1 (broadcast+collapse) | 方案 2 (保留 b 循环) |
// |------------|---------------------------|---------------------|
// | x1 内存     | 4×3×8 = 96 elem (膨胀 3x) | 4×1×8 = 32 elem (原始) |
// | 核 0 搬运   | 48 elem (含冗余)           | 16 elem (无冗余)     |
// | 循环层数     | 2 层 (d_AB + d_C)          | 3 层 (d_A + d_B + d_C) |
// | Codegen     | 需 fold pass 恢复 broadcast | 已有 PureBroadcast 路径 |
```

### Pre-Check

```cpp
struct CollapsePreCheck {
  SmallVector<int>                G;                 // 广播轴剪枝后的 G
  DenseMap<Value, InputClass>     inputClasses;      // boundary input → 最严格的 class
  bool hasB2 = false;
};

CollapsePreCheck preCheck(const GroupInfo &info, ArrayRef<int> G) {
  CollapsePreCheck result;
  result.G = SmallVector<int>(G);

  for (auto op : info.topoMembers) {
    auto lop = cast<linalg::LinalgOp>(op);
    for (auto [operand, map] :
         llvm::zip(lop.getInputs(), lop.getIndexingMapsArray())) {
      if (!isBoundaryInput(operand, info)) continue;
      auto cls = classifyInput(map, G);
      auto &stored = result.inputClasses[operand];
      // B2 优先级最高（严格排序：A < C < B2）
      if (classRank(cls) > classRank(stored)) stored = cls;
      if (cls == InputClass::B2) result.hasB2 = true;
    }
  }
  return result;
}
```

### B2 处理 — 生成两个 Variant

**Variant 1（Preserve）— 插 linalg.transpose，no_collapse barrier**:

```cpp
GroupInfo fixupB2Variant1(OpBuilder &builder, GroupInfo info,
                            const CollapsePreCheck &check) {
  for (auto &[boundary, cls] : check.inputClasses) {
    if (cls != InputClass::B2) continue;
    for (auto op : info.topoMembers) {
      auto lop = cast<linalg::LinalgOp>(op);
      for (auto [idx, operand] : enumerate(lop.getInputs())) {
        if (operand != boundary) continue;
        if (classifyInput(lop.getIndexingMapsArray()[idx], check.G)
            != InputClass::B2) continue;
        auto perm = computePermToCanonical(lop.getIndexingMapsArray()[idx], check.G);
        builder.setInsertionPoint(op);
        auto transposeOp = builder.create<linalg::TransposeOp>(
            op->getLoc(), operand,
            createTransposedInit(builder, operand, perm), perm);
        transposeOp->setAttr("vector_plan.no_collapse", builder.getUnitAttr());
        lop->setOperand(idx, transposeOp.getResult());
        insertBefore(info.topoMembers, op, transposeOp);
      }
    }
  }
  // 标记：含 no_collapse op，loop nest 走原始 G-axes 路径
  info.noCollapse = true;
  return info;
}
```

**Variant 2（Eliminate）— 改 consumer map + load_with_transpose 标记**:

```cpp
GroupInfo fixupB2Variant2(OpBuilder &builder, GroupInfo info,
                            const CollapsePreCheck &check) {
  DenseSet<Value> marked;
  for (auto &[boundary, cls] : check.inputClasses) {
    if (cls != InputClass::B2) continue;
    for (auto op : info.topoMembers) {
      auto lop = cast<linalg::LinalgOp>(op);
      for (auto [idx, operand] : enumerate(lop.getInputs())) {
        if (operand != boundary) continue;
        // 把 consumer map 改为 canonical（B2 → C）
        setIndexingMapToCanonical(lop, idx, check.G);
      }
    }
    // boundary input 打标（每个只打一次）
    if (!marked.count(boundary)) {
      // boundary 通常是 BlockArgument；在 func arg 上设 attr
      if (auto blockArg = dyn_cast<BlockArgument>(boundary))
        blockArg.getOwner()->getParentOp()->setAttr(
            "vector_plan.load_with_transpose", builder.getUnitAttr());
      marked.insert(boundary);
    }
  }
  return info;
}
```

### 测试用例（Phase 1）

```mlir
// test/Conversion/VectorPlan/broadcast-absorb-basic.mlir
// RUN: mlir-opt --vector-plan-broadcast-absorb %s | FileCheck %s
// linalg.broadcast + linalg.generic → 吸收为缺轴 generic
func.func @kernel_group0(%x: tensor<4x8xf16>, %y: tensor<4x3x8xf16>)
    -> tensor<4x3x8xf16> { ... }
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>

// test/Conversion/VectorPlan/broadcast-absorb-fallback.mlir
// 非 generic 消费者 → 保留 linalg.broadcast 作为普通 op
// CHECK: linalg.broadcast
// CHECK: tensor.reshape

// test/Conversion/VectorPlan/collapse-prune-middle-axis.mlir
// G = [d0, d1, d2]，d1 广播 → 整体不 collapse（两边碎片 size < 2）
// CHECK-NOT: tensor.collapse_shape

// test/Conversion/VectorPlan/collapse-prune-tail-axis.mlir
// G = [d0, d1, d2]，d2 广播 → 保留 [d0, d1] collapse
// CHECK: tensor.collapse_shape {{.*}} [[0, 1], [2]]

// test/Conversion/VectorPlan/tile-fuse-collapse-c.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
// LayerNorm [B, S, H] → 候选 G = {d_B, d_S}；scale/bias [H] → Case A；其余 → Case C
func.func @kernel_group0(%input: tensor<4x8x16xf16>,
                          %scale: tensor<16xf16>,
                          %bias: tensor<16xf16>) -> tensor<4x8x16xf16> {
  // ... linalg ops
}
// CHECK: tensor.collapse_shape {{.*}} into tensor<32x16xf16>
// CHECK: scf.for

// test/Conversion/VectorPlan/tile-fuse-collapse-b2.mlir
// B2 case：transpose+pointwise，input [S, B] 乱序
// CHECK: linalg.transpose  ← B2 Variant 1
//  或
// CHECK-NOT: linalg.transpose  ← B2 Variant 2（改 map + load_with_transpose）
```

---

## Phase 2: TilePlanGen — VectorGroup 基础路径

文件：`lib/Conversion/VectorPlan/TileFuse/TilePlanGen.cpp`

**适用条件**：`enableReductionSplit=false`，无 B2。

```cpp
TilePlan genVectorTilePlan(const CollapsedGroupInfo &info,
                            OpBuilder &builder, Location loc) {
  TilePlan plan;
  plan.group = &info;

  SmallVector<int> tileableIdx, reductionIdx;
  for (auto [i, ax] : enumerate(info.collapsedAxes)) {
    if (ax.role == AxisRole::Parallel)  tileableIdx.push_back(i);
    if (ax.role == AxisRole::Reduction) reductionIdx.push_back(i);
  }
  assert(!tileableIdx.empty() && "VectorGroup must have at least one tileable axis");

  // 第一个 tileable 轴：两级 (XBLOCK Outer + XBLOCK_SUB Inner)
  {
    int axIdx = tileableIdx[0];
    Value extent = getAxisExtentValue(info.collapsedAxes[axIdx], builder, loc);
    Value xblock    = insertFuncArg(builder, "XBLOCK", 256);
    Value xblockSub = insertFuncArg(builder, "XBLOCK_SUB", 64);
    plan.tileable.push_back({
      TileParam{"XBLOCK",     xblock,    IntegerAttr::get(i64, 256),
                axIdx, TileLevel::Outer, AxisRole::Parallel},
      TileParam{"XBLOCK_SUB", xblockSub, IntegerAttr::get(i64, 64),
                axIdx, TileLevel::Inner, AxisRole::Parallel},
    });
    // blockDimExprs[0] = ceildiv(extent, XBLOCK)
    plan.blockDimExprs.push_back(
        builder.create<arith::CeilDivSIOp>(loc, extent, xblock).getResult());
  }

  // 其余 tileable 轴：一级 Inner，defaultValue = full dim
  for (int i = 1; i < (int)tileableIdx.size(); ++i) {
    int axIdx = tileableIdx[i];
    Value extent = getAxisExtentValue(info.collapsedAxes[axIdx], builder, loc);
    plan.tileable.push_back({
      TileParam{formatv("XBLOCK_SUB_{0}", i), extent,
                /*default=*/getShapeDimAttr(builder, axIdx),
                axIdx, TileLevel::Inner, AxisRole::Parallel},
    });
  }

  // Reduction 轴：Full（不切）
  for (auto [j, axIdx] : enumerate(reductionIdx)) {
    Value extent = getAxisExtentValue(info.collapsedAxes[axIdx], builder, loc);
    plan.full.push_back(
      TileParam{formatv("RBLOCK_{0}", j), extent,
                getShapeDimAttr(builder, axIdx),
                (int)axIdx, TileLevel::Full, AxisRole::Reduction}
    );
  }

  return plan;
}
```

**广播轴处理（BAII L2 落地）**：

- 所有 `BCast(G)` 中的轴进 `plan.full`，`TileLevel::Full`，**step = 1 逐值迭代**。
  这是 L2 的关键语义：每个广播轴值对应一次外层循环迭代，使 tile 内层的 Load 在该循环内
  loop-invariant，GroupEmitter 才能将其主动上浮。若 step = extent（只跑 1 次）则退化为
  "整块加载"，Load hoist 无从发生。
- Escape：若某广播轴 `|axis| > kMaxFullLoopIters`（默认 2048，pass option 可覆盖），
  降级为 `tileable` 的单级 Inner（step > 1，防止循环深度/指令 overhead 爆炸）。降级后
  该轴 Load **不再可上浮**（成为普通 tile 轴），语义仍正确但性能退化到同 Case C。
- 多条广播轴按 **post-collapse idx** 升序（= canonical 原始序）进入 `plan.full`（D5：
  YAGNI，不引入未验证的代价模型）。
- `CollapsedGroupInfo::broadcastAxes`（见 §数据模型影响）在 Pre-Check 阶段填充，
  TilePlanGen 直接读取，避免重复遍历 `indexingMaps`。

```cpp
// TilePlanGen 中广播轴分支
// 约定：axIdx 是 post-collapse idx；broadcastExtents 单独维护（广播轴不在 collapsedAxes 里）
for (int axIdx : info.broadcastAxes) {
  Value extent = info.broadcastAxisExtents.at(axIdx);
  if (auto sz = getStaticSize(extent); sz && *sz > kMaxFullLoopIters) {
    // Escape 到 tileable Inner（step = 某 tile size，非 1）
    plan.tileable.push_back({
      TileParam{formatv("BCAST_TILE_{0}", axIdx),
                insertFuncArg(builder, formatv("BCAST_TILE_{0}", axIdx), 256),
                extent, axIdx, TileLevel::Inner, AxisRole::Parallel},
    });
  } else {
    // BCast Full：step = 1（由 ssa 字段承载），defaultValue = extent（供 UB 估算）
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    plan.full.push_back(
      TileParam{formatv("BCAST_{0}", axIdx), /*ssa=tile size=*/one,
                /*defaultValue=extent=*/getShapeDimAttr(builder, axIdx),
                axIdx, TileLevel::Full, AxisRole::Parallel}
    );
  }
}
```

> **TileParam 语义收紧**：`ssa` 统一表示 **"step / tile size"**（每次 loop 推进多少）；
> `defaultValue` 表示该轴的 **extent**（用于 UB 预算 / block_dim 计算）。
> 原"其余 tileable 轴 ssa = extent"的用法修正为：若某轴意图"不切一次 full 遍历"，
> 应使用 `TileLevel::Full` + ssa=extent（step=extent 跑 1 次拿整块），与 BCast Full 的
> step=1 明确区分。

**UB 预算 check**（**在 Phase 2 TilePlanGen 末尾**，不是 Phase 3）：检查
`∏(|非 reduction 广播轴|) × innerTileSize × sizeof(dtype) ≤ UB_capacity`。不满足时
**在 plan 生成阶段就 escape**，避免 Phase 3 回滚：优先降低默认 `XBLOCK_SUB`；仍不足则
对最大广播轴执行 `kMaxFullLoopIters` escape 路径（降级 tileable）。Phase 3 只消费 plan，
永不改动。

**分核候选回退**：若剔除广播轴后分核候选总可分核元素数 `< 核数 × 最小 tile`，
保留原 canonicalAxes 顺序但 emit 时 warn（非 verifier fail），由上层 / 人工评估。

**enableReductionSplit=true 修改**（Phase 4 实现，此处列出接口）：

```cpp
// full[j] 改为：
TileParam{formatv("RBLOCK_{0}", j),
          insertFuncArg(builder, formatv("RBLOCK_{0}", j), defaultVal),
          defaultVal,
          (int)axIdx, TileLevel::Inner,  // ← Full → Inner
          AxisRole::Reduction}
```

### 分核模型与两级 Tile 语义

TilePlan 通过 `TileLevel` + `ascendc.parallel` 属性表达**分核**与**每核处理量**：

```
数据总量 N
├── XBLOCK (Outer, ascendc.parallel)  ← 分核粒度：每核负责 XBLOCK 大小的数据块
│   └── XBLOCK_SUB (Inner)            ← 每核每次处理量：每次从 UB 取 XBLOCK_SUB 大小
```

| 概念 | 参数 | TileLevel | IR 表达 |
|------|------|-----------|---------|
| 核数 | `ceildiv(extent, XBLOCK)` | — | `blockDimExprs[0]`，Codegen 用于 dispatch |
| 每核总工作量 | `XBLOCK` | `Outer` | 外层 `scf.for` 带 `ascendc.parallel` |
| 每核每次处理 | `XBLOCK_SUB` | `Inner` | 内层 `scf.for`，范围 `[0, XBLOCK)` |
| Reduction 轴 | `RBLOCK` | `Full` / `Inner` | `Full` 时不建 loop；`Inner` 时建 RBLOCK loop |

生成的 loop nest 结构（VectorGroup，单 tileable 轴）：

```
// 核数 = ceildiv(N, XBLOCK)
scf.for XB = 0 to N step XBLOCK {ascendc.parallel}      // 分核边界
  scf.for XS = 0 to XBLOCK step XBLOCK_SUB              // 核内分批
    composedIV = XB + XS                                 // 实际 tensor offset
    extract_slice tensor[composedIV] size=XBLOCK_SUB
    compute...
    insert_slice → iterArg
```

CubeGroup 类似但为 2D 分核（`blockDimExprs` 有两个元素）：

```
// grid_y = ceildiv(M, BM),  grid_x = ceildiv(N, BN)
scf.for BM  {ascendc.parallel}     // M 维分核
  scf.for BN  {ascendc.parallel}   // N 维分核
    scf.for Tb_M                    // 核内 M 维分批
      scf.for Tb_N                  // 核内 N 维分批
        fill → acc                  // accumulator init
        scf.for t_K                 // K 维累加
          matmul(A_slice, B_slice, acc)
        epilogue(acc_result)        // bias_add, relu 等
```

**Codegen 接口**：Codegen 通过 `ascendc.parallel` 标记识别 dispatch loop，将其转换为
`block_idx` 索引（1D 或 2D）；`blockDimExprs` 告诉 runtime 需要多少个核。
Codegen 不从 loop 结构反推核数——核数由 `tiling.infos.block_dim` 权威提供。

---

## Phase 3: LoopNestBuilder + SliceComputer + GroupEmitter

### LoopNestBuilder

文件：`lib/Conversion/VectorPlan/TileFuse/LoopNestBuilder.cpp`

```cpp
struct LoopNestResult {
  DenseMap<int, Value> loopIVs;      // collapsed axis idx → composedIV
  Block               *innermostBody;
  SmallVector<Value>   iterArgs;     // group boundary outputs 的 iter arg
};

**Loop order 规则（D1，BAII L2 落地）**：从外到内顺序为

```
1. bind_multicore 轴（Outer tileable，带 ascendc.parallel）
2. BCast(G) 中的轴（plan.full，按 canonicalAxes 原始序）
3. tile 内层（Inner tileable，XBLOCK_SUB 等）
4. reduction 轴（plan.full 的 RBLOCK_*；enableReductionSplit=true 时走 Phase 4）
```

设计意图：
- NPU blockIdx 自然在最外；
- 每核独立持有 broadcast 复用数据（BCast full 在 tile 内层之外 → 每核 load 一次由硬件
  `broadcast_l2` 展开）；
- reduction 在 broadcast 内层满足 `x[B,1,K]*y[B,S,K] → sum_K` 类语义（广播轴在 reduction
  轴外层，累加器在广播轴内层独立 init/flush）。

**Load 主动上浮规则（TileFuse 职责，不假设后端 LICM）**：对每个 `linalg.generic` input I，
若 `indexingMap(I)` 在某广播轴 `a_b` 上缺失，则对应 `extract_slice` 的 offsets 静态上就
不引用 `a_b` 的 IV——GroupEmitter 在发射时主动将其插入到 `a_b` 循环之前（见下文
`computeHoistPoint`）。硬件 `broadcast_l2` 的展开由 codegen 在搬运到 UB 时局部决定，
与上浮分层解耦。

```cpp
LoopNestResult buildLoopNest(OpBuilder &builder, Location loc,
                               const TilePlan &plan,
                               ArrayRef<Value> initTensors) {
  LoopNestResult result;

  // loop order（D1）：
  //   Outer tileable（ascendc.parallel） → BCast full → Inner tileable → Reduction
  // Reduction 轴 Full 且非 reduction-split 时不建 loop；
  // BCast full 轴总是建 loop（不切，上界 = full extent、step = 1 或 extent/1）
  SmallVector<TileParam> loopParams;
  // 1. Outer tileable
  for (auto &tileVec : plan.tileable)
    for (auto &tp : tileVec)
      if (tp.level == TileLevel::Outer)
        loopParams.push_back(tp);
  // 2. BCast full（plan.full 中 role == Parallel 且属于 broadcastAxes；step=1）
  for (auto &tp : plan.full)
    if (tp.role == AxisRole::Parallel &&
        llvm::is_contained(plan.group->broadcastAxes, tp.axisIdx))
      loopParams.push_back(tp);
  // 3. Inner tileable
  for (auto &tileVec : plan.tileable)
    for (auto &tp : tileVec)
      if (tp.level == TileLevel::Inner)
        loopParams.push_back(tp);
  // 4. Reduction full 轴由 Phase 4 / Phase 3 后段处理（非 split 时通常不建 loop）

  // 判断某 TileParam 是否 BCast Full（step=1，upperBound=extent）
  auto isBCastFull = [&](const TileParam &tp) {
    return tp.level == TileLevel::Full && tp.role == AxisRole::Parallel &&
           llvm::is_contained(plan.group->broadcastAxes, tp.axisIdx);
  };

  SmallVector<Value> currentIterArgs(initTensors.begin(), initTensors.end());

  // 记录每个 axis 的 Outer tile size，供 Inner loop 确定上界
  DenseMap<int, Value> outerTileSize;

  for (auto &tp : loopParams) {
    // 上界规则：
    //   Inner + 同 axis 已有 Outer: outerTileSize（核内分批在核的 XBLOCK 范围内）
    //   BCast Full / 其他: full extent
    Value upperBound;
    if (tp.level == TileLevel::Inner && outerTileSize.count(tp.axisIdx))
      upperBound = castToIndex(outerTileSize[tp.axisIdx], builder, loc);
    else
      upperBound = castToIndex(
          getAxisExtentValue(*plan.group, tp.axisIdx, builder, loc),
          builder, loc);

    // step 规则：BCast Full 强制 step=1（逐值迭代），其他用 tp.ssa（tile size）
    Value step = isBCastFull(tp)
        ? builder.create<arith::ConstantIndexOp>(loc, 1).getResult()
        : tp.ssa;

    auto forOp = builder.create<scf::ForOp>(
        loc,
        builder.create<arith::ConstantIndexOp>(loc, 0),
        upperBound,
        castToIndex(step, builder, loc),
        currentIterArgs);

    if (tp.level == TileLevel::Outer) {
      forOp->setAttr("ascendc.parallel", builder.getUnitAttr());
      outerTileSize[tp.axisIdx] = tp.ssa;
    }

    builder.setInsertionPointToStart(forOp.getBody());

    // composedIV：同一 axis 已有 Outer IV 时，叠加 Inner IV
    //   outerIV 取值: 0, XBLOCK, 2*XBLOCK, ...
    //   innerIV 取值: 0, XBLOCK_SUB, 2*XBLOCK_SUB, ..., XBLOCK-XBLOCK_SUB
    //   composedIV = outerIV + innerIV → 实际 tensor offset
    if (result.loopIVs.count(tp.axisIdx)) {
      Value outerIV = result.loopIVs[tp.axisIdx];
      Value innerIV = forOp.getInductionVar();
      result.loopIVs[tp.axisIdx] =
          builder.create<arith::AddIOp>(loc, outerIV, innerIV);
    } else {
      result.loopIVs[tp.axisIdx] = forOp.getInductionVar();
    }

    currentIterArgs = SmallVector<Value>(forOp.getRegionIterArgs());
  }

  result.innermostBody = builder.getInsertionBlock();
  result.iterArgs      = currentIterArgs;
  return result;
}
```

### SliceComputer

文件：`lib/Conversion/VectorPlan/TileFuse/SliceComputer.cpp`

```cpp
struct SliceParams {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides; // 全为 1
};

SliceParams computeSlice(OpBuilder &builder, Location loc,
                          AffineMap indexingMap,
                          const DenseMap<int, Value> &loopIVs,
                          const TilePlan &plan,
                          Value tensor) {
  SliceParams params;
  auto zero = builder.getIndexAttr(0);
  auto tensorType = cast<RankedTensorType>(tensor.getType());

  for (auto [dimIdx, expr] : enumerate(indexingMap.getResults())) {
    params.strides.push_back(builder.getIndexAttr(1));

    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      int g = dimExpr.getPosition();
      if (loopIVs.count(g)) {
        // 该 G-axis 在当前 loop 层有 IV → tile
        params.offsets.push_back(loopIVs.at(g));
        params.sizes.push_back(
            castToIndex(getTileSizeForAxis(plan, g), builder, loc));
      } else {
        // Reduction 轴或未参与当前 loop 的轴 → full dim
        params.offsets.push_back(zero);
        params.sizes.push_back(
            builder.create<tensor::DimOp>(loc, tensor, dimIdx).getResult());
      }
    } else {
      // 常量 0 或其他 affine expr（如 broadcast 维）→ full dim
      params.offsets.push_back(zero);
      params.sizes.push_back(
          builder.create<tensor::DimOp>(loc, tensor, dimIdx).getResult());
    }
  }
  return params;
}
```

### GroupEmitter

文件：`lib/Conversion/VectorPlan/TileFuse/GroupEmitter.cpp`

**主动 Load 上浮（BAII L2 的性能收益落地）**：

对每个 boundary input 的 `extract_slice`，判断其 `offsets` 是否依赖某广播轴的 IV。若不依赖，
**GroupEmitter 将 `extract_slice` 插入到对应广播轴 `scf.for` 之前**（即插入到该广播轴循环
在上一级 Block 的 insertion point），使其在该轴循环外仅执行一次。上浮判定以"offsets 中不
出现该广播轴的 IV"为充要条件——这是静态可检测的，不依赖后端 LICM。

```cpp
// 对给定 extract_slice 计算可上浮的最外层 loop（返回该 loop 之前的插入点）
Block::iterator computeHoistPoint(ArrayRef<OpFoldResult> offsets,
                                   const LoopNestResult &loopNest,
                                   const CollapsedGroupInfo &info) {
  Block::iterator hoistPt = loopNest.innermostBody->begin();
  // 从内到外遍历 BCast loops；首个 "offsets 不引用其 IV" 的 loop 就是最外可上浮点
  for (int ab : llvm::reverse(info.broadcastAxes)) {
    Value iv = loopNest.loopIVs.at(ab);
    if (offsetsReferenceValue(offsets, iv)) break; // 依赖该轴 → 不能再往外
    scf::ForOp forOp = getForOpOwning(iv);
    hoistPt = Block::iterator(forOp); // 上浮到该 for 之前
  }
  return hoistPt;
}
```

```cpp
using TiledValueMap = DenseMap<Value, Value>;

void emitGroup(OpBuilder &builder, Location loc,
               const CollapsedGroupInfo &info,
               const TilePlan &plan,
               const LoopNestResult &loopNest) {
  TiledValueMap tiledValues;

  for (linalg::LinalgOp op : info.topoMembers) {
    SmallVector<Value> newOperands;

    for (auto [idx, operand] : enumerate(op->getOperands())) {
      if (tiledValues.count(operand)) {
        // interior value：前序 op 的 tiled 结果，直接使用
        newOperands.push_back(tiledValues[operand]);
      } else if (isBoundaryInput(operand, info)) {
        // boundary input：emit extract_slice，并主动上浮到最外可行点
        auto map = op.getIndexingMapsArray()[idx];
        auto slice = computeSlice(builder, loc, map,
                                   loopNest.loopIVs, plan, operand);
        OpBuilder::InsertionGuard guard(builder);
        auto hoistPt = computeHoistPoint(slice.offsets, loopNest, info);
        builder.setInsertionPoint(hoistPt->getBlock(), hoistPt);
        auto extractSlice = builder.create<tensor::ExtractSliceOp>(
            loc, operand, slice.offsets, slice.sizes, slice.strides);
        newOperands.push_back(extractSlice);
      } else {
        newOperands.push_back(operand); // 常量、scalar
      }
    }

    // emit tiled op
    auto *tiledOp = builder.clone(*op);
    for (auto [i, v] : enumerate(newOperands))
      tiledOp->setOperand(i, v);
    for (auto [orig, tiled] : llvm::zip(op->getResults(), tiledOp->getResults()))
      tiledValues[orig] = tiled;
  }

  // emit insert_slice + scf.yield for boundary outputs
  SmallVector<Value> yieldVals;
  for (auto [outVal, iterArg] :
       llvm::zip(info.boundaryOut, loopNest.iterArgs)) {
    // output indexing map 是 identity → offsets 同 XBLOCK/XBLOCK_SUB loop IVs
    auto insertSlice = builder.create<tensor::InsertSliceOp>(
        loc, tiledValues[outVal], iterArg,
        getOutputOffsets(loopNest, plan),
        getOutputSizes(plan),
        /*strides=*/ones(builder, loc, info.collapsedAxes.size()));
    yieldVals.push_back(insertSlice);
  }
  builder.create<scf::YieldOp>(loc, yieldVals);
}
```

### 测试用例（Phase 3）

```mlir
// test/Conversion/VectorPlan/tile-fuse-vector-pointwise.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s

func.func @kernel_group0(%x: tensor<?xf16>, %y: tensor<?xf16>) -> tensor<?xf16> {
  %out = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%x, %y) outs(%init: tensor<?xf16>) { ... }
  return %out
}
// CHECK: scf.for %[[XB:.*]] = %c0 to {{.*}} step %[[XBLOCK:.*]] {ascendc.parallel
// CHECK:   scf.for %[[XS:.*]] = %c0 to %[[XBLOCK]] step {{.*}} {
// CHECK:     %[[OFF:.*]] = arith.addi %[[XB]], %[[XS]]
// CHECK:     %[[S1:.*]] = tensor.extract_slice %x[%[[OFF]]]
// CHECK:     %[[S2:.*]] = tensor.extract_slice %y[%[[OFF]]]
// CHECK:     linalg.generic ins(%[[S1]], %[[S2]])
// CHECK:     tensor.insert_slice

// test/Conversion/VectorPlan/tile-fuse-vector-reduce-pointwise.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
func.func @kernel_group0(%in: tensor<?x?xf16>) -> tensor<?xf16> {
  %r = linalg.reduce { arith.addf } ins(%in) outs(...) dimensions = [1]
  %out = linalg.generic { ... } ins(%r) outs(...)  // epilogue
  return %out
}
// CHECK: scf.for %[[XB:.*]] {{.*}} step %[[XBLOCK:.*]] {ascendc.parallel
// CHECK:   scf.for %[[XS:.*]] = %c0 to %[[XBLOCK]]
// CHECK:     %[[OFF:.*]] = arith.addi %[[XB]], %[[XS]]
// CHECK:     %[[slice:.*]] = tensor.extract_slice %in[%[[OFF]], 0]
// CHECK:     %[[r:.*]] = linalg.reduce
// CHECK:     linalg.generic ins(%[[r]])  ← epilogue 使用 reduce 结果（无 extract_slice）
// CHECK:     tensor.insert_slice

// test/Conversion/VectorPlan/tile-fuse-vector-sibling.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
// 3 个 sibling：共享 boundary input x
func.func @kernel_group0(%x: tensor<?xf16>, %a: tensor<?xf16>,
                          %b: tensor<?xf16>, %c: tensor<?xf16>)
    -> (tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) {
  %s1 = linalg.generic { ... } ins(%x, %a) outs(...)
  %s2 = linalg.generic { ... } ins(%x, %b) outs(...)
  %s3 = linalg.generic { ... } ins(%x, %c) outs(...)
  return %s1, %s2, %s3
}
// 同一 loop body 内，x 只 extract_slice 一次
// CHECK: scf.for %[[XB:.*]] {{.*}} step %[[XBLOCK:.*]] {ascendc.parallel
// CHECK:   scf.for %[[XS:.*]] = %c0 to %[[XBLOCK]]
// CHECK:     %[[OFF:.*]] = arith.addi %[[XB]], %[[XS]]
// CHECK:     tensor.extract_slice %x[%[[OFF]]]
// CHECK-COUNT-3: linalg.generic
// CHECK-COUNT-3: tensor.insert_slice

// test/Conversion/VectorPlan/loop-order-broadcast-outside-tile.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
// attention mask 广播: scores[B,H,S,T] + mask[1,1,S,T]
// d_B, d_H 为广播轴 → loop nest 中 d_B, d_H 在 tile 内层之外
func.func @kernel_group0(%scores: tensor<?x?x?x?xf16>,
                          %mask: tensor<?x?xf16>) -> tensor<?x?x?x?xf16> { ... }
// CHECK: scf.for %[[XB:.*]] {{.*}} {ascendc.parallel   ← 分核外层
// CHECK:   scf.for %[[BCAST_B:.*]]                      ← d_B（BCast full）
// CHECK:     scf.for %[[BCAST_H:.*]]                    ← d_H（BCast full）
// CHECK:       scf.for %[[XS:.*]]                       ← tile 内层
// CHECK:         tensor.extract_slice %mask             ← Load hoist 候选
```

---

## Phase 4: Reduce Split（accumulator pattern）

`enableReductionSplit=true` 时，对含 reduction 的 VectorGroup 启用。

```cpp
void emitGroupWithReductionSplit(OpBuilder &builder, Location loc,
                                  const CollapsedGroupInfo &info,
                                  const TilePlan &plan,
                                  const LoopNestResult &parallelLoopNest) {
  // 把 topoMembers 分成两段：pre-reduction（含 reduce op）+ epilogue
  auto [preReduction, epilogue] = splitAtReductionBoundary(info.topoMembers);

  // 1. 在 parallel loop body 内 emit linalg.fill（init accumulator）
  auto accType = getReductionAccType(info);
  auto empty   = builder.create<tensor::EmptyOp>(loc, accType, ValueRange{});
  Value acc    = builder.create<linalg::FillOp>(
      loc, TypeRange{accType}, ValueRange{zero, empty}).getResult(0);

  // 2. 建 RBLOCK scf.for
  const TileParam &rblock = plan.full[0]; // RBLOCK_0
  Value rExtent = getAxisExtentValue(info, rblock.axisIdx, builder, loc);
  auto rForOp = builder.create<scf::ForOp>(
      loc,
      builder.create<arith::ConstantIndexOp>(loc, 0),
      castToIndex(rExtent, builder, loc),
      castToIndex(rblock.ssa, builder, loc),
      ValueRange{acc});

  // 3. pre-reduction ops → emit 进 RBLOCK body
  builder.setInsertionPointToStart(rForOp.getBody());
  TiledValueMap tiledValues;
  tiledValues[getAccInit(preReduction)] = rForOp.getRegionIterArgs()[0];
  LoopNestResult rblockLoopNest = parallelLoopNest;
  rblockLoopNest.loopIVs[rblock.axisIdx] = rForOp.getInductionVar();
  emitOps(builder, loc, preReduction, plan, rblockLoopNest, tiledValues);
  builder.create<scf::YieldOp>(loc, tiledValues[getReductionResult(preReduction)]);

  // 4. epilogue ops → emit 在 RBLOCK loop 之后
  builder.setInsertionPointAfter(rForOp);
  tiledValues[getReductionResult(preReduction)] = rForOp.getResult(0);
  emitOps(builder, loc, epilogue, plan, parallelLoopNest, tiledValues);
}
```

### 测试用例（Phase 4）

```mlir
// test/Conversion/VectorPlan/tile-fuse-vector-softmax.mlir
// RUN: mlir-opt --vector-plan-tile-fuse="enable-reduction-split=true" %s \
// RUN:   | FileCheck %s

func.func @kernel_group0(%in: tensor<?x?xf16>) -> tensor<?x?xf16> {
  // max reduce → sub → exp → sum reduce → div
}
// CHECK: scf.for %[[XB:.*]] {ascendc.parallel
// CHECK:   scf.for %[[XS:.*]]
// CHECK:     linalg.fill  ← acc init（在 RBLOCK loop 之前）
// CHECK:     scf.for %[[RB:.*]] = %c0 to {{.*}} step {{.*}} {  ← RBLOCK loop
// CHECK:       linalg.reduce
// CHECK:     }
// CHECK:     linalg.generic  ← epilogue（在 RBLOCK loop 之后）
```

---

## Phase 5: B2 Variant 1（no_collapse transpose 路径）

B2 input 存在时，TileFusePass 生成两份独立的 IR（Variant 1 / Variant 2），
由 Autotuner 在运行时选优。两份 IR 分别对应 `kernel_groupN_v1.mlir` /
`kernel_groupN_v2.mlir`，各自走完整的 TilePlan + LoopNest 流程。

```cpp
// TileFusePass.cpp 顶层，Collapse 之后：
if (check.hasB2) {
  // 生成两个 CollapsedGroupInfo，分别走完整的 TilePlan + LoopNest
  GroupInfo v1Info = fixupB2Variant1(builder, cloneGroupInfo(info), check);
  GroupInfo v2Info = fixupB2Variant2(builder, cloneGroupInfo(info), check);
  // 对 v1 执行后续步骤 → 产出 Variant 1 IR
  // 对 v2 执行后续步骤 → 产出 Variant 2 IR
} else {
  // 无 B2：单一路径
}

// Variant 1 路径识别（loop nest 建立时使用）：
bool useOriginalAxes = llvm::any_of(info.topoMembers, [](linalg::LinalgOp op) {
  return op->hasAttr("vector_plan.no_collapse");
});
// useOriginalAxes=true → LoopNestBuilder 用原始 G-axes（各轴独立 IV）
// SliceComputer 对 B2 input 用原始 B2 map，对 output 用 canonical map
```

### 测试用例（Phase 5）

```mlir
// test/Conversion/VectorPlan/tile-fuse-vector-b2-v1.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s

// input T [S, B]，consumer map: (d0,d1)→(d1,d0)（B2）
// Variant 1：插 linalg.transpose [S,B]→[B,S]
func.func @kernel_group0(%T: tensor<?x?xf16>, %side: tensor<?x?xf16>)
    -> tensor<?x?xf16> { ... }
// CHECK: linalg.transpose %T perm = [1, 0]  ← no_collapse barrier
// CHECK: scf.for %[[IV0:.*]]  ← 原始 d0 loop
// CHECK:   scf.for %[[IV1:.*]]  ← 原始 d1 loop
// CHECK:     tensor.extract_slice %T[%[[IV1]], %[[IV0]]]  ← B2 map offset
```

---

## Phase 6: B2 Variant 2（load_with_transpose 路径）

```cpp
// GroupEmitter 中，boundary input 处理追加判断：
if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
  auto parentFunc = cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
  if (parentFunc->hasAttr("vector_plan.load_with_transpose")) {
    // 生成 ConfusionTranspose 占位 op（GM→UB 搬运时完成重排）
    auto confOp = builder.create<ConfusionTransposeOp>(loc, operand, ...);
    newOperands.push_back(confOp);
    continue;
  }
}
```

### 测试用例（Phase 6）

```mlir
// test/Conversion/VectorPlan/tile-fuse-vector-b2-v2.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s

// Variant 2：consumer map 改为 canonical，boundary input 打 load_with_transpose
func.func @kernel_group0(%T: tensor<?x?xf16>, %side: tensor<?x?xf16>)
    -> tensor<?x?xf16> { ... }
// CHECK-NOT: linalg.transpose  ← 无 transpose node
// CHECK: tensor.collapse_shape  ← 整体 collapse，无 barrier
// CHECK: scf.for %[[XB:.*]]  ← collapsed loop
// CHECK:   confusion_transpose %T  ← Load 侧重排
```

---

## Phase 7: CubeGroup — 层次化 Loop Nest

### TilePlanGen（CubeGroup 分支）

```cpp
TilePlan genCubeTilePlan(const CubeGroupInfo &info,
                          OpBuilder &builder, Location loc) {
  TilePlan plan;
  auto matmul = info.matmul;
  Value lhs = matmul.getInputs()[0]; // [M, K]
  Value rhs = matmul.getInputs()[1]; // [K, N]

  int mRank = cast<RankedTensorType>(lhs.getType()).getRank();
  Value Mext = builder.create<tensor::DimOp>(loc, lhs, mRank - 2);
  Value Next = builder.create<tensor::DimOp>(loc, rhs,
      cast<RankedTensorType>(rhs.getType()).getRank() - 1);
  Value Kext = builder.create<tensor::DimOp>(loc, lhs, mRank - 1);

  // M 轴：两级
  plan.tileable.push_back({
    TileParam{"BM",   insertFuncArg(builder, "BM",   128), Mext,
              mAxisIdx, TileLevel::Outer, AxisRole::Parallel},
    TileParam{"Tb_M", insertFuncArg(builder, "Tb_M",  64), Mext,
              mAxisIdx, TileLevel::Inner, AxisRole::Parallel},
  });
  // N 轴：两级
  plan.tileable.push_back({
    TileParam{"BN",   insertFuncArg(builder, "BN",   128), Next,
              nAxisIdx, TileLevel::Outer, AxisRole::Parallel},
    TileParam{"Tb_N", insertFuncArg(builder, "Tb_N", 128), Next,
              nAxisIdx, TileLevel::Inner, AxisRole::Parallel},
  });
  // K 轴：Inner（必须切，非 Full）
  plan.full.push_back(
    TileParam{"t_K", insertFuncArg(builder, "t_K", 64), Kext,
              kAxisIdx, TileLevel::Inner, AxisRole::Reduction}
  );

  // 2D blockDimExprs
  plan.blockDimExprs.push_back(
      builder.create<arith::CeilDivSIOp>(loc, Mext,
          plan.tileable[0][0].ssa).getResult()); // grid_y
  plan.blockDimExprs.push_back(
      builder.create<arith::CeilDivSIOp>(loc, Next,
          plan.tileable[1][0].ssa).getResult()); // grid_x

  return plan;
}
```

### CubeGroup 发射

```cpp
void emitCubeGroup(OpBuilder &builder, Location loc,
                    const CubeGroupInfo &info, const TilePlan &plan) {
  // 4 层 loop：BM(Outer) → BN(Outer) → Tb_M(Inner) → Tb_N(Inner)
  auto outerLoopNest = buildCubeOuterLoops(builder, loc, plan);

  // innermost body（Tb_N 内）：accumulator pattern
  auto accType = getMatmulOutputType(info.matmul);
  auto empty   = builder.create<tensor::EmptyOp>(loc, accType, ValueRange{});
  Value acc    = builder.create<linalg::FillOp>(
      loc, TypeRange{accType}, ValueRange{zero, empty}).getResult(0);

  // t_K loop
  const TileParam &tK = plan.full[0];
  Value Kext = getAxisExtentValue(info, tK.axisIdx, builder, loc);
  auto tkForOp = builder.create<scf::ForOp>(
      loc,
      builder.create<arith::ConstantIndexOp>(loc, 0),
      castToIndex(Kext, builder, loc),
      castToIndex(tK.ssa, builder, loc),
      ValueRange{acc});
  tkForOp->setAttr("ascendc.prologue",
                    builder.getStringAttr("lhs:A1->A2,rhs:B1->B2"));

  // t_K body：只发射 matmul
  builder.setInsertionPointToStart(tkForOp.getBody());
  TiledValueMap tiledValues;
  tiledValues[info.matmul.getInputs()[0]] =
      getASlice(builder, loc, info, outerLoopNest, tkForOp.getInductionVar(), plan);
  tiledValues[info.matmul.getInputs()[1]] =
      getBSlice(builder, loc, info, outerLoopNest, tkForOp.getInductionVar(), plan);
  tiledValues[getAccInit(info.matmul)] = tkForOp.getRegionIterArgs()[0];
  emitOp(builder, loc, info.matmul, tiledValues, plan, outerLoopNest);
  // ascendc.unit 标注必须设置在 loop body 内的实际 tiled matmul op 上
  tiledValues[info.matmul->getResult(0)].getDefiningOp()
      ->setAttr("ascendc.unit", builder.getStringAttr("AiCore.Cube"));
  builder.create<scf::YieldOp>(loc, tiledValues[info.matmul->getResult(0)]);

  // epilogue ops after t_K loop
  builder.setInsertionPointAfter(tkForOp);
  Value matmulResult = tkForOp.getResult(0);
  tiledValues[info.matmul->getResult(0)] = matmulResult;

  for (linalg::LinalgOp epi : getEpilogueOps(info)) {
    emitOp(builder, loc, epi, tiledValues, plan, outerLoopNest);
    // epilogue ops 标注为 AiCore.Vector
    tiledValues[epi->getResult(0)].getDefiningOp()
        ->setAttr("ascendc.unit", builder.getStringAttr("AiCore.Vector"));
  }
```

### 测试用例（Phase 7）

```mlir
// test/Conversion/VectorPlan/tile-fuse-cube-matmul-bias-relu.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s

func.func @kernel_group1(%A: tensor<?x?xf16>, %B: tensor<?x?xf16>,
                          %bias: tensor<?xf16>) -> tensor<?x?xf16> {
  %mm  = linalg.matmul ins(%A, %B) outs(...)
  %add = linalg.generic { ... } ins(%mm, %bias) outs(...)  // bias add
  %relu = linalg.generic { ... } ins(%add) outs(...)       // relu
  return %relu
}
// CHECK:      scf.for %[[BM:.*]] {{.*}} {ascendc.parallel
// CHECK-NEXT:   scf.for %[[BN:.*]] {{.*}} {ascendc.parallel
// CHECK:          scf.for %[[TBM:.*]]
// CHECK:            scf.for %[[TBN:.*]]
// CHECK:              linalg.fill  ← acc init
// CHECK:              scf.for %[[TK:.*]] {{.*}} ascendc.prologue
// CHECK:                linalg.matmul
// CHECK:              linalg.generic  ← bias add，AiCore.Vector
// CHECK:              linalg.generic  ← relu，AiCore.Vector

// test/Conversion/VectorPlan/tile-fuse-cube-layernorm-matmul.mlir
// RUN: mlir-opt --vector-plan-tile-fuse %s | FileCheck %s
// （LayerNorm VectorGroup + matmul+epilogue CubeGroup 各自独立文件，分别验证）
```

---

## 数据模型影响

### 现有结构复用

- **`CollapsedGroupInfo::axisMap`** 已有的 `original axis idx -> collapsed axis idx; -1 if not collapsed`
  语义原生承载 BAII L1：广播轴在此映射为 `-1`。
- **`TilePlan::full`** 已有的"完整循环轴"语义原生承载广播轴不切：广播轴进 `full`，不进 `tileable`。
- **`TilePlan::tileable`** 与 **`TilePlan::blockDimExprs`** 不受影响。

### 新增字段（最小）

```cpp
struct CollapsedGroupInfo : GroupInfo {
  // ... 已有字段
  llvm::SmallVector<int>        broadcastAxes;        // post-collapse idx，升序
  llvm::DenseMap<int, Value>    broadcastAxisExtents; // post-collapse idx → extent SSA
};
```

约定见 §术语与索引空间约定：`broadcastAxes` 存 **post-collapse idx**，值范围
`[|collapsedAxes|, n)`。`broadcastAxisExtents` 承担广播轴的 extent 查询（广播轴不在
`collapsedAxes` 里，无法用 `collapsedAxes[i].extent` 查）。

两字段在 Pre-Check 阶段一并填充，后续 Phase 2 / Phase 3 / verifier 直接读取。

### 不新增

不引入 `ViewOpAttrInfo` 一类的 Load 侧属性结构——MLIR `linalg.generic::indexing_maps`
已承载"哪些轴对哪个 operand 广播"的信息，无需二次建模。

---

## 边界情形

### 全 group 共同广播轴

若某轴 `a ∈ BCast(G)` 且**所有** boundary input 都在 `a` 上缺失，该轴对 group 的实际数据
依赖为 0，属于纯"外层重复"。本设计不特殊处理——这类轴应在 canonical axes 推导阶段就被
归零或外提，不会到达 TileFuse。BroadcastAbsorb 后若仍存在，走通用 BAII 路径（进 `full`、
完整遍历），语义正确但有少量冗余循环开销。Future optimization：canonical 外提。

### 广播轴紧邻 reduction

形如 `x[B, 1, K] * y[B, S, K] → sum_K` 的结构，d_S 是广播轴、d_K 是 reduction 轴。

- D1 loop order `[分核] → [broadcast] → [tile] → [reduction]` 自动保证 d_S 在 d_K 外层，
  reduction 累加器在广播轴内层独立累加，语义正确。
- 无需额外约束，此例作为 BAII L2 成立的佐证。

### Dynamic shape

- 当 `indexingMap` **语法上缺失**某轴：仍按广播处理（语义上已确定）。
- 当 `indexingMap` **包含**某轴但运行时可能 size=1：不作为广播处理（由硬件/运行时 stride=0 机制处理）。

即 **BAII 判据是 affine map 形态，不是 size**。

### 吸收失败的 `linalg.broadcast`

见 §前置 Pass BroadcastAbsorb "失败后的语义"小节：保留为普通 elementwise op，物化搬运一次，
不破坏 BAII。

### Cube group

`GroupInfo::Kind::Cube` 跳过本设计全部逻辑（Collapse 恒等映射、TilePlan 由 matmul 专有路径生成）。
matmul 的 bias 广播等由硬件直接处理。

### 迭代空间与资源约束（Case A/B/C）

广播轴进 `TilePlan::full` 会产生嵌套 full-loop，存在以下潜在 corner case：

**Case A — 多广播轴嵌套**：若 `BCast(G)` 中存在多条大 size 广播轴（典型 attention
`scores[B,H,S,T] + mask[1,1,S,T] + alibi[1,H,S,T]` → 广播轴集合 `{d_B, d_H}`），每核
循环次数为 `∏(|广播轴|) × iter(tile)`。`B=1, H=32` 量级推理可接受；`B=32, H=32` 训练
场景可能使软件循环 overhead 支配 vector 执行时间。

**Case B — 广播轴 + reduction 的累加器占用**：形如 `sum_K(x[B,1,K] * y[B,S,K])`，
reduction 累加器必须在广播轴内层独立 init/flush，UB 占用约
`∏(|非 reduction 广播轴|) × tile_size × sizeof(dtype)`。

**Case C — 广播轴挤占分核候选**：广播轴不参与 tile 也不参与分核，若剩余非广播轴总元素数
不足以填满分核粒度，核利用率下降（性能问题，不破坏 BAII）。

**缓解规则（v1 实施）**：

1. `kMaxFullLoopIters` escape（默认 2048）：单条广播轴超阈值时降级 tileable。
2. 累加器 UB 预算 assert：不满足时优先降低内层 tile_size，仍不足则对最大广播轴降级。
3. 分核候选不足 warn（非 verifier fail）。

**Future work (v2+)**：
- 相邻一致广播轴合并（同组 boundary input 同时缺失的相邻轴合成为单一 full-loop）
- 硬件多维 broadcast_l2 的 codegen 优化

---

## Verifier: BAII L1/L2 校验

TileFuse pass 结束时校验 BAII：

```cpp
// L1: Collapse 独立性
for (auto &cg : module.getAllCollapsedGroups()) {
  auto bcast = computeBCast(cg);
  for (int ai : bcast)
    assert(cg.axisMap[ai] == -1 &&
           "BAII L1 violation: broadcast axis must not be collapsed");
}

// L2: Loop-nest 外层性
for (auto &tp : module.getAllTilePlans()) {
  auto bcast = computeBCast(tp.group);
  for (int ab : bcast) {
    int bcastDepth = loopDepth(tp, ab);
    for (auto *consumer : tileLoopsConsuming(ab)) {
      assert(bcastDepth < loopDepth(consumer) &&
             "BAII L2 violation: broadcast axis must be outer to tile loops");
    }
  }
}
```

违反即 `signalPassFailure()`，打印违反的 axis idx 与相关 op 位置，便于定位。

### 测试用例（Verifier）

```mlir
// test/Conversion/VectorPlan/broadcast-axis-invariant-violation.mlir
// RUN: not mlir-opt --vector-plan-tile-fuse --verify-diagnostics %s 2>&1 | FileCheck %s
// 人工构造违反 BAII 的 TilePlan（广播轴出现在 collapsedAxes 中）
// CHECK: BAII L1 violation
```

---

## 实施范围（v1）

### 包含
- BroadcastAbsorb pass（基本 pattern：`linalg.broadcast → linalg.generic`）
- Phase 1 广播轴剪枝 + A/B2/C 三类分类
- `CollapsedGroupInfo::broadcastAxes` 字段
- Phase 2 广播轴进 `TilePlan::full` + `kMaxFullLoopIters` escape
- Phase 3 Loop order 规则（D1）
- 累加器 UB 预算 assert + 分核候选不足 warn
- BAII verifier（L1 + L2）
- Phase 1 / Phase 3 / Verifier 的 FileCheck 测试

### 端到端（nanoGPT 级）测试
- LayerNorm（`[B,S,H]` input，`[H]` scale/bias）：d_H 不进 collapse、scale/bias Load hoist 最外
- Attention mask 广播（`[1,1,S,S] → [B,H,S,S]`）：d_B/d_H 同时作为广播轴、按原序排列
- Elementwise + reduction 混合（GELU 后 sum）：broadcast + reduction 顺序不冲突

### 回归
- 现有 `foundation-smoke.mlir` 等 group-pipeline 测试保持 green
- 新增 broadcast case 加入 CI

### 不包含（future work）
- BroadcastAbsorb 对 `tensor.reshape` 等中间 op 的跨越吸收
- 全 group 共同广播轴的 canonical 外提
- 相邻一致广播轴合并
- 硬件多维 broadcast_l2 codegen 优化
- Autotuner 对 BAII 的"受控违反"路径

### 不支持
- Cube group（正交，由 matmul 专有路径处理）
- Dynamic shape 下基于 size=1 的运行时广播识别
