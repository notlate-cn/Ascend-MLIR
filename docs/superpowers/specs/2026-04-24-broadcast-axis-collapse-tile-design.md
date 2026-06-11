# Broadcast 轴的 Collapse 与 Tile 方案

**日期**：2026-04-24
**范围**：VectorPlan TileFuse pass 对含广播语义的 input 的处理策略
**关联文档**：
- `docs/vector-plan/03-tile-fuse.md` §Phase 1 Collapse / §Phase 3 LoopNest
- `docs/vector-plan/01-data-model.md`（`GroupInfo` / `CollapsedGroupInfo` / `TilePlan`）
- AF 参考：`ge-eco/.../can_fuse/backend/asc_graph_axis_mapping.cpp`、
  `ge-eco/.../can_fuse/backend/backend_utils.h`、
  `ge-eco/.../doc/schedule_ATT轴处理全流程说明.md`

---

## 1. 背景与问题

VectorPlan 的 TileFuse pass 在 Phase 1 (Collapse) 会把候选 collapse 组 G 中的连续同类型轴折成一条轴，进而做 tile 与分核。当 group 内某个 boundary input 在 G 的某条轴上是广播（indexing_map 在该轴缺失）时，原 B1 方案提出"**补齐**"路径——先插入 `linalg.broadcast` 把该 input 展开到目标形状，再与其它 input 一起 collapse 与 tile。

这条路径的问题已在 03-tile-fuse.md §114-213 的数值推演中被证实：

- `linalg.broadcast` + `collapse_shape` + `extract_slice` 的复合形态中，一旦 tile 切分了 collapsed 轴，单个 tile 在原 3D 空间里不再是矩形（会跨越 broadcast 轴边界、且 broadcast 轴外层轴自增）。
- 元素级索引 `brc_flat[aAB, c] = x1[aAB/3, 0, c]` 在数学上仍可溯源，但"Load 对某循环迭代器不变"这个 LICM/hoist 所需的模式**不成立**——broadcast 轴 d_B 已被折进 aAB，aAB 的内层迭代会同时改变 d_A 和 d_B 分量。
- 结果：实际搬运量膨胀（示例中 12 行搬运但仅 4 行独立数据，效率 33%）。
- 唯一的救回路径是强制 tile size 为 reassoc 内层 size 的倍数（strip-mine 出 b_inner 独立循环再 hoist），但这等价于反解 collapse——即退回到"不 collapse broadcast 轴"的方案。

AF（AutoFuse，CANN 图编译器）在三年以上的工程实践中对 broadcast 轴采用的不是"补齐"路径，而是：

- broadcast 语义不落为图上节点，而是以 `ViewOpAttrInfo::broadcast_info` 挂在 Load 边上；
- ATT schedule 搜索空间**硬跳过** broadcast split 轴及其派生轴（`ShouldSkipAxis` + `IsFromReduceSplit`）；
- Reorder 里 broadcast 轴优先级高于普通轴；
- 一句总结："Broadcast 在 ATT 中的核心不是'怎么切'，而是**明确哪些轴不能按普通方式去搜**"（§4.7.2）。

本设计将 AF 的这条工程经验映射到 VectorPlan，确立一条硬不变量 **Broadcast Axis Independence Invariant**，并给出在现有 pass/data model 上的落地方案。

---

## 2. 核心不变量：Broadcast Axis Independence Invariant

对一个 VectorGroup `G` 与其 canonical iteration axes `A = {a_0, ..., a_{n-1}}`，定义**广播轴集合** `BCast(G)`：

```
BCast(G) = { a_i ∈ A | ∃ boundary input I ∈ inputs(G),
                       indexingMap(I) 在 a_i 上缺失 }
```

即：只要 group 内任一 boundary input 在某轴上是广播（affine map 结果中不出现该 dim），该轴即为广播轴。

**Broadcast Axis Independence Invariant L1（Collapse 独立性）**：

> `BCast(G)` 中的任何轴不得出现在 `CollapsedGroupInfo::collapsedAxes` 内。
>
> 等价地：`axisMap[i] == -1` 对所有 `a_i ∈ BCast(G)` 成立。

**Broadcast Axis Independence Invariant L2（Loop-nest 外层性）**：

> 在 `TilePlan` 对应的 loop nest 中，对任一广播轴 `a_b ∈ BCast(G)` 与任一消费 `a_b` 的 tile 循环轴 `a_t`（`a_t ∉ BCast(G)` 且 `map(I).results` 中包含 `a_b` 或 `a_t`，I 为 boundary input），`a_b` 的循环层级必须在 `a_t` 之外。

**范围**：Broadcast Axis Independence Invariant 仅对 `GroupInfo::Kind::Vector` 生效。`Kind::Cube` 跳过 Collapse（恒等映射）且 matmul 的广播（如 bias）由硬件直接吸收，本设计不覆盖。

**强度**：硬不变量。TileFuse pass 在产出 `CollapsedGroupInfo` 与 `TilePlan` 时**必须**保证 Broadcast Axis Independence Invariant 成立；任何违反通过 pass verifier 报错而非 warn。

---

## 3. 设计决策

下表列出所有开放问题的定型选择，每项附理由锚点。

| 编号 | 决策 | 理由摘要 |
|---|---|---|
| D1 | **Loop 排序**：`[分核 tile 外层] → [broadcast full] → [tile 内层] → [reduction]` | NPU blockIdx 自然在最外；每核独立持有 broadcast 复用；reduction 在 broadcast 内层满足 `x[B,1,K]*y[B,S,K]→sum_K` 类语义 |
| D2 | **广播表示**：上游保留 `linalg.broadcast`；TileFuse 入口加 **BroadcastAbsorb** 预处理吸收进 `linalg.generic::indexing_maps`；TileFuse 内部零显式 broadcast 节点 | 对 torch-mlir 等上游友好；TileFuse 核心逻辑纯净；吸收不掉的情形作为普通 elementwise 物化 |
| D3 | **Collapse 剪枝**：按 broadcast 轴切分 candidate G，复用 `size ≥ 2` 过滤 | 与 `findCandidateGroups` 语义一致；实现为独立小函数，职责清晰 |
| D4 | **广播轴 tile**：永不 tile，进 `TilePlan::full`；超阈值 `kMaxFullLoopIters`（初值 2048）时 escape 到 `tileable`（v1 实施，见 §6.6） | 简化 hoist 规则；nanoGPT 级结构 broadcast 轴 size 在 full-loop 可承受范围；escape 兜底多广播轴嵌套爆炸 |
| D5 | **多广播轴排序**：按 `canonicalAxes` 原始序保持 | 正确性无差别；复用率与内存布局自然对齐；YAGNI 拒绝未验证的代价模型 |
| D6 | **不变量强度**：硬不变量；违反则 pass verifier 失败 | AF 工程实践支持；codegen 可做强假设；debug 定位明确 |

---

## 4. 架构与 Pass 拓扑影响

### 4.1 新增 Pass：BroadcastAbsorb

**位置**：TileFuse 入口前，作为 TileFuse 的前置 pass 或 TileFuse 内部的首个 phase。

**输入**：含 `linalg.broadcast` + `linalg.generic` 的 kernel_group{N}.mlir

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
- `linalg.broadcast` 的消费者不是 `linalg.generic`（例如被 `tensor.reshape` 等非仿射 op 截断）
- 下游 generic 的 iteration domain 与 broadcast 目标形状不匹配（罕见）

### 4.2 修改 Pass：TileFuse Phase 1 (Collapse)

增加 **广播轴剪枝** 子步骤，位于 `findCandidateGroups` 产出 candidate 之后、B2 分类之前：

```cpp
// 输入：candidate groups Gs，广播轴集合 BCast
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

**剪枝后的 Pre-Check 分类**：原 A/B1/B2/C 四类收缩为 **A/B2/C** 三类（B1 消失——任何 "G 内轴部分缺失" 的情形已在剪枝阶段被切开）。

### 4.3 修改 Pass：TileFuse Phase 3 (LoopNest + Emit)

**Loop order 生成规则**（D1 的具体化）：

```
loop nest 从外到内 =
  1. bind_multicore 轴（分核 tile 外层）
  2. BCast(G) 中的轴，按 canonicalAxes 原始序
  3. tile 内层（XBLOCK_SUB 等）
  4. reduction 轴（RBLOCK）
```

**Load 发射规则**：对每个 `linalg.generic` input I：
- 若 `indexingMap(I)` 在某轴 `a_b` 上缺失，该 Load 在 `a_b` 循环内自然 loop-invariant
- Codegen 侧的 LICM pass 将其 hoist 到 `a_b` 循环之外
- Hoist 后单份数据通过硬件 `broadcast_l2` 展开（硬件判定由 codegen 局部完成，非 TileFuse 决策）

### 4.4 Verifier

TileFuse pass 结束时校验 Broadcast Axis Independence Invariant：

```cpp
// 伪代码
for (auto &cg : module.getAllCollapsedGroups()) {
  auto bcast = computeBCast(cg);
  for (int ai : bcast) assert(cg.axisMap[ai] == -1);  // Broadcast Axis Independence Invariant L1
}
for (auto &tp : module.getAllTilePlans()) {
  auto bcast = computeBCast(tp.group);
  for (auto *consumer : tileLoopsConsuming(bcast)) {
    assert(loopDepth(broadcastAxis) < loopDepth(consumer));  // Broadcast Axis Independence Invariant L2
  }
}
```

---

## 5. 数据模型影响

### 5.1 现有结构复用

- **`CollapsedGroupInfo::axisMap`** 已有的 `original axis idx -> collapsed axis idx; -1 if not collapsed` 语义原生承载 Broadcast Axis Independence Invariant L1：广播轴在此映射为 `-1`。
- **`TilePlan::full`** 已有的"完整循环轴"语义原生承载 D4：广播轴进 `full`，不进 `tileable`。
- **`TilePlan::tileable`** 与 **`TilePlan::blockDimExprs`** 不受影响。

### 5.2 新增字段（最小）

为便于 verifier 与 LoopNest 快速查询，在 `CollapsedGroupInfo` 增加：

```cpp
struct CollapsedGroupInfo : GroupInfo {
  // ... 已有字段
  llvm::SmallVector<int> broadcastAxes;  // BCast(G) 的原始轴 idx 列表，升序
};
```

`broadcastAxes` 在 Pre-Check 阶段填充，后续 Phase 3 与 verifier 直接读取，避免重复遍历 `indexingMaps`。

### 5.3 不新增

不引入 `ViewOpAttrInfo` 一类的 Load 侧属性结构——MLIR 的 `linalg.generic::indexing_maps` 已经承载了"哪些轴对哪个 operand 是广播"的信息，无需二次建模。

---

## 6. 边界与例外

### 6.1 全 group 共同广播轴

若某轴 `a` ∈ `BCast(G)` 且**所有** boundary input 都在 `a` 上缺失，该轴对 group 的实际数据依赖为 0，属于纯"外层重复"。本设计不特殊处理——这类轴在 canonical axes 推导阶段就应该被归零或外提，不会到达 TileFuse。在 BroadcastAbsorb 后的 Pre-Check 中，这类轴若仍存在，走通用 Broadcast Axis Independence Invariant 路径（进 `full`、完整遍历），语义正确但有少量冗余循环开销。可作为 future optimization。

### 6.2 广播轴紧邻 reduction

形如 `x[B, 1, K] * y[B, S, K] → sum_K` 的结构，d_S 是广播轴、d_K 是 reduction 轴。

- D1 的排序规则 `[分核] → [broadcast] → [tile] → [reduction]` 自动保证 d_S 在 d_K 外层，满足 reduction 累加器在广播轴内层独立累加的语义。
- 无需额外约束，本例作为 Broadcast Axis Independence Invariant L2 成立的佐证。

### 6.3 Dynamic shape

若某 input 在某轴上的 size 动态，无法静态判定"广播"：

- 当 `indexingMap` 语法上缺失该轴：仍按广播处理（语义上已确定）。
- 当 `indexingMap` 包含该轴但运行时可能 size=1：不作为广播处理（由硬件/运行时的 stride=0 机制处理）。

即：**Broadcast Axis Independence Invariant 的判据是 affine map 形态，不是 size**。

### 6.4 吸收失败的 `linalg.broadcast`

若 BroadcastAbsorb 无法把某 `linalg.broadcast` 融入下游 generic：
- 该 op 在 TileFuse 眼中是一个普通 elementwise op（与 `linalg.add` 同层）
- 其输出已经是"物化后的完整形状"，后续 consumer 的 indexing_map 不缺轴
- 因此其下游的 group 里不会因它产生 `BCast` 贡献——代价是一次物化搬运，但不破坏 Broadcast Axis Independence Invariant
- v1 接受此物化开销；后续可通过扩展 BroadcastAbsorb 的 pattern 覆盖更多 case 以降低

### 6.5 Cube group

`GroupInfo::Kind::Cube` 跳过本设计全部逻辑（Collapse 恒等映射、TilePlan 由 matmul 专有路径生成）。matmul 的 bias 广播等由硬件直接处理。

### 6.6 迭代空间与资源约束

广播轴进 `TilePlan::full` 会产生嵌套 full-loop，存在以下潜在 corner case：

**Case A — 多广播轴嵌套**：若 `BCast(G)` 中存在多条大 size 广播轴（典型 attention `scores[B,H,S,T] + mask[1,1,S,T] + alibi[1,H,S,T]` → 广播轴集合 `{d_B, d_H}`），每核循环次数为 `∏(|广播轴|) × iter(tile)`。对 `B=1, H=32` 量级的推理场景可接受；训练场景 `B=32, H=32` 可能使软件循环 overhead 支配 vector 部分执行时间。

**Case B — 广播轴 + reduction 的累加器占用**：形如 `sum_K(x[B,1,K] * y[B,S,K])` 的结构，reduction 累加器必须在广播轴内层独立 init/flush，UB 占用约为 `∏(|非 reduction 广播轴|) × tile_size × sizeof(dtype)`。

**Case C — 广播轴挤占分核候选**：广播轴不参与 tile 也不参与分核，若剩余非广播轴的总元素数不足以填满分核粒度，核利用率下降。属于性能问题，不破坏 Broadcast Axis Independence Invariant。

**缓解规则**（v1 实施）：

1. **广播轴 size 上限（D4 的 escape 路径具体化）**：单条广播轴 `|axis| > kMaxFullLoopIters` 时，降级为 `tileable`。推荐初值 `kMaxFullLoopIters = 2048`，可通过 pass option 覆盖。降级后该轴 Load 不再可 hoist 到循环外，但循环深度可控。
2. **累加器 UB 预算 assert**（Phase 3 emit 时）：检查 `∏(|非 reduction 广播轴|) × tile_size × sizeof(dtype) ≤ UB_capacity`。不满足时触发 escape：优先降低最内层 tile_size；仍不足则对最大的广播轴执行规则 1 的降级。
3. **分核候选回退**：若剔除广播轴后分核候选的总可分核元素数 `< 核数 × 最小 tile`，保留原 canonicalAxes 顺序但 emit 时 warn（非 verifier fail），由上层 pass / 人工评估是否需要调整 group 边界。

**Future work（v2+）**：

- **相邻一致广播轴合并**：若 `canonicalAxes` 中相邻的多条广播轴被**同一组** boundary input 同时缺失（例如 `mask[1,1,S,T]` 对 `d_B` 和 `d_H` 同时广播），可合成为单一 full-loop 以减少循环嵌套深度与指令 overhead。v1 不做，按原始序保留。
- **硬件多维 broadcast_l2**：若硬件支持在一次 Load 展开中同时对多个轴广播，codegen 可在保持软件多层循环的同时共享单份 Load，这是 codegen 局部优化，不影响本设计。

---

## 7. 测试策略

### 7.1 单元测试（pass 级别）

位置：`test/Conversion/VectorPlan/`

- `broadcast-absorb-basic.mlir`：`linalg.broadcast` + `linalg.generic` → 吸收为缺轴 generic
- `broadcast-absorb-fallback.mlir`：非 generic 消费者，保留 `linalg.broadcast` 作为普通 op
- `collapse-prune-middle-axis.mlir`：`G = [d0, d1, d2]`、`d1` 广播 → 剪枝出 `[d0]`（丢弃）与 `[d2]`（丢弃），整体无 collapse
- `collapse-prune-tail-axis.mlir`：`G = [d0, d1, d2]`、`d2` 广播 → 保留 `[d0, d1]`
- `loop-order-broadcast-outside-tile.mlir`：FileCheck loop 嵌套顺序符合 D1
- `broadcast-axis-invariant-violation.mlir`：人工构造违反 Broadcast Axis Independence Invariant 的 TilePlan → expect verifier 报错

### 7.2 端到端（nanoGPT 级别）

- LayerNorm（`[B, S, H]` input，`[H]` scale/bias）：验证 d_H 不进 collapse、scale/bias 的 Load hoist 到最外
- Attention mask 广播（`[1, 1, S, S]` → `[B, H, S, S]`）：验证 d_B, d_H 同时作为广播轴、二者按原始序排列
- Element-wise + reduction 的混合 pattern（GELU 后 sum）：验证 broadcast + reduction 顺序不冲突

### 7.3 回归

- 现有 `foundation-smoke.mlir` 等 group-pipeline 测试保持 green
- 新增 broadcast case 加入 CI

---

## 8. 实施范围（v1）

### 包含
- BroadcastAbsorb pass（基本 pattern：`linalg.broadcast → linalg.generic`）
- TileFuse Phase 1 的广播轴剪枝逻辑
- TileFuse Phase 3 的 loop order 规则调整
- `CollapsedGroupInfo::broadcastAxes` 字段
- Broadcast Axis Independence Invariant verifier（L1 + L2）
- §6.6 缓解规则 1（`kMaxFullLoopIters` escape，初值 2048）
- §6.6 缓解规则 2（累加器 UB 预算 assert + tile 降级 escape）
- §6.6 缓解规则 3（分核候选不足 warn）
- 7.1 中列出的单元测试

### 不包含（future work）
- BroadcastAbsorb 对 `tensor.reshape` 等中间 op 的跨越吸收
- 全 group 共同广播轴的 canonical 外提优化（§6.1）
- 相邻一致广播轴合并（§6.6 future work）
- 硬件多维 broadcast_l2 的 codegen 优化（§6.6 future work）
- Autotuner 对 Broadcast Axis Independence Invariant 的"受控违反"路径

### 不支持
- Cube group（正交，由 matmul 专有路径处理）
- Dynamic shape 下基于 size=1 的运行时广播识别

---

## 9. 设计检查清单

- [x] 与 AF ATT 的 broadcast 处理理念一致（§4.7.2 "不是怎么切，而是哪些轴不能按普通方式搜"）
- [x] 复用 MLIR linalg 原生能力（indexing_maps 缺轴），不引入冗余建模
- [x] 复用现有 `CollapsedGroupInfo::axisMap == -1` 与 `TilePlan::full` 语义
- [x] 硬不变量 + verifier，契约清晰可 debug
- [x] 兼容 torch-mlir 等上游（显式 `linalg.broadcast` 由前置 pass 吸收）
- [x] 广播 + reduction 结构语义正确（§6.2）
- [x] 测试覆盖：吸收、剪枝、loop order、verifier 违反四类场景
- [x] Scope 明确，v1 可实施

---

## 10. 参考资料

- AF axis mapping: `ge-eco/.../can_fuse/backend/asc_graph_axis_mapping.cpp`
- AF ViewOpAttrInfo: `ge-eco/.../can_fuse/backend/backend_utils.h` §ViewOpAttrInfo
- AF ATT 轴处理: `ge-eco/.../doc/schedule_ATT轴处理全流程说明.md` §4.3.5 / §4.5 / §4.7.2
- 本项目 03-tile-fuse.md §114-213（方案 1/2 的数值推演）