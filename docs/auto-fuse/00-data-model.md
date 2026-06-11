# Vector Plan Generation — 各阶段关键数据结构

**Date:** 2026-04-23  
**Scope:** `auto-fuse` pass 流水线内部各阶段的数据结构定义、所有权与关键不变量  
**Companion docs:**  
- 架构总纲：[00-architecture.md](./00-architecture.md)  
- TileInfo 设计：[04-tile-info.md](./04-tile-info.md)

---

## 1. 流水线总览

```
func::FuncOp (linalg-on-tensor)
        │
        ▼  Phase 1: Group Analysis
  auto_fuse.group_id / topo_index  ← IR attribute(Outline Pass 消费后 strip)
        │
        ▼  Outline Pass
  kernel_group{N}.mlir               ← 每个 group 一个独立文件
        │
        ▼  Phase 3a: Group Collapse
  CollapsedGroupInfo                  ← 同时修改 IR（插入 collapse_shape）
        │
        ▼  Phase 3b: TilePlan Generation
  SmallVector<TilePlan>              ← 同时修改 func signature（追加 index args）
        │
        ▼  Phase 3c: TilePlan Realization
  （IR 就地变形：scf.for + fuse）
        │
        ▼  Phase 3d: Module Metadata
  module attr: tiling.infos          ← TileInfo（权威）
  module attr: tiling.tiles          ← compat 投影
  module attr: tiling.shapes         ← compat 投影
```

**所有权原则**：每个阶段的产物在 `runOnOperation` 的同一 scope 里存活，任何 `SmallVector` 在构建完成后不再 push_back，从而保证其他结构持有的视图/指针稳定。

---

## 2. 核心枚举类型

```cpp
enum class AxisRole : uint8_t {
  Parallel,   // Tileable 轴
  Reduction,  // Full / Inner 轴
};

enum class TileLevel : uint8_t {
  Outer,  // 外层多核分派：XBLOCK / BM / BN
  Inner,  // 内层 UB 批：XBLOCK_SUB / Tb_M / RBLOCK_sub / t_K
  Full,   // 完整轴：RBLOCK（v1 不切）
};

enum class TileFieldKind : uint8_t {
  TunableTile, // autotuner 搜索的 tile 参数
  FixedTile,   // 固定 tile 值或默认 full 的字段
  ShapeDim,    // 运行时 shape 透传
  Derived,     // 由其他字段 / shape 推导；不进 TilingData bytes
};

enum class GroupKind : uint8_t {
  Vector,
  Cube,
};
```

---

## 3. Phase 1 产物：Group Annotation (IR attribute)

Pass 1 不产生 C++ 持久结构，产出是 **per-op IR attribute**：

```mlir
%0 = linalg.reduce { ... }
     {auto_fuse.group_id = 0 : i32,
      auto_fuse.topo_index = 2 : i32}
```

Pass 1 内部使用轻量 `FusionGroup` 结构进行迭代融合，详见 [01-group-analysis.md](./01-group-analysis.md)。

---

## 4. Outline Pass 产物：GroupInfo

Outline Pass 从 IR attribute + def-use 重建 `GroupInfo`，然后 outline 为独立 func。
Pass 2 对 kernel func 内所有 linalg op 重跑上确界推导，得到相同的 canonical axes。

```cpp
struct AxisInfo {
  std::string name;       // "B", "S", "H"；合并轴如 "BS"；可为空
  int64_t     staticSize; // ShapedType::kDynamic 表示动态
  AxisRole    kind;
};

struct GroupInfo {
  GroupKind                     kind;
  SmallVector<linalg::LinalgOp> topoMembers;   // 拓扑序（确定性遍历顺序）
  SmallVector<linalg::LinalgOp> sinks;         // group 末端（无 group 内消费者）
  SmallVector<AxisInfo>         canonicalAxes; // 上确界推导所得
  SmallVector<Value>            boundaryIn;    // group 外输入 tensor
  SmallVector<Value>            boundaryOut;   // group 外输出 tensor
};
```

**关键不变量**：
- `topoMembers` 里每个 op 都在同一 `Block` 里
- `canonicalAxes` 完整覆盖 group 的 iteration domain（对所有成员取 join）
- Tileable ∩ Reduction = ∅（由上确界规则保证）

---

## 5. Phase 3a 产物：CollapsedGroupInfo

```cpp
struct CollapsedGroupInfo : GroupInfo {
  SmallVector<AxisInfo>         collapsedAxes; // collapse 后的轴，≤ canonicalAxes.size()
  SmallVector<int>              axisMap;       // canonicalAxes[i] → collapsedAxes[axisMap[i]]
  SmallVector<int>              broadcastAxes; // BAII 广播轴 (post-collapse idx)
  DenseMap<int, Value>          broadcastAxisExtents;
};
```

**IR 副作用**（Collapse 同时改写 IR）：
- Case C 成员：调用 `linalg::collapseOpIterationDims`
- Group 边界：插入 `tensor.collapse_shape` / `tensor.expand_shape`

**关键不变量**：
- 若无法 collapse，则 `collapsedAxes == canonicalAxes`，`axisMap` 是恒等映射
- **BAII L1 保证**：广播轴的 axisMap 映射为 -1（不被 merge 到其他轴）
- v1 目标：LayerNorm / Softmax 压到 1 Tileable + 1 Full（2D）

### CubeGroup 专用扩展

```cpp
struct CubeGroupInfo : GroupInfo {
  linalg::LinalgOp matmul;
  linalg::LinalgOp epilogueAnchor; // 最后 epilogue op；无 epilogue 时 == matmul
};
```

CubeGroup 不做 Collapse（恒等映射），M/N 轴始终独立。

---

## 6. Phase 3b 产物：TilePlan

```cpp
struct TileParam {
  std::string  name;         // "XBLOCK" / "XBLOCK_SUB" / "RBLOCK_0" / "BM" 等
  Value        ssa;          // 追加到 func signature 的 index arg
  OpFoldResult defaultValue; // 静态 shape → IntegerAttr；动态 shape → Value
  int32_t      axisIdx;      // 对应 collapsedAxes 的下标
  TileLevel    level;        // Outer / Inner / Full
  AxisRole     role;         // Parallel / Reduction
};

struct TilePlan {
  int groupIdx; // 指向 SmallVector<CollapsedGroupInfo>[groupIdx]
                // 用下标代替裸指针，避免 vector realloc 后悬空

  // VectorGroup：
  //   tileable[0] = {XBLOCK(Outer), XBLOCK_SUB(Inner)}
  //   tileable[i>0] = {XBLOCK_SUB_i(Inner)}
  // CubeGroup：
  //   tileable[M] = {BM(Outer), Tb_M(Inner)}
  //   tileable[N] = {BN(Outer), Tb_N(Inner)}
  //   tileable[batch_i] = {XBLOCK_i(Outer), XBLOCK_SUB_i(Inner)}
  SmallVector<SmallVector<TileParam>> tileable;

  // VectorGroup full[j] = {RBLOCK_j(Full 或 Inner)}
  // CubeGroup   full[K] = {t_K(Inner)}
  SmallVector<TileParam> full;

  // VectorGroup：1 个元素 ceildiv(extent, XBLOCK)
  // CubeGroup：  2 个元素 ceildiv(M, BM), ceildiv(N, BN)
  SmallVector<OpFoldResult, 2> blockDimExprs;
};
```

**IR 副作用**（TilePlan 同时改写 func signature）：
- 往 func 末尾批量追加所有 tile index args
- 追加前记录 `originalArgCount`，Phase 3d 用此区分原始 tensor args 和 tile index args

**关键不变量**：
- 每个 `TileParam` 都有显式 `level`（不做位置推断）
- `blockDimExprs[0]` = `ceildiv(extent, xblockSSA)`

---

## 7. Phase 3c 产物：tiled IR

Phase 3c 不产生新的 C++ 数据结构，产出是**就地变形的 IR**。

**关键不变量**（FileCheck 必须断言）：
1. 外层 `scf.for` 带 `{ascendc.parallel = true}`
2. Group 内所有 linalg op 都落在最内层 loop body
3. Reduction 产出 tensor 的 static shape = tile size（非原始维长）

---

## 8. Phase 3d 产物：TileInfo / module attributes

详细设计见 [04-tile-info.md](./04-tile-info.md)。

**从 TilePlan 到 TileInfo 的转换边界**：

| TilePlan 持有的信息 | → | TileInfo 持有的信息 |
|---|---|---|
| `OpFoldResult defaultValue` | → | `ValueExpr defaultExpr` |
| `Value blockDimValues[i]` | → | `ValueExpr blockDimExprs[i]` |
| `TileLevel level` | → | `TileFieldSpec.level` |
| `int groupIdx + collapsedAxes` | → | `TileAxisInfo` list |

**module 上最终写入的三个属性**：

| 属性 | 权威性 | 消费方 |
|---|---|---|
| `tiling.infos` | **权威** | PrepareForEmit、AutoTuner（Phase C 后） |
| `tiling.tiles` | compat 投影 | 当前 autotuner（Phase C 之前） |
| `tiling.shapes` | compat 投影 | 当前 autotuner（同上） |

---

## 9. 跨阶段所有权与生命期

```
runOnOperation() 栈帧
├── int originalArgCount                         ← Phase 3b 追加 args 之前记录
├── SmallVector<CollapsedGroupInfo>  collapsed   ← Phase 3a 产物
│   └── 构建完成后不再 push_back（保证地址稳定）
├── SmallVector<TilePlan>            plans       ← Phase 3b 产物
│   └── plans[i].groupIdx 是 collapsed 的下标，不是指针
└── （Phase 3c / 3d 直接操作 IR，不产生新的持久数据结构）
```

**规则**：
- `TilePlan` 用 `groupIdx`（下标）引用 `CollapsedGroupInfo`，不用裸指针
- `AxisInfo.name` 用 `std::string`（拥有存储），不用 `StringRef`（会悬空）
- Phase 3c 从 `SCFTileAndFuseResult` 取第二次 tile 的目标 op，不持有原始 `root`（已被 erase）

---

## 10. 架构性决策点

| # | 决策点 | 影响 | 决策 |
|---|---|---|---|
| A | 两级 tile 续接的目标 op 来源 | Phase 3c 结构 | 从 `SCFTileAndFuseResult.tiledAndFusedOps.back()` 取 |
| B | `TileLevel` 在 `TileParam` 中存储方式 | TileInfo 正确性 | 显式存储（§6 已定稿） |
| C | `buildTileInfo` 中 shape dim 字段覆盖范围 | TilingData ABI | v1：所有 `index < originalArgCount` 的 ShapedType args |
