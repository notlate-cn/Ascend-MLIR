# Vector Plan Generation — 各阶段关键数据结构

**Date:** 2026-04-14  
**Scope:** `vector-plan-generation` pass 内部各阶段的数据结构定义、所有权与关键不变量  
**Companion docs:**  
- 实施方案：`docs/superpowers/plans/2026-04-10-vector-plan-generation-final.md`  
- TileInfo 设计：`docs/superpowers/specs/2026-04-14-tile-info-design.md`

---

## 1. 流水线总览

```
func::FuncOp (linalg-on-tensor)
        │
        ▼  Task 2: Chain Analysis
  SmallVector<ChainInfo>
        │
        ▼  Task 3: Chain Collapse
  SmallVector<CollapsedChainInfo>   ← 同时修改 IR（插入 collapse_shape）
        │
        ▼  Task 4: TilePlan Generation
  SmallVector<TilePlan>             ← 同时修改 func signature（追加 index args）
        │
        ▼  Task 5: TilePlan Realization
  （IR 就地变形：scf.for + fuse）
        │
        ▼  Task 6: Module Metadata
  module attr: tiling.infos         ← TileInfo（权威）
  module attr: tiling.tiles         ← compat 投影
  module attr: tiling.shapes        ← compat 投影
```

**所有权原则**：每个阶段的产物在 `runOnOperation` 的同一 scope 里存活，任何 `SmallVector` 在构建完成后不再 push_back，从而保证其他结构持有的视图/指针稳定。

---

## 2. 各阶段数据结构

---

### 2.1 Task 2 产物：`ChainInfo`

```cpp
enum class AxisKind : uint8_t { Parallel, Reduction };

struct AxisInfo {
  std::string name;       // 如 "B", "S", "H"；合并轴如 "BS"；可为空
  int64_t     staticSize; // ShapedType::kDynamic 表示动态
  AxisKind    kind;
};

struct ChainInfo {
  linalg::LinalgOp              root;
  SmallVector<linalg::LinalgOp> members;      // 程序序，root 在末尾
  SmallVector<AxisInfo>         canonicalAxes; // root 的 iter space

  // chain 边界
  SmallVector<Value>            boundaryIn;   // 从 chain 外流入的 tensor
  SmallVector<Value>            boundaryOut;  // 从 chain 流出的 tensor

  std::string                   cutReason;    // 本 chain 因何被切开
};
```

**关键不变量**：
- `members` 里每个 op 都在同一 `Block` 里，且在 `root` 之前（program order）。
- `canonicalAxes` 完整覆盖 root 的 iteration domain，顺序与 root iterator_types 对齐。
- `Tileable` ∩ `Full` = ∅（轴冲突时在吸收判据处切开 chain）。

---

### 2.2 Task 3 产物：`CollapsedChainInfo`

```cpp
struct CollapsedChainInfo : ChainInfo {
  SmallVector<AxisInfo> collapsedAxes; // collapse 后的轴，≤ canonicalAxes.size()
  SmallVector<int>      axisMap;       // canonicalAxes[i] 映射到 collapsedAxes[axisMap[i]]
};
```

**IR 副作用**（Task 3 同时改写 IR）：
- 为 Case C 成员调用 `linalg::collapseOpIterationDims`
- 为 Case B 成员插入 `linalg.broadcast`（打 `vector_plan.no_collapse = true`）
- 在 chain 边界插入 `tensor.collapse_shape` / `tensor.expand_shape`

**关键不变量**：
- 若无法 collapse（所有轴都是候选组边界打断），则 `collapsedAxes == canonicalAxes`，`axisMap` 是恒等映射。
- v1 目标：LayerNorm / Softmax 压到 1 Tileable + 1 Full（2D）。

---

### 2.3 Task 4 产物：`TilePlan`

```cpp
enum class TileLevel : uint8_t {
  Outer,  // 外层多核分派，如 XBLOCK
  Inner,  // 内层 UB 批，如 XBLOCK_SUB / XBLOCK_SUB_i
  Full,   // 完整轴，如 RBLOCK_j（默认 = 维长）
};

struct TileParam {
  std::string  name;         // "XBLOCK" / "XBLOCK_SUB" / "RBLOCK_0" 等
  Value        ssa;          // 追加到 func signature 的 index arg
  OpFoldResult defaultValue; // 静态 shape → IntegerAttr；动态 shape → Value
  int32_t      axisIdx;      // 对应 collapsedAxes 的下标
  TileLevel    level;        // Outer / Inner / Full
                             // ← 必须在 Task 4 显式赋值，Task 6 直接读取
};

struct TilePlan {
  int chainIdx; // 指向 SmallVector<CollapsedChainInfo>[chainIdx]
                // 用下标代替裸指针，避免 vector realloc 后悬空

  // tileable[0] 永远是 {XBLOCK(Outer), XBLOCK_SUB(Inner)} 两个参数
  // tileable[i>0] 永远是 {XBLOCK_SUB_i(Inner)} 一个参数
  SmallVector<SmallVector<TileParam>> tileable;

  // full[j] 永远是 {RBLOCK_j(Full)} 一个参数
  SmallVector<TileParam> full;

  // ceildiv(extent(tileable[0].axis), XBLOCK)
  // v1 只填一个元素；将来 2D grid 追加 [1]
  SmallVector<Value, 2> blockDimValues;
};
```

**IR 副作用**（Task 4 同时改写 func signature）：
- 往 func 末尾追加所有 chain 的 tile index args（一次性批量追加，顺序为 chain0 params, chain1 params, ...）。
- 追加前记录 `originalArgCount = func.getNumArguments()`，Task 6 用此边界区分原始 tensor args 和 tile index args。

**关键不变量**：
- 每个 `TileParam` 都有 `level`；Task 6 的 `buildTileInfo` 直接读，不做位置推断。
- `blockDimValues[0]` = `arith::ceildivui(materialize(extent), xblockSSA)`，在 Task 4 用 `getValueOrCreateConstantOp(rw, loc, extent)` 处理静态/动态 extent。

---

### 2.4 Task 5 产物：tiled IR

Task 5 不产生新的 C++ 数据结构，产出是**就地变形的 IR**。

**两级 tile 的正确续接方式**：

```cpp
// 第一次 tile：outer split（XBLOCK）
SCFTileAndFuseResult outer = tileConsumerAndFuseProducersUsingSCF(
    rw, tilingInterface, outerOpts);
// outer.replacements 持有 {original op → tiled op within outer loop}

// 取出第一次 tile 后的目标 op，用于第二次 tile
auto tiledRoot = cast<TilingInterface>(
    outer.tiledAndFusedOps.back()); // 或从 replacements[chainRoot] 取

// 第二次 tile：inner split（XBLOCK_SUB），在 tiledRoot 上调用
SCFTileAndFuseResult inner = tileConsumerAndFuseProducersUsingSCF(
    rw, tiledRoot, innerOpts);
```

**关键不变量**（FileCheck 必须显式断言）：
1. 外层 `scf.for` 带 `{ascendc.parallel = true}`。
2. chain 内所有 linalg op 都落在最内层 loop body，loop 外不剩 chain 成员。
3. reduction 产出 tensor 的 static shape = tile size（不是原始维长）。

---

### 2.5 Task 6 产物：`TileInfo` / module attributes

**从 `TilePlan` 到 `TileInfo` 的转换边界**：

```
TilePlan 持有的信息               →  TileInfo 持有的信息
────────────────────────────────────────────────────────
OpFoldResult defaultValue         →  ValueExpr defaultExpr
Value blockDimValues[i]           →  ValueExpr blockDimExprs[i]
TileLevel level                   →  TileFieldSpec.level
int chainIdx + collapsedAxes      →  TileAxisInfo list
int axisIdx                       →  TileFieldSpec.axisIndex
```

**func args 的边界划分**（决定 ShapeDim 字段范围）：

```
func.func @kernel(
  %arg0: tensor<...>,  // ← originalArgCount 之内的 ShapedType args
  %arg1: tensor<...>,  //    只取这部分做 shape dim 字段
  %arg2: tensor<...>,  //    通常是 inputs；outputs 是否纳入由具体策略决定（见下）
  %XBLOCK: index,      // ← originalArgCount 之后，tile index args，跳过
  %XBLOCK_SUB: index,
  %RBLOCK_0: index
)
```

> **待定决策**：输出 tensor 的 shape dim 是否写入 TileInfo.fields（影响 TilingData ABI 大小）。
> v1 建议：只写 `index < originalArgCount` 的 ShapedType args，输入/输出都写；
> 如果输出 shape 完全由输入 shape 决定（静态 shape v1 场景），也可只写输入。
> 无论哪种，必须与 `tiling.shapes` 的 Step 6.2 保持一致。

**`opFoldResultToValueExpr` 支持的模式**（v1 最小集）：

| OpFoldResult 形态 | 转成 ValueExpr |
|---|---|
| `IntegerAttr(n)` | `{Const, n}` |
| `tensor.dim(%arg_i, %c_j)` | `{ShapeDim, argIndex=i, dimIndex=j}` |
| `arith.ceildivui(x, y)` | `{CeilDiv, lhs=cvt(x), rhs=cvt(y)}` |
| `arith.muli(x, y)` | `{Mul, lhs=cvt(x), rhs=cvt(y)}` |

其余模式在 v1 直接报错（`emitError` + return failure），不做 fallback。

**module 上最终写入的三个属性**：

| 属性 | 权威性 | 消费方 |
|---|---|---|
| `tiling.infos` | **权威** | PrepareForEmit（Task 6 之后）、AutoTuner（Migration Phase C 后） |
| `tiling.tiles` | compat 投影 | 当前 autotuner（Migration Phase C 之前） |
| `tiling.shapes` | compat 投影 | 当前 autotuner（同上） |

---

## 3. 跨阶段所有权与生命期

```
runOnOperation() 栈帧
├── SmallVector<ChainInfo>           chains      ← Task 2 产物
├── SmallVector<CollapsedChainInfo>  collapsed   ← Task 3 产物
│   └── 构建完成后不再 push_back（保证地址稳定）
├── int originalArgCount                         ← Task 4 追加 args 之前记录
├── SmallVector<TilePlan>            plans       ← Task 4 产物
│   └── plans[i].chainIdx 是 collapsed 的下标，不是指针
└── （Task 5 / Task 6 直接操作 IR，不产生新的持久数据结构）
```

**规则**：
- `TilePlan` 用 `chainIdx`（下标）引用 `CollapsedChainInfo`，不用裸指针，避免 `SmallVector` realloc 后悬空。
- `AxisInfo.name` 用 `std::string`（拥有存储），不用 `StringRef`（视图，会在 collapse 计算时悬空）。
- Task 5 从 `SCFTileAndFuseResult.tiledAndFusedOps` / `.replacements` 取第二次 tile 的目标 op，不持有 Task 4 传入的原始 `root`（已被 erase）。

---

## 4. 三个架构性决策点

以下三点需要在实现开始前明确，不是实现时边写边决定：

| # | 决策点 | 影响 | 建议 |
|---|---|---|---|
| A | 两级 tile 续接：第二次 `tileConsumerAndFuseProducersUsingSCF` 的目标 op 从哪取 | Task 5 整体结构 | 从第一次的 `SCFTileAndFuseResult.tiledAndFusedOps.back()` 取；若 upstream 不支持两次调用，退到单次 tile + 后置 split loop pass |
| B | `TileLevel` 在 `TileParam` 中显式存储，还是 Task 6 从位置推断 | TileInfo 正确性 | 显式存储（已在 §2.3 中定稿），Task 4 生成规则中赋值 |
| C | `buildTileInfo` 中 shape dim 字段覆盖哪些 args | TilingData ABI layout | v1：所有 `index < originalArgCount` 的 ShapedType args（含输入和输出），与 `tiling.shapes` 对齐 |
