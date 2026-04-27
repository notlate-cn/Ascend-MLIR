# TileInfo Design

**Date:** 2026-04-14  
**Scope:** Vector Phase 1 中 `TilePlan -> TilingData -> AutoTuner` 的连接接口  
**Status:** Draft for review

---

## 1. Problem

当前流水线里，tile 相关信息被拆散在三个层次里：

1. **Phase 1 / Plan Generation** 持有 `TilePlan`：
   - 哪些轴是 Parallel / Reduction
   - 哪些参数是 `XBLOCK` / `XBLOCK_SUB` / `RBLOCK_0`
   - 每个参数的默认值、轴归属、outer/inner/full 语义

2. **PrepareForEmit / step7** 物化 `TilingData`：
   - 把若干 tile 参数和 shape dim 组织成 `emitasc.py_struct<"TilingData", ...>`
   - 供 codegen、runtime 打包和 host tiling 使用

3. **AutoTuner** 需要搜索视图：
   - 哪些字段可调
   - 每个字段的候选值是什么
   - `block_dim` 如何由 shape 和 tile 计算
   - 哪些字段只是 shape 透传，不参与搜索

当前问题在于：

- `TilePlan` 是**语义对象**，但它混有 MLIR 内部对象（如 `Value` / `OpFoldResult`），不适合直接跨阶段暴露；
- `TilingData` 是**ABI 载体**，字段顺序和名字对 runtime 很重要，但它本身不该承担全部语义；
- autotuner 目前更接近从 step7 `emitasc.py_struct` 或 `tiling_space.json` **反推语义**，这会丢失 `outer/inner/full`、`axisIndex`、`defaultExpr` 等信息；
- 多 chain / 多 plan 时，平铺的 `tiling.tiles` / `tiling.shapes` 无法唯一表达“哪个字段属于哪个 kernel plan”；
- 动态 shape 下，`defaultValue` 可能来自 shape dim 表达式，而不是纯常量，现有平铺 schema 不足以表达。

因此需要一个**稳定、可序列化、单一权威**的中间层：**TileInfo**。

---

## 2. Goals

- 在 `TilePlan` 与 `TilingData` / autotuner 之间引入一个**明确的连接数据结构**：`TileInfo`。
- 让 **PrepareForEmit** 和 **AutoTuner** 都消费同一个权威 schema，而不是各自重新推断。
- 支持：
  - 多 chain
  - 多 plan
  - 静态 shape
  - 动态 shape 默认值表达（即使 v1 不做 e2e 动态测试）
- 保持 runtime ABI 简单：`RunArgs::tiling` 仍然是 packed bytes，不改现有执行接口。
- 把“哪些字段可调、哪些字段固定、哪些字段来自 shape”表达清楚。

---

## 3. Non-Goals

- 不在 v1 引入新 dialect。
- 不在 runtime 做 fusion 决策或 plan 选择。
- 不要求 autotuner 直接理解完整 `TilePlan`。
- 不要求 `TilingData` 承载全部编译语义。
- 不在 v1 做完整的动态 shape e2e 验证。

---

## 4. Layering

推荐的分层如下：

```text
TilePlan
  -> TileInfo
      -> (A) PrepareForEmit 生成 TilingData / emitasc.py_struct
      -> (B) 导出 AutoTuner 使用的搜索视图
  -> autotuner 搜索得到 TunedTileValues
      -> host tiling / runtime pack bytes
```

三层职责严格区分：

### 4.1 `TilePlan`
编译器内部语义对象，回答：

- 哪些轴参与 tiling
- 每个 tile param 作用于哪根轴
- outer / inner / full 是什么语义
- 默认值表达式是什么

### 4.2 `TileInfo`
稳定连接层，回答：

- 哪些字段会进入 `TilingData`
- 字段顺序是什么
- 哪些字段可调、哪些固定、哪些来自 shape
- `block_dim` 怎么算
- 这些信息属于哪个 chain / plan

### 4.3 `TilingData`
ABI 层的平铺 struct，回答：

- 最终 kernel 接收的字段顺序和字段名
- 每个字段的基础数据类型
- runtime 如何 pack bytes

### 4.4 `AutoTuner` 搜索视图
只关心：

- 可搜索字段
- 候选值集合 / 范围
- shape 透传字段
- `block_dim` 表达式

**重要原则：autotuner 不直接消费 `TilePlan`；PrepareForEmit 不反向推断 `TilePlan`。两者都消费 `TileInfo`。**

---

## 5. TileInfo 是单一权威源头

### 5.1 为什么 step7 `emitasc.py_struct` 不能作为唯一权威

step7 的 `emitasc.py_struct<"TilingData", ...>` 很适合做 ABI 物化，但它只天然表达：

- 字段名
- 字段顺序
- 字段类型

它**不能充分表达**：

- 哪个字段对应哪根 collapsed axis
- `XBLOCK` 是 outer split 还是 inner tile
- `RBLOCK_0` 是 reduction full 还是 tunable split
- 默认值是常量还是 shape 表达式
- 该字段是否允许 autotune
- 该字段属于哪个 chain / plan

因此 step7 产物应该是 **TileInfo 的一个派生物**，不是语义权威。

### 5.2 为什么 `tiling.tiles` / `tiling.shapes` 不够

平铺的：

```mlir
module attributes {
  tiling.tiles  = [...]
  tiling.shapes = [...]
}
```

在“单 kernel / 单 plan”时足够轻量，但在以下场景表达力不足：

- 一个 module 里有多个 chain
- 一个 chain 有多个 plan
- 需要知道某个 tile 字段的 `level = outer/inner/full`
- 需要表达默认值是 `shape(arg=0, dim=1)` 这种符号表达式
- 需要表达某个字段是 fixed 还是 tunable
- 需要记录 `block_dim_expr`

因此 `tiling.tiles` / `tiling.shapes` 可以保留为**兼容投影**或**调试视图**，但不应再是唯一权威源头。

---

## 6. Core Data Model

## 6.1 基本概念

### AxisRole

```cpp
enum class AxisRole : uint8_t {
  Parallel,
  Reduction,
};
```

### TileLevel

```cpp
enum class TileLevel : uint8_t {
  Outer,   // 外层分核，例如 XBLOCK
  Inner,   // 内层 tile，例如 XBLOCK_SUB / XBLOCK_SUB_i
  Full,    // 默认 full，例如 RBLOCK_0 = dim_size
};
```

### TileFieldKind

```cpp
enum class TileFieldKind : uint8_t {
  TunableTile, // autotuner 搜索的 tile 参数
  FixedTile,   // 固定 tile 值或默认 full 的字段
  ShapeDim,    // 运行时 shape 透传
  Derived,     // 由其他字段/shape 推导；通常不进 TilingData bytes
};
```

### ShapeRef

```cpp
struct ShapeRef {
  int32_t argIndex;
  int32_t dimIndex;
};
```

表示一个 primitive shape 绑定：`tensor arg[argIndex]` 的 `dimIndex` 维。

---

## 6.2 可序列化表达式：ValueExpr

为了支持：

- 常量默认值
- 来自 shape 的默认值
- `block_dim = ceildiv(M, XBLOCK)`
- collapse 后的复合表达式（如 `B*S`）

TileInfo 需要一个**可序列化**的小表达式系统，而不是直接暴露 `OpFoldResult` / `Value`。

推荐最小 DSL：

```cpp
struct ValueExpr {
  enum Kind {
    Const,
    ShapeDim,
    FieldRef,
    Mul,
    Add,
    CeilDiv,
    Min,
    Max,
  } kind;

  int64_t constValue;
  ShapeRef shape;
  std::string fieldId;
  // 使用 shared_ptr 而非 unique_ptr，使 ValueExpr 可拷贝，
  // 从而 TileFieldSpec / TileInfo 等持有它的结构体也保持可拷贝语义。
  std::shared_ptr<ValueExpr> lhs;
  std::shared_ptr<ValueExpr> rhs;
};
```

v1 必须支持的节点只有：

- `Const`
- `ShapeDim`
- `FieldRef`
- `Mul`
- `CeilDiv`

其余节点只做预留。

> **注意**：`ValueExpr` 使用 `shared_ptr` 而非 `unique_ptr`，是为了让整个
> `TileInfo` 保持可拷贝语义（SmallVector push_back、std::optional 赋值、
> MLIR attribute builder 均依赖此性质）。若日后切换为 arena 分配，
> 可将 `shared_ptr<ValueExpr>` 替换为 `ValueExpr*`（arena 管理生命周期），
> 接口形状不变。

---

## 6.3 `TileInfo` 建议结构

```cpp
struct TileAxisInfo {
  int32_t axisIndex;           // 对应 collapsed axis 的下标
  std::string axisName;        // 如 "BS" / "H" / "S_col"
  AxisRole role;               // Parallel / Reduction
  ValueExpr extentExpr;        // 这根轴的逻辑大小（primitive shape 组合）
};

struct SearchSpace {
  bool enabled = false;
  SmallVector<int64_t> candidates;
};

struct TileFieldSpec {
  std::string fieldId;         // 稳定语义 ID，如 "tile.xblock"
  std::string abiName;         // TilingData 字段名，如 "XBLOCK"
  std::string abiType;         // v1 统一用 "i64"
  // 在 TilingData struct 中的顺序。
  // 只有 packable 字段（TunableTile / FixedTile / ShapeDim）才有值；
  // Derived 字段此项为 std::nullopt，不进入 TilingData。
  std::optional<int32_t> abiIndex;
  TileFieldKind kind;          // TunableTile / FixedTile / ShapeDim / Derived

  // 语义信息
  std::optional<int32_t> axisIndex;
  std::optional<TileLevel> level;

  // 值来源
  std::optional<ShapeRef> shapeBinding;  // kind=ShapeDim 时必填
  std::optional<ValueExpr> defaultExpr;  // TunableTile / FixedTile 时填写

  // 搜索空间
  std::optional<SearchSpace> search;     // TunableTile 时填写
};

struct TileInfo {
  std::string kernelId;        // 如 "chain0_plan0"
  int32_t chainId;
  int32_t planId;

  SmallVector<TileAxisInfo> axes;
  SmallVector<TileFieldSpec> fields;

  // 不进入 TilingData，但 autotuner / runtime 打包时需要。
  // 支持多维 grid（如 Ascend 二维 block grid 场景），
  // v1 只填一个元素；若将来需要二维，追加第二个元素即可，
  // 消费方按 index 取 blockDimExprs[0] / blockDimExprs[1]。
  SmallVector<ValueExpr, 2> blockDimExprs;
};
```

---

## 7. Invariants

`TileInfo` 必须满足以下不变量：

1. **单一归属**：每个 `TileInfo` 只对应一个 `(chainId, planId)`。
2. **稳定 ID**：`fieldId` 用于语义对齐；`abiName` 仅用于 ABI / 可读性。
3. **ABI 顺序权威**：`abiIndex`（`std::optional<int32_t>`）是 packable 字段在 `TilingData` struct、step7 `emitasc.py_struct`、runtime pack 中的顺序唯一权威。只有 `TunableTile` / `FixedTile` / `ShapeDim` 字段持有有效 `abiIndex`；`Derived` 字段的 `abiIndex` 必须为 `std::nullopt`。
4. **Shape 只绑定 primitive dim**：`ShapeDim` 只能引用原始 tensor 参数的 primitive 维度，不直接存储 `B*S` 之类 composite binding。
5. **表达式不持有 MLIR Value**：跨阶段的数据结构只能持有 `ValueExpr`，不能持有 SSA Value。
6. **AutoTuner 只搜索 `TunableTile`**：`ShapeDim` 透传，`FixedTile` 不搜索，`Derived` 不直接 pack（`abiIndex` 为空）。
7. **blockDimExprs 不靠反推**：`block_dim` 的语义必须在 `TileInfo.blockDimExprs` 里显式表达，而不是由 autotuner 自己从 loop 结构猜。v1 只填 `blockDimExprs[0]`；二维 grid 场景追加 `blockDimExprs[1]`。

---

## 8. Interface Boundaries

## 8.1 `TilePlan -> TileInfo`

这一层是**语义降维**：

- 输入：`TilePlan`（编译期对象）
- 输出：`TileInfo`（稳定序列化对象）

转换规则：

1. `TilePlan` 中每个 collapsed axis 变成一条 `TileAxisInfo`
2. `TilePlan` 中每个 tile param 变成一个 `TileFieldSpec`
3. 原始 tensor 参数的 primitive dim 绑定变成 `ShapeDim` 字段
4. `blockDimExprs` 由 plan 生成阶段显式给出（v1 只填一个元素）

**注意**：
- `TilePlan.defaultValue` 若是 `OpFoldResult`，必须在这一层转成 `ValueExpr`
- `TilePlan` 中的 `Value` / `OpFoldResult` 不允许继续外流

---

## 8.2 `TileInfo -> TilingData`

这一层是**ABI 物化**：

- `PrepareForEmit` 不再通过“扫描 i64 args + memref.dim”反向猜字段；
- 它应当直接读取 `TileInfo.fields`，按 `abiIndex` 构造：
  - `emitasc.py_struct<"TilingData", ...>`
  - `emitasc.member` 的字段顺序
  - host tiling 需要的字段布局

推荐规则：

- `TunableTile` / `FixedTile` / `ShapeDim` → 进入 `TilingData`
- `Derived` → 默认不进入 `TilingData` bytes，除非某个后端显式要求

也就是说，`TilingData` 是 `TileInfo.fields` 的一个 ABI 投影，而不是另一个独立 schema。

---

## 8.3 `TileInfo -> AutoTuner`

这一层是**搜索投影**：

autotuner 不需要看到完整 `TileInfo`，只需要：

- 可调字段：`kind = TunableTile`
- 对应候选值：`search.candidates`
- shape 透传字段：`kind = ShapeDim`
- 固定字段：`kind = FixedTile`
- `blockDimExprs`

因此可以从 `TileInfo` 派生出一个更瘦的 `AutotuneSpec`：

```cpp
struct AutotuneParam {
  std::string fieldId;
  std::string abiName;
  SmallVector<int64_t> candidates;
};

struct AutotuneSpec {
  std::string kernelId;
  SmallVector<AutotuneParam> params;
  SmallVector<TileFieldSpec> fixedFields;
  SmallVector<TileFieldSpec> shapeFields;
  // 与 TileInfo.blockDimExprs 对应，v1 只有一个元素。
  SmallVector<ValueExpr, 2> blockDimExprs;
};
```

### 关键规则

- autotuner **不从 step7 loop 结构反推 `block_dim`**
- autotuner **不从字段名前缀猜字段角色**（例如不靠 `TB_` / `dim_arg` 前缀）
- autotuner **不需要理解 Chain / Collapse 语义**，只消费 `TileInfo` 的投影结果

---

## 8.4 AutoTuner 输出 -> runtime

autotuner 最终输出的是选中的 tile 值：

```cpp
struct TunedTileValues {
  std::string kernelId;
  SmallVector<std::pair<std::string, int64_t>> values; // fieldId -> chosen value
};
```

### ValueExpr 求值接口

`FixedTile` 字段的 `defaultExpr` 以及 `blockDimExprs` 在打包时需要对 `ValueExpr` 求值。
必须在 host tiling 层提供如下接口：

```cpp
/// 对一个 ValueExpr 求值。
/// @param shapeValues  dispatch 输入中每个 tensor arg 的各维 shape 值，
///                     shapeValues[argIndex][dimIndex]
/// @param tuned        autotuner 选出的 tile 值（TunableTile 字段查这里）
int64_t evalExpr(const ValueExpr& expr,
                 ArrayRef<SmallVector<int64_t>> shapeValues,
                 const TunedTileValues& tuned);
```

`evalExpr` 只在 host tiling（CPU 侧）调用，不进入 kernel，不进入 runtime 热路径。

### 打包流程

runtime / host tiling 在打包 bytes 时：

1. 读取 `TileInfo.fields`
2. 按 `abiIndex`（有值的 packable 字段）从小到大遍历
3. 对于：
   - `TunableTile`：从 `TunedTileValues` 取值
   - `FixedTile`：调用 `evalExpr(defaultExpr, shapeValues, tuned)`
   - `ShapeDim`：从 dispatch 输入的实际 shape 取值（等价于 `evalExpr(ShapeDim expr, ...)`）
4. pack 成 `RunArgs::tiling`

这样 runtime 不需要知道 `TilePlan`，只需要知道 `TileInfo`。

---

## 9. MLIR Representation

推荐把 `TileInfo` 作为 **module-level attribute** 挂在 IR 上，名称建议：

```mlir
tiling.infos
```

示意：

```mlir
module attributes {
  tiling.infos = [
    {
      kernel = "chain0_plan0",
      chain = 0 : i64,
      plan = 0 : i64,
      axes = [
        {axis = 0 : i64, name = "BS", role = "parallel",
         extent = {op = "mul",
                   lhs = {op = "shape_dim", arg = 0 : i64, dim = 0 : i64},
                   rhs = {op = "shape_dim", arg = 0 : i64, dim = 1 : i64}}},
        {axis = 1 : i64, name = "H", role = "reduction",
         extent = {op = "shape_dim", arg = 0 : i64, dim = 2 : i64}}
      ],
      fields = [
        {id = "tile.xblock", abi_name = "XBLOCK", abi_type = "i64",
         abi_index = 0 : i64, kind = "tunable", axis = 0 : i64,
         level = "outer", default = {op = "const", value = 256 : i64},
         search = [64 : i64, 128 : i64, 256 : i64]},
        {id = "tile.xblock_sub", abi_name = "XBLOCK_SUB", abi_type = "i64",
         abi_index = 1 : i64, kind = "tunable", axis = 0 : i64,
         level = "inner", default = {op = "const", value = 64 : i64},
         search = [32 : i64, 64 : i64]},
        {id = "tile.rblock0", abi_name = "RBLOCK_0", abi_type = "i64",
         abi_index = 2 : i64, kind = "fixed", axis = 1 : i64,
         level = "full",
         default = {op = "shape_dim", arg = 0 : i64, dim = 2 : i64}},
        {id = "shape.arg0.0", abi_name = "dim_arg0_0", abi_type = "i64",
         abi_index = 3 : i64, kind = "shape_dim", from_arg = 0 : i64, dim = 0 : i64},
        {id = "shape.arg0.1", abi_name = "dim_arg0_1", abi_type = "i64",
         abi_index = 4 : i64, kind = "shape_dim", from_arg = 0 : i64, dim = 1 : i64},
        {id = "shape.arg0.2", abi_name = "dim_arg0_2", abi_type = "i64",
         abi_index = 5 : i64, kind = "shape_dim", from_arg = 0 : i64, dim = 2 : i64}
      ],
      block_dim = [{op = "ceildiv",
                    lhs = {op = "shape_dim", arg = 0 : i64, dim = 0 : i64},
                    rhs = {op = "field_ref", id = "tile.xblock"}}]
    }
  ]
}
```

`block_dim` 改为数组（对应 `blockDimExprs`），v1 固定只有一个元素。

### MLIR 属性类型化建议

当前 `tiling.infos` 使用裸 `DictionaryAttr` / `ArrayAttr` 嵌套，存在以下风险：

- 字段名拼写错误不会报错
- 没有 verifier，invariants 只能在运行时发现违反
- `kind = "tunable"` 等字符串 enum 无法做穷举检查

**推荐**：为 `TileInfo`、`TileFieldSpec`、`ValueExpr` 定义 TableGen `AttrDef`（挂在现有 dialect 下，不需要引入新 dialect）。至少对 `ValueExpr` 和顶层 `TileInfo` 做类型化，`TileFieldSpec` 可先用 `DictionaryAttr` 过渡。

这不违反 Non-Goals 中"不引入新 dialect"的约束，只是在现有 dialect 里增加 Attribute 定义。

### 与现有 `tiling.tiles` / `tiling.shapes` 的关系

推荐规则：

- `tiling.infos`：**权威表示**
- `tiling.tiles` / `tiling.shapes`：可选兼容投影

兼容投影适用于：

- 单 kernel / 单 plan 模块
- 调试输出
- 旧 autotuner / 旧工具短期兼容

但新实现不应只依赖这两个平铺字段。

---

## 10. Example: LayerNorm

假设 collapse 之后的链是二维：

- axis 0: `B*S`，Parallel
- axis 1: `H`，Reduction

则 `TilePlan` 语义上会产生：

- `XBLOCK`：axis 0 的 outer split
- `XBLOCK_SUB`：axis 0 的 inner tile
- `RBLOCK_0`：axis 1 的 full reduction，默认值为 `H`

对应的 `TileInfo`：

- `axes[0] = {axisIndex=0, role=Parallel, extentExpr=B*S}`
- `axes[1] = {axisIndex=1, role=Reduction, extentExpr=H}`
- `fields`：
  - `tile.xblock`
  - `tile.xblock_sub`
  - `tile.rblock0`
  - `shape.arg0.0`
  - `shape.arg0.1`
  - `shape.arg0.2`
- `blockDimExprs[0] = ceildiv(B*S, XBLOCK)`（v1 只填一个元素），取决于 Phase 1 对 block-parallel 轴的定义

**重点**：这里的 `RBLOCK_0` 虽然默认 full、通常不参与搜索，但它仍然应当出现在 `TileInfo.fields` 里，从而保持“每根轴都有明确参数语义”的完整性。

---

## 11. Migration Plan

### Phase A：引入 TileInfo，但不破坏现有运行接口

- Phase 1 新增 `TilePlan -> TileInfo` 转换
- module 上新增 `tiling.infos`
- runtime ABI 不变，`RunArgs::tiling` 仍然是 raw bytes

### Phase B：PrepareForEmit 改为消费 TileInfo

- 不再扫描 i64 参数顺序猜 `TB_M/TB_N/...`
- 直接按 `tiling.infos[*].fields[abiIndex]` 生成 `TilingData`

### Phase C：AutoTuner 改为消费 TileInfo 投影

- 直接读取 `tiling.infos`，**不保留旧代码路径的 fallback**
- Phase C 完成后，旧的"从 step7 `emitasc.py_struct` 反推"逻辑应直接删除
- 若 `tiling.infos` 缺失，应 hard-fail（明确错误信息），而非静默降级到旧路径
- step7 `emitasc.py_struct` 可在 Phase C 期间做**一次性一致性校验**（对比两条路径的输出），校验通过后即可移除

### Phase D：旧平铺元数据降级为兼容视图

- `tiling.tiles` / `tiling.shapes` 保留，但不再是唯一权威

---

## 12. Deferred Items

以下内容可延期，不阻塞 v1：

1. `Derived` 字段是否需要进入 `TilingData`
2. `ValueExpr` 是否要支持更多运算（如 `Sub` / `FloorDiv` / `Select`）
3. `abiType` 是否需要支持 `i32`
4. `TileInfo` 是否需要额外导出成 sidecar JSON，供非 MLIR 工具直接消费
5. 多 plan 情况下，module 中“默认 plan”的选择与标记方式

---

## 13. Recommended Rule of Thumb

一句话总结：

> `TilePlan` 负责描述“怎么切”；`TileInfo` 负责描述“怎么传”；`AutoTuner` 只负责描述“怎么搜”。

这是 v1 最稳的接口边界。