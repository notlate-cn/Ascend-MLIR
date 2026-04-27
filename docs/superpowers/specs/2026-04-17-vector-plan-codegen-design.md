# Vector Plan Codegen 设计方案

**Date:** 2026-04-17
**Status:** Draft for review
**Depends on:** `docs/superpowers/specs/2026-04-14-vector-plan-unified-design.md`

---

## 1. 总体目标

将 vector-plan-tile-fuse（Pass 2）输出的 tiled+fused tensor-semantic IR
自动转换为 AscendC C++ kernel，覆盖 VectorGroup 和 CubeGroup 两类 group。

**输入**：`kernel_group{N}.mlir`（Pass 2 产出，含 `ascendc.*` 注解 + `tiling.infos` module attribute）
**输出**：`kernel_group{N}.cpp`（CANN 标准 AscendC C++ kernel）

---

## 2. 流水线

```
kernel_group{N}.mlir
  │
  ├─ one-shot-bufferize          [现有，复用]
  │    bufferize-function-boundaries=true
  │    allow-return-allocs-from-loops=true
  │    function-boundary-type-conversion=identity-layout-map
  ├─ annotate-ascendc-kernel-kind [现有，复用]
  ├─ cse
  ├─ ascendc-buffer-placement    [现有，复用]
  ├─ linalg-to-ascendc           [现有，复用]
  ├─ ascendc-parallelize         [扩展：N-D dispatch]
  ├─ canonicalize / cse
  ├─ ascendc-prepare-for-emit    [Phase B 迁移：读 tiling.infos]
  ├─ canonicalize-cann-signature [现有，复用]
  │
  └─ afir-translate -mlir-to-cann → kernel_group{N}.cpp
```

VectorGroup 与 CubeGroup 在同一流水线内处理，差异由 `ascendc.*` 注解和
`tiling.infos` 驱动，流水线本身无分支。

---

## 3. 接口契约：Pass 2 → Codegen

Pass 2 与 codegen 之间通过两类信息通信：

### 3.1 `tiling.infos`（module attribute，权威来源）

`tiling.infos` 是 Pass 2 与 codegen 之间的**唯一结构化接口**。
codegen passes 不从 IR 结构（循环层数、函数签名）反推语义。

每个 kernel func 对应一条 `TileInfo`，包含：
- `axes`：canonical 轴列表（含 name、role=parallel/reduction、extent 表达式）
- `fields`：tile 参数（`TunableTile`/`FixedTile`/`ShapeDim`/`Derived`），含 abi_index
- `block_dim`：**dispatch shape 的权威来源**（ValueExpr 数组，N 个元素表示 N-D dispatch）

### 3.2 `ascendc.*` 注解（IR attribute，循环/op 句柄标记）

| Annotation | 位置 | 语义 |
|---|---|---|
| `ascendc.parallel` | `scf.for` | 标记该循环对应 dispatch 的某一维（句柄，不决定维度数） |
| `ascendc.prologue` | `scf.for` | 该循环入口前执行的 data-move 任务（`"lhs:A1->A2,rhs:B1->B2"`）|
| `ascendc.epilogue` | `scf.for` | 该循环出口后执行的 data-move 任务 |
| `ascendc.unit` | linalg op | 计算单元（`"AiCore.Cube"` / `"AiCore.Vector"`）|

**`ascendc.parallel` 的语义约束**：
- 它是循环句柄标记，不是 dispatch 维度的来源
- dispatch 维度（N-D 还是 1D、各维 grid 大小）由 `tiling.infos.block_dim` 决定
- codegen pass 通过 `ascendc.parallel` 定位需要替换的循环，通过 `block_dim` 确定替换方式

---

## 4. `ascendc-parallelize`：N-D dispatch 扩展

### 4.1 现状与问题

现有 pass 在函数 entry block 找顶层 `scf.for`，每个独立替换为
`get_block_idx` + `muli`，是隐式 1D 设计。无法处理 CubeGroup 的
`for_BM { for_BN {…} }` 2D 并行结构。

### 4.2 新设计

**输入**：函数上的 `tiling.infos`，标有 `ascendc.parallel` 的 `scf.for` 集合

**算法**：

```
Step 1：从 tiling.infos.block_dim 读取 N 和各维 grid_i
Step 2：收集函数内所有标有 ascendc.parallel 的 scf.for，按嵌套深度排序
         （outermost-first，对应 block_dim[0], block_dim[1], ...）
Step 3：生成 N-D 线性化展开
         block_idx = ascendc.get_block_idx()
         // 从内向外 unpack（row-major）
         idx[N-1] = block_idx % grid[N-1]
         idx[N-2] = (block_idx / grid[N-1]) % grid[N-2]
         ...
         idx[0]   = block_idx / (grid[1] * ... * grid[N-1])
         // 每个 idx[i] 乘以对应 step，得到 loop IV 的替代值
Step 4：把 N 层 ascendc.parallel scf.for 替换为 idx[i] * step[i]，
         保留 inner 循环（Tb/XBLOCK_SUB/t_K 层）在 scf.if guard 内
```

**N=1（VectorGroup）**：退化为现有行为，无额外开销。
**N=2（CubeGroup）**：生成 `block_idx / grid_n` 和 `block_idx % grid_n`。
**N≥3（未来场景）**：自动支持，pass 无需修改。

### 4.3 与 `block_dim` 的绑定

`grid_i` 的值在 dispatch 时由 host 计算（`total_blocks = ∏ grid_i`）。
pass 在 kernel 函数内只生成 div/mod 链，不固化 grid 数值——grid 大小
来自 tiling params（tunable，运行时传入）。

---

## 5. `ascendc-prepare-for-emit`：Phase B 迁移

### 5.1 现状与问题

现有 pass 扫描函数签名末尾的 `i64` 参数，按位置/名称前缀反推字段角色，
构造 `TilingData` struct。这是对 Pass 2 输出格式的脆弱假设。

### 5.2 新设计

Phase B：pass 直接读取 `tiling.infos.fields`，按 `abi_index` 顺序构造
`TilingData`：

```
for field in sorted(tiling_info.fields, key=abi_index):
    if field.kind in {TunableTile, FixedTile, ShapeDim}:
        add to TilingData struct at position abi_index
    // Derived 字段默认不进入 TilingData bytes（§10.2）
```

好处：
- 字段顺序由 `tiling.infos` 显式给出，不靠扫描位置反推
- 新增/删除 tiling 参数不影响 pass 逻辑
- VectorGroup（XBLOCK/XBLOCK_SUB）和 CubeGroup（BM/BN/Tb_M/Tb_N/t_K）
  统一走同一路径

### 5.3 迁移路径

| Phase | 内容 | 触发条件 |
|---|---|---|
| Phase A（当前）| PrepareForEmit 扫描 i64 args | 现有 examples 继续工作 |
| Phase B | 读 `tiling.infos.fields`，i64 args 扫描作为 fallback | Pass 2 `tiling.infos` 格式稳定后 |
| Phase C | 删除 fallback，hard-fail 若 `tiling.infos` 缺失 | 全部 examples 迁移完成后 |

---

## 6. VectorGroup vs CubeGroup 差异对照

| | VectorGroup | CubeGroup |
|---|---|---|
| dispatch 维度 | 1D（XBLOCK 轴）| 2D（BM × BN）|
| `block_dim` 元素数 | 1 | 2 |
| `ascendc.parallel` 循环数 | 1（`for_XBLOCK`）| 2（`for_BM`, `for_BN`）|
| `ascendc.unit` | 无（全 Vector）| matmul=Cube，epilogue=Vector |
| `ascendc.prologue` on K loop | 无 | `"lhs:A1->A2,rhs:B1->B2"` |
| buffer-placement 内存层 | VECIN/VECOUT/VECCALC | A1/A2/B1/B2/CO1 + VECIN/VECOUT |
| parallelize pass 行为 | N=1 退化路径 | N=2 路径，div/mod 展开 |
| PrepareForEmit | 同一路径（读 tiling.infos）| 同一路径 |

---

## 7. AutoTuner：Solver 抽象

### 7.1 现状

AutoTuner 当前读取手写 `tiling_space.json`，枚举 `values` 全量笛卡尔积，
逐个跑模拟器选最优，输出 `tiling_func.cpp`。

### 7.2 Solver 接口

将候选生成逻辑抽象为 `Solver` 接口，其余流程（编译、模拟、选优、输出）完全共享：

```
Solver 接口
  └─ generateCandidates(TilingSpace, shape) → vector<Config>

[通用流程，两个 Solver 共享]
  编译 kernel 一次
  → 对每个 Config：pack tiling bytes，计算 block_dim，跑模拟器
  → 选 cycle_count 最小的 Config
  → 输出 tiling_func.cpp
```

| Solver | `generateCandidates` 逻辑 | 候选数量 |
|---|---|---|
| `DefaultSolver` | 每个 tunable param 从 `values` 取小/中/大三个代表值，笛卡尔积 | O(3^N)，N≤3 时约 9–27 个 |
| `GridSearchSolver` | 全部 `values` 笛卡尔积 | 全量 |

命令行：`--solver=default`（新默认）/ `--solver=search`（全搜索）。

`tiling_space.json` 格式不变，现有 examples 完全兼容。

### 7.3 与 tiling.infos 的关系

Pass 2 生成 `tiling.infos` 后，需要一个轻量转换步骤生成 `tiling_space.json`：

```
tiling.infos (MLIR module attribute)
    ↓  tiling-infos-to-json（MLIR pass 或独立工具）
tiling_space.json
    ↓
AutoTuner --solver=default
    ↓
tiling_func.cpp
```

转换规则：
- `TunableTile` field → `{fixed: false, values: search.candidates}`
- `ShapeDim` field → `{fixed: true, shape_key: "argN_dimM"}`
- `block_dim` ValueExpr 数组 → `block_dim_expr` 乘积字符串（`"ceil(M/BM)*ceil(N/BN)"`）

### 7.4 扩展路径

未来只需实现新 `Solver` 的 `generateCandidates`，注册后通过 `--solver=<name>` 触发：
- `MLSolver`：用模型预测最优 config
- `BayesianSolver`：贝叶斯优化迭代搜索

---

## 8. 已知边界与暂不处理

- **`ComputeConversion.cpp` 覆盖范围**：当前处理 `linalg.generic`（pointwise/reduction body）和 `linalg.matmul`/`linalg.fill`。Pass 2 产出中若出现现有 pattern 未覆盖的 generic body，需在 `ComputeConversion.cpp` 内新增 pattern——迭代扩展，遇到时再处理。
- **3D+ dispatch 验证**：设计支持 N≥3，但无具体 group 类型触发，暂不写测试。
- **`network.mlir` host launch codegen**：coordinator func 的 host 调用生成不在本文档范围。