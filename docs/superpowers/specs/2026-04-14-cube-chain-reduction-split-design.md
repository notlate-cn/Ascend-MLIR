# Cube Chain Fusion 与 Reduction Splitting 设计

**Date:** 2026-04-14  
**Scope:** 在 `vector-plan-generation` pass 中新增 Cube Chain（matmul + epilogue 融合）和 Reduction Splitting（RBLOCK < full）两项特性  
**Status:** Draft for review  
**依赖文档：**
- `docs/superpowers/specs/2026-04-14-tile-info-design.md`（TileInfo 数据模型）
- `docs/superpowers/specs/2026-04-14-vector-plan-data-model.md`（各阶段数据结构）
- `docs/superpowers/plans/2026-04-10-vector-plan-generation-final.md`（Phase 1 实施方案）

---

## 1. Problem

### 1.1 Cube Chain 融合缺失

当前 `vector-plan-generation` 把 `linalg.matmul` 作为硬边界切链，matmul 和其后的
epilogue（bias_add、relu、layernorm-affine 等）始终生成两个独立 kernel：

```
// 当前：两次 kernel launch
kernel_0: matmul(A, B) → C               ← Cube kernel
kernel_1: bias_add(C, bias) → relu → D   ← Vector kernel
```

matmul epilogue 的 pointwise 计算量远小于 matmul 本身，两次 kernel launch 的开销（
host 调度、global memory 读写 C）是不必要的。典型案例：QKV projection、FFN linear、
attention output projection。

### 1.2 Reduction Splitting 缺失

当前 vector plan 中 reduction tile 参数（`RBLOCK_j`）默认等于完整维长（Full），理由是
v1 目标网络的 reduction 维度能放进 UB。但以下场景会突破这个假设：

1. **大序列长度 LayerNorm**：`[B, S=8192, H=4096]`，H 维 4096×f16=8KB，可能超出 UB。
2. **Cube K 轴**：matmul 的 K 维必须分块（BK），BK 对应 L0A/L0B buffer 容量而不是 UB，
   且有硬件对齐约束（FP16 要求 BK 是 16 的倍数）。
3. **两者统一**：Cube BK 和 Vector RBLOCK_sub 在数据模型上是同一语义
   （`TileLevel=Inner, AxisRole=Reduction`），应使用统一接口。

---

## 2. Goals

- 在 Chain Analysis 中识别 **CubeChain**：matmul 作为驱动算子，向前吸收 epilogue
  consumer。
- 在 TilePlan Generation 中为 CubeChain 生成正确的 **2D block dispatch + BK tunable**
  参数。
- 为 VectorChain 新增 **Reduction Splitting** 支持：`RBLOCK_j` 从 `FixedTile/Full`
  变为可选的 `TunableTile/Inner`。
- 在 `SearchSpace` 中补充 **硬件约束**（对齐、上界），统一 cube BK 和 vector RBLOCK_sub
  的搜索空间表达。
- 保持 VectorChain Phase 1 的产物不变（已有测试不回归）。

## 3. Non-Goals

- 不在本设计中实现 Cube kernel 的多级 buffer pipeline（L1→L0 double buffer）。
- 不做混合 chain（同一 chain 内既有 cube op 又有 vector reduction）。
- 不做 reduction splitting 的 parallel reduction（split-reduce + combine pattern）。
- 不做动态 shape 下 BK 的运行时选择。

---

## 4. 核心统一性

在讨论具体设计之前，先确认数据模型层面的统一性：

```
TileLevel × AxisRole 的完整矩阵：

                  AxisRole=Parallel          AxisRole=Reduction
TileLevel=Outer   XBLOCK（vector 分核）       —
                  BM / BN（cube 2D 分核）
TileLevel=Inner   XBLOCK_SUB（vector UB 批）  BK（cube L0 K tile）
                                             RBLOCK_sub（vector reduction split）
TileLevel=Full    —                           RBLOCK（vector v1，不切）
```

**Cube BK 和 Vector RBLOCK_sub 落在同一格**：`TileLevel=Inner, AxisRole=Reduction`。
现有 `TileFieldSpec` 数据结构已经可以表达两者，只是参数的默认值和约束不同。

这意味着：
- `TileInfo` 数据模型**不需要新字段**即可表达两者。
- 需要扩展的只有 `SearchSpace`（约束）和 TilePlan 生成规则（默认行为）。

---

## 5. 数据模型扩展

### 5.1 SearchSpace 补充约束

```cpp
struct SearchSpace {
  bool enabled = false;
  SmallVector<int64_t> candidates;

  // 新增：硬件/buffer 对齐约束。
  // 所有 candidates 必须满足 value % alignment == 0。
  // vector RBLOCK_sub：无约束（alignment 留空）。
  // cube BK：alignment = 矩阵指令的 K 粒度（FP16 = 16，BF16 = 16，INT8 = 32）。
  std::optional<int64_t> alignment;

  // 新增：上界约束，由 buffer 容量决定。
  // 搜索时自动过滤超出上界的候选值。
  // vector RBLOCK_sub：可选（UB 大小限制）。
  // cube BK：必填（L0A/L0B buffer 容量，单位：element 数）。
  std::optional<ValueExpr> upperBound;
};
```

**候选值生成规则**（替代 v1 的空 candidates）：

```
candidates = { v | v = k × alignment, k ≥ 1, v ≤ eval(upperBound) }
             按 2 的幂次过滤（常见策略，可由 autotuner 覆盖）
```

### 5.2 ChainKind 与 CubeChainInfo

在 `ChainInfo` 中新增 `kind` 字段，并引入专用的 `CubeChainInfo`：

```cpp
enum class ChainKind : uint8_t {
  Vector, // 现有：linalg Vector op 组成
  Cube,   // 新增：以 linalg.matmul/BatchMatmul 为锚点
};

// CubeChain 专用结构，继承 ChainInfo 的公共字段
struct CubeChainInfo {
  ChainKind kind = ChainKind::Cube;

  linalg::LinalgOp matmul;       // 锚点
  linalg::LinalgOp epilogueRoot; // 最后一个 epilogue op，tile 的起点
                                 // 若无 epilogue，epilogueRoot == matmul

  // 所有成员（含 prologue + matmul + epilogue），程序序排列
  SmallVector<linalg::LinalgOp> members;

  SmallVector<AxisInfo> canonicalAxes; // M / N / K / Batch 轴
  SmallVector<Value>    boundaryIn;
  SmallVector<Value>    boundaryOut;
};
```

**注意**：`ChainKind` 只影响 TilePlan 生成规则；TileInfo 数据模型对两种 chain 统一，
不需要在 `TileInfo` 里存 `ChainKind`（PrepareForEmit 通过 `TileLevel`/`AxisRole`
即可区分 1D/2D dispatch）。

### 5.3 TileParam 无需修改

现有 `TileParam`（`name`, `ssa`, `defaultValue: OpFoldResult`, `axisIdx`,
`level: TileLevel`）已经足够表达：

| 参数 | kind | level | axisRole | 说明 |
|------|------|-------|----------|------|
| XBLOCK | TunableTile | Outer | Parallel | Vector 分核 |
| XBLOCK_SUB | TunableTile | Inner | Parallel | Vector UB 批 |
| RBLOCK（v1） | FixedTile | Full | Reduction | Vector 不切 reduction |
| RBLOCK_sub（新） | TunableTile | Inner | Reduction | Vector reduction split |
| TB_M | TunableTile | Outer | Parallel | Cube 分核 grid_y |
| Tb_M | TunableTile | Inner | Parallel | Cube UB M 批 |
| TB_N | TunableTile | Outer | Parallel | Cube 分核 grid_x |
| Tb_N | TunableTile | Inner | Parallel | Cube UB N 批 |
| t_K（新） | TunableTile | Inner | Reduction | Cube L0 K tile，必须切 |

---

## 6. Chain Analysis 扩展

### 6.1 VectorChain：不变

现有反向生长逻辑不变。唯一的变化：`linalg.matmul` 继续是硬边界（VectorChain 不吸收
cube op）。

### 6.2 CubeChain：识别、成员判定与生长规则

#### 6.2.1 matmul 锚点识别

```
matmul 锚点判定：
  op 是 linalg.matmul / linalg.batch_matmul，
  或 linalg.generic 满足：恰好一根 reduction 轴，indexing_maps 满足矩阵乘形式
  （lhs: (m,k)->(...,m,k)，rhs: (k,n)->(...,k,n)，result: (m,n)->(...,m,n)）
```

#### 6.2.2 Epilogue 吸收：以整个 VectorChain 为单位

**吸收粒度**：CubeChain 不逐 op 扫描 epilogue，而是以已形成的 **VectorChain 为整体单位**
吸收，与 AutoFuse 对齐（AutoFuse 先让 vector 融合收敛，Cube 再吸收已成型的 vector 子图）。

```
对每个 VectorChain vc（VectorChain 扫描已完成）：
  若 vc.root 是 matmul 直接 consumer（vc 消费 matmul 的输出 tensor）：
    // Fan-out 检查（对应 AutoFuse ReducePartitionMultipleCitations）：
    // matmul 输出只能被唯一一个 VectorChain 消费；
    // 若 matmul 输出同时被多个 VectorChain 消费 → 全部不吸收，epilogueRoot = matmul
    若 matmul 输出 tensor 被多于一个 VectorChain 引用 → 跳过，epilogueRoot = matmul
    检查 vc 整体是否满足 E1–E4（见下）：
      若满足 → 整体吸收 vc 为 epilogue，从 VectorChain 列表移除 vc
      若不满足 → 不吸收、不拆散 vc（保守策略）
  若 matmul 输出直接到 func.return（无 epilogue VectorChain）：
    epilogueRoot = matmul，无 epilogue 成员
```

**整体满足的条件 E1–E4**（对 vc 内每个 op C 均成立）：

```
E1. C 是 linalg Vector op，iterator_types 全为 "parallel"

E2. C 没有 in-place 写或 aliasing

E3. C 的 indexing_map 对 matmul 输出轴是 affine projective
    （identity / broadcast / rank-drop；不能有 permutation 导致 M/N 轴混合）
    **E3b**（AutoFuse §11.3 对应规则）：
    matmul 输出到 epilogueRoot 的路径上不能存在 tensor.expand_shape /
    linalg.broadcast 等显式 view node；vc 的 side-input 不能有 batch 轴 broadcast

E4. C 的所有输入要么来自 chain 内已有成员，要么是 chain 外的常量 / 纯标量
```

**不吸收的场景**（对应 AutoFuse §11.5）：
- vc 内全为纯 shape/view op（expand_dims / reshape / squeeze）→ 不吸收

#### 6.2.3 Prologue 成员判定（移植自 Inductor `scheduler.py:5128-5185`）

从 matmul 的 A 或 B 输入沿 def-use chain 向后（producer 方向）扫描，op P 可加入 prologue iff：

```
P1. P 是 linalg Vector op，iterator_types 全为 "parallel"
    （对应 Inductor: not node1.is_reduction()）

P2. P 没有 in-place 写或 aliasing
    （对应 Inductor: not node1.has_aliasing_or_mutation()）

P3. P 的输出只被 matmul 或 chain 内其他 prologue op 消费——单消费者约束
    （对应 Inductor scheduler.py:5158-5182 的最严格限制）：

    对 prologue 链上每个中间 op Q：
      Q 的每个输出结果只能被 prologue 链内的下一个 op 或 matmul 自身消费，
      不能被 chain 外任何 op 引用。

    若 P 的输出同时被 matmul 和其他 op 消费 → P 不能 fuse，必须物化。

P4. P 的 indexing_map 对 A/B 的输入轴是 affine projective
```

**启发式过滤**（移植自 Inductor `scheduler.py:4919-4976`）：

```
H1. 内存放大检查：
    fuse 后 prologue 引入的额外读取字节数 ≤ 原本写入字节数 × 1.1
    （Inductor 的 BYTES_THRESHOLD_MULTIPLIER = 1.1）
    超出则不 fuse（prologue 反而增加了 global memory 流量）

H2. 不允许 constant_pad_nd 等引入非对齐读取的 op 作为 prologue

H3. 若模板是低精度（f16/bf16），prologue 不能引入需要 f32 中间结果的 upcast
    （否则寄存器压力显著上升，不合算）
```

> **v1 实现策略**：H1–H3 属于性能启发式，v1 可先跳过仅保证正确性（只执行 P1–P4），
> 待有 profiling 数据后再加入启发式过滤。

#### 6.2.4 CubeChain 不做 collapse

M 轴和 N 轴始终保持独立，在 `collapseChains` 中对 `CubeChain` 直接返回恒等映射：

```cpp
if (isa<CubeChainInfo>(chain)) {
  cc.collapsedAxes = chain.canonicalAxes;
  cc.axisMap = identity;
  continue;
}
```

#### 6.2.3 Prologue 成员判定（移植自 Inductor `scheduler.py:5128-5185`）

从 matmul 的 A 或 B 输入沿 def-use chain 向后（producer 方向）扫描，op P 可加入 prologue iff：

```
P1. P 是 linalg Vector op，iterator_types 全为 "parallel"

P2. P 没有 in-place 写或 aliasing

P3. P 的输出只被 matmul 或 chain 内其他 prologue op 消费——单消费者约束：
    对 prologue 链上每个中间 op Q：
      Q 的每个输出结果只能被 prologue 链内的下一个 op 或 matmul 自身消费，
      不能被 chain 外任何 op 引用。

P4. P 的 indexing_map 对 A/B 的输入轴是 affine projective
```

**启发式过滤**（移植自 Inductor `scheduler.py:4919-4976`）：

```
H1. 内存放大检查：
    fuse 后 prologue 引入的额外读取字节数 ≤ 原本写入字节数 × 1.1
H2. 不允许 constant_pad_nd 等引入非对齐读取的 op 作为 prologue
H3. 若模板是低精度（f16/bf16），prologue 不能引入需要 f32 中间结果的 upcast
```

> **v1 实现策略**：H1–H3 属于性能启发式，v1 可先跳过仅保证正确性（只执行 P1–P4）。

#### 6.2.4 CubeChain 不做 collapse

M 轴和 N 轴始终保持独立，在 `collapseChains` 中对 `CubeChain` 直接返回恒等映射：

```cpp
if (isa<CubeChainInfo>(chain)) {
  cc.collapsedAxes = chain.canonicalAxes;
  cc.axisMap = identity;
  continue;
}
```

### 6.3 两类 Chain 的共存

**扫描顺序：先 VectorChain，再 CubeChain**，与 AutoFuse 对齐：

```
Step 1  VectorChain 扫描（matmul 保持硬边界，不吸收 cube op）
        → 产出 SmallVector<VectorChainInfo> vectorChains
          （所有非 matmul op 按现有反向生长逻辑归链）

Step 2  CubeChain 扫描
        → 遍历图中每个 matmul 锚点
        → 按 §6.2.2 尝试以整个 VectorChain 为单位吸收 epilogue
        → 按 §6.2.3 扫描 prologue（逐 op，P1–P4）
        → 被吸收的 VectorChain 从 vectorChains 列表移除
        → 产出 SmallVector<CubeChainInfo> cubeChains
```

**原因**：VectorChain 先把图跑稳（对应 AutoFuse DEFAULT 优先级），
CubeChain 再来吸收已成型的 VectorChain 块（对应 AutoFuse LOW 优先级）。
避免 CubeChain 贪婪抢走可能更适合 VectorChain 的 epilogue op。

**不拆散原则**：CubeChain 只能整体吸收 VectorChain，不能从中截取部分 op；
若整体不满足 E1–E4 条件，该 VectorChain 保持独立，CubeChain 的 epilogueRoot = matmul。

---

## 7. TilePlan Generation 扩展

### 7.1 VectorChain：Reduction Splitting（可选）

**当前行为**：所有 `full[j]` 参数的 `kind=FixedTile, level=Full`，不参与搜索。

**新增行为**（opt-in，由 pass option 控制）：

```cpp
Option<"enableReductionSplit", "enable-reduction-split", "bool", "false",
       "Whether to make RBLOCK tunable (Inner) instead of fixed full">
```

当 `enableReductionSplit=true` 时，为每根 Full 轴生成：

```
full[j] = { name="RBLOCK_j", kind=TunableTile, level=Inner,
            defaultValue=min(dim_size, UB_CAPACITY / elem_size),
            search={ enabled=true, candidates=[...2的幂次...],
                     upperBound=UB_CAPACITY/elem_size } }
```

**Epilogue 数量上限**（对应 AutoFuse `max_reduce_can_fuse_elementwise_nums`）：

当 `enableReductionSplit=true` 时，VectorChain 内 reduction 之后的 epilogue 成员数
（即消费 reduction 输出的 pointwise op 数量）不应无限扩张。过长的 epilogue 链会让
accumulator 的 combine 阶段变复杂，schedule 模板也可能再次拆分。

```cpp
Option<"maxReduceEpilogueOps", "max-reduce-epilogue-ops", "int32_t", "3",
       "Max pointwise ops after a reduction when enableReductionSplit=true">
```

超出上限时，在 Chain Analysis 阶段切断 epilogue 生长（保守回退到 FixedTile/Full），
而不是在 Realization 时失败。

**注意**：reduction splitting 需要在 Task 5 Realization 里额外处理 reduction 的
累加器初始化与 combine（见 §8.2）。

### 7.2 CubeChain：BM / BN / BK 参数生成

**识别轴**：

```
M 轴 = matmul lhs 的第 -2 维（或 batch_matmul 的第 -2 维）
N 轴 = matmul rhs 的第 -1 维
K 轴 = matmul lhs 的第 -1 维（= rhs 的第 -2 维）
Batch 轴 = batch_matmul 的前缀维（若有）
```

**生成规则**（对应 examples 中实测的 5 参数结构：TB_M/TB_N/Tb_M/Tb_N/t_K）：

```
// Batch 轴（若有）：与 VectorChain 相同，两级 split
batch[i] = { name="XBLOCK_i"(Outer), name="XBLOCK_SUB_i"(Inner) }

// M 轴：两级，与 VectorChain 的 tileable[0] 结构对齐
tileable[M] = {
  { name="TB_M", kind=TunableTile, level=Outer,   // 分核，驱动 grid_y
    defaultValue=128,
    search={ enabled=true, candidates=[64,128,256], alignment=HW_M_ALIGN } },
  { name="Tb_M", kind=TunableTile, level=Inner,   // UB 内 M 批次
    defaultValue=64,
    search={ enabled=true, candidates=[32,64,128], alignment=HW_M_ALIGN } }
}

// N 轴：两级，结构同 M 轴
tileable[N] = {
  { name="TB_N", kind=TunableTile, level=Outer,   // 分核，驱动 grid_x
    defaultValue=128,
    search={ enabled=true, candidates=[64,128,256], alignment=HW_N_ALIGN } },
  { name="Tb_N", kind=TunableTile, level=Inner,   // UB 内 N 批次
    defaultValue=128,
    search={ enabled=true, candidates=[64,128,256], alignment=HW_N_ALIGN } }
}

// K 轴：一级 Inner tunable（不是 Full）
full[K] = { name="t_K", kind=TunableTile, level=Inner,
            defaultValue=64,                        // 对应 L0A/L0B 容量
            search={ enabled=true, candidates=[64,128,256],
                     alignment=HW_K_ALIGN,
                     upperBound=L0A_CAPACITY/elem_size } }
```

**blockDimExprs**：CubeChain 产生 2D grid，表达为乘积与 tiling_space.json 对齐：

```cpp
// 对应 tiling_space.json: "block_dim_expr": "ceil(M/TB_M) * ceil(N/TB_N)"
// blockDimExprs 存两个元素，序列化时可合并为乘积或保留分离形式
blockDimExprs[0] = ceildiv(M_extent, TB_M);  // grid_y
blockDimExprs[1] = ceildiv(N_extent, TB_N);  // grid_x
```

**硬件常量来源**（从 target attr 或 pass option 读取）：

```cpp
Option<"hwMatmulMAlign", "hw-matmul-m-align", "int64_t", "16", "">
Option<"hwMatmulNAlign", "hw-matmul-n-align", "int64_t", "16", "">
Option<"hwMatmulKAlign", "hw-matmul-k-align", "int64_t", "16", "">
Option<"l0aCapacityBytes", "l0a-capacity-bytes", "int64_t", "65536", "">
```

---

## 8. Realization 扩展

### 8.1 CubeChain：反向 fusion（与 VectorChain 方向一致）

CubeChain 的 realization 与 VectorChain **方向相同**：以 `epilogueRoot` 为 tile 起点，
向后（producer 方向）fuse，matmul 作为 epilogue 的 producer 被拉入。这与 examples 中
transform 脚本的实际做法一致（先 tile leaky_relu，再 fuse_into_containing_op matmul）。

**三轮 tile + fuse 步骤**：

```
// 轮 1：TB 层 [TB_M, TB_N]，产生分核 loop
tile epilogueRoot [TB_M, TB_N]
fuse epilogue 中间 ops 进 for_TB_N
fuse matmul 进 for_TB_N          ← matmul 作为 producer 被反向 fuse
标注 for_TB_M / for_TB_N: ascendc.parallel = true
标注 for_TB_N: prologue = "lhs:GM->A1,rhs:GM->B1,bias:GM->VECIN"
              epilogue = "result:VECOUT->GM"

// 轮 2：Tb 层 [Tb_M, Tb_N]，在 TB 结果上再 tile
tile tiled_epilogueRoot [Tb_M, Tb_N]
fuse epilogue 中间 ops 进 for_Tb_N
fuse matmul_TB 进 for_Tb_N

// 轮 3：K 轴 [0, 0, t_K]，单独 tile matmul
tile matmul_Tb [0, 0, t_K]
标注 for_K: prologue = "lhs:A1->A2,rhs:B1->B2"
            epilogue = "acc:CO1->VECIN"
标注 matmul_final: ascendc.unit = "AiCore.Cube"
标注 epilogue ops:  ascendc.unit = "AiCore.Vector"

// 若有 prologue compute（如 dequant）：
// prologue op 满足 P1-P4 + 单消费者约束后，作为 matmul 的 B-input producer
// 被 fuse 进 for_K loop（自然落在 K-slice 粒度）
```

**目标产出 IR 结构**（与 examples/matmul-add-leakyrelu 一致）：

```
scf.for %TB_M {ascendc.parallel}
  scf.for %TB_N {ascendc.parallel, prologue=..., epilogue=...}
    scf.for %Tb_M
      scf.for %Tb_N
        scf.for %t_K {prologue=A1->A2/B1->B2, epilogue=CO1->VECIN}
          linalg.matmul  [AiCore.Cube]
          // prologue ops（如 dequant）在此处 fuse
        linalg.bias_add  [AiCore.Vector]   // epilogue 在 Tb 层
        linalg.relu      [AiCore.Vector]
```

**fusionControlFn**：只 fuse `chain.members` 里的 op，单消费者约束已在 Chain
Analysis 阶段（P3）保证，Realization 层不再重复检查。

### 8.2 VectorChain Reduction Splitting：accumulator init 与 combine

当 `enableReductionSplit=true` 时，reduction 不再是单次 full loop，变成：

```
// split-reduce 产出 IR 结构：
tensor.empty → acc_init (0)
scf.for %xb = 0 to BS step XBLOCK {    // ascendc.parallel
  scf.for %xs = 0 to XBLOCK step XBLOCK_SUB {
    partial_acc = tensor.empty
    linalg.fill(0, partial_acc)
    scf.for %rb = 0 to H step RBLOCK_sub { // reduction split
      partial_acc += x[xs, rb:rb+RBLOCK_sub]
    }
    // partial_acc 是当前 XBLOCK_SUB 块的完整 reduction 结果
    affine_normalize(partial_acc) → y[xs]
  }
}
```

**注意**：本设计 Non-Goals 中明确**不做 parallel reduction**（split-k 然后 combine）。
reduction splitting 这里只做**串行分块**：每个 XBLOCK_SUB 内完整跑完 H 维，不分给
多个 core。这与 Cube BK 的语义相同（不是 split-k）。

---

## 9. TileInfo 输出

### 9.1 VectorChain with Reduction Split 示例

```mlir
// LayerNorm [16, 4096]，enableReductionSplit=true
tiling.infos = [{
  kernel = "chain0_plan0",
  axes = [
    {axis=0, name="BS", role="parallel",
     extent={op="shape_dim", arg=0, dim=0}},
    {axis=1, name="H",  role="reduction",
     extent={op="shape_dim", arg=0, dim=1}}
  ],
  fields = [
    {id="tile.xblock",    abi_name="XBLOCK",     kind="tunable",
     level="outer", axis=0, abi_index=0,
     default={op="const", value=256},
     search={candidates=[64,128,256]}},
    {id="tile.xblock_sub",abi_name="XBLOCK_SUB", kind="tunable",
     level="inner", axis=0, abi_index=1,
     default={op="const", value=64},
     search={candidates=[32,64]}},
    {id="tile.rblock0",   abi_name="RBLOCK_0",   kind="tunable",
     level="inner", axis=1, abi_index=2,           // ← Inner，不再是 Full
     default={op="const", value=256},
     search={candidates=[128,256,512],
             upper_bound={op="const", value=1024}}},
    ...shape fields...
  ],
  block_dim=[{op="ceildiv",
              lhs={op="shape_dim",arg=0,dim=0},
              rhs={op="field_ref",id="tile.xblock"}}]
}]
```

### 9.2 CubeChain 示例

```mlir
// matmul [M=1024, K=4096, N=2048] + bias_add + relu
tiling.infos = [{
  kernel = "chain0_plan0",
  axes = [
    {axis=0, name="M", role="parallel",  extent={op="shape_dim",arg=0,dim=0}},
    {axis=1, name="N", role="parallel",  extent={op="shape_dim",arg=1,dim=1}},
    {axis=2, name="K", role="reduction", extent={op="shape_dim",arg=0,dim=1}}
  ],
  fields = [
    {id="tile.bm", abi_name="BM", kind="tunable",
     level="outer", axis=0, abi_index=0,
     default={op="const", value=128},
     search={candidates=[64,128,256], alignment=16}},
    {id="tile.bn", abi_name="BN", kind="tunable",
     level="outer", axis=1, abi_index=1,
     default={op="const", value=128},
     search={candidates=[64,128,256], alignment=16}},
    {id="tile.bk", abi_name="BK", kind="tunable",
     level="inner", axis=2, abi_index=2,    // ← reduction 轴，Inner
     default={op="const", value=256},
     search={candidates=[128,256,512],
             alignment=16,
             upper_bound={op="const", value=4096}}},  // L0 容量
    ...shape fields...
  ],
  block_dim=[
    {op="ceildiv", lhs={op="shape_dim",arg=0,dim=0},
                   rhs={op="field_ref",id="tile.bm"}},  // grid_y
    {op="ceildiv", lhs={op="shape_dim",arg=1,dim=1},
                   rhs={op="field_ref",id="tile.bn"}}   // grid_x
  ]
}]
```

---

## 10. 与现有 TileInfo 设计的关系

本设计对 `docs/superpowers/specs/2026-04-14-tile-info-design.md` 的修改：

| 修改项 | 内容 |
|--------|------|
| `SearchSpace` | 新增 `alignment: optional<int64_t>` 和 `upperBound: optional<ValueExpr>` |
| `TileLevel` | 语义不变；明确 `Inner` 同时适用于 Parallel 轴（XBLOCK_SUB）和 Reduction 轴（BK / RBLOCK_sub） |
| `TileFieldKind` | 不变；`FixedTile` 对应 RBLOCK v1（Full），`TunableTile` 对应 BK 和 RBLOCK_sub |
| `TileInfo.blockDimExprs` | 不变（已是 `SmallVector<..., 2>`，CubeChain 填两个元素） |
| Invariant 7 | 补充说明：CubeChain 的 `blockDimExprs` 必须有两个元素（grid_y, grid_x） |

对 `docs/superpowers/specs/2026-04-14-vector-plan-data-model.md` 的修改：

| 修改项 | 内容 |
|--------|------|
| `ChainInfo` | 新增 `ChainKind kind` 字段 |
| `TilePlan` 生成规则 | 新增 CubeChain 分支（BM/BN/BK），VectorChain 新增 reduction split opt-in |
| Task 5 续接 | CubeChain 走 `fuseProducerOfSlice` 前向融合路径 |

---

## 11. 迁移与兼容性

| 阶段 | 兼容性 |
|------|--------|
| Phase 1 VectorChain（现有测试） | 完全不变；`enableReductionSplit` 默认 false |
| Phase 1 TileInfo 格式 | 完全兼容；SearchSpace 新字段对 v1 vector 为空 |
| `tiling.tiles` / `tiling.shapes` | 不变；compat 投影逻辑不涉及 ChainKind |
| AutoTuner | 读 `search.candidates` 逻辑不变；新增 `alignment` 过滤步骤 |
| PrepareForEmit | 通过 `TileLevel=Outer` 的个数判断 1D/2D dispatch，不需要读 ChainKind |

---

## 12. Deferred

1. **Split-K parallel reduction**：BK split 后多个 core 各自计算 partial sum，最后 combine——这是更激进的优化，留到有性能压力时。
2. **动态 BK**：运行时根据实际 K 大小选 BK——留到动态 shape 支持整包时。
3. **Cube + Vector 混合 chain**：matmul 输出经过 reduce 后再接 pointwise（如 attention score sum + normalize）——需要额外的 chain 分析，留到 v2。
4. **多 stage pipeline**（L1→L0 double buffer）：Cube kernel 的 buffer 流水线优化——由 Phase 2 codegen 负责，不在本 pass 处理。


```python
for i in [0, A]:
    sum[i][k] = 0
    for j in [0, B]:
        for k in [0, C]:
            sum[i][k] += input[i][j][k]

for i in [0, A]:
    sum[i][k] = 0
    for j0 in [0, B0]:
        
        for j1 in [0, B1]:
            for k in [0, C]:
                sum [i][k] += input[i][j0 * B1 + j1][k]

```



