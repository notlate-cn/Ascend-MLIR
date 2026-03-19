# AscendC Buffer Placement 新方案设计

## 背景：当前方案的问题

当前 `AscendCBufferPlacementPass` 的工作方式：

1. 解析 for 循环上的字符串 annotation（`ascendc.prologue = "lhs:GM->A1,rhs:GM->B1"`）
2. 通过 `findRoleSubview` 启发式扫描 for 循环 body，用 role 字符串（`"lhs"`/`"rhs"`）关联到实际的 SSA Value
3. 创建 alloc + 插入 memref.copy

核心问题：**annotation 里的 role 字符串与 Op 的 SSA Value 没有直接关联**，需要靠 arg 序号、循环嵌套深度等启发式手段来匹配，遇到非标准结构容易误匹配。

---

## 新方案总览

### 设计原则

- **compute op 上只标单元类型**（`ascendc.unit`），其余由 pass 自动推导
- **placement 推导**：基于数据流分析，从 compute op 的固定规则 + 操作数使用关系推导
- **搬运路径**：硬件相关的固定路径表，写死在 pass 里
- **搬运时机**：由操作数的支配关系自动决定，不需要循环层级标签
- **人工干预**：通过 memref 类型上的 memory_space 或手写 memref.copy 覆盖

---

## 两类信息的归属

| 信息 | 归属层级 | 表达方式 |
|------|---------|---------|
| 搬运什么（src/dst） | Op 级，就是操作数本身 | SSA Value 直接引用 |
| 搬入哪里（placement） | Op 级，由硬件规则决定 | 自动推导，或 memref memory_space 覆盖 |
| 搬运时机（插入位置） | 调度级，由支配关系决定 | 自动推导，或手写 memref.copy 覆盖 |
| 硬件单元类型 | Op 级 | `ascendc.unit` annotation |

---

## Annotation 最小集

用户（或 transform 脚本）只需标注：

```mlir
linalg.matmul {ascendc.unit = "AiCore.Cube"}

linalg.elementwise {ascendc.unit = "AiCore.Vector"}

linalg.generic {
  ascendc.unit = "AiCore.Vector",
  library_call = "gather_by_index"
}
```

不需要 role 字符串，不需要循环层级标签，不需要显式描述搬运方向。

---

## Placement 自动推导

### Compute Op 固定规则

| ascendc.unit | 操作数 | 目标 placement |
|-------------|--------|---------------|
| AiCore.Cube | ins[0] (lhs) | A2 |
| AiCore.Cube | ins[1] (rhs) | B2 |
| AiCore.Cube | outs (acc) | CO1 |
| AiCore.Vector | ins[*] | VECIN 或 VECCALC（见下） |
| AiCore.Vector | outs | VECOUT 或 VECCALC（见下） |

### 中间结果规则（数据流传播）

Vector op 的 ins/outs placement 通过数据流分析决定：

```
算法：
1. 所有 compute op 的 outs 初始化为"待定"
2. 如果 outs 被另一个 compute op 的 ins 直接使用 → VECCALC（中间结果，零搬运）
3. 如果 outs 最终写回 func arg（GM）→ VECOUT
4. ins 的 placement：
   - 来自 GM（func arg / subview）→ VECIN（需要 GM→VECIN 搬运）
   - 来自另一个 compute op 的 outs → 继承该 outs 的 placement（零搬运）
   - 来自 Cube op 的 outs（CO1）→ VECIN，通过 fixpipe 搬运
```

### 示例：matmul + add + relu

```
func arg %bias (GM)
    ↓ subview → ins[1] of add       需要 GM→VECIN 搬运

matmul outs → CO1
    ↓ fixpipe → ins[0] of add       CO1→VECIN，已搬运

add outs → VECCALC                  被 relu 的 ins[0] 使用，中间结果，零搬运

relu outs → VECOUT                  写回 func arg，需要 VECOUT→GM 搬运
```

---

## 搬运路径表

硬件相关，固定写死在 pass 里：

| src | dst | 路径 | 搬运 Op |
|-----|-----|------|---------|
| GM | A2 | GM→A1→A2 | data_copy_l2 + data_copy_l0 |
| GM | B2 | GM→B1→B2 | data_copy_l2 + data_copy_l0 |
| GM | VECIN | GM→VECIN | data_copy_l2 |
| CO1 | VECIN | CO1→VECIN | fixpipe |
| VECOUT | GM | VECOUT→GM | data_copy_l2 |
| A1 | A2 | A1→A2 | data_copy_l0 |
| B1 | B2 | B1→B2 | data_copy_l0 |

多级路径（如 GM→A1→A2）会生成两个中间 alloc 和两次 copy。

---

## 搬运时机：支配关系自动决定

Pass 对每个需要搬运的操作数，将 alloc + copy 插入到：

> **操作数 def 点** 和 **compute op 使用点** 之间，尽量提升到最外层合法位置

规则：
- `%subview_a` 定义在 Tb 循环内 → A1 alloc 最早插在 Tb 循环入口
- `%k_subview_a` 定义在 K 循环内 → A2 alloc 最早插在 K 循环入口
- 如果操作数定义在循环外（func arg）→ alloc 可提升到函数入口

这样不需要任何循环层级标签，时机完全由 SSA 支配关系决定。

---

## 人工干预

### 覆盖 Placement

直接在 memref 类型上写 memory_space，pass 跳过该操作数的推导：

```mlir
// 强制 ins[0] 使用 VECIN（memory_space=9）而不是自动推导的 VECCALC
linalg.elementwise {ascendc.unit = "AiCore.Vector"}
  ins(%add_result : memref<?xf32, 9>)   // 9 = VECIN
  outs(%relu_result : memref<?xf32, 10>) // 10 = VECOUT
```

### 覆盖搬运时机

手写 memref.copy 到目标位置，pass 检测到已有 copy 就不再插入：

```mlir
// 把 bias 搬运提升到 TB 循环外（比自动推导更大粒度的预取）
scf.for %m_outer = ... {
  %bias_ub = memref.alloc() : memref<?xf32, 9>  // VECIN
  memref.copy %bias, %bias_ub                    // 手写，pass 不重复插入
  scf.for %k = ... {
    linalg.elementwise ...
  }
}
```

### 优先级

```
手写 memory_space on memref type  >  手写 memref.copy  >  自动推导
```

Pass 检查顺序：
1. 操作数已有 memory_space → 跳过 placement 推导
2. 已有 copy 到目标位置 → 跳过 copy 插入
3. 否则 → 自动推导 + 自动插入

---

## 与当前方案对比

| | 当前方案 | 新方案 |
|--|---------|-------|
| compute op annotation | `ascendc.unit` | `ascendc.unit`（不变） |
| for 循环 annotation | `ascendc.prologue = "lhs:GM->A1,..."` | 不需要 |
| 操作数关联方式 | role 字符串 + `findRoleSubview` 启发式扫描 | SSA Value 直接引用 |
| 中间结果 placement | 需要手动标注 | 数据流自动推导 |
| 搬运时机 | 手写在 annotation 里 | 支配关系自动决定 |
| 人工干预 | 修改 annotation 字符串 | memref memory_space 或手写 copy |
| 多场景通用性 | 依赖 matmul 固定结构 | 数据流分析，与循环结构无关 |

---

## 新 Pass 实现步骤

1. **收集 compute ops**：walk 找所有带 `ascendc.unit` 的 op
2. **数据流分析**：对每个 op 的 operands，判断来源（GM/中间结果/CO1）
3. **查路径表**：根据 src placement → dst placement 确定搬运路径
4. **确定插入位置**：找操作数 def 点和 use 点之间最外层合法插入点
5. **生成 alloc + copy**：按路径表插入中间 alloc 和 memref.copy（已有则跳过）
6. **更新操作数**：将 compute op 的 operand 从原 GM subview 改为新 alloc

`AscendCBufferPlacementPass` 中可删除：`findRoleSubview`、`insertCopiesForLoop`、`LoopAnnotation` 解析，约 400 行。
