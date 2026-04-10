先回顾一下 Phase 1 输出的 IR 状态，再制定 Phase 2 的目标和计划。

---

## Phase 1 输出状态回顾

```
func 参数：memref（无 memory_space，GM 语义）
片上 alloc：memref（带 #ascendc.space<*> via integer memory_space）
计算 op：linalg.matmul / linalg.elementwise（保持原样）
搬运 op：memref.copy（带 memory_space 组合表达搬运语义）
```

---

## Phase 2 总目标

```
输入：Phase 1 输出的 IR（memref + memory_space + linalg + memref.copy）
输出：纯 ascendc dialect IR，可直接 codegen 为 AscendC C++ kernel
```

整个 Phase 2 按职责拆分为 4 个独立 Pass，顺序执行。

---

## Pass 2-A：MemrefToAscendCTensorPass（类型系统转换）

**职责**：把所有 `memref` 替换为 ascendc 的类型系统。

**输入 → 输出**：

```mlir
// 输入
%alloc_2 = memref.alloc(%6, %dim_1) : memref<?x?xf32, 1 : i32>   // A1
%arg0 : memref<?x?xf32, strided<[?, ?], offset: ?>>               // GM

// 输出
%tbuf_A1  = ascendc.tbuf : !ascendc.tbuf<a1>
%tensor_A1 = ascendc.tbuf.get_tensor %tbuf_A1 : !ascendc.local_tensor<f32>
%tensor_GM = ascendc.global_tensor : !ascendc.global_tensor<f32>
```

**转换规则**：

| 输入 memref memory_space | 输出 ascendc 类型 | 分配 op |
|---|---|---|
| 无（func 参数） | `GlobalTensor` | `ascendc.global_tensor` |
| `1/2/3/4/7`（A1/A2/B1/B2/CO1） | `LocalTensor` via `TBuf` | `ascendc.tbuf` + `tbuf.get_tensor` |
| `9/10/11`（VECIN/VECOUT/VECCALC） | `LocalTensor` via `TQue` 或 `TBuf` | `ascendc.queue` 或 `ascendc.tbuf` |

`memref.subview` → `ascendc.local_tensor.subindex` 或直接消除（CO1 的 subview 特殊处理）。

`memref.dealloc` → `ascendc.tbuf` 生命周期由 TPipe 管理，直接删除。

**依赖**：需要插入 `ascendc.pipe`（TPipe）作为资源管理器，在 func 入口创建，出口销毁。

---

## Pass 2-B：CopyToAscendCDataMovePass（搬运 op 替换）

**职责**：把 `memref.copy` 替换为对应硬件通道的 ascendc 搬运 op，并插入 TQue 同步。

**转换规则**（根据 src/dst 的 memory_space 组合判断）：

| src memory_space | dst memory_space | 替换为 | 硬件通道 |
|---|---|---|---|
| GM（无） | `1/3`（A1/B1） | `ascendc.data_copy_l2` | MTE2 |
| GM（无） | `9`（VECIN bias） | `ascendc.data_copy_l2` | MTE2 |
| `1/3`（A1/B1） | `2/4`（A2/B2） | `ascendc.data_copy_l0` | MTE1 |
| `7`（CO1） | `9`（VECIN） | `ascendc.fixpipe` | FIX |
| `10`（VECOUT） | GM（无） | `ascendc.data_copy_l2` | MTE3 |

每个 `data_copy_l2` / `data_copy_l0` 前后需要插入 `ascendc.pipe_barrier`，保证流水线顺序正确：

```mlir
// MTE2 搬运前后的同步示例
ascendc.pipe_barrier {pipe = #ascendc.pipe<mte2>}
ascendc.data_copy_l2 %dst, %src, %count
ascendc.pipe_barrier {pipe = #ascendc.pipe<mte2>}
```

**TQue 同步模型**（VECIN/VECOUT 使用 TQue 而非 TBuf）：

```mlir
// VECIN：生产者（DataCopy）→ EnQue → 消费者（Vector op）→ DeQue
ascendc.queue.enque %vecin_queue, %tensor
// ... vector op ...
ascendc.queue.deque %vecin_queue
```

---

## Pass 2-C：LinalgToAscendCComputePass（计算 op lowering）

分两部分。

### Part 1：linalg.matmul → Matmul 状态机

```mlir
// 输入
linalg.matmul ins(%A2, %B2) outs(%CO1)

// 输出（matmul 状态机展开）
%mm = ascendc.matmul.init ... : !ascendc.matmul<...>
ascendc.matmul.set_tensor_a %mm, %A2
ascendc.matmul.set_tensor_b %mm, %B2
ascendc.matmul.iterate %mm      // 触发一次 CUBE 计算
ascendc.matmul.get_tensor_c %mm, %CO1
ascendc.matmul.end %mm
```

`matmul.init` 的参数需要从 A2/B2/CO1 的类型中推导：`CubeFormat`（nZ/ND）、数据类型、tile size。

for_K 的多次迭代对应多次 `matmul.iterate`，`matmul.init` 和 `matmul.end` 提升到 for_K 之外。

### Part 2：linalg.elementwise → ascendc Vector op

```mlir
// add
linalg.elementwise kind=add ins(%VECIN, %bias) outs(%VECCALC)
→ ascendc.add %VECCALC, %VECIN, %bias : !ascendc.local_tensor<f32>

// max（relu）
linalg.elementwise kind=max_signed ins(%VECCALC, %zero) outs(%VECOUT)
→ ascendc.max %VECOUT, %VECCALC, %zero : !ascendc.local_tensor<f32>
```

零标量可用 `arith.constant` + broadcast 替代全量零 buffer（消除 Phase 1 残留的 `%alloc` 问题）。

---

## Pass 2-D：VectorizationPass（向量化）

**职责**：把标量/tile 级的 ascendc Vector op 向量化，生成最终的向量化指令序列。

这一步对应 AscendC 的 `RepeatTimes` / `RepeatStride` 参数展开，把逻辑上的 `[Tb_M × Tb_N]` tile 操作映射到硬件的向量宽度（256B / 512B 对齐的计算单元）。

```mlir
// 向量化前
ascendc.add %dst, %src0, %src1  // tile 级，[Tb_M × Tb_N] 个元素

// 向量化后
ascendc.add %dst, %src0, %src1 {repeat_times = 4, dst_rep_stride = 8, ...}
// 展开为硬件 BLK 宽度对齐的重复计算
```

具体参数来源于 `Tb_M × Tb_N`（tile size）和硬件向量宽度（910B 的 FP32 向量宽度为 64 个元素 / 256B）。

**向量化的实现策略**：

1. 分析每个 Vector op 的 operand shape（来自 `LocalTensor` 的 shape 参数）
2. 计算 `repeat_times = total_elements / vector_width`
3. 设置 `RepeatParams`（`dstRepStride`、`srcRepStride` 等）
4. 生成带 repeat 参数的最终 Vector op

---

## 四个 Pass 的执行顺序和依赖关系

```
Phase 1 输出
    ↓
Pass 2-A: MemrefToAscendCTensorPass
    memref → GlobalTensor / LocalTensor / TBuf / TQue
    ↓
Pass 2-B: CopyToAscendCDataMovePass
    memref.copy → DataCopyL2 / DataCopyL0 / Fixpipe + PipeBarrier
    ↓
Pass 2-C: LinalgToAscendCComputePass
    linalg.matmul → matmul 状态机
    linalg.elementwise → ascendc.add / ascendc.max
    ↓
Pass 2-D: VectorizationPass
    tile 级 Vector op → 带 RepeatParams 的硬件指令
    ↓
纯 ascendc dialect IR → Codegen → AscendC C++ kernel
```

**Pass 之间的关键接口约定**：

| 接口 | 由谁产生 | 由谁消费 |
|---|---|---|
| `memref` + `memory_space` integer | Phase 1 | Pass 2-A |
| `ascendc.LocalTensor` / `GlobalTensor` | Pass 2-A | Pass 2-B / 2-C |
| `memref.copy`（带 memory_space 组合） | Phase 1 | Pass 2-B |
| `ascendc.data_copy_*` / `ascendc.fixpipe` | Pass 2-B | Codegen |
| `linalg.matmul` / `linalg.elementwise` | Phase 1 | Pass 2-C |
| tile 级 `ascendc.add` / `ascendc.max` | Pass 2-C | Pass 2-D |
| 带 RepeatParams 的 Vector op | Pass 2-D | Codegen |

---

## 工作量估计

| Pass | 复杂度 | 主要难点 |
|---|---|---|
| 2-A 类型转换 | 中 | subview → LocalTensor 的 offset 语义转换；TPipe 生命周期管理 |
| 2-B 搬运替换 | 中 | PipeBarrier 插入时机；TQue EnQue/DeQue 配对 |
| 2-C matmul 展开 | 高 | matmul 状态机参数推导；for_K 的 iterate 循环结构变换 |
| 2-C Vector lowering | 低 | 一对一替换，规则简单 |
| 2-D 向量化 | 高 | RepeatParams 计算；动态 shape 的向量化策略 |

建议按 **2-C Vector → 2-B → 2-A → 2-C matmul → 2-D** 的顺序开发，先做简单的，建立测试基础设施后再攻难点。