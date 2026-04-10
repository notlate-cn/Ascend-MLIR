## 总体评估

仔细阅读了端到端测试用例产生的结果：examples/matmul-add-relu-sum/output_step3_buffer_placement.mlir，
发现了几个严重的语义错误。

---

## 问题 1：搬运粒度完全错误（严重）

**第 22–35 行，for_TB_N 的 prologue**：

```mlir
// A1：把整个 %arg0（全量 A 矩阵）搬到 alloc_5
%alloc_5 = memref.alloc(%dim_2, %dim_4) : memref<?x?xf32, 1 : i32>  // A1
memref.copy %arg0, %alloc_5 ...

// B1：把整个 %arg1（全量 B 矩阵）搬到 alloc_10
%alloc_10 = memref.alloc(%dim_7, %dim_9) : memref<?x?xf32, 3 : i32>  // B1
memref.copy %arg1, %alloc_10 ...

// VECIN：把整个 %arg2（全量 bias 矩阵）搬到 alloc_15
%alloc_15 = memref.alloc(%dim_12, %dim_14) : memref<?x?xf32, 9 : i32>  // VECIN
memref.copy %arg2, %alloc_15 ...
```

`%dim_2, %dim_4` 来自 `memref.dim %arg0`，即 A 矩阵的**完整尺寸**，而不是当前 TB tile 的尺寸 `[%6, %dim_16]`（`%6` 是 `affine.min` 计算的 TB_M 大小）。

正确的行为：A1 buffer 只需要存放当前 AiCore 负责的那块 `[TB_M × K]`，而不是全量 `[M × K]`。这是设计上最核心的错误——**L1 片上存储根本放不下整个矩阵**。

根本原因在 `findSourceMemref` 里：它找到的 `srcMemref` 是函数参数 `%arg0`（全量），然后 `createLocalMemref` 用 `srcType.getShape()` 分配了和 `%arg0` 一样大的 alloc。

**修复方向**：prologue 搬运的 alloc 大小应该和**当前循环层的 subview 大小一致**，而不是 source 的完整大小。具体来说，`createLocalMemref` 的 shape 应该来自当前 forOp body 里对应 subview 的 sizes，而不是 `arg0` 的全量尺寸。

---

## 问题 2：for_K prologue 搬运的 src 是全量 A1/B1，而不是当前 K 切片（严重）

**第 60–67 行，for_K 的 prologue**：

```mlir
// A2：把整个 alloc_5（全量 A1）搬到 alloc_34
%alloc_34 = memref.alloc(%dim_31, %dim_33) : memref<?x?xf32, 2 : i32>  // A2
memref.copy %alloc_5, %alloc_34 ...

// B2：把整个 alloc_10（全量 B1）搬到 alloc_39
%alloc_39 = memref.alloc(%dim_36, %dim_38) : memref<?x?xf32, 4 : i32>  // B2
memref.copy %alloc_10, %alloc_39 ...
```

同样的问题：A2 应该只存放 `[Tb_M × t_K]`，而不是整个 `alloc_5`。`dim_31/dim_33` 来自 `memref.dim %alloc_5`，即 A1 的完整尺寸。

L0A 的容量只有几十 KB，专门为单次 CUBE 指令设计，根本容不下整个 A1 的内容。

---

## 问题 3：CO1 的 epilogue（CO1→VECIN）没有出现

**for_K epilogue** 应该在第 79 行之前插入 CO1→VECIN 的 fixpipe/copy，但实际输出里：

```mlir
// for_K 的唯一 epilogue 是：
%alloc_47 = memref.alloc(%dim_44, %dim_46) : memref<?x?xf32, 9 : i32>  // VECIN
memref.copy %alloc_25, %alloc_47 : memref<?x?xf32, 7 : i32> to memref<?x?xf32, 9 : i32>
```

虽然 `memref.copy` 从 `7 : i32`（CO1）到 `9 : i32`（VECIN）看起来语义是对的，但问题是这个 copy 在 `for_K` 循环体**内部**，而不是 epilogue（for_K 结束后）。

CO1→VECIN 的 fixpipe 必须在 K 轴**所有迭代完成后**才能执行（因为 CO1 是 K 轴的累加结果），放在循环体内部意味着每次 K 迭代都触发一次，语义完全错误。

---

## 问题 4：VECOUT→GM 的 epilogue 写回目标错误（严重）

**第 90–92 行，for_TB_N epilogue**：

```mlir
memref.copy %alloc_21, %subview_18 : memref<?x?xf32, 10 : i32> to memref<?x?xf32, ...>
// 然后紧接着：
memref.copy %alloc_5, %arg3 : memref<?x?xf32, 1 : i32> to memref<?x?xf32, ...>
```

VECOUT（`alloc_21`, `10 : i32`) 写回的目标是 `%subview_18`，这是 `%arg3`（输出矩阵 C）的一个 subview，这部分是对的。但紧接着又把 `alloc_5`（A1，`1 : i32`）写回了 `%arg3`——把 L1 里的 A 矩阵内容写回到输出矩阵 C，这是完全错误的多余搬运。这条多余的 copy 来自 `findSourceMemref("result", ...)` 找到的 `arg3` 与 A1 buffer 的错误组合。

---

## 问题 5：add 的 ins[0] 使用的是 CO1 原始 buffer，而不是 VECIN

**第 82 行**：

```mlir
linalg.elementwise kind=add
    ins(%alloc_25, %subview_26 : memref<?x?xf32, 7 : i32>, ...)
```

`%alloc_25` 是 CO1（`7 : i32`），add 直接读 CO1，而不是读 for_K epilogue 插入的 VECIN（`%alloc_47`, `9 : i32`）。这说明虽然 pass 插入了 CO1→VECIN 的 copy，但 add op 的 operand 没有被更新为 VECIN 的值。

这是因为 Phase 1 只是在循环边界插入了新的 alloc 和 copy，但没有把 linalg op 的 operand 重定向到新的 VECIN buffer。Phase 2 需要处理这个重定向，但目前 IR 的连接关系是断的。

---

### 问题1：linalg.matmul 的 ins 读的是原始 GM subview，不是 A2/B2
代码行：examples/matmul-add-relu-sum/output_step3_buffer_placement.mlir:52

问题分析：matmul的ins应该是%alloc_20和%alloc_22，分别对应 A2/B2。可能是 pass 在 for_K prologue 里插入了 A2/B2 的 alloc 和 copy，但没有把 linalg.matmul 的 ins operand 替换为 %alloc_20 和 %alloc_22

修复方向：在插入 A2/B2 的 copy 之后，把 matmul 的 ins 替换为新的 A2/B2 buffer
```CPP
// 插入 A2 copy 后
matmulOp.getInputsMutable()[0].assign(alloc_A2);
// 插入 B2 copy 后
matmulOp.getInputsMutable()[1].assign(alloc_B2);
```

### 问题2：matmul 的 ins 用的是 GM subview 而非 A1/B1 subview（中等）
代码行：examples/matmul-add-relu-sum/output_step3_buffer_placement.mlir:45
```mlir
%subview_19 ← %subview_10[0, %arg13] ← %subview[%arg11, 0] ← %arg0 (GM)
%subview_21 ← %subview_11[%arg13, 0] ← %subview_3[0, %arg12] ← %arg1 (GM)
```
问题分析：理想情况下，A1→A2 的 copy 应该以 %alloc_2（A1，1 : i32）的 subview 为 source，而不是直接从 GM subview。实际上 A1→A2 的 copy（第 47 行）的 source 也是 GM（%subview_19），不是 %alloc_2（A1）。这意味着 A1 buffer 在搬运链上也是死代码——数据流是 GM → A2，完全跳过了 GM → A1 → A2 的两级搬运。对硬件来说，GM→A2 是非法的（MTE1 通道只能做 L1↔L0，不能直接访问 GM）。正确的搬运链必须是 GM →(MTE2)→ A1 →(MTE1)→ A2。

修复方向：for_K 的 prologue 处理时，A2 的 source 应该是 %alloc_2（A1）的对应 subview，而不是 GM 的 subview。pass 需要知道"A1 buffer 里哪一块对应当前 K 迭代"，即 memref.subview %alloc_2[0, %arg13] [%8, %10] [1, 1]。

### 问题3：CO1 初始化缺失（严重）
代码行：examples/matmul-add-relu-sum/output_step3_buffer_placement.mlir:41-42
```mlir
%alloc_13 = memref.alloc(%8, %9) : memref<?x?xf32, 7 : i32>  // CO1
memref.copy %subview_12, %alloc_13  // 从 GM 的 C subview 初始化 CO1
```

问题分析：这里把 %subview_12（C 矩阵的 GM subview，即已有的部分结果）复制到 CO1 作为初始值，目的是让 for_K 的多次迭代能在 CO1 上累加。这会导致第一次 matmul 时 CO1 的初始值是 %arg3 的原始内容（可能是随机值），而不是 0。

修复方向：把 memref.copy %subview_12, %alloc_13 替换为 linalg.fill 零初始化

### 问题 4：add 的 ins[1]（bias）来自 GM subview，不是 VECIN buffer（中等）
代码行：examples/matmul-add-relu-sum/output_step3_buffer_placement.mlir:58
```mlir
linalg.elementwise kind=add
    ins(%alloc_14, %subview_15 :
        memref<?x?xf32, 9 : i32>,           // VECIN，正确
        memref<?x?xf32, strided<[?, ?], offset: ?>>)  // ← 无 memory_space，GM subview
    outs(%alloc_16 : memref<?x?xf32, 11 : i32>)
```
问题分析：%subview_15（第 56 行）来自 %subview_6，来自 %arg2（bias，GM）。bias 已经在 for_TB_N 的 prologue 搬运到了 %alloc_7（VECIN，9 : i32，第 29–30 行），但 add 的 ins[1] 没有使用 %alloc_7，仍然直接读 GM。同样是 operand 重定向的问题。

### 问题 5：alloc_13（CO1）和 alloc_20/alloc_22（A2/B2）缺少 dealloc
第 41 行的 %alloc_13、第 46 行的 %alloc_20、第 49 行的 %alloc_22 在循环体结束时没有对应的 memref.dealloc。相比之下，%alloc_14（第 62 行）和 %alloc_16（第 63 行）有 dealloc。片上存储资源有限，必须显式管理生命周期。

---

### 问题1：add 的 ins[1] 使用全量 VECIN bias，而非当前 tile 的 subview（严重）
行号：examples/matmul-add-relu-sum/output_step3_buffer_placement.mlir:56

问题分析：%alloc_7 是 for_TB_N prologue 创建的 bias VECIN buffer，大小是 [%6, %7]（即整个 TB tile：TB_M × TB_N）。
但此处 add 在 for_Tb_M × for_Tb_N 的最内层，每次只处理 [%8, %9]（即 tb_M × tb_N）大小的 tile。%alloc_11（VECIN，CO1 搬运来的）大小是 [%8, %9]，而 %alloc_7 大小是 [%6, %7]，两者尺寸不匹配，linalg.elementwise 会报错或产生错误结果。

期望：用 %alloc_7 的 subview
```mlir
%subview_bias = memref.subview %alloc_7[%arg11, %arg12] [%8, %9] [1, 1]
    : memref<?x?xf32, 9:i32> to memref<?x?xf32, strided<[?, 1], offset: ?>, 9:i32>
linalg.elementwise kind=add
    ins(%alloc_11, %subview_bias : ...)
```

### 问题2：CO1→VECIN 的 copy（alloc_11） 的 dealloc 缺失（低）
行号：examples/matmul-add-relu-sum/output_step3_buffer_placement.mlir:53

### 问题3：%alloc_9（VECOUT）初始化也应该用fill
行号：examples/matmul-add-relu-sum/output_step3_buffer_placement.mlir:32
我建议把所有的output空间的初始化统一成fill

---

### 问题 1：max ins[1] 仍然没有 memory_space（中等）

**第 58–60 行**：

```mlir
%subview_14 = memref.subview %subview_8[%arg11, %arg12] [%8, %9] [1, 1]
    : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>

linalg.elementwise kind=max_signed
    ins(%alloc_12, %subview_14 :
        memref<?x?xf32, 11:i32>,                          // VECCALC ✅
        memref<?x?xf32, strided<[?, 1], offset: ?>>)      // ← 无 memory_space ❌
```

`%subview_14` 来自 `%subview_8`（第 31 行），来自 `%alloc`（第 14 行），而 `%alloc` 是裸 `memref<?x?xf32>`，没有 memory_space。

这个问题上一版就提到了，零张量 `%alloc` 应该换成在 `for_Tb_N` 内动态分配的带 memory_space 的小 buffer：

```mlir
// 在 for_Tb_N 内，alloc_10 之后：
%zero_buf = memref.alloc(%8, %9) : memref<?x?xf32, 9 : i32>   // VECIN
linalg.fill ins(%cst : f32) outs(%zero_buf : memref<?x?xf32, 9 : i32>)

linalg.elementwise kind=max_signed
    ins(%alloc_12, %zero_buf : memref<?x?xf32, 11:i32>, memref<?x?xf32, 9:i32>)
    outs(%subview_15 : ...)
memref.dealloc %zero_buf
```

同时，第 14–15 行的全量 `%alloc` 和 `linalg.fill`、第 31 行的 `%subview_8` 均可删除。

---

### 问题 2：`%alloc_2`（A1）和 `%alloc_4`（B1）缺少 dealloc（低）

扫描整个 IR，`for_TB_N` 循环体内：

- `%alloc_7`（VECIN bias，`9:i32`）：无 dealloc ❌
- `%alloc_2`（A1，`1:i32`）：无 dealloc ❌
- `%alloc_4`（B1，`3:i32`）：无 dealloc ❌
- `%alloc_9`（VECOUT，`10:i32`）：第 67 行有 dealloc ✅

这三个 buffer 在 `for_TB_N` 的 epilogue（第 66 行 copy 之后）应该各加一行 `memref.dealloc`：

```mlir
memref.copy %alloc_9, %subview_5 ...
memref.dealloc %alloc_9
memref.dealloc %alloc_2   // A1
memref.dealloc %alloc_4   // B1
memref.dealloc %alloc_7   // VECIN bias
```