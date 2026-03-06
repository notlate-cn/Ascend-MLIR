# fc_add_relu → AscendC Kernel 编译流水线 Roadmap

**目标**：将 `fc_add_relu.mlir`（linalg-on-tensor）全自动编译为 AscendC kernel C++ 代码，支持动态 Shape 编译期生成 + 执行期 Shape 特化优化。

---

## 现状（已完成）

```
Step 1: fc_add_relu.mlir
        ↓ transform_tile_and_fuse_3level（Transform Dialect）
Step 2: output_step1_tile_and_fuse_3level.mlir
        ↓ One-Shot Bufferize
Step 3: output_step2_bufferized.mlir
        ↓ AscendCBufferPlacementPass（已实现，未push）
Step 4: output_step3_buffer_placement.mlir
        ✓ memref 带 memory_space（1:i32 ~ 11:i32）
        ✓ 显式 memref.copy 占位符
        ✓ ascendc.* annotation 已清除
```

---

## 编译期流水线（全符号化，支持动态 Shape）

### Step 5：CopyToAscendCDataMovePass

**目标**：将 `memref.copy` 替换为对应的 AscendC 搬运 op。

**输入**：output_step3_buffer_placement.mlir（带 memory_space 的 memref.copy）

**转换规则**（根据 src/dst memory_space 判断）：

| src memory_space | dst memory_space | AscendC op |
|---|---|---|
| GM (无/0) | A1(1) / B1(3) / VECIN(9) | `asc.data_copy_l2 dst, src, count` |
| A1(1) | A2(2) | `asc.data_copy_l0 dst, src, params` |
| B1(3) | B2(4) | `asc.data_copy_l0 dst, src, params` |
| CO1(7) | VECIN(9) | `asc.fixpipe dst, src, workspace, params` |
| VECOUT(10) | GM (无/0) | `asc.data_copy_l2 dst, src, count` |

**关键实现点**：
- `data_copy_l2` 的 `calCount` 参数：动态 shape 场景下用 `memref.dim` + `arith` 计算元素数，静态 shape 直接折叠为常量
- `data_copy_l0` 的 `repeatParams`：构造 `!asc.data_copy_params`，包含 stride/repeat 信息
- `fixpipe` 的 `intriParams`：构造 `!asc.fixpipe_params<f32>`，用 `asc.construct` 创建默认参数
- Pass 名：`--ascendc-copy-to-datamove`
- 文件位置：`lib/Conversion/AscendCCopyToDataMove/`
- 测试：`test/Conversion/ascendc-copy-to-datamove.mlir`

**输出**：output_step4_datamove.mlir

---

### Step 6：LinalgToAscendCComputePass

**目标**：将 linalg 计算 op 一对一映射到 AscendC Dialect 的计算 op。

**输入**：output_step3_buffer_placement.mlir（直接对接 Step 4 输出）

**Pass 名**：`--linalg-to-ascendc-compute`
**文件位置**：`lib/Conversion/LinalgToAscendCCompute/`
**测试**：`test/Conversion/linalg-to-ascendc-compute.mlir`

---

#### 6a. linalg.matmul → AscendC Matmul 状态机

输入 IR（来自 output_step3_buffer_placement.mlir:116-118）：
```mlir
linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<?x?xf32, 7 : i32>)
// ... K-loop ...
linalg.matmul
  ins(%alloc_15, %alloc_17 : memref<?x?xf32, 2 : i32>, memref<?x?xf32, 4 : i32>)
  outs(%subview_18 : memref<?x?xf32, strided<[?, 1]>, 7 : i32>)
```

转换后：
```mlir
// linalg.fill ins(0) outs(CO1) → 不产生 op（matmul.iterate_all 内部初始化）
// 或保留 linalg.fill 待后续 pass 处理

// linalg.matmul → matmul 状态机
// MatmulType: !asc.matmul<a2, ND, f32, false, NONE, b2, ND, f32, false, NONE,
//                          co1, ND, f32, false, NONE, gm, ND, f32, {}>
%mm : !asc.matmul<a2, ND, f32, false, NONE, b2, ND, f32, false, NONE,
                  co1, ND, f32, false, NONE, gm, ND, f32, {}>
asc.matmul.set_tensor_a %mm, %alloc_15, %false
  : !asc.matmul<...>, memref<?x?xf32, 2:i32>, i1
asc.matmul.set_tensor_b %mm, %alloc_17, %false
  : !asc.matmul<...>, memref<?x?xf32, 4:i32>, i1
asc.matmul.iterate_all %mm, %subview_18, %false
  : !asc.matmul<...>, memref<?x?xf32, 7:i32>, i1
```

**关键问题**：`asc.matmul.init` 需要 `AscendC_TCubeTiling` 操作数，该 tiling 信息来自执行期；
**暂定方案**：Step 6 只生成 `set_tensor_a/b` + `iterate_all`，`init`/`end` 由后续 Step 5 结合 TQue 插入，或单独插入一个 `AscendC_MatmulInitPlaceholderPass`。

---

#### 6b. linalg.elementwise kind=add → asc.add_l2

输入 IR（:138-140）：
```mlir
linalg.elementwise kind=#linalg.elementwise_kind<add>
  ins(%alloc_9, %subview_11 : memref<?x?xf32, 9:i32>, memref<?x?xf32, strided<...>, 9:i32>)
  outs(%alloc_10 : memref<?x?xf32, 11:i32>)
```

转换后（使用 `BinaryTemplateL2Op`，签名 `(dst, src0, src1, calCount)`）：
```mlir
// calCount = 元素总数 = dim0 * dim1（动态时用 memref.dim + arith.muli 计算）
%n0 = memref.dim %alloc_10, %c0 : memref<?x?xf32, 11:i32>
%n1 = memref.dim %alloc_10, %c1 : memref<?x?xf32, 11:i32>
%count = arith.muli %n0, %n1 : index
asc.add_l2 %alloc_10, %alloc_9, %subview_11, %count
  : memref<?x?xf32, 11:i32>, memref<?x?xf32, 9:i32>, memref<?x?xf32, strided<...>, 9:i32>, index
```

---

#### 6c. linalg.fill + linalg.elementwise kind=max_signed → asc.duplicate_l2 + asc.max_l2

输入 IR（:148-155）：
```mlir
// linalg.fill 产生一个全零的 VECIN buffer（零值操作数）
%alloc_13 = memref.alloc(%8, %9) : memref<?x?xf32, 9 : i32>
linalg.fill ins(%cst : f32) outs(%alloc_13 : memref<?x?xf32, 9 : i32>)

linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
  ins(%alloc_10, %alloc_13 : memref<?x?xf32, 11:i32>, memref<?x?xf32, 9:i32>)
  outs(%subview_12 : memref<?x?xf32, strided<...>, 10:i32>)
```

观察：`linalg.fill` + `max_signed` 的组合语义是 ReLU（`max(x, 0)`）。
转换策略：匹配「fill 产生的全零 buffer 作为 max_signed 的 ins[1]」模式，消除 fill+alloc，改用标量零直接传入 `duplicate_l2`：

```mlir
// 消除 %alloc_13 + linalg.fill，改用 duplicate 将标量 0 广播到 dst
// （或：直接用 max_l2 的第三操作数传标量，取决于 max_l2 API 是否支持 scalar-tensor）

// 方案 A：保留 fill→duplicate，仅替换 elementwise
asc.duplicate_l2 %alloc_13, %cst, %count
  : memref<?x?xf32, 9:i32>, f32, index
asc.max_l2 %subview_12, %alloc_10, %alloc_13, %count
  : memref<..., 10:i32>, memref<..., 11:i32>, memref<..., 9:i32>, index

// 方案 B（更激进）：若 max_l2 不支持，先 fold fill+max 为 relu-like 专用 op（待调研）
```

**建议先实现方案 A**，语义明确，不依赖额外调研。

---

#### 6d. linalg.fill（非零值初始化）→ asc.duplicate_l2

对于 `linalg.fill ins(%cst) outs(%CO1_buf)` 这种对 CO1 的初始化（:87）：
```mlir
linalg.fill ins(%cst : f32) outs(%alloc_8 : memref<?x?xf32, 7 : i32>)
```

转换后：
```mlir
asc.duplicate_l2 %alloc_8, %cst, %count : memref<?x?xf32, 7:i32>, f32, index
```

（`duplicate_l2` 签名：`(dst: LocalTensor, scalar: AnyType, calCount: AnyType)`）

---

#### 转换规则汇总

| 输入 linalg op | 判断条件 | 输出 asc op |
|---|---|---|
| `linalg.matmul` | outs memory_space == 7（CO1） | `asc.matmul.set_tensor_a/b` + `asc.matmul.iterate_all` |
| `linalg.fill` | outs memory_space == 7（CO1 初始化） | `asc.duplicate_l2` |
| `linalg.fill` | outs memory_space == 9（全零占位） | 与下一个 `max_signed` 合并，见 6c |
| `linalg.elementwise add` | — | `asc.add_l2` |
| `linalg.elementwise max_signed`（ins[1] 来自 fill） | 匹配 fill+max 模式 | `asc.duplicate_l2` + `asc.max_l2` |

**不转换**（保持原样传给后续 pass）：
- `memref.alloc` / `memref.dealloc` / `memref.subview` / `memref.copy`
- `scf.for` / `affine.min` / `arith.*`

**输出**：output_step5_compute.mlir（linalg op 全部替换，其余 IR 不变）

---

### Step 7：EmitAscPass（AscendC C++ 代码生成）

**目标**：使用已有的 `EmitAsc` Dialect（`externals/pyasc/include/ascir/Dialect/EmitAsc/`）将 Step 6 输出转为 AscendC C++ kernel 代码。

**关键事项**：
- EmitAsc 是已有基础设施，调研其对 `LocalTensor<T, pos>` 类型的要求
- 可能需要在 Step 6 之后插入一个类型转换 pass：`memref<..., pos:i32>` → `!asc.local_tensor<f32, pos>`
- 或者直接在 LinalgToAscendCComputePass 中输出带 `local_tensor` 类型的 op

**待调研**：
- [ ] EmitAsc 的输入 IR 形态约定（读 `EmitAsc/IR/Ops.td`）
- [ ] `asc.local_tensor` 和 `asc.global_tensor` 的类型定义
- [ ] 现有 EmitAsc 用例（`externals/pyasc/test/` 下是否有示例）

---

## 执行期流水线（Shape 已知，基于 Affine 做优化）

执行期收到具体的 M/N/K 值后，对编译期生成的 kernel 进行特化优化。

### Step E1：Shape 特化（Specialization）

将动态 Shape 的 `?` 替换为具体值，触发以下优化：

```
memref<?x?xf32, 1:i32>  →  memref<128x256xf32, 1:i32>
```

- 使用 MLIR 标准的 `--inline` + `--canonicalize` 将 `i64` 参数折叠为常量
- 使用 `--scf-for-loop-range-folding` 消除 `affine.min`（边界 tile 退化为整 tile）
- Pass 名：通过 pass pipeline 组合，无需新 pass

### Step E2：Affine 转换与循环优化

Shape 已知后，Affine 分析可做更多优化：

| 优化 | MLIR Pass | 收益 |
|---|---|---|
| 消除冗余 alloc/dealloc | `--buffer-loop-hoisting` | 减少片上内存分配次数 |
| 循环展开 | `--affine-loop-unroll` | 消除小循环 overhead |
| 向量化 | `--affine-vectorize` | 利用 SIMD（如适用） |
| 循环平铺再优化 | `--affine-loop-tile` | 进一步调整 cache 局部性 |
| 内存访问合并 | `--affine-data-copy-generate` | 探索 L1/L0 的最优 tiling |

### Step E3：执行期 Kernel 调度

对于需要运行时 dispatch 的场景（如 M/N/K 变化频繁）：

- 编译期为若干代表性 Shape 生成多个特化 kernel
- 执行期根据实际 Shape 选择最近的 kernel（shape matching table）
- 长期目标：结合 Tuner 自动搜索最优 tile 参数（arg4~arg8）

---

## 推荐实现顺序

```
优先级 1（一对一映射，最清晰）：
  Step 6: LinalgToAscendCComputePass
    → linalg.matmul → asc matmul 状态机 op 序列
    → linalg.elementwise add → asc.add_l2
    → linalg.elementwise max_signed + linalg.fill → asc.max_l2（消除 fill，标量化零值）
    → 验证：output_step5_compute.mlir 只含 asc.* op + memref + scf

优先级 2（依赖 Step 7 的 local_tensor 类型约定确定后实现）：
  Step 5: CopyToAscendCDataMovePass
    → 将 memref.copy 替换为 asc 搬运 op
    → 同时处理 PipeBarrier 插入和 TQue EnQue/DeQue 配对
    → 验证：output_step4_datamove.mlir 可通过 mlir-opt verify

优先级 3（代码输出）：
  Step 7: 调研 EmitAsc，打通到 C++ 代码生成
    → 调研 local_tensor / global_tensor 类型约定
    → 输出 fc_add_relu_kernel.cpp

优先级 4（执行期优化）：
  Step E1+E2: Shape 特化 + Affine 优化
    → 建立 shape → optimized kernel 的 pipeline
```

---

## 里程碑检查点

| 里程碑 | 验证方式 | 标志 |
|---|---|---|
| M1: Step 5 完成 | `afir-opt ... --ascendc-copy-to-datamove \| FileCheck` | 无 memref.copy，有 asc.data_copy_* / asc.fixpipe |
| M2: Step 6 完成 | `afir-opt ... --linalg-to-ascendc-compute \| FileCheck` | 无 linalg.*，有 asc.matmul.* / asc.vec_* |
| M3: Step 7 完成 | 输出合法 C++ | `asc.data_copy_l2(...)` 等 API 调用出现在 .cpp 中 |
| M4: 执行期 E1 完成 | 具体 Shape 下 affine.min 消失 | `--canonicalize` 后 tile 大小全为常量 |

---

## 关键文件索引

| 文件 | 用途 |
|---|---|
| `examples/matmul-add-relu-sum/run.sh` | 端到端运行脚本（逐步扩展） |
| `lib/Conversion/AscendCBufferPlacement/` | Step 4（已完成） |
| `lib/Conversion/AscendCCopyToDataMove/` | Step 5（待实现） |
| `lib/Conversion/LinalgToAscendCCompute/` | Step 6（待实现） |
| `externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpDataCopy.td` | DataCopyL2/L0 op 定义 |
| `externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpFixpipe.td` | Fixpipe op 定义 |
| `externals/pyasc/include/ascir/Dialect/Asc/IR/Adv/Matmul.td` | matmul 状态机 op 定义 |
| `externals/pyasc/include/ascir/Dialect/EmitAsc/` | C++ 代码生成（Step 7） |
