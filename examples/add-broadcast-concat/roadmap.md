# ewop-broadcast-concat → AscendC Kernel 编译流水线 Roadmap

**目标**：将 `step0_input.mlir`（linalg-on-tensor）编译为 AscendC kernel C++ 代码。
**场景**：ElementwiseOp（add+mul）+ BroadcastOp + ConcatOp

---

## 计算图

```
input_a[M]     ──┐ (broadcast map)
input_b[M,N]   ──┤  Op1: broadcast+add  ──→ C[M,N]
                          │
input_c[M]     ──┐ (broadcast map)      │
input_d[M,N]   ──┤  Op2: broadcast+mul  ──→ D[M,N]
                                         │
                         Concat(axis=0) ─┘
                              │
                         output[2M, N]
```

---

## 设计决策

### 1. Concat 的表示方式

**选择**：两个 `tensor.insert_slice` 写入同一输出 tensor 的不同 subview。

**理由**：
- linalg 层面无需新增 concat op
- bufferization 后变成两个 `memref.subview` 写操作，subview 的 offset 自然编码了 concat 位置
- `DataMoveConversion` 中 VECOUT→GM 的 `data_copy_l2` 使用 `global_tensor` +
  `global_tensor_set_global_buffer` 设置 GM 指针，当 GM 指针是一个带 offset 的 subview 时，
  offset 自动保留在 GM 地址计算中
- 无需在 `ComputeConversion` 或 `DataMoveConversion` 中新增 `asc.concat` 处理路径

**对比 asc.concat 方案**：
- `asc.concat` 接受 variadic `local_tensor` 输入，轴和 calCount 参数
- 需要在 ComputeConversion 中识别 linalg.generic 的 concat 语义（复杂）
- insert_slice 方案更简洁，且与现有两个示例的管线完全统一

### 2. 纯并行 generic 的下沉路径

**问题**：现有 `ComputeConversion.cpp` 中对 `linalg.generic` 仅处理含 reduction 的情况（line 246 `if (!hasReduction) continue`）。

**解决**：在 reduction generic 循环之后新增 pure-parallel generic 处理块：
- 复用相同的 `isBroadcastMap`、`allocVeccalc`、`readTensor` 等辅助函数
- Step 1：广播输入用 `broadcast_l2`，非广播输入用 `readTensor`
- Step 2：body 内联（addf→`add_l2`，mulf→`mul_l2`，maximumf→`max_l2`）
- Step 3：无 reduce 步骤；将 accumLt enqueue 到输出 queue（若有），让 DataMoveConversion 负责写回

### 3. linalg.elementwise mul 支持

现有代码只处理 `add` 和 `max_signed`。新增 `mul` → `MulL2Op`：
- `ComputeConversion.cpp` elementwise filter 条件中加入 `mul`
- kind switch 中加入 `MulL2Op` 分支
- reduction generic body walker 和 pure-parallel body walker 均加入 `arith.MulFOp → MulL2Op`

### 4. Tiling 策略

Op1 和 Op2 均为 `["parallel","parallel"]`（M×N），独立 tile：
- 仅对 d0（M 轴）做两级切分：TB_M（核间）/ Tb_M（UB 批次）
- d1（N 轴）不切，整 N 在 UB 内处理（与 broadcast-add-reduce 一致）
- 两个 generic 分别 tile，各自产生独立的 scf.for 嵌套
- concat 的两个 insert_slice 自动继承各自 tiled generic 的 M offset

---

## 编译流水线

```
Step 0: step0_input.mlir
        ─ 原始 linalg-on-tensor IR

Step 1: --linalg-fuse-elementwise-ops
        ─ 本场景 no-op（Op1/Op2 输出独立）
        ─ 保留此阶段以与其他示例保持一致

Step 2: --transform-interpreter step2_transform.mlir
        ─ Op1 和 Op2 各自的 TB/Tb 两级 M 轴 tiling
        ─ 标注 ascendc.parallel / prologue / epilogue / unit

Step 3: --one-shot-bufferize
        ─ tensor → memref
        ─ insert_slice → memref.subview 写（concat 位置编码在 offset 中）

Step 4: --ascendc-buffer-placement
        ─ 推导 VECIN(9) / VECOUT(10) memory_space
        ─ 插入 memref.copy 占位（GM→VECIN，VECOUT→GM）
        ─ VECOUT→GM 的 copy target 是带 offset 的 subview，保留 concat 位置

Step 5: --linalg-to-ascendc
        ─ pure-parallel generic → broadcast_l2 + add_l2 / mul_l2
        ─ memref.copy GM→VECIN → data_copy_l2
        ─ memref.copy VECOUT→GM → data_copy_l2（target subview offset 保留）

Step 6: --ascendc-parallelize
        ─ TB scf.for → get_block_idx 单维多核调度

Step 7: --ascendc-prepare-for-emit
        ─ 添加 aicore / global 属性，解包 TilingData struct

Step 8: ascir-translate -mlir-to-ascendc
        ─ 生成 AscendC C++ kernel 源码
```

---

## 代码修改

只修改一个现有文件：

### `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`

1. **reduction generic body walker**（~line 427）：新增 `arith.MulFOp → MulL2Op`
2. **linalg.elementwise filter**（~line 538）：加入 `ElementwiseKind::mul`
3. **linalg.elementwise kind switch**（~line 589）：加入 `MulL2Op` 分支
4. **新增 pure-parallel generic 处理块**：在 reduction generic 循环之后、
   `linalg.matmul` 循环之前，处理 `iterator_types = ["parallel", ...]` 的情况

---

## 新增文件

```
examples/ewop-broadcast-concat/
├── step0_input.mlir       ✓ 原始 linalg IR
├── step2_transform.mlir   ✓ Transform tiling 脚本
├── run.sh                 ✓ 端到端流水线脚本
└── roadmap.md             ✓ 本文档
```

---

## 里程碑

| 里程碑 | 验证方式 |
|---|---|
| M1: step2 Tiling | `afir-opt --transform-interpreter step2_transform.mlir --canonicalize --cse \| FileCheck` → 有 `scf.for` + `ascendc.parallel` |
| M2: step3 Bufferize | 无 tensor op，有 `memref.subview` 携带 `[%dim_m, 0]` offset |
| M3: step4 Buffer Placement | `9 : i32` / `10 : i32` 出现，有 `memref.copy` |
| M4: step5 AscendC | 有 `ascendc.broadcast_l2` + `ascendc.add_l2` + `ascendc.mul_l2` |
| M5: C++ 生成 | `Broadcast(...)` + `Add(...)` + `Mul(...)` 调用出现在 .cpp 中 |

---

## 关键文件索引

| 文件 | 用途 |
|---|---|
| `examples/ewop-broadcast-concat/step0_input.mlir` | 原始计算图 |
| `examples/ewop-broadcast-concat/step2_transform.mlir` | Tiling 调度脚本 |
| `examples/ewop-broadcast-concat/run.sh` | 端到端运行脚本 |
| `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp` | 核心修改：pure-parallel + mul 支持 |
| `externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecBinary.td` | `mul_l2` op 定义 |
| `externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecBroadcastExt.td` | `broadcast_l2` op 定义 |
| `externals/pyasc/include/ascir/Dialect/Asc/IR/Basic/OpVecDataMoveExt.td` | `concat` op 定义（备用） |
