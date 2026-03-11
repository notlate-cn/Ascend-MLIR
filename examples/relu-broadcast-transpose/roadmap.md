# ewop-broadcast-transpose 实现路线图

## 计算图

```
输入: A[M,N], bias[N], scale[M]

Op1: relu(A[M,N]) + broadcast_col(bias[N]) → B[M,N]
     列广播：bias[N] 沿 M 轴扩展
     relu = max(x, 0)

Op2: Transpose(B[M,N]) → C[N,M]
     行列互换：C[i,j] = B[j,i]

Op3: C[N,M] * broadcast_col(scale[M]) → D[N,M]
     列广播：scale[M] 沿 N 轴扩展
```

## 设计决策

### Transpose 的 linalg 表达方式

Transpose 用 `linalg.generic` 表达（不用 `linalg.transpose`），这样可以复用现有的 pure-parallel 路径：

```mlir
linalg.generic {
  indexing_maps = [#transpose_map, #full_access_map],  // (d0,d1)->(d1,d0), identity
  iterator_types = ["parallel", "parallel"],
  library_call = "transpose"
} ins(%b : tensor<?x?xf16>) outs(%c : tensor<?x?xf16>) {
^bb0(%val: f16, %out: f16):
  linalg.yield %val : f16  // 直接 yield，无计算
}
```

优势：
- 无需修改 tiling、bufferize、buffer-placement passes
- Transform tiling 可以直接对 transpose generic 操作

### ComputeConversion.cpp 修改

在 `parallelGenericOps` 循环最前面添加 `isTransposeGeneric` 检测：

```cpp
// 检测模式：1 input, 1 output, input map = (d0,d1)->(d1,d0), output = identity
if (isTransposeGeneric(genOp)) {
  Value srcLt = readTensor(builder, loc, inMemref);   // VECIN deque
  Value dstLt = writeTensor(builder, loc, outMemref); // VECOUT alloc
  builder.create<TransposeOp>(loc, dstLt, srcLt);     // ascendc.transpose
  if (Value q = ctx.getQueue(outMemref))
    builder.create<TQueBindEnqueTensorOp>(loc, q, dstLt);
  genOp.erase();
  continue;
}
```

### Tiling 策略

Transpose 的迭代空间是 [N, M]（d0=N, d1=M）：
- 对 d0（N 轴）做两级切分 TB/Tb
- d1（M 轴）不切，整块在 UB 内转置

每个 tile 的数据流：
- 读入 B[i*Tb:i*Tb+Tb, :] → VECIN（Tb×M 元素）
- ascendc.transpose → VECOUT（M×Tb 元素放在 C 的对应位置）
- 写回 C[i*Tb:i*Tb+Tb, :] → GM

### 与现有场景的统一性

| 特性              | broadcast-add-reduce | relu-broadcast-split | ewop-broadcast-transpose |
|-------------------|---------------------|---------------------|--------------------------|
| Pass pipeline     | ✓ 相同              | ✓ 相同              | ✓ 相同                   |
| 内存层次          | GM/VECIN/VECOUT      | GM/VECIN/VECOUT      | GM/VECIN/VECOUT           |
| 多核调度          | get_block_idx        | get_block_idx        | get_block_idx             |
| 新增 op           | broadcast_l2+reduce  | broadcast_l2+mul     | **ascendc.transpose**     |
| library_call 区分 | N/A（单 op）         | ✓                   | ✓                         |

## 编译流水线步骤

| 文件                         | Pass                              | 描述                              |
|------------------------------|-----------------------------------|-----------------------------------|
| `step0_input.mlir`           | —                                 | linalg-on-tensor 源码             |
| `step1_fused.mlir`           | `--linalg-fuse-elementwise-ops`   | 尝试融合（transpose 阻断，3→3）    |
| `step2_tiled.mlir`           | `--transform-interpreter`         | TB/Tb 两级循环（3个 op 各自）      |
| `step3_bufferized.mlir`      | `--one-shot-bufferize`            | memref-based IR                   |
| `step4_buffer_placement.mlir`| `--ascendc-buffer-placement`      | memory_space 标注（VECIN/VECOUT）  |
| `step5_ascendc.mlir`         | `--linalg-to-ascendc`             | AscendC ops（含 ascendc.transpose）|
| `step6_parallelize.mlir`     | `--ascendc-parallelize`           | get_block_idx 多核调度             |
| `step7_kernel.mlir`          | `--ascendc-prepare-for-emit`      | 完整 kernel IR                    |
| `step8_kernel.cpp`           | `ascir-translate -mlir-to-ascendc`| AscendC C++ kernel 源码           |

## 预期生成的 C++ kernel 结构

```cpp
extern "C" __global__ __aicore__ void ewop_broadcast_transpose(
    half* input_a, half* bias, half* scale,
    __gm__ TilingData* tiling_data_ptr,
    half* output
) {
    // ... TilingData 加载 ...
    uint32_t block_idx = GetBlockIdx();

    // Op1: relu + broadcast_col(bias) + add → B[M,N]
    for (uint32_t i = ...) {
        // DataCopy A tile [i:i+Tb, :] → VECIN
        // broadcast_l2(bias_veccalc, bias_vecin, [Tb,N], [N], 2)
        // duplicate_l2(zero_veccalc, 0.0, Tb*N)
        // max_l2(relu_veccalc, a_vecin, zero_veccalc, Tb*N)  // relu
        // add_l2(b_vecout, relu_veccalc, bias_veccalc, Tb*N) // add bias
        // DataCopy B tile → GM
    }

    // Op2: Transpose B[M,N] → C[N,M]
    for (uint32_t i = ...) {
        // DataCopy B[:, i:i+Tb] → VECIN (Tb 列)
        AscendC::Transpose(c_vecout, b_vecin);  // 转置
        // DataCopy C tile [i:i+Tb, :] → GM
    }

    // Op3: C[N,M] * broadcast_col(scale[M]) → D[N,M]
    for (uint32_t i = ...) {
        // DataCopy C tile [i:i+Tb, :] → VECIN
        // broadcast_l2(scale_veccalc, scale_vecin, [Tb,M], [M], 2)
        // mul_l2(d_vecout, c_vecin, scale_veccalc, Tb*M)
        // DataCopy D tile → GM
    }
}
```

## 已修改的文件

- `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`：
  - 新增 `isTransposeGeneric()` lambda（检测置换 indexing_map）
  - 在 `parallelGenericOps` 循环中添加 transpose 分支，emit `ascendc.transpose`
