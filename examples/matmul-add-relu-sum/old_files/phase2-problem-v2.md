**所有严重问题的根本原因是同一个**：prologue 里的 alloc/enque 与循环内的 alloc/enque 存在嵌套冲突，queue 容量为 1 时无法同时持有两个 tensor。需要严格保证每个 queue 的 **alloc→enque→deque→free** 是完整配对的，且 enque 和对应的 deque 在**同一循环层级**。

## 问题 1：A1/B1 在 for_K 内每次 deque/free，仍然只 enque 一次（严重）

这是上版的核心问题，本版未修复。

**第 60 行**：A1 在 for_TB_N prologue 只 enque 了一次。  
**第 103、113 行**：for_K 内每次迭代都 `deque %1(A1)` + `free_tensor %1`。

for_K 有多次迭代，第一次 deque 后 A1 队列就空了，第二次迭代的 `deque_tensor %1` 会阻塞或出错。

B1 同理（第 73 行 enque 一次，第 117、122 行每次迭代都 deque/free）。

**对照样例的正确模式**：

```
// for_K 外：A1 deque 一次，做整体 SplitA（A1→A2）
A1.deque → LoadData 全量 A (for mBlocks) → A2.enque → A1.free

// for_K 内：只操作 A2（按 K slice 取 subview）
A2.deque[slice] → mmad → A2.free
```

但当前 IR 的 tiling 结构（for_K 是 K 维度的分块循环）和样例不完全相同，需要确定：是把 A1 全量转 A2 后在 for_K 内取 A2 subview，还是 for_K 内每次只搬一个 K slice 的 A1→A2。

如果是后者（每次只搬一个 K slice），则 A1 的 `alloc/enque` 也应该在 for_K 内，而不是在 prologue。**A1 的 enque 位置和 deque 位置必须在同一循环层级**。

---

## 问题 2：CO1 queue 在 for_K 内被 alloc 两次（严重）

**第 95–97 行**（for_Tb_N prologue）：
```mlir
%57 = ascendc.que_bind.alloc_tensor %5   // CO1 alloc + enque
ascendc.duplicate_l2 %57, %cst, ...
ascendc.que_bind.enque_tensor %5, %57    // 队列已满(容量1)
```

**第 125 行**（for_K 内）：
```mlir
%86 = ascendc.que_bind.alloc_tensor %5   // CO1 再次 alloc，但队列已满
```

队列容量为 1，prologue 已经 enque 了，for_K 内再 alloc 会阻塞。

此外，**`cmatrixInitVal=false`** 意味着 mmad 硬件自动清零 CO1，第 96 行的 `duplicate_l2` 零初始化 CO1 是多余的。

**正确模式**：删除 prologue 的 CO1 alloc/duplicate/enque，for_K 内直接 alloc → mmad → enque，for_K 结束后 deque → fixpipe。

---

## 问题 3：VECOUT 在 prologue 被零初始化并 enque，for_Tb_N 内 max 又 alloc/enque（严重）

**第 85–87 行**（prologue）：
```mlir
%50 = ascendc.que_bind.alloc_tensor %4   // VECOUT alloc
ascendc.duplicate_l2 %50, %cst, ...     // 零初始化（不必要）
ascendc.que_bind.enque_tensor %4, %50   // enque，队列满
```

**第 156–158 行**（for_Tb_N 内，max）：
```mlir
%67 = ascendc.que_bind.alloc_tensor %4   // VECOUT 再次 alloc，队列已满
ascendc.max_l2 %67, ...
ascendc.que_bind.enque_tensor %4, %67
```

VECOUT queue 容量为 1，prologue enque 后满了，for_Tb_N 内无法再 alloc。

另外第 163 行（for_Tb_M/Tb_N 循环**外**）：
```mlir
%51 = ascendc.que_bind.deque_tensor %4   // VECOUT deque
```

这是在整个 for_Tb_M/Tb_N 结束后才 deque，意味着每次 for_Tb_N 迭代产生的 VECOUT 结果都在队列里积压（但队列容量只有 1，早就满了）。

**正确模式**：
- 删除 prologue 的 VECOUT alloc/duplicate/enque
- for_Tb_N 内 max 完成后 enque，**紧接着在同一循环层**（或 epilogue）deque → data_copy_l2 → free
- 或者把 deque/copy/free 移到 for_Tb_N 循环内

---

## 问题 4：bias VECIN 在 for_Tb_N 内每次迭代 deque，只 enque 一次（严重）

**第 83 行**：bias VECIN（`%3`）在 for_TB_N prologue enque 一次。  
**第 144 行**：for_Tb_N 内每次迭代都 `deque_tensor %3`。  
**第 149 行**：每次迭代后 `free_tensor %3`。

for_Tb_N 有多次迭代，第一次 deque 后 `%3` 空了，后续迭代 deque 会阻塞。

bias 在整个 for_Tb_M/Tb_N 过程中是**只读**的，正确做法：
- bias deque 一次，保存在循环外的 Value
- 循环内用 `subview` 取当前 tile 对应的 slice
- 整个 for_Tb_M/Tb_N 结束后才 free

---

## 问题 5：`nd2nz_params` 参数可能有误（中等）

**第 58 行**（A 矩阵的 nd2nz_params）：
```mlir
%37 = ascendc.construct !ascendc.nd2nz_params(
    %34,      // %27 as i16：TB_M（行数）
    %35,      // %33 as i16：dim_1/16（K/16，列块数）
    %34,      // %27 as i16：重复？
    %36,      // dim_1 as i16：K（列数）
    %34, %c0_i16, %34, %c0_i16)
```

对照 AscendC `DataCopyParams`（nd2nz 使用）的字段：`blockCount, blockLen, srcStride, dstStride`（或类似）。

样例的 `CopyND2NZ` 里：
```cpp
DataCopy(dst[dstOffset], src[srcOffset],
    { height, 1, uint16_t(width/16 - 1), 0 });
// { blockCount=height, blockLen=1, srcStride=width/16-1, dstStride=0 }
```

当前 IR 传入了 8 个参数，需要对照 `nd2nz_params` 的 td 定义确认每个字段的语义，特别是 `%34`（TB_M）重复出现了 4 次，可能有字段填错。