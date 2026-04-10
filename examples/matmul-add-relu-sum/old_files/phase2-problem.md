## 问题 1：所有 TBuf/GlobalTensor 都是新创建的，与实际 buffer 断开（严重）

这是延续上一版讨论的核心问题，现在在输出 IR 里看得非常清楚。

以 A1→A2 的搬运为例（第 47–52 行）：

```mlir
%49 = ascendc.tbuf : <a1>          // ← 新创建的空 A1 TBuf
%50 = ascendc.tbuf.get_tensor %49  // ← 从新 TBuf 取 tensor
%51 = ascendc.tbuf : <a2>          // ← 新创建的空 A2 TBuf
%52 = ascendc.tbuf.get_tensor %51  // ← 从新 TBuf 取 tensor
ascendc.data_copy_l0 %52, %50, %53
```

同时第 59–64 行 matmul 又创建了另一套新的 A2/B2/CO1 TBuf：

```mlir
%59 = ascendc.tbuf : <a2>   // ← data_copy_l0 的目标是 %52，但 matmul 读的是 %60
%60 = ascendc.tbuf.get_tensor %59
```

`data_copy_l0` 往 `%52` 写，`matmul.set_tensor_a` 读 `%60`——**两个完全不同的 TBuf 对象，没有任何数据流连接**。每次 `ascendc.tbuf` 在 AscendC 语义里都是声明一个新的 buffer 变量，所以数据实际上没有从 A1 流向 matmul。

同样的问题出现在：
- 第 30–34 行的 VECIN(bias) 和第 77–80 行 add 的 ins[1] VECIN 是两个不同的 TBuf
- 第 69–72 行 fixpipe 的 dst VECIN 和第 37–40 行 add 的 ins[0] VECIN 是两个不同的 TBuf
- 第 43–44 行的 CO1 和第 63–64 行 matmul 的 CO1 是两个不同的 TBuf

**根本原因**：`memref.copy` 有明确的 src/dst operand 连接（`copy %src, %dst`），但 pass 在把 copy 转换为 `data_copy_*` 时，用 `memrefToLocalTensor` 为 src 和 dst 分别创建了新的 `tbuf`，而不是复用 Phase 1 已经 alloc 的那个 buffer 对应的 tbuf。

**修复方向**：需要在 Phase 1 的 `memref.alloc` → Phase 2 的 `ascendc.tbuf` 之间建立一对一的映射。具体做法是在 pass 入口维护一个 `DenseMap<Value, Value> allocToTbuf`，遍历所有 `memref.alloc` 并提前创建对应的 tbuf，后续所有 pattern 通过查表复用，而不是重新创建。

---

## 问题 2：matmul 使用 `iterate_all` 在 for_K 内部（严重）

**第 67 行**在 `for_K`（第 46–68 行）内部调用了 `matmul.iterate_all`。

`iterate_all` 的语义是"一次性完成 K 轴所有迭代"，它内部隐含了一个完整的 K 循环。当前 IR 结构是外层 `scf.for %K` 循环 + 内层 `iterate_all`，等于把 K 轴迭代做了两次嵌套，语义重复且错误。

正确选择是二选一：
- **保留 `scf.for %K` 循环，内部用 `matmul.iterate`**（每次 K 迭代调用一次）
- **去掉 `scf.for %K`，用 `matmul.iterate_all`**（状态机自己管理 K 循环）

对于当前流水线，for_K 是 Phase 1 已有的循环结构，建议保留并改用 `matmul.iterate`。

---

## 问题 3：add 的两个 ins 都是新创建的 VECIN TBuf（严重）

**第 75–81 行**：

```mlir
%37 = ascendc.tbuf : <vecin>   // ← add 的 ins[0]：应该是 fixpipe 写入的 VECIN
%38 = ascendc.tbuf.get_tensor %37
%39 = ascendc.tbuf : <vecin>   // ← add 的 ins[1]：应该是 bias VECIN
%40 = ascendc.tbuf.get_tensor %39
ascendc.add_l2 %36, %38, %40, %27
```

两个 VECIN 都是新创建的空 TBuf，既没有连接到 fixpipe 的输出，也没有连接到 bias 的 data_copy_l2 的目标。这是问题 1 在 add op 上的具体体现。

---

## 问题 4：`duplicate_l2` 用于零初始化 CO1 和 VECOUT 但单位不对（中等）

**第 37 行**：

```mlir
ascendc.duplicate_l2 %21, %cst, %19 : ..., f32, index
// %19 = arith.muli %6, %7  ← TB tile 的元素个数
```

**第 45 行**：

```mlir
ascendc.duplicate_l2 %29, %cst, %27 : ..., f32, index
// %27 = arith.muli %25, %26  ← tb tile 的元素个数
```

`duplicate_l2` 的 count 参数通常以 **repeat 次数**（BLK 为单位）计量，而不是元素个数。这里传入的是元素个数（index 类型），需要确认 `DuplicateL2Op` 的 td 定义里 count 的语义究竟是元素个数还是 BLK 数。

如果是 BLK 数，则需要除以向量宽度（64 for f32）：

```mlir
%blk_size = arith.constant 64 : index
%repeat = arith.ceildivui %27, %blk_size : index
ascendc.duplicate_l2 %29, %cst, %repeat
```

---

## 问题 5：`data_copy_l2` 的 count 语义同上（中等）

第 24、29、34、97 行的 `data_copy_l2` 传入的 count 也是元素个数（index 类型）。同样需要确认 `DataCopyL2Op` 期望的是元素个数还是字节数/BLK数。

AscendC 的 `DataCopy` API 接受的 count 是**元素个数**（`calCount` 参数），所以这里用元素个数是正确的，但类型需要是 `uint32`，不是 `index`。

建议统一加一个 `arith.index_cast ... : index to i32` 转换。

---

## 问题 6：`data_copy_l0` 共用同一个 `%53`（DataCopyParams）（低）

**第 52 和 57 行**：

```mlir
%53 = ascendc.construct !ascendc.data_copy_params()
ascendc.data_copy_l0 %52, %50, %53   // A1→A2
ascendc.data_copy_l0 %57, %55, %53   // B1→B2，复用同一个 %53
```

A→A2 和 B1→B2 的 DataCopyParams 可能需要不同的配置（stride、transposed 等），共用一个 params 对象存在风险。如果当前默认构造参数对两者都适用，可以接受，但建议分别创建：

```mlir
%53_a = ascendc.construct !ascendc.data_copy_params()
ascendc.data_copy_l0 %52, %50, %53_a
%53_b = ascendc.construct !ascendc.data_copy_params()
ascendc.data_copy_l0 %57, %55, %53_b
```

---


**核心问题只有一个**：`memrefToLocalTensor` 每次都创建新的 TBuf，导致所有 buffer 的数据流在 ascendc dialect 层面完全断开。解决了这个问题，其他问题大部分会自然消失。