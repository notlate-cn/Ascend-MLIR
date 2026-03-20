# prepare-for-emit view chain 通用化方案

## 问题

`AscendCPrepareForEmitPass` 的 7b 步骤负责将 `memref.subview` + `set_global_buffer` 展平为 flat pointer + offset，以消除 ascir-translate 不认识的 memref op。

当前 7b 的走链逻辑只能穿过 `memref.subview` 和 `memref.cast`，遇到 `expand_shape`、`collapse_shape`、`transpose` 等 view op 就会中断，导致这些 op 残留到 ascir-translate 阶段报错。

此外，1D 和 2D subview 链分成两条独立代码路径（`resolveSubviewChain1D` 和 2D 分支），col stride 的获取依赖 TilingData 间接查找，对 promoted alloc 等场景存在盲区。

### 受影响的 case

| case | 失败原因 |
|------|---------|
| test_add | 2D promoted alloc，col stride 查不到 |
| test_broadcast_add_reduce | expand_shape 阻断走链 |
| 未来含 transpose/collapse_shape 的 case | 同理 |

## 已完成的单点修复

1. **7a stride 修复**：`strides(rank, 1)` → `strides(rank, kDynamic)`，修复 promoted alloc 的 `memref.cast` 类型不兼容问题。
2. **7b/7c stride fallback**：当 TilingData 和 promotedArgDynSizes 都查不到 col stride 时，从 subview 的 type 读 `strides[0]` 作为 fallback。

这两个修复解决了 test_add 的问题，但 expand_shape 等 view op 的走链中断问题仍未解决。

## 方案 B：基于 stride 的统一 flat offset 计算

### 核心思想

不区分 1D/2D，不依赖 TilingData 查 stride。在走链过程中，每遇到一个 subview，直接从其 source type 的 stride 计算 flat offset 贡献；其他 view op 一律透传。

### 走链规则

| op 类型 | 内存偏移 | 处理方式 |
|---------|:-------:|---------|
| `memref.subview` | 有 | `flat_offset += sum(offset[i] * src_stride[i])` |
| `memref.cast` | 无 | 透传：`cur = op.getSource()` |
| `memref.expand_shape` | 无 | 透传：`cur = op.getSrc()` |
| `memref.collapse_shape` | 无 | 透传：`cur = op.getSrc()` |
| `memref.transpose` | 无 | 透传：`cur = op.getSource()` |
| `memref.reinterpret_cast` | 无 | 透传：`cur = op.getSource()` |
| `BlockArgument` | — | 到达 root，结束 |

### 为什么 view op 可以透传

所有 view op（expand_shape、collapse_shape、transpose、cast 等）都只是重新解释同一块内存的 shape/stride，不移动指针。stride 的变化由 MLIR 类型系统自动传播到上层 subview 的 source type 中。方案 B 在 subview 层读 source type 的 stride 时，自然拿到了经过 view op 变换后的正确值。

### 伪代码

```cpp
auto resolveViewChain = [&](Value start, OpBuilder &b,
                            Location loc) -> std::pair<BlockArgument, Value> {
    Value cur = start;
    Value flatOffset = b.create<arith::ConstantIndexOp>(loc, 0);

    while (true) {
        if (auto sv = cur.getDefiningOp<memref::SubViewOp>()) {
            // subview 贡献 offset：sum(offset[i] * src_stride[i])
            auto srcTy = cast<MemRefType>(sv.getSource().getType());
            auto [strides, _] = srcTy.getStridesAndOffset();
            auto offsets = sv.getMixedOffsets();
            for (unsigned i = 0; i < offsets.size(); ++i) {
                Value off = materialize(offsets[i]);
                if (strides[i] != ShapedType::kDynamic) {
                    Value stride = b.create<arith::ConstantIndexOp>(loc, strides[i]);
                    Value contrib = b.create<arith::MulIOp>(loc, off, stride);
                    flatOffset = b.create<arith::AddIOp>(loc, flatOffset, contrib);
                } else {
                    // dynamic stride: 从 TilingData 或其他途径获取（保留现有 fallback）
                }
            }
            cur = sv.getSource();
            continue;
        }
        // 所有 view op 透传
        if (auto op = cur.getDefiningOp<memref::CastOp>()) { cur = op.getSource(); continue; }
        if (auto op = cur.getDefiningOp<memref::ExpandShapeOp>()) { cur = op.getSrc(); continue; }
        if (auto op = cur.getDefiningOp<memref::CollapseShapeOp>()) { cur = op.getSrc(); continue; }
        if (auto op = cur.getDefiningOp<memref::TransposeOp>()) { cur = op.getSource(); continue; }
        if (auto op = cur.getDefiningOp<memref::ReinterpretCastOp>()) { cur = op.getSource(); continue; }
        // 到达 root
        if (auto ba = dyn_cast<BlockArgument>(cur))
            return {ba, flatOffset};
        return {BlockArgument{}, Value{}}; // 无法解析
    }
};
```

### 正确性验证

**test_add**（2D → 2D）：

```
subview %cast[%17, 0]       src strides=[64,1]  → %17*64 + 0*1 = %17*64
subview %subview[%arg4, 0]  src strides=[64,1]  → %arg4*64 + 0*1 = %arg4*64
cast → 透传
%arg3 → root

flat_offset = (%arg4 + %17) * 64  ✓
```

**test_broadcast_add_reduce**（1D → expand → 2D）：

```
subview %expand[%10, 0]     src strides=[1,1]   → %10*1 + 0*1 = %10
subview %subview[%arg4, 0]  src strides=[1,1]   → %arg4*1 + 0*1 = %arg4
expand_shape → 透传
%arg0 → root

flat_offset = %arg4 + %10  ✓
```

**test_sigmoid_mul_reduce**（1D → 1D）：

```
subview %alloc[%10] [%12] [1]   src strides=[1]  → %10*1 = %10
subview %subview[%arg4] [%n] [1] src strides=[1]  → %arg4*1 = %arg4
cast → 透传
%arg3 → root

flat_offset = %arg4 + %10  ✓
```

### 改动范围

| 位置 | 改动 |
|------|------|
| 7b `setGlobalBufferOps` 循环 (L375-463) | 删除 1D/2D 分支，替换为统一的 `resolveViewChain` |
| 7c `resolveGMChain` lambda (L485-534) | 同上 |
| `resolveSubviewChain1D` (L344-366) | 删除（不再需要） |

### 与现有修复的关系

- **7a stride 修复**（`kDynamic`）：保留，确保 promoted alloc 的 `memref.cast` 合法
- **7b stride fallback**（从 type 读 stride）：被方案 B 吸收——方案 B 本身就是从 type 读 stride，不再需要单独的 fallback