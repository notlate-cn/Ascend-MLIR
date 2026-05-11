# Ascend Phase 5C Full Pipeline Bridge Report

## 结论

最小普通用例已经可以通过完整 `Phase 0 -> Phase 5` 新主线：

```bash
afir-opt ordinary.mlir \
  --ascend-normalize \
  --ascend-kernelize \
  --ascend-schedule \
  --ascend-realize='materialization-mode=memory-space-annotate' \
  --ascend-compute-lower
```

target-aware 链路也已通过：

```bash
afir-opt ordinary.mlir \
  --ascend-normalize \
  --ascend-kernelize \
  --ascend-schedule \
  --ascend-realize='placement-mode=target-aware cann-root=<cann-root> soc=<soc> materialization-mode=memory-space-annotate' \
  --ascend-compute-lower \
  --ascend-parallelize \
  --ascend-prepare-for-emit \
  --ascend-canonicalize-cann-signature
```

对应测试：

- `test/Conversion/ascend-full-pipeline-ordinary-smoke.mlir`

## 已关闭的断点

此前断点在 `ascend-realize` 输出与 `ascend-compute-lower` 输入契约之间：

- `ascend-realize` 只把 tensor IR bufferize 成默认 GM memref。
- Phase 5 parallel generic lowering 要求 output 是 on-chip memory space。
- 因 output 仍是 GM，`ascend-compute-lower` 会跳过 lowering 并 fail-closed。

本轮 bridge 在 `memory-space-annotate` 模式下为最小普通 vector output 做 value-level materialization：

- 保留 one-shot bufferize 生成的 GM result alloc，作为函数返回 / ABI output。
- 为 Phase 5 支持的最终 vector output 新建 `VECOUT` alloc。
- 将 `linalg` output init 改写到 `VECOUT` alloc。
- 在 `linalg` 后插入 `memref.copy`，表示 `VECOUT -> GM` epilogue。
- Phase 5 现有 `DataMoveConversion` 将该 copy 降为 `ascendc.data_copy_l2`。

## 范围边界

本轮只覆盖保守 MVP：

- all-parallel vector output
- `linalg.generic` body 中只包含 Phase 5 当前支持的 `arith.addf` / `arith.mulf` / `arith.maximumf` / `arith.constant`
- `linalg.elementwise` 的 add / mul / max
- output alloc 只允许当前 writer 和外部 return/cast 使用
- 不桥接跨 kernel temporary
- 不桥接 reduction、matmul、gather、transpose 或动态多 kernel DAG

## 仍需增强

完整 workspace/copy materialization 仍是后续增强：

1. 为 workspace slot / subview 生成真实 value-level IR。
2. 为 `GM -> VECIN/A1/B1` 预取路径生成显式 copy，而不是依赖 Phase 5 compute lowering 内部临时 copy。
3. 支持跨 kernel value handoff。
4. 支持 reduction / matmul / gather / transpose 的 output bridge。
5. 将普通 demo 从旧式 pre-lowered 路径迁移到完整 Phase 0 -> Phase 5 脚本。
