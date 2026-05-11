# Ascend Phase 5C Full Pipeline Gap Report

## 结论

普通用例当前可以通过 Phase 5 官方后端入口跑通旧式 demo 路径，但还不能直接通过完整 `Phase 0 -> Phase 5` 新主线。

已固定的完整链路 smoke：

```bash
afir-opt ordinary.mlir \
  --ascend-normalize \
  --ascend-kernelize \
  --ascend-schedule \
  --ascend-realize='materialization-mode=one-shot-bufferize' \
  --ascend-compute-lower
```

target-aware memory-space annotate 链路也已固定为同一断点：

```bash
afir-opt ordinary.mlir \
  --ascend-normalize \
  --ascend-kernelize \
  --ascend-schedule \
  --ascend-realize='placement-mode=target-aware cann-root=<cann-root> soc=<soc> materialization-mode=memory-space-annotate' \
  --ascend-compute-lower
```

当前预期失败：

```text
error: ascend-compute-lower left a lowerable operation behind
```

对应测试：

- `test/Conversion/ascend-full-pipeline-gap-smoke.mlir`

## 断点

断点在 `ascend-realize` 输出与 `ascend-compute-lower` 输入契约之间。

`ascend-realize` 的 `one-shot-bufferize` / `memory-space-annotate` 当前只完成：

- tensor -> memref bufferization
- 已证明 vector temporary 的 memory space annotation

它还没有完成：

- value-level workspace alloc/subview materialization
- GM -> VECIN / A1 / B1 显式 copy materialization
- VECOUT / CO1 -> GM epilogue copy materialization
- ordinary output buffer 的 on-chip memory space 写入

因此进入 `ascend-compute-lower` 时，普通 `linalg.generic` 的输出仍是默认 GM memref。Phase 5 复用的 AscendC lowering 对 parallel generic 和 elementwise lowering 都要求输出是 on-chip memory space；GM output 会被保守跳过，随后 fail-closed verifier 报告残留 `linalg.generic`。

## 代码证据

- `lib/Conversion/Ascend/Realize/RealizePass.cpp`：`materialization-mode=one-shot-bufferize` 只调用 One-Shot Bufferize；`memory-space-annotate` 只追加 memory space annotation。
- `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`：parallel generic lowering 在 output memory space <= 0 时跳过。
- `lib/Conversion/LinalgToAscendC/ComputeConversion.cpp`：elementwise lowering 同样要求 output memory space > 0。
- `lib/Conversion/Ascend/Backend/ComputeLoweringPass.cpp`：lowering 后若仍残留 `memref.copy` 或 `linalg` op，直接 fail-closed。

## 验收影响

当前阶段可以验收：

- Phase 0 -> Phase 3B：Normalize / Kernelize / Schedule / Realize MVP 链路与 plan/materialization slice。
- Phase 5：正式 backend 入口、support matrix、ABI lowering、runtime artifact emitters。
- 普通 demo：通过旧式前处理 + Phase 5 官方入口跑通 runtime-session sim。
- 完整 Phase 0 -> Phase 5 普通用例：已有 expected-fail smoke 固定断点，尚未 positive。

下一阶段要把该 smoke 从 expected-fail 改为 positive，需要补齐 Realize 到 Phase 5 的 value-level materialization bridge。

## 下一步桥接目标

1. 在 `MemoryRealizationDriver` 中为普通 vector elementwise 生成 value-level placement result。
2. 为 GM inputs 物化 GM -> VECIN copy，为 output 物化 VECOUT -> GM copy。
3. 为中间/输出 buffer 写入 Phase 5 可识别的 memory space。
4. 将 `ascend-full-pipeline-gap-smoke.mlir` 改为 positive LIT，并补充普通 full-pipeline demo 命令。
