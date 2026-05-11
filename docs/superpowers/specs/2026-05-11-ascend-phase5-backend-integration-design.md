# Ascend Phase 5 Backend Integration Design

## Goal

Phase 5 将 Phase 3B 产出的 memory-realized IR 对接到现有 AscendC / CANN 后端，形成可验证的 Translate / Runtime Artifact 链路。

本阶段不把已有后端代码机械复制到 `Conversion/Ascend` 后再删除旧目录。正确路径是先建立 `Conversion/Ascend/Backend` 正式边界，显式化支持矩阵和失败诊断，复用现有可工作的 lowering / ABI / translation 能力，再按测试覆盖逐步迁移旧入口。

## Current State

已有可复用能力：

- `lib/Conversion/LinalgToAscendC`：将部分 `linalg` / `memref.copy` 降到 `ascendc` dialect。
- `lib/Conversion/AscendCBufferPlacement`：原型 placement / DataCopy 插入逻辑。
- `lib/Conversion/AscendCParallelize`：原型 block dispatch lowering。
- `lib/Conversion/AscendCPrepareForEmit`：生成 kernel 侧 `TilingData` / emitasc 准备。
- `lib/Conversion/CanonicalizeCannSignature`：规整 CANN 标准 kernel signature。
- `lib/Target/CannKernel` 与 `tools/afir-translate`：把 CANN signature MLIR 翻译为 AscendC C++，并已有 `tiling_space.json` skeleton 输出。

当前缺口：

- 支持范围隐含在代码分支中，没有统一 support matrix。
- V2-6 中的 `ComputeLoweringDriver`、`BackendABILoweringDriver`、`HostTilingEmitter`、`RuntimeManifestBuilder` 没有正式 Ascend pipeline 边界。
- `tiling_space.json` 仍是 skeleton，不满足 V2-6 `schema_version = "2.0"` 规范。
- Runtime manifest 和 host tiling C ABI 产物还没有编译器生成路径。
- 旧 pass 名称仍是原型路径，不能作为 Phase 5 的最终用户入口。

## Decision

采用“封装、收敛、迁移、删除”的策略。

1. 在 `include/Conversion/Ascend/Backend` 和 `lib/Conversion/Ascend/Backend` 下新增 Phase 5 正式边界。
2. Phase 5 正式 pass 先复用现有后端实现，不复制一份逻辑。
3. 新增 `AscendBackendSupportMatrix`，把现有特殊判断变成可测试的支持声明。
4. 新增 verifier / diagnostic，所有 unsupported op、unsupported memory path、unsupported ABI shape 必须明确失败，不允许静默跳过。
5. 旧原型入口先保留，等新入口覆盖同等测试并通过 review 后，再单独计划迁移或删除。

## Alternatives Considered

### Approach A: Copy backend code into `Conversion/Ascend` and delete old code

不采用。短期看目录整齐，但会引入重复逻辑、测试迁移风险和 review 噪声。现有后端包含 translator、dialect lowering、ABI 处理和运行时工具链假设，直接搬迁容易破坏已有测试。

### Approach B: Keep old backend untouched and only在文档中标注

不采用。这样 Phase 5 仍然没有正式代码边界，后续 pipeline 入口、support matrix、runtime artifact 都会继续散落在旧原型路径里。

### Approach C: Add Ascend backend boundary and reuse old implementation

采用。该方案最小化重写风险，同时把 Phase 5 的职责和验收标准收敛到新边界上。旧实现作为内部能力被复用，外部入口逐步迁到 `ascend-*` 命名。

## Architecture

Phase 5 拆成四个可独立 review 的单元：

1. `AscendBackendSupportMatrix`
   - 声明支持的 compute op、movement path、memory space、ABI 形态、translation artifact。
   - 提供 verifier 可复用的查询接口。
   - 所有未声明支持的组合默认失败。

2. `ComputeLoweringDriver`
   - 负责 Phase 5 compute lowering 边界。
   - 首轮复用 `linalg-to-ascendc` 的实际 lowering 能力。
   - lowering 后验证所有受支持 kernel 内不再残留应被 lowering 的 `linalg` / movement op。
   - 不做新的 schedule / placement / memory decision。

3. `BackendABILoweringDriver`
   - 对齐 V2-6 的 ABI lowering 边界。
   - 首轮复用 `ascendc-parallelize`、`ascendc-prepare-for-emit`、`canonicalize-cann-signature` 能力。
   - 输出满足 CANN 标准签名的 `AscendC Kernel MLIR`。
   - 验证 `cann.num_inputs`、workspace 参数、tiling py_struct 和字段顺序。

4. Runtime artifact emitters
   - `HostTilingABI` 从 CANN signature MLIR 提取 kernel ABI。
   - `HostTilingEmitter` 生成 host tiling C ABI source。
   - `TilingSpaceExporter` 输出 V2-6 `schema_version = "2.0"` 的 `tiling_space.json`。
   - `RuntimeManifestBuilder` 输出静态 shape / 单 kernel MVP 的 `runtime_manifest.json`，动态 guard 和多 kernel DAG 先显式报 unsupported 或输出保守 manifest。

## Pipeline Surface

新增或对齐的用户入口：

```text
--ascend-compute-lower
--ascend-parallelize
--ascend-prepare-for-emit
--ascend-canonicalize-cann-signature
afir-translate -mlir-to-cann \
  --tiling-space-out=<path> \
  --runtime-manifest-out=<path> \
  --host-tiling-out=<path>
```

旧入口仍保留：

```text
--linalg-to-ascendc
--ascendc-parallelize
--ascendc-prepare-for-emit
--canonicalize-cann-signature
```

旧入口的删除不属于本设计的首轮实现范围。删除条件是新入口覆盖同等测试、文档和 review 结论。

## Support Matrix MVP

首轮支持范围按现有代码事实收敛：

| 类别 | MVP 支持 | Unsupported 行为 |
|---|---|---|
| Data movement | `GM->A1/B1`、`GM->VECIN`、`A1->A2`、`B1->B2`、`CO1->VECIN`、`VECOUT->GM` | pass 失败并报告 source/dest memory space |
| Compute | supported `linalg.matmul`、fill、elementwise add/max/relu、部分 reduction | pass 失败并报告 op name 与 operand memory spaces |
| Memory space | 现有 AscendC TPosition integer memory space | 未知 memory space 失败 |
| ABI | 单个 `func.func` CANN kernel signature，`cann.num_inputs`，workspace `memref<ui8>`，tiling `!emitasc.py_struct` | ABI verifier 失败 |
| Runtime artifact | 静态 shape、单 kernel、字段顺序与 py_struct 一致 | 动态 guard / 多 kernel DAG 首轮显式 unsupported，后续增强 |

该 matrix 是 Phase 5 的验收核心：支持项必须有 positive test，不支持项必须有 negative diagnostic test。

## Runtime Artifact MVP

`tiling_space.json` 输出必须满足 V2-6 schema：

- `schema_version = "2.0"`
- `kernel`
- `soc`
- `block_dim_expr`
- `workspace_size_expr`
- `tiling_params`
- 静态 shape 时可输出 `shapes`
- `fixed: true` 字段来自 tiling py_struct 中的 shape 参数，不从其他位置猜测

`runtime_manifest.json` 首轮输出静态 shape / 单 kernel schema：

- `kernelName`
- `shapeBucketKey`
- `guardSet`
- `tilingSchema`
- `scheduleEntries`
- `abiSignature`
- `cacheKey`
- `workspaceSizeExpr`
- `workspaceSizeBytes`
- `shapeArgOrder`
- `kernelGraph`

`HostTilingEmitter` 首轮生成 C ABI source，而不是在 compiler pass 内直接调用外部编译器生成 `.so`。`.so` 打包可以作为后续工具链集成任务，但 C ABI source 的内容和符号必须按 V2-6 固定。

## Error Handling

- 新 Ascend backend verifier 默认 fail-closed。
- 不支持的 op / memory path / ABI 形态必须 `emitError`，诊断中包含 op 名称、kernel 名称和关键 memory space / ABI 字段。
- 旧原型 pass 中原本“跳过不转换”的路径，在新入口下必须由 wrapper verifier 捕获。
- Runtime artifact 生成失败时不得只打印 warning；新入口必须返回 failure。

## Testing

首轮测试分四层：

1. Support matrix unit tests
   - 支持路径返回 supported。
   - 未声明路径返回 unsupported reason。

2. Conversion LIT
   - 新 `--ascend-compute-lower` 输出与现有 `--linalg-to-ascendc` 等价。
   - negative case 检查 unsupported memory path 报错。
   - 新 ABI wrapper pass 输出 CANN signature。

3. Target translation LIT
   - `tiling_space.json` 满足 `schema_version = "2.0"`。
   - `runtime_manifest.json` 输出静态单 kernel manifest。
   - `host_tiling.cpp` 输出 C ABI 符号。

4. Integration smoke
   - 使用 `examples/transformer/transformer_dynamic.mlir` 作为最终验收场景之一。
   - Phase 5 首轮只要求明确报告支持到哪一层；未支持的 transformer 子图必须有清晰 unsupported 诊断，不允许静默成功。

所有编译和测试在 xvm/docker 中执行，主机只做代码开发。

## Tracking And Review

Phase 5 每个任务需要完成：

- 计划项更新到 `docs/Ascend-MLIR-V2-Implementation-Tracking.zh.md`
- spec compliance review
- code quality review
- xvm/docker focused build
- focused LIT / unit tests
- code naming guard `test/tools/check_ascend_no_v2_code_naming.sh`
- `git diff --check -- . ':!AGENTS.md'`

## Non-Goals

- 不在首轮删除旧 backend 目录。
- 不重写 `lib/Target/CannKernel` translator。
- 不在 Phase 5 中新增 schedule / placement / memory planning 决策。
- 不把 Phase 3B deferred 的 full workspace/subview/copy materialization 强行塞进 Phase 5。
- 不承诺首轮支持完整 transformer 动态 shape 后端代码生成；首轮目标是支持矩阵内可成功，不支持矩阵外明确失败。
