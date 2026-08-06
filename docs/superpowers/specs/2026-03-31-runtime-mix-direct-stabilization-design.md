# RuntimeMix Direct Backend Stabilization Design

**Date:** 2026-03-31

## Goal

把 `RuntimeMix` 的 direct backend 从“`baremix` 样例已跑通的原型”收敛成“可以并回 `lib/Runtime` 的稳定后端”。本阶段不追求支持任意 host 参数 ABI，也不直接启动并回动作；重点是把当前已经验证通过的 direct pipeline 固化成可重复、可定位、可扩展的实现。

稳定化的验收标准是：

- 在 xvm 上，`examples/baremix-test/run.sh` 连续重跑稳定通过
- `mix-compiler` / `mix-validator` 的行为与产物契约明确，不再依赖隐式 fallback
- direct backend 的核心阶段边界清晰，错误能定位到具体 stage
- 当前实现不绑定 `baremix` 图结构，只绑定一套明确的 mix host ABI
- 代码结构已经按未来并回 `lib/Runtime` 的方向组织

## Scope

### In Scope

- 固化 `RuntimeMix` direct backend 的阶段边界与产物契约
- 固化 `mix-validator` 优先使用 `mix_runner` 的执行路径
- 把当前实现中的 `baremix` 专用假设收敛为显式 ABI 约束
- 为后续并回 `lib/Runtime` 定义文件级映射和替换策略
- 增加最小但稳定的 xvm 回归入口

### Out of Scope

- 本阶段不新增用户可见 CLI 参数
- 本阶段不实现“从源码自动推导任意 host 参数 ABI”
- 本阶段不统一 vec/cube/mix 三类后端
- 本阶段不直接删除 `RuntimeMix`
- 本阶段不把代码并回 `lib/Runtime`

## Current State

当前 direct backend 已经能在 xvm 上完成：

1. `bisheng -E` preprocess
2. toolkit `extract_host_stub.py` / `update_host_stub.py`
3. AIC/AIV compile
4. merge device object
5. `ascendc_pack_kernel`
6. 产出 packed `.so` 和 `mix_runner`
7. `mix-validator` 调 `mix_runner` 跑 sim
8. `verify_result.py` 过精度

这证明架构主路径成立，但距离“稳定后端”仍有三个问题：

1. 某些行为仍然依赖样例路径和当前调试经验，而不是清晰契约
2. host ABI 约束没有被明确建模，只是散落在 runner 和 validator 中
3. `RuntimeMix` 与 `lib/Runtime` 的收敛边界还不够清楚，继续开发容易长出平行逻辑

## Recommended Approach

建议按四个阶段完成稳定化，而不是直接开始“更泛化的图结构支持”。

### Approach A: 继续直接泛化到更多样例

优点：

- 表面上进展快
- 可以更早暴露通用性问题

缺点：

- 当前 ABI 和 artifact 边界还没固化
- 一旦第二个样例失败，很难判断是 pipeline 问题还是 ABI 问题
- 容易把临时兼容逻辑直接焊死在后端里

### Approach B: 先做稳定化，再泛化（推荐）

优点：

- 先把“已成立的 baremix 路径”收成稳定后端
- 后面接第二个样例时，更容易区分“实现 bug”和“接口边界不够”
- 最符合“将来并回 `lib/Runtime`”的目标

缺点：

- 短期内不会立刻增加更多图结构 coverage

### Approach C: 直接开始并回 `lib/Runtime`

优点：

- 不会长期保留 `RuntimeMix`

缺点：

- 现在并回会和主线并行修改互相干扰
- 会把尚未稳定的 ABI 和实现细节直接扩散到正式 runtime

结论：先走 Approach B。

## Design

### 1. Functional Stability

把当前闭环从“偶然跑通”变成“固定入口、固定行为、固定产物”。

要求：

- `mix-compiler` 对同一输入重复运行，产物布局稳定
- `mix-validator` 对 direct artifact 必须优先走 `mix_runner`
- direct packed fallback 只保留为显式受限的兜底路径，不能再抢主路径
- `examples/baremix-test/run.sh` 作为固定 acceptance 入口

这一步不增加新能力，只减少行为上的不确定性。

### 2. Structural Stability

把 `MixDirectBackend` 收敛成清晰的阶段式 pipeline。

建议的内部阶段为：

1. source analyze
2. preprocess
3. generated-config parse
4. device compile
5. device merge
6. host stub finalize
7. pack
8. artifact assemble

要求：

- 每个阶段有明确输入输出
- manifest/debug 输出反映真实路径，而不是推测值
- 任何工具失败时，错误信息至少包含 stage 名、命令和关键输入路径

这一步的目标不是追求抽象优雅，而是确保问题能被快速定位。

### 3. ABI Stability

本阶段明确支持的“通用 mix host ABI”是：

- GM inputs
- GM outputs
- workspace
- tiling

这里的“通用”含义是：

- 不绑定 `baremix` 的内部算子图
- 不绑定某个固定 mix 结构
- 但仍要求 kernel 的 host 交互符合 AscendC 当前这套通用 launcher ABI

为了避免把限制散落在代码里，约束要收敛到 artifact/runner 配置层。当前已知约束包括：

- 输入个数
- 输出个数
- 输出 dtype / shape
- tiling 字节来源与布局
- workspace 大小

这些约束本阶段可以仍然由样例提供或由 artifact 描述，但不能继续硬编码在多个位置。

### 4. Mergeability

`RuntimeMix` 只是隔离开发区，因此本阶段必须明确未来的并回映射。

| Current `RuntimeMix` area | Future merge target |
| --- | --- |
| `RuntimeMix/MixDirectBackend.*` | `Runtime/Compiler.*` `mix` branch |
| `RuntimeMix/MixCommandBuilder.*` | helper layer under `Runtime/Compiler.*` or adjacent runtime mix command-builder split |
| `RuntimeMix/MixSourceAnalyzer.*` | source-analysis portion of `Runtime/Compiler.*` mix branch |
| `RuntimeMix/MixArtifact.h` | `Runtime` artifact/runner config model |
| `RuntimeMix/Executor.*` | `Runtime/Executor.*` |
| `RuntimeMix/NpyIO.*` | `Runtime/NpyIO.*` |
| `RuntimeMix/TilingSchema.*` | `Runtime/TilingSchema.*` or equivalent mix runner metadata layer |
| `RuntimeMix/Compiler.*` | fold into `Runtime/Compiler.*` as the public mix compile entry |
| `RuntimeMix/HostRunnerGen.*` | `Runtime/HostRunnerGen.*` |
| `tools/mix-validator` runner-first path | `Runtime/SimValidator.*` and `tools/validator` |
| `tools/mix-compiler` CLI shell | `tools/compiler` once the direct mix backend replaces the temporary entrypoint |

禁止事项：

- 不再把新的核心逻辑只写进 `tools/mix-compiler`
- 不再把新的核心逻辑只写进 `examples/baremix-test/run.sh`

目标状态是“核心逻辑尽量留在 `RuntimeMix` 库层，工具只做薄壳”。当前仍存在一个已知例外：`tools/mix-validator` 还保留运行时环境探测、direct fallback 和 baremix 路径相关逻辑，这部分属于后续真正并回 `Runtime` 时仍要继续收敛的遗留项，不视为本阶段已经完成。

## File-Level Direction

本阶段建议重点收口这些文件：

- `include/RuntimeMix/MixArtifact.h`
  明确 direct backend 的稳定产物字段和可空约束

- `lib/RuntimeMix/MixDirectBackend.cpp`
  收敛阶段边界、错误上下文和 manifest 输出

- `lib/RuntimeMix/MixCommandBuilder.cpp`
  收敛所有外部命令构造，避免工具命令散落

- `tools/mix-validator/mix_validator_main.cpp`
  固化 runner-first 执行策略，限制 direct packed fallback

- `examples/baremix-test/run.sh`
  仅保留用户验收入口职责，不承载核心逻辑

## Validation Strategy

稳定化阶段只要求一个强验收样例：`baremix`。

必须验证三层：

1. Compile validation
   - packed `.so` 生成
   - `mix_runner` 生成
   - manifest 字段完整

2. Execution validation
   - `mix-validator` 走 `mix_runner`
   - simulator 成功执行

3. Precision validation
   - `max_abs_diff == 0`
   - `mean_abs_diff == 0`
   - `verify_result.py` 输出 `test pass`
   - `golden.bin` 与 `actual.bin` md5 一致

所有验证都必须在 xvm 上完成，不在 host 上声明成功。

## Milestones

### Milestone 1: Contract Stabilization

内容：

- 固化 `MixArtifact`
- 固化 validator 的 runner-first 路径
- 固化 manifest 字段

验收：

- `mix-validator` 不再意外落回 direct packed 路径
- artifact 布局和字段可重复

### Milestone 2: Pipeline Stabilization

内容：

- 收敛 `MixDirectBackend` 的 stage 边界
- 收敛错误上下文
- 收敛命令构造责任

验收：

- 任一 stage 失败时能快速定位
- backend 代码结构可直接映射到未来 `Compiler.cpp`

### Milestone 3: ABI Stabilization

内容：

- 把当前 mix host ABI 约束集中表达
- 从 baremix 专用描述收敛到通用 runner/config 描述

验收：

- backend 不依赖 baremix 内部图结构
- 当前 ABI 限制被显式表达，而不是隐式散落

### Milestone 4: Merge Readiness

内容：

- 标出每个 `RuntimeMix` 核心模块未来归并位置
- 避免继续在临时工具层增长逻辑

验收：

- 可以为每个核心文件给出未来归并落点
- 不存在“只能留在 RuntimeMix”的核心逻辑

## Risks

### 1. 误把 ABI 限制当作图结构限制

如果第二个样例失败，必须先判断是 host ABI 不匹配，还是 compile pipeline 不通，不能直接在 backend 里加样例特判。

### 2. 工具层继续承载核心逻辑

如果 `mix-compiler` / `mix-validator` 继续长逻辑，未来并回时会把稳定化成果打散。

### 3. 过早并回主线

在 direct backend 还没稳定前合并进 `lib/Runtime`，会把实验性行为扩散到主路径，也更容易与并行修改冲突。

## Decision

下一阶段按“先稳定，再泛化”推进。

顺序是：

1. 固化 contract
2. 固化 pipeline
3. 固化 ABI
4. 准备并回

只有这四步完成后，才开始用第二个样例检验真正的通用性。
