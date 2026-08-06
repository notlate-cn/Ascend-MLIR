# RuntimeMix Direct Packed Alignment Design

**Date:** 2026-03-31

## Goal

把 `RuntimeMix` 当前 `mix` 验证语义收敛到与现有 `lib/Runtime` 一致：

- `mix-validator` 以 direct packed 执行为主
- `mix_runner` 保留为调试/对照产物，而不是长期默认前提
- 当前 baremix 用例继续在 xvm 上稳定 `PASS`

这一步不再围绕“修一个 direct packed 崩溃 bug”展开，因为当前代码下强制 `--force-direct-packed` 已经在 xvm 上通过。

## Scope

### In Scope

- 调整 `RuntimeMix` 的 `mix-validator` 执行优先级
- 明确 direct packed 是当前 `mix` 校验的主语义
- 保留 `mix_runner` 产物和 runner 路径用于调试、对照和回归
- 更新 README / 验收说明

### Out of Scope

- 不修改 `lib/Runtime`
- 不做通用混合结构泛化
- 不删除 `mix_runner`
- 不重写 `Executor::RunPackedMixFile(...)`
- 不改变当前 baremix ABI 接受面

## Current State

当前已经成立的事实是：

1. `bash examples/baremix-test/run.sh` 在 xvm 上通过
2. `mix_runner` 路径稳定
3. 强制 `--force-direct-packed` 后，direct packed 路径也已经 `PASS`

这意味着：

- direct packed 已经具备当前 baremix 场景的可用性
- 当前真正需要收口的是“默认语义”和“文档/入口”
- 继续按“修 direct packed bug”写计划会制造错误目标

## Approaches

### Approach A: 语义收敛，推荐

把 `RuntimeMix` 的 `mix-validator` 从 runner-first 改成 direct-packed-first，同时保留 runner 作为调试/回归路径。

优点：

- 直接对齐现有 `lib/Runtime`
- 改动面小
- 不会继续把 runner 变成 RuntimeMix 的长期核心语义

缺点：

- 需要重新整理当前 README、注释和接受标准

### Approach B: 保持 dual-path 平权

把 runner 和 direct packed 都保留成同级主路径。

优点：

- 兼容性看起来更保守

缺点：

- 语义不清楚
- 后面并回 `lib/Runtime` 时还要再收一次口

### Approach C: 彻底删掉 runner

优点：

- 最干净

缺点：

- 会失去当前已经证明稳定的调试/对照产物
- 本阶段没有必要

结论：先走 Approach A。

## Design

### 1. Direct Packed Becomes the Primary Mix Validation Path

当前 `RuntimeMix` 的 `mix-validator` 应明确遵循：

- 默认先走 direct packed
- 只有在显式调试/对照需要时才走 runner

这样它的行为就和现有 [tools/validator/validator_main.cpp](/Volumes/GM9/code/Ascend-MLIR/tools/validator/validator_main.cpp) 的 `mix` 路径一致，本质上都围绕 `Executor::RunPackedMixFile(...)`。

### 2. Keep Runner as a Debug/Comparison Artifact

runner 仍然保留，原因只有三个：

1. 当前已有稳定 runner 作为对照
2. 对 pack / stub / launch 差异做调试时仍然有价值
3. baremix acceptance 入口目前已经依赖它的存在来生成完整 artifact

但 runner 不再被描述为“正常验证的唯一主路径”。

### 3. Preserve Current ABI Surface

本阶段继续只接受当前 baremix ABI：

- 3 inputs
- 1 output
- fixed workspace
- fixed tiling source

这一步只改“执行语义”，不改“ABI 接受面”。

### 4. Validation Strategy

必须在 xvm 上验证两类结果：

1. Default path
   - 默认 `mix-validator` 行为走 direct packed
   - `PASS`
   - `test pass`

2. Runner comparison path
   - runner 路径仍可显式使用
   - 结果和 direct packed 一致

这样可以同时保证：

- 语义已经对齐 `lib/Runtime`
- runner 仍然可用于调试和回归

## Expected Outcome

完成后，`RuntimeMix` 应呈现为：

- `mix` 校验默认语义：direct packed
- `mix_runner`：辅助调试/对照产物
- `examples/baremix-test` 文档与行为一致

## Risks

### 1. 入口切换导致已有脚本认知不一致

需要同步更新 README 和命令说明，避免继续把 runner-first 当成主路径。

### 2. Runner 路径失去回归价值

不能因为语义切换就让 runner 路径变成无人维护的死代码。必须保留一个明确的 comparison/debug 入口。

### 3. 把“当前可用”误当成“已经泛化”

本阶段只说明 baremix 上 current direct packed 已经成立，不代表已经支持更一般的 mix 结构。

## Decision

下一步按以下顺序实现：

1. 调整 `mix-validator` 的执行优先级为 direct-packed-first
2. 保留 runner 的显式 comparison/debug 入口
3. 更新 README 与验证说明
4. 在 xvm 上验证 default path 和 runner comparison path 都正确
