# RuntimeMix Direct Packed Debug Design

**Date:** 2026-03-31

## Goal

把 `RuntimeMix` 的 `mix` 执行语义对齐现有 `lib/Runtime`：

- `validator` 在 `mix` 场景下以 direct packed 执行为主
- `mix_runner` 不再是长期默认依赖
- 先以 `baremix` 为验收样例，把 `Executor::RunPackedMixFile(...)` 在 xvm simulator 下修稳

本阶段不是做通用 mix 泛化，也不是并回 `lib/Runtime`。重点是把当前 direct packed 路径作为一个独立 bug 彻底定位并修复。

## Scope

### In Scope

- 对比稳定 runner 路径和 `RunPackedMixFile(...)` 的真实执行差异
- 在 `RuntimeMix` 中修复 direct packed 路径
- 让 `mix-validator` 在禁用 `mix_runner` 时，仍能在 `baremix` 上 `PASS`
- 保持当前 artifact / ABI metadata 设计不变

### Out of Scope

- 不修改 `lib/Runtime`
- 不启动混合结构泛化
- 不重新设计 host ABI
- 不删除 runner 路径，只降低它的主路径地位

## Current Problem

当前 `RuntimeMix` 已知行为是：

- 正常 runner 路径稳定，xvm 上能 `PASS`
- direct packed fallback 已经改成 metadata-driven
- 但一旦强制禁用 `mix_runner`，`Executor::RunPackedMixFile(...)` 在 simulator 下会报 `div by 0`、`invalid ldst addr`

这说明：

- 问题已经不在 manifest ABI metadata
- 问题也不在 validator 的输入/输出装载逻辑
- 根因在 direct packed 的执行模型与稳定 runner 路径之间仍有 ABI / launch 差异

## Approaches

### Approach A: 单点修 `RunPackedMixFile(...)`，推荐

先把 runner 稳定路径和 direct packed 路径逐项对比，确认最小差异，再只修 `RuntimeMix/Executor.cpp`。

优点：

- 最符合现有 `lib/Runtime` 语义
- 改动集中
- 后面并回 `lib/Runtime` 最顺

缺点：

- 需要先完成一轮严格的 root-cause diff

### Approach B: 把 runner 逻辑内嵌进 `Executor`

优点：

- 容易快速复用当前稳定行为

缺点：

- 会把 `Executor` 变成“内嵌 runner”
- 和未来并回 `lib/Runtime` 的方向相反

### Approach C: 保持 runner-first，不再修 direct packed

优点：

- 风险最低

缺点：

- 明确不符合“对齐现有 `lib/Runtime`”的目标

结论：先走 Approach A。

## Design

### 1. Root-Cause Diff First

先不改代码，先列出 direct packed 与稳定 runner 路径的关键差异。至少检查：

1. `dlopen` flag 与 preload 顺序
2. stream / device 初始化
3. workspace 分配大小和来源
4. tiling 字节内容、长度和拷贝方式
5. `aclrtlaunch_<kernel>` 调用签名与参数顺序
6. launch 前后同步、D2H 顺序
7. packed `.so` 中 host stub 的预期行为

这一轮的目标是拿到一个明确假设：

> “direct packed 失败是因为 X，而 runner 稳定是因为 Y”

在没有这个假设之前，不允许直接改 `Executor`。

### 2. Fix Only `RunPackedMixFile(...)`

确认根因后，只修改：

- `include/RuntimeMix/Executor.h`，如果需要补最小辅助接口
- `lib/RuntimeMix/Executor.cpp`

修复原则：

- 只修 direct packed 的 root cause
- 不顺手改 validator 结构
- 不顺手改 backend artifact
- 不把 runner 逻辑大段搬进 `Executor`

如果需要复用 packed host stub 的某些语义，可以抽成很薄的 helper，但不能把 `main.cpp` 风格逻辑整个塞进 runtime。

### 3. Keep ABI Surface Narrow

本阶段仍然只接受当前 baremix 的 3-input / 1-output mix surface。

也就是说：

- 可以继续要求 direct packed 当前只支持一个明确接受面
- 但必须在这个接受面内稳定通过
- 不在本阶段扩大输入输出形态

### 4. Validation Strategy

必须在 xvm 上做两类验证：

1. Normal path
   - `bash examples/baremix-test/run.sh`
   - 继续保持 `PASS`

2. Direct packed path
   - 强制禁用或绕过 `mix_runner`
   - 让 `mix-validator` 走 `Executor::RunPackedMixFile(...)`
   - `PASS`
   - `test pass`
   - md5 一致

同时保留最小诊断输出，用于证明 direct packed 走的确实不是 runner 路径。

## Expected Outcome

完成后，`RuntimeMix` 应达到：

- `mix` validator 的主语义与现有 `lib/Runtime` 一致
- runner 可以保留，但不再是必须依赖
- direct packed 路径在 baremix 上稳定可用

## Risks

### 1. Root cause 不在 `Executor`

如果最后发现 packed host stub 本身和 runner 使用的 `.so` / loader 环境有结构性差异，那么修复点可能要上移到 pack / stub 生成阶段。

这不视为本设计失败，但必须先通过 root-cause diff 证实，不能预先假设。

### 2. 模拟器依赖环境差异

runner 路径和 direct packed 路径可能对 preload 或 `LD_LIBRARY_PATH` 有细微依赖差异。本阶段允许补齐这些差异，但必须明确记录为 direct packed 执行前提，而不是继续留成隐式行为。

### 3. 修成“baremix only workaround”

本阶段只用 baremix 验收，但修复不能是只在 validator 里特殊绕过一条路径。修复点必须落在 direct packed 执行模型本身。

## Decision

下一步按以下顺序执行：

1. 对比 runner 和 direct packed 的执行差异，锁定 root cause
2. 只修 `RuntimeMix/Executor.cpp`
3. 在 xvm 上验证 direct packed baremix `PASS`
4. 再决定是否进入更一般的 mix 执行泛化
