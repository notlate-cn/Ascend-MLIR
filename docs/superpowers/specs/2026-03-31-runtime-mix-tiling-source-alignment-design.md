# RuntimeMix Tiling Source Alignment Design

**Date:** 2026-03-31

## Goal

修复 `RuntimeMix` direct packed 路径和稳定 runner 路径之间最关键的不一致：tiling 来源不同。

本阶段目标是让：

- runner 使用的 tiling
- direct packed fallback 使用的 tiling

来自同一份 artifact 语义，而不是一边走真实 `GenerateTiling(...)`，另一边走 validator 内部的手写 `baremix_fixed_blob`。

## Scope

### In Scope

- 重新定义当前 baremix artifact 中 tiling 的真实来源
- 让 direct packed path 读取/复用和 runner 同源的 tiling 结果
- 只以 baremix 为验收样例

### Out of Scope

- 不修改 `lib/Runtime`
- 不泛化到任意 mix kernel
- 不继续深挖 ACL-backed `Executor` 路径
- 不改变当前输入输出 ABI 接受面

## Current Evidence

当前已经确认：

1. runner 路径稳定通过
2. direct packed 路径稳定失败，表现为：
   - `div by 0`
   - `invalid ldst addr`
   - `EXIT:124`
3. ACL-backed `Executor` 尝试没有解决问题
4. runner 和 direct packed 当前喂给 kernel 的 tiling 并不同源：
   - runner 路径在 [lib/RuntimeMix/MixDirectBackend.cpp](/Volumes/GM9/code/Ascend-MLIR/lib/RuntimeMix/MixDirectBackend.cpp) 中真实调用 `GenerateTiling(...)`
   - direct packed 路径在 [tools/mix-validator/mix_validator_main.cpp](/Volumes/GM9/code/Ascend-MLIR/tools/mix-validator/mix_validator_main.cpp) 中使用 `buildBaremixTiling()` 的手写字节

所以当前最强根因是：

> direct packed 失败不是单纯执行后端问题，而是它收到的 tiling 与 runner 实际执行时使用的 tiling 不一致。

## Approaches

### Approach A: 让 artifact 产出真实 tiling，并让 direct packed 读取它，推荐

优点：

- 直接对齐 runner 和 direct packed 输入
- 保持 artifact 驱动设计
- 后面泛化时也更合理

缺点：

- 需要在 backend 产物里明确落一个真实 tiling 文件或等价元数据

### Approach B: 在 validator 里复刻 `GenerateTiling(...)`

优点：

- 改动看起来集中在 validator

缺点：

- 又会复制一份 runner/backend 逻辑
- 和当前“同源 artifact”方向相反

### Approach C: 把 runner 生成的 tiling 逻辑直接塞进 `Executor`

优点：

- 可能也能工作

缺点：

- 继续让 runtime 背负不该背的 build-time 逻辑

结论：先走 Approach A。

## Design

### 1. Tiling Must Become a Real Artifact

当前 manifest 里写的是：

- `abi_tiling_mode=fixed_bytes`
- `abi_tiling_source=baremix_fixed_blob`

但这已经和真实 runner 语义不一致。

本阶段要把 tiling 变成 artifact 中可追踪、可复用的真实产物。推荐形式：

- 在 build/output 目录中落一个真实 tiling 二进制文件
- manifest 记录它的路径和来源

例如：

- `abi_tiling_mode=generated_file`
- `abi_tiling_source=<artifact-relative tiling path>`

### 2. Runner and Direct Packed Must Read the Same Tiling Result

runner 路径可以继续通过 `GenerateTiling(...)` 生成这个文件，或者在 build 阶段直接产出这份文件。

但 direct packed path 不能再自己拼一份手写 blob。它必须：

- 读取 artifact 中同一份 tiling 文件
- 或者读取与该文件完全等价、明确记录的 artifact 数据

### 3. Keep Current ABI Surface Narrow

本阶段仍然只支持 baremix 当前接受面：

- 3 inputs
- 1 output
- fixed workspace
- current matmul/bias shape

变化只在 tiling 来源，不在 ABI 形态。

### 4. Validation

必须在 xvm 上验证：

1. `bash examples/baremix-test/run.sh` 仍然通过
2. 强制 direct packed：
   - 不再 timeout
   - `PASS`
   - `test pass`
   - md5 一致
3. manifest / artifact 中能看到真实 tiling 来源，而不是 `baremix_fixed_blob`

## Expected Outcome

完成后，`RuntimeMix` 应达到：

- runner 和 direct packed 使用同源 tiling
- direct packed 在 baremix 上通过
- 当前 root cause 从“执行模型差异”收敛为“tiling artifact 对齐完成”

## Risks

### 1. Tiling 对齐后仍然失败

如果同源 tiling 后 direct packed 仍失败，再回头看 `Executor` 执行模型差异。那时范围会更清晰，因为可以排除 tiling 输入不一致。

### 2. 把 generated tiling 塞错阶段

tiling 产物应由 backend/artifact 持有，不应再次散落到 validator 或 runtime 的内部硬编码里。

## Decision

下一步按以下顺序做：

1. 在 artifact 中落真实 tiling 产物并更新 manifest
2. 让 direct packed path 消费这份真实 tiling
3. 在 xvm 上验证 baremix direct packed `PASS`
