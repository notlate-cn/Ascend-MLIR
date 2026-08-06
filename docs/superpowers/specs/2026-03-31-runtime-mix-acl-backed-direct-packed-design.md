# RuntimeMix ACL-Backed Direct Packed Design

**Date:** 2026-03-31

## Goal

验证并实现一个最小修复：让 `RuntimeMix::Executor::RunPackedMixFile(...)` 在 `baremix` 上改用与稳定 runner 一致的 `acl/aclrt` 执行模型，从而修复当前 direct packed 路径在 xvm simulator 下的 `div by 0` / `invalid ldst addr` / timeout 问题。

本阶段的目标不是泛化 mix 执行，也不是重写整个 runtime，而是验证：

> packed mix launcher 当前期望的执行模型，是否本质上更接近 ACL/aclrt，而不是当前 `Executor` 的底层 `rt*` 调用模型。

## Scope

### In Scope

- 对 direct packed 路径做最小 ACL-backed 执行验证
- 只围绕 `RunPackedMixFile(...)` 收敛修复
- 用 `baremix` 作为唯一验收样例
- 保持当前 artifact / ABI metadata / validator 接口不变

### Out of Scope

- 不修改 `lib/Runtime`
- 不泛化到任意 mix kernel
- 不删除 runner 路径
- 不重新设计 ABI metadata
- 不重写 pack / stub 生成

## Current Evidence

当前已确认的事实：

1. `mix_runner` 路径稳定通过  
   它使用的是：
   - `aclInit`
   - `aclrtSetDevice`
   - `aclrtCreateStream`
   - `aclrtMalloc`
   - `aclrtMemcpy`
   - `ACLRT_LAUNCH_KERNEL`
   - `aclrtSynchronizeStream`

2. 强制 direct packed 路径稳定失败  
   在 xvm 上可复现：
   - `div by 0`
   - `invalid ldst addr`
   - `timeout 25`
   - `EXIT:124`

3. direct packed 当前使用的是底层 `rt*` 模型  
   `RunPackedMixFile(...)` 走的是：
   - `rtStreamCreate_`
   - `rtMalloc_`
   - `rtMemcpy_`
   - `rtStreamSynchronize_`

4. 补 `preloadPackedMixDeps()` 语义没有解决问题  
   说明根因更像“执行模型不匹配”，而不是“缺依赖 preload”。

## Approaches

### Approach A: 在 `RunPackedMixFile(...)` 中切到 ACL-backed launch，推荐

优点：

- 直接验证当前最强假设
- 与稳定 runner 的执行模型最接近
- 如果成立，后面并回 `lib/Runtime` 也更有方向

缺点：

- 会让 `Executor` 的 mix packed 路径和 vec/cube 二进制路径采用不同执行后端

### Approach B: 继续修 `rt*` 路径

优点：

- 表面上更贴近现有 `Executor` 结构

缺点：

- 当前证据已经表明这条路可疑
- 容易继续在错误抽象上打补丁

### Approach C: 让 direct packed 退回 runner-only

优点：

- 风险最小

缺点：

- 明确不符合和现有 `lib/Runtime` mix validator 语义对齐的目标

结论：先走 Approach A。

## Design

### 1. Keep the Fix Narrow

本阶段只允许围绕这些文件改动：

- `include/RuntimeMix/Executor.h`
- `lib/RuntimeMix/Executor.cpp`
- 必要时，`tools/mix-validator/mix_validator_main.cpp` 仅保留 debug forcing 开关，不再扩大职责

修复不能扩到：

- `MixDirectBackend`
- manifest ABI metadata
- runner 生成逻辑

### 2. ACL-Backed Mix Packed Execution Path

在 `RunPackedMixFile(...)` 中，为 packed mix 增加一条 ACL-backed 执行分支。它的语义要尽量贴近稳定 runner：

1. `aclInit`
2. `aclrtSetDevice`
3. `aclrtCreateStream`
4. `aclrtMallocHost` / `aclrtMalloc`
5. `aclrtMemcpy`
6. `aclrtlaunch_<kernel>`
7. `aclrtSynchronizeStream`
8. `aclrtMemcpy` D2H
9. ACL 资源释放

这里的关键不是把 runner `main.cpp` 全部搬进来，而是把 packed mix launch 的最小必要执行语义搬进 `Executor`。

### 3. Preserve Current Baremix ABI Surface

本阶段仍然只支持当前 baremix 接受面：

- 3 inputs
- 1 output
- fixed workspace
- fixed tiling bytes

如果 ACL-backed 执行修复成功，也只证明当前 baremix packed mix 路径成立，不代表已经支持更一般的 mix kernel。

### 4. Validation

必须在 xvm 上验证三件事：

1. 正常 `run.sh` 仍然通过
2. `--force-direct-packed` 不再 timeout
3. direct packed 路径输出：
   - `PASS`
   - `test pass`
   - md5 与 golden 一致

## Expected Outcome

完成后，应得到：

- `RunPackedMixFile(...)` 在 baremix 上稳定可用
- direct packed 路径和 runner 路径结果一致
- 根因被明确定位为执行模型差异，而不是 ABI metadata 或 preload 缺失

## Risks

### 1. ACL-backed path 仍然失败

如果 ACL-backed `RunPackedMixFile(...)` 仍然失败，说明根因可能在：

- packed host stub 自身预期
- pack 产物结构
- runner 里未显式建模的额外执行前提

到那时再上移调查范围，而不是现在提前发散。

### 2. Executor mix path 与其他路径分叉

本阶段允许 `mix` packed 先走 ACL-backed 特化路径，因为当前问题本来就只存在于 packed mix。后续并回 `lib/Runtime` 时再决定是否统一抽象。

## Decision

下一步按以下顺序做：

1. 在 `Executor` 中实现最小 ACL-backed packed mix 执行路径
2. 保留当前 direct repro 开关用于 xvm 验证
3. 以 baremix 为唯一样例验证成功
4. 再决定是否收紧 validator 默认语义
