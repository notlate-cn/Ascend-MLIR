# RuntimeMix Second Sample Split ReLU Design

**Date:** 2026-03-31

## Goal

用第二个、结构差异足够大的 mix 样例验证当前 `RuntimeMix` 的 artifact-backed tiling 机制不是 baremix-only 修补。

本阶段选择的目标样例是：

- [examples/matmul-add-relu-sum/fc_relu_split_mix.cpp](/Volumes/GM9/code/Ascend-MLIR/examples/matmul-add-relu-sum/fc_relu_split_mix.cpp)

成功标准不是“支持任意 mix kernel”，而是：

- `RuntimeMix` 在第二个 mix 样例上也能完成 compile
- 产出真实 tiling artifact
- runner 和 direct packed 使用同源 tiling
- xvm 上执行与精度通过

## Why This Sample

选择 `fc_relu_split_mix.cpp` 的原因：

1. 它和 baremix 有足够差异  
   仍然是 AIC/AIV 混合结构，但激活部分和入口命名比 leakyrelu 更干净。

2. 它和当前 direct backend 的 sample-aware 路线更匹配  
   源码里有明确的：
   - `fc_relu_split_mix_origin`
   - `auto_gen_fc_relu_split_kernel`
   - `.ascend.meta.fc_relu_split_*`

3. 它比 `fc_leakyrelu_mix.cpp` 更适合当“第二个稳定样例”  
   前一个候选在当前 xvm / simulator 环境里，连 proto packed 路径都不稳定，不能再当验证基线。

## Scope

### In Scope

- 为 `fc_relu_split_mix.cpp` 做一个最小 `RuntimeMix` 验证入口
- 复用当前 compile / artifact / manifest / direct packed / runner 机制
- 让第二个样例也产出真实 tiling artifact
- 在 xvm 上验证 runner 与 direct packed 都可用

### Out of Scope

- 不做“任意 mix kernel 自动接入”
- 不并回 `lib/Runtime`
- 不统一所有 `examples/matmul-add-relu-sum/*mix*.cpp`
- 不重构 `RuntimeMix` 公共 ABI 模型
- 不继续 debug `fc_leakyrelu_mix.cpp` 这个样例本身

## Recommended Approach

### Approach A: 单独做一个第二样例验证入口，推荐

做一个独立、最小的 split-relu example/runner 脚本，只服务于验证 `RuntimeMix` 当前机制是否能覆盖第二个 mix 样例。

优点：

- 范围清楚
- 失败时容易定位是样例适配问题还是 runtime 问题
- 不会把多个候选样例混在一起

缺点：

- 会新增一个临时验证入口

### Approach B: 直接复用当前 `examples/leakyrelu-mix-test/`

优点：

- 少建一个目录

缺点：

- 语义会变得混乱
- 目录名和实际样例不一致，后续更难维护

### Approach C: 一次接两到三个 ReLU 变体

优点：

- 覆盖面更大

缺点：

- 当前阶段不需要
- 一旦失败，很难知道是哪一层没抽象好

结论：先走 Approach A。

## Design

### 1. Add a Second Focused RuntimeMix Example

建议新增一个专门的验证入口：

- `examples/relu-split-mix-test/`

它的职责和当前 `examples/baremix-test/` 类似，但只服务于 `fc_relu_split_mix.cpp`。

该目录只需要：

- 一个 `run.sh`
- 一个简短 `README.md`
- 指向现有数据生成 / 校验脚本的最小 glue

### 2. Keep Artifact-Backed Tiling Contract

这个第二样例必须沿用当前已经在 baremix 上成立的契约：

- backend 产出真实 `tiling.bin`
- manifest 记录：
  - `abi_tiling_mode=generated_file`
  - `abi_tiling_source=<artifact-relative path>`
- direct packed 与 runner 共用这份 tiling

本阶段不允许为第二样例再发明一套 tiling 机制。

### 3. Keep ABI Explicit

第二样例可以有自己的：

- input names
- output names
- shapes
- dtype
- workspace bytes

但这些都必须继续通过 artifact/manifest 显式表达，不允许在 validator 里再单独硬编码一份 `fc_relu_split_mix` 常量表。

### 4. Data and Naming

当前设计建议 runtime kernel 名使用：

- `fc_relu_split`

而源码文件仍然是：

- `fc_relu_split_mix.cpp`

原因是源码里的 mix metadata section 已经按：

- `.ascend.meta.fc_relu_split_0_mix_aic`
- `.ascend.meta.fc_relu_split_0_mix_aiv`

命名，这和 `fc_relu_split` 更一致。

数据文件命名维持和当前样例族一致的最小约定：

- `input_a.npy`
- `input_b.npy`
- `input_bias.npy`
- `output.npy`

然后在 example `run.sh` 中做最小 `.npy -> .bin` glue，落到 artifact ABI 期望的文件名。

### 5. Validation

必须在 xvm 上验证四层：

1. compile 成功
2. 真实 tiling artifact 存在
3. runner 路径通过
4. `--force-direct-packed` 也通过，并与 runner md5 一致

## Expected Outcome

完成后，应能明确回答：

- 当前 `RuntimeMix` 的 artifact-backed tiling 机制至少适用于 baremix 和 split-relu 两个结构不同的 mix 样例

如果失败，也能明确知道失败点是在：

- 样例 ABI 适配
- tiling 生成
- packed mix execution

而不是继续停留在“只有 baremix 通”的状态。

## Risks

### 1. 第二样例仍可能暴露新的 ABI 细节

如果 `fc_relu_split_mix.cpp` 需要与 baremix 不同的 ABI 描述，可能会暴露出当前 metadata 模型仍然过窄。

这不是坏事，正是第二样例的价值。

### 2. 运行时 kernel 名与源码文件名不一致

这一点必须在 backend、manifest 和 example run script 中统一处理，不能一半按 `fc_relu_split_mix`，一半按 `fc_relu_split`。

## Decision

下一步按以下顺序推进：

1. 新增 `examples/relu-split-mix-test/`
2. 为 `fc_relu_split_mix.cpp` 增加一个最小 `RuntimeMix` 验证入口
3. 复用当前 artifact-backed tiling 契约
4. 在 xvm 上验证 runner 和 direct packed 都通过
