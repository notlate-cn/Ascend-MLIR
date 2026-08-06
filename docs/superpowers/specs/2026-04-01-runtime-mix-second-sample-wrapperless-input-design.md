# RuntimeMix Second Sample Wrapperless Input Design

**Date:** 2026-04-01

## Goal

将第二个 RuntimeMix 验证样例的编译输入从原始
`fc_relu_split_mix.cpp` 切换为显式的 wrapperless helper
`fc_relu_split_wrapperless.cpp`，先验证当前 direct backend 的
compile / run / accuracy 闭环，而不在这一阶段同时承担“兼容手写
auto_gen_* wrapper 源码”的职责。

## Why

当前证据已经明确：

- 原始 `fc_relu_split_mix.cpp` 与官方 AscendC auto-gen 机制冲突
- 这个冲突不是 RuntimeMix 独有问题
- wrapperless helper 已经证明可被官方 sample-style CMake 编译到
  `libascendc_kernels_sim.so`

因此，下一步最有价值的是用 wrapperless helper 作为编译基线，
验证 RuntimeMix 的执行链，而不是继续把源码 wrapper 兼容和执行
链验证混在一起。

## Scope

### In Scope

- 更新 `examples/relu-split-mix-test/run.sh` 的 `KERNEL_SRC`
- 如有必要，更新 README 说明第二样例当前使用 wrapperless helper 作为
  编译输入基线
- 在 xvm 上重新验证 RuntimeMix 第二样例的 compile / runner /
  direct-packed / accuracy 结果

### Out of Scope

- 不改原始 `fc_relu_split_mix.cpp`
- 不在本阶段为 RuntimeMix 新增“手写 auto_gen wrapper 兼容层”
- 不泛化到任意 mix kernel
- 不并回 `lib/Runtime`

## Approaches

### Approach A: 直接切换 second-sample 的编译输入，推荐

将 `examples/relu-split-mix-test/run.sh` 的 `KERNEL_SRC` 改为
`fc_relu_split_wrapperless.cpp`，其余 ABI、数据脚本、validator 验收
逻辑保持不变。

优点：
- 范围最小
- 直接验证 RuntimeMix 执行链
- 与已经确认可编译的官方基线一致

缺点：
- 第二样例暂时不再直接消费原始样例源码

### Approach B: 新增一个平行的 wrapperless test 目录

优点：
- 与原 split-relu test 目录完全隔离

缺点：
- 多一个临时入口
- 当前没有必要

### Approach C: 同时支持 mix.cpp 与 wrapperless.cpp 输入

优点：
- 功能更完整

缺点：
- 超出当前目标
- 会把执行链验证和源码兼容层混在一起

结论：走 Approach A。

## Design

### 1. Keep RuntimeMix Backend Stable

`RuntimeMix` backend 本阶段不新增新的 sample-aware 源码兼容逻辑。
当前验证只要求 direct backend 能正确处理 wrapperless 编译视图。

### 2. Switch Example Input Only

`examples/relu-split-mix-test/run.sh` 中：

- `KERNEL_SRC` 改为
  `examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp`
- `KERNEL_NAME` 继续使用 `fc_relu_split`

这样可保持：

- metadata section 命名一致
- artifact 命名一致
- data / manifest / validator 约定不变

### 3. Preserve Existing Validation Contract

成功标准保持不变：

1. compile 成功
2. `out/tiling.bin` 存在
3. runner 路径通过
4. `--force-direct-packed` 通过
5. runner / direct / golden md5 一致

## Risks

### 1. Wrapperless helper 仍可能暴露新的执行期差异

即便 compile 基线成立，运行期仍可能暴露新的 ABI / tiling /
pack 差异。这是本阶段要验证的真实问题。

### 2. README 语义需要同步

文档必须明确：
- 原始 `fc_relu_split_mix.cpp` 仍是语义来源
- `fc_relu_split_wrapperless.cpp` 只是当前 RuntimeMix/官方工具链
  验证用的编译输入基线

## Decision

下一步执行顺序：

1. 更新 second-sample run script 的编译输入
2. 同步 README
3. 在 xvm 上重新跑完整闭环
4. 根据结果决定是否进入运行期 root-cause debugging
