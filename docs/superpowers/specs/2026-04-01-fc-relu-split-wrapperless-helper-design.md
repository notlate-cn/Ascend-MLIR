# FC ReLU Split Wrapperless Helper Design

**Date:** 2026-04-01

## Goal

为 `fc_relu_split_mix.cpp` 增加一个仓库内显式保留的 wrapperless helper 源文件，用作官方 AscendC toolchain 与 RuntimeMix 的公共编译基线。

本阶段只验证一件事：

- 源码中手写 `auto_gen_*` wrapper 与官方 auto-gen 冲突时，wrapperless helper 可以绕过这个冲突。

## Why

当前已在 xvm 上确认：

- 原样 `fc_relu_split_mix.cpp` 走官方 AscendC CMake 会在 auto-gen 阶段失败
- 原样 `fc_leakyrelu_mix.cpp` 也会在同一类 `auto_gen_*_origin` 调用点失败
- 临时 wrapperless 视图可以越过这一 compile blocker

因此我们需要一个可复用、可回归、可被两条编译链共享的显式 helper，而不是继续把 workaround 藏在临时脚本或字符串生成里。

## Scope

### In Scope

- 新增一个仓库内显式 helper 源文件
- 保留原 `fc_relu_split_mix.cpp` 不改语义
- 让官方 AscendC CMake/toolchain 能接受这个 helper，至少越过当前 auto-gen compile blocker
- 为后续 RuntimeMix 第二样例验证提供同一个基线文件

### Out of Scope

- 不修改原 `fc_relu_split_mix.cpp`
- 不在这一阶段解决 simulator 执行错误
- 不在这一阶段处理 merge/pack 后续问题
- 不推广到所有 mix 样例

## Approaches

### Approach A: 只在脚本中临时生成 wrapperless 视图

优点：
- 不新增仓库文件

缺点：
- 不可复用
- 不利于回归
- 根因和 workaround 不可见

### Approach B: 新增显式 helper 源文件，推荐

建议新增：

- `examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp`

职责：
- 保留真实 kernel body
- 保留 `.ascend.meta.fc_relu_split_*`
- 不包含手写 `auto_gen_fc_relu_split_kernel` wrapper

优点：
- 公共基线明确
- 官方 CMake 与 RuntimeMix 可共同消费
- 便于后续决定是否把原样例收敛为这一形态

### Approach C: 直接改原 `fc_relu_split_mix.cpp`

优点：
- 长期可能更干净

缺点：
- 当前风险过高
- 还没验证完整执行链前，不应该直接改原始样例

结论：先走 Approach B。

## Design

### File

新增：

- `examples/matmul-add-relu-sum/fc_relu_split_wrapperless.cpp`

### Content Rules

该 helper 文件应：

- 复用 `fc_relu_split_mix.cpp` 当前 kernel body 逻辑
- 保留最终 kernel 名为 `fc_relu_split`
- 保留：
  - `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`
  - AIC matmul + bias 路径
  - AIV relu 路径
  - `.ascend.meta.fc_relu_split_0_mix_aic`
  - `.ascend.meta.fc_relu_split_0_mix_aiv`
- 不包含手写 `auto_gen_fc_relu_split_kernel` wrapper
- 不包含源码内再次包装 workspace/tiling 的逻辑

### Validation Target

本阶段验证目标只到 compile 基线：

1. 官方 AscendC CMake/toolchain 直接编 `fc_relu_split_wrapperless.cpp`
2. 至少越过当前 `auto_gen_fc_relu_split_kernel_origin` 这类 compile blocker
3. 将该 helper 作为下一阶段 RuntimeMix 第二样例的编译输入候选

## Success Criteria

成功标准是：

- helper 文件入仓
- 官方 CMake/toolchain 不再在 `auto_gen_*_origin` 点失败
- 我们可以明确说“当前 compile blocker 已从源码 wrapper 冲突移除”

不是：

- simulator 执行通过
- 精度通过

## Decision

下一步：

1. 新增 `fc_relu_split_wrapperless.cpp`
2. 用 xvm 上的官方 AscendC CMake/toolchain 编译它
3. 只确认 compile blocker 是否消失
