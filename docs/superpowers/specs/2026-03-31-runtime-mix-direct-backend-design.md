# RuntimeMix Direct Backend Design

**Date:** 2026-03-31

## Goal

在 `RuntimeMix` 中实现方案 B：不再复用 AscendC sample CMake 工具链，而是直接在 `RuntimeMix` 内编排 `bisheng + ld.lld + merge + pack + host stub/launcher`，形成一条通用 mix-kernel 编译与执行链。

该方案的最终目标不是长期保留 `RuntimeMix` 目录，而是在功能稳定后并回 `lib/Runtime`。因此，本设计要求：

- 当前先在 `RuntimeMix` 中隔离开发，避免和并行中的 `lib/Runtime` 修改互相干扰
- 模块边界、命名和产物模型从第一天起就向 `lib/Runtime` 对齐
- 方案 B 打通后，合并回 `lib/Runtime` 时主要是搬移和整合模块，而不是重写逻辑

## Scope

### In Scope

- 在 `RuntimeMix` 中新增 direct mix backend
- 保持 `mix-compiler` 的用户接口不变
- 保持 `mix-validator` 的 CLI 语义不变
- 支持“单个 AscendC 源文件 -> 单个 mix artifact”的通用编译闭环
- 产出与方案 A 兼容的 `MixArtifact`
- 在 xvm 上完成编译、sim 执行和精度校验

### Out of Scope

- 本阶段不把代码并回 `lib/Runtime`
- 本阶段不统一 vec/cube/mix 三类 backend
- 本阶段不改现有 `compiler` / `validator` 主入口
- 本阶段不要求自动从源码完全推断任意 host 参数 ABI

## Constraints

### 1. External Interface Stability

除非绝对必要，不新增用户必须传入的新参数。输入接口保持与现有 `RuntimeMix` 和 `lib/Runtime` 风格一致：

- kernel 源文件路径
- kernel 名
- soc 版本
- 输出目录

### 2. Generality

方案 B 不能绑定 baremix 的固定图结构，也不能写死某个特定算子组合。它必须支持一般性的 mix kernel，即：

- AIC/AIV 混合结构由源码本身决定
- backend 只负责通用编译、链接、pack、launcher 生成
- backend 不对内部算子拓扑做特化假设

### 3. Mergeability

`RuntimeMix` 只是临时隔离区，不是长期架构。模块必须按未来并入 `lib/Runtime` 的方式组织：

- 公共产物模型优先稳定
- 阶段性 backend 与 orchestration 分离
- sample 风格工程生成逻辑不能继续成为长期核心实现

## Recommended Architecture

采用分阶段 pipeline，而不是把全部逻辑塞进单个大类。

### Stable Boundary

对外只保留两层稳定边界：

1. 编译输入配置
2. `MixArtifact`

这样方案 A 和方案 B 可以在同一个外层工具下切换，后续合并回 `lib/Runtime` 时也只需要保留通用边界。

### Core Components

#### 1. `MixCompilerDriver`

职责：

- 接收现有 CLI 和配置对象
- 调度 source analyze、compile、link、pack、artifact assemble
- 作为未来并入 `Runtime::Compiler` 的 mix 分支入口

输入：

- kernel 源路径
- kernel 名
- soc 版本
- 输出目录

输出：

- `MixArtifact`

#### 2. `MixSourceAnalyzer`

职责：

- 从现有输入推导 mix backend 所需的标准命名和编译信息
- 不要求用户额外传 mix 专用参数

输出包括：

- AIC 入口名：`<kernel>_0_mix_aic`
- AIV 入口名：`<kernel>_0_mix_aiv`
- launcher 符号名：`aclrtlaunch_<kernel>`
- `bisheng` 所需宏定义集合
- host stub 模板替换变量

这一层不解析内部图结构，只负责通用命名和编译约定推导。

#### 3. `MixCompileStage`

职责：

- 直接调用 `bisheng`
- 为 AIC/AIV 分别构建编译命令
- 输出 AIC/AIV 对象文件

关键要求：

- 内部注入 `__MIX_CORE_MACRO__=1`
- 分别注入对应 auto-gen 入口宏
- 对 cube/vec 两个分支注入正确 target 宏

#### 4. `MixLinkStage`

职责：

- 直接调用 `ld.lld`
- 将 AIC/AIV 对象完成必要的 `-r`、`-Ttext`、merge
- 输出最终 device object

它只处理 device 侧对象拼装，不负责 host stub 和最终 `.so`。

#### 5. `MixStubGen`

职责：

- 生成通用 host stub 源文件与 launcher 头文件
- 提供稳定的 `aclrtlaunch_<kernel>` 导出符号

这里的“通用”含义是：

- 不绑定 baremix 名称
- 不绑定固定算子图
- 但先沿用 AscendC 当前 mix launcher 的通用 ABI 约定：GM inputs / outputs / workspace / tiling

本阶段不做“从任意源码自动推断 host 参数签名”的更大前端能力。

#### 6. `MixPackStage`

职责：

- 调用 `ascendc_pack_kernel` 或等价 pack 工具
- 将 host stub object 与 merged device object 打包成最终可执行载体
- 产出最终 `.so`

#### 7. `MixArtifactBuilder`

职责：

- 统一收集产物路径
- 组装成当前稳定的 `MixArtifact`

`MixArtifact` 继续至少包含：

- kernel 名
- soc 版本
- 构建目录
- 安装目录
- packed kernel `.so` 路径
- launcher 头文件目录
- host runner 路径
- manifest 路径

#### 8. `MixValidatorBackend`

职责：

- 保持当前 validator 外部 CLI 不变
- 消费 direct backend 产出的 `MixArtifact`
- 在 xvm 上调用 sim 执行并完成结果比对

validator 不是方案 B 的重点改写对象。只要 artifact 模型保持稳定，它应尽量保持不变。

## Data Flow

### Compile Flow

1. `mix-compiler` 解析 CLI
2. 调用 `MixCompilerDriver`
3. `MixSourceAnalyzer` 生成标准命名、宏和模板变量
4. `MixCompileStage` 生成 AIC/AIV 对象
5. `MixLinkStage` 生成 merged device object
6. `MixStubGen` 生成 host stub / launcher 文件
7. `MixPackStage` 生成 packed `.so`
8. `MixArtifactBuilder` 返回 `MixArtifact`
9. `mix-compiler` 打印标准化产物路径

### Validate Flow

1. `mix-validator` 读取 artifact root
2. 定位 runner、launcher header、packed `.so`
3. 准备输入和 golden
4. 在 xvm 上执行 sim
5. 收集输出
6. 输出 diff 指标和 PASS/FAIL

## Backward Compatibility Strategy

方案 A 与方案 B 必须在一段时间内并存，以便对比验证。

建议策略：

- 保留现有 `AscendCMixCompiler` 作为方案 A backend
- 新增 direct backend driver，名称明确区分
- `mix-compiler` 内部通过编译期选择或临时配置切换 backend
- 两个 backend 产出的 `MixArtifact` 保持兼容

这样在 B 方案未稳定前，仍然可以用 A 方案作为回归基线。

## Implementation Milestones

### Milestone 1: Direct Compile Skeleton

目标：

- `MixSourceAnalyzer`
- `MixCompileStage`
- `MixLinkStage`

验收：

- AIC/AIV 对象生成成功
- merged device object 生成成功

### Milestone 2: Stub + Pack

目标：

- `MixStubGen`
- `MixPackStage`

验收：

- 成功生成 launcher 头和 packed `.so`
- 产物结构与当前 `MixArtifact` 兼容

### Milestone 3: End-to-End Sim

目标：

- direct backend 接入 `mix-compiler`
- `mix-validator` 在不改 CLI 语义下消费 direct artifact

验收：

- 在 xvm 上完成 sim 执行
- baremix 用例 PASS

### Milestone 4: Generality Check

目标：

- 追加至少一个不同于 baremix 的 mix kernel 样例

验收：

- 第二个样例也能通过 direct backend 打通
- 证明实现没有偷偷耦合 baremix 固定结构

## Testing Strategy

必须全部在 xvm 中验证。

最低测试链：

1. 构建 `RuntimeMix` 库
2. 构建 `mix-compiler` 和 `mix-validator`
3. direct backend 编译 baremix
4. 生成输入和 golden
5. 执行 sim
6. 校验 PASS

除 baremix 之外，Milestone 4 需要补一个第二样例。

## Merge Plan

当方案 B 稳定后，按下面方式并入 `lib/Runtime`：

- 保留稳定的 artifact model
- 将 direct mix backend 模块迁入 `lib/Runtime`
- 删除仅为方案 A 服务的 sample wrapper 逻辑
- 再决定是否把 `mix-compiler` / `mix-validator` 折叠进现有主工具

因此，本阶段代码组织必须避免：

- 过度依赖 `RuntimeMix` 特有命名
- 把 sample 目录结构写死在关键 backend 中
- 让 validator 依赖某个 backend 私有目录布局

## Risks

### 1. Host Stub ABI Risk

如果当前对 launcher ABI 的抽象不准确，可能出现：

- `.so` 能生成但无法执行
- runner 能启动但 AIC/AIV 协作不正确

缓解方式：

- 用方案 A 产物作为基线对照
- 保持 `MixArtifact` 和 launcher 命名一致

### 2. Source Analysis Risk

如果对 mix 入口命名和宏推导不稳定，可能导致：

- AIC/AIV 编译成功但符号不匹配
- pack 后导出的 launcher 不可用

缓解方式：

- 把 analyzer 限定在“标准命名推导”，不做过度语义分析

### 3. Generality Risk

如果只用 baremix 验证，很容易把 backend 写成“baremix-specific but looks generic”。

缓解方式：

- 将第二样例验证作为里程碑，而不是后续可选项

## Scope Check

该设计聚焦于方案 B 的 direct mix backend，不扩展到 vec/cube 统一编译框架，也不提前处理最终合并提交策略，范围适合单独进入 implementation plan。
