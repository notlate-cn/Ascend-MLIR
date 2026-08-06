# RuntimeMix AscendC Wrapper Design

**Date:** 2026-03-31

**Goal**

在 `RuntimeMix` 中增加一条独立于 `lib/Runtime` 的 mix-kernel toolchain，先通过复用 AscendC CMake 工具链实现“源码 -> sim 可执行产物 -> 执行校验”的完整闭环，并保持未来可替换为直接 `bisheng + lld + pack` 后端。

## Recommended Approach

采用双工具边界：

- `mix-compiler` 只负责编译 orchestration
- `mix-validator` 只负责执行与校验

这比单工具模式更稳，因为：

- 编译产物模型可以先稳定下来
- 执行器可以专门适配 mix 的 ACL launcher 语义
- 后续把方案 A 替换成方案 B 时，只换 compiler backend，不改 validator CLI

## Architecture

### 1. RuntimeMix Compiler Backend

新增一个 `RuntimeMix` 专用的编译驱动层，内部生成临时 AscendC CMake 工程并调用现有 sample 风格工具链。

职责：

- 接收 kernel 源文件、kernel 名、soc、输出目录
- 生成最小 CMake 工程骨架
- 复用 `examples/baremix-test/cmake/npu_lib.cmake`
- 调用 `cmake configure/build/install`
- 收集最终产物路径

它不直接暴露 CMake target 名和内部目录布局给上层。

### 2. RuntimeMix Artifact Model

编译阶段统一输出一个稳定产物描述，而不是让上层直接猜目录结构。

核心字段：

- packed kernel so 路径
- launcher 头文件目录
- 可执行 host runner 路径
- 输出目录
- kernel 名
- soc 版本

这样后面从 AscendC CMake backend 切到 direct backend 时，只要还能产出同一份 artifact model，上层工具无需重写。

### 3. RuntimeMix Validator Backend

新增一个 mix 专用 validator，执行方式对齐当前 `examples/baremix-test` 已验证可跑的 sample-style ACL sim 流程，而不是复用现有 `Runtime/Executor` 的 `.bin + rtKernelLaunch` 直跑模式。

职责：

- 准备输入二进制或 `.npy`
- 动态链接 compiled artifact 对应的 launcher
- 调用 host runner 或等价入口执行 sim
- 收集输出
- 对比 golden，给出 diff 指标

### 4. CLI Split

保留两个独立工具：

- `mix-compiler`
  - 输入：kernel 源码、kernel 名、soc、输出目录
  - 输出：artifact manifest 或标准化路径打印
- `mix-validator`
  - 输入：compiled artifact、输入数据、golden、执行参数
  - 输出：PASS/FAIL、误差指标

这一步不修改现有 `compiler` / `validator`。

## File Layout

新增或修改这些区域：

- `include/RuntimeMix/`
  - 新增 backend 接口与 artifact 定义
- `lib/RuntimeMix/`
  - 新增 AscendC CMake wrapper 实现
  - 新增 mix validator 实现
- `tools/mix-compiler/`
  - 从当前 smoke test 扩成真正 CLI
- `tools/mix-validator/`
  - 新增独立 validator 工具
- `examples/baremix-test/`
  - 作为第一阶段验证用例，仅复用现有可跑 kernel/data/脚本，不再承担 runtime 实现职责

## Data Flow

### Compile Flow

1. `mix-compiler` 解析 CLI
2. 调用 `RuntimeMix` 编译 backend
3. backend 生成临时 CMake 工程
4. backend 执行 AscendC CMake configure/build/install
5. backend 返回标准化 artifact model
6. `mix-compiler` 打印产物路径或写 manifest

### Validate Flow

1. `mix-validator` 读取 artifact model 或命令行指定路径
2. 加载输入与 golden
3. 调用 mix validator backend 执行 sim
4. 收集输出并计算误差
5. 返回标准化结果

## Error Handling

错误分三类：

- 输入错误
  - 源文件不存在
  - kernel 名为空
  - 数据文件缺失
- 编译错误
  - CMake configure/build/install 失败
  - 产物缺失
- 执行错误
  - launcher 加载失败
  - sim 运行失败
  - 输出文件缺失或校验失败

外层工具只暴露摘要错误，详细 stderr/stdout 写入构建目录日志文件，避免终端输出被 CMake 噪声淹没。

## First Milestone

第一阶段只要求以下能力：

- `mix-compiler` 能基于 `examples/baremix-test/baremix_custom.cpp` 产出可执行 mix artifact
- `mix-validator` 能基于该 artifact 在 xvm 上跑通 sim
- `verify_result.py` 等价通过，输出 PASS

非目标：

- 不接入现有 `lib/Runtime`
- 不支持 vec/cube 一般化统一
- 不实现方案 B 的 direct `bisheng` backend
- 不合并到现有 `compiler` / `validator`

## Testing

必须在 xvm 验证，不在 host 验证。

最小验证链：

1. 构建 `AscendCRuntimeMix`
2. 构建 `mix-compiler` 和 `mix-validator`
3. 用 `baremix_custom.cpp` 编译出 artifact
4. 用 baremix 输入和 golden 跑 sim
5. 校验 PASS

## Scope Check

这个 spec 只覆盖方案 A 的 mix 编译和执行闭环，不涉及方案 B 的 direct backend，因此适合单独出一份 implementation plan。
