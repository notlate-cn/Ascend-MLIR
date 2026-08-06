# RuntimeMix Validator Portability Design

**Date:** 2026-03-31

## Goal

收敛 `tools/mix-validator/mix_validator_main.cpp` 中的运行时环境探测逻辑，使其对齐 `lib/Runtime` 现有的通用风格，去掉 `/home/niu/...` 这类机器私有 fallback，同时不改变当前 baremix 在 xvm 上的验证行为。

这个阶段只解决可移植性问题，不处理 runner ABI 的 baremix 硬编码。

## Scope

### In Scope

- 修改 `tools/mix-validator/mix_validator_main.cpp` 中的 toolkit/runtime 路径探测
- 参考 `lib/Runtime/Executor.cpp` / `lib/Runtime/HostRunnerGen.cpp` 的探测模式
- 保持 `examples/baremix-test/run.sh` 在 xvm 上可继续通过
- 保持 `mix-validator` CLI 不变

### Out of Scope

- 不修改 `lib/Runtime`
- 不修改 `lib/RuntimeMix` 的 runner ABI
- 不修改 `mix-compiler`
- 不做通用公共 helper 抽取到共享库层

## Current Problem

当前 `mix-validator` 里仍有多条 `/home/niu/...` 私有 fallback。这样的问题是：

- 行为依赖当前开发机布局
- 换一台 xvm、换 toolkit 安装位置，可能 silently 命中错误路径
- 与 `lib/Runtime` 里已经存在的更通用探测风格不一致

这会影响后续把 `RuntimeMix` 并回 `lib/Runtime`。

## Existing Runtime Pattern

`lib/Runtime` 里最值得对齐的是：

- `lib/Runtime/Executor.cpp`
  - 先依赖 `ASCEND_HOME_PATH`
  - 再兼容 `<arch>/simulator/<SOC>/lib`
  - 再兼容 `tools/simulator/<SOC>/lib`

- `lib/Runtime/HostRunnerGen.cpp`
  - 也是按 `ASCEND_HOME_PATH` + 多布局兼容思路组织

这说明正确方向不是继续写机器私有路径，而是：

1. 环境变量优先
2. 标准安装布局探测
3. 找不到时明确报错

## Recommended Approach

### Approach A: 只保留环境变量

优点：

- 最干净
- 最不容易误判

缺点：

- 需要每次都手动保证环境变量已设置
- 当前 `xvm` 使用体验会退步

### Approach B: 环境变量优先 + 标准布局探测 + 明确报错（推荐）

顺序：

1. `ASCEND_HOME_PATH`
2. `ASCEND_TOOLKIT_HOME`
3. `$HOME/Ascend/latest`
4. `$HOME/Ascend/ascend-toolkit/latest`
5. `/usr/local/Ascend/ascend-toolkit/latest`

然后在该 root 下继续探测：

- `lib64`
- `<arch>/simulator/<SOC>/lib`
- `tools/simulator/<SOC>/lib`

优点：

- 去掉机器私有路径
- 保留当前 xvm 可用性
- 行为和 `lib/Runtime` 的现有风格一致

### Approach C: 直接把这套探测抽成共享 helper

优点：

- 最终形态更好

缺点：

- 范围超出当前目标
- 会碰到 `lib/Runtime` / `RuntimeMix` 的共享边界设计

结论：先做 Approach B。

## Design

### 1. `findAscendHome()`

去掉所有 `/home/niu/...` 私有候选，保留：

- `ASCEND_HOME_PATH`
- `ASCEND_TOOLKIT_HOME`
- `$HOME/Ascend/latest`
- `$HOME/Ascend/ascend-toolkit/latest`
- `/usr/local/Ascend/ascend-toolkit/latest`

要求：

- 继续检查该路径下是否有运行时库
- 如果全部失败，返回明确错误，而不是静默猜路径

### 2. `findAscendLib64()`

在确定 `ascendHome` 后，只按标准布局探测：

- `${ascendHome}/lib64`
- `${ascendHome}/aarch64-linux/lib64`
- `${ascendHome}/arm64-linux/lib64`

不再额外枚举开发机私有目录。

### 3. `findSimulatorLibDir()`

只保留标准 simulator 布局：

- `${ascendHome}/aarch64-linux/simulator/${soc}/lib`
- `${ascendHome}/arm64-linux/simulator/${soc}/lib`
- `${ascendHome}/tools/simulator/${soc}/lib`

同样不再列出 `/home/niu/...` 私有候选。

### 4. Error Handling

找不到路径时，要输出明确错误，例如：

- 找不到 `ASCEND_HOME`
- 找不到 `lib64`
- 找不到 simulator `lib`

而不是后面在 `dlopen` 或 simulator 启动时才模糊失败。

## Validation

必须在 xvm 上验证：

1. 强制通过标准探测找到 toolkit，而不是机器私有 fallback
2. `bash examples/baremix-test/run.sh` 仍然通过
3. `mix-validator` 仍然输出 `PASS`
4. `verify_result.py` 仍然输出 `test pass`

如果需要，可以在验证时打印最终命中的：

- `ascendHome`
- `ascendLib64`
- `simulatorLibDir`

但这些调试输出不应长期污染用户默认输出。

## Risks

### 1. xvm 当前正好依赖某条私有路径

如果删掉 `/home/niu/...` 后失败，说明真正需要的是补标准探测，而不是把私有路径加回来。

### 2. `mix-validator` 和 `Runtime` 风格仍然轻微不一致

本阶段接受“风格对齐”，不要求做到共享 helper。

## Decision

下一步按以下顺序实现：

1. 去掉 `mix-validator` 中的 `/home/niu/...` 私有 fallback
2. 对齐到 `Runtime` 风格的标准路径探测
3. 在 xvm 上回归 baremix 验证

等这一步完成后，再进入 runner ABI 从 baremix 特化收敛到 artifact/config 驱动。
