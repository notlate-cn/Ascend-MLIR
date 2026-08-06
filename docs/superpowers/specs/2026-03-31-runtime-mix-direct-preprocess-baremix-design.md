# RuntimeMix Direct Backend Preprocess Baremix Design

## Goal

在不修改 `lib/Runtime` 的前提下，把 `RuntimeMix` 的 B 方案推进到一个可验收状态：

- `mix-compiler` 不再直接对原始 `baremix_custom.cpp` 做 AIC/AIV 双编译
- 改为先走 toolkit 的 preprocess/host-stub 生成链
- 再继续使用 `RuntimeMix` 自己的 direct compile/merge/pack/validate 路径
- 最终在 xvm 上通过 `examples/baremix-test` 的 baremix 用例，并达到精度一致

本阶段的验收目标只有一个：`baremix-test` 在 B backend 下跑通且精度通过。通用化、非 baremix 样例支持和接口并回 `lib/Runtime` 都不在本阶段范围内。

## Context

当前 `RuntimeMix` direct backend 已经打通了这些部分：

- `mix-compiler` / `mix-validator` 的 CLI 边界
- `bisheng` AIC/AIV 编译
- toolkit `merge_obj.sh` / `merge_mix_obj.sh` 的接入方向
- packed `.so` 输出
- runner 与 validator 的基本执行框架

当前未打通的根因也已经明确：

- 直接对原始 `baremix_custom.cpp` 做 AIC/AIV 编译时，两边对象都会导出同名 `baremix_custom`
- 一旦切到 toolkit 官方 merge 逻辑，xvm 上会在 final merge 时触发 duplicate symbol
- 这说明问题不在 validator，也不在 pack，而在 compile 前缺少 toolkit 的 preprocess/split-source 生成阶段

已在 xvm 上验证过，toolkit 的 `extract_host_stub.py` 可以在不走 legacy CMake 的情况下工作，只要提供：

- 一个 `bisheng -E` 生成的预处理产物
- AIC/AIV 对象路径
- 一个最小 `compile_commands.json`

它会生成：

- `auto_gen_baremix_custom.cpp`
- `host_stub.cpp`
- `aic_config.cmake`
- `aiv_config.cmake`
- `host_config.cmake`
- `aclrtlaunch_baremix_custom.h`

这说明 B backend 的正确下一阶段不是继续修手写 `MixStubTemplate`，而是把 toolkit preprocess/stub generation 正式纳入 direct backend。

## Scope

### In Scope

- 在 `RuntimeMix` 内新增 preprocess 子阶段
- 复用 toolkit 脚本：
  - `extract_host_stub.py`
  - `update_host_stub.py`
- 让 direct backend 使用 toolkit 生成的：
  - `auto_gen_*.cpp`
  - `host_stub.cpp`
  - `aclrtlaunch_*.h`
  - `aic_config.cmake`
  - `aiv_config.cmake`
- 让 baremix 用例在 xvm 上通过编译、sim 执行和精度校验
- 尽量保持 `mix-compiler` / `mix-validator` CLI 不变

### Out of Scope

- 通用 mix kernel 泛化支持
- 多样例兼容性验证
- `RuntimeMix` 并回 `lib/Runtime`
- 删除 A backend
- 清理所有遗留手写模板代码

## Design Options

### Option 1: Full legacy CMake embedding

把 legacy CMake 的 preprocess/device/host 三段都嵌入 `RuntimeMix`。

优点：

- 与 toolkit 最接近
- 最不容易漏掉隐藏步骤

缺点：

- 会把 B backend 做成半个 CMake wrapper
- 与“未来并回 `lib/Runtime`”的目标冲突
- 调试和接口收敛都会更差

### Option 2: Minimal toolkit preprocess reuse

只复用 toolkit 中必须的 preprocess/stub generation 步骤，其余 compile/merge/pack/executor 继续留在 `RuntimeMix` direct backend。

优点：

- 保留 B backend 的 direct pipeline 结构
- 只引入目前已验证必需的工具链语义
- 便于后续再做通用化

缺点：

- 需要自行解析 `aic_config.cmake` / `aiv_config.cmake`
- 需要自己生成最小 `compile_commands.json`

### Option 3: Continue hand-written equivalence

继续扩展 `MixStubTemplate` 和 hand-written split logic，直到行为等价 toolkit。

优点：

- 理论上最“纯”

缺点：

- 已经验证投入高、回报低
- 当前已知缺的是 toolkit 生成语义，不适合继续硬拗

### Recommendation

采用 Option 2。

这是本阶段最短路径，也最符合 `RuntimeMix` 作为临时隔离开发区、未来并回 `lib/Runtime` 的目标。它把真正不可缺少的 toolkit 逻辑局限在 preprocess/stub generation，而不把整个 direct backend 退化成 CMake wrapper。

## Architecture

本阶段在现有 B backend 上增加一个新子阶段：

1. `MixPreprocessStage`
2. `MixDeviceCompileStage`
3. `MixDeviceMergeStage`
4. `MixPackStage`
5. `MixRunner/ValidatorStage`

其中变化最大的是前两段。

### MixPreprocessStage

输入：

- 原始 kernel 源文件路径
- kernel name
- soc version
- output root

输出：

- 预处理产物路径
- `compile_commands.json`
- `auto_gen_*.cpp`
- `host_stub.cpp`
- `aclrtlaunch_*.h`
- `aic_config.cmake`
- `aiv_config.cmake`
- `host_config.cmake`

具体流程：

1. 用 `bisheng -E` 生成预处理产物
2. 生成最小 `compile_commands.json`
3. 调 `extract_host_stub.py`
4. 调 `update_host_stub.py`

实现上不依赖 legacy CMake，只直接调用 toolkit 脚本和 `bisheng`。

### MixDeviceCompileStage

输入不再是原始 `baremix_custom.cpp`，而是 `MixPreprocessStage` 生成的 `auto_gen_baremix_custom.cpp`。

编译参数不再手写 AIC/AIV 命名宏，而是从 `aic_config.cmake` / `aiv_config.cmake` 解析：

- `MIX_SOURCES`
- 对应 source 的 `COMPILE_DEFINITIONS`

这一步的目标不是通用 CMake parser，而是一个只覆盖本阶段产物格式的最小 parser：

- 解析 `set(MIX_SOURCES ...)`
- 解析 `set_source_files_properties(... COMPILE_DEFINITIONS "...")`
- 把 `;` 分隔的 compile definitions 还原成 `-D...`

### MixDeviceMergeStage

保留官方 merge 脚本：

- `merge_obj.sh`
- `merge_mix_obj.sh`

但输入改成 `auto_gen_*.cpp` 生成出来的 AIC/AIV 对象。这样 final merge 才能避免当前 duplicate symbol 的问题。

### MixPackStage

不再使用手写 `MixStubTemplate` 作为主路径。

host stub 直接使用 toolkit 生成的 `host_stub.cpp`，并在 `update_host_stub.py` 之后进入：

- host stub compile
- `ascendc_pack_kernel.sh`
- host `.so` link

现有 hand-written `MixStubTemplate` 保留为过渡代码，但本阶段不再作为 baremix 主路径。

### MixRunner / ValidatorStage

外部接口保持不变：

- `mix-compiler`
- `mix-validator`

runner 路径和 validator 路径都继续消费 `MixArtifact`。

本阶段不额外扩展 direct validator 的通用能力；目标只是在 baremix 用例上恢复正确行为并过精度。

## File Changes

### Files to Modify

- `include/RuntimeMix/MixCommandBuilder.h`
  - 增加 preprocess/script command builder 接口
- `lib/RuntimeMix/MixCommandBuilder.cpp`
  - 新增 `bisheng -E`、`extract_host_stub.py`、`update_host_stub.py`、最小 `compile_commands.json` 辅助命令构造
- `include/RuntimeMix/MixDirectBackend.h`
  - 如有需要，补充 preprocess artifact 内部结构声明
- `lib/RuntimeMix/MixDirectBackend.cpp`
  - 接入 preprocess 阶段
  - 切换 device compile 输入到 `auto_gen_*.cpp`
  - 切换 host stub 输入到 toolkit 生成的 `host_stub.cpp`
  - 解析 `aic_config.cmake` / `aiv_config.cmake`
- `tools/mix-compiler/mix_compiler_main.cpp`
  - 不改 CLI，只在必要时更新输出字段
- `tools/mix-validator/mix_validator_main.cpp`
  - 只做必要适配，不改 CLI 语义

### Files That May Be Deprecated But Not Removed In This Phase

- `include/RuntimeMix/MixStubTemplate.h`
- `lib/RuntimeMix/MixStubTemplate.cpp`

这两者在 baremix 主路径中不再作为首选实现，但本阶段不主动删除，避免扩大改动面。

## Data Flow

### Compiler Flow

1. `mix-compiler` 接收原始 kernel path
2. `MixDirectBackend` 调 `MixPreprocessStage`
3. `MixPreprocessStage` 生成 split source / host stub / headers / configs
4. `MixDirectBackend` 解析 `aic_config.cmake` / `aiv_config.cmake`
5. 用解析结果编 AIC/AIV device objects
6. 用官方 merge 脚本生成 final `device.o`
7. 编译 toolkit 生成的 `host_stub.cpp`
8. pack device object 到 host stub
9. link 得到 packed `.so`
10. 继续产出 host runner 和 manifest

### Validator Flow

validator 对外不变：

1. 读取 manifest
2. 优先尝试 direct packed path
3. 失败时回退到 runner path
4. 比较输出与 golden

本阶段要求 baremix 最终至少有一条路径稳定通过并精度一致。

## Error Handling

本阶段错误要尽量在最接近根因的位置暴露：

- preprocess 失败：直接报 `bisheng -E` 或 toolkit 脚本失败
- config parse 失败：明确报是 `aic_config.cmake` 还是 `aiv_config.cmake` 内容不符合当前 parser 预期
- merge 失败：保留官方 merge 脚本 stderr
- pack / register 失败：继续保留 packed `.so`、manifest、generated sources 供 xvm 上复现

不要在本阶段引入复杂 recovery 逻辑。目标是先把 baremix 主路径跑通。

## Testing

### Primary Acceptance Test

在 xvm 上执行：

```bash
cd /home/niu/code/Ascend-MLIR
bash examples/baremix-test/run.sh
```

成功标准：

- `mix-compiler` 成功
- `mix-validator` 成功
- `verify_result.py` 输出 `test pass`
- `golden.bin` 与 `actual.bin` 一致

### Secondary Diagnostic Checks

需要保留这些调试检查能力：

- AIC/AIV 对象不再都导出同名 `baremix_custom`
- `merge_mix_obj.sh` 不再报 duplicate symbol
- manifest 中记录 preprocess/generated 产物路径

## Risks

### Risk 1: `extract_host_stub.py` 仍隐含依赖更多 CMake 语义

缓解：

- 目前已在 xvm 上用最小输入成功生成产物，说明这条风险可控
- 本阶段先只覆盖 baremix，避免过早泛化

### Risk 2: `aic_config.cmake` / `aiv_config.cmake` 解析器过于脆弱

缓解：

- 只做覆盖当前产物格式的最小 parser
- parser 失败时明确报错，不静默 fallback

### Risk 3: 即使 preprocess 接入后，direct packed path 仍失败

缓解：

- 本阶段验收不是“必须 direct path 成功”，而是“B backend 最终过精度”
- 允许先靠 runner path 过验收，只要 packed artifact 确实由 B backend 生成

## Success Criteria

本阶段完成的判定标准：

1. `MixDirectBackend` 不再直接编译原始 `baremix_custom.cpp`
2. `MixDirectBackend` 正式消费 toolkit 生成的 preprocess/split-source/stub 产物
3. xvm 上 `bash examples/baremix-test/run.sh` 通过
4. 精度验证通过

## Deferred Work

完成 baremix 后再进入下一阶段：

- 抽象通用 preprocess artifact 模型
- 支持更多 mix kernel 结构
- 清理 hand-written template 路径
- 评估何时把 `RuntimeMix` 并回 `lib/Runtime`
