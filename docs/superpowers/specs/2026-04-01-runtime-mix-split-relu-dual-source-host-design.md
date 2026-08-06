# RuntimeMix Split-ReLU Dual-Source Host Design

**Goal**

让 `fc_relu_split` 第二样例在保持当前 device 路径稳定的前提下，补回官方 host launcher 生成链，验证 RuntimeMix 能产出同时包含：

- 正确的 packed mix device section
- 正确的 `aclrtlaunch_fc_relu_split`

的最终 `.so`。

本阶段只针对 `fc_relu_split` 第二样例，不做通用 mix 自动推断。

---

## Problem Statement

当前证据已经把问题范围收缩得很清楚：

1. `fc_relu_split` 的 device 侧已经对齐官方 control。
   - RuntimeMix 和官方 wrapperless control 的 AIC/AIV/merged `device.o` 在 strip debug 后逐字节一致。
   - 所以问题不在 device compile 或 merge。

2. host 侧 launcher 生成链没有对齐官方。
   - `BareMixInvocation` 这类官方可运行样例会生成非空的 `host_dir/.../host_bisheng_obj/.../*.cpp.o`。
   - `recompile_binary.py` 会把 `host_dir` 里的额外 host objects 重新链接进最终 `libascendc_kernels_sim.so`。
   - 最终 `.so` 中因此出现 `aclrtlaunch_<kernel>`。

3. `fc_relu_split_wrapperless.cpp` 虽然能作为 device 侧编译基线，但它去掉了原始样例里的手写 wrapper：
   - 原始样例有 `auto_gen_fc_relu_split_kernel(...)`
   - wrapperless 版本只保留 `fc_relu_split(...)`
   - 结果是官方 host_bisheng 路径失去可识别的 launcher 入口，`host_dir` 为空，最终 `.so` 不含 `aclrtlaunch_*`

结论：

`fc_relu_split` 需要两套源码视图：

- device 视图：wrapperless，避免 device preprocess/auto-gen 冲突
- host 视图：原始 mix 源码，保留 launcher wrapper，供 host_bisheng 生成 host object

---

## Proposed Design

### 1. Dual-Source Policy For Split-ReLU

只对 `fc_relu_split` 路径引入 dual-source policy：

- device preprocess / AIC / AIV compile：
  - 输入继续使用 `fc_relu_split_wrapperless.cpp`
- host launcher compile：
  - 输入改用原始 `fc_relu_split_mix.cpp`

这个策略只存在于 RuntimeMix 的 split-relu 样例分支，不影响：

- baremix
- fc_leakyrelu
- 现有 manual mix 分支

### 2. Add Explicit Host Compile Stage

在 `MixDirectBackend` 中新增一条 host compile 子链，语义对齐官方 sample：

1. 使用原始 host source
2. 使用 `bisheng`
3. 使用官方 host flags：
   - `-DTILING_KEY_VAR=0`
   - `-include <...>/aclrtlaunch_triple_chevrons_func.h`
   - `--cce-host-only`
   - `-fcce-kernel-launch-custom`
   - `-DONE_CORE_DUMP_SIZE=1048576`
4. 输出 host object 到 RuntimeMix 自己的 `host_dir`

本阶段不尝试泛化为任意 mix kernel 的 host-source selector。只给 split-relu 明确指定：

- device source path
- host source path

### 3. Add Recompile Step

在当前 `pack + host link` 之后，新增官方等价的 recompile step：

- 复用 toolkit 的 `recompile_binary.py`
- `--root-dir` 指向 RuntimeMix 当前 artifact root
- `--target-name` 使用当前 target 名
- `--add-dir` 指向 RuntimeMix 生成的 host_dir

目标效果是把 host_bisheng 产出的 host object 重新注入最终 `.so`，得到：

- packed kernel registration section
- `aclrtlaunch_fc_relu_split`

### 4. Artifact/Manifest Extensions

manifest 需要显式记录 dual-source host 信息，至少补充：

- `host_source_path`
- `host_bisheng_object`
- `host_bisheng_cmd`
- `recompile_cmd`
- `host_object_dir`

这样后续再 debug 时，不需要反推 RuntimeMix 内部决策。

---

## Files To Modify

### Core RuntimeMix

- `include/RuntimeMix/MixCommandBuilder.h`
- `lib/RuntimeMix/MixCommandBuilder.cpp`
- `lib/RuntimeMix/MixDirectBackend.cpp`

### Example / Docs

- `examples/relu-split-mix-test/README.md`

不修改：

- `lib/Runtime`
- `tools/compiler`
- `tools/validator`
- 原始 `fc_relu_split_mix.cpp`
- `fc_relu_split_wrapperless.cpp`

---

## Validation Plan

### Stage 1: Structure Validation

在 xvm 上验证 RuntimeMix split-relu artifact 现在同时产出：

- device object
- packed `.so`
- `host_dir/.../*.o`
- manifest 中的 host_bisheng / recompile 记录

### Stage 2: Symbol Validation

验证最终 `.so` 出现：

- `aclrtlaunch_fc_relu_split`

并且不只是 `__ascend_kernel_...`

### Stage 3: Execution Validation

仍然只以第二样例入口验收：

```bash
bash examples/relu-split-mix-test/run.sh
```

目标：

- runner 路径通过
- `--force-direct-packed` 行为与当前 validator 语义一致
- 精度通过

---

## Non-Goals

本阶段不做：

- 任意 mix kernel 的通用 dual-source 自动推断
- 把 RuntimeMix 直接并回 `lib/Runtime`
- 重写手工 stub 模板体系
- 调整 baremix 已通过路径

---

## Risks

### 1. Host source 仍可能触发原始 wrapper 冲突

缓解：

- host 路径不走 device preprocess generated source
- host_bisheng 直接吃原始 source，仿照官方 sample

### 2. RuntimeMix target 布局与 `recompile_binary.py` 预期不一致

缓解：

- 严格按官方脚本需求组织：
  - `link.txt`
  - `host_dir`
  - target name

### 3. 即便补上 host_bisheng + recompile，split-relu 仍可能有样例级 simulator 问题

缓解：

- 先以 `.so` 是否出现 `aclrtlaunch_fc_relu_split` 作为中间验收
- 如果符号恢复但执行仍失败，再继续看 launcher/runtime 层

---

## Success Criteria

本阶段完成的标准是：

1. RuntimeMix split-relu artifact 中出现非空 `host_dir` host object
2. 最终 `.so` 出现 `aclrtlaunch_fc_relu_split`
3. xvm 上 `bash examples/relu-split-mix-test/run.sh` 通过

如果只做到 1 和 2，没有做到 3，本阶段仍然算“部分收敛，但未完成”。
