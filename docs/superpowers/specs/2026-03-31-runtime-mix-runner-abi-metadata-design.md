# RuntimeMix Runner ABI Metadata Design

**Date:** 2026-03-31

## Goal

把当前 `RuntimeMix` direct backend 中 runner 的 baremix 硬编码 ABI，从代码内散落的常量收敛为 artifact/config 驱动的显式 metadata。

本阶段的目标不是“自动推断任意 mix kernel 的 host ABI”，而是：

- 保持当前 baremix 接受面不变
- 把 ABI 细节显式写入 artifact/manifest
- 让 runner 和 validator 从同一份 metadata 消费这些 ABI 信息

这样下一步再扩到更一般的 mix 结构时，不会继续被 baremix 常量绑死。

## Scope

### In Scope

- 为当前 baremix 接受面引入显式 ABI metadata
- 修改 `RuntimeMix` direct backend 生成并消费这份 metadata
- 修改 runner/validator 路径，使其不再直接硬编码 baremix 文件名、shape、dtype、workspace、tiling 细节
- 保持 xvm 上 `examples/baremix-test/run.sh` 通过

### Out of Scope

- 不自动从 AscendC 源码推断任意 host ABI
- 不支持任意 mix kernel 图结构
- 不修改 `lib/Runtime`
- 不并回 `RuntimeMix` 到 `Runtime`

## Current Problem

虽然当前 baremix 在 xvm 上已经通过，但 runner 仍然把这些内容写死在代码里：

- 输入文件名
- 输入 shape / dtype
- 输出 shape / dtype
- 固定 workspace 大小
- 固定 tiling 来源与字节生成方式

与此同时，manifest 里目前只有粗粒度字段，例如：

- `abi_inputs=3`
- `abi_outputs=1`
- `abi_workspace_mode=fixed`
- `abi_tiling_mode=fixed_bytes`

这不够支撑 runner 和 validator 脱离 baremix 常量。

## Recommended Approach

### Approach A: 继续局部硬编码，只把常量集中一下

优点：

- 改动最少

缺点：

- ABI 仍然不在 artifact 中
- 只是把硬编码换了位置
- 后续泛化仍然困难

### Approach B: 显式 ABI metadata 下沉，推荐

把 baremix 当前需要的 host ABI 细节显式写入 artifact/manifest，然后让 runner/validator 读取并消费这份 metadata。

优点：

- 去掉分散的 baremix 常量
- runner 和 validator 行为可以由 artifact 驱动
- 为下一阶段真正泛化打基础

缺点：

- 需要定义一份小型 metadata 模型

### Approach C: 直接做自动推断任意 host ABI

优点：

- 一步到位

缺点：

- 范围过大
- 当前没有足够稳定的前端基础

结论：先做 Approach B。

## Design

### 1. ABI Metadata Model

为当前 baremix 接受面引入一份显式 ABI 描述，至少覆盖这四部分：

1. `inputs`
2. `outputs`
3. `workspace`
4. `tiling`

建议结构如下：

- `inputs`
  - `name`
  - `file`
  - `dtype`
  - `shape`

- `outputs`
  - `name`
  - `file`
  - `dtype`
  - `shape`

- `workspace`
  - `mode`
  - `bytes`

- `tiling`
  - `mode`
  - `source`

本阶段 `source` 可以是：

- `fixed_bytes`
- 或者更具体的 baremix 标识

但不能再只留一句模糊描述而不记录细节。

### 2. Artifact / Manifest Emission

`MixDirectBackend` 在编译完成后，除了现有字段，还要把这份 ABI metadata 写入 artifact/manifest。

要求：

- metadata 来自一个集中构造点
- 不允许 runner 和 validator 再分别写一份自己的 baremix 常量

本阶段可以继续使用 manifest 文本，不强制引入 JSON 文件，只要信息表达清楚且可稳定解析。

### 3. Runner Generation

runner 生成逻辑必须改成消费 ABI metadata，而不是直接写死：

- `x1_gm.bin`
- `x2_gm.bin`
- `bias.bin`
- 输出 shape / dtype
- workspace bytes
- tiling 生成方式

注意：

- 这不要求 runner 变成真正的通用解释器
- 只要求它的当前 baremix 行为由 metadata 驱动

也就是说，本阶段可以继续生成“面向当前 ABI 的 runner”，但它必须从 metadata 注入，而不是从散落常量注入。

### 4. Validator Alignment

`mix-validator` 的 runner-first 路径继续保留，但 direct fallback 也必须读取同一份 ABI metadata，避免：

- runner 走一套 baremix 常量
- fallback 再复制另一套 baremix 常量

本阶段允许 direct fallback 仍然只支持当前 baremix 接受面，但要求：

- ABI 细节来自 artifact/manifest
- 不再直接在 validator 里写死 shape/file/dtype

### 5. Documentation

README 和 manifest 中的 ABI 描述要同步更新：

- README 说明当前是“metadata-driven baremix ABI”
- manifest 从粗粒度升级到足够支撑 runner/validator 的明细字段

## Validation

必须在 xvm 上验证：

1. `bash examples/baremix-test/run.sh` 仍然通过
2. `mix-validator` 输出 `PASS`
3. `verify_result.py` 输出 `test pass`
4. manifest 中能看到具体的 input/output/workspace/tiling metadata
5. runner 和 fallback 不再直接依赖原来的 baremix shape/file 常量副本

## Risks

### 1. 只换了 manifest，没有真正消掉代码硬编码

如果 runner/validator 最后仍然保留一份独立 baremix 常量，那这一步没有达到目标。

### 2. metadata 设计过大

本阶段只覆盖当前 baremix 接受面，不要把“未来所有 mix kernel”一起设计进来。

### 3. validator fallback 和 runner 脱节

这两个路径必须消费同一份 ABI 描述，否则后面仍然会分叉。

## Decision

下一步按以下顺序实现：

1. 定义当前 baremix 所需的显式 ABI metadata
2. 在 `MixDirectBackend` 中集中生成并写入 manifest
3. 让 runner 生成改为消费 metadata
4. 让 direct fallback 也消费同一份 metadata
5. 在 xvm 上回归 baremix 验证
