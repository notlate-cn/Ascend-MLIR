# RuntimeMix Split-ReLU Official-Style Device Compile Design

**Date:** 2026-04-01

## Goal

为 `fc_relu_split_wrapperless.cpp` 这条第二样例路径新增一条
official-style device compile command，使 preprocess-generated source
不再复用当前手工 mix 宏路径，而是尽量对齐官方 AscendC sample-style
CMake 的 device 编译语义。

## Why

当前 root cause 已经收敛：

- 官方 wrapperless 基线的 AIC/AIV `device.o` 都是 `36464` 字节，merged
  `device.o` 是 `43952` 字节
- `RuntimeMix` 当前这条路径的 AIC/AIV 对象只有 `1120` 字节，merged
  `device.o` 只有 `1520` 字节
- 所以后续 `RegisterAscendBinary failed: 107000` 只是后果，真正问题在
  device compile 语义

官方 `flags.make` 已经说明对 preprocess-generated source 的 device 编译
语义与当前 `buildBishengCommand(...)` 不同：

- 官方只用 `HAVE_TILING/HAVE_WORKSPACE/TILING_KEY_VAR` 和一组 `asc/tikcfw`
  include
- 不继续注入 `__MIX_CORE_MACRO__`、`auto_gen_*=`、`__DAV_C220_*__`

## Scope

### In Scope

- 在 `MixCommandBuilder` 中新增 official-style device compile builder
- 只在 split-relu wrapperless 的 preprocess-generated source 路径使用
- 保持 host stub / pack / runner / validator 逻辑不变
- 在 xvm 上验证 device object 尺寸和注册行为是否向官方基线收敛

### Out of Scope

- 不重构整个 `RuntimeMix` backend
- 不泛化到所有 mix kernel
- 不处理原始 `fc_relu_split_mix.cpp` 的 wrapper 兼容
- 不改 `lib/Runtime`

## Approaches

### Approach A: 新增 official-style device compile builder，推荐

在 `MixCommandBuilder` 中新增一条专门给 preprocess-generated source 用的
builder，直接复刻官方 `flags.make` 的核心语义：

- `CXX_DEFINES`: `HAVE_TILING/HAVE_WORKSPACE/TILING_KEY_VAR`
- `CXX_INCLUDES`: `asc/...` + `tikcfw/...`
- `CXX_FLAGS`: `--cce-aicore-arch=dav-c220-{cube,vec}` 等官方 flags
- 不注入当前手工 mix 宏

优点：
- 根因对齐
- 改动面小
- 不污染 baremix 已工作的路径

缺点：
- 当前先是 sample-specific 路径，不是最终通用抽象

### Approach B: 继续在现有 `buildBishengCommand(...)` 上加 if 分支

优点：
- 文件改动更少

缺点：
- 会把两种本质不同的编译语义揉在一个 builder 里
- 后面更难维护

### Approach C: 直接调用官方 device sub-build

优点：
- 语义最接近官方

缺点：
- 会把 direct backend 重新拉回外部 CMake 黑盒

结论：走 Approach A。

## Design

### 1. Add a Separate Builder

在：
- `include/RuntimeMix/MixCommandBuilder.h`
- `lib/RuntimeMix/MixCommandBuilder.cpp`

新增一个 builder，例如：

- `buildPreprocessedDeviceCompileCommand(...)`

它只接受：
- generated source path
- output object path
- core type

内部按官方 `flags.make` 组织：
- `bisheng`
- `-DHAVE_TILING -DHAVE_WORKSPACE -DTILING_KEY_VAR=0`
- `appendAscIncludes(...)`
- `appendTikcppIncludes(...)`
- `-g --cce-disable-kernel-global-attr-check`
- `--cce-aicore-arch=dav-c220-{cube,vec}`
- `--cce-aicore-only --cce-auto-sync`
- `-mllvm -cce-aicore-stack-size=0x8000`
- `-mllvm -cce-aicore-function-stack-size=0x8000`
- `-mllvm -cce-aicore-record-overflow=true`
- `-mllvm -cce-aicore-addr-transform`
- `-mllvm -cce-aicore-dcci-insert-for-scalar=false`
- `-O3 -std=c++17 --cce-aicore-lang`
- `-include <asc_devkit_version.h>`

### 2. Use It Only For Preprocessed Split-ReLU Path

在 `MixDirectBackend.cpp` 中：
- 当 kernel 是 split-relu 且走 preprocess-generated source 时
- AIC/AIV compile command 改用新 builder
- 其他路径继续使用现有 `buildBishengCommand(...)`

### 3. Validation

先验证两层：

1. 编译产物收敛：
- `fc_relu_split_aic.o`
- `fc_relu_split_aiv.o`
- merged `device.o`

尺寸应明显接近官方基线，不再是 `1120/1120/1520`

2. 执行收敛：
- xvm 上重新跑 `bash examples/relu-split-mix-test/run.sh`
- 观察 `RegisterAscendBinary failed: 107000` 是否消失
- 若仍失败，再继续查下一层

## Risks

### 1. 尺寸接近不代表执行一定通过

这一步主要验证 device compile 语义是否被拉回正确方向；即使对象尺寸回到官方量级，后面仍可能有 merge/pack 差异。

### 2. 这仍然是 sample-aware 路径

当前只为 split-relu wrapperless 路径加 official-style device compile，后续如果要并回 `lib/Runtime`，还需要再抽象。

## Decision

下一步执行顺序：

1. 新增 official-style device compile builder
2. 只在 split-relu preprocess-generated path 上使用
3. 在 xvm 上重跑并先对比 object size
4. 再判断注册错误是否消失
