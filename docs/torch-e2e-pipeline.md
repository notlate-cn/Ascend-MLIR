# Torch E2E Pipeline: 端到端编译验证流程

## 概述

Torch E2E 测试框架将 PyTorch 模型通过 MLIR 多层 lowering 编译为 Ascend NPU C++ kernel，
再通过 autotuner 在模拟器上搜索最优 tiling 配置并验证数值正确性。

```
PyTorch Module
  │  torch.export + torch-mlir
  ▼
step0: linalg on tensors (动态 shape)
  │  afir-opt passes (stage 0b-7b)
  ▼
step8: C++ kernel + tiling_space.json
  │  autotuner (compile → search → verify)
  ▼
tiling_func.cpp (最优配置) + 数值验证 PASS/FAIL
```

## 目录结构

```
python/torch/
  torch2linalg/              # 纯转换库
    convert.py                # torch_to_linalg(): PyTorch → linalg MLIR
  framework/                  # 测试框架
    tensor_spec.py            # TensorSpec: 动态/静态维度描述
    pipeline.py               # @torch_e2e_test 装饰器 + MLIR pipeline + autotuner 调用

tests/torch_e2e/
  conftest.py                 # 环境变量 (ASCEND_HOME_PATH, PATH, LD_LIBRARY_PATH)
  test_broadcast_add_reduce.py
  test_elementwise.py

output/torch_e2e/<test_name>/ # 每个测试的全部中间产物
```

## 测试用例定义

```python
from framework import torch_e2e_test, TensorSpec

@torch_e2e_test
def test_broadcast_add_reduce():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)
    return Model(), [TensorSpec(("M",)),       # a: shape (M,)
                     TensorSpec(("M", "N"))]   # b: shape (M, N)
```

### TensorSpec 维度类型

| 类型 | 示例 | 含义 | IR 中 | 默认大小 |
|------|------|------|-------|---------|
| `int` | `32` | 静态维度 | 固定常量 | 不适用 |
| `str` | `"M"` | 命名动态维度（同名共享 Dim） | `?` | 64 |
| `None` | `None` | 匿名动态维度（各自独立） | `?` | 64 |

---

## 编译阶段：各 Stage 产物

### Stage 0: torch → linalg

```
torch.nn.Module → torch.export (FX graph) → torch-mlir → linalg on tensors
```

- `TensorSpec.dynamic_dims()` 构建 `dynamic_shapes`，告诉 `torch.export` 哪些维度是动态的
- `TensorSpec.make_sample()` 生成 dummy tensor（shape 64），仅用于 trace
- 调用 `torch_mlir.fx.export_and_import()`，func_name 固定为 `"kernel"`
- **产物**: `step0_linalg.mlir`

### Stage 0b: Fold Unit-Extent Dims

```
afir-opt --linalg-fold-unit-extent-dims
```

消除 `tensor.expand_shape` 等 unsqueeze 引入的单位维度。

- **产物**: `step0b_folded.mlir`

### Stage 1: Fusion

```
afir-opt --linalg-fuse-elementwise-ops
```

将多个 elementwise / reduction op 融合成单个 `linalg.generic`。

- **产物**: `step1_fused.mlir`

### Stage 2: Tiling (Transform Interpreter)

`_generate_transform_script()` 读取 step1 IR 中 `linalg.generic` 的 `iterator_types`，
找到**第一个 parallel 维度**，自动生成 Transform Dialect 脚本，注入两级 tiling：

| 层级 | 参数 | 语义 | 标注 |
|------|------|------|------|
| 外层 (TB) | `%tb_m` | 核间并行切分 | `ascendc.parallel = true` |
| 内层 (Tb) | `%tb_inner_m` | UB 批次切分 | `ascendc.prologue = "src:GM->VECIN"`, `ascendc.epilogue = "dst:VECOUT->GM"` |

`%tb_m` 和 `%tb_inner_m` 是**参数化的 tiling size**，不是固定值。后续 autotuner 搜索最优值。

切分策略：只切第一个 parallel 维度，其余维度 `tile_size=0`（不切）。

```
afir-opt --transform-interpreter --canonicalize --cse
```

- **产物**: `step2_transform.mlir`（含 transform 脚本）, `step2_tiled.mlir`

### Stage 3: Bufferize

```
afir-opt --one-shot-bufferize=\
    bufferize-function-boundaries=true \
    allow-return-allocs-from-loops=true \
    function-boundary-type-conversion=identity-layout-map
```

从 tensor 语义转为 memref（buffer）语义。

- **产物**: `step3_bufferized.mlir`

### Stage 4: Buffer Placement

```
afir-opt --ascendc-buffer-placement
```

将 buffer 放到 NPU 正确的存储层级（GM / UB 等）。

- **产物**: `step4_buffer_placement.mlir`

### Stage 5: Linalg → AscendC

```
afir-opt --linalg-to-ascendc --canonicalize --cse
```

将 linalg 操作降到 AscendC dialect，映射到 NPU 向量指令。

- **产物**: `step5_ascendc.mlir`

### Stage 6: Parallelize

```
afir-opt --ascendc-parallelize --canonicalize --cse
```

处理 `ascendc.parallel` 标注，生成多核并行代码（核间任务划分）。

- **产物**: `step6_parallelize.mlir`

### Stage 7: Prepare For Emit

```
afir-opt --ascendc-prepare-for-emit --canonicalize --cse
```

最终 IR 清理，为 C++ 代码生成做准备。

- **产物**: `step7_kernel.mlir`

### Stage 7b: Canonicalize CANN Signature

```
afir-opt --canonicalize-cann-signature
```

规范化函数签名，符合 CANN 框架要求（TilingData struct、GM_ADDR 参数等）。

- **产物**: `step7_cann.mlir`

### Stage 7c: 移除 cf.assert (**workaround**)

用正则移除 `cf.assert`（动态 shape 的 broadcast 检查）和仅被 assert 使用的 `arith.cmpi`。

> **原因**: afir-translate 不支持 cf dialect，无法 codegen cf.assert。
> **代价**: 丢失运行时 shape 校验。

### Stage 8: C++ Codegen

```
afir-translate -mlir-to-cann step7_cann.mlir \
    -o step8_kernel.cpp \
    --tiling-space-out step8_kernel.tiling_space.json
```

翻译为 AscendC C++ kernel 源码，同时输出 tiling 参数的搜索空间定义。

- **产物**: `step8_kernel.cpp`, `step8_kernel.tiling_space.json`

---

## Tiling 机制

### tiling_space.json 结构

```json
{
  "block_dim_expr": "ceil(arg0_dim0/TB_M)",
  "kernel": "kernel",
  "soc": "Ascend910B1",
  "tiling_params": [
    {"name": "TB_M",       "fixed": false, "type": "int64", "values": [16, 32, 64]},
    {"name": "TB_N",       "fixed": false, "type": "int64", "values": [16, 32, 64]},
    {"name": "dim_arg0_0", "fixed": true,  "type": "int64", "shape_key": "arg0_dim0"},
    {"name": "dim_arg1_0", "fixed": true,  "type": "int64", "shape_key": "arg1_dim0"},
    {"name": "dim_arg1_1", "fixed": true,  "type": "int64", "shape_key": "arg1_dim1"}
  ]
}
```

| 字段 | 含义 |
|------|------|
| `fixed: false` | 搜索参数——autotuner 遍历 `values` 中的所有值 |
| `fixed: true` | Shape 参数——从 `--shape` 参数中按 `shape_key` 查找，传入 kernel |
| `block_dim_expr` | 运行时公式，根据 shape 和 tiling 参数计算 block 数量 |
| `shape_key` | 映射到 `--shape arg0_dim0=64` 中的 key |

### block_dim 计算

`block_dim_expr` 支持的语法：

- `ceil(X/Y)` → `(X + Y - 1) / Y`（向上取整除法）
- `X/Y` → 整数除法
- 变量名从 `--shape` 和搜索参数的当前值中查找

例：`ceil(arg0_dim0/TB_M)`，当 `arg0_dim0=64, TB_M=16` 时 → `block_dim = 4`。

### 生成的 kernel 中 tiling 如何工作

```cpp
// kernel 入口：每个 block 处理 TB_M 行
uint32_t block_id = GetBlockIdx();              // 0..block_dim-1
uint32_t start = block_id * TB_M;               // 该 block 起始行
uint32_t count = min(TB_M, total_rows - start); // 该 block 处理行数

// 内层循环：每次处理 TB_N 行（UB 批次）
for (uint32_t i = 0; i < count; i += TB_N) {
    uint32_t batch = min(TB_N, count - i);
    // GM → VECIN (prologue)
    DataCopy(local_buf, global_buf + offset, batch * cols);
    // 计算
    Add(out_buf, a_buf, b_buf, batch * cols);
    // VECOUT → GM (epilogue)
    DataCopy(global_buf + out_offset, out_buf, batch);
}
```

---

## 验证阶段：Autotuner

### 工作原理

```
                    ┌─────────────────────────────────────────┐
                    │              Autotuner                   │
                    │                                         │
step8_kernel.cpp ──→│  1. Compile (.cpp → .o → .bin)  [一次]  │
                    │     bisheng → ld.lld                    │
                    │                                         │
tiling_space.json ─→│  2. 枚举所有 tiling 组合               │
                    │     TB_M ∈ [16,32,64] × TB_N ∈ [16,32,64]│
                    │     = 9 种配置                           │
input_*.npy ───────→│                                         │
expected_0.npy ────→│  3. 对每个配置:                          │
                    │     ├ 计算 block_dim                     │
                    │     ├ Pack tiling 参数                   │
                    │     ├ 模拟器执行 kernel                  │
                    │     ├ 对比 actual vs expected            │
                    │     ├ PASS → 记录 cycles + actual       │
                    │     └ FAIL → dump actual, 立即停止      │
                    │                                         │
                    │  4. 选 cycles 最小的 PASS 配置           │──→ tiling_func.cpp
                    │  5. 可选: msprof 性能分析                │──→ perf_out/
                    └─────────────────────────────────────────┘
```

### 编译流程 (Compiler)

| 阶段 | 工具 | 命令 |
|------|------|------|
| .cpp → .o | bisheng | `bisheng -c -x cce -O3 --cce-aicore-arch=dav-c220-vec ...` |
| .o → .bin | ld.lld | `ld.lld -m aicorelinux -Ttext=0 kernel.o -static -o kernel.bin` |

- bisheng 位于 `$ASCEND_HOME_PATH/toolkit/tools/ccec_compiler/bin/bisheng`
- kernel 只编译一次，所有 tiling 配置共享同一个 binary（tiling 作为 kernel 参数传入）

### 精度验证 (SimValidator)

```
容差公式: |actual[i] - expected[i]| ≤ atol + rtol × |expected[i]|
默认值:   atol = 1.0, rtol = 0.01
```

- 支持的数据类型: F16, BF16, F32, INT8, INT32, INT64（统一转 float 比较）
- 从模拟器日志 (`core*_summary_log`) 中提取 cycle count

### Fail-Fast 策略

精度验证失败时**立即停止搜索**，dump actual 输出用于 debug：

- `build_e2e/actual.npy` — best/failed config 的 kernel 输出
- `build_e2e/actual.txt` — 人类可读格式
- `build_e2e/expected.txt` — 参考值

### 输出产物 (tiling_func.cpp)

```cpp
// Auto-generated by autotuner
// best config: TB_M=16 TB_N=16  cycles=6985  max_diff=0.0625

struct TilingData { int64_t TB_M, TB_N, dim_arg0_0, dim_arg1_0, dim_arg1_1; };

void get_tiling(int64_t arg0_dim0, int64_t arg1_dim0, int64_t arg1_dim1,
                TilingData* out) {
    out->TB_M = 16;                    // 搜索得到的最优值
    out->TB_N = 16;
    out->dim_arg0_0 = arg0_dim0;       // 运行时 shape 直传
    out->dim_arg1_0 = arg1_dim0;
    out->dim_arg1_1 = arg1_dim1;
}

int get_block_dim(int64_t arg0_dim0, int64_t arg1_dim0, int64_t arg1_dim1) {
    return static_cast<int>((arg0_dim0 + 16 - 1) / 16);  // ceil(arg0_dim0 / TB_M)
}
```

### 可选: 性能分析 (--perf-report)

1. `HostRunnerGen` 生成独立 runner 可执行文件 (`g++ -O2 runner.cpp -ldl -o runner`)
2. 通过 `msprof op simulator` 调用 runner，生成性能报告
3. 产物在 `perf_out/` 目录下

---

## 全部产物清单

```
output/torch_e2e/test_broadcast_add_reduce/
├── step0_linalg.mlir                # Stage 0:  torch → linalg
├── step0b_folded.mlir               # Stage 0b: fold unit dims
├── step1_fused.mlir                 # Stage 1:  fusion
├── step2_transform.mlir             # Stage 2:  含 transform 脚本
├── step2_tiled.mlir                 # Stage 2:  tiling 后
├── step3_bufferized.mlir            # Stage 3:  bufferize
├── step4_buffer_placement.mlir      # Stage 4:  buffer placement
├── step5_ascendc.mlir               # Stage 5:  AscendC IR
├── step6_parallelize.mlir           # Stage 6:  多核并行
├── step7_kernel.mlir                # Stage 7:  prepare for emit
├── step7_cann.mlir                  # Stage 7b: CANN signature
├── step8_kernel.cpp                 # Stage 8:  C++ kernel
├── step8_kernel.tiling_space.json   # Stage 8:  tiling 搜索空间
├── input_0.npy                      # Stage 9:  输入数据
├── input_1.npy
├── expected_0.npy                   # Stage 9:  PyTorch reference 输出
├── tiling_func.cpp                  # Stage 10: 最优 tiling 函数
├── build_e2e/                       # Autotuner 编译产物
│   ├── kernel.o                     #   编译中间产物
│   ├── kernel.bin                   #   NPU 可执行 binary
│   ├── runner.cpp                   #   host runner 源码 (--perf-report)
│   ├── runner                       #   host runner 可执行文件
│   ├── actual.npy                   #   best config 的 kernel 输出
│   ├── actual.txt                   #   人类可读 actual
│   └── expected.txt                 #   人类可读 expected
└── perf_out/                        # 性能报告 (--perf-report)
```

---

## 当前 Patch / Workaround 一览

| # | 位置 | 问题 | 临时方案 | 未来应做 |
|---|------|------|---------|---------|
| 1 | `_patch_tiling_space()` | afir-translate 生成的 tiling_space.json 中搜索参数没有 `values` | 硬编码填充 `[16, 32, 64]` | afir-translate 应生成合理的 min/max/step 或 values |
| 2 | `_patch_tiling_space()` | `block_dim_expr` 为空，导致 block_dim=1，多核 tiling 无效 | 自动生成 `ceil(shape_key/first_search_param)`，取第一个 fixed 参数的 shape_key 和第一个搜索参数 | afir-translate 应从 IR 的 `ascendc.parallel` 标注推导出正确的 block_dim_expr |
| 3 | Step 7c | afir-translate 不支持 `cf.assert`（cf dialect 未注册） | 正则移除 cf.assert 和相关 cmpi 指令 | 在 afir-translate 中注册 cf dialect，或在 codegen 前加 pass 消除 assert |
| 4 | autotuner `_Exit(0)` | 模拟器后台线程在 exit() 时竞争导致 segfault | 用 `_Exit(0)` 跳过析构函数 | 正确关闭模拟器资源，或用独立进程隔离 |
| 5 | `_generate_transform_script()` | 只处理单个 linalg.generic，只切第一个 parallel 维度 | 生成简单的两级 tiling，非 parallel 维度 tile_size=0 | 支持多 generic（多算子未完全融合）、多维度切分、reduction 维度切分 |
| 6 | `_cleanup_sim_dumps()` | 模拟器在 cwd 留下大量 .dump/.vcd/profile 文件 | 运行后清理 | 让 autotuner/模拟器支持指定 dump 输出目录 |

## 未来泛化方向

### 1. Tiling 策略泛化

当前只对第一个 parallel 维度做两级切分。需要支持：

- **多维度切分**: 对多个 parallel 维度分别 tiling（如 M 和 N 各自切分）
- **Reduction 维度切分**: 沿 reduction 维度切分，需要额外的 partial reduce + merge 逻辑
- **多算子场景**: fusion 未完全合并时，IR 中有多个 linalg.generic，需要分别 tiling

### 2. block_dim_expr 自动推导

当前由 Python 端 patch。应在 MLIR 层面，从 `ascendc.parallel` 标注和 tiling 参数
自动推导出 block_dim_expr 并写入 tiling_space.json。

### 3. 搜索空间自动生成

当前 `[16, 32, 64]` 是硬编码。应根据：

- 硬件约束（UB 大小、对齐要求）
- 算子特征（数据量、计算密度）
- Shape 范围

自动生成合理的搜索范围。

### 4. Fused Kernel 正确性

`--linalg-to-ascendc` 在处理融合了非 reduce 运算（如 relu + reduce）的 generic 时，
可能丢弃前置计算。需要支持 body 中有多个计算 op 的 generic lowering。

### 5. 多输出支持

当前假设单输出（`expected_0.npy`）。需要扩展为支持多输出 kernel 的验证。