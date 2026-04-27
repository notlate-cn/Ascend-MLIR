# PyTorch → NPU Kernel 端到端编译路径设计文档

**日期**：2026-03-20
**状态**：部分实现
**范围**：torch2linalg 前端 + e2e 测试框架 + linalg-to-ascendc pipeline 对接

---

## 1. 背景与目标

### 现状

Ascend-MLIR 已有从 linalg → AscendC → C++ kernel → bisheng → binary 的编译路径（以 `examples/broadcast-add-reduce` 为参考），但存在以下问题：

- 所有 example 的 linalg IR（step0）均为**手写**，无法从真实神经网络模型自动生成
- 缺少从前端框架（PyTorch/JAX/TensorFlow）到 linalg 的接入路径
- `--linalg-to-ascendc` pass 存在 bug（无法处理 `linalg.fill` + reduce pattern），step0 → step5 的自动化 lowering 尚未完全打通
- 没有以 PyTorch 源码为输入的端到端测试框架

### 目标

1. **torch2linalg 前端**：以 `torch.nn.Module` 为入口，通过 torch-mlir 自动生成 linalg MLIR IR
2. **e2e 测试框架**：以 PyTorch 模型为输入，串联 torch → linalg → (pipeline) → kernel → 执行 → 验证
3. **对接现有 pipeline**：确保 torch-mlir 输出的 linalg IR 能被 `--linalg-to-ascendc` 等后续 pass 处理

### 不在范围内

- `--linalg-to-ascendc` pass 的 bug 修复（单独处理）
- JAX / TensorFlow 前端接入（未来通过 StableHLO → linalg 路径接入）
- `torch.compile` backend 集成（成熟期目标）
- Dynamic shape 支持

---

## 2. 整体架构

```
┌───────────────────────────────────────────────────────────────┐
│  E2E 测试框架 (E2ETestRunner)                                  │
│  python3 -m torch2linalg.test_framework                       │
└──────┬────────────────────────────────────┬───────────────────┘
       │                                    │
       ▼                                    ▼
┌──────────────┐                   ┌──────────────────┐
│ torch2linalg │                   │ torch reference   │
│              │                   │                   │
│ torch.export │                   │ model(input)      │
│ → torch-mlir │                   │ → expected output │
│ → linalg.mlir│                   │                   │
└──────┬───────┘                   └────────┬──────────┘
       │                                    │
       ▼                                    │
┌──────────────────────────────┐            │
│ 现有 MLIR Pipeline (afir-opt)│            │
│                              │            │
│ linalg.mlir                  │            │
│  → --linalg-to-ascendc       │            │
│  → --ascendc-parallelize     │            │
│  → --ascendc-prepare-for-emit│            │
│  → C++ kernel emit           │            │
│  → bisheng compile           │            │
│  → simulator execute         │            │
│  → actual output             │            │
└──────────┬───────────────────┘            │
           │                                │
           ▼                                ▼
      ┌─────────────────────────────────────────┐
      │  numpy compare (allclose)               │
      │  actual vs expected                     │
      └─────────────────────────────────────────┘
```

**松耦合设计**：前端（torch2linalg）和后端（afir-opt pipeline）通过 `.mlir` 文件解耦。后端可以独立开发和测试，不依赖 PyTorch 环境。

---

## 3. 技术验证结果

### 3.1 torch-mlir 安装

- **环境**：conda env `torch-mlir`，Python 3.10
- **安装源**：`pip install --pre torch-mlir -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels`
- **版本**：torch-mlir 20260319.756 + torch 2.10.0+cpu

### 3.2 torch-mlir 输出 IR 分析

以 `BroadcastAddReduce` 模型为例：

```python
class BroadcastAddReduce(torch.nn.Module):
    def forward(self, a, b):  # a: [M], b: [M, N]
        return (a.unsqueeze(1) + b).sum(dim=1)
```

torch-mlir 输出的 linalg IR：

```mlir
#map = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%arg0: tensor<32xf16>, %arg1: tensor<32x64xf16>) -> tensor<32xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [32, 1]
        : tensor<32xf16> into tensor<32x1xf16>
    %0 = tensor.empty() : tensor<32x64xf16>
    // broadcast + add 合并为一个 linalg.generic
    %1 = linalg.generic {
        indexing_maps = [#map, #map1, #map1],
        iterator_types = ["parallel", "parallel"]
    } ins(%expanded, %arg1 : tensor<32x1xf16>, tensor<32x64xf16>)
      outs(%0 : tensor<32x64xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %5 = arith.addf %in, %in_0 : f16
      linalg.yield %5 : f16
    } -> tensor<32x64xf16>
    // reduce sum
    %2 = tensor.empty() : tensor<32xf16>
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<32xf16>) -> tensor<32xf16>
    %4 = linalg.generic {
        indexing_maps = [#map1, #map2],
        iterator_types = ["parallel", "reduction"]
    } ins(%1 : tensor<32x64xf16>) outs(%3 : tensor<32xf16>) {
    ^bb0(%in: f16, %out: f16):
      %5 = arith.addf %in, %out : f16
      linalg.yield %5 : f16
    } -> tensor<32xf16>
    return %4 : tensor<32xf16>
  }
}
```

### 3.3 与手写 step0 的关键差异

| 维度 | 手写 step0 | torch-mlir 输出 |
|------|-----------|----------------|
| broadcast | 独立 `linalg.generic` + `affine_map<(d0,d1)->(d0)>` | `tensor.expand_shape` + `affine_map<(d0,d1)->(d0,0)>` 融合在 add 中 |
| add + broadcast | 分两步 | 一步完成（broadcast 隐含在 indexing_map 中） |
| reduce 初始化 | 无 `linalg.fill` | 有 `linalg.fill` 初始化为 0 |
| shape | dynamic (`tensor<?xf16>`) | static (`tensor<32xf16>`) |
| 函数名 | 自定义 | `main` |

### 3.4 torch-mlir 不做算子融合

验证了 `add + mul` 模型：torch-mlir 为每个 `aten.*` 算子生成独立的 `linalg.generic`，**不做算子间 fusion**。Fusion 由下游编译器（我们的 pipeline）负责。

### 3.5 现有 pipeline 兼容性

- `afir-opt` 能正常解析 torch-mlir 输出的 IR
- `--linalg-to-ascendc` 在处理 `linalg.fill` 时崩溃（手写 step0 也触发同样的 bug）
- 后续 pass（`--ascendc-parallelize` 等）从 step5 开始工作正常

---

## 4. 组件设计

### 4.1 torch2linalg 模块

**位置**：`python/torch2linalg/`

```
python/torch2linalg/
  __init__.py          # 导出 torch_to_linalg
  convert.py           # 核心转换：torch.nn.Module → linalg MLIR
  test_framework.py    # E2ETestRunner 测试框架
```

**核心 API**：

```python
from torch2linalg import torch_to_linalg

mlir_text = torch_to_linalg(
    model,                    # torch.nn.Module
    sample_inputs,            # list[torch.Tensor]，用于 tracing shape
    output_path="out.mlir",   # 可选，保存到文件
    dtype=torch.float16,      # 目标 dtype
)
```

**实现**：
1. `model.to(dtype).eval()` + 输入转 dtype
2. `torch_mlir.fx.export_and_import(model, *inputs, output_type=LINALG_ON_TENSORS)`
3. `module.operation.get_asm()` 获取 MLIR 文本

**依赖**：torch-mlir（pip 包，不编译源码），仅在 `torch-mlir` conda 环境中可用。

### 4.2 E2E 测试框架

**位置**：`python/torch2linalg/test_framework.py`

**核心类**：

```python
from torch2linalg.test_framework import E2ETestRunner, PipelineConfig

runner = E2ETestRunner(build_dir="./build_e2e")
result = runner.run(model, sample_inputs, config=PipelineConfig(
    # MLIR pipeline 完全打通后使用：
    # afir_opt_passes=["--linalg-to-ascendc", "--ascendc-parallelize", ...],
    # 当前使用已有手写 kernel：
    kernel_cpp=Path("step8_kernel.cpp"),
    kernel_name="broadcast_add_reducesum",
    tiling_func=lambda: struct.pack("6q", TB_M, TB_N, M, N, M, N),
    block_dim_func=lambda: (M + TB_M - 1) // TB_M,
))
```

**测试流程（6 个 Stage）**：

| Stage | 动作 | 当前状态 |
|-------|------|---------|
| 1 | torch.nn.Module → linalg MLIR | 已实现 |
| 2 | linalg → AscendC (afir-opt) | 待 `--linalg-to-ascendc` 修复 |
| 3 | AscendC → C++ kernel | 待实现，可用手写 kernel 替代 |
| 4 | PyTorch reference 计算 | 已实现 |
| 5 | bisheng 编译 + simulator 执行 | 已实现（需 Ascend 环境） |
| 6 | 精度对比 (numpy allclose) | 已实现 |

**PipelineConfig** 提供了灵活的配置：
- `afir_opt_passes`：afir-opt pass 序列，pipeline 打通后填入
- `kernel_cpp`：提供手写 kernel 可跳过 MLIR pipeline
- `tiling_func` / `block_dim_func`：自定义 tiling 参数
- `rtol` / `atol`：精度容忍度

---

## 5. 后续工作

### 5.1 近期（打通 pipeline）

1. **修复 `--linalg-to-ascendc` 的 `linalg.fill` bug**
   - `linalg.fill` + reduce 的 pattern 未正确处理
   - 手写 step0 和 torch-mlir 输出均触发同一 bug
2. **处理 torch-mlir 的 broadcast 表达差异**
   - torch-mlir 用 `tensor.expand_shape` + `indexing_map (d0,0)` 表达 broadcast
   - `--linalg-to-ascendc` 需要能识别这种 pattern

### 5.2 中期（扩展算子覆盖）

3. **支持通用 elementwise 链**
   - 任意 elementwise 组合（add/mul/exp/relu/...）+ 可选 broadcast + 可选 reduce
   - 在 linalg 层通过 `--linalg-fuse-elementwise-ops` 做 fusion
4. **Transform dialect 自动 tiling**
   - Phase 1：手动 Transform dialect 脚本
   - Phase 2：自动 tiling + autotuner 搜索参数

### 5.3 远期（多前端 + 产品化）

5. **JAX 接入**：`jax.export` → StableHLO → `stablehlo-to-linalg`（已有 stablehlo 依赖）
6. **TensorFlow 接入**：tf-mlir → mhlo → linalg
7. **torch.compile backend**：注册 `@torch.compile(backend="ascend")`

---

## 6. 依赖关系

```
torch-mlir (pip, conda env: torch-mlir)
  ├── torch 2.10.0+cpu
  └── torch-mlir 20260319.756

Ascend-MLIR (本项目)
  ├── afir-opt (含 --linalg-to-ascendc 等 passes)
  ├── python/runtime (bisheng 编译 + executor)
  └── python/torch2linalg (新增)

Ascend 工具链 (bisheng + libruntime_camodel)
  └── 通过 ASCEND_HOME_PATH 环境变量配置
```

---

## 7. 运行方式

```bash
# 1. 安装 torch-mlir 环境（一次性）
conda create -n torch-mlir python=3.10 -y
conda activate torch-mlir
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --pre torch-mlir \
    -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels

# 2. 设置 PATH
source examples/env.sh

# 3. 运行 e2e 测试
PYTHONPATH=python conda run -n torch-mlir \
    python3 python/torch2linalg/test_framework.py

# 4. 仅做 torch → linalg 转换
PYTHONPATH=python conda run -n torch-mlir python3 -c "
import torch
from torch2linalg import torch_to_linalg

"
class MyModel(torch.nn.Module):
    def forward(self, a, b):
        return (a.unsqueeze(1) + b).sum(dim=1)

torch_to_linalg(MyModel(), [torch.randn(32), torch.randn(32, 64)],
                output_path='output.mlir')
```