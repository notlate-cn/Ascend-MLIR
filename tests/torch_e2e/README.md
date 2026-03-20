# Torch E2E 测试

PyTorch 模型 → linalg MLIR → AscendC C++ kernel 端到端测试。

## 环境准备

conda 环境 `torch-mlir` 关键依赖：

| 包 | 版本 |
|----|------|
| Python | 3.10.20 |
| torch | 2.10.0+cpu |
| torch-mlir | 20260319.756 |
| numpy | 2.2.6 |
| pytest | 9.0.2 |

```bash
# 1. 激活 torch-mlir conda 环境
conda activate torch-mlir

# 2. 设置 afir-opt / ascir-translate 路径
source examples/env.sh
```

## 运行测试

```bash
# 全部测试
conda run -n torch-mlir python -m pytest tests/torch_e2e/ -v -s

# 单个文件
conda run -n torch-mlir python -m pytest tests/torch_e2e/test_elementwise.py -v -s

# 单个用例
conda run -n torch-mlir python -m pytest tests/torch_e2e/test_elementwise.py::test_add -v -s

# 按关键词过滤
conda run -n torch-mlir python -m pytest tests/torch_e2e/ -v -s -k "reduce"
```

## 编写测试用例

```python
import torch
from torch2linalg import torch_e2e_test

@torch_e2e_test
def test_my_op():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    return Model(), [torch.randn(32, 64, dtype=torch.float16),
                     torch.randn(32, 64, dtype=torch.float16)]
```

`@torch_e2e_test` 为零参数装饰器，`dtype` 由测试用例在构造 tensor 时显式指定。
带参数的模块需同时 cast：`model = nn.Linear(64, 32).to(torch.float16)`。

## Pipeline 阶段

```
torch.nn.Module
  → [torch-mlir] step0_linalg.mlir
  → [step1] --linalg-fuse-elementwise-ops
  → [step2] --transform-interpreter
  → [step3] --one-shot-bufferize
  → [step4] --ascendc-buffer-placement
  → [step5] --linalg-to-ascendc
  → [step6] --ascendc-parallelize
  → [step7] --ascendc-prepare-for-emit
  → [step8] ascir-translate → step8_kernel.cpp
```

中间产物保存在 `output/torch_e2e/<test_name>/`，便于调试。

## 判定标准

测试通过条件：成功生成 C++ kernel 文件（`step8_kernel.cpp`）。