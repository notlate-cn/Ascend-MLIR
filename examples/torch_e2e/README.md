# Torch E2E 测试

PyTorch 模型 → linalg MLIR → AscendC C++ kernel → compile → 数值验证 端到端测试。

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

# 2. 设置 afir-opt / afir-translate / autotuner 路径
source examples/env.sh
```

## 运行测试

```bash
# 全部测试
conda run -n torch-mlir python -m pytest examples/torch_e2e/ -v -s

# 单个文件
conda run -n torch-mlir python -m pytest examples/torch_e2e/test_elementwise.py -v -s

# 单个用例
conda run -n torch-mlir python -m pytest examples/torch_e2e/test_elementwise.py::test_add -v -s

# 按关键词过滤
conda run -n torch-mlir python -m pytest examples/torch_e2e/ -v -s -k "reduce"
```

## 编写测试用例

```python
import torch
from torch2linalg import torch_e2e_test, TensorSpec

@torch_e2e_test
def test_my_op():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    return Model(), [TensorSpec((None, None), torch.float16),
                     TensorSpec((None, None), torch.float16)]
```

`TensorSpec(shape, dtype)` 描述张量的形状模式：
- `None` 维度 → 动态（IR 里是 `?`，运行时传入具体值）
- 具体数字 → 静态（IR 里是固定值）
- `dtype` 默认 `torch.float16`
- 动态维度默认用 64 实例化测试数据

## Pipeline 阶段

```
torch.nn.Module + TensorSpec
  → [torch-mlir] step0_linalg.mlir（动态 shape）
  → [step1] --linalg-fuse-elementwise-ops
  → [step2] --ascend-normalize
  → [step3] --ascend-kernelize
  → [step4] --ascend-schedule
  → [step5] --ascend-realize
  → [step6] --ascend-compute-lower
  → [step7] --ascend-parallelize + --ascend-prepare-for-emit
  → [step7b] --ascend-canonicalize-cann-signature
  → [step8] afir-translate -mlir-to-cann → step8_kernel.cpp + tiling_space.json
  → [step9-10] autotuner → compile + tiling 搜索 + 数值验证
```

中间产物保存在 `output/torch_e2e/<test_name>/`，便于调试。

## 判定标准

测试通过条件：autotuner 成功完成 compile + tiling 搜索 + 数值正确性验证。
