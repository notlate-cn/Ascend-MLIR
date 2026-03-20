# Torch E2E 测试

PyTorch 模型 → linalg MLIR → AscendC C++ kernel 端到端测试。

## 环境准备

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

@torch_e2e_test(tiling={"TB_M": 16, "TB_N": 4})
def test_my_op():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    return Model(), [torch.randn(32, 64), torch.randn(32, 64)]
```

### `@torch_e2e_test` 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `tiling` | Tiling 参数，如 `{"TB_M": 16, "TB_N": 4}`，自动生成 transform 脚本 | None（跳过 tiling） |
| `transform_mlir` | 手写 transform 脚本路径，优先级高于 `tiling` | None |
| `kernel_cpp` | 跳过 MLIR pipeline，直接使用已有 C++ kernel | None |
| `rtol` / `atol` | 精度容忍度 | 1e-2 / 1.0 |
| `dtype` | 目标数据类型 | torch.float16 |

## Pipeline 阶段

```
torch.nn.Module
  → [torch-mlir] step0_linalg.mlir
  → [step1] --linalg-fuse-elementwise-ops
  → [step2] --transform-interpreter（tiling）
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