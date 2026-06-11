# PyTorch E2E 测试框架设计文档

**日期**：2026-03-20
**状态**：待实现
**前置**：[torch-to-npu-e2e-design](2026-03-20-torch-to-npu-e2e-design.md)

---

## 1. 目标

建立一个以 PyTorch 模型为输入的端到端测试框架，满足：

- **前后端分离**：测试用例只定义模型和输入，不关心编译/执行细节
- **后端全自动**：kernel name 自动生成，tiling 未来由 autotuner 推导
- **留调试口**：可选覆盖 tiling 参数或指定已有 kernel，方便人工调试
- **基于 pytest**：复用 pytest 的用例发现、过滤、报告等基础设施

---

## 2. 架构

```
tests/torch_e2e/                    ← 测试用例（前端，只关心模型+输入）
  test_broadcast_add_reduce.py
  test_relu_reduce.py
  test_add_mul.py
  conftest.py                       ← pytest fixture 注册

python/torch2linalg/                ← torch → linalg 转换
  convert.py                        ← torch_to_linalg()
  pytest_plugin.py                  ← torch_e2e_test 装饰器 + pipeline 执行

python/runtime/                     ← 编译 + 执行（现有）
  compiler.py
  executor.py
```

**数据流**：

```
测试用例                              pytest_plugin (后端)
────────                              ──────────────────
@torch_e2e_test                       1. 调用 func() 拿到 (model, inputs)
def test_xxx():                       2. torch_to_linalg(model, inputs) → .mlir
    return Model(), inputs    ──→     3. afir-opt pipeline → AscendC → C++ kernel
                                      4. bisheng compile → binary
                                      5. simulator execute → actual output
                                      6. model(*inputs) → expected output (torch reference)
                                      7. np.allclose(actual, expected)
```

---

## 3. 测试用例写法

### 基本用法

```python
# tests/torch_e2e/test_broadcast_add_reduce.py
import torch
from torch2linalg.pytest_plugin import torch_e2e_test

@torch_e2e_test
def test_broadcast_add_reduce():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)
    return Model(), [torch.randn(32), torch.randn(32, 64)]

@torch_e2e_test
def test_relu_reduce():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.relu().sum(dim=1)
    return Model(), [torch.randn(32, 64)]
```

### 手动调试

```python
@torch_e2e_test(
    tiling_override={"TB_M": 16, "TB_N": 4},
    kernel_cpp="examples/broadcast-add-reduce/step8_kernel.cpp",
)
def test_broadcast_add_reduce_manual():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)
    return Model(), [torch.randn(32), torch.randn(32, 64)]
```

### 调整精度容忍度

```python
@torch_e2e_test(rtol=1e-3, atol=0.5)
def test_high_precision():
    ...
```

---

## 4. `torch_e2e_test` 装饰器

### 签名

```python
def torch_e2e_test(
    func=None,
    *,
    tiling_override: dict | None = None,
    kernel_cpp: str | Path | None = None,
    rtol: float = 1e-2,
    atol: float = 1.0,
    dtype: torch.dtype = torch.float16,
):
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `tiling_override` | 手动覆盖 tiling 参数，绕过 autotuner | None（自动推导） |
| `kernel_cpp` | 跳过 MLIR pipeline，使用已有 C++ kernel | None（走完整 pipeline） |
| `rtol` | 相对精度容忍度 | 1e-2 |
| `atol` | 绝对精度容忍度 | 1.0 |
| `dtype` | 目标数据类型 | torch.float16 |

### 执行流程

```python
def torch_e2e_test(func, **kwargs):
    def test_wrapper():
        model, inputs = func()

        # 1. torch → linalg
        mlir_text = torch_to_linalg(model, inputs, dtype=kwargs["dtype"])

        # 2. linalg → kernel (afir-opt pipeline 或已有 kernel)
        if kwargs["kernel_cpp"]:
            kernel_cpp = kwargs["kernel_cpp"]
        else:
            kernel_cpp = run_afir_opt_pipeline(mlir_text, kwargs["tiling_override"])

        # 3. compile + execute
        actual = compile_and_execute(kernel_cpp, inputs)

        # 4. torch reference
        expected = run_torch_reference(model, inputs, kwargs["dtype"])

        # 5. compare
        assert np.allclose(actual, expected, rtol=kwargs["rtol"], atol=kwargs["atol"])

    return test_wrapper
```

---

## 5. 后端 pipeline 细节

### 5.1 kernel name 生成

自动从 torch-mlir 输出的 MLIR 函数名派生，或使用测试函数名：

```
test_broadcast_add_reduce → kernel name: broadcast_add_reduce
```

### 5.2 tiling 参数

优先级（高→低）：
1. `tiling_override` — 测试用例手动指定
2. `tiling_space.json` — 如果 example 目录下有，使用 autotuner 搜索
3. 默认启发式 — 根据 tensor shape 和 UB 大小计算（fallback）

### 5.3 工作目录

每个测试用例在 `output/torch_e2e/<test_name>/` 下生成中间产物：

```
output/torch_e2e/test_broadcast_add_reduce/
  step0_linalg.mlir      # torch-mlir 输出
  ascendc.mlir            # afir-opt 输出（pipeline 打通后）
  kernel.cpp              # C++ kernel（自动生成或手动指定）
  kernel.bin              # bisheng 编译产物
```

便于调试时检查每个阶段的输出。

---

## 6. conftest.py

```python
# tests/torch_e2e/conftest.py
import sys
from pathlib import Path

# 添加 python/ 到路径
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
```

---

## 7. 运行方式

```bash
# 前提：torch-mlir 环境 + Ascend 工具链
conda activate torch-mlir
source examples/env.sh

# 跑全部 e2e 测试
PYTHONPATH=python pytest tests/torch_e2e/ -v

# 跑单个测试
PYTHONPATH=python pytest tests/torch_e2e/test_broadcast_add_reduce.py -v

# 过滤
PYTHONPATH=python pytest tests/torch_e2e/ -v -k "broadcast"

# 详细输出（显示每个 stage 的日志）
PYTHONPATH=python pytest tests/torch_e2e/ -v -s
```

---

## 8. 后续演进

| 阶段 | 内容 |
|------|------|
| 当前 | `kernel_cpp` 手动指定，验证前端 + 后端串联 |
| `--linalg-to-ascendc` 修复后 | 去掉 `kernel_cpp`，走完整 MLIR pipeline |
| autotuner 集成后 | 去掉 `tiling_override`，tiling 全自动推导 |
| CI 集成 | 在有 Ascend 环境的 CI 节点上定期运行 |
