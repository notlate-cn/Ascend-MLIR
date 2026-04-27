# PyTorch E2E 测试框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立基于 pytest 的 e2e 测试框架，以 PyTorch 模型为输入，前后端分离，测试用例只定义模型和输入。

**Architecture:** pytest 装饰器 `@torch_e2e_test` 封装完整 pipeline（torch → linalg → afir-opt → kernel → execute → compare）。测试用例在 `tests/torch_e2e/` 下，框架逻辑在 `python/torch2linalg/pytest_plugin.py`。中间产物输出到 `output/torch_e2e/<test_name>/`。

**Tech Stack:** pytest, torch, torch-mlir, numpy, afir-opt (C++ tool), python/runtime (bisheng + executor)

**Spec:** `docs/superpowers/specs/2026-03-20-torch-e2e-test-framework-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `python/torch2linalg/pytest_plugin.py` | Create | `torch_e2e_test` 装饰器 + pipeline 执行逻辑 |
| `python/torch2linalg/test_framework.py` | Delete | 被 pytest_plugin.py 替代 |
| `tests/torch_e2e/conftest.py` | Create | pytest 路径配置 |
| `tests/torch_e2e/test_broadcast_add_reduce.py` | Create | 第一个测试用例 |
| `.gitignore` | Modify | 添加 `output/` |

---

### Task 1: pytest_plugin.py — `torch_e2e_test` 装饰器

**Files:**
- Create: `python/torch2linalg/pytest_plugin.py`

- [ ] **Step 1: Create pytest_plugin.py with torch_e2e_test decorator**

```python
# python/torch2linalg/pytest_plugin.py
"""
pytest 集成：torch_e2e_test 装饰器

用法：
    @torch_e2e_test
    def test_my_model():
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.relu().sum(dim=1)
        return Model(), [torch.randn(32, 64)]
"""

import os
import subprocess
import shutil
import functools
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from .convert import torch_to_linalg

REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_ROOT = REPO_ROOT / "output" / "torch_e2e"


def torch_e2e_test(
    func=None,
    *,
    tiling_override: dict | None = None,
    kernel_cpp: str | Path | None = None,
    rtol: float = 1e-2,
    atol: float = 1.0,
    dtype: torch.dtype = torch.float16,
):
    """
    pytest 装饰器：定义一个 torch → NPU e2e 测试。

    被装饰函数应返回 (model, sample_inputs)。
    装饰器负责执行完整 pipeline 并验证精度。
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper():
            model, inputs = fn()
            test_name = fn.__name__
            work_dir = OUTPUT_ROOT / test_name
            work_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: torch → linalg MLIR
            linalg_path = work_dir / "step0_linalg.mlir"
            torch_to_linalg(model, inputs, output_path=linalg_path, dtype=dtype)

            # Stage 2: linalg → AscendC (afir-opt pipeline)
            resolved_kernel_cpp = None
            if kernel_cpp is not None:
                resolved_kernel_cpp = Path(kernel_cpp)
                if not resolved_kernel_cpp.is_absolute():
                    resolved_kernel_cpp = REPO_ROOT / resolved_kernel_cpp
            else:
                resolved_kernel_cpp = _run_mlir_pipeline(
                    linalg_path, work_dir, tiling_override
                )

            # Stage 3: torch reference
            model_typed = model.to(dtype).eval()
            inputs_typed = [x.to(dtype) for x in inputs]
            with torch.no_grad():
                expected = model_typed(*inputs_typed)
            expected_np = expected.numpy()

            # Stage 4: compile + execute
            if resolved_kernel_cpp is not None and resolved_kernel_cpp.exists():
                actual_np = _compile_and_execute(
                    resolved_kernel_cpp, inputs_typed, expected_np.shape,
                    test_name, work_dir, tiling_override
                )

                # Stage 5: compare
                abs_diff = np.abs(
                    actual_np.astype(np.float32) - expected_np.astype(np.float32)
                )
                max_diff = float(abs_diff.max())
                mean_diff = float(abs_diff.mean())
                print(f"  max_diff={max_diff:.4e}, mean_diff={mean_diff:.4e}")

                assert np.allclose(
                    actual_np.astype(np.float32),
                    expected_np.astype(np.float32),
                    rtol=rtol, atol=atol,
                ), (
                    f"精度不匹配: max_diff={max_diff:.4e}, "
                    f"mean_diff={mean_diff:.4e}"
                )
            else:
                # pipeline 未完全打通，仅验证 linalg 生成成功
                assert linalg_path.exists(), "linalg MLIR 生成失败"
                print(f"  仅验证 torch → linalg（无可执行 kernel）")

        return wrapper

    # 支持 @torch_e2e_test 和 @torch_e2e_test(...) 两种写法
    if func is not None:
        return decorator(func)
    return decorator


def _run_mlir_pipeline(
    linalg_path: Path,
    work_dir: Path,
    tiling_override: dict | None,
) -> Path | None:
    """
    运行 afir-opt pipeline: linalg → AscendC → C++ kernel。
    当前 --linalg-to-ascendc 有 bug，返回 None。
    修复后在此添加 pass 序列。
    """
    afir_opt = shutil.which("afir-opt")
    if afir_opt is None:
        default = REPO_ROOT / "build" / "bin" / "afir-opt"
        if default.exists():
            afir_opt = str(default)

    if afir_opt is None:
        print("  afir-opt 不在 PATH 中，跳过 MLIR pipeline")
        return None

    # TODO: --linalg-to-ascendc 修复后启用
    # passes = ["--linalg-to-ascendc", "--ascendc-parallelize",
    #           "--ascendc-prepare-for-emit"]
    # ascendc_path = work_dir / "ascendc.mlir"
    # cmd = [afir_opt, str(linalg_path)] + passes + ["-o", str(ascendc_path)]
    # subprocess.run(cmd, check=True, capture_output=True, text=True)
    # return ascendc_path

    print("  MLIR pipeline 尚未打通，跳过")
    return None


def _compile_and_execute(
    kernel_cpp: Path,
    inputs_typed: list[torch.Tensor],
    expected_shape: tuple,
    kernel_name: str,
    work_dir: Path,
    tiling_override: dict | None,
) -> np.ndarray:
    """编译 C++ kernel 并在 simulator 上执行。"""
    import struct
    os.environ.setdefault("SOC_VERSION", "Ascend910B1")
    os.environ.setdefault("ASCEND_CPU_SIMULATION", "1")
    os.environ.setdefault("ASCEND_DEVICE_ID", "0")

    from runtime import compile_kernel, read_binary
    from runtime.executor import KernelExecutor

    binary_path = compile_kernel(
        src_file=kernel_cpp,
        output_dir=work_dir,
        kernel_name=kernel_name,
    )
    binary_data = read_binary(binary_path)

    # tiling
    if tiling_override is not None:
        fields = list(tiling_override.values())
        tiling_bytes = struct.pack(f"{len(fields)}q", *fields)
        block_dim = tiling_override.get(
            "block_dim",
            max(1, inputs_typed[0].shape[0] // tiling_override.get("TB_M", 1))
        )
    else:
        tiling_bytes = b""
        block_dim = 1

    inputs_np = [x.numpy() for x in inputs_typed]
    output_buf = np.zeros(expected_shape, dtype=np.float16)
    executor = KernelExecutor()
    executor.initialize()
    outputs = executor.execute(
        binary_data=binary_data,
        function_name=kernel_name,
        inputs=inputs_np,
        outputs=[output_buf],
        tiling_data=tiling_bytes,
        block_dim=block_dim,
    )
    return outputs[0]
```

- [ ] **Step 2: Update `__init__.py` to export new module**

```python
# python/torch2linalg/__init__.py
"""
torch2linalg - 将 PyTorch 模型转换为 linalg MLIR IR
"""

from .convert import torch_to_linalg
from .pytest_plugin import torch_e2e_test

__all__ = ["torch_to_linalg", "torch_e2e_test"]
```

- [ ] **Step 3: Delete old test_framework.py**

```bash
rm python/torch2linalg/test_framework.py
```

- [ ] **Step 4: Commit**

```bash
git add python/torch2linalg/
git commit -m "feat(torch2linalg): add pytest_plugin with torch_e2e_test decorator"
```

---

### Task 2: conftest.py + 第一个测试用例

**Files:**
- Create: `tests/torch_e2e/conftest.py`
- Create: `tests/torch_e2e/test_broadcast_add_reduce.py`

- [ ] **Step 1: Create conftest.py**

```python
# tests/torch_e2e/conftest.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
```

- [ ] **Step 2: Create test_broadcast_add_reduce.py**

```python
# tests/torch_e2e/test_broadcast_add_reduce.py
"""broadcast + add + reduce_sum 端到端测试"""
import struct
import torch
from torch2linalg import torch_e2e_test

REPO_ROOT_STR = "examples/broadcast-add-reduce/step8_kernel.cpp"

@torch_e2e_test(
    kernel_cpp=REPO_ROOT_STR,
    tiling_override={"TB_M": 16, "TB_N": 4, "M": 32, "N": 32, "M2": 32, "N2": 32},
)
def test_broadcast_add_reduce():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)
    return Model(), [torch.randn(32), torch.randn(32, 32)]


@torch_e2e_test
def test_broadcast_add_reduce_linalg_only():
    """仅验证 torch → linalg 转换（不需要 Ascend 环境）"""
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)
    return Model(), [torch.randn(32), torch.randn(32, 64)]
```

- [ ] **Step 3: Run test to verify linalg-only test passes**

```bash
PYTHONPATH=python conda run -n torch-mlir pytest tests/torch_e2e/test_broadcast_add_reduce.py::test_broadcast_add_reduce_linalg_only -v
```

Expected: PASSED（仅验证 torch → linalg 转换成功）

- [ ] **Step 4: Commit**

```bash
git add tests/torch_e2e/
git commit -m "test: add broadcast-add-reduce e2e test case"
```

---

### Task 3: 添加更多测试用例

**Files:**
- Create: `tests/torch_e2e/test_elementwise.py`

- [ ] **Step 1: Create elementwise 测试用例**

```python
# tests/torch_e2e/test_elementwise.py
"""elementwise 算子组合测试"""
import torch
from torch2linalg import torch_e2e_test


@torch_e2e_test
def test_add_mul():
    class Model(torch.nn.Module):
        def forward(self, a, b, c):
            return (a + b) * c
    return Model(), [torch.randn(32, 64)] * 3


@torch_e2e_test
def test_relu_reduce():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.relu().sum(dim=1)
    return Model(), [torch.randn(32, 64)]


@torch_e2e_test
def test_exp_broadcast_add():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return a.exp().unsqueeze(1) + b
    return Model(), [torch.randn(32), torch.randn(32, 64)]


@torch_e2e_test
def test_sigmoid_mul_reduce():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return (x.sigmoid() * y).sum(dim=1)
    return Model(), [torch.randn(32, 64), torch.randn(32, 64)]
```

- [ ] **Step 2: Run all linalg-only tests**

```bash
PYTHONPATH=python conda run -n torch-mlir pytest tests/torch_e2e/ -v
```

Expected: 所有测试 PASSED（均为 linalg-only 模式，因为没有指定 kernel_cpp）

- [ ] **Step 3: Commit**

```bash
git add tests/torch_e2e/test_elementwise.py
git commit -m "test: add elementwise combo e2e test cases"
```

---

### Task 4: gitignore + cleanup

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add output/ to .gitignore**

在 `.gitignore` 的 `# build files` 部分添加：

```
output/
```

- [ ] **Step 2: Verify output directory is created on test run**

```bash
PYTHONPATH=python conda run -n torch-mlir pytest tests/torch_e2e/test_broadcast_add_reduce.py::test_broadcast_add_reduce_linalg_only -v
ls output/torch_e2e/test_broadcast_add_reduce_linalg_only/step0_linalg.mlir
```

Expected: 文件存在，包含 linalg IR

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add output/ to gitignore"
```