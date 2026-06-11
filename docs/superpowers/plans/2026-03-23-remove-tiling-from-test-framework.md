# 移除测试框架中的编译/运行时参数 - 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 简化 `@torch_e2e_test` 装饰器为零参数，移除 tiling/rtol/atol/kernel_cpp/dtype/transform_mlir，dtype 由测试用例显式控制。

**Architecture:** 三处改动：(1) `convert.py` 移除 dtype 参数 (2) `pytest_plugin.py` 简化装饰器和 pipeline (3) 测试用例和文档更新。所有变更在一个 commit 中原子完成。

**Tech Stack:** Python, pytest, torch-mlir

**Spec:** `docs/superpowers/specs/2026-03-23-remove-tiling-from-test-framework-design.md`

---

### Task 1: 修改 `convert.py` — 移除 dtype 参数

**Files:**
- Modify: `python/torch2linalg/convert.py:15-35`

- [ ] **Step 1: 修改 `torch_to_linalg` 签名和实现**

移除 `dtype` 参数，不再做隐式类型转换：

```python
def torch_to_linalg(
    model: nn.Module,
    sample_inputs: list[torch.Tensor],
    output_path: Optional[str | Path] = None,
) -> str:
    """
    将 PyTorch 模型转换为 linalg MLIR IR 文本。

    Args:
        model: PyTorch 模型（调用方负责 dtype）
        sample_inputs: 示例输入张量（调用方负责 dtype）
        output_path: 可选，输出 .mlir 文件路径

    Returns:
        linalg MLIR IR 文本
    """
    model = model.eval()
    inputs = tuple(sample_inputs)
```

变更点：
- 移除 `dtype: torch.dtype = torch.float16` 参数
- `model.to(dtype).eval()` → `model.eval()`
- `tuple(x.to(dtype) for x in sample_inputs)` → `tuple(sample_inputs)`
- 更新 docstring

---

### Task 2: 简化 `pytest_plugin.py` — 装饰器和 pipeline

**Files:**
- Modify: `python/torch2linalg/pytest_plugin.py:30-102` (装饰器)
- Modify: `python/torch2linalg/pytest_plugin.py:240-290` (pipeline)

- [ ] **Step 1: 移除死 import 和更新模块 docstring**

移除 `import torch.nn as nn`（第 21 行）和 `import numpy as np`（第 22 行），Stage 9 移除后不再使用。

更新模块 docstring（第 1-11 行）中的示例，tensor 显式指定 dtype：

```python
"""
pytest 集成：torch_e2e_test 装饰器

用法：
    @torch_e2e_test
    def test_my_model():
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.relu().sum(dim=1)
        return Model(), [torch.randn(32, 64, dtype=torch.float16)]
"""
```

- [ ] **Step 2: 简化装饰器为零参数**

替换整个 `torch_e2e_test` 函数（第 30-102 行）为：

```python
def torch_e2e_test(func):
    """
    pytest 装饰器：定义一个 torch → NPU e2e 测试。

    被装饰函数应返回 (model, sample_inputs)。
    装饰器执行完整 pipeline 并验证 kernel 生成成功。
    """
    @functools.wraps(func)
    def wrapper():
        model, inputs = func()
        test_name = func.__name__
        work_dir = OUTPUT_ROOT / test_name
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: torch → linalg MLIR
        linalg_path = work_dir / "step0_linalg.mlir"
        print(f"\n[Stage 1] torch → linalg MLIR")
        torch_to_linalg(model, inputs, output_path=linalg_path)
        print(f"  输出: {linalg_path}")

        # Stage 2-8: MLIR pipeline → C++ kernel
        print(f"\n[Stage 2-8] MLIR pipeline → C++ kernel")
        resolved_kernel_cpp = _run_mlir_pipeline(linalg_path, work_dir)

        # 验证 C++ kernel 生成
        assert resolved_kernel_cpp is not None and resolved_kernel_cpp.exists(), \
            f"C++ kernel 生成失败: {resolved_kernel_cpp}"
        print(f"\n  C++ kernel 生成成功: {resolved_kernel_cpp}")

    return wrapper
```

关键变更：
- 移除所有可选参数（`tiling`, `rtol`, `atol`, `kernel_cpp`, `dtype`, `transform_mlir`）
- 移除 `func=None` + `decorator` 两层包装
- 移除 `kernel_cpp` 分支
- 移除 Stage 9 torch reference 计算
- `torch_to_linalg` 调用不再传 `dtype`
- `_run_mlir_pipeline` 调用不再传 `transform_mlir`, `tiling`

- [ ] **Step 3: 简化 `_run_mlir_pipeline` 签名和 Stage 2 逻辑**

修改 `_run_mlir_pipeline` 签名（第 240-245 行）：

```python
def _run_mlir_pipeline(
    linalg_path: Path,
    work_dir: Path,
) -> Path | None:
```

替换 Stage 2 条件逻辑（第 269-290 行）为：

```python
    # ── Stage 2: Tiling (Transform Interpreter) ──
    transform_path = _generate_transform_script(step1, work_dir)

    if transform_path is not None:
        step2 = work_dir / "step2_tiled.mlir"
        print(f"  [step2] --transform-interpreter ({transform_path.name})")
        if not _run_afir_opt(transform_path, step2,
                             ["--transform-interpreter", "--canonicalize", "--cse"],
                             afir_opt):
            return None
    else:
        print(f"  [step2] 跳过（无 parallel 维度，无法自动 tiling）")
        step2 = step1
```

变更点：
- 移除 `transform_mlir` 和 `tiling` 参数
- 总是调用 `_generate_transform_script()`
- 更新跳过时的打印信息

---

### Task 3: 更新测试用例 — 移除 tiling，显式指定 dtype

**Files:**
- Modify: `tests/torch_e2e/test_broadcast_add_reduce.py`
- Modify: `tests/torch_e2e/test_elementwise.py`

- [ ] **Step 1: 更新 `test_broadcast_add_reduce.py`**

```python
"""broadcast + add + reduce_sum 端到端测试"""
import torch
from torch2linalg import torch_e2e_test


@torch_e2e_test
def test_broadcast_add_reduce():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)
    return Model(), [torch.randn(32, dtype=torch.float16),
                     torch.randn(32, 64, dtype=torch.float16)]
```

- [ ] **Step 2: 更新 `test_elementwise.py`**

所有 5 个测试函数统一修改，移除 `tiling=...`，tensor 加 `dtype=torch.float16`：

```python
"""elementwise 算子组合测试"""
import torch
from torch2linalg import torch_e2e_test


@torch_e2e_test
def test_add_mul():
    class Model(torch.nn.Module):
        def forward(self, a, b, c):
            return (a + b) * c
    return Model(), [torch.randn(32, 64, dtype=torch.float16)] * 3


@torch_e2e_test
def test_relu_reduce():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.relu().sum(dim=1)
    return Model(), [torch.randn(32, 64, dtype=torch.float16)]


@torch_e2e_test
def test_exp_broadcast_add():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return a.exp().unsqueeze(1) + b
    return Model(), [torch.randn(32, dtype=torch.float16),
                     torch.randn(32, 64, dtype=torch.float16)]


@torch_e2e_test
def test_sigmoid_mul_reduce():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return (x.sigmoid() * y).sum(dim=1)
    return Model(), [torch.randn(32, 64, dtype=torch.float16),
                     torch.randn(32, 64, dtype=torch.float16)]


@torch_e2e_test
def test_add():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    return Model(), [torch.randn(32, 64, dtype=torch.float16),
                     torch.randn(32, 64, dtype=torch.float16)]
```

---

### Task 4: 更新 README

**Files:**
- Modify: `tests/torch_e2e/README.md`

- [ ] **Step 1: 更新示例代码和参数表**

更新「编写测试用例」示例：

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

移除参数表（装饰器无参数），更新 pipeline 说明中 step2 的描述：

```
→ [step2] --transform-interpreter（自动生成符号化 tiling）
```

---

### Task 5: Commit

- [ ] **Step 1: 提交所有变更**

```bash
git add python/torch2linalg/convert.py \
        python/torch2linalg/pytest_plugin.py \
        tests/torch_e2e/test_broadcast_add_reduce.py \
        tests/torch_e2e/test_elementwise.py \
        tests/torch_e2e/README.md
git commit -m "refactor: 移除测试框架中的 tiling 等编译/运行时参数

- 装饰器简化为零参数 @torch_e2e_test
- dtype 由测试用例显式指定
- pipeline 默认自动生成符号化 Transform 脚本
- 移除 tiling/rtol/atol/kernel_cpp/transform_mlir 参数"
```