# Torch E2E Pipeline Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the pytest torch e2e test framework with the latest run.sh pipeline (stages 0-10), adding dynamic shape support via TensorSpec and end-to-end numerical verification via autotuner.

**Architecture:** Refactor `python/torch2linalg/pytest_plugin.py` in-place (method A). Add `TensorSpec` dataclass. Split pipeline into `_run_mlir_pipeline` (stage 1-8) + `_run_autotuner` (stage 9-10). Update all test cases to use TensorSpec.

**Tech Stack:** Python 3.10, PyTorch, torch-mlir, pytest, afir-opt, afir-translate, autotuner (C++ CLI tools)

**Spec:** `docs/superpowers/specs/2026-03-25-torch-e2e-pipeline-alignment-design.md`

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `python/torch2linalg/tensor_spec.py` | Create | TensorSpec dataclass |
| `python/torch2linalg/convert.py` | Modify (line 15-43) | Add `dynamic_shapes` param |
| `python/torch2linalg/__init__.py` | Modify (line 13-16) | Export TensorSpec |
| `python/torch2linalg/pytest_plugin.py` | Modify (full rewrite of decorator + pipeline) | Decorator, pipeline, autotuner |
| `tests/torch_e2e/test_elementwise.py` | Modify (all test functions) | Migrate to TensorSpec |
| `tests/torch_e2e/test_broadcast_add_reduce.py` | Modify (all test functions) | Migrate to TensorSpec |
| `tests/torch_e2e/README.md` | Modify | Update docs |

---

### Task 1: Create TensorSpec

**Files:**
- Create: `python/torch2linalg/tensor_spec.py`
- Modify: `python/torch2linalg/__init__.py`

- [x] **Step 1: Create `tensor_spec.py`**

```python
"""TensorSpec: 张量形状描述符，支持动态/静态维度声明。"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.export import Dim


@dataclass
class TensorSpec:
    """描述张量的形状模式和数据类型。

    shape 中的 None 表示动态维度，具体数字表示静态维度。
    例：(None, None) 全动态，(32, None) 第一维静态。
    """
    shape: tuple
    dtype: torch.dtype = torch.float16

    def make_sample(self) -> torch.Tensor:
        """用默认值(64)填充 None 维度，生成具体 tensor。"""
        concrete = tuple(64 if d is None else d for d in self.shape)
        return torch.randn(concrete, dtype=self.dtype)

    def dynamic_dims(self) -> dict[int, Any]:
        """返回 torch.export 需要的 dynamic_shapes 映射。

        返回 {dim_index: Dim("d{dim_index}")} 仅包含 None 维度。
        """
        return {i: Dim(f"d{i}") for i, d in enumerate(self.shape) if d is None}
```

- [x] **Step 2: Update `__init__.py` to export TensorSpec**

Modify `python/torch2linalg/__init__.py`:

```python
"""
torch2linalg - 将 PyTorch 模型转换为 linalg MLIR IR

依赖：
  pip install --pre torch-mlir -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
  pip install torch --index-url https://download.pytorch.org/whl/cpu

用法：
  from torch2linalg import torch_to_linalg
  mlir_text = torch_to_linalg(model, sample_inputs)
"""

from .convert import torch_to_linalg
from .pytest_plugin import torch_e2e_test
from .tensor_spec import TensorSpec

__all__ = ["torch_to_linalg", "torch_e2e_test", "TensorSpec"]
```

- [ ] **Step 3: Commit** (deferred to end)

---

### Task 2: Add `dynamic_shapes` to `torch_to_linalg`

**Files:**
- Modify: `python/torch2linalg/convert.py:15-35`

- [x] **Step 1: Add `dynamic_shapes` parameter**

Change the function signature and the `export_and_import` call in `python/torch2linalg/convert.py`:

```python
def torch_to_linalg(
    model: nn.Module,
    sample_inputs: list[torch.Tensor],
    dynamic_shapes: dict | None = None,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    将 PyTorch 模型转换为 linalg MLIR IR 文本。

    Args:
        model: PyTorch 模型（调用方负责 dtype）
        sample_inputs: 示例输入张量（调用方负责 dtype）
        dynamic_shapes: 可选，torch.export 的动态维度映射
        output_path: 可选，输出 .mlir 文件路径

    Returns:
        linalg MLIR IR 文本
    """
    model = model.eval()
    inputs = tuple(sample_inputs)

    # torch-mlir: export + import → linalg
    kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS)
    if dynamic_shapes is not None:
        kwargs["dynamic_shapes"] = dynamic_shapes
    module = export_and_import(model, *inputs, **kwargs)
    mlir_text = module.operation.get_asm()

    # 可选：写入文件
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(mlir_text)

    return mlir_text
```

Note: We conditionally pass `dynamic_shapes` so that existing callers without dynamic shapes are unaffected.

- [ ] **Step 2: Commit** (deferred to end)

---

### Task 3: Rewrite `pytest_plugin.py` — decorator + pipeline + autotuner

This is the main refactoring task. The file `python/torch2linalg/pytest_plugin.py` is rewritten to:
1. Decorator accepts `TensorSpec` instead of raw tensors
2. Pipeline split into `_run_mlir_pipeline(work_dir)` + `_run_autotuner(work_dir, num_inputs, shape_str)`
3. Stage 7b added, stage 8 updated (afir-translate -mlir-to-cann), old ascir-translate + transform strip hack removed
4. Autotuner invocation added (stage 9-10)

**Files:**
- Modify: `python/torch2linalg/pytest_plugin.py` (full rewrite)

- [x] **Step 1: Rewrite `pytest_plugin.py`**

Replace the entire content of `python/torch2linalg/pytest_plugin.py` with:

```python
"""
pytest 集成：torch_e2e_test 装饰器

用法：
    @torch_e2e_test
    def test_my_model():
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.relu().sum(dim=1)
        return Model(), [TensorSpec((None, None), torch.float16)]
"""

import re
import subprocess
import shutil
import functools
from pathlib import Path

import numpy as np
import torch

from .convert import torch_to_linalg
from .tensor_spec import TensorSpec

REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_ROOT = REPO_ROOT / "output" / "torch_e2e"


# ================================================================
# Public API: torch_e2e_test decorator
# ================================================================

def torch_e2e_test(func):
    """
    pytest 装饰器：定义一个 torch → NPU e2e 测试。

    被装饰函数应返回 (model, specs: list[TensorSpec])。
    装饰器执行完整 pipeline 并验证数值正确性。
    """
    @functools.wraps(func)
    def wrapper():
        model, specs = func()
        test_name = func.__name__
        work_dir = OUTPUT_ROOT / test_name
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. 从 TensorSpec 生成具体 tensor
        inputs = [spec.make_sample() for spec in specs]

        # 2. PyTorch reference run → 存 npy
        print(f"\n[Stage 0] 生成 reference data")
        model_eval = model.eval()
        for i, tensor in enumerate(inputs):
            np.save(work_dir / f"input_{i}.npy", tensor.numpy())
        with torch.no_grad():
            expected = model_eval(*inputs)
        np.save(work_dir / "expected_0.npy", expected.numpy())
        print(f"  inputs: {[t.shape for t in inputs]}")
        print(f"  expected: {expected.shape}")

        # 3. 构建 dynamic_shapes
        dynamic_shapes = {}
        for i, spec in enumerate(specs):
            dims = spec.dynamic_dims()
            if dims:
                dynamic_shapes[i] = dims
        dynamic_shapes = dynamic_shapes or None

        # 4. torch → linalg（动态 shape）
        print(f"\n[Stage 0] torch → linalg MLIR")
        torch_to_linalg(model, inputs, dynamic_shapes,
                         output_path=work_dir / "step0_linalg.mlir")
        print(f"  输出: {work_dir / 'step0_linalg.mlir'}")

        # 5. MLIR pipeline (stage 1-8)
        print(f"\n[Stage 1-8] MLIR pipeline")
        assert _run_mlir_pipeline(work_dir), "MLIR pipeline 失败"

        # 6. Autotuner (stage 9-10)
        print(f"\n[Stage 9-10] Autotuner (compile + tune + verify)")
        shape_str = _build_shape_str(specs, inputs)
        num_inputs = len(inputs)
        assert _run_autotuner(work_dir, num_inputs, shape_str), \
            "Autotuner 验证失败"

        print(f"\n  ✓ 测试通过: {test_name}")

    return wrapper


# ================================================================
# Helpers
# ================================================================

def _build_shape_str(specs: list[TensorSpec], samples: list[torch.Tensor]) -> str:
    """从 TensorSpec + 具体 sample 构建 autotuner --shape 参数。

    格式: "arg0_dim0=64,arg0_dim1=64,arg1_dim0=64,arg1_dim1=64"
    """
    parts = []
    for arg_idx, (spec, sample) in enumerate(zip(specs, samples)):
        for dim_idx, size in enumerate(sample.shape):
            parts.append(f"arg{arg_idx}_dim{dim_idx}={size}")
    return ",".join(parts)


def _find_tool(name: str) -> str | None:
    """查找工具路径，优先 PATH，其次 build/bin/"""
    path = shutil.which(name)
    if path:
        return path
    default = REPO_ROOT / "build" / "bin" / name
    if default.exists():
        return str(default)
    return None


def _run_afir_opt(input_path: Path, output_path: Path, passes: list[str],
                  afir_opt: str) -> bool:
    """运行 afir-opt，成功返回 True"""
    cmd = [afir_opt] + passes + [str(input_path), "-o", str(output_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  失败: {' '.join(passes)}")
        print(f"  stderr: {proc.stderr[:500]}")
        return False
    return True


# ================================================================
# MLIR Pipeline: stage 1-8
# ================================================================

def _generate_transform_script(fused_path: Path, work_dir: Path) -> Path | None:
    """
    读取 step1 fused IR，注入通用 transform 序列，生成 step2_transform.mlir。

    适用场景：单个 linalg.generic（fusion 后最常见的情况）。
    切分策略：沿第一个 parallel 维度做两级 tile（TB 核间 + Tb UB批次），
    其余维度不切（tile_size=0）。
    """
    content = fused_path.read_text()

    match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', content)
    if not match:
        print(f"  [transform-gen] 未找到 linalg.generic iterator_types，跳过")
        return None

    iters = [s.strip().strip('"') for s in match.group(1).split(",")]
    ndims = len(iters)

    par_idx = next((i for i, t in enumerate(iters) if t == "parallel"), None)
    if par_idx is None:
        print(f"  [transform-gen] 无 parallel 维度，跳过 tiling")
        return None

    def _tile_sizes(param_name: str) -> str:
        sizes = ["0"] * ndims
        sizes[par_idx] = f"%{param_name}"
        return "[" + ", ".join(sizes) + "]"

    tb_tile = _tile_sizes("tb_m")
    tb_inner_tile = _tile_sizes("tb_inner_m")

    new_content = content.replace(
        "module {",
        "module attributes {transform.with_named_sequence} {",
        1,
    )

    transform_seq = f"""
  // ═══════════ 自动生成的 Transform 调度脚本 ═══════════
  transform.named_sequence @__transform_main(
      %root : !transform.any_op {{transform.readonly}}
  ) {{
    %func = transform.structured.match ops{{["func.func"]}} in %root
        : (!transform.any_op) -> !transform.any_op
    %func_new, %tb_m, %tb_inner_m =
        transform.func.add_index_args %func, 2
            : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)

    %generic = transform.structured.match ops{{["linalg.generic"]}} in %func_new
        : (!transform.any_op) -> !transform.any_op

    %tiled_tb, %loop_tb =
        transform.structured.tile_using_for %generic
            tile_sizes {tb_tile}
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %true_param = transform.param.constant true -> !transform.any_param
    transform.annotate %loop_tb "ascendc.parallel"
        = %true_param : !transform.any_op, !transform.any_param

    %tiled_inner, %loop_inner =
        transform.structured.tile_using_for %tiled_tb
            tile_sizes {tb_inner_tile}
                : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %prologue = transform.param.constant "src:GM->VECIN" -> !transform.any_param
    %epilogue = transform.param.constant "dst:VECOUT->GM" -> !transform.any_param
    transform.annotate %loop_inner "ascendc.prologue"
        = %prologue : !transform.any_op, !transform.any_param
    transform.annotate %loop_inner "ascendc.epilogue"
        = %epilogue : !transform.any_op, !transform.any_param

    %vector_unit = transform.param.constant "AiCore.Vector" -> !transform.any_param
    transform.annotate %tiled_inner "ascendc.unit"
        = %vector_unit : !transform.any_op, !transform.any_param

    transform.yield
  }}
"""

    last_brace = new_content.rfind("}")
    new_content = new_content[:last_brace] + transform_seq + "}\n"

    out_path = work_dir / "step2_transform.mlir"
    out_path.write_text(new_content)
    print(f"  [transform-gen] iterator_types={iters}, tile parallel dim d{par_idx}")
    return out_path


def _run_mlir_pipeline(work_dir: Path) -> bool:
    """
    运行 MLIR pipeline（stage 1-8）:
      step1: --linalg-fuse-elementwise-ops
      step2: --transform-interpreter
      step3: --one-shot-bufferize
      step4: --ascendc-buffer-placement
      step5: --linalg-to-ascendc
      step6: --ascendc-parallelize
      step7: --ascendc-prepare-for-emit
      step7b: --canonicalize-cann-signature
      step8: afir-translate -mlir-to-cann
    """
    # 入口校验
    linalg_path = work_dir / "step0_linalg.mlir"
    assert linalg_path.exists(), f"缺少 {linalg_path}"

    afir_opt = _find_tool("afir-opt")
    if afir_opt is None:
        print("  afir-opt 不在 PATH 中，跳过")
        return False

    # ── Stage 1: Fusion ──
    step1 = work_dir / "step1_fused.mlir"
    print(f"  [step1] --linalg-fuse-elementwise-ops")
    if not _run_afir_opt(linalg_path, step1,
                         ["--linalg-fuse-elementwise-ops"], afir_opt):
        return False

    # ── Stage 2: Tiling (Transform Interpreter) ──
    transform_path = _generate_transform_script(step1, work_dir)

    if transform_path is not None:
        step2 = work_dir / "step2_tiled.mlir"
        print(f"  [step2] --transform-interpreter ({transform_path.name})")
        if not _run_afir_opt(transform_path, step2,
                             ["--transform-interpreter", "--canonicalize", "--cse"],
                             afir_opt):
            return False
    else:
        print(f"  [step2] 跳过（无 parallel 维度，无法自动 tiling）")
        step2 = step1

    # ── Stage 3: Bufferize ──
    step3 = work_dir / "step3_bufferized.mlir"
    bufferize_opts = ("--one-shot-bufferize="
                      "bufferize-function-boundaries=true "
                      "allow-return-allocs-from-loops=true "
                      "function-boundary-type-conversion=identity-layout-map")
    print(f"  [step3] --one-shot-bufferize")
    if not _run_afir_opt(step2, step3, [bufferize_opts, "--cse"], afir_opt):
        return False

    # ── Stage 4: Buffer Placement ──
    step4 = work_dir / "step4_buffer_placement.mlir"
    print(f"  [step4] --ascendc-buffer-placement")
    if not _run_afir_opt(step3, step4, ["--ascendc-buffer-placement"], afir_opt):
        return False

    # ── Stage 5: Linalg → AscendC ──
    step5 = work_dir / "step5_ascendc.mlir"
    print(f"  [step5] --linalg-to-ascendc")
    if not _run_afir_opt(step4, step5,
                         ["--linalg-to-ascendc", "--canonicalize", "--cse"],
                         afir_opt):
        return False

    # ── Stage 6: Parallelize ──
    step6 = work_dir / "step6_parallelize.mlir"
    print(f"  [step6] --ascendc-parallelize")
    if not _run_afir_opt(step5, step6,
                         ["--ascendc-parallelize", "--canonicalize", "--cse"],
                         afir_opt):
        return False

    # ── Stage 7: Prepare For Emit ──
    step7 = work_dir / "step7_kernel.mlir"
    print(f"  [step7] --ascendc-prepare-for-emit")
    if not _run_afir_opt(step6, step7,
                         ["--ascendc-prepare-for-emit", "--canonicalize", "--cse"],
                         afir_opt):
        return False

    # ── Stage 7b: Canonicalize CANN Signature ──
    step7b = work_dir / "step7_cann.mlir"
    print(f"  [step7b] --canonicalize-cann-signature")
    if not _run_afir_opt(step7, step7b,
                         ["--canonicalize-cann-signature"], afir_opt):
        return False

    # ── Stage 8: C++ Codegen (CANN standard) ──
    afir_translate = _find_tool("afir-translate")
    if afir_translate is None:
        print(f"  [step8] afir-translate 不在 PATH 中，跳过 codegen")
        return False

    step8_cpp = work_dir / "step8_kernel.cpp"
    step8_tiling = work_dir / "step8_kernel.tiling_space.json"
    print(f"  [step8] afir-translate -mlir-to-cann")
    cmd = [afir_translate, "-mlir-to-cann", str(step7b),
           "-o", str(step8_cpp),
           "--tiling-space-out", str(step8_tiling)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  失败: afir-translate -mlir-to-cann")
        print(f"  stderr: {proc.stderr[:500]}")
        return False

    print(f"  C++ kernel: {step8_cpp}")
    print(f"  tiling space: {step8_tiling}")
    return True


# ================================================================
# Autotuner: stage 9-10 (compile + tune + verify)
# ================================================================

def _run_autotuner(work_dir: Path, num_inputs: int, shape_str: str) -> bool:
    """
    调用 autotuner 完成 compile + tiling 搜索 + 数值验证。

    autotuner 内部自动编译 kernel，搜索最优 tiling 参数，并验证数值正确性。
    """
    # 入口校验
    for i in range(num_inputs):
        npy = work_dir / f"input_{i}.npy"
        assert npy.exists(), f"缺少 {npy}"
    expected_npy = work_dir / "expected_0.npy"
    assert expected_npy.exists(), f"缺少 {expected_npy}"
    assert (work_dir / "step8_kernel.cpp").exists(), "缺少 step8_kernel.cpp"
    assert (work_dir / "step8_kernel.tiling_space.json").exists(), \
        "缺少 step8_kernel.tiling_space.json"
    assert shape_str, "shape_str 不能为空"

    autotuner = _find_tool("autotuner")
    if autotuner is None:
        print("  autotuner 不在 PATH 中，跳过")
        return False

    input_npys = ",".join(
        str(work_dir / f"input_{i}.npy") for i in range(num_inputs)
    )
    cmd = [
        autotuner,
        "--space", str(work_dir / "step8_kernel.tiling_space.json"),
        "--kernel", str(work_dir / "step8_kernel.cpp"),
        "--inputs", input_npys,
        "--expected", str(expected_npy),
        "--shape", shape_str,
        "--output", str(work_dir / "tiling_func.cpp"),
    ]
    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  失败: autotuner")
        print(f"  stdout: {proc.stdout[-500:]}")
        print(f"  stderr: {proc.stderr[:500]}")
        return False

    print(f"  autotuner stdout:\n{proc.stdout[-300:]}")
    return True
```

- [x] **Step 2: Verify the file has no syntax errors**

Run: `cd /home/gser/code/Ascend-MLIR && python -c "import ast; ast.parse(open('python/torch2linalg/pytest_plugin.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit** (deferred to end)

---

### Task 4: Migrate test cases to TensorSpec

**Files:**
- Modify: `tests/torch_e2e/test_elementwise.py`
- Modify: `tests/torch_e2e/test_broadcast_add_reduce.py`

- [x] **Step 1: Rewrite `test_elementwise.py`**

Replace the entire content of `tests/torch_e2e/test_elementwise.py` with:

```python
"""elementwise 算子组合测试"""
import torch
from torch2linalg import torch_e2e_test, TensorSpec


@torch_e2e_test
def test_add_mul():
    class Model(torch.nn.Module):
        def forward(self, a, b, c):
            return (a + b) * c
    return Model(), [TensorSpec((None, None))] * 3


@torch_e2e_test
def test_relu_reduce():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.relu().sum(dim=1)
    return Model(), [TensorSpec((None, None))]


@torch_e2e_test
def test_exp_broadcast_add():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return a.exp().unsqueeze(1) + b
    return Model(), [TensorSpec((None,)),
                     TensorSpec((None, None))]


@torch_e2e_test
def test_sigmoid_mul_reduce():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return (x.sigmoid() * y).sum(dim=1)
    return Model(), [TensorSpec((None, None)),
                     TensorSpec((None, None))]


@torch_e2e_test
def test_add():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    return Model(), [TensorSpec((None, None)),
                     TensorSpec((None, None))]
```

- [x] **Step 2: Rewrite `test_broadcast_add_reduce.py`**

Replace the entire content of `tests/torch_e2e/test_broadcast_add_reduce.py` with:

```python
"""broadcast + add + reduce_sum 端到端测试"""
import torch
from torch2linalg import torch_e2e_test, TensorSpec


@torch_e2e_test
def test_broadcast_add_reduce():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)
    return Model(), [TensorSpec((None,)),
                     TensorSpec((None, None))]
```

- [ ] **Step 3: Commit** (deferred to end)

---

### Task 5: Update README

**Files:**
- Modify: `tests/torch_e2e/README.md`

- [x] **Step 1: Update README.md**

Replace the entire content of `tests/torch_e2e/README.md` with:

```markdown
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
  → [step2] --transform-interpreter（自动生成 transform 脚本）
  → [step3] --one-shot-bufferize
  → [step4] --ascendc-buffer-placement
  → [step5] --linalg-to-ascendc
  → [step6] --ascendc-parallelize
  → [step7] --ascendc-prepare-for-emit
  → [step7b] --canonicalize-cann-signature
  → [step8] afir-translate -mlir-to-cann → step8_kernel.cpp + tiling_space.json
  → [step9-10] autotuner → compile + tiling 搜索 + 数值验证
```

中间产物保存在 `output/torch_e2e/<test_name>/`，便于调试。

## 判定标准

测试通过条件：autotuner 成功完成 compile + tiling 搜索 + 数值正确性验证。
```

- [ ] **Step 2: Commit** (deferred to end)

---

### Task 6: Smoke test

- [x] **Step 1: Verify imports work**

Run: `cd /home/gser/code/Ascend-MLIR && conda run -n torch-mlir python -c "from torch2linalg import torch_e2e_test, TensorSpec, torch_to_linalg; print('imports OK')"`

Expected: `imports OK`

- [x] **Step 2: Run a single test case end-to-end**

Run: `cd /home/gser/code/Ascend-MLIR && conda run -n torch-mlir python -m pytest tests/torch_e2e/test_elementwise.py::test_add -v -s`

Expected: Test runs through all stages. If tools (afir-opt, afir-translate, autotuner) are on PATH, expect PASS. If not, expect a clear assertion error indicating which tool is missing.

- [x] **Step 3: If test passes, run all tests**

Run: `cd /home/gser/code/Ascend-MLIR && conda run -n torch-mlir python -m pytest tests/torch_e2e/ -v -s`

Observe results. Some tests may fail at specific pipeline stages — this is expected if the MLIR passes don't handle all patterns yet. The important thing is that the framework infrastructure works correctly.