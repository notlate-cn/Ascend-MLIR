# 移除测试框架中的编译/运行时参数

**日期**：2026-03-23
**状态**：待实现
**前置**：[torch-e2e-test-framework-design](2026-03-20-torch-e2e-test-framework-design.md)

---

## 1. 背景

当前 e2e 测试框架要求每个测试用例通过装饰器指定 tiling 等编译/运行时参数：

```python
@torch_e2e_test(tiling={"TB_M": 16, "TB_N": 4})
def test_add_mul():
    ...
```

问题：
- **tiling 不属于编译期** — 具体 tiling 值应在 runtime 阶段由 autotune 模块生成，编译期应使用符号化 tiling
- **`_generate_transform_script()` 已经是符号化的** — tile size 为 `%tb_m`, `%tb_inner_m`，不依赖 tiling dict。tiling dict 仅作为"是否生成 Transform 脚本"的条件判断
- **`rtol`/`atol` 无使用场景** — 当前测试只做到 kernel 生成，没有 compile + execute 验证
- **`kernel_cpp` 与测试目标矛盾** — 测试目标是验证 torch → kernel 生成通路，跳过 pipeline 没有意义
- **`dtype` 应由测试用例控制** — 框架不应隐式做类型转换，测试用例应显式指定 tensor dtype，以自然支持 mixed dtype / cast 场景

---

## 2. 目标

- 装饰器变为零参数：`@torch_e2e_test`
- 移除 `tiling`、`rtol`、`atol`、`kernel_cpp`、`dtype`、`transform_mlir` 所有参数
- `dtype` 由测试用例在构造 tensor 时显式指定
- Pipeline 默认总是自动生成符号化 Transform 脚本
- 测试通过标准：kernel C++ 代码能正常生成（Stage 0-8 成功）

---

## 3. 设计

### 3.1 装饰器接口

**改前：**
```python
def torch_e2e_test(
    func=None,
    *,
    transform_mlir: str | Path | None = None,
    tiling: dict | None = None,
    kernel_cpp: str | Path | None = None,
    rtol: float = 1e-2,
    atol: float = 1.0,
    dtype: torch.dtype = torch.float16,
):
```

**改后：**
```python
def torch_e2e_test(func):
```

不再需要 `func=None` + `decorator` 两层包装，因为没有可选参数。

### 3.2 装饰器内部逻辑

**改前：**
```python
def decorator(fn):
    @functools.wraps(fn)
    def wrapper():
        model, inputs = fn()
        ...
        torch_to_linalg(model, inputs, output_path=linalg_path, dtype=dtype)
        ...
        if kernel_cpp is not None:
            resolved_kernel_cpp = Path(kernel_cpp)
        else:
            resolved_kernel_cpp = _run_mlir_pipeline(
                linalg_path, work_dir, transform_mlir, tiling
            )
        ...
        model_typed = model.to(dtype).eval()
        inputs_typed = [x.to(dtype) for x in inputs]
    return wrapper
```

**改后：**
```python
@functools.wraps(func)
def wrapper():
    model, inputs = func()
    ...
    torch_to_linalg(model, inputs, output_path=linalg_path)
    ...
    resolved_kernel_cpp = _run_mlir_pipeline(linalg_path, work_dir)
    ...
    # 移除 Stage 9 torch reference（当前不做验证）
    assert resolved_kernel_cpp is not None and resolved_kernel_cpp.exists()
return wrapper
```

关键变更：
- `torch_to_linalg()` 不再传 `dtype`，从 tensor 实际 dtype 推导
- 直接调用 `_run_mlir_pipeline()`，不再有 `kernel_cpp` 分支
- 移除 Stage 9 torch reference 计算（当前不做精度验证）
- 移除 `tiling` 打印语句

### 3.3 `_run_mlir_pipeline()` 签名

**改前：**
```python
def _run_mlir_pipeline(
    linalg_path: Path,
    work_dir: Path,
    transform_mlir: str | Path | None,
    tiling: dict | None = None,
) -> Path | None:
```

**改后：**
```python
def _run_mlir_pipeline(
    linalg_path: Path,
    work_dir: Path,
) -> Path | None:
```

Stage 2 条件逻辑简化为总是自动生成 Transform 脚本：

```python
# 改前
if transform_mlir is not None:
    transform_path = Path(transform_mlir)
elif tiling is not None:
    transform_path = _generate_transform_script(step1, work_dir)
else:
    transform_path = None

# 改后
transform_path = _generate_transform_script(step1, work_dir)
```

若 `_generate_transform_script` 返回 None（如无 parallel 维度），跳过 tiling 并打印 `"[step2] 跳过（无 parallel 维度，无法自动 tiling）"`。

Pipeline 返回 `Path | None`，失败时返回 None，装饰器 assert 检查。

### 3.4 `convert.py` 变更

**改前：**
```python
def torch_to_linalg(
    model: nn.Module,
    sample_inputs: list[torch.Tensor],
    output_path: Optional[str | Path] = None,
    dtype: torch.dtype = torch.float16,
) -> str:
    model = model.to(dtype).eval()
    inputs = tuple(x.to(dtype) for x in sample_inputs)
    ...
```

**改后：**
```python
def torch_to_linalg(
    model: nn.Module,
    sample_inputs: list[torch.Tensor],
    output_path: Optional[str | Path] = None,
) -> str:
    model = model.eval()
    inputs = tuple(sample_inputs)
    ...
```

移除 `dtype` 参数，不再做隐式类型转换。调用方负责确保 model 参数和 input tensor 的 dtype 一致。

### 3.5 测试用例

**改前：**
```python
@torch_e2e_test(tiling={"TB_M": 16, "TB_N": 4})
def test_add_mul():
    class Model(torch.nn.Module):
        def forward(self, a, b, c):
            return (a + b) * c
    return Model(), [torch.randn(32, 64)] * 3
```

**改后：**
```python
@torch_e2e_test
def test_add_mul():
    class Model(torch.nn.Module):
        def forward(self, a, b, c):
            return (a + b) * c
    return Model(), [torch.randn(32, 64, dtype=torch.float16)] * 3
```

`dtype` 在 `torch.randn()` 时显式指定，框架不做隐式转换。

带参数的模块需要同时 cast model：

```python
@torch_e2e_test
def test_linear():
    model = torch.nn.Linear(64, 32).to(torch.float16)
    return model, [torch.randn(8, 64, dtype=torch.float16)]
```

所有测试文件在同一个 commit 中原子迁移。

---

## 4. 涉及文件

| 文件 | 变更 |
|------|------|
| `python/torch2linalg/pytest_plugin.py` | 移除所有装饰器参数；简化装饰器为单层；移除 `_run_mlir_pipeline` 的 `tiling`/`transform_mlir` 参数；移除 Stage 9；简化 Stage 2 条件逻辑 |
| `python/torch2linalg/convert.py` | `torch_to_linalg()` 移除 `dtype` 参数，从 tensor 实际 dtype 推导 |
| `tests/torch_e2e/test_broadcast_add_reduce.py` | 移除 `tiling` 参数，tensor 显式指定 dtype |
| `tests/torch_e2e/test_elementwise.py` | 移除所有 5 处 `tiling` 参数，tensor 显式指定 dtype |
| `tests/torch_e2e/README.md` | 更新示例代码和参数表 |

不涉及 MLIR pass 或 C++ 代码的改动。

---

## 5. 后续演进

| 阶段 | 内容 |
|------|------|
| 当前 | 移除编译/运行时参数，自动生成符号化 Transform 脚本 |
| autotuner 集成后 | 在 runtime 阶段通过 autotune 模块搜索最优 tiling 配置 |
| compile + execute 就绪后 | 按需加回 `rtol`/`atol` 用于精度验证 |