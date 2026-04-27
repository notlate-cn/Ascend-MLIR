# Torch E2E Pipeline Alignment 设计文档

## 背景

现有 `pytest_plugin.py` 的 pipeline 停留在旧版本（stage 1-8），与 `examples/broadcast-add-reduce/run.sh` 已经脱节：

- 缺少 stage 7b（`--canonicalize-cann-signature`）
- 使用旧工具名（`ascir-translate` → 应为 `afir-translate`）
- 缺少 stage 9（compile）和 stage 10（autotuner run + verify）
- 测试用例使用静态 shape，IR 里维度是固定值
- 判定标准只检查文件生成，没有数值验证

## 目标

1. pytest pipeline 完整对齐 run.sh 的 stage 0-10
2. 支持动态 shape（IR 里维度为 `?`）
3. 端到端数值正确性验证（通过 autotuner）

## 限制

- **仅支持单输出模型**：compiler 工具当前限制 `num-outputs=1`，多输出模型暂不支持。
- **autotuner 内部已集成 compile**：不需要单独调用 compiler（stage 9），autotuner 会自动编译 kernel。

## 设计

### 1. TensorSpec

新增 `python/torch2linalg/tensor_spec.py`：

```python
@dataclass
class TensorSpec:
    shape: tuple          # (None, None) 全动态, (32, None) 第一维静态
    dtype: torch.dtype = torch.float16

    def make_sample(self) -> torch.Tensor:
        """用默认值填充 None 维度，生成具体 tensor"""
        concrete = tuple(64 if d is None else d for d in self.shape)
        return torch.randn(concrete, dtype=self.dtype)

    def dynamic_dims(self) -> dict[int, Dim]:
        """返回 torch.export 需要的 dynamic_shapes 映射"""
        return {i: Dim(f"d{i}") for i, d in enumerate(self.shape) if d is None}
```

测试用例写法：

```python
@torch_e2e_test
def test_add():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    return Model(), [TensorSpec((None, None), torch.float16),
                     TensorSpec((None, None), torch.float16)]
```

### 2. torch_to_linalg 改造

`convert.py` 新增 `dynamic_shapes` 参数透传给 `export_and_import`：

```python
def torch_to_linalg(
    model: nn.Module,
    sample_inputs: list[torch.Tensor],
    dynamic_shapes: dict | None = None,
    output_path: Optional[str | Path] = None,
) -> str:
    model = model.eval()
    inputs = tuple(sample_inputs)
    module = export_and_import(
        model, *inputs,
        output_type=OutputType.LINALG_ON_TENSORS,
        dynamic_shapes=dynamic_shapes,
    )
    mlir_text = module.operation.get_asm()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(mlir_text)
    return mlir_text
```

### 3. Pipeline 拆分

将现有的 `_run_mlir_pipeline` 拆为两个函数：

#### `_run_mlir_pipeline(work_dir) -> bool`

负责 stage 1-8（纯 MLIR 编译 + codegen，不需要具体 shape）：

- Stage 1: `--linalg-fuse-elementwise-ops`
- Stage 2: `--transform-interpreter`（自动生成 transform 脚本）
- Stage 3: `--one-shot-bufferize`
- Stage 4: `--ascendc-buffer-placement`
- Stage 5: `--linalg-to-ascendc`
- Stage 6: `--ascendc-parallelize`
- Stage 7: `--ascendc-prepare-for-emit`
- **Stage 7b**: `--canonicalize-cann-signature`（新增，去除 transform ops，生成 CANN 标准签名）
- **Stage 8**: `afir-translate -mlir-to-cann --tiling-space-out step8_kernel.tiling_space.json`（替换旧 ascir-translate -mlir-to-ascendc，同时去掉手动 strip transform.named_sequence 的 hack）

入口校验：

```python
linalg_path = work_dir / "step0_linalg.mlir"
assert linalg_path.exists(), f"缺少 {linalg_path}"
```

#### `_run_autotuner(work_dir, num_inputs, shape_str) -> bool`

负责 stage 9-10（autotuner 一站式完成 compile + tiling 搜索 + 数值验证）：

入口校验：

```python
for i in range(num_inputs):
    assert (work_dir / f"input_{i}.npy").exists(), f"缺少 input_{i}.npy"
assert (work_dir / "expected_0.npy").exists(), "缺少 expected_0.npy"
assert (work_dir / "step8_kernel.cpp").exists(), "缺少 step8_kernel.cpp"
assert (work_dir / "step8_kernel.tiling_space.json").exists(), "缺少 tiling_space.json"
assert shape_str, "shape_str 不能为空"
```

调用 autotuner：

```
autotuner --space step8_kernel.tiling_space.json \
          --kernel step8_kernel.cpp \
          --inputs input_0.npy,input_1.npy \
          --expected expected_0.npy \
          --shape "arg0_dim0=64,arg0_dim1=64,..."
```

### 4. 装饰器流程

```python
def wrapper():
    model, specs = func()
    work_dir = OUTPUT_ROOT / func.__name__
    # 清理 & 创建 work_dir

    # 1. 生成具体 tensor
    inputs = [spec.make_sample() for spec in specs]

    # 2. PyTorch reference run → 存 npy
    for i, tensor in enumerate(inputs):
        np.save(work_dir / f"input_{i}.npy", tensor.numpy())
    expected = model(*inputs)
    np.save(work_dir / "expected_0.npy", expected.numpy())

    # 3. 构建 dynamic_shapes
    dynamic_shapes = {
        i: spec.dynamic_dims() for i, spec in enumerate(specs)
        if spec.dynamic_dims()
    }

    # 4. torch → linalg（动态 shape）
    torch_to_linalg(model, inputs, dynamic_shapes,
                     output_path=work_dir / "step0_linalg.mlir")

    # 5. MLIR pipeline (stage 1-8)
    assert _run_mlir_pipeline(work_dir)

    # 6. Autotuner (stage 9-10)
    shape_str = _build_shape_str(specs, inputs)
    assert _run_autotuner(work_dir, len(inputs), shape_str)
```

### 5. shape_str 生成

```python
def _build_shape_str(specs: list[TensorSpec], samples: list[torch.Tensor]) -> str:
    parts = []
    for arg_idx, (spec, sample) in enumerate(zip(specs, samples)):
        for dim_idx, size in enumerate(sample.shape):
            parts.append(f"arg{arg_idx}_dim{dim_idx}={size}")
    return ",".join(parts)
```

### 6. work_dir 约定

`output/torch_e2e/<test_name>/` 下的文件布局：

| 文件 | 来源 |
|------|------|
| `input_0.npy`, `input_1.npy`, ... | 装饰器生成 |
| `expected_0.npy` | 装饰器生成（PyTorch reference） |
| `step0_linalg.mlir` | 装饰器生成（torch-mlir） |
| `step1_fused.mlir` ~ `step7_kernel.mlir` | `_run_mlir_pipeline` |
| `step7_cann.mlir` | `_run_mlir_pipeline`（stage 7b） |
| `step8_kernel.cpp` | `_run_mlir_pipeline`（stage 8） |
| `step8_kernel.tiling_space.json` | `_run_mlir_pipeline`（stage 8） |
| autotuner 输出（tiling_func.cpp 等） | `_run_autotuner`（stage 9-10） |

## 改动清单

| 文件 | 改动 |
|------|------|
| `python/torch2linalg/tensor_spec.py` | **新增** TensorSpec dataclass |
| `python/torch2linalg/__init__.py` | 导出 TensorSpec |
| `python/torch2linalg/convert.py` | 加 `dynamic_shapes` 参数 |
| `python/torch2linalg/pytest_plugin.py` | 装饰器改用 TensorSpec；pipeline 拆为 `_run_mlir_pipeline` + `_run_autotuner`；新增 stage 7b、更新 stage 8、新增 autotuner 调用；去掉旧 ascir-translate 逻辑和 transform strip hack |
| `tests/torch_e2e/test_elementwise.py` | 用例改用 TensorSpec |
| `tests/torch_e2e/test_broadcast_add_reduce.py` | 用例改用 TensorSpec |
| `tests/torch_e2e/README.md` | 更新用例写法和 pipeline 说明 |