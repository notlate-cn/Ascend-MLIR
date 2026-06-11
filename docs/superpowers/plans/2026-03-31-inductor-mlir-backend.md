# Inductor → MLIR Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal inductor backend that outputs fused LoopIR as linalg MLIR, enabling reuse of inductor's fusion capabilities.

**Architecture:** Register a custom scheduling backend (MLIRScheduling) that inherits SIMDScheduling's fusion logic, intercept FusedSchedulerNode after fusion via `_post_fusion_custom_pass`, extract LoopIR via custom OpsHandler trace, emit linalg.generic MLIR, and plug into existing Ascend-MLIR pipeline at stage 2.

**Tech Stack:** PyTorch Inductor, Python, MLIR, pytest

---

## File Structure

```
python/inductor_backend/
  __init__.py              # setup_inductor_backend() 入口
  scheduling.py            # MLIRScheduling class (继承 SIMDScheduling)
  post_fusion_pass.py      # _post_fusion_custom_pass 实现
  ops_handler.py           # MLIROpsHandler (inner_fn → MLIR)
  mlir_emitter.py          # MLIR 文本生成工具

test/inductor_e2e/
  conftest.py             # pytest fixture (环境配置)
  test_broadcast_add_reduce.py  # POC 验证用例
```

---

### Task 1: Create project structure and base files

**Files:**
- Create: `python/inductor_backend/__init__.py`
- Create: `python/inductor_backend/scheduling.py`
- Create: `python/inductor_backend/post_fusion_pass.py`
- Create: `python/inductor_backend/ops_handler.py`
- Create: `python/inductor_backend/mlir_emitter.py`
- Create: `test/inductor_e2e/conftest.py`
- Create: `test/inductor_e2e/test_broadcast_add_reduce.py`

- [ ] **Step 1: Create package __init__.py**

```python
# python/inductor_backend/__init__.py

from .scheduling import MLIRScheduling, setup_inductor_backend
from .post_fusion_pass import create_post_fusion_pass

__all__ = ['MLIRScheduling', 'setup_inductor_backend', 'create_post_fusion_pass']
```

- [ ] **Step 2: Create base scheduling.py with MLIRScheduling class**

```python
# python/inductor_backend/scheduling.py

from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.codegen.common import register_backend_for_device

class MLIRScheduling(SIMDScheduling):
    """
    最小 backend，只为让 inductor 的融合逻辑跑起来。
    can_fuse_vertical/horizontal 完全继承 SIMDScheduling。
    codegen 相关方法空实现——我们在 post_fusion_pass 中自己处理。
    """

    def codegen_node(self, node):
        # 不走 inductor codegen
        pass

    def codegen_template(self, *args, **kwargs):
        pass

    def codegen_node_schedule(self, *args, **kwargs):
        pass

    def define_kernel(self, src_code, node_schedule, kernel):
        # SIMDScheduling 要求 override，空实现
        pass

    def codegen_sync(self):
        # BaseScheduling 要求 override
        pass

    def benchmark_fused_nodes(self, nodes):
        # BaseScheduling 要求 override，返回 (cost, name)
        return (0.0, "npu")

    def flush(self):
        pass

    def ready_to_flush(self):
        return False
```

- [ ] **Step 3: Create base post_fusion_pass.py**

```python
# python/inductor_backend/post_fusion_pass.py

from typing import List
from torch._inductor.scheduler import BaseSchedulerNode

def create_post_fusion_pass(output_dir: str):
    """
    创建一个 post-fusion pass，将 FusedSchedulerNode 转换为 linalg MLIR。
    output_dir: 输出 .mlir 文件的目录
    """
    def post_fusion_pass(nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
        # POC: 先简单打印节点信息
        for i, node in enumerate(nodes):
            print(f"Node {i}: {type(node).__name__}")
        return nodes
    return post_fusion_pass
```

- [ ] **Step 4: Create base ops_handler.py**

```python
# python/inductor_backend/ops_handler.py

from torch._inductor.ops_handler import OpsHandler

class MLIROpsHandler(OpsHandler):
    """
    拦截 inner_fn 的标量操作，构建 linalg.generic body。

    注意 inner_fn 调用约定：
    - Pointwise: inner_fn(index)          — index 是 sympy 表达式序列
    - Reduction: inner_fn(index, rindex)   — 多一个 reduction 索引参数
    """

    def __init__(self):
        self.operations = []
        self.ssa_counter = 0

    def _new_ssa(self):
        op_id = f"%{self.ssa_counter}"
        self.ssa_counter += 1
        return op_id

    def constant(self, value, dtype):
        op_id = self._new_ssa()
        self.operations.append({
            'type': 'constant', 'id': op_id,
            'value': value, 'dtype': dtype
        })
        return op_id

    def load(self, name, index):
        op_id = self._new_ssa()
        self.operations.append({
            'type': 'load', 'id': op_id,
            'name': name, 'index': index
        })
        return op_id

    def store(self, name, index, value, mode=None):
        self.operations.append({
            'type': 'store',
            'name': name, 'index': index, 'value': value
        })

    def store_reduction(self, name, index, value):
        """reduction 结果写回（OpsHandler 必需方法）"""
        self.operations.append({
            'type': 'store_reduction',
            'name': name, 'index': index, 'value': value
        })

    def add(self, a, b):
        op_id = self._new_ssa()
        self.operations.append({
            'type': 'add', 'id': op_id, 'lhs': a, 'rhs': b
        })
        return op_id

    def mul(self, a, b):
        op_id = self._new_ssa()
        self.operations.append({
            'type': 'mul', 'id': op_id, 'lhs': a, 'rhs': b
        })
        return op_id

    def reduction(self, dtype, src_dtype, reduction_type, value):
        """reduction combiner（sum/max 等），返回 SSA id"""
        op_id = self._new_ssa()
        self.operations.append({
            'type': 'reduction', 'id': op_id,
            'dtype': dtype, 'src_dtype': src_dtype,
            'reduction_type': reduction_type,  # "sum", "max", etc.
            'value': value
        })
        return op_id

    def index_expr(self, expr, dtype):
        """sympy 索引表达式（OpsHandler 必需方法）"""
        op_id = self._new_ssa()
        self.operations.append({
            'type': 'index_expr', 'id': op_id,
            'expr': str(expr), 'dtype': dtype
        })
        return op_id

    def indirect_indexing(self, x, size, check=True, wrap_neg=True):
        """间接索引（OpsHandler 必需方法）— POC 直接返回 sympy 符号"""
        import sympy
        return sympy.Symbol(f"indirect_{self.ssa_counter}")

    def masked(self, mask, body, other):
        """带 mask 的计算（OpsHandler 必需方法）— POC 直接执行 body"""
        return body()

    def to_dtype(self, value, dtype):
        # POC: 暂时不处理 dtype 转换
        return value

    def where(self, condition, input, other):
        op_id = self._new_ssa()
        self.operations.append({
            'type': 'where', 'id': op_id,
            'cond': condition, 'a': input, 'b': other
        })
        return op_id
```

- [ ] **Step 5: Create base mlir_emitter.py**

```python
# python/inductor_backend/mlir_emitter.py

class MLIREmitter:
    """
    生成 linalg MLIR 文本。
    """

    def __init__(self):
        self.lines = []
        self.indent_level = 0
        self.ssa_counter = 0

    def emit(self, text: str):
        indent = "  " * self.indent_level
        self.lines.append(f"{indent}{text}")

    def new_ssa(self) -> str:
        ssa = f"%{self.ssa_counter}"
        self.ssa_counter += 1
        return ssa

    def emit_module_header(self):
        self.emit('module {')
        self.indent_level += 1
        self.emit('func.func @main() {')
        self.indent_level += 1

    def emit_module_footer(self):
        self.indent_level -= 1
        self.emit('}')
        self.indent_level -= 1
        self.emit('}')

    def emit_linalg_generic(
        self,
        inputs: list,
        outputs: list,
        iterator_types: list,
        body_ops: list
    ):
        """
        生成一个 linalg.generic op。

        inputs: [(name, type), ...]
        outputs: [(name, type), ...]
        iterator_types: ['parallel', 'parallel', 'reduction', ...]
        body_ops: ops_handler 产生的操作列表
        """
        # 输入参数
        for name, ty in inputs:
            self.emit(f'%{name} = tensor.empty() : tensor<{ty}>')

        # 输出参数
        for name, ty in outputs:
            self.emit(f'%{name} = tensor.empty() : tensor<{ty}>')

        # linalg.generic 头部
        in_types = ', '.join([f'tensor<{ty}>' for _, ty in inputs])
        out_types = ', '.join([f'tensor<{ty}>' for _, ty in outputs])
        iter_str = ', '.join([f'"{it}"' for it in iterator_types])

        self.emit(f'%result = linalg.generic {{indexing_maps = [], iterator_types = [{iter_str}]}}')
        self.emit(f'  ins({in_types}) : {in_types} -> ({out_types}) {{')

        # POC 阶段：简单输出两个参数的加法（Task 4 将扩展为完整实现）
        self.emit('  ^bb0(%arg0: f32, %arg1: f32):')
        self.emit('    %add = arith.addf %arg0, %arg1 : f32')
        self.emit('    linalg.yield %add : f32')

        self.emit('  }')

    def get_text(self) -> str:
        return '\n'.join(self.lines)
```

- [ ] **Step 6: Create test conftest.py**

```python
# test/inductor_e2e/conftest.py

import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'python'))
```

- [ ] **Step 7: Create base test file**

```python
# test/inductor_e2e/test_broadcast_add_reduce.py

import torch

def test_inductor_backend_registration():
    """测试 MLIRScheduling 能正确注册并运行 inductor 融合"""
    from inductor_backend import setup_inductor_backend

    # 注册 backend
    setup_inductor_backend()

    # 简单模型
    class SimpleModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = SimpleModel()
    example_inputs = (torch.randn(4, 4), torch.randn(4, 4))

    # 运行 inductor
    with torch._inductor.config.patch("backend", "npu"):
        compiled = torch.compile(model, backend="inductor")
        result = compiled(*example_inputs)

        # 验证数值正确性
        expected = model(*example_inputs)
        assert torch.allclose(result, expected, atol=1e-5), "Inductor backend output mismatch"

    print("✓ Backend registration and inductor fusion work")
```

- [ ] **Step 8: Commit initial structure**

```bash
git add python/inductor_backend/ test/inductor_e2e/
git commit -m "feat(inductor): add base structure for inductor-mlir backend"
```

---

### Task 2: Complete setup_inductor_backend function

**Files:**
- Modify: `python/inductor_backend/scheduling.py`

- [ ] **Step 1: Implement setup_inductor_backend function**

```python
# python/inductor_backend/scheduling.py

from torch._inductor.codegen.common import register_backend_for_device

class MLIRScheduling(SIMDScheduling):
    # Task 1 中定义的类

def setup_inductor_backend(post_fusion_pass=None):
    """
    注册 MLIRScheduling 为 inductor 的 npu backend。

    Args:
        post_fusion_pass: 可选的自定义 pass，在融合完成后执行
    """
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen

    # register_backend_for_device 完整签名（torch 2.10+）：
    #   device, device_scheduling, device_wrapper_codegen,
    #   device_cpp_wrapper_codegen=None, device_fx_wrapper_codegen=None,
    #   device_custom_pass=None, device_custom_config=None
    register_backend_for_device(
        device="npu",
        device_scheduling=MLIRScheduling,
        device_wrapper_codegen=PythonWrapperCodegen,
    )

    # 如果提供了 post_fusion_pass，设置配置
    if post_fusion_pass is not None:
        import torch._inductor.config
        torch._inductor.config._post_fusion_custom_pass = post_fusion_pass
```

- [ ] **Step 2: Run test to verify registration**

```bash
cd /home/gser/code/Ascend-MLIR
python -m pytest test/inductor_e2e/test_broadcast_add_reduce.py::test_inductor_backend_registration -v
```

Expected: PASS with "✓ Backend registration and inductor fusion work"

- [ ] **Step 3: Commit**

```bash
git add python/inductor_backend/scheduling.py
git commit -m "feat(inductor): add setup_inductor_backend function"
```

---

### Task 3: Implement FusedSchedulerNode extraction

**Files:**
- Modify: `python/inductor_backend/post_fusion_pass.py`

- [ ] **Step 1: Implement LoopIR node extraction**

```python
# python/inductor_backend/post_fusion_pass.py

from typing import List, Optional
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.scheduler import FusedSchedulerNode
from torch._inductor.ir import ComputedBuffer, Pointwise, Reduction

def _extract_loops_from_node(node: BaseSchedulerNode):
    """
    从 SchedulerNode 提取 LoopIR 信息。

    Returns:
        List of dicts with keys: node_type, ranges, reduction_ranges, inner_fn
    """
    loops_info = []

    if isinstance(node, FusedSchedulerNode):
        # 融合节点：遍历子节点
        for sub in node.get_nodes():
            buf = sub.node  # ComputedBuffer
            data = buf.data  # Pointwise 或 Reduction
            loops_info.append({
                'node_type': type(data).__name__,
                'node': data,
                'ranges': data.ranges,
                'reduction_ranges': getattr(data, 'reduction_ranges', []),
                'inner_fn': data.inner_fn,
                'dtype': data.dtype,
                'device': data.get_device()
            })
    else:
        # 单个节点
        buf = node.node
        if hasattr(buf, 'data'):
            data = buf.data
            if isinstance(data, (Pointwise, Reduction)):
                loops_info.append({
                    'node_type': type(data).__name__,
                    'node': data,
                    'ranges': data.ranges,
                    'reduction_ranges': getattr(data, 'reduction_ranges', []),
                    'inner_fn': data.inner_fn,
                    'dtype': data.dtype,
                    'device': data.get_device()
                })

    return loops_info


def create_post_fusion_pass(output_dir: str, verbose: bool = True):
    """
    创建一个 post-fusion pass，将 FusedSchedulerNode 转换为 linalg MLIR。

    Args:
        output_dir: 输出 .mlir 文件的目录
        verbose: 是否打印调试信息

    Returns:
        可以传给 _post_fusion_custom_pass 的函数
    """
    def post_fusion_pass(nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
        if verbose:
            print(f"\n=== Post-Fusion Pass (output_dir={output_dir}) ===")
            print(f"Total nodes: {len(nodes)}")

        all_loops = []
        for i, node in enumerate(nodes):
            node_type = type(node).__name__
            if verbose:
                print(f"Node {i}: {node_type}")

            loops = _extract_loops_from_node(node)
            if verbose:
                for j, loop in enumerate(loops):
                    print(f"  Loop[{j}]: {loop['node_type']}, "
                          f"dtype={loop['dtype']}, "
                          f"ranges={loop['ranges']}, "
                          f"reduction_ranges={loop['reduction_ranges']}")

            all_loops.extend(loops)

        if verbose:
            print(f"Total LoopIR nodes extracted: {len(all_loops)}")
            print("=== End Post-Fusion Pass ===\n")

        # POC 阶段：暂不生成 MLIR，Task 4 将实现完整转换
        # 必须返回 nodes，inductor 才能继续（虽然 codegen 是空的）
        return nodes

    return post_fusion_pass
```

- [ ] **Step 2: Update test to use create_post_fusion_pass**

```python
# test/inductor_e2e/test_broadcast_add_reduce.py

import torch

def test_inductor_backend_registration():
    """测试 MLIRScheduling 能正确注册并运行 inductor 融合"""
    from inductor_backend import setup_inductor_backend, create_post_fusion_pass

    # 创建输出目录
    import os
    output_dir = "/tmp/inductor_e2e_output"
    os.makedirs(output_dir, exist_ok=True)

    # 创建 post-fusion pass
    post_fusion_pass = create_post_fusion_pass(output_dir, verbose=True)

    # 注册 backend
    setup_inductor_backend(post_fusion_pass=post_fusion_pass)

    # 简单模型
    class SimpleModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = SimpleModel()
    example_inputs = (torch.randn(4, 4), torch.randn(4, 4))

    # 运行 inductor
    with torch._inductor.config.patch("backend", "npu"):
        compiled = torch.compile(model, backend="inductor")
        result = compiled(*example_inputs)

        # 验证数值正确性
        expected = model(*example_inputs)
        assert torch.allclose(result, expected, atol=1e-5), "Inductor backend output mismatch"

    print("✓ Backend registration and inductor fusion work")
```

- [ ] **Step 3: Run test to verify LoopIR extraction**

```bash
cd /home/gser/code/Ascend-MLIR
python -m pytest test/inductor_e2e/test_broadcast_add_reduce.py::test_inductor_backend_registration -v -s
```

Expected: PASS with console output showing "Total LoopIR nodes extracted: X"

- [ ] **Step 4: Commit**

```bash
git add python/inductor_backend/post_fusion_pass.py test/inductor_e2e/test_broadcast_add_reduce.py
git commit -m "feat(inductor): implement FusedSchedulerNode extraction"
```

---

### Task 4: Implement inner_fn trace → linalg MLIR conversion

**Files:**
- Modify: `python/inductor_backend/mlir_emitter.py`
- Modify: `python/inductor_backend/post_fusion_pass.py`

- [ ] **Step 1: Enhance MLIREmitter for full linalg.generic**

```python
# python/inductor_backend/mlir_emitter.py

class MLIREmitter:
    """生成 linalg MLIR 文本。"""

    def __init__(self):
        self.lines = []
        self.indent_level = 0
        self.ssa_counter = 0

    def emit(self, text: str):
        indent = "  " * self.indent_level
        self.lines.append(f"{indent}{text}")

    def new_ssa(self) -> str:
        ssa = f"%{self.ssa_counter}"
        self.ssa_counter += 1
        return ssa

    def emit_module_header(self):
        self.emit('module {')
        self.indent_level += 1

    def emit_module_footer(self):
        self.indent_level -= 1
        self.emit('}')

    def emit_func_header(self, name: str, inputs: list):
        """
        inputs: [(name, type), ...]
        """
        args = ', '.join([f'%{n}: {t}' for n, t in inputs])
        self.emit(f'func.func @{name}({args}) {{')
        self.indent_level += 1

    def emit_func_footer(self):
        self.indent_level -= 1
        self.emit('}')

    def emit_linalg_generic(
        self,
        name: str,
        inputs: list,
        outputs: list,
        indexing_maps: list,
        iterator_types: list,
        body_ops: list,
        dtype: str = "f32"
    ):
        """
        生成一个 linalg.generic op。

        inputs: [(name, type), ...]
        outputs: [(name, type), ...]
        indexing_maps: affine_map 文本列表
        iterator_types: ['parallel', 'parallel', 'reduction', ...]
        body_ops: ops_handler 产生的操作列表
        """
        # 输入参数
        for i, (n, t) in enumerate(inputs):
            self.emit(f'%{n} = tensor.empty() : tensor<{t}>')

        # 输出参数
        for n, t in outputs:
            self.emit(f'%{n} = tensor.empty() : tensor<{t}>')

        # linalg.generic 头部
        in_names = ', '.join([f'%{n}' for n, _ in inputs])
        out_names = ', '.join([f'%{n}' for n, _ in outputs])
        in_types = ', '.join([f'tensor<{t}>' for _, t in inputs])
        out_types = ', '.join([f'tensor<{t}>' for _, t in outputs])
        iter_str = ', '.join([f'"{it}"' for it in iterator_types])
        maps_str = ', '.join([f'affine_map<({m})>' for m in indexing_maps])

        self.emit(f'%{name} = linalg.generic {{')
        self.emit(f'  indexing_maps = [{maps_str}],')
        self.emit(f'  iterator_types = [{iter_str}]}}')
        self.emit(f'  ins({in_names}) : {in_types} outs({out_names}) -> ({out_types}) {{')

        # Body
        self.emit('  ^bb0(')
        self.indent_level += 1

        # 生成 block 参数
        num_args = len(inputs) + len(outputs)
        block_args = ', '.join([f'%arg{i}: {dtype}' for i in range(num_args)])
        self.emit(block_args + '):')

        # 生成 body 操作
        for op in body_ops:
            self._emit_body_op(op)

        self.indent_level -= 1
        self.emit('  }')

    def _emit_body_op(self, op: dict, elem_dtype: str = "f32"):
        op_type = op['type']

        if op_type == 'constant':
            self.emit(f"{op['id']} = arith.constant {op['value']} : {self._mlir_dtype(op['dtype'])}")

        elif op_type == 'load':
            # Load 映射为 block 参数，不需要 emit（参数由 ^bb0 绑定）
            pass

        elif op_type in ('store', 'store_reduction'):
            self.emit(f"linalg.yield {op['value']} : {elem_dtype}")

        elif op_type == 'add':
            self.emit(f"{op['id']} = arith.addf {op['lhs']}, {op['rhs']} : {elem_dtype}")

        elif op_type == 'mul':
            self.emit(f"{op['id']} = arith.mulf {op['lhs']}, {op['rhs']} : {elem_dtype}")

        elif op_type == 'reduction':
            # reduction combiner: 根据 reduction_type 选择对应 arith op
            rt = op['reduction_type']
            if rt == 'sum':
                self.emit(f"{op['id']} = arith.addf {op['value']}, %acc : {elem_dtype}")
            elif rt == 'max':
                self.emit(f"{op['id']} = arith.maximumf {op['value']}, %acc : {elem_dtype}")
            # POC: 更多 reduction_type 后续扩展

        elif op_type == 'index_expr':
            # POC: 索引表达式暂不 emit，仅记录
            pass

    def _mlir_dtype(self, torch_dtype) -> str:
        """转换 PyTorch dtype 到 MLIR 类型"""
        import torch
        if torch_dtype == torch.float32:
            return "f32"
        elif torch_dtype == torch.float16:
            return "f16"
        elif torch_dtype == torch.int32:
            return "i32"
        elif torch_dtype == torch.int64:
            return "i64"
        else:
            return "f32"  # POC 默认

    def get_text(self) -> str:
        return '\n'.join(self.lines)
```

- [ ] **Step 2: Implement MLIR generation in post_fusion_pass**

```python
# python/inductor_backend/post_fusion_pass.py

# 在 create_post_fusion_pass 函数中，将 Task 3 的简单实现替换为：

def post_fusion_pass(nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
    if verbose:
        print(f"\n=== Post-Fusion Pass (output_dir={output_dir}) ===")

    from .mlir_emitter import MLIREmitter
    from .ops_handler import MLIROpsHandler

    emitter = MLIREmitter()
    emitter.emit_module_header()

    all_loops = []
    for i, node in enumerate(nodes):
        node_type = type(node).__name__
        if verbose:
            print(f"Node {i}: {node_type}")

        loops = _extract_loops_from_node(node)
        all_loops.extend(loops)

        for j, loop in enumerate(loops):
            if verbose:
                print(f"  Loop[{j}]: {loop['node_type']}, dtype={loop['dtype']}")

            # 调用 inner_fn 捕获操作
            handler = MLIROpsHandler()

            # ranges 和 reduction_ranges 是独立的列表
            # ranges = parallel 维度, reduction_ranges = reduction 维度
            ranges = loop['ranges']
            reduction_ranges = loop['reduction_ranges']
            num_parallel = len(ranges)
            num_reduction = len(reduction_ranges)
            total_dims = num_parallel + num_reduction

            iterator_types = ['parallel'] * num_parallel + ['reduction'] * num_reduction

            # inner_fn 调用约定不同：
            #   Pointwise:  inner_fn(index)           — index 长度 = len(ranges)
            #   Reduction:  inner_fn(index, rindex)    — 多一个 reduction 索引
            import sympy
            index_vars = [sympy.Symbol(f"d{k}") for k in range(num_parallel)]
            rindex_vars = [sympy.Symbol(f"d{num_parallel + k}") for k in range(num_reduction)]

            try:
                if loop['node_type'] == 'Reduction':
                    loop['inner_fn'](index_vars, rindex_vars)
                else:
                    loop['inner_fn'](index_vars)
            except Exception as e:
                if verbose:
                    print(f"    ⚠ Skipping: {e}")
                continue

            # 生成 linalg.generic
            dtype = emitter._mlir_dtype(loop['dtype'])

            # 输入/输出 tensor 名称 (从 handler 提取)
            input_names = _extract_input_names(handler.operations)
            output_name = _extract_output_name(handler.operations)

            # POC indexing maps: 输入用 identity, 输出根据是否有 reduction 调整
            dim_vars = ', '.join([f'd{k}' for k in range(total_dims)])
            all_dims_map = f'{dim_vars} -> ({dim_vars})'
            # 输出 map 仅包含 parallel 维度
            parallel_dims = ', '.join([f'd{k}' for k in range(num_parallel)])
            out_map = f'{dim_vars} -> ({parallel_dims})' if num_reduction > 0 else all_dims_map
            indexing_maps = [all_dims_map] * len(input_names) + [out_map]

            emitter.emit_linalg_generic(
                name=f"op_{i}_{j}",
                inputs=[(n, dtype) for n in input_names],
                outputs=[(output_name, dtype)],
                indexing_maps=indexing_maps,
                iterator_types=iterator_types,
                body_ops=handler.operations,
                dtype=dtype
            )

    emitter.emit_module_footer()

    # 写入文件
    mlir_path = f"{output_dir}/inductor_linalg.mlir"
    with open(mlir_path, "w") as f:
        f.write(emitter.get_text())

    if verbose:
        print(f"✓ MLIR written to {mlir_path}")
        print(f"Total nodes: {len(nodes)}, Total LoopIR: {len(all_loops)}")
        print("=== End Post-Fusion Pass ===\n")

    return nodes


def _extract_input_names(operations: list) -> list:
    """从操作列表中提取输入 tensor 名称"""
    names = []
    for op in operations:
        if op['type'] == 'load':
            name = op['name']
            if name not in names:
                names.append(name)
    return names


def _extract_output_name(operations: list) -> str:
    """从操作列表中提取输出 tensor 名称（最后一个 store/store_reduction 操作）"""
    for op in reversed(operations):
        if op['type'] in ('store', 'store_reduction'):
            return op['name']
    # 如果没有 store，说明 inner_fn trace 不完整
    raise ValueError("No store/store_reduction found in operations — inner_fn trace 可能不完整")
```

- [ ] **Step 3: 验证 MLIREmitter._mlir_dtype 作为实例方法可用**

`_mlir_dtype` 保持为实例方法（Task 4 Step 1 已定义），在 post_fusion_pass 中通过 `emitter._mlir_dtype(...)` 调用。不要改为 `@staticmethod`，以保持一致性。

- [ ] **Step 4: Run test to verify MLIR generation**

```bash
cd /home/gser/code/Ascend-MLIR
python -m pytest test/inductor_e2e/test_broadcast_add_reduce.py::test_inductor_backend_registration -v -s
```

Expected: PASS and `/tmp/inductor_e2e_output/inductor_linalg.mlir` contains valid linalg MLIR

- [ ] **Step 5: Commit**

```bash
git add python/inductor_backend/mlir_emitter.py python/inductor_backend/post_fusion_pass.py
git commit -m "feat(inductor): implement inner_fn to linalg MLIR conversion"
```

---

### Task 5: Create broadcast-add-reduce test with fusion verification

**Files:**
- Modify: `test/inductor_e2e/test_broadcast_add_reduce.py`

- [ ] **Step 1: Add broadcast-add-reduce test case**

```python
# test/inductor_e2e/test_broadcast_add_reduce.py

import torch

def test_inductor_backend_registration():
    """测试 MLIRScheduling 能正确注册并运行 inductor 融合"""
    # ... (existing code) ...


def test_broadcast_add_reduce_fusion():
    """
    验证 inductor 能将 elementwise + reduction 融合成一个 kernel。

    这是当前 linalg-fuse-elementwise-ops 做不到的，
    但 inductor 可以实现。
    """
    from inductor_backend import setup_inductor_backend, create_post_fusion_pass

    import os
    output_dir = "/tmp/inductor_e2e_output"
    os.makedirs(output_dir, exist_ok=True)

    post_fusion_pass = create_post_fusion_pass(output_dir, verbose=True)
    setup_inductor_backend(post_fusion_pass=post_fusion_pass)

    class BroadcastAddReduceModel(torch.nn.Module):
        def forward(self, a, b):
            # a: [M], b: [M, N] → broadcast → add → reduce
            return (a.unsqueeze(1) + b).sum(dim=1)

    model = BroadcastAddReduceModel()
    example_inputs = (torch.randn(128), torch.randn(128, 16))

    # 运行 inductor
    with torch._inductor.config.patch("backend", "npu"):
        compiled = torch.compile(model, backend="inductor")
        result = compiled(*example_inputs)

        # 验证数值正确性
        expected = model(*example_inputs)
        assert torch.allclose(result, expected, atol=1e-5), "Output mismatch"

    print("✓ broadcast-add-reduce fusion works")
```

- [ ] **Step 2: Run test to verify fusion**

```bash
cd /home/gser/code/Ascend-MLIR
python -m pytest test/inductor_e2e/test_broadcast_add_reduce.py::test_broadcast_add_reduce_fusion -v -s
```

Expected: PASS and output shows FusedSchedulerNode containing both elementwise and reduction

- [ ] **Step 3: Commit**

```bash
git add test/inductor_e2e/test_broadcast_add_reduce.py
git commit -m "test(inductor): add broadcast-add-reduce fusion test"
```

---

### Task 6: Connect to existing Ascend-MLIR pipeline (stage 2+)

> **注意**：原 Task 6 和 Task 7 合并为一个 Task，避免依赖断裂。

**Files:**
- Modify: `python/torch/framework/pipeline.py`
- Modify: `test/inductor_e2e/test_broadcast_add_reduce.py`

**关键约束**：
1. `torch_e2e_test` 装饰器必须保持 `func=None, *, ...` 双调用模式（兼容 `@torch_e2e_test` 和 `@torch_e2e_test(...)`）
2. `_run_afir_opt` 实际签名是 `(input_path: Path, output_path: Path, passes: list[str], afir_opt: str)`
3. `_generate_transform_script` 签名是 `(fused_path: Path, work_dir: Path) -> Path | None`
4. 所有路径统一使用 `Path` 对象（不用字符串拼接）

- [ ] **Step 1: 添加 _run_mlir_pipeline_from_inductor 函数**

从 inductor 生成的 MLIR 入口，跳过 stage 0/0a/0b/1，从 stage 2 (tiling) 开始。
复用 `_run_mlir_pipeline` 中 stage 2-8 的真实 pass 和参数。

```python
# python/torch/framework/pipeline.py — 在 _run_mlir_pipeline 后面添加

def _run_mlir_pipeline_from_inductor(work_dir: Path) -> bool:
    """
    从 inductor 生成的 linalg MLIR 开始执行 pipeline（stage 2-8）。

    跳过 stage 0 (torch-mlir)、0a (eliminate-cf-assert)、
    0b (fold-unit-extent-dims)、1 (fuse-elementwise-ops)。
    inductor 已完成融合，输出的 MLIR 直接进入 tiling。
    """
    inductor_mlir = work_dir / "inductor_linalg.mlir"
    assert inductor_mlir.exists(), f"缺少 {inductor_mlir}"

    afir_opt = _find_tool("afir-opt")
    if afir_opt is None:
        print("  afir-opt 不在 PATH 中，跳过")
        return False

    # ── Stage 2: Tiling (Transform Interpreter) ──
    # inductor MLIR 相当于 step1_fused.mlir，直接作为 transform 输入
    transform_path = _generate_transform_script(inductor_mlir, work_dir)

    if transform_path is not None:
        step2 = work_dir / "step2_tiled.mlir"
        print(f"  [step2] --transform-interpreter ({transform_path.name})")
        if not _run_afir_opt(transform_path, step2,
                             ["--transform-interpreter", "--canonicalize", "--cse"],
                             afir_opt):
            return False
    else:
        print(f"  [step2] 跳过（无 parallel 维度，无法自动 tiling）")
        step2 = inductor_mlir

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

    # ── Stage 8: C++ Codegen ──
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
```

- [ ] **Step 2: 扩展 torch_e2e_test 装饰器，添加 use_inductor 参数**

保持 `func=None, *` 双调用模式兼容性。

```python
# python/torch/framework/pipeline.py — 替换现有 torch_e2e_test 定义

def torch_e2e_test(func=None, *, verify_shapes: dict[str, int] | None = None,
                   use_inductor: bool = False):
    """
    pytest 装饰器：定义一个 torch → NPU e2e 测试。

    被装饰函数应返回 (model, specs: list[TensorSpec])。

    Args:
        verify_shapes: 验证阶段使用的具体维度大小，如 {"M": 64, "N": 128}。
        use_inductor: 若 True，使用 inductor 融合替代 linalg-fuse-elementwise-ops。
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper():
            model, specs = fn()
            test_name = fn.__name__
            work_dir = OUTPUT_ROOT / test_name
            if work_dir.exists():
                shutil.rmtree(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

            if use_inductor:
                # ── Inductor 路径：inductor 融合 → linalg MLIR → stage 2+ ──
                from inductor_backend import setup_inductor_backend, create_post_fusion_pass

                post_fusion_pass = create_post_fusion_pass(
                    str(work_dir), verbose=True)
                setup_inductor_backend(post_fusion_pass=post_fusion_pass)

                # 生成 trace 输入并运行 inductor
                trace_inputs = [spec.make_sample() for spec in specs]
                import torch._inductor.config
                with torch._inductor.config.patch("backend", "npu"):
                    compiled = torch.compile(model, backend="inductor")
                    _ = compiled(*trace_inputs)

                # inductor post_fusion_pass 已将 MLIR 写到 work_dir/inductor_linalg.mlir
                print(f"\n[Stage 2-8] MLIR pipeline (from inductor)")
                assert _run_mlir_pipeline_from_inductor(work_dir), \
                    "MLIR pipeline (inductor path) 失败"

            else:
                # ── 原有路径（不改动） ──
                # 1. dynamic_shapes + trace
                dim_pool = {}
                shapes_list = []
                has_dynamic = False
                for spec in specs:
                    dims = spec.dynamic_dims(dim_pool)
                    if dims:
                        has_dynamic = True
                        shapes_list.append(dims)
                    else:
                        shapes_list.append({})
                dynamic_shapes = tuple(shapes_list) if has_dynamic else None

                trace_inputs = [spec.make_sample() for spec in specs]

                # 2. torch → linalg
                print(f"\n[Stage 0] torch → linalg MLIR")
                mlir_text = torch_to_linalg(model, trace_inputs, dynamic_shapes)
                (work_dir / "step0_linalg.mlir").write_text(mlir_text)

                # 3. MLIR pipeline (stage 0a-8)
                print(f"\n[Stage 0b-8] MLIR pipeline")
                assert _run_mlir_pipeline(work_dir), "MLIR pipeline 失败"

            # ── 验证阶段（两条路径共用） ──
            actual_shapes = verify_shapes or {}
            verify_inputs = [spec.make_sample(actual_shapes) for spec in specs]

            print(f"\n[Stage 9] 生成 reference data")
            model_eval = model.eval()
            for i, tensor in enumerate(verify_inputs):
                np.save(work_dir / f"input_{i}.npy", tensor.numpy())
            with torch.no_grad():
                expected = model_eval(*verify_inputs)
            np.save(work_dir / "expected_0.npy", expected.numpy())

            print(f"\n[Stage 10] Autotuner (compile + tune + verify)")
            tiling_space = work_dir / "step8_kernel.tiling_space.json"
            shape_str = _build_shape_str(tiling_space, verify_inputs)
            assert _run_autotuner(work_dir, len(verify_inputs), shape_str), \
                "Autotuner 验证失败"

            print(f"\n  ✓ 测试通过: {test_name}")

        return wrapper

    # 支持 @torch_e2e_test 和 @torch_e2e_test(...) 两种用法
    if func is not None:
        return decorator(func)
    return decorator
```

- [ ] **Step 3: 更新测试用例**

```python
# test/inductor_e2e/test_broadcast_add_reduce.py

import torch
from torch.framework.pipeline import torch_e2e_test
from torch.framework.tensor_spec import TensorSpec


@torch_e2e_test(use_inductor=True)
def test_broadcast_add_reduce_with_pipeline():
    """端到端测试：inductor → MLIR → AscendC → NPU"""
    class BroadcastAddReduceModel(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)

    return BroadcastAddReduceModel(), [
        TensorSpec(("M",)), TensorSpec(("M", "N"))
    ]
```

- [ ] **Step 4: 运行端到端测试**

```bash
cd /home/gser/code/Ascend-MLIR
python -m pytest test/inductor_e2e/test_broadcast_add_reduce.py::test_broadcast_add_reduce_with_pipeline -v -s
```

Expected: PASS，生成 step8_kernel.cpp 和 step8_kernel.tiling_space.json

- [ ] **Step 5: Commit**

```bash
git add python/torch/framework/pipeline.py test/inductor_e2e/test_broadcast_add_reduce.py
git commit -m "feat(inductor): connect inductor MLIR to existing pipeline (stage 2+)"
```

---

### Task 7: Documentation and cleanup

**Files:**
- Modify: `docs/torch-e2e-pipeline.md`

- [ ] **Step 1: Update documentation with inductor path**

```markdown
# 在 docs/torch-e2e-pipeline.md 添加新章节

## Inductor → MLIR 路径

当 `use_inductor=True` 时，pipeline 使用 inductor 的融合能力：

1. 运行 inductor 获取融合后的 LoopIR
2. 通过 `_post_fusion_custom_pass` 拦截 FusedSchedulerNode
3. 转换为 linalg MLIR (`inductor_linalg.mlir`)
4. 从 stage 2 (tiling) 开始执行剩余 pipeline

这种方式复用了 inductor 的融合逻辑，支持：
- elementwise + reduction 融合
- reduction + elementwise 融合
- 更强大的 cost model
```

- [ ] **Step 2: Add README for inductor_backend**

```python
# python/inductor_backend/README.md

# Inductor Backend for Ascend-MLIR

这个模块将 PyTorch Inductor 的融合能力接入到 Ascend-MLIR pipeline。

## 架构

```
torch.compile → inductor (decompose + lower + fuse)
    → _post_fusion_custom_pass (intercept FusedSchedulerNode)
    → translate LoopIR → linalg MLIR
    → connect to existing Ascend-MLIR pipeline (stage 2+)
```

## 使用方式

```python
from inductor_backend import setup_inductor_backend, create_post_fusion_pass

# 注册 backend
setup_inductor_backend(
    post_fusion_pass=create_post_fusion_pass(output_dir="/tmp/inductor_mlir")
)

# 运行 inductor
with torch._inductor.config.patch("backend", "npu"):
    compiled = torch.compile(model, backend="inductor")
    result = compiled(*example_inputs)
```

## 当前限制 (POC)

- 仅支持简单的 elementwise + reduction 融合
- 索引表达式仅支持简单的 affine 模式
- 不支持复杂的 reduction 类型（仅 sum）
- 不支持动态 shape

## 未来方向

- 扩展 support reduction 类型（max, min, argmax 等）
- 支持 matmul epilogue 融合
- 改进 index 表达式 → affine_map 转换
- 接入 autotuner 搜索最优 tiling
```

- [ ] **Step 3: Final test run**

```bash
cd /home/gser/code/Ascend-MLIR
python -m pytest test/inductor_e2e/ -v
```

Expected: All tests pass

- [ ] **Step 4: Commit documentation**

```bash
git add docs/torch-e2e-pipeline.md python/inductor_backend/README.md
git commit -m "docs(inductor): add documentation for inductor-mlir path"
```