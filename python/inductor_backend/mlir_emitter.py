import torch


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
        indexing_maps: affine_map 表达式文本列表（不含 affine_map<...> 包装）
        iterator_types: ['parallel', 'parallel', 'reduction', ...]
        body_ops: ops_handler 产生的操作列表
        """
        for n, t in inputs:
            self.emit(f'%{n} = tensor.empty() : tensor<{t}>')
        for n, t in outputs:
            self.emit(f'%{n} = tensor.empty() : tensor<{t}>')

        in_names = ', '.join([f'%{n}' for n, _ in inputs])
        out_names = ', '.join([f'%{n}' for n, _ in outputs])
        in_types = ', '.join([f'tensor<{t}>' for _, t in inputs])
        out_types = ', '.join([f'tensor<{t}>' for _, t in outputs])
        iter_str = ', '.join([f'"{it}"' for it in iterator_types])
        maps_str = ', '.join([f'affine_map<{m}>' for m in indexing_maps])

        self.emit(f'%{name} = linalg.generic {{')
        self.emit(f'  indexing_maps = [{maps_str}],')
        self.emit(f'  iterator_types = [{iter_str}]}}')
        self.emit(f'  ins({in_names} : {in_types}) outs({out_names} : {out_types}) {{')

        num_args = len(inputs) + len(outputs)
        block_args = ', '.join([f'%arg{i}: {dtype}' for i in range(num_args)])
        self.emit(f'  ^bb0({block_args}):')

        self.indent_level += 1
        for op in body_ops:
            self._emit_body_op(op, dtype)
        self.indent_level -= 1

        self.emit('  }')

    def _emit_body_op(self, op: dict, elem_dtype: str = "f32"):
        op_type = op['type']
        if op_type == 'constant':
            self.emit(f"{op['id']} = arith.constant {op['value']} : {self._mlir_dtype(op['dtype'])}")
        elif op_type == 'load':
            pass  # load maps to block args, no emit needed
        elif op_type in ('store', 'store_reduction'):
            self.emit(f"linalg.yield {op['value']} : {elem_dtype}")
        elif op_type == 'add':
            self.emit(f"{op['id']} = arith.addf {op['lhs']}, {op['rhs']} : {elem_dtype}")
        elif op_type == 'mul':
            self.emit(f"{op['id']} = arith.mulf {op['lhs']}, {op['rhs']} : {elem_dtype}")
        elif op_type == 'reduction':
            rt = op['reduction_type']
            if rt == 'sum':
                self.emit(f"{op['id']} = arith.addf {op['value']}, %acc : {elem_dtype}")
            elif rt == 'max':
                self.emit(f"{op['id']} = arith.maximumf {op['value']}, %acc : {elem_dtype}")
        elif op_type == 'index_expr':
            pass  # POC: not emitted

    def _mlir_dtype(self, torch_dtype) -> str:
        if torch_dtype == torch.float32:
            return "f32"
        elif torch_dtype == torch.float16:
            return "f16"
        elif torch_dtype == torch.int32:
            return "i32"
        elif torch_dtype == torch.int64:
            return "i64"
        return "f32"

    def get_text(self) -> str:
        return '\n'.join(self.lines)