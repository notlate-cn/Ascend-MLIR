"""
MLIREmitter: 生成 linalg MLIR 文本。

核心方法：
  emit_linalg_generic_kernel() — 生成完整 func.func + linalg.generic
"""

import torch


class MLIREmitter:
    """生成 linalg MLIR 文本。"""

    def __init__(self):
        self.lines = []
        self.indent_level = 0

    # ── 基础 emit ────────────────────────────────────────────────

    def emit(self, text: str):
        indent = "  " * self.indent_level
        self.lines.append(f"{indent}{text}")

    def emit_module_header(self):
        self.emit('module {')
        self.indent_level += 1

    def emit_module_footer(self):
        self.indent_level -= 1
        self.emit('}')

    def get_text(self) -> str:
        return '\n'.join(self.lines)

    # ── 类型辅助 ─────────────────────────────────────────────────

    def _mlir_dtype(self, torch_dtype) -> str:
        """PyTorch dtype → MLIR 类型字符串。"""
        dtype_map = {
            torch.float32: "f32",
            torch.float16: "f16",
            torch.bfloat16: "bf16",
            torch.float64: "f64",
            torch.int32: "i32",
            torch.int64: "i64",
            torch.int8: "i8",
            torch.bool: "i1",
        }
        return dtype_map.get(torch_dtype, "f32")

    @staticmethod
    def _tensor_type(shape_dims: list, dtype_str: str) -> str:
        """生成 MLIR tensor 类型，如 '128x16xf32'。"""
        if not shape_dims:
            return dtype_str  # scalar
        return 'x'.join(str(d) for d in shape_dims) + 'x' + dtype_str

    # ── 核心：完整 kernel 生成 ───────────────────────────────────

    def emit_linalg_generic_kernel(
        self,
        func_name: str,
        input_specs: list,
        output_spec: tuple,
        iterator_types: list,
        body_ops: list,
        load_id_to_arg: dict,
        acc_arg: str,
        reduction_type: str | None,
        elem_dtype: str = "f32",
    ):
        """
        生成一个完整的 func.func 包裹 linalg.generic kernel。

        Args:
            func_name:      函数名，如 "kernel_0_0"
            input_specs:    [(buf_name, shape_dims, affine_map_str), ...]
            output_spec:    (buf_name, shape_dims, affine_map_str)
            iterator_types: ['parallel', 'parallel', 'reduction', ...]
            body_ops:       MLIROpsHandler 产生的操作列表
            load_id_to_arg: {load_ssa_id: '%argN'} 映射，用于替换 body 中的引用
            acc_arg:        输出 block 参数名，如 '%arg2'（accumulator）
            reduction_type: 'sum', 'max', None（pointwise）
            elem_dtype:     元素类型字符串，如 'f32'
        """
        out_name, out_shape, out_amap = output_spec

        # ── func.func 签名 ──
        arg_decls = []
        for name, shape_dims, _ in input_specs:
            ttype = self._tensor_type(shape_dims, elem_dtype)
            arg_decls.append(f'%{name}: tensor<{ttype}>')
        ret_type = self._tensor_type(out_shape, elem_dtype)
        args_str = ', '.join(arg_decls)
        self.emit(f'func.func @{func_name}({args_str}) -> tensor<{ret_type}> {{')
        self.indent_level += 1

        # ── 初始化输出 tensor ──
        self.emit(f'%{out_name}_init = tensor.empty() : tensor<{ret_type}>')

        # ── linalg.generic 头部 ──
        in_names_str = ', '.join(f'%{n}' for n, _, _ in input_specs)
        in_types_str = ', '.join(
            f'tensor<{self._tensor_type(s, elem_dtype)}>'
            for _, s, _ in input_specs
        )
        out_init_str = f'%{out_name}_init'
        out_type_str = f'tensor<{ret_type}>'

        iter_str = ', '.join(f'"{it}"' for it in iterator_types)
        all_amaps = [amap for _, _, amap in input_specs] + [out_amap]
        maps_str = ', '.join(f'affine_map<{m}>' for m in all_amaps)

        # linalg.generic: {attr_dict} ins(...) outs(...) { body } -> result_type
        # (全部属性和 ins/outs 在同一逻辑行，-> type 在 body 后)
        self.emit(
            f'%{out_name} = linalg.generic '
            f'{{indexing_maps = [{maps_str}], iterator_types = [{iter_str}]}} '
            f'ins({in_names_str} : {in_types_str}) '
            f'outs({out_init_str} : {out_type_str}) {{'
        )

        # ── ^bb0 block 参数 ──
        num_inputs = len(input_specs)
        block_args = ', '.join(
            f'%arg{k}: {elem_dtype}' for k in range(num_inputs + 1)
        )
        self.emit(f'^bb0({block_args}):')
        self.indent_level += 1

        # ── body 操作 ──
        inner_result = self._emit_body(body_ops, load_id_to_arg, elem_dtype)

        # ── reduction 累加 or pointwise yield ──
        if reduction_type is not None and inner_result is not None:
            result_ref = load_id_to_arg.get(inner_result, inner_result)
            self._emit_reduction_combine(result_ref, acc_arg, reduction_type,
                                         elem_dtype)
        else:
            if inner_result is not None:
                result_ref = load_id_to_arg.get(inner_result, inner_result)
                self.emit(f'linalg.yield {result_ref} : {elem_dtype}')

        self.indent_level -= 1
        self.emit(f'}} -> {out_type_str}')

        # ── return ──
        self.emit(f'return %{out_name} : {out_type_str}')
        self.indent_level -= 1
        self.emit('}')
        self.emit('')  # 空行分隔

    def _subst(self, ssa_id: str, load_id_to_arg: dict) -> str:
        """将 load SSA ID 替换为对应 block arg 名，其余原样返回。"""
        return load_id_to_arg.get(ssa_id, ssa_id)

    def _emit_body(self, body_ops: list, load_id_to_arg: dict,
                   elem_dtype: str) -> str | None:
        """
        遍历 body 操作并 emit（跳过 load 和 store），返回最终结果 SSA ID。
        """
        last_result = None
        for op in body_ops:
            op_type = op['type']

            if op_type == 'load':
                # load 已经映射到 block arg，无需 emit
                last_result = op['id']

            elif op_type in ('store', 'store_reduction'):
                # store 的 value 就是 inner_fn 的最终结果
                last_result = op.get('value')

            elif op_type == 'constant':
                mlir_t = self._mlir_dtype(op['dtype']) if hasattr(op.get('dtype'), 'dtype') else elem_dtype
                self.emit(f"{op['id']} = arith.constant {op['value']} : {mlir_t}")
                last_result = op['id']

            elif op_type == 'add':
                lhs = self._subst(op['lhs'], load_id_to_arg)
                rhs = self._subst(op['rhs'], load_id_to_arg)
                self.emit(f"{op['id']} = arith.addf {lhs}, {rhs} : {elem_dtype}")
                last_result = op['id']

            elif op_type == 'mul':
                lhs = self._subst(op['lhs'], load_id_to_arg)
                rhs = self._subst(op['rhs'], load_id_to_arg)
                self.emit(f"{op['id']} = arith.mulf {lhs}, {rhs} : {elem_dtype}")
                last_result = op['id']

            elif op_type == 'reduction':
                # inductor 的 reduction combiner（Pointwise unroll 路径）
                # 在 linalg.generic 里用 acc_arg 处理，此处跳过
                last_result = op['id']

            elif op_type == 'where':
                cond = self._subst(op['cond'], load_id_to_arg)
                a = self._subst(op['a'], load_id_to_arg)
                b = self._subst(op['b'], load_id_to_arg)
                self.emit(f"{op['id']} = arith.select {cond}, {a}, {b} : {elem_dtype}")
                last_result = op['id']

            # index_expr, indirect_indexing: 跳过（POC 不处理）

        return last_result

    def _emit_reduction_combine(self, new_val: str, acc_arg: str,
                                 reduction_type: str, elem_dtype: str):
        """
        生成 reduction 累加指令并 yield。

        new_val:        inner_fn 结果（待累加值）
        acc_arg:        accumulator block arg，如 '%arg2'
        reduction_type: 'sum', 'max', 'min', etc.
        """
        result_id = '%_reduce_result'
        if reduction_type == 'sum':
            self.emit(f"{result_id} = arith.addf {new_val}, {acc_arg} : {elem_dtype}")
        elif reduction_type in ('max', 'amax'):
            self.emit(f"{result_id} = arith.maximumf {new_val}, {acc_arg} : {elem_dtype}")
        elif reduction_type in ('min', 'amin'):
            self.emit(f"{result_id} = arith.minimumf {new_val}, {acc_arg} : {elem_dtype}")
        else:
            # 默认 sum
            self.emit(f"{result_id} = arith.addf {new_val}, {acc_arg} : {elem_dtype}")
        self.emit(f"linalg.yield {result_id} : {elem_dtype}")
