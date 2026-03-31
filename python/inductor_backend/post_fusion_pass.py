"""
Post-fusion pass: 从 inductor 融合后的 LoopIR 节点提取信息，生成 linalg.generic MLIR。

落盘的 inductor 中间结果：
  {output_dir}/fx_graph.py         — inductor 接收到的 FX graph
  {output_dir}/loop_ir.json        — 融合后的 LoopIR 节点结构
  {output_dir}/inductor_linalg.mlir — 生成的 linalg MLIR
"""

import json
import os
import sympy
from typing import List

from torch._inductor.scheduler import BaseSchedulerNode, FusedSchedulerNode
from torch._inductor.ir import Pointwise, Reduction


# ================================================================
# FX graph 捕获
# ================================================================

def _install_fx_graph_capturer(output_dir: str):
    """Patch compile_fx_inner 以落盘 FX graph。"""
    try:
        import torch._inductor.compile_fx as cfx
        orig_inner = cfx.compile_fx_inner

        def patched_inner(*args, **kwargs):
            # 第一个位置参数是 gm (GraphModule)
            gm = args[0] if args else kwargs.get('gm')
            if gm is not None:
                try:
                    fx_path = os.path.join(output_dir, "fx_graph.py")
                    with open(fx_path, "w") as f:
                        # gm.code 是 Python 代码字符串；fallback 到 str(gm.graph)
                        text = getattr(gm, 'code', None) or str(gm.graph)
                        f.write(text)
                except Exception as e:
                    print(f"[MLIRBackend] FX graph write failed: {e}")
            return orig_inner(*args, **kwargs)

        cfx.compile_fx_inner = patched_inner
    except Exception as e:
        print(f"[MLIRBackend] FX capturer install failed: {e}")


# ================================================================
# LoopIR 提取
# ================================================================

def _extract_loops_from_node(node: BaseSchedulerNode):
    """从 SchedulerNode / FusedSchedulerNode 提取 LoopIR 信息列表。"""
    loops_info = []

    if isinstance(node, FusedSchedulerNode):
        for sub in node.get_nodes():
            buf = sub.node
            data = getattr(buf, 'data', None)
            if data is None:
                continue
            loops_info.append({
                'node_type': type(data).__name__,
                'data': data,
                'scheduler_node': sub,
                'ranges': data.ranges,
                'reduction_ranges': getattr(data, 'reduction_ranges', []),
                'reduction_type': getattr(data, 'reduction_type', None),
                'inner_fn': data.inner_fn,
                'dtype': data.dtype,
            })
    else:
        buf = getattr(node, 'node', None)
        if buf is not None:
            data = getattr(buf, 'data', None)
            if data is not None and isinstance(data, (Pointwise, Reduction)):
                loops_info.append({
                    'node_type': type(data).__name__,
                    'data': data,
                    'scheduler_node': node,
                    'ranges': data.ranges,
                    'reduction_ranges': getattr(data, 'reduction_ranges', []),
                    'reduction_type': getattr(data, 'reduction_type', None),
                    'inner_fn': data.inner_fn,
                    'dtype': data.dtype,
                })

    return loops_info


# ================================================================
# Index 分析：sympy 表达式 → affine_map + shape
# ================================================================

def _infer_tensor_info(index_expr, index_vars, rindex_vars, ranges, reduction_ranges):
    """
    从 load 的 sympy index 推断 tensor shape 和 affine_map。

    Returns:
        shape_dims: list[int]   — 各维度大小
        affine_map: str         — 如 "(d0, d1) -> (d0)"
    """
    all_vars = index_vars + rindex_vars
    all_sizes = list(ranges) + list(reduction_ranges)

    try:
        free_syms = index_expr.free_symbols
    except AttributeError:
        free_syms = set()

    used = [(i, v) for i, v in enumerate(all_vars) if v in free_syms]

    num_total = len(all_vars)
    all_dim_str = ', '.join(f'd{k}' for k in range(num_total))

    if not used:
        return [], f'({all_dim_str}) -> ()'

    shape_dims = [int(all_sizes[i]) for i, _ in used]
    used_dim_str = ', '.join(str(v) for _, v in used)
    affine_map = f'({all_dim_str}) -> ({used_dim_str})'

    return shape_dims, affine_map


# ================================================================
# Post-fusion pass factory
# ================================================================

def create_post_fusion_pass(output_dir: str, verbose: bool = True,
                            capture_fx_graph: bool = True):
    """
    创建 post-fusion pass（传给 setup_inductor_backend）。

    Args:
        output_dir:       落盘目录
        verbose:          是否打印调试信息
        capture_fx_graph: 是否落盘 FX graph
    """
    os.makedirs(output_dir, exist_ok=True)

    if capture_fx_graph:
        _install_fx_graph_capturer(output_dir)

    def post_fusion_pass(nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
        from torch._inductor import virtualized
        from .mlir_emitter import MLIREmitter
        from .ops_handler import MLIROpsHandler

        if verbose:
            print(f"\n=== Post-Fusion Pass (output_dir={output_dir}) ===")
            print(f"Total nodes after fusion: {len(nodes)}")

        emitter = MLIREmitter()
        emitter.emit_module_header()

        loop_ir_records = []

        for i, node in enumerate(nodes):
            if verbose:
                print(f"Node {i}: {type(node).__name__}")

            loops = _extract_loops_from_node(node)
            for j, loop in enumerate(loops):
                ranges = loop['ranges']
                reduction_ranges = loop['reduction_ranges']
                num_parallel = len(ranges)
                num_reduction = len(reduction_ranges)
                iterator_types = (['parallel'] * num_parallel
                                  + ['reduction'] * num_reduction)

                if verbose:
                    print(f"  Loop[{j}]: {loop['node_type']}, "
                          f"ranges={ranges}, rranges={reduction_ranges}, "
                          f"reduction_type={loop['reduction_type']}")

                index_vars = [sympy.Symbol(f'd{k}') for k in range(num_parallel)]
                rindex_vars = [sympy.Symbol(f'd{num_parallel + k}')
                               for k in range(num_reduction)]

                # ── inner_fn trace ──
                handler = MLIROpsHandler()
                virtualized.V.set_ops_handler(handler)
                fn_result = None
                try:
                    if loop['node_type'] == 'Reduction':
                        fn_result = loop['inner_fn'](index_vars, rindex_vars)
                    else:
                        fn_result = loop['inner_fn'](index_vars)
                except Exception as e:
                    if verbose:
                        print(f"    inner_fn failed: {e}")
                    continue
                finally:
                    virtualized.V.set_ops_handler(None)

                # OpsValue unwrap（inductor wraps handler return values）
                if hasattr(fn_result, 'value'):
                    fn_result = fn_result.value

                # 若没有 store op，用 inner_fn 返回值合成
                has_store = any(op['type'] in ('store', 'store_reduction')
                                for op in handler.operations)
                if not has_store and fn_result is not None:
                    out_name = loop['scheduler_node'].get_name()
                    handler.operations.append({
                        'type': 'store',
                        'name': out_name,
                        'index': index_vars,
                        'value': fn_result,
                    })

                # ── 分析 load：推断 affine_map 和 shape ──
                loads = [op for op in handler.operations if op['type'] == 'load']
                seen_names: list = []
                first_load_by_name: dict = {}
                for op in loads:
                    if op['name'] not in seen_names:
                        seen_names.append(op['name'])
                        first_load_by_name[op['name']] = op

                if not seen_names:
                    if verbose:
                        print(f"    no input tensors, skipping")
                    continue

                input_specs = []
                for name in seen_names:
                    idx_expr = first_load_by_name[name]['index']
                    shape_dims, amap = _infer_tensor_info(
                        idx_expr, index_vars, rindex_vars, ranges, reduction_ranges)
                    input_specs.append((name, shape_dims, amap))

                # 输出 spec
                out_store = next(
                    (op for op in reversed(handler.operations)
                     if op['type'] in ('store', 'store_reduction')),
                    None)
                if out_store is None:
                    if verbose:
                        print(f"    no output store, skipping")
                    continue

                out_name = out_store['name']
                out_shape = [int(r) for r in ranges]
                num_total_dims = num_parallel + num_reduction
                all_dim_str = ', '.join(f'd{k}' for k in range(num_total_dims))
                par_dim_str = ', '.join(f'd{k}' for k in range(num_parallel))
                out_amap = (f'({all_dim_str}) -> ({par_dim_str})'
                            if num_reduction > 0
                            else f'({all_dim_str}) -> ({all_dim_str})')
                output_spec = (out_name, out_shape, out_amap)

                # ── load SSA ID → block arg 映射 ──
                load_id_to_arg: dict = {}
                arg_idx = 0
                for name in seen_names:
                    for op in loads:
                        if op['name'] == name and op['id'] not in load_id_to_arg:
                            load_id_to_arg[op['id']] = f'%arg{arg_idx}'
                    arg_idx += 1
                acc_arg = f'%arg{arg_idx}'  # accumulator（输出 block arg）

                dtype_str = emitter._mlir_dtype(loop['dtype'])

                if verbose:
                    print(f"    inputs={[(n, s) for n,s,_ in input_specs]}, "
                          f"output={out_name}{out_shape}, "
                          f"ops={len(handler.operations)}")

                loop_ir_records.append({
                    'node': i,
                    'loop': j,
                    'node_type': loop['node_type'],
                    'ranges': [str(r) for r in ranges],
                    'reduction_ranges': [str(r) for r in reduction_ranges],
                    'reduction_type': str(loop['reduction_type']),
                    'iterator_types': iterator_types,
                    'input_specs': [(n, s, m) for n, s, m in input_specs],
                    'output_spec': list(output_spec),
                    'ops': [{k: str(v) for k, v in op.items()}
                            for op in handler.operations],
                    'load_id_to_arg': load_id_to_arg,
                })

                emitter.emit_linalg_generic_kernel(
                    func_name=f"kernel_{i}_{j}",
                    input_specs=input_specs,
                    output_spec=output_spec,
                    iterator_types=iterator_types,
                    body_ops=handler.operations,
                    load_id_to_arg=load_id_to_arg,
                    acc_arg=acc_arg,
                    reduction_type=loop['reduction_type'],
                    elem_dtype=dtype_str,
                )

        emitter.emit_module_footer()

        mlir_path = os.path.join(output_dir, "inductor_linalg.mlir")
        with open(mlir_path, "w") as f:
            f.write(emitter.get_text())

        json_path = os.path.join(output_dir, "loop_ir.json")
        with open(json_path, "w") as f:
            json.dump(loop_ir_records, f, indent=2, default=str)

        if verbose:
            print(f"MLIR  -> {mlir_path}")
            print(f"LoopIR -> {json_path}")
            print(f"=== End Post-Fusion Pass ===\n")

        return nodes

    return post_fusion_pass