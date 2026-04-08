import os
import sympy
from typing import List
from torch._inductor.scheduler import BaseSchedulerNode, FusedSchedulerNode
from torch._inductor.ir import Pointwise, Reduction


def _extract_loops_from_node(node: BaseSchedulerNode):
    """从 SchedulerNode 提取 LoopIR 信息。"""
    loops_info = []

    if isinstance(node, FusedSchedulerNode):
        for sub in node.get_nodes():
            buf = sub.node
            data = getattr(buf, 'data', None)
            if data is None:
                continue
            loops_info.append({
                'node_type': type(data).__name__,
                'node': data,
                'scheduler_node': sub,
                'ranges': data.ranges,
                'reduction_ranges': getattr(data, 'reduction_ranges', []),

                'inner_fn': data.inner_fn,
                'dtype': data.dtype,
                'device': data.get_device()
            })
    else:
        buf = node.node
        if hasattr(buf, 'data'):
            data = buf.data
            if isinstance(data, (Pointwise, Reduction)):
                loops_info.append({
                    'node_type': type(data).__name__,
                    'node': data,
                    'scheduler_node': node,
                    'ranges': data.ranges,
                    'reduction_ranges': getattr(data, 'reduction_ranges', []),
                    'inner_fn': data.inner_fn,
                    'dtype': data.dtype,
                    'device': data.get_device()
                })

    return loops_info


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
    raise ValueError("No store/store_reduction found in operations — inner_fn trace 可能不完整")


def create_post_fusion_pass(output_dir: str, verbose: bool = True):
    """
    创建一个 post-fusion pass，将 FusedSchedulerNode 转换为 linalg MLIR。
    """
    def post_fusion_pass(nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
        from torch._inductor import virtualized
        from .mlir_emitter import MLIREmitter
        from .ops_handler import MLIROpsHandler

        if verbose:
            print(f"\n=== Post-Fusion Pass (output_dir={output_dir}) ===")

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

                handler = MLIROpsHandler()

                # ranges = parallel 维度, reduction_ranges = reduction 维度（独立列表）
                ranges = loop['ranges']
                reduction_ranges = loop['reduction_ranges']
                num_parallel = len(ranges)
                num_reduction = len(reduction_ranges)
                total_dims = num_parallel + num_reduction

                iterator_types = ['parallel'] * num_parallel + ['reduction'] * num_reduction

                # inner_fn 调用约定：
                #   Pointwise: inner_fn(index)          — index 长度 = len(ranges)
                #   Reduction: inner_fn(index, rindex)  — 多一个 reduction 索引
                index_vars = [sympy.Symbol(f"d{k}") for k in range(num_parallel)]
                rindex_vars = [sympy.Symbol(f"d{num_parallel + k}") for k in range(num_reduction)]

                # Set the ops handler so inner_fn calls are intercepted
                virtualized.V.set_ops_handler(handler)
                try:
                    if loop['node_type'] == 'Reduction':
                        fn_result = loop['inner_fn'](index_vars, rindex_vars)
                    else:
                        fn_result = loop['inner_fn'](index_vars)
                except Exception as e:
                    if verbose:
                        print(f"    ⚠ Skipping loop[{j}]: inner_fn failed: {e}")
                    virtualized.V.set_ops_handler(None)
                    continue
                finally:
                    virtualized.V.set_ops_handler(None)

                # inner_fn may return the result without emitting a store.
                # In that case, synthesize a store op using the scheduler node name.
                has_store = any(op['type'] in ('store', 'store_reduction')
                                for op in handler.operations)
                if not has_store and fn_result is not None:
                    output_buf_name = loop['scheduler_node'].get_name()
                    handler.operations.append({
                        'type': 'store',
                        'name': output_buf_name,
                        'index': index_vars,
                        'value': fn_result
                    })

                dtype = emitter._mlir_dtype(loop['dtype'])

                input_names = _extract_input_names(handler.operations)
                try:
                    output_name = _extract_output_name(handler.operations)
                except ValueError as e:
                    if verbose:
                        print(f"    ⚠ Skipping loop[{j}]: {e}")
                    continue

                if not input_names:
                    if verbose:
                        print(f"    ⚠ Skipping loop[{j}]: no input tensors found")
                    continue

                # POC indexing maps: 输入用 identity，输出仅含 parallel 维度
                dim_vars = ', '.join([f'd{k}' for k in range(total_dims)])
                all_dims_map = f'({dim_vars}) -> ({dim_vars})'
                parallel_dims = ', '.join([f'd{k}' for k in range(num_parallel)])
                out_map = (f'({dim_vars}) -> ({parallel_dims})'
                           if num_reduction > 0 else all_dims_map)
                indexing_maps = [all_dims_map] * len(input_names) + [out_map]

                if verbose:
                    print(f"    inputs={input_names}, output={output_name}, "
                          f"ops={len(handler.operations)}")

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

        os.makedirs(output_dir, exist_ok=True)
        mlir_path = os.path.join(output_dir, "inductor_linalg.mlir")
        with open(mlir_path, "w") as f:
            f.write(emitter.get_text())

        if verbose:
            print(f"✓ MLIR written to {mlir_path}")
            print(f"Total nodes: {len(nodes)}, Total LoopIR: {len(all_loops)}")
            print("=== End Post-Fusion Pass ===\n")

        return nodes

    return post_fusion_pass