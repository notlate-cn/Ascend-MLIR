from typing import List
from torch._inductor.scheduler import BaseSchedulerNode, FusedSchedulerNode
from torch._inductor.ir import ComputedBuffer, Pointwise, Reduction


def _extract_loops_from_node(node: BaseSchedulerNode):
    """
    从 SchedulerNode 提取 LoopIR 信息。

    Returns:
        List of dicts with keys: node_type, node, ranges, reduction_ranges, inner_fn, dtype, device
    """
    loops_info = []

    if isinstance(node, FusedSchedulerNode):
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
        return nodes

    return post_fusion_pass