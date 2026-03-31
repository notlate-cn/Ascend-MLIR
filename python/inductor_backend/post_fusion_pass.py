from typing import List
from torch._inductor.scheduler import BaseSchedulerNode


def create_post_fusion_pass(output_dir: str, verbose: bool = True):
    """
    创建一个 post-fusion pass，将 FusedSchedulerNode 转换为 linalg MLIR。
    output_dir: 输出 .mlir 文件的目录
    """
    def post_fusion_pass(nodes: List[BaseSchedulerNode]) -> List[BaseSchedulerNode]:
        for i, node in enumerate(nodes):
            print(f"Node {i}: {type(node).__name__}")
        return nodes
    return post_fusion_pass