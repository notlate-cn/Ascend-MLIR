from torch._inductor.codegen.simd import SIMDScheduling
from torch._inductor.codegen.common import register_backend_for_device


class MLIRScheduling(SIMDScheduling):
    """
    最小 backend，只为让 inductor 的融合逻辑跑起来。
    can_fuse_vertical/horizontal 完全继承 SIMDScheduling。
    codegen 相关方法空实现——我们在 post_fusion_pass 中自己处理。
    """

    def codegen_node(self, node):
        pass

    def codegen_template(self, *args, **kwargs):
        pass

    def codegen_node_schedule(self, *args, **kwargs):
        pass

    def define_kernel(self, src_code, node_schedule, kernel):
        pass

    def codegen_sync(self):
        pass

    def benchmark_fused_nodes(self, nodes):
        return (0.0, "npu")

    def flush(self):
        pass

    def ready_to_flush(self):
        return False


def setup_inductor_backend(post_fusion_pass=None):
    """Stub — implemented in Task 2."""
    pass