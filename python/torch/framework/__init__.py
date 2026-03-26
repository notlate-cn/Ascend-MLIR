"""
framework - Torch → NPU 端到端测试框架

用法：
    from framework import torch_e2e_test, TensorSpec

    @torch_e2e_test
    def test_my_model():
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.relu().sum(dim=1)
        return Model(), [TensorSpec(("M", "N"), torch.float16)]
"""

from .tensor_spec import TensorSpec
from .pipeline import torch_e2e_test

__all__ = ["torch_e2e_test", "TensorSpec"]