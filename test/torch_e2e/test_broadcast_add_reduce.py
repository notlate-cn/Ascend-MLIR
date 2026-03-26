"""broadcast + add + reduce_sum 端到端测试"""
import torch
from framework import torch_e2e_test, TensorSpec


@torch_e2e_test(verify_shapes={"M": 128, "N": 16})
def test_broadcast_add_reduce():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)
    return Model(), [TensorSpec(("M",)),
                     TensorSpec(("M", "N"))]