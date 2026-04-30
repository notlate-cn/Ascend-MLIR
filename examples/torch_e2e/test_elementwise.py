"""elementwise, broadcast, reduce 算子组合测试"""
import torch
from framework import torch_e2e_test, TensorSpec


@torch_e2e_test
def test_add_mul():
    class Model(torch.nn.Module):
        def forward(self, a, b, c):
            return (a + b) * c

    return Model(), [TensorSpec(("M", "N"))] * 3


@torch_e2e_test
def test_add_reduce():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a + b).sum(dim=1)

    return Model(), [TensorSpec(("M", "N"))] * 2


@torch_e2e_test(verify_shapes={"M": 128, "N": 16})
def test_broadcast_add_reduce():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)

    return Model(), [TensorSpec(("M",)),
                     TensorSpec(("M", "N"))]