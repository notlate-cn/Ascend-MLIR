"""elementwise 算子组合测试"""
import torch
from framework import torch_e2e_test, TensorSpec


@torch_e2e_test
def test_add_mul():
    class Model(torch.nn.Module):
        def forward(self, a, b, c):
            return (a + b) * c
    return Model(), [TensorSpec(("M", "N"))] * 3


@torch_e2e_test
def test_relu_reduce():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.relu().sum(dim=1)
    return Model(), [TensorSpec(("M", "N"))]


@torch_e2e_test
def test_exp_broadcast_add():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return a.exp().unsqueeze(1) + b
    return Model(), [TensorSpec(("M",)),
                     TensorSpec(("M", "N"))]


@torch_e2e_test
def test_sigmoid_mul_reduce():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return (x.sigmoid() * y).sum(dim=1)
    return Model(), [TensorSpec(("M", "N")),
                     TensorSpec(("M", "N"))]
