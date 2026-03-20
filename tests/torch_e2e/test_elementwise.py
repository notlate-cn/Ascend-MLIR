"""elementwise 算子组合测试"""
import torch
from torch2linalg import torch_e2e_test


@torch_e2e_test
def test_add_mul():
    class Model(torch.nn.Module):
        def forward(self, a, b, c):
            return (a + b) * c
    return Model(), [torch.randn(32, 64, dtype=torch.float16)] * 3


@torch_e2e_test
def test_relu_reduce():
    class Model(torch.nn.Module):
        def forward(self, x):
            return x.relu().sum(dim=1)
    return Model(), [torch.randn(32, 64, dtype=torch.float16)]


@torch_e2e_test
def test_exp_broadcast_add():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return a.exp().unsqueeze(1) + b
    return Model(), [torch.randn(32, dtype=torch.float16),
                     torch.randn(32, 64, dtype=torch.float16)]


@torch_e2e_test
def test_sigmoid_mul_reduce():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return (x.sigmoid() * y).sum(dim=1)
    return Model(), [torch.randn(32, 64, dtype=torch.float16),
                     torch.randn(32, 64, dtype=torch.float16)]


@torch_e2e_test
def test_add():
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y
    return Model(), [torch.randn(32, 64, dtype=torch.float16),
                     torch.randn(32, 64, dtype=torch.float16)]