"""broadcast + add + reduce_sum 端到端测试"""
import torch
from torch2linalg import torch_e2e_test


@torch_e2e_test
def test_broadcast_add_reduce():
    class Model(torch.nn.Module):
        def forward(self, a, b):
            return (a.unsqueeze(1) + b).sum(dim=1)
    return Model(), [torch.randn(32, dtype=torch.float16),
                     torch.randn(32, 64, dtype=torch.float16)]