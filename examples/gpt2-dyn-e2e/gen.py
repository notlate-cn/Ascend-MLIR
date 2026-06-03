"""Generate the dynamic-seq mini-GPT model + inputs for gpt2-dyn-e2e.

A small from-scratch GPT (2 blocks, n_embd=64, 4 heads) with the causal mask
built IN-GRAPH via torch.triu (so the exported MLIR has a dynamic `?` seq dim,
not a baked [S,S] constant). Exercises the full dynamic-seq path: attention
(dynamic Q/K/V + value-less causal-mask sentinel), LayerNorm (dynamic broadcast
raised by --lower-broadcast-extract), MLP/erf-GELU, residuals.

Writes model.mlir (dynamic seq) + x.npy + expected.npy for --seq. The default
seq (48) is deliberately a value the model was never "compiled for" — a pass
proves the seq extent is resolved at runtime.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "torch"))
from torch2linalg import torch_to_linalg  # noqa: E402


class Attn(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.proj = nn.Linear(n_embd, n_embd, bias=True)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, S, C = x.shape
        H, D = self.n_head, self.head_dim
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        # In-graph dynamic causal mask (triu over the runtime seq length).
        att = att + torch.triu(torch.full((S, S), float("-inf"), dtype=x.dtype), 1)
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, S, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=True)

    def forward(self, x):
        return self.proj(nn.functional.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = Attn(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, n_embd=64, n_head=4, n_layer=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.lnf(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq", type=int, default=48)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(0)
    model = GPT().eval()

    S = torch.export.Dim("S", min=2, max=256)
    sample = torch.randn(1, 16, 64)
    mlir = torch_to_linalg(model, [sample], {"x": {1: S}})
    open(os.path.join(args.out_dir, "model.mlir"), "w").write(mlir)

    x = torch.randn(1, args.seq, 64)
    np.save(os.path.join(args.out_dir, "x.npy"), x.numpy())
    np.save(os.path.join(args.out_dir, "expected.npy"),
            model(x).detach().numpy())
    print(f"gpt2-dyn: model.mlir + x/expected (seq={args.seq})")


if __name__ == "__main__":
    main()
