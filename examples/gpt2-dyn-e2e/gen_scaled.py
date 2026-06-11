"""Generate a REAL-GPT-2-dimension dynamic-seq transformer stack.

Same dynamic-seq path as gen.py, but at real GPT-2 small dims (n_embd=768,
12 heads) and configurable depth. Weights are large at these dims, so they are
lifted to npy INPUTS via params_to_buffers (inlining ~85M params as MLIR
constants makes a multi-100MB file g++ can't compile — the static-GPT-2 lesson).

Input is the hidden state [1,?,n_embd] (forward-from-hidden; the embedding /
lm_head are host-side / out of scope here). Causal mask is built IN-GRAPH via
torch.triu over the dynamic seq.

Emits model.mlir + input_0..input_{N-1}.npy (weights + hidden, in export
signature order) + hidden_input_index.txt + expected.npy.
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
    def __init__(self, n_embd, n_head, n_layer):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.lnf(x)


def params_to_buffers(mod):
    for name, p in list(mod.named_parameters(recurse=False)):
        delattr(mod, name)
        mod.register_buffer(name, p.data, persistent=False)
    for child in mod.children():
        params_to_buffers(child)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-embd", type=int, default=768)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--seq", type=int, default=48)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = Path(args.out_dir)

    torch.manual_seed(0)
    model = GPT(args.n_embd, args.n_head, args.n_layer).eval()
    hidden = torch.randn(1, args.seq, args.n_embd)
    with torch.no_grad():
        expected = model(hidden).detach()
    np.save(out / "expected.npy", expected.numpy())

    params_to_buffers(model)
    S = torch.export.Dim("S", min=2, max=2048)
    ep = torch.export.export(model, (hidden,), dynamic_shapes={"x": {1: S}})
    bufs = dict(model.named_buffers())
    n, hidden_idx = 0, None
    for s in ep.graph_signature.input_specs:
        kind = str(s.kind)
        if "PARAMETER" in kind:
            continue
        if "BUFFER" in kind:
            t = bufs[s.target]
        else:
            t = hidden
            hidden_idx = n
        np.save(out / f"input_{n}.npy", t.detach().cpu().numpy())
        n += 1
    (out / "hidden_input_index.txt").write_text(str(hidden_idx))

    mlir = torch_to_linalg(model, [hidden], {"x": {1: S}})
    (out / "model.mlir").write_text(mlir)
    print(f"gpt2-dyn scaled: n_embd={args.n_embd} n_head={args.n_head} "
          f"n_layer={args.n_layer} seq={args.seq} -> {n} inputs "
          f"(hidden at {hidden_idx})")


if __name__ == "__main__":
    main()
