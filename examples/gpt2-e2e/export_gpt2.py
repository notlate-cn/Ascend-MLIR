#!/usr/bin/env python3
"""Tiny GPT-2 (12 layers, hidden=64, vocab=1024) -> linalg MLIR + reference npy.

HuggingFace's GPT2 in transformers >= 5.5 uses a dynamic mask dispatch
(create_causal_mask + ALL_MASK_ATTENTION_FUNCTIONS) that torch.export can't
trace. So we ship a minimal from-scratch implementation that mirrors the GPT-2
architecture (pre-LN, multi-head attention with causal mask, GELU FFN with 4x
hidden, weight-tied lm_head) but uses only torch.export-friendly ops.

Goal: show the same V-V fusion pattern BERT demonstrated for GELU, scaled
across 12 layers, so the compiler emits ~12 fused GELU+bias kernels.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "torch"))
from torch2linalg import torch_to_linalg  # noqa: E402


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, seq_len):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.proj = nn.Linear(n_embd, n_embd, bias=True)
        # Static causal mask as a buffer: lower-triangular ones, -inf above
        # diagonal (additive mask).  Registering as a buffer means
        # torch.export lifts it to a constant.
        mask = torch.full((seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, S, C = x.shape
        H, D = self.n_head, self.head_dim
        qkv = self.qkv(x)                            # [B, S, 3C]
        q, k, v = qkv.split(C, dim=-1)               # each [B, S, C]
        q = q.view(B, S, H, D).transpose(1, 2)       # [B, H, S, D]
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale # [B, H, S, S]
        att = att + self.mask                        # additive causal mask
        att = torch.softmax(att, dim=-1)
        y = att @ v                                  # [B, H, S, D]
        y = y.transpose(1, 2).contiguous().view(B, S, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=True)

    def forward(self, x):
        return self.proj(nn.functional.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, seq_len)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab, n_embd, n_head, n_layer, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, seq_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab, bias=False)
        # Register position indices as a buffer so torch.export lifts them
        # as a constant rather than synthesizing a dynamic arange.
        self.register_buffer(
            "pos_ids",
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )

    def forward(self, input_ids):
        x = self.tok_emb(input_ids) + self.pos_emb(self.pos_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    model = MiniGPT(
        vocab=args.vocab, n_embd=args.n_embd, n_head=args.n_head,
        n_layer=args.n_layer, seq_len=args.seq,
    ).eval().to(dtype)
    input_ids = torch.randint(0, args.vocab, (1, args.seq), dtype=torch.long)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        expected = model(input_ids)
    np.save(outdir / "expected_0.npy", expected.numpy())

    n_inputs = 0
    for nm, buf in model.named_buffers():
        np.save(outdir / f"input_{n_inputs}.npy", buf.detach().cpu().numpy())
        n_inputs += 1
    np.save(outdir / f"input_{n_inputs}.npy", input_ids.numpy())
    n_inputs += 1

    mlir_text = torch_to_linalg(model, [input_ids], None)
    (outdir / "step0_linalg.mlir").write_text(mlir_text)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params}  ({n_params * 4 / 1e6:.1f} MB fp32)")
    print(f"input_ids={tuple(input_ids.shape)} expected={tuple(expected.shape)} dtype={args.dtype}")
    print(f"wrote: {outdir}/step0_linalg.mlir, input_0..{n_inputs - 1}.npy, expected_0.npy")


if __name__ == "__main__":
    main()
