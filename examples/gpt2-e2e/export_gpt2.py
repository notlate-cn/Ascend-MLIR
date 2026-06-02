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
        h = self.fc(x)
        # gelu_new (tanh approximation, matches HF). Written out with explicit
        # multiplies for the cube: torch's gelu(approximate="tanh") lowers x**3
        # to math.fpowi, which the AscendC codegen cannot lower; h*h*h becomes
        # math.mulf (supported) and is numerically identical here.
        g = 0.5 * h * (1.0 + torch.tanh(
            0.7978845608028654 * (h + 0.044715 * h * h * h)))
        return self.proj(g)


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

    def embed(self, input_ids):
        """Host-side embedding: returns hidden_states [B, S, C]."""
        return self.tok_emb(input_ids) + self.pos_emb(self.pos_ids)

    def forward_from_hidden(self, hidden):
        x = hidden
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def forward(self, input_ids):
        return self.forward_from_hidden(self.embed(input_ids))


class FromHiddenWrapper(nn.Module):
    """Exposes MiniGPT.forward_from_hidden as the module forward for export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden):
        return self.model.forward_from_hidden(hidden)


def params_to_buffers(mod):
    """Recursively convert nn.Parameters to non-persistent buffers.

    torch-mlir inlines PARAMETERs as constants (static C++ arrays) but lifts
    BUFFERs as runtime inputs. At 124M params, inlining produces a ~1GB MLIR and
    multi-GB C++ that g++ cannot compile, so we turn weights into npy-fed inputs.
    """
    for name, p in list(mod.named_parameters(recurse=False)):
        delattr(mod, name)
        mod.register_buffer(name, p.data, persistent=False)
    for child in mod.children():
        params_to_buffers(child)


def export_from_hidden(model, args, outdir):
    """Port real GPT-2 weights, build hidden on host, export forward-from-hidden.

    torch.export lifts nn.Buffers (pos_ids + per-layer causal masks) as network
    inputs (params are inlined as constants). We must dump ALL inputs in the
    exact signature order: pos_ids, mask_0..mask_11, hidden. The hidden index is
    recorded so the generation driver knows which input to vary each step.
    """
    from port_gpt2_weights import port_weights, load_tokenizer
    from transformers import GPT2LMHeadModel

    port_weights(model, GPT2LMHeadModel.from_pretrained("gpt2").eval())
    tok = load_tokenizer()
    ids = tok(args.prompt, return_tensors="pt").input_ids[0]
    pad = torch.zeros(1, args.seq, dtype=torch.long)
    pad[0, : ids.shape[0]] = ids

    with torch.no_grad():
        hidden = model.embed(pad).detach()          # host-side embedding
        expected = model.forward_from_hidden(hidden)
    np.save(outdir / "expected_0.npy", expected.numpy())

    # Drop embedding tables (host-only, already used) so they aren't lifted as
    # large unused network inputs; then turn weights into npy-fed inputs.
    del model.tok_emb
    del model.pos_emb
    wrapper = FromHiddenWrapper(model).eval()
    params_to_buffers(wrapper)
    ep = torch.export.export(wrapper, (hidden,))
    bufs = dict(wrapper.named_buffers())
    n = 0
    hidden_idx = None
    for s in ep.graph_signature.input_specs:
        kind = str(s.kind)
        if "PARAMETER" in kind:
            continue  # inlined as constants in linalg
        if "BUFFER" in kind:
            t = bufs[s.target]
        else:  # USER_INPUT == hidden
            t = hidden
            hidden_idx = n
        np.save(outdir / f"input_{n}.npy", t.detach().cpu().numpy())
        n += 1
    (outdir / "hidden_input_index.txt").write_text(str(hidden_idx))

    mlir_text = torch_to_linalg(wrapper, [hidden], None)
    (outdir / "step0_linalg.mlir").write_text(mlir_text)

    print(f"prompt={args.prompt!r} real_tokens={ids.shape[0]}")
    print(f"hidden={tuple(hidden.shape)} expected={tuple(expected.shape)}")
    print(f"wrote {n} inputs (hidden at index {hidden_idx}) + expected_0 + linalg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-hidden", action="store_true",
                    help="port real GPT-2 weights, lift embedding to host, "
                         "export forward-from-hidden (input = hidden_states)")
    ap.add_argument("--prompt", default="The capital of France is",
                    help="prompt used to build hidden_states for --from-hidden")
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

    if args.from_hidden:
        export_from_hidden(model, args, outdir)
        return

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
