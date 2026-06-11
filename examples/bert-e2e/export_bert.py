#!/usr/bin/env python3
"""Tiny BERT layer -> linalg MLIR + reference npy, for the network_runner pipeline.

Feeds hidden_states directly into a HuggingFace BertLayer (skips embedding lookup
and attention mask) so the matmul/attention/layernorm/vector backbone is what gets
exercised. Outputs step0_linalg.mlir, input_0.npy, expected_0.npy into --outdir.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Make torch2linalg importable (mirrors examples/torch_e2e/conftest.py).
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "torch"))
from torch2linalg import torch_to_linalg  # noqa: E402

from transformers import BertConfig  # noqa: E402
from transformers.models.bert.modeling_bert import BertLayer  # noqa: E402


class BertLayerWrapper(nn.Module):
    """Single BertLayer taking hidden_states, returning the layer output tensor."""
    def __init__(self, config):
        super().__init__()
        self.layer = BertLayer(config)

    def forward(self, hidden_states):
        out = self.layer(hidden_states)
        # transformers 5.5.4 returns a tensor; older versions a tuple. Handle both.
        return out[0] if isinstance(out, (tuple, list)) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--intermediate", type=int, default=None,
                    help="FFN intermediate size (default 4*hidden)")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    inter = args.intermediate or 4 * args.hidden

    config = BertConfig(
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        intermediate_size=inter,
        num_hidden_layers=1,
        max_position_embeddings=max(args.seq, 16),
        vocab_size=32,                 # unused (we skip embeddings) but must be set
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )

    model = BertLayerWrapper(config).eval().to(dtype)
    # batch=1; hidden_states shape [B, seq, hidden]
    hidden = torch.randn(1, args.seq, args.hidden, dtype=dtype)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Reference output from PyTorch.
    with torch.no_grad():
        expected = model(hidden)
    np.save(outdir / "input_0.npy", hidden.numpy())
    np.save(outdir / "expected_0.npy", expected.numpy())

    # torch.export -> linalg (static shapes; no dynamic_shapes for bring-up).
    mlir_text = torch_to_linalg(model, [hidden], None)
    (outdir / "step0_linalg.mlir").write_text(mlir_text)

    print(f"hidden_states={tuple(hidden.shape)} expected={tuple(expected.shape)} dtype={args.dtype}")
    print(f"wrote: {outdir}/step0_linalg.mlir, input_0.npy, expected_0.npy")


if __name__ == "__main__":
    main()
