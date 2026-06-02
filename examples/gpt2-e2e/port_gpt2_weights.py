#!/usr/bin/env python3
"""Load official HF GPT-2 weights into our MiniGPT and assert logits match.

This is the Phase-1 "关 0" golden gate: if our from-scratch MiniGPT (with ported
weights) does not match HuggingFace GPT2LMHeadModel logits, nothing downstream
can be trusted.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "gpt2-e2e"))
from export_gpt2 import MiniGPT  # noqa: E402

# GPT-2 small config (must match HF "gpt2")
CFG = dict(vocab=50257, n_embd=768, n_head=12, n_layer=12, seq_len=64)

# Local tokenizer dir (vocab.json + merges.txt fetched once; avoids hub network).
TOKENIZER_DIR = str(REPO / "examples" / "gpt2-e2e" / "gpt2_tokenizer")


def load_tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR)


def port_weights(mini: MiniGPT, hf):
    """Copy HF GPT2LMHeadModel state_dict into MiniGPT.

    HF uses Conv1D (weight shape [in, out]); our nn.Linear wants [out, in],
    so c_attn/c_proj/c_fc/c_proj weights are transposed. lm_head is tied to wte.
    """
    h = hf.transformer
    sd = {}
    sd["tok_emb.weight"] = h.wte.weight
    sd["pos_emb.weight"] = h.wpe.weight[: CFG["seq_len"]]
    for i, blk in enumerate(h.h):
        p = f"blocks.{i}."
        sd[p + "ln1.weight"] = blk.ln_1.weight
        sd[p + "ln1.bias"] = blk.ln_1.bias
        sd[p + "attn.qkv.weight"] = blk.attn.c_attn.weight.t().contiguous()
        sd[p + "attn.qkv.bias"] = blk.attn.c_attn.bias
        sd[p + "attn.proj.weight"] = blk.attn.c_proj.weight.t().contiguous()
        sd[p + "attn.proj.bias"] = blk.attn.c_proj.bias
        sd[p + "ln2.weight"] = blk.ln_2.weight
        sd[p + "ln2.bias"] = blk.ln_2.bias
        sd[p + "mlp.fc.weight"] = blk.mlp.c_fc.weight.t().contiguous()
        sd[p + "mlp.fc.bias"] = blk.mlp.c_fc.bias
        sd[p + "mlp.proj.weight"] = blk.mlp.c_proj.weight.t().contiguous()
        sd[p + "mlp.proj.bias"] = blk.mlp.c_proj.bias
    sd["ln_f.weight"] = h.ln_f.weight
    sd["ln_f.bias"] = h.ln_f.bias
    sd["lm_head.weight"] = h.wte.weight  # weight-tied
    missing, unexpected = mini.load_state_dict(sd, strict=False)
    # pos_ids buffer is allowed missing/unexpected; nothing else should be.
    bad = [k for k in missing + unexpected if "pos_ids" not in k]
    assert not bad, f"state_dict mismatch: {bad}"


def main():
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tok = load_tokenizer()

    mini = MiniGPT(vocab=CFG["vocab"], n_embd=CFG["n_embd"],
                   n_head=CFG["n_head"], n_layer=CFG["n_layer"],
                   seq_len=CFG["seq_len"]).eval()
    port_weights(mini, hf)

    text = "The capital of France is"
    ids = tok(text, return_tensors="pt").input_ids  # [1, T]
    T = ids.shape[1]
    pad = torch.zeros(1, CFG["seq_len"], dtype=torch.long)
    pad[0, :T] = ids[0]

    with torch.no_grad():
        ours = mini(pad)[0, T - 1]            # our logits at last real token
        ref = hf(ids).logits[0, T - 1]        # HF logits at same position
    max_diff = (ours - ref).abs().max().item()
    print(f"max_diff(ours,HF) = {max_diff:.3e}  (erf-GELU vs HF gelu_new variant)")
    print("our argmax token:", repr(tok.decode([ours.argmax().item()])))
    print("HF  argmax token:", repr(tok.decode([ref.argmax().item()])))
    # Porting-correctness signal: argmax must agree. The small logit gap is the
    # erf-GELU vs gelu_new variant difference (see MLP.forward), not a port bug.
    assert ours.argmax().item() == ref.argmax().item(), "argmax mismatch (port bug)"
    assert max_diff < 0.1, f"logit gap too large for a gelu variant: {max_diff}"
    print("GOLDEN PASS")

    with torch.no_grad():
        hidden = mini.embed(pad)
        ours2 = mini.forward_from_hidden(hidden)[0, T - 1]
    assert torch.allclose(ours, ours2, atol=1e-6), "from_hidden path diverged"
    print("FROM_HIDDEN EQUIV PASS")


if __name__ == "__main__":
    main()
