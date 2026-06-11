#!/usr/bin/env python3
"""Host-side token-by-token generation driver for GPT-2 on our pipeline.

Compile-once/run-many: the network binary is built once (Task 3); each step the
host computes embedding -> overwrites the hidden input npy -> invokes the binary
with all network inputs -> reads logits -> greedy-picks the next token. No
KV-cache; the full [1,N] window is recomputed each step.

The exported network takes 161 inputs (pos_ids + 12 causal masks + ~147 weight
tensors + hidden). All but `hidden` are constant across steps, so they are
staged once from the artifact dir; only the hidden input (at the recorded index)
is rewritten per step.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "gpt2-e2e"))
from export_gpt2 import MiniGPT  # noqa: E402
from port_gpt2_weights import CFG, port_weights, load_tokenizer  # noqa: E402


def build_model():
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    mini = MiniGPT(vocab=CFG["vocab"], n_embd=CFG["n_embd"], n_head=CFG["n_head"],
                   n_layer=CFG["n_layer"], seq_len=CFG["seq_len"]).eval()
    port_weights(mini, hf)
    return mini, load_tokenizer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, help="compiled network_test binary")
    ap.add_argument("--artifact", required=True,
                    help="export dir with input_*.npy + hidden_input_index.txt")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--io-dir", default="/tmp/gpt2_gen_io")
    args = ap.parse_args()

    art = Path(args.artifact)
    n_inputs = len(list(art.glob("input_*.npy")))
    hidden_idx = int((art / "hidden_input_index.txt").read_text().strip())

    # Stage the constant inputs (everything except hidden) once.
    io_dir = Path(args.io_dir)
    io_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_inputs):
        if i != hidden_idx:
            shutil.copy(art / f"input_{i}.npy", io_dir / f"input_{i}.npy")
    in_paths = [str(io_dir / f"input_{i}.npy") for i in range(n_inputs)]
    out_path = io_dir / "out0.npy"

    mini, tok = build_model()
    N = CFG["seq_len"]
    ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()

    for _ in range(args.max_new):
        pos = len(ids) - 1
        if pos >= N - 1:
            print("[generate] window full, stopping")
            break
        pad = torch.zeros(1, N, dtype=torch.long)
        pad[0, : len(ids)] = torch.tensor(ids)
        with torch.no_grad():
            hidden = mini.embed(pad).numpy()
        np.save(io_dir / f"input_{hidden_idx}.npy", hidden)  # vary only hidden

        if out_path.exists():
            out_path.unlink()
        cmd = [str(args.binary)]
        for p in in_paths:
            cmd += ["--input", p]
        cmd += ["--output", str(out_path)]
        subprocess.run(cmd, check=True)

        logits = np.load(out_path)
        nxt = int(logits[0, pos].argmax())  # greedy
        ids.append(nxt)
        print(f"[{len(ids)-1}] +{tok.decode([nxt])!r} -> {tok.decode(ids)!r}",
              flush=True)
        if nxt == tok.eos_token_id:
            break

    print("FINAL:", repr(tok.decode(ids)))


if __name__ == "__main__":
    main()
