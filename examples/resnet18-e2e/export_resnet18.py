#!/usr/bin/env python3
"""torchvision ResNet-18 -> linalg MLIR + reference npy, for the network_runner pipeline.

Random-init weights (no checkpoint download), batch=1, 224x224 fp32 input.
Outputs step0_linalg.mlir, input_0.npy, expected_0.npy into --outdir.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "torch"))
from torch2linalg import torch_to_linalg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    model = resnet18(weights=None).eval().to(dtype)
    x = torch.randn(args.batch, 3, args.size, args.size, dtype=dtype)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        expected = model(x)
    np.save(outdir / "expected_0.npy", expected.numpy())

    mlir_text = torch_to_linalg(model, [x], None)
    (outdir / "step0_linalg.mlir").write_text(mlir_text)

    # ResNet's BatchNorm buffers (running_mean, running_var, num_batches_tracked)
    # are NOT inlined as constants by torch.export — they become function inputs.
    # Dump them in named_buffers() order (= torch.export's input order), with the
    # image tensor at the end. The number of inputs matches the kernel func's
    # arg count in step0_linalg.mlir.
    n_inputs = 0
    for name, buf in model.named_buffers():
        np.save(outdir / f"input_{n_inputs}.npy", buf.detach().cpu().numpy())
        n_inputs += 1
    np.save(outdir / f"input_{n_inputs}.npy", x.numpy())
    n_inputs += 1

    print(f"input={tuple(x.shape)} expected={tuple(expected.shape)} dtype={args.dtype}")
    print(f"wrote: {outdir}/step0_linalg.mlir, input_0..{n_inputs - 1}.npy ({n_inputs} inputs), expected_0.npy")


if __name__ == "__main__":
    main()
