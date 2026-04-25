"""Generate Q/K/V inputs and PyTorch reference output for aclnn-attn-e2e.

Reference: scaled dot-product attention without masking (full bidirectional).
  out = softmax(Q @ K^T / sqrt(D)) @ V

Output files (all float16, BNSD layout):
  q.npy      [1, 2, 16, 8]
  k.npy      [1, 2, 16, 8]
  v.npy      [1, 2, 16, 8]
  mask.npy   [1, 2, 16, 16]   zeros → no additive bias effect
  init.npy   [1, 2, 16, 8]    zeros → placeholder
  expected.npy [1, 2, 16, 8]  reference attention output

Usage:
  python3 gen_ref.py [--out-dir DIR] [--seed SEED]
"""

import argparse
import math
import numpy as np
from pathlib import Path


def softmax_f32(x, axis=-1):
    """Stable softmax in float32."""
    x = x.astype(np.float32)
    x -= x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sdpa_reference(q, k, v):
    """Scaled dot-product attention; all inputs [B, N, S, D] float16."""
    q = q.astype(np.float32)
    k = k.astype(np.float32)
    v = v.astype(np.float32)
    B, N, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = np.einsum("bnsd,bnkd->bnsk", q, k) * scale  # [B, N, S, S]
    probs = softmax_f32(scores, axis=-1)  # [B, N, S, S]
    out = np.einsum("bnsk,bnkd->bnsd", probs, v)  # [B, N, S, D]
    return out.astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)

    B, N, S, D = 1, 2, 16, 8

    q = rng.uniform(-0.5, 0.5, (B, N, S, D)).astype(np.float16)
    k = rng.uniform(-0.5, 0.5, (B, N, S, D)).astype(np.float16)
    v = rng.uniform(-0.5, 0.5, (B, N, S, D)).astype(np.float16)
    mask = np.zeros((B, N, S, S), dtype=np.float16)  # no additive bias
    init = np.zeros((B, N, S, D), dtype=np.float16)  # placeholder

    expected = sdpa_reference(q, k, v)

    np.save(out_dir / "q.npy", q)
    np.save(out_dir / "k.npy", k)
    np.save(out_dir / "v.npy", v)
    np.save(out_dir / "mask.npy", mask)
    np.save(out_dir / "init.npy", init)
    np.save(out_dir / "expected.npy", expected)

    print(f"q:        {q.shape} {q.dtype}  range [{float(q.min()):.3g}, {float(q.max()):.3g}]")
    print(f"k:        {k.shape} {k.dtype}")
    print(f"v:        {v.shape} {v.dtype}")
    print(
        f"expected: {expected.shape} {expected.dtype}  range [{float(expected.min()):.3g}, {float(expected.max()):.3g}]")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
