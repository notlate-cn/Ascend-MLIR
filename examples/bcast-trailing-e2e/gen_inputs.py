#!/usr/bin/env python3
"""Inputs for bcast-trailing-e2e: out[a,b,c] = x[a,b,c] + y[a,b]  (y broadcast over c)."""
import argparse, numpy as np, os
SEED = 42

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=int, default=8)
    p.add_argument("--b", type=int, default=4)
    p.add_argument("--c", type=int, default=32)
    p.add_argument("--outdir", type=str, default=".")
    args = p.parse_args()
    rng = np.random.default_rng(SEED)
    x = (rng.random((args.a, args.b, args.c), dtype=np.float32) * 4 - 2).astype(np.float32)
    y = (rng.random((args.a, args.b), dtype=np.float32) * 4 - 2).astype(np.float32)
    expected = (x + y[:, :, None]).astype(np.float32)
    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "x.npy"), x)
    np.save(os.path.join(args.outdir, "y.npy"), y)
    np.save(os.path.join(args.outdir, "expected.npy"), expected)
    print(f"x shape={x.shape}, y shape={y.shape}, expected shape={expected.shape}")
    print(f"x: [{x.min():.3f}, {x.max():.3f}]  y: [{y.min():.3f}, {y.max():.3f}]  expected: [{expected.min():.3f}, {expected.max():.3f}]")
    print("Done.")

if __name__ == "__main__":
    main()
