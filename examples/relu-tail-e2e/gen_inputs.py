#!/usr/bin/env python3
import argparse, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1010)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    x = (rng.random(args.n, dtype=np.float32) * 4 - 2).astype(np.float32)
    expected = np.maximum(x, 0).astype(np.float32)
    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "input.npy"), x)
    np.save(os.path.join(args.outdir, "expected.npy"), expected)
    print(f"x={x.shape} expected={expected.shape}")

if __name__ == "__main__":
    main()
