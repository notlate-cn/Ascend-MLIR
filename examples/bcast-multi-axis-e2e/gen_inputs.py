#!/usr/bin/env python3
"""Inputs for bcast-multi-axis-e2e: out[d0,d1,d2] = b[d0,d1,d2] + a[d1]."""
import argparse, numpy as np, os
SEED = 42

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d0", type=int, default=4)
    p.add_argument("--d1", type=int, default=8)
    p.add_argument("--d2", type=int, default=16)
    p.add_argument("--outdir", type=str, default=".")
    a = p.parse_args()
    rng = np.random.default_rng(SEED)
    av = (rng.random((a.d1,), dtype=np.float32) * 4 - 2).astype(np.float32)
    bv = (rng.random((a.d0, a.d1, a.d2), dtype=np.float32) * 4 - 2).astype(np.float32)
    expected = (bv + av[None, :, None]).astype(np.float32)
    os.makedirs(a.outdir, exist_ok=True)
    np.save(os.path.join(a.outdir, "a.npy"), av)
    np.save(os.path.join(a.outdir, "b.npy"), bv)
    np.save(os.path.join(a.outdir, "expected.npy"), expected)
    print(f"a shape={av.shape}, b shape={bv.shape}, expected shape={expected.shape}")
    print(f"a: [{av.min():.3f}, {av.max():.3f}]  b: [{bv.min():.3f}, {bv.max():.3f}]  expected: [{expected.min():.3f}, {expected.max():.3f}]")
    print("Done.")

if __name__ == "__main__":
    main()
