#!/usr/bin/env python3
"""Generate test inputs for combo-elewise-reduce-e2e.

  y[d0,d1,d2]  = a[d0,d1,d2] + b[d0,d1,d2]
  out[d0,d1]   = sum_{d2}( y[d0,d1,d2] )
"""

import argparse, os
import numpy as np

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=int, default=4)
    ap.add_argument("--d1", type=int, default=8)
    ap.add_argument("--d2", type=int, default=32)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    shape = (args.d0, args.d1, args.d2)
    a = (rng.random(shape, dtype=np.float32) * 4 - 2).astype(np.float32)
    b = (rng.random(shape, dtype=np.float32) * 4 - 2).astype(np.float32)
    expected = (a + b).sum(axis=2).astype(np.float32)

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "a.npy"), a)
    np.save(os.path.join(args.outdir, "b.npy"), b)
    np.save(os.path.join(args.outdir, "expected.npy"), expected)
    print(f"a={a.shape} b={b.shape} expected={expected.shape}")
    print(f"expected range: [{expected.min():.3f}, {expected.max():.3f}]")


if __name__ == "__main__":
    main()
