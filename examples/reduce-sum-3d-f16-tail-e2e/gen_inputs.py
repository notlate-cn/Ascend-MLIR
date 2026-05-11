#!/usr/bin/env python3
"""Generate test inputs for reduce-sum-3d f16 E2E.
   out[d0,d1] = sum_{d2}( x[d0,d1,d2] ), f16."""

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
    x = (rng.random((args.d0, args.d1, args.d2), dtype=np.float32) * 2 - 1).astype(np.float16)
    # Compute in f32 then cast (mirrors hardware accumulator); for small d2 (32)
    # f16 sum precision is fine with tight tolerance.
    expected = x.astype(np.float32).sum(axis=2).astype(np.float16)

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "x.npy"), x)
    np.save(os.path.join(args.outdir, "expected.npy"), expected)
    print(f"x={x.shape} f16, expected={expected.shape} f16")
    print(f"expected range: [{float(expected.min()):.3f}, {float(expected.max()):.3f}]")


if __name__ == "__main__":
    main()
