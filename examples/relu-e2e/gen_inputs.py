#!/usr/bin/env python3
"""Generate test inputs for relu-e2e example.

Computation:
  output[N] = max(input[N], 0.0)   (element-wise ReLU, float32)
"""

import argparse
import numpy as np
import os

N_DEFAULT = 1024
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=N_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    N = args.N
    rng = np.random.default_rng(SEED)

    inp = (rng.random(N, dtype=np.float32) * 4 - 2).astype(np.float32)
    expected = np.maximum(inp, 0.0).astype(np.float32)

    d = args.outdir
    np.save(os.path.join(d, "input.npy"), inp)
    np.save(os.path.join(d, "expected.npy"), expected)

    print(f"N={N}, seed={SEED}")
    print(f"input:    {inp.shape} {inp.dtype}  [{inp.min():.3f}, {inp.max():.3f}]")
    print(f"expected: {expected.shape} {expected.dtype}  [{expected.min():.3f}, {expected.max():.3f}]")
    print(f"negative fraction: {(inp < 0).mean():.1%}")
    print("Done.")


if __name__ == "__main__":
    main()
