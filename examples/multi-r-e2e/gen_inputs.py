#!/usr/bin/env python3
"""Generate test inputs for multi-r-e2e (adjacent multi-reduce-axes).

Computation:
  out[a] = sum_{r1, r2}( x[a, r1, r2] )

Only `x` is an input; the output (rank-1) is allocated by the runtime.
The kernel zero-initializes its per-tile accumulator internally.
"""

import argparse
import numpy as np
import os

A_DEFAULT, R1_DEFAULT, R2_DEFAULT = 8, 16, 32
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=int, default=A_DEFAULT)
    parser.add_argument("--r1", type=int, default=R1_DEFAULT)
    parser.add_argument("--r2", type=int, default=R2_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    x = (rng.random((args.a, args.r1, args.r2),
                    dtype=np.float32) * 4 - 2).astype(np.float32)
    expected = x.sum(axis=(1, 2)).astype(np.float32)

    d = args.outdir
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "x.npy"), x)
    np.save(os.path.join(d, "expected.npy"), expected)

    print(f"x shape={x.shape}, expected shape={expected.shape}")
    print(f"expected: {expected}")
    print("Done.")


if __name__ == "__main__":
    main()
