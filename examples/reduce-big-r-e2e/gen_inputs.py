#!/usr/bin/env python3
"""Generate test inputs for reduce-big-r-e2e example.

Computation:
  out[a] = sum_{r}( x[a, r] )

Only `x` is an input; the output buffer is allocated by the runtime (the
kernel zeros its per-tile accumulator internally).
"""

import argparse
import numpy as np
import os

A_DEFAULT, R_DEFAULT = 8, 65536
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=int, default=A_DEFAULT)
    parser.add_argument("--r", type=int, default=R_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    x = (rng.random((args.a, args.r), dtype=np.float32) * 4 - 2).astype(np.float32)
    expected = x.sum(axis=1).astype(np.float32)

    d = args.outdir
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "x.npy"), x)
    np.save(os.path.join(d, "expected.npy"), expected)

    print(f"x shape={x.shape}, expected shape={expected.shape}")
    print(f"x:        [{x.min():.3f}, {x.max():.3f}]")
    print(f"expected: [{expected.min():.3f}, {expected.max():.3f}]")
    print("Done.")


if __name__ == "__main__":
    main()
