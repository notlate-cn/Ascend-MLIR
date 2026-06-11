#!/usr/bin/env python3
"""Generate test inputs for leading-reduce-e2e (RA pattern + FullLoad).

Computation:
  out[d1] = sum_{d0}( x[d0, d1] )

Only `x` is an input; the output buffer is allocated by the runtime (the
kernel zeros its per-tile accumulator internally).
"""

import argparse
import numpy as np
import os

D0_DEFAULT, D1_DEFAULT = 8, 1024
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0", type=int, default=D0_DEFAULT)
    parser.add_argument("--d1", type=int, default=D1_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    x = (rng.random((args.d0, args.d1), dtype=np.float32) * 4 - 2).astype(np.float32)
    expected = x.sum(axis=0).astype(np.float32)

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
