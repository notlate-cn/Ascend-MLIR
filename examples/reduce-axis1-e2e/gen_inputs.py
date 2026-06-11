#!/usr/bin/env python3
"""Generate test inputs for reduce-axis1-e2e example.

Computation:
  out[d0, d2] = sum_{d1}( x[d0, d1, d2] )   (reduce the MIDDLE axis)

Only `x` is an input; the output buffer is allocated by the runtime (the
kernel zeros its per-tile accumulator internally).
"""

import argparse
import numpy as np
import os

D0_DEFAULT, D1_DEFAULT, D2_DEFAULT = 4, 8, 32
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0", type=int, default=D0_DEFAULT)
    parser.add_argument("--d1", type=int, default=D1_DEFAULT)
    parser.add_argument("--d2", type=int, default=D2_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    x = (rng.random((args.d0, args.d1, args.d2), dtype=np.float32) * 4 - 2).astype(np.float32)
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
