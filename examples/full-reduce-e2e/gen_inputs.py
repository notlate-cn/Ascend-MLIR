#!/usr/bin/env python3
"""Generate test inputs for full-reduce-e2e (RCore template).

Computation:
  out = sum_{d0}( x[d0] )   (scalar)

Only `x` is an input; the output (rank-0 scalar) is allocated by the runtime.
The kernel zero-initializes its per-core accumulator internally.
"""

import argparse
import numpy as np
import os

D0_DEFAULT = 256
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0", type=int, default=D0_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    x = (rng.random((args.d0,), dtype=np.float32) * 4 - 2).astype(np.float32)
    expected = np.array(x.sum(), dtype=np.float32)

    d = args.outdir
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "x.npy"), x)
    np.save(os.path.join(d, "expected.npy"), expected)

    print(f"x shape={x.shape}, expected (scalar)={expected.item():.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
