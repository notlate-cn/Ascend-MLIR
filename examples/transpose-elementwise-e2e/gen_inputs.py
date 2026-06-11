#!/usr/bin/env python3
"""Generate test inputs for transpose-elementwise-e2e.

Computation:
  out = relu(transpose(x, [1,0]))     x : [M,N] f16  ->  out : [N,M] f16

Only `x` is an input; the output buffer is allocated by the runtime.
"""

import argparse
import numpy as np
import os

M_DEFAULT, N_DEFAULT = 16, 32
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=M_DEFAULT)
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    x = (rng.random((args.m, args.n), dtype=np.float32) * 4 - 2).astype(np.float16)
    expected = np.maximum(x.T, np.float16(0.0)).astype(np.float16)

    d = args.outdir
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "x.npy"), x)
    np.save(os.path.join(d, "expected.npy"), expected)

    print(f"x shape={x.shape}, expected shape={expected.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
