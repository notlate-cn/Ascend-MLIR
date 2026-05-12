#!/usr/bin/env python3
"""Generate test inputs for transpose-preserve-e2e.

Computation:
  t      = transpose(x, [1,0])     x : [M,N] f16  ->  t : [N,M] f16
  out_a  = relu(t)
  out_b  = t * 2

Only `x` is an input; the two output buffers are allocated by the runtime.
"""

import argparse
import numpy as np
import os

M_DEFAULT, N_DEFAULT = 16, 32
SEED = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=M_DEFAULT)
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    x = (rng.random((args.m, args.n), dtype=np.float32) * 4 - 2).astype(np.float16)
    t = x.T
    expected_a = np.maximum(t, np.float16(0.0)).astype(np.float16)
    expected_b = (t * np.float16(2.0)).astype(np.float16)

    d = args.outdir
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "x.npy"), x)
    np.save(os.path.join(d, "expected_a.npy"), expected_a)
    np.save(os.path.join(d, "expected_b.npy"), expected_b)

    print(f"x shape={x.shape}, out shape={t.shape} (x2)")
    print("Done.")


if __name__ == "__main__":
    main()
