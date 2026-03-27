#!/usr/bin/env python3
"""Generate test inputs for add-broadcast-concat example.

Computation:
  C[M,N] = input_a[M] + input_b[M,N]   (broadcast add)
  D[M,N] = input_c[M] * input_d[M,N]   (broadcast mul)
  output[2M,N] = concat(C, D, axis=0)

All inputs are float16, values in [-1, 1].
"""

import argparse
import numpy as np

M_DEFAULT = 640
N_DEFAULT = 500
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=M_DEFAULT)
    parser.add_argument("--N", type=int, default=N_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    M, N = args.M, args.N
    rng = np.random.default_rng(SEED)

    def rand(shape):
        return (rng.random(shape, dtype=np.float32) * 2 - 1).astype(np.float16)

    input_a = rand((M,))
    input_b = rand((M, N))
    input_c = rand((M,))
    input_d = rand((M, N))

    C = input_a[:, None] + input_b
    D = input_c[:, None] * input_d
    output = np.concatenate([C, D], axis=0)

    import os
    d = args.outdir
    np.save(os.path.join(d, "input_a.npy"), input_a)
    np.save(os.path.join(d, "input_b.npy"), input_b)
    np.save(os.path.join(d, "input_c.npy"), input_c)
    np.save(os.path.join(d, "input_d.npy"), input_d)
    np.save(os.path.join(d, "output.npy"),  output)

    print(f"M={M}, N={N}, seed={SEED}")
    print(f"input_a: {input_a.shape} {input_a.dtype}  [{input_a.min():.3f}, {input_a.max():.3f}]")
    print(f"input_b: {input_b.shape} {input_b.dtype}")
    print(f"output:  {output.shape} {output.dtype}")
    print("Done.")


if __name__ == "__main__":
    main()
