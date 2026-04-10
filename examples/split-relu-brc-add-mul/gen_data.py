#!/usr/bin/env python3
"""Generate test inputs/expected output for split-relu-brc-add-mul example.

Computation:
  a0 = input_a[0:M/2, :]          (top half)
  a1 = input_a[M/2:M, :]          (bottom half)
  out0[M/2,N] = relu(a0) + bias0[:,None]  * scale0[None,:]  (brc add then brc mul)
  out1[M/2,N] = relu(a1) + bias1[:,None]  * scale1[None,:]
  output[M,N] = concat([out0, out1], axis=0)

All tensors are float16.
"""

import argparse
import numpy as np
import os

M_DEFAULT = 640
N_DEFAULT = 512  # Must be multiple of 16 (AscendC DataCopy alignment for f16)
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=M_DEFAULT)
    parser.add_argument("--N", type=int, default=N_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    M, N = args.M, args.N
    HM = M // 2
    rng = np.random.default_rng(SEED)

    def rand(shape):
        return (rng.random(shape, dtype=np.float32) * 2 - 1).astype(np.float16)

    input_a = rand((M, N))
    bias0   = rand((HM,))
    bias1   = rand((HM,))
    scale0  = rand((N,))
    scale1  = rand((N,))

    a0 = input_a[:HM, :]
    a1 = input_a[HM:, :]

    # relu + brc_add + brc_mul, compute in float32 for accuracy
    out0 = (np.maximum(a0.astype(np.float32), 0.0)
            + bias0[:, None].astype(np.float32)) * scale0[None, :].astype(np.float32)
    out1 = (np.maximum(a1.astype(np.float32), 0.0)
            + bias1[:, None].astype(np.float32)) * scale1[None, :].astype(np.float32)
    output = np.concatenate([out0, out1], axis=0).astype(np.float16)

    d = args.outdir
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "input_a.npy"), input_a)
    np.save(os.path.join(d, "bias0.npy"),   bias0)
    np.save(os.path.join(d, "bias1.npy"),   bias1)
    np.save(os.path.join(d, "scale0.npy"),  scale0)
    np.save(os.path.join(d, "scale1.npy"),  scale1)
    np.save(os.path.join(d, "output.npy"),  output)

    print(f"M={M}, N={N}, HM={HM}, seed={SEED}")
    print(f"input_a : {input_a.shape} {input_a.dtype}")
    print(f"bias0   : {bias0.shape}  {bias0.dtype}")
    print(f"bias1   : {bias1.shape}  {bias1.dtype}")
    print(f"scale0  : {scale0.shape}  {scale0.dtype}")
    print(f"scale1  : {scale1.shape}  {scale1.dtype}")
    print(f"output  : {output.shape} {output.dtype}")
    print("Done.")


if __name__ == "__main__":
    main()
