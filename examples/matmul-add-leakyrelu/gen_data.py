#!/usr/bin/env python3
"""Generate input/expected-output npy files for matmul_add_leakyrelu kernel.

Computation: E[m,n] = leakyrelu(A[m,k] @ B[k,n] + bias[n], alpha=0.001)
  A:    (M, K)  float16
  B:    (K, N)  float16
  bias: (N,)    float32  (1-D, broadcast over M rows)
  E:    (M, N)  float32
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M",       type=int, default=128)
    parser.add_argument("--K",       type=int, default=256)
    parser.add_argument("--N",       type=int, default=128)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    A    = rng.integers(-10, 10, (M, K)).astype(np.float16)
    B    = rng.integers(-10, 10, (K, N)).astype(np.float16)
    bias = rng.integers(1, 10, (N,)).astype(np.float32)

    matmul = A.astype(np.float32) @ B.astype(np.float32)   # [M, N]
    added  = matmul + bias                                   # broadcast [N] -> [M,N]
    alpha  = 0.001
    output = np.where(added >= 0, added, added * alpha).astype(np.float32)

    np.save(out_dir / "input_a.npy",    A)
    np.save(out_dir / "input_b.npy",    B)
    np.save(out_dir / "input_bias.npy", bias)
    np.save(out_dir / "output.npy",     output)
    np.save(out_dir / "input0.npy",     A)
    np.save(out_dir / "input1.npy",     B)
    np.save(out_dir / "input2.npy",     bias)
    np.save(out_dir / "output0.npy",    output)

    print(f"input_a:    {A.shape} {A.dtype}")
    print(f"input_b:    {B.shape} {B.dtype}")
    print(f"input_bias: {bias.shape} {bias.dtype}")
    print(f"output:     {output.shape} {output.dtype}  "
          f"range [{output.min():.4g}, {output.max():.4g}]")


if __name__ == "__main__":
    main()
