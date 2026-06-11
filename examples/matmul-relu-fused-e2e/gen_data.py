#!/usr/bin/env python3
"""Generate input/expected-output npy files for matmul+relu fused kernel.

Computation: D[m,n] = relu(A[m,k] @ B[k,n])
  A:    (M, K)  float16
  B:    (K, N)  float16
  init: (M, N)  float32  (DPS matmul output; here we treat as zeros)
  D:    (M, N)  float32
"""
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M",       type=int, default=32)
    parser.add_argument("--K",       type=int, default=16)
    parser.add_argument("--N",       type=int, default=64)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    A    = rng.integers(-10, 10, (M, K)).astype(np.float16)
    B    = rng.integers(-10, 10, (K, N)).astype(np.float16)
    init = np.zeros((M, N), dtype=np.float32)

    matmul = A.astype(np.float32) @ B.astype(np.float32)   # [M, N]
    output = np.maximum(matmul, 0.0).astype(np.float32)    # relu

    np.save(out_dir / "input_a.npy",    A)
    np.save(out_dir / "input_b.npy",    B)
    np.save(out_dir / "input_init.npy", init)
    np.save(out_dir / "output.npy",     output)
    np.save(out_dir / "input0.npy",     A)
    np.save(out_dir / "input1.npy",     B)
    np.save(out_dir / "input2.npy",     init)
    np.save(out_dir / "output0.npy",    output)

    print(f"input_a:    {A.shape} {A.dtype}")
    print(f"input_b:    {B.shape} {B.dtype}")
    print(f"input_init: {init.shape} {init.dtype}")
    print(f"output:     {output.shape} {output.dtype}  "
          f"range [{output.min()}, {output.max()}]")


if __name__ == "__main__":
    main()
