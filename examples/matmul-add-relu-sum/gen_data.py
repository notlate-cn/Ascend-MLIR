"""Generate input/expected-output npy files for matmul-add-relu-sum.

Computation: output[m,n] = max(0, sum_k(A[m,k] * B[k,n]) + bias[m,n])
  A:    (M, K)  float32
  B:    (K, N)  float32
  bias: (M, N)  float32
  out:  (M, N)  float32  (ReLU of matmul+bias)

Usage:
  python3 gen_data.py [--M M] [--K K] [--N N] [--seed SEED] [--out-dir DIR]
"""

import argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M",       type=int, default=512)
    parser.add_argument("--K",       type=int, default=256)
    parser.add_argument("--N",       type=int, default=640)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    # Use small range to keep matmul accumulation from overflowing f32
    A    = rng.uniform(-0.1, 0.1, (M, K)).astype(np.float32)
    B    = rng.uniform(-0.1, 0.1, (K, N)).astype(np.float32)
    bias = rng.uniform(-0.1, 0.1, (M, N)).astype(np.float32)

    # Reference: matmul in float64 then cast, to maximise numerical accuracy
    matmul = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
    added  = matmul + bias
    output = np.maximum(added, 0.0).astype(np.float32)

    np.save(out_dir / "input_a.npy",  A)
    np.save(out_dir / "input_b.npy",  B)
    np.save(out_dir / "input_bias.npy", bias)
    np.save(out_dir / "output.npy",   output)

    print(f"input_a:    {A.shape} {A.dtype}")
    print(f"input_b:    {B.shape} {B.dtype}")
    print(f"input_bias: {bias.shape} {bias.dtype}")
    print(f"output:     {output.shape} {output.dtype}  range [{output.min():.4g}, {output.max():.4g}]")

if __name__ == "__main__":
    main()
