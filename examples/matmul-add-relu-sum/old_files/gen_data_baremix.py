"""Generate input/expected-output npy for baremix_custom reference kernel.

Computation: leakyrelu(A[fp16,M,K] @ B[fp16,K,N] + bias[fp32,N], alpha=0.001)
  A:    (M, K)  float16
  B:    (K, N)  float16
  bias: (N,)    float32
  out:  (M, N)  float32

Matches scripts/gen_data.py from BareMixInvocation: M=128, N=128, K=256.
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

    # Reference: same as BareMixInvocation/scripts/gen_data.py
    alpha  = 0.001
    result = (A.astype(np.float32) @ B.astype(np.float32)) + bias  # broadcast bias[N]
    output = np.where(result >= 0, result, result * alpha).astype(np.float32)

    np.save(out_dir / "input_a.npy",    A)
    np.save(out_dir / "input_b.npy",    B)
    np.save(out_dir / "input_bias.npy", bias)
    np.save(out_dir / "output.npy",     output)

    print(f"input_a:    {A.shape} {A.dtype}")
    print(f"input_b:    {B.shape} {B.dtype}")
    print(f"input_bias: {bias.shape} {bias.dtype}")
    print(f"output:     {output.shape} {output.dtype}  "
          f"range [{output.min():.4g}, {output.max():.4g}]")


if __name__ == "__main__":
    main()
