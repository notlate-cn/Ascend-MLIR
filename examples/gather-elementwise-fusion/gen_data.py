"""Generate input/expected-output npy files for gather-elementwise-fusion.

Computation: out[i, j] = relu(data[i, indices[j]]) + bias[j]
  data:    (M, N)   f16   input data matrix
  indices: (K,)     i64   gather column indices (values in [0, index_high))
  bias:    (K,)     f16   per-column bias added after relu+gather
  out:     (M, K)   f16   output

Runtime constraints:
  - M >= 1, N >= 1, K >= 1
  - K <= index_high <= N
  - one row of data[N] should fit in UB VECCALC for the current gather lowering
  - indices values in [0, N), all distinct (for reproducibility)
  - default index_high = N - 16 to avoid the final real-NPU Gather tail datablock

Usage:
  python3 gen_data.py [--m M] [--n N] [--k K]
      [--index-high HIGH] [--seed SEED] [--out-dir DIR]
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",       type=int, default=16,
                        help="Number of rows in data")
    parser.add_argument("--n",       type=int, default=640,
                        help="Number of columns in data (gather source width)")
    parser.add_argument("--k",       type=int, default=128,
                        help="Gather output width K (indices[K] -> data columns)")
    parser.add_argument("--index-high", type=int, default=None,
                        help="Exclusive upper bound for generated gather indices")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    index_high = args.index_high if args.index_high is not None else N - 16
    assert M >= 1, "M must be >= 1"
    assert N >= 1, "N must be >= 1"
    assert K >= 1, "K must be >= 1"
    assert index_high >= 1, "index_high must be >= 1"
    assert index_high <= N, "index_high must be <= N"
    assert K <= index_high, (
        "K must be <= index_high so generated indices are distinct and "
        "stay inside the real-NPU-safe gather range"
    )

    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    # data[M, N]: random f16
    data = rng.uniform(-1.0, 1.0, (M, N)).astype(np.float32).astype(np.float16)

    # indices[K]: K distinct column indices in [0, index_high), cast to i64
    indices = rng.choice(index_high, size=K, replace=False).astype(np.int64)

    # bias[K]: random f16
    bias = rng.uniform(-0.5, 0.5, (K,)).astype(np.float32).astype(np.float16)

    # Expected output: out[i, j] = relu(data[i, indices[j]]) + bias[j]
    # Compute in float32 for precision, then cast to f16
    data_f32 = data.astype(np.float32)
    bias_f32 = bias.astype(np.float32)
    gathered = data_f32[:, indices]          # (M, K)
    relu_gathered = np.maximum(gathered, 0.0)  # relu
    out_f32 = relu_gathered + bias_f32[None, :]  # broadcast bias over rows
    out = out_f32.astype(np.float16)

    np.save(out_dir / "input_data.npy",    data)
    np.save(out_dir / "input_indices.npy", indices)
    np.save(out_dir / "input_bias.npy",    bias)
    np.save(out_dir / "output_out.npy",    out)

    print(f"input_data:    {data.shape} {data.dtype}")
    print(
        f"input_indices: {indices.shape} {indices.dtype}  "
        f"range [{indices.min()}, {indices.max()}], index_high={index_high}"
    )
    print(f"input_bias:    {bias.shape} {bias.dtype}")
    print(f"output_out:    {out.shape} {out.dtype}  range [{out.min():.4g}, {out.max():.4g}]")


if __name__ == "__main__":
    main()
