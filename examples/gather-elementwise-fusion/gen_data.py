"""Generate input/expected-output npy files for gather-elementwise-fusion.

Computation: out[i, j] = relu(data[i, indices[j]]) + bias[j]
  data:    (M, N)   f16   input data matrix
  indices: (K,)     i64   gather column indices (values in [0, N))
  bias:    (K,)     f16   per-column bias added after relu+gather
  out:     (M, K)   f16   output

Tiling constraints (参数设计规则):
  - K >= 16  (DataCopy half requires >= 16 elements per transfer)
  - TB_M divides M evenly (no tail block for simplicity)
  - Tb_M = 1  (row-by-row gather; one row of data[N] fits in UB VECCALC)
  - indices values are distinct and stay within the requested index range

Usage:
  python3 gen_data.py [--m M] [--n N] [--k K] [--index-high H] [--seed SEED] [--out-dir DIR]
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",       type=int, default=16,
                        help="Number of rows in data (must be divisible by TB_M=16)")
    parser.add_argument("--n",       type=int, default=640,
                        help="Number of columns in data (gather source width)")
    parser.add_argument("--k",       type=int, default=128,
                        help="Gather output width K (>= 16, indices[K] -> data columns)")
    parser.add_argument("--index-high", type=int, default=None,
                        help="Exclusive upper bound for generated indices. Default: N")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    index_high = N if args.index_high is None else args.index_high
    assert K >= 16, "K must be >= 16 for DataCopy alignment"
    assert M % 16 == 0, "M must be divisible by TB_M=16"
    assert 0 < index_high <= N, "index_high must be in (0, N]"
    assert K <= index_high, "K must be <= index_high for distinct indices"

    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    # data[M, N]: random f16
    data = rng.uniform(-1.0, 1.0, (M, N)).astype(np.float32).astype(np.float16)

    # indices[K]: K distinct column indices in [0, index_high), cast to i64.
    # Real AscendC Gather can access a tail 32B datablock from the source row;
    # examples keep away from the final half datablock until tail padding is
    # represented as part of the runtime input contract.
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
    print(f"input_indices: {indices.shape} {indices.dtype}  range [{indices.min()}, {indices.max()}]")
    print(f"input_bias:    {bias.shape} {bias.dtype}")
    print(f"output_out:    {out.shape} {out.dtype}  range [{out.min():.4g}, {out.max():.4g}]")


if __name__ == "__main__":
    main()
