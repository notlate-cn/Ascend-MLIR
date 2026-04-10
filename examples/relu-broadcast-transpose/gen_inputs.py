"""Generate input/expected-output npy files for relu-broadcast-transpose.

Computation: out[n, m] = relu(data0[m, 0]) + data1[n, m]
  data0: (M, 1)   column vector, relu applied element-wise
  data1: (N, M)   2-D input matrix
  out:   (N, M)   result

Usage:
  python3 gen_inputs.py [--m M] [--n N] [--seed SEED] [--out-dir DIR]
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",       type=int, default=640)
    parser.add_argument("--n",       type=int, default=500)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    M, N = args.m, args.n
    out_dir = Path(args.out_dir)

    rng = np.random.default_rng(args.seed)
    data0 = rng.uniform(-1.0, 1.0, (M, 1)).astype(np.float32).astype(np.float16)
    data1 = rng.uniform(-1.0, 1.0, (N, M)).astype(np.float32).astype(np.float16)

    # out[n, m] = relu(data0[m, 0]) + data1[n, m]
    # relu(data0[:,0]) has shape [M]; broadcast along n-axis → [N, M]
    relu_col = np.maximum(data0[:, 0].astype(np.float32), 0.0)  # [M]
    out = (relu_col[np.newaxis, :] + data1.astype(np.float32)).astype(np.float16)  # [N, M]

    np.save(out_dir / "input_data0.npy", data0)
    np.save(out_dir / "input_data1.npy", data1)
    np.save(out_dir / "output_expected.npy", out)

    print(f"input_data0:      {data0.shape} {data0.dtype}")
    print(f"input_data1:      {data1.shape} {data1.dtype}")
    print(f"output_expected:  {out.shape} {out.dtype}  range [{out.min():.4g}, {out.max():.4g}]")


if __name__ == "__main__":
    main()
