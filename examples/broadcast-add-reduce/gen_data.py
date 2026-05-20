"""Generate input/expected-output npy files for broadcast-add-reduce.

Computation: c[m] = sum_n( a[m] + b[m, n] )
  a: (M,)    broadcast row vector
  b: (M, N)  2-D input matrix
  c: (M,)    reduce-sum result

Usage:
  python3 gen_data.py [--m M] [--n N] [--seed SEED] [--out-dir DIR]
"""

import argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",       type=int, default=640)
    parser.add_argument("--n",       type=int, default=512)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    M, N = args.m, args.n
    out_dir = Path(args.out_dir)

    rng = np.random.default_rng(args.seed)
    a = rng.uniform(-1.0, 1.0, (M,)).astype(np.float32).astype(np.float16)
    b = rng.uniform(-1.0, 1.0, (M, N)).astype(np.float32).astype(np.float16)

    # Expected output: float32 accumulation then cast to half, matching the
    # generated f16 reduction path, which reduces through f32 intermediates.
    c = (a[:, None].astype(np.float32) + b.astype(np.float32)).sum(axis=1).astype(np.float16)

    np.save(out_dir / "input_a.npy",  a)
    np.save(out_dir / "input_b.npy",  b)
    np.save(out_dir / "output_c.npy", c)

    print(f"input_a:  {a.shape} {a.dtype}")
    print(f"input_b:  {b.shape} {b.dtype}")
    print(f"output_c: {c.shape} {c.dtype}  range [{c.min():.4g}, {c.max():.4g}]")

if __name__ == "__main__":
    main()
