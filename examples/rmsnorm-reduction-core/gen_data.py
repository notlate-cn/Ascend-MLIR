#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = rng.uniform(-1.0, 1.0, size=(args.m, args.n)).astype(np.float16)
    square = (x.astype(np.float32) * x.astype(np.float32)).astype(np.float16)
    sumsq = square.astype(np.float32).sum(axis=1).astype(np.float16)
    expected = (x.astype(np.float32) * sumsq[:, None].astype(np.float32)).astype(
        np.float16
    )

    np.save(out_dir / "input_x.npy", x)
    np.save(out_dir / "output_expected.npy", expected)

    print(f"M={args.m}, N={args.n}, seed={args.seed}")
    print(f"input_x: ({args.m}, {args.n}) float16")
    print(f"output:  ({args.m}, {args.n}) float16")


if __name__ == "__main__":
    main()
