#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = rng.uniform(-1.0, 1.0, size=(args.n,)).astype(np.float16)
    b = rng.uniform(0.0, 2.0, size=(args.n,)).astype(np.float16)
    c = rng.uniform(0.5, 1.5, size=(args.n,)).astype(np.float16)
    expected = ((a + b) * c).astype(np.float16)

    np.save(out_dir / "input_a.npy", a)
    np.save(out_dir / "input_b.npy", b)
    np.save(out_dir / "input_c.npy", c)
    np.save(out_dir / "output_expected.npy", expected)

    print(f"N={args.n}, seed={args.seed}")
    print(f"input_a: ({args.n},) float16")
    print(f"input_b: ({args.n},) float16")
    print(f"input_c: ({args.n},) float16")
    print(f"output:  ({args.n},) float16")


if __name__ == "__main__":
    main()
