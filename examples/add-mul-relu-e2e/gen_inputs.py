#!/usr/bin/env python3
"""Generate test inputs for add-mul-relu-e2e example.

Computation:
  out[d0,d1,d2] = max(a[d0,d1,d2] + b[d0,d1,d2] * c[d0,d1,d2], 0.0)
"""

import argparse
import numpy as np
import os

D0_DEFAULT, D1_DEFAULT, D2_DEFAULT = 4, 8, 32
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0", type=int, default=D0_DEFAULT)
    parser.add_argument("--d1", type=int, default=D1_DEFAULT)
    parser.add_argument("--d2", type=int, default=D2_DEFAULT)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    shape = (args.d0, args.d1, args.d2)
    rng = np.random.default_rng(SEED)

    a = (rng.random(shape, dtype=np.float32) * 4 - 2).astype(np.float32)
    b = (rng.random(shape, dtype=np.float32) * 4 - 2).astype(np.float32)
    c = (rng.random(shape, dtype=np.float32) * 4 - 2).astype(np.float32)
    inter1 = (b * c).astype(np.float32)
    inter2 = (a + inter1).astype(np.float32)
    expected = np.maximum(inter2, 0.0).astype(np.float32)

    d = args.outdir
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "a.npy"), a)
    np.save(os.path.join(d, "b.npy"), b)
    np.save(os.path.join(d, "c.npy"), c)
    np.save(os.path.join(d, "expected.npy"), expected)
    np.save(os.path.join(d, "expected_inter1.npy"), inter1)
    np.save(os.path.join(d, "expected_inter2.npy"), inter2)

    total = args.d0 * args.d1 * args.d2
    print(f"shape={shape}, total={total}, seed={SEED}")
    print(f"a:        [{a.min():.3f}, {a.max():.3f}]")
    print(f"b:        [{b.min():.3f}, {b.max():.3f}]")
    print(f"c:        [{c.min():.3f}, {c.max():.3f}]")
    print(f"expected: [{expected.min():.3f}, {expected.max():.3f}]  "
          f"({(expected > 0).mean():.1%} positive)")
    print("Done.")


if __name__ == "__main__":
    main()
