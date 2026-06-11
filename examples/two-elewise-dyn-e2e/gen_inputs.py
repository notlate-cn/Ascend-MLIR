"""Generate inputs and expected outputs for two-elewise-dyn-e2e.

Dynamic-shape twin of two-elewise-e2e. The model has two independent
elementwise chains (2 fused ops each) over fully dynamic 3D tensors:
  out0 = (a + b) * e   (kernel_group0)
  out1 = (c * d) + f   (kernel_group1)
The concrete shape here (default 2x3x8) is arbitrary — a passing run proves
the axis extents are resolved at runtime from the input npy shapes, not baked
into the kernel.
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--d0", type=int, default=2)
    ap.add_argument("--d1", type=int, default=3)
    ap.add_argument("--d2", type=int, default=8)
    args = ap.parse_args()
    od = args.out_dir
    os.makedirs(od, exist_ok=True)
    rng = np.random.default_rng(0)

    shape = (args.d0, args.d1, args.d2)
    a = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    b = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    c = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    d = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    e = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    f = (rng.standard_normal(shape) * 0.5).astype(np.float16)

    expected0 = ((a.astype(np.float32) + b.astype(np.float32))
                 * e.astype(np.float32)).astype(np.float16)
    expected1 = ((c.astype(np.float32) * d.astype(np.float32))
                 + f.astype(np.float32)).astype(np.float16)

    for name, arr in [("a", a), ("b", b), ("c", c), ("d", d),
                      ("e", e), ("f", f),
                      ("expected0", expected0), ("expected1", expected1)]:
        np.save(os.path.join(od, f"{name}.npy"), arr)
        print(f"{name}: {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    main()
