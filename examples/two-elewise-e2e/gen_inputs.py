"""Generate inputs and expected outputs for two-elewise-e2e.

The model has two independent elementwise chains (2 fused ops each):
  out0 = (a + b) * e   (kernel_group0)
  out1 = (c * d) + f   (kernel_group1)
Shape: 4x4 fp16 (matches the model.mlir).
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    od = args.out_dir
    os.makedirs(od, exist_ok=True)
    rng = np.random.default_rng(0)

    shape = (4, 4)
    a = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    b = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    c = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    d = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    e = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    f = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    init0 = np.zeros(shape, dtype=np.float16)
    init1 = np.zeros(shape, dtype=np.float16)

    expected0 = ((a.astype(np.float32) + b.astype(np.float32))
                 * e.astype(np.float32)).astype(np.float16)
    expected1 = ((c.astype(np.float32) * d.astype(np.float32))
                 + f.astype(np.float32)).astype(np.float16)

    for name, arr in [("a", a), ("b", b), ("c", c), ("d", d),
                      ("e", e), ("f", f),
                      ("init0", init0), ("init1", init1),
                      ("expected0", expected0), ("expected1", expected1)]:
        np.save(os.path.join(od, f"{name}.npy"), arr)
        print(f"{name}: {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    main()
