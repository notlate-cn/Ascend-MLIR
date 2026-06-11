"""Inputs for autotune-stress-elewise (32×64 fp16, two independent elementwise)."""
import argparse, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    od = args.out_dir
    os.makedirs(od, exist_ok=True)
    rng = np.random.default_rng(0)
    shape = (32, 64)
    a = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    b = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    c = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    d = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    init0 = np.zeros(shape, dtype=np.float16)
    init1 = np.zeros(shape, dtype=np.float16)
    e0 = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    e1 = (c.astype(np.float32) * d.astype(np.float32)).astype(np.float16)
    for name, arr in [("a", a), ("b", b), ("c", c), ("d", d),
                       ("init0", init0), ("init1", init1),
                       ("expected0", e0), ("expected1", e1)]:
        np.save(os.path.join(od, f"{name}.npy"), arr)
        print(f"{name}: {arr.shape}")


if __name__ == "__main__":
    main()
