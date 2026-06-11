"""Inputs for autotune-multivar-bcast: out[m,n] = a[m,n] + b[n]."""
import argparse, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    od = args.out_dir
    os.makedirs(od, exist_ok=True)
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((256, 64)) * 0.5).astype(np.float16)
    b = (rng.standard_normal((64,)) * 0.5).astype(np.float16)
    init = np.zeros((256, 64), dtype=np.float16)
    expected = (a.astype(np.float32) +
                b.astype(np.float32)[None, :]).astype(np.float16)
    for name, arr in [("a", a), ("b", b), ("init", init), ("expected", expected)]:
        np.save(os.path.join(od, f"{name}.npy"), arr)
        print(f"{name}: {arr.shape}")


if __name__ == "__main__":
    main()
