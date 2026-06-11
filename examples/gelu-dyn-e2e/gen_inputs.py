"""Generate inputs + expected output for gelu-dyn-e2e.

The model is `gelu(x + b)` on `tensor<1x?x3072xf32>` — the seq dim is DYNAMIC
(torch.export Dim). The concrete inputs here use seq=48 (deliberately != the
GPT-2 static 64) so a passing run proves the seq extent is resolved at runtime
from the input npy shape, not baked into the kernel.

erf-GELU, fp32: out = (x+b) * 0.5 * (1 + erf((x+b) / sqrt(2))).
"""
import argparse
import os

import numpy as np

try:
    from scipy.special import erf  # type: ignore
except ImportError:
    import math
    erf = np.vectorize(math.erf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq", type=int, default=48)
    args = ap.parse_args()
    od = args.out_dir
    os.makedirs(od, exist_ok=True)
    rng = np.random.default_rng(0)

    x = (rng.standard_normal((1, args.seq, 3072)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((3072,)) * 0.5).astype(np.float32)

    h = x + b
    expected = (h * 0.5 * (1.0 + erf(h / np.sqrt(2.0)))).astype(np.float32)

    for name, arr in [("x", x), ("b", b), ("expected", expected)]:
        np.save(os.path.join(od, f"{name}.npy"), arr)
        print(f"{name}: {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    main()
