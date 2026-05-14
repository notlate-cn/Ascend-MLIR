"""Generate inputs for kg1-hang-repro.

Reference math (full H4 baseline):
  q'  = q*scale + bias                              (kernel_group0)
  fa  = softmax(q'·k^T/sqrt(d) + mask) · v          (aclnn host-mode CPU ref)
  out = fa + kg1_bias                               (kernel_group1, dynamic)

For H1 (middle replaced with memcpy): out = q' + kg1_bias (kg0_out passed through).
Caller selects which expected via --variant {h4,h1}.
"""
import argparse
import os

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x).astype(np.float32)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", default="h4", choices=["h4", "h1"])
    args = ap.parse_args()
    od = args.out_dir
    os.makedirs(od, exist_ok=True)
    rng = np.random.default_rng(0)

    shape = (1, 1, 2, 8)
    mask_shape = (1, 1, 2, 2)
    q = (rng.standard_normal(shape) * 0.1).astype(np.float16)
    k = (rng.standard_normal(shape) * 0.1).astype(np.float16)
    v = (rng.standard_normal(shape) * 0.1).astype(np.float16)
    scale = np.full(shape, 1.0, dtype=np.float16)
    bias  = np.zeros(shape, dtype=np.float16)
    mask  = np.zeros(mask_shape, dtype=np.float16)
    init_pre  = np.zeros(shape, dtype=np.float16)
    init_fa   = np.zeros(shape, dtype=np.float16)
    kg1_bias  = np.full(shape, 1.0, dtype=np.float16)
    init_post = np.zeros(shape, dtype=np.float16)

    # kernel_group0
    q_prime = (q.astype(np.float32) * scale.astype(np.float32)
               + bias.astype(np.float32)).astype(np.float16)
    # aclnn FA (host-mode CPU ref)
    d = q_prime.shape[-1]
    qf = q_prime.astype(np.float32); kf = k.astype(np.float32); vf = v.astype(np.float32)
    scores = np.einsum("bnqd,bnkd->bnqk", qf, kf) / np.sqrt(d) + mask.astype(np.float32)
    attn = softmax(scores, axis=-1)
    fa = np.einsum("bnqk,bnkd->bnqd", attn, vf).astype(np.float16)

    if args.variant == "h1":
        # H1: middle is memcpy(kg0_out → kg1_in), so kg1 input == q_prime.
        kg1_in = q_prime
    else:
        kg1_in = fa
    out = (kg1_in.astype(np.float32) + kg1_bias.astype(np.float32)).astype(np.float16)

    for name, arr in [("q", q), ("scale", scale), ("bias", bias),
                       ("k", k), ("v", v), ("mask", mask),
                       ("init_pre", init_pre), ("init_fa", init_fa),
                       ("kg1_bias", kg1_bias), ("init_post", init_post),
                       ("expected", out)]:
        np.save(os.path.join(od, f"{name}.npy"), arr)
        print(f"{name}: {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    main()
