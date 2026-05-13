"""Generate inputs and expected.npy for mixed-attn-e2e.

Reference math mirrors the host-mode FlashAttention in lib/Runtime/AclnnOps.cpp:
  q' = q*scale + bias                              (kernel_group0)
  out = softmax(q'·k^T/sqrt(d) + mask) · v         (aclnn host-mode CPU ref)
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
    init_pre = np.zeros(shape, dtype=np.float16)
    init_fa  = np.zeros(shape, dtype=np.float16)

    # kernel_group0: q' = q*scale + bias  (with scale=1, bias=0 → q' == q)
    q_prime = (q.astype(np.float32) * scale.astype(np.float32)
               + bias.astype(np.float32)).astype(np.float16)
    # FlashAttentionScore reference (BNSD): out = softmax(qk^T/sqrt(d)+mask) @ v
    d = q_prime.shape[-1]
    qf = q_prime.astype(np.float32); kf = k.astype(np.float32); vf = v.astype(np.float32)
    scores = np.einsum("bnqd,bnkd->bnqk", qf, kf) / np.sqrt(d) + mask.astype(np.float32)
    attn = softmax(scores, axis=-1)
    out = np.einsum("bnqk,bnkd->bnqd", attn, vf).astype(np.float16)

    for name, arr in [("q", q), ("scale", scale), ("bias", bias),
                       ("k", k), ("v", v), ("mask", mask),
                       ("init_pre", init_pre), ("init_fa", init_fa),
                       ("expected", out)]:
        np.save(os.path.join(od, f"{name}.npy"), arr)
        print(f"{name}: {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    main()
