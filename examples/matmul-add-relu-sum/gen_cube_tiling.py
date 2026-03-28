#!/usr/bin/env python3
"""
gen_cube_tiling.py — Generate TCubeTiling bytes for fc_relu_mix.cpp.

Usage:
  python3 gen_cube_tiling.py --M 128 --K 64 --N 128 --block-dim 1 \
    --out-json tiling_space_mix.json [--print-tiling]

Outputs:
  tiling_space_mix.json — TCubeTiling-based tiling schema for validator.
  (Optionally prints tiling param string for --tiling-params.)
"""
import argparse
import json
import struct
import sys

# TCubeTiling field names in declaration order (int32_t each).
# From: aarch64-linux/asc/include/adv_api/kernel_tiling.h struct TCubeTiling
TCUBE_FIELDS = [
    "usedCoreNum", "M", "N", "Ka", "Kb",
    "singleCoreM", "singleCoreN", "singleCoreK",
    "baseM", "baseN", "baseK",
    "depthA1", "depthB1",
    "stepM", "stepN",
    "isBias", "transLength", "iterateOrder", "shareMode",
    "shareL1Size", "shareL0CSize", "shareUbSize",
    "batchM", "batchN", "singleBatchM", "singleBatchN",
    "stepKa", "stepKb",
    "depthAL1CacheUB", "depthBL1CacheUB",
    "dbL0A", "dbL0B", "dbL0C",
    "ALayoutInfoB", "ALayoutInfoS", "ALayoutInfoN", "ALayoutInfoG", "ALayoutInfoD",
    "BLayoutInfoB", "BLayoutInfoS", "BLayoutInfoN", "BLayoutInfoG", "BLayoutInfoD",
    "CLayoutInfoB", "CLayoutInfoS1", "CLayoutInfoN", "CLayoutInfoG", "CLayoutInfoS2",
    "BatchNum", "mxTypePara",
]

def compute_tiling(M: int, K: int, N: int, block_dim: int) -> dict:
    """Compute TCubeTiling fields for float32 matmul on 910B."""
    assert M % block_dim == 0, f"M={M} must be divisible by block_dim={block_dim}"
    single_m = M // block_dim
    base_m = min(64, single_m)
    base_n = min(128, N)
    base_k = min(64, K)
    base_m = max(16, (base_m // 16) * 16)
    base_n = max(16, (base_n // 16) * 16)
    base_k = max(64, (base_k // 64) * 64)

    t = {f: 0 for f in TCUBE_FIELDS}
    t["usedCoreNum"] = block_dim
    t["M"]           = M
    t["N"]           = N
    t["Ka"]          = K
    t["Kb"]          = K
    t["singleCoreM"] = single_m
    t["singleCoreN"] = N
    t["singleCoreK"] = K
    t["baseM"]       = base_m
    t["baseN"]       = base_n
    t["baseK"]       = base_k
    t["depthA1"]     = 1
    t["depthB1"]     = 1
    t["stepM"]       = 1
    t["stepN"]       = 1
    return t

def make_schema(tiling: dict) -> dict:
    """Build tiling_space_mix.json content."""
    params = []
    for name in TCUBE_FIELDS:
        params.append({
            "name": name,
            "type": "int32",
            "fixed": True,
            "value": tiling[name],
        })
    return {
        "kernel": "fc_relu",
        "kernel_file": "fc_relu_mix.cpp",
        "soc": "Ascend910B1",
        "block_dim_expr": "usedCoreNum",
        "tiling_params": params,
        "shapes": {
            "M": tiling["M"],
            "K": tiling["Ka"],
            "N": tiling["N"],
        },
    }

def make_tiling_params_str(tiling: dict) -> str:
    """Return comma-separated name=value string for --tiling-params."""
    return ",".join(f"{k}={v}" for k, v in tiling.items())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M",         type=int, required=True)
    ap.add_argument("--K",         type=int, required=True)
    ap.add_argument("--N",         type=int, required=True)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--out-json",  default="tiling_space_mix.json")
    ap.add_argument("--print-tiling", action="store_true",
                    help="Print tiling-params string to stdout")
    args = ap.parse_args()

    tiling = compute_tiling(args.M, args.K, args.N, args.block_dim)
    schema = make_schema(tiling)

    with open(args.out_json, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Wrote {args.out_json}")

    if args.print_tiling:
        print(make_tiling_params_str(tiling))

if __name__ == "__main__":
    main()
