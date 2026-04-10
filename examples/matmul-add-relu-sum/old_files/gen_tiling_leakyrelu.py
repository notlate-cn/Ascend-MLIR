#!/usr/bin/env python3
"""
gen_tiling_leakyrelu.py — Generate TCubeTiling for fc_leakyrelu_mix.cpp.

Mirrors baremix_custom_tiling.cpp:
  M=128, N=128, K=256, fp16 x fp16 -> fp32, bias fp32[N], baseM=128, baseN=128.

Usage:
  python3 gen_tiling_leakyrelu.py [--M M] [--K K] [--N N] [--block-dim 1]
      --out-json tiling_leakyrelu.json [--print-tiling]
"""
import argparse
import json

# TCubeTiling field names in declaration order (int32_t each).
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
    """Compute TCubeTiling fields for fp16 x fp16 -> fp32 matmul with bias on 910B1.

    Field values derived from MatmulApiTiling with:
      SetAType(GM, ND, FLOAT16), SetBType(GM, ND, FLOAT16),
      SetCType(GM, ND, FLOAT), SetBiasType(GM, ND, FLOAT),
      SetOrgShape/SetShape(128,128,256), SetBias(true),
      SetTraverse(FIRSTM), SetFixSplit(128,128,-1), SetBufferSpace(-1,-1,-1)
    on Ascend910B1. Non-trivial fields (shareL1Size, depthA1/B1, stepKa/Kb,
    dbL0A/B/C) must match what the tiling API produces for the kernel to
    correctly configure L1/L0 buffers at runtime.
    """
    assert M % block_dim == 0, f"M={M} must be divisible by block_dim={block_dim}"
    single_m = M // block_dim

    t = {f: 0 for f in TCUBE_FIELDS}
    t["usedCoreNum"]  = block_dim
    t["M"]            = M
    t["N"]            = N
    t["Ka"]           = K
    t["Kb"]           = K
    t["singleCoreM"]  = single_m
    t["singleCoreN"]  = N
    t["singleCoreK"]  = K
    t["baseM"]        = 128
    t["baseN"]        = 128
    t["baseK"]        = 128   # API outputs 128 for K=256
    t["depthA1"]      = 2     # from API: double-buffered L1
    t["depthB1"]      = 2
    t["stepM"]        = 1
    t["stepN"]        = 1
    t["isBias"]       = 1
    t["shareL1Size"]  = 131584  # from API (bytes)
    t["shareL0CSize"] = 65536   # from API (bytes)
    t["stepKa"]       = 2       # from API
    t["stepKb"]       = 2
    t["dbL0A"]        = 2       # from API: L0A double-buffer
    t["dbL0B"]        = 2
    t["dbL0C"]        = 1
    t["batchM"]       = 1
    t["batchN"]       = 1
    t["singleBatchM"] = 1
    t["singleBatchN"] = 1
    return t


def make_schema(tiling: dict, kernel_name: str) -> dict:
    params = []
    for name in TCUBE_FIELDS:
        params.append({
            "name": name,
            "type": "int32",
            "fixed": True,
            "value": tiling[name],
        })
    return {
        "kernel": kernel_name,
        "kernel_file": "fc_leakyrelu_mix.cpp",
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
    return ",".join(f"{k}={tiling[k]}" for k in TCUBE_FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M",          type=int, default=128)
    ap.add_argument("--K",          type=int, default=256)
    ap.add_argument("--N",          type=int, default=128)
    ap.add_argument("--block-dim",  type=int, default=1)
    ap.add_argument("--kernel-name", default="fc_leakyrelu")
    ap.add_argument("--out-json",   default="tiling_leakyrelu.json")
    ap.add_argument("--print-tiling", action="store_true")
    args = ap.parse_args()

    tiling = compute_tiling(args.M, args.K, args.N, args.block_dim)
    schema = make_schema(tiling, args.kernel_name)

    with open(args.out_json, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Wrote {args.out_json}")

    if args.print_tiling:
        print(make_tiling_params_str(tiling))


if __name__ == "__main__":
    main()
