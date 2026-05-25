#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def shell_int_default(script: str, name: str) -> int:
    match = re.search(rf"^{re.escape(name)}=([0-9]+)$", script, re.MULTILINE)
    if not match:
        fail(f"missing shell default: {name}")
    return int(match.group(1))


def python_int_constant(script: str, name: str) -> int:
    pattern = rf"^{re.escape(name)}\s*=\s*([0-9]+)"
    match = re.search(pattern, script, re.MULTILINE)
    if not match:
        fail(f"missing python constant: {name}")
    return int(match.group(1))


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        fail("usage: check_split_defaults.py RUN_MAINLINE GEN_DATA TILING_SPACE")

    run_mainline = Path(argv[1]).read_text(encoding="utf-8")
    gen_data = Path(argv[2]).read_text(encoding="utf-8")
    tiling_space = json.loads(Path(argv[3]).read_text(encoding="utf-8"))

    expected_m = 640
    expected_n = 512
    if shell_int_default(run_mainline, "M") != expected_m:
        fail("run-mainline M default drifted")
    if shell_int_default(run_mainline, "N") != expected_n:
        fail("run-mainline N default drifted")
    if python_int_constant(gen_data, "M_DEFAULT") != expected_m:
        fail("gen_data M default drifted")
    if python_int_constant(gen_data, "N_DEFAULT") != expected_n:
        fail("gen_data N default drifted")

    if tiling_space.get("block_dim_expr") != "ceil(HM / TB_M)":
        fail(f"unsupported block_dim_expr: {tiling_space.get('block_dim_expr')}")
    tb_m = next(
        (field for field in tiling_space.get("tiling_params", [])
         if field.get("name") == "TB_M"),
        None,
    )
    if tb_m is None:
        fail("missing TB_M tiling param")
    if int(tb_m.get("min", -1)) != 32 or int(tb_m.get("max", -1)) != 32:
        fail("split TB_M default must stay 32 for the real-NPU launch")

    if not re.search(r"^TILE_M=32$", run_mainline, re.MULTILINE):
        fail("run-mainline TILE_M default drifted")
    if not re.search(r'^BLOCK_DIM=""$', run_mainline, re.MULTILINE):
        fail("run-mainline BLOCK_DIM should be derived when not overridden")
    if "BLOCK_DIM=$(((HM + TILE_M - 1) / TILE_M))" not in run_mainline:
        fail("run-mainline BLOCK_DIM is not derived from HM/TILE_M")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
