#!/usr/bin/env python3
import json
import math
import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def shell_default(script: str, name: str) -> int:
    match = re.search(rf"^{re.escape(name)}=([0-9]+)$", script, re.MULTILINE)
    if not match:
        fail(f"missing shell default: {name}")
    return int(match.group(1))


def argparse_default(script: str, option: str) -> int:
    pattern = rf'add_argument\("{re.escape(option)}".*?default=([0-9]+)'
    match = re.search(pattern, script, re.DOTALL)
    if not match:
        fail(f"missing argparse default: {option}")
    return int(match.group(1))


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        fail("usage: check_gather_defaults.py RUN_MAINLINE GEN_DATA TILING_SPACE")

    run_mainline = Path(argv[1]).read_text(encoding="utf-8")
    gen_data = Path(argv[2]).read_text(encoding="utf-8")
    tiling_space = json.loads(Path(argv[3]).read_text(encoding="utf-8"))

    shapes = tiling_space["shapes"]
    expected = {"M": int(shapes["M"]), "N": int(shapes["N"]), "K": int(shapes["K"])}

    actual_run = {
        "M": shell_default(run_mainline, "M"),
        "N": shell_default(run_mainline, "N"),
        "K": shell_default(run_mainline, "K"),
    }
    if actual_run != expected:
        fail(f"run-mainline defaults drifted: expected {expected}, got {actual_run}")

    actual_gen = {
        "M": argparse_default(gen_data, "--m"),
        "N": argparse_default(gen_data, "--n"),
        "K": argparse_default(gen_data, "--k"),
    }
    if actual_gen != expected:
        fail(f"gen_data defaults drifted: expected {expected}, got {actual_gen}")

    if tiling_space.get("block_dim_expr") != "ceil(M / TB_M)":
        fail(f"unsupported block_dim_expr: {tiling_space.get('block_dim_expr')}")
    tb_m = next(
        (field for field in tiling_space.get("tiling_params", [])
         if field.get("name") == "TB_M"),
        None,
    )
    if tb_m is None:
        fail("missing TB_M tiling param")
    tb_m_values = tb_m.get("values")
    if tb_m_values:
        tile_m = int(tb_m_values[0])
    elif tb_m.get("min") == tb_m.get("max"):
        tile_m = int(tb_m["min"])
    else:
        fail("TB_M must have a fixed default for the gather real-NPU smoke")

    expected_block_dim = math.ceil(expected["M"] / tile_m)
    actual_block_dim = shell_default(run_mainline, "BLOCK_DIM")
    if actual_block_dim != expected_block_dim:
        fail(
            "run-mainline BLOCK_DIM drifted: "
            f"expected {expected_block_dim}, got {actual_block_dim}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
