#!/usr/bin/env python3
"""Generate NPU microcase input and golden .npy files.

This intentionally uses only the Python standard library so the xvm/remote
preparation path does not depend on NumPy being installed before runtime
verification begins.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def write_f16_npy(path: Path, shape: tuple[int, ...], values: list[float]) -> None:
    if len(values) != num_elements(shape):
        raise ValueError(f"{path}: value count does not match shape")
    path.parent.mkdir(parents=True, exist_ok=True)
    shape_text = f"({shape[0]},)" if len(shape) == 1 else str(shape)
    header = {
        "descr": "'<f2'",
        "fortran_order": "False",
        "shape": shape_text,
    }
    header_text = (
        "{'descr': "
        + header["descr"]
        + ", 'fortran_order': "
        + header["fortran_order"]
        + ", 'shape': "
        + header["shape"]
        + ", }"
    )
    prefix_len = 10
    header_len = len(header_text) + 1
    padding = (16 - ((prefix_len + header_len) % 16)) % 16
    full_header = (header_text + " " * padding + "\n").encode("latin1")
    with path.open("wb") as f:
        f.write(b"\x93NUMPY")
        f.write(bytes([1, 0]))
        f.write(struct.pack("<H", len(full_header)))
        f.write(full_header)
        for value in values:
            f.write(struct.pack("<e", value))


def write_i64_tiling(path: Path, values: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for value in values:
            f.write(struct.pack("<q", value))


def num_elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def round_f16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def write_case_arrays(root: Path) -> None:
    const_dir = root / "const640"
    const_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(const_dir / "expected.npy", (640,), [1.0] * 640)

    copy_dir = root / "copy640"
    copy_dir.mkdir(parents=True, exist_ok=True)
    copy_input = [i / 17.0 for i in range(640)]
    write_f16_npy(copy_dir / "input.npy", (640,), copy_input)
    write_f16_npy(copy_dir / "expected.npy", (640,), copy_input)

    copy_tbuf_dir = root / "copy_tbuf640"
    copy_tbuf_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(copy_tbuf_dir / "input.npy", (640,), copy_input)
    write_f16_npy(copy_tbuf_dir / "expected.npy", (640,), copy_input)

    copy_scalar_dir = root / "copy_scalar640"
    copy_scalar_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(copy_scalar_dir / "input.npy", (640,), copy_input)
    write_f16_npy(copy_scalar_dir / "expected.npy", (640,), copy_input)

    copy_params_dir = root / "copy_params640"
    copy_params_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(copy_params_dir / "input.npy", (640,), copy_input)
    write_f16_npy(copy_params_dir / "expected.npy", (640,), copy_input)

    copy_wait_dir = root / "copy_wait640"
    copy_wait_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(copy_wait_dir / "input.npy", (640,), copy_input)
    write_f16_npy(copy_wait_dir / "expected.npy", (640,), copy_input)

    const_with_input_dir = root / "const_with_input640"
    const_with_input_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(const_with_input_dir / "input.npy", (640,), copy_input)
    write_f16_npy(const_with_input_dir / "expected.npy", (640,), [2.0] * 640)

    relu_dir = root / "relu_only"
    relu_dir.mkdir(parents=True, exist_ok=True)
    relu_input = [-8.0 + (16.0 * i / 639.0) for i in range(640)]
    write_f16_npy(relu_dir / "input.npy", (640,), relu_input)
    write_f16_npy(
        relu_dir / "expected.npy", (640,), [max(x, 0.0) for x in relu_input]
    )

    broadcast_dir = root / "broadcast_add"
    broadcast_dir.mkdir(parents=True, exist_ok=True)
    base = [i / 64.0 for i in range(640)]
    bias = [i / 128.0 for i in range(1280)]
    expected = [base[i % 640] + bias[i] for i in range(1280)]
    write_f16_npy(broadcast_dir / "input.npy", (1, 640), base)
    write_f16_npy(broadcast_dir / "bias.npy", (2, 640), bias)
    write_f16_npy(broadcast_dir / "expected.npy", (2, 640), expected)

    relu_diag_data0 = [
        round_f16(-1.0 + (2.0 * (i % 640) / 639.0)) for i in range(640)
    ]
    relu_diag_data1 = [
        round_f16(((r * 17 + c * 3) % 257) / 128.0 - 1.0)
        for r in range(500)
        for c in range(640)
    ]
    relu_diag_broadcast = [
        max(relu_diag_data0[c], 0.0) for _r in range(500) for c in range(640)
    ]
    relu_diag_full = [
        round_f16(max(relu_diag_data0[c], 0.0) + relu_diag_data1[r * 640 + c])
        for r in range(500)
        for c in range(640)
    ]

    relu_broadcast_dir = root / "relu_diag_broadcast_store"
    relu_broadcast_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(relu_broadcast_dir / "input_data0.npy", (640,), relu_diag_data0)
    write_f16_npy(
        relu_broadcast_dir / "expected.npy", (500, 640), relu_diag_broadcast
    )

    relu_strided_dir = root / "relu_diag_strided_copy"
    relu_strided_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(
        relu_strided_dir / "input_data1.npy", (500, 640), relu_diag_data1
    )
    write_f16_npy(
        relu_strided_dir / "expected.npy", (500, 640), relu_diag_data1
    )

    relu_full_dir = root / "relu_diag_broadcast_add"
    relu_full_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(relu_full_dir / "input_data0.npy", (640,), relu_diag_data0)
    write_f16_npy(relu_full_dir / "input_data1.npy", (500, 640), relu_diag_data1)
    write_f16_npy(relu_full_dir / "expected.npy", (500, 640), relu_diag_full)

    relu_generated_dir = root / "relu_diag_generated_buffers"
    relu_generated_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(
        relu_generated_dir / "input_data0.npy", (640,), relu_diag_data0
    )
    write_f16_npy(
        relu_generated_dir / "input_data1.npy", (500, 640), relu_diag_data1
    )
    write_f16_npy(
        relu_generated_dir / "expected.npy", (500, 640), relu_diag_full
    )

    relu_tiling_dir = root / "relu_diag_tiling_abi"
    relu_tiling_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(relu_tiling_dir / "input_data0.npy", (640,), relu_diag_data0)
    write_f16_npy(relu_tiling_dir / "input_data1.npy", (500, 640), relu_diag_data1)
    write_f16_npy(relu_tiling_dir / "expected.npy", (500, 640), relu_diag_full)
    write_i64_tiling(relu_tiling_dir / "tiling.bin", (32, 32, 640, 500, 640, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    write_case_arrays(Path(args.out_dir))


if __name__ == "__main__":
    main()
