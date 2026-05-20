#!/usr/bin/env python3
"""Generate data for real-NPU multi-kernel scheduling examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MICROCASE_DIR = Path(__file__).resolve().parents[1] / "real-npu-microcases"
sys.path.insert(0, str(MICROCASE_DIR))

from gen_data import write_f16_npy  # noqa: E402


def write_arrays(root: Path) -> None:
    serial_dir = root / "serial-two-kernel"
    serial_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(serial_dir / "expected.npy", (640,), [1.0] * 640)

    fork_dir = root / "fork-join"
    fork_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(fork_dir / "expected.npy", (640,), [3.0] * 640)

    producer_b_dir = root / "data" / "producer_b"
    producer_b_dir.mkdir(parents=True, exist_ok=True)
    write_f16_npy(
        producer_b_dir / "input.npy",
        (640,),
        [i / 17.0 for i in range(640)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    write_arrays(Path(args.out_dir))


if __name__ == "__main__":
    main()
