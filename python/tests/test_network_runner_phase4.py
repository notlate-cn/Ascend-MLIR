"""Phase 4 test: per-kernel autotune via existing autotuner.

Run with:
    source /home/gser/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH="..."   # see task 10 instructions
    PYTHONPATH=python python3 -m pytest python/tests/test_network_runner_phase4.py -v

Phase 4 requires:
  - ASCEND_HOME_PATH for CANN include/lib paths.
  - Simulator LD_LIBRARY_PATH (libruntime_camodel.so) for the sim run.
  - build/bin/autotuner to be present.

Note: autotuner runs the simulator once per tiling combination per kernel,
so this test may take several minutes on the twochain model.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO   = Path(__file__).resolve().parents[2]
RUNNER = REPO / "python/network_runner.py"

TWOCHAIN_MLIR = """\
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @twochain(%a: tensor<?x?xf16>, %b: tensor<?x?xf16>,
                    %c: tensor<?x?xf16>, %d: tensor<?x?xf16>,
                    %i0: tensor<?x?xf16>, %i1: tensor<?x?xf16>)
    -> (tensor<?x?xf16>, tensor<?x?xf16>) {
  %x = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel","parallel"]}
       ins(%a, %b : tensor<?x?xf16>, tensor<?x?xf16>) outs(%i0 : tensor<?x?xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<?x?xf16>
  %y = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel","parallel"]}
       ins(%c, %d : tensor<?x?xf16>, tensor<?x?xf16>) outs(%i1 : tensor<?x?xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<?x?xf16>
  return %x, %y : tensor<?x?xf16>, tensor<?x?xf16>
}
"""


def _make_inputs(inputs_dir: Path) -> list[str]:
    paths = []
    for i in range(6):
        a = np.zeros((128, 64), dtype=np.float16)
        p = inputs_dir / f"in{i}.npy"
        np.save(p, a)
        paths.append(str(p))
    return paths


def _make_expected(inputs_dir: Path) -> list[str]:
    paths = []
    for i in range(2):
        e = np.zeros((128, 64), dtype=np.float16)
        p = inputs_dir / f"exp{i}.npy"
        np.save(p, e)
        paths.append(str(p))
    return paths


@pytest.mark.skipif(
    "ASCEND_HOME_PATH" not in os.environ,
    reason="ASCEND_HOME_PATH not set; phase 4 needs CANN env",
)
def test_phase4_autotune_twochain(tmp_path):
    """Phase 4 should produce per-kernel best JSON and aggregated tilings_best.json."""
    model = tmp_path / "model.mlir"
    model.write_text(TWOCHAIN_MLIR)

    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    in_paths = _make_inputs(inputs_dir)
    exp_paths = _make_expected(inputs_dir)

    work = tmp_path / "work"

    res = subprocess.run(
        [
            "python3", str(RUNNER),
            "--input-linalg", str(model),
            "--inputs",   *in_paths,
            "--expected", *exp_paths,
            "--workdir",  str(work),
            # max-phase defaults to 4, so no need to pass it explicitly
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, (
        f"network_runner failed.\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )

    # Per-kernel best JSON files
    for kid in ("kernel_group0", "kernel_group1"):
        best_path = work / f"{kid}_best.json"
        assert best_path.exists(), f"missing {best_path}"
        bc = json.loads(best_path.read_text())

        config = bc.get("config", {})
        assert "XBLOCK" in config, \
            f"{kid}_best.json missing config.XBLOCK; config={config}"
        assert "XBLOCK_SUB" in config, \
            f"{kid}_best.json missing config.XBLOCK_SUB; config={config}"

        best = bc.get("best", {})
        assert int(best.get("block_dim", 0)) >= 1, \
            f"{kid}_best.json best.block_dim < 1; best={best}"

    # Aggregated tilings_best.json
    tilings_path = work / "tilings_best.json"
    assert tilings_path.exists(), "missing tilings_best.json"
    tb = json.loads(tilings_path.read_text())

    assert set(tb.keys()) >= {"kernel_group0", "kernel_group1"}, \
        f"tilings_best.json missing kernels; keys={list(tb.keys())}"

    for kid in ("kernel_group0", "kernel_group1"):
        entry = tb[kid]
        assert "XBLOCK" in entry, \
            f"tilings_best.json[{kid}] missing XBLOCK; entry={entry}"
        assert "XBLOCK_SUB" in entry, \
            f"tilings_best.json[{kid}] missing XBLOCK_SUB; entry={entry}"
        assert "_block_dim" in entry, \
            f"tilings_best.json[{kid}] missing _block_dim; entry={entry}"
        assert int(entry["_block_dim"]) >= 1, \
            f"tilings_best.json[{kid}] _block_dim < 1; entry={entry}"
