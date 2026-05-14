"""Phase 3 test: default-tilings build + g++ link + sim run + dump intermediates.

Run with:
    source examples/env.sh   # sets ASCEND_HOME_PATH + simulator LD_LIBRARY_PATH
    PYTHONPATH=python python3 -m pytest python/tests/test_network_runner_phase3.py -v

Phase 3 requires:
  - ASCEND_HOME_PATH for the CANN include/lib paths.
  - Simulator LD_LIBRARY_PATH (libruntime_camodel.so) for the actual sim run.
    Without it, the compile+link step still passes; only the run step fails.
"""
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
    """Write six 128x64 float16 zero tensors and return their paths."""
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
    reason="ASCEND_HOME_PATH not set; phase 3 needs CANN env",
)
def test_phase3_default_build_links(tmp_path):
    """Phase 3 should compile network_host_default.cpp + link to a runnable binary."""
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
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, (
        f"network_runner failed.\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )

    # Phase 3 artifacts
    assert (work / "tilings_default.json").exists(), "missing tilings_default.json"
    assert (work / "network_host_default.cpp").exists(), "missing network_host_default.cpp"
    assert (work / "network_test_default").exists(), "missing linked binary"

    # Sim run should have written intermediates
    inter = work / "intermediates_default"
    assert inter.is_dir(), f"intermediates dir not created: {inter}"
    npys = list(inter.glob("*.npy"))
    assert npys, f"no .npy files in {inter}; listing: {list(inter.iterdir())}"

    # Each kernel should have at least one output npy
    assert any("kernel_group0" in f.name for f in npys), \
        f"no kernel_group0 npy in {inter}"
    assert any("kernel_group1" in f.name for f in npys), \
        f"no kernel_group1 npy in {inter}"


@pytest.mark.skipif(
    "ASCEND_HOME_PATH" not in os.environ,
    reason="ASCEND_HOME_PATH not set; phase 3 needs CANN env",
)
def test_phase3_tilings_default_json_structure(tmp_path):
    """tilings_default.json should have _block_dim + tunable params per kernel."""
    import json

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
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr

    td = json.loads((work / "tilings_default.json").read_text())
    for kid in ("kernel_group0", "kernel_group1"):
        assert kid in td, f"{kid} missing from tilings_default.json"
        entry = td[kid]
        assert "_block_dim" in entry, f"_block_dim missing for {kid}"
        assert entry["_block_dim"] >= 1, f"_block_dim < 1 for {kid}"
        # Should have at least one tunable param (XBLOCK)
        non_bd = {k: v for k, v in entry.items() if k != "_block_dim"}
        assert non_bd, f"no tunable params for {kid} in tilings_default.json"
