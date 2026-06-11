"""Phase 5 test: best-tilings build + final run + verify against expected."""
import os, subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
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


@pytest.mark.skipif("ASCEND_HOME_PATH" not in os.environ, reason="needs CANN env")
def test_phase5_twochain_end_to_end(tmp_path):
    """Run twochain through all 5 phases; final output should match the
    numpy reference (add for kernel_group0, mul for kernel_group1)."""
    # Use 4x4 = 16 elements — exactly one minimum tile (XBLOCK_min = 16).
    # With block_dim=1 (block_dim_expr is empty in the generated tiling space
    # for this flat-parallel pattern), block 0 covers elements [0, XBLOCK),
    # which equals all 16 elements when total == XBLOCK.  Larger tensors would
    # need block_dim > 1 (a known limitation of TilePlanGen, out of T11 scope).
    # 4x4 avoids unit-extent dims (which would be collapsed by
    # --linalg-fold-unit-extent-dims and emit unsupported tensor.collapse_shape).
    rng = np.random.default_rng(42)
    a = rng.standard_normal((128, 64)).astype(np.float16) * 0.1
    b = rng.standard_normal((128, 64)).astype(np.float16) * 0.1
    c = rng.standard_normal((128, 64)).astype(np.float16) * 0.1
    d = rng.standard_normal((128, 64)).astype(np.float16) * 0.1
    i0 = np.zeros((128, 64), dtype=np.float16)
    i1 = np.zeros((128, 64), dtype=np.float16)
    # Compute reference in f16 to match kernel precision (kernel operates in f16).
    exp0 = (a + b).astype(np.float16)
    exp1 = (c * d).astype(np.float16)

    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    for name, arr in [("a", a), ("b", b), ("c", c), ("d", d), ("i0", i0), ("i1", i1)]:
        np.save(inputs_dir / f"{name}.npy", arr)
    np.save(inputs_dir / "exp0.npy", exp0)
    np.save(inputs_dir / "exp1.npy", exp1)

    model = tmp_path / "model.mlir"
    model.write_text(TWOCHAIN_MLIR)

    work = tmp_path / "work"
    res = subprocess.run(
        [
            "python3", str(RUNNER),
            "--input-linalg", str(model),
            "--inputs",
                str(inputs_dir / "a.npy"),  str(inputs_dir / "b.npy"),
                str(inputs_dir / "c.npy"),  str(inputs_dir / "d.npy"),
                str(inputs_dir / "i0.npy"), str(inputs_dir / "i1.npy"),
            "--expected",
                str(inputs_dir / "exp0.npy"), str(inputs_dir / "exp1.npy"),
            "--workdir", str(work),
            "--atol", "1e-3", "--rtol", "1e-2",
        ],
        capture_output=True, text=True, timeout=600,
    )
    assert res.returncode == 0, (
        f"runner failed.\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )

    # Phase 5 artifacts
    assert (work / "network_host.cpp").exists()
    assert (work / "network_test").exists()
    assert (work / "outputs" / "out0.npy").exists()
    assert (work / "outputs" / "out1.npy").exists()

    # Numerical check (independent of the runner's own verdict)
    got0 = np.load(work / "outputs" / "out0.npy")
    got1 = np.load(work / "outputs" / "out1.npy")
    assert np.allclose(got0.astype(np.float32), exp0.astype(np.float32),
                       atol=1e-3, rtol=1e-2), f"max_diff={np.max(np.abs(got0.astype(np.float32) - exp0.astype(np.float32)))}"
    assert np.allclose(got1.astype(np.float32), exp1.astype(np.float32),
                       atol=1e-3, rtol=1e-2), f"max_diff={np.max(np.abs(got1.astype(np.float32) - exp1.astype(np.float32)))}"

    # The runner's stdout should contain the PASS line
    assert "network.output[0]" in res.stdout
    assert "network.output[1]" in res.stdout
    assert "PASS" in res.stdout
