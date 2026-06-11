"""End-to-end pytest for examples/two-elewise-e2e.

Exercises the auto-outline path: a single --input-linalg model.mlir
containing two independent linalg.generic ops should be outlined into
kernel_group0 + kernel_group1, then run through all 5 phases.

Requires CANN env (ASCEND_HOME_PATH + simulator LD_LIBRARY_PATH).
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SH = REPO / "examples/two-elewise-e2e/run.sh"


@pytest.mark.skipif(
    "ASCEND_HOME_PATH" not in os.environ,
    reason="ASCEND_HOME_PATH not set; two-elewise-e2e needs CANN env",
)
def test_two_elewise_e2e_runs_to_pass(tmp_path):
    """Run examples/two-elewise-e2e/run.sh end-to-end; expect exit 0 + 2 PASS lines."""
    env = os.environ.copy()
    env["WORK"] = str(tmp_path / "build_e2e")
    res = subprocess.run(
        ["bash", str(RUN_SH)],
        env=env, capture_output=True, text=True, timeout=900,
    )
    assert res.returncode == 0, (
        f"run.sh failed.\n--- stdout (tail) ---\n{res.stdout[-2000:]}\n"
        f"--- stderr (tail) ---\n{res.stderr[-2000:]}"
    )
    assert "network.output[0]" in res.stdout
    assert "network.output[1]" in res.stdout
    # Both outputs must PASS (not just one).
    assert res.stdout.count("PASS") >= 2, (
        f"expected 2 PASS lines, got:\n{res.stdout[-2000:]}"
    )

    work = Path(env["WORK"])
    assert (work / "outputs" / "out0.npy").exists()
    assert (work / "outputs" / "out1.npy").exists()
    inter = work / "intermediates_default"
    # Both kernels' intermediates must have been dumped
    assert (inter / "kernel_group0_in_0.npy").exists()
    assert (inter / "kernel_group0_out_0.npy").exists()
    assert (inter / "kernel_group1_in_0.npy").exists()
    assert (inter / "kernel_group1_out_0.npy").exists()
