"""End-to-end pytest for examples/mixed-attn-e2e.

Runs the full 5-phase network_runner pipeline through a mixed
AscendC + aclnn network (kernel_group0 → __aclnn_flash_attention).
Tests both that the runner exits 0 and that the dumped intermediates + final
output exist.

Requires CANN env (ASCEND_HOME_PATH + simulator LD_LIBRARY_PATH).
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_SH = REPO / "examples/mixed-attn-e2e/run.sh"


@pytest.mark.skipif(
    "ASCEND_HOME_PATH" not in os.environ,
    reason="ASCEND_HOME_PATH not set; mixed-attn-e2e needs CANN env",
)
def test_mixed_attn_e2e_runs_to_pass(tmp_path):
    """Run examples/mixed-attn-e2e/run.sh end-to-end; expect exit 0 + PASS line."""
    env = os.environ.copy()
    # Drive WORK to a tmp dir so we don't trample the example's build_e2e.
    env["WORK"] = str(tmp_path / "build_e2e")
    res = subprocess.run(
        ["bash", str(RUN_SH)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert res.returncode == 0, (
        f"run.sh failed.\n--- stdout (tail) ---\n{res.stdout[-2000:]}\n"
        f"--- stderr (tail) ---\n{res.stderr[-2000:]}"
    )
    assert "network.output[0]" in res.stdout
    assert "PASS" in res.stdout

    work = Path(env["WORK"])
    assert (work / "outputs" / "out0.npy").exists()
    # kernel_group0 ran on the sim and intermediates were dumped
    inter = work / "intermediates_default"
    assert (inter / "kernel_group0_in_0.npy").exists()
    assert (inter / "kernel_group0_out_0.npy").exists()
