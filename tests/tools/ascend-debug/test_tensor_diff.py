import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))
from ascend_debug import ingest


def _mk(tmp_path):
    wd = tmp_path / "wd"
    (wd / "outputs").mkdir(parents=True)
    # output0 matches expected0; output1 diverges beyond atol
    np.save(wd / "outputs" / "out0.npy", np.zeros((4,), dtype=np.float32))
    np.save(wd / "expected0.npy",        np.zeros((4,), dtype=np.float32))
    np.save(wd / "outputs" / "out1.npy", np.ones((4,), dtype=np.float32))
    np.save(wd / "expected1.npy",        np.zeros((4,), dtype=np.float32))
    return wd


def test_build_tensor_diff(tmp_path):
    wd = _mk(tmp_path)
    run = tmp_path / "run"; (run / "summaries").mkdir(parents=True)
    ok = ingest._build_tensor_diff(wd, run, atol=1e-2, rtol=1e-2)
    assert ok is True
    td = json.loads((run / "summaries" / "tensor_diff.json").read_text())
    assert td["comparison_count"] == 2
    assert td["failed_count"] == 1
    assert td["status"] == "FAIL"
    by_id = {c["id"]: c for c in td["comparisons"]}
    assert by_id["network.output[0]"]["status"] == "PASS"
    assert by_id["network.output[1]"]["status"] == "FAIL"
    assert by_id["network.output[1]"]["max_abs_error"] == 1.0


def test_build_tensor_diff_no_outputs(tmp_path):
    wd = tmp_path / "wd2"; (wd).mkdir()
    run = tmp_path / "run2"; run.mkdir()
    # no outputs/ → returns False, writes nothing
    assert ingest._build_tensor_diff(wd, run, atol=1e-2, rtol=1e-2) is False
    assert not (run / "summaries" / "tensor_diff.json").exists()
