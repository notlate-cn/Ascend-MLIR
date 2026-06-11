import json
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))

from ascend_debug import ingest


def _make_workdir(tmp_path):
    wd = tmp_path / "wd"
    (wd / "groups").mkdir(parents=True)
    (wd / "groups" / "network.json").write_text(json.dumps({
        "function": "f",
        "inputs": [{"name": "a", "shape": [4], "dtype": "f16"}],
        "kernels": [{"id": "kernel_group0", "kind": "ascendc",
                     "args": [{"from": "input", "name": "a"}],
                     "results": [{"name": "r0", "shape": [4], "dtype": "f16"}]}],
        "outputs": [],
    }))
    # network lowering stage dumps (subset present, out of full set)
    (wd / "model_recognized.mlir").write_text("// recognized\nmodule {}\n")
    (wd / "model_unit_folded.mlir").write_text("// unit\nmodule {}\n")
    (wd / "_outlined_combined.mlir").write_text("// outlined\nmodule {}\n")
    return wd


def test_ingest_registers_present_network_stages(tmp_path):
    wd = _make_workdir(tmp_path)
    out = tmp_path / "run"
    ingest.ingest_run(types.SimpleNamespace(network_workdir=wd, out=out))
    m = json.loads((out / "manifest.json").read_text())
    names = [s["name"] for s in m["stages"]]
    # only the 3 present, in pipeline order, symbolized/transpose-folded absent
    assert names == ["recognized", "unit-folded", "outlined"]
    # each registered stage file was actually copied
    for s in m["stages"]:
        assert (out / s["path"]).exists()
    # orders strictly increasing
    orders = [s["order"] for s in m["stages"]]
    assert orders == sorted(orders) and len(set(orders)) == len(orders)


def test_ingest_fallback_when_no_named_stages(tmp_path):
    wd = tmp_path / "wd2"
    (wd / "groups").mkdir(parents=True)
    (wd / "groups" / "network.json").write_text(json.dumps({"kernels": []}))
    (wd / "groups" / "kernel_group0.mlir").write_text("module {}\n")
    out = tmp_path / "run2"
    ingest.ingest_run(types.SimpleNamespace(network_workdir=wd, out=out))
    m = json.loads((out / "manifest.json").read_text())
    assert len(m["stages"]) >= 1  # fell back to kernel_group0.mlir / network IR
