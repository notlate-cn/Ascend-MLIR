import json, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "python/network_runner.py"


def test_phase1_input_network_mixed(tmp_path):
    """Hand-written mixed network.mlir → runner copies, runs emit-network-json,
    produces network.json with both ascendc and aclnn kinds."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "network.mlir").write_text("""
module {
  func.func private @kernel_group0(tensor<8xf16>) -> tensor<8xf16>
  func.func private @__aclnn_softmax(tensor<8xf16>) -> tensor<8xf16>
      attributes {aclnn.op = "Softmax"}
  func.func @model(%x: tensor<8xf16>) -> tensor<8xf16> {
    %a = call @kernel_group0(%x) : (tensor<8xf16>) -> tensor<8xf16>
    %s = call @__aclnn_softmax(%a) : (tensor<8xf16>) -> tensor<8xf16>
    return %s : tensor<8xf16>
  }
}
""")
    # placeholder per-kernel .mlir (not exercised in phase 1)
    (src / "kernel_group0.mlir").write_text("module {}\n")

    work = tmp_path / "work"
    res = subprocess.run([
        "python3", str(RUNNER),
        "--input-network", str(src),
        "--inputs", "/dev/null",
        "--expected", "/dev/null",
        "--workdir", str(work),
        "--max-phase", "1",
    ], capture_output=True, text=True)
    assert res.returncode == 0, f"stderr: {res.stderr}"

    nj = work / "groups" / "network.json"
    assert nj.exists(), f"expected {nj}"
    d = json.loads(nj.read_text())
    assert d["function"] == "model"
    kinds = [k["kind"] for k in d["kernels"]]
    assert kinds == ["ascendc", "aclnn"], kinds


def test_phase1_input_linalg_two_chain(tmp_path):
    """Linalg model.mlir → runner runs outline → produces all-ascendc network.json."""
    model = tmp_path / "model.mlir"
    model.write_text("""
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @twochain(%a: tensor<128x64xf16>, %b: tensor<128x64xf16>,
                    %c: tensor<128x64xf16>, %d: tensor<128x64xf16>,
                    %i0: tensor<128x64xf16>, %i1: tensor<128x64xf16>)
    -> (tensor<128x64xf16>, tensor<128x64xf16>) {
  %x = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel","parallel"]}
       ins(%a, %b : tensor<128x64xf16>, tensor<128x64xf16>) outs(%i0 : tensor<128x64xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.addf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<128x64xf16>
  %y = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel","parallel"]}
       ins(%c, %d : tensor<128x64xf16>, tensor<128x64xf16>) outs(%i1 : tensor<128x64xf16>) {
  ^bb0(%p: f16, %q: f16, %o: f16):
    %v = arith.mulf %p, %q : f16
    linalg.yield %v : f16
  } -> tensor<128x64xf16>
  return %x, %y : tensor<128x64xf16>, tensor<128x64xf16>
}
""")
    work = tmp_path / "work"
    res = subprocess.run([
        "python3", str(RUNNER),
        "--input-linalg", str(model),
        "--inputs", "/dev/null",
        "--expected", "/dev/null",
        "--workdir", str(work),
        "--max-phase", "1",
    ], capture_output=True, text=True)
    assert res.returncode == 0, f"stderr: {res.stderr}"

    nj = work / "groups" / "network.json"
    assert nj.exists()
    d = json.loads(nj.read_text())
    assert d["function"] == "twochain"
    assert all(k["kind"] == "ascendc" for k in d["kernels"])
    assert len(d["kernels"]) == 2
