"""Phase 2 test: per-kernel codegen → translate → compile.

Run with:
    source /home/gser/Ascend/ascend-toolkit/set_env.sh
    PYTHONPATH=python python3 -m pytest python/tests/test_network_runner_phase2.py -v
"""
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "python/network_runner.py"

TWOCHAIN_MLIR = """\
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
"""


def test_phase2_codegen_compile_twochain(tmp_path):
    """Phase 2: each ascendc kernel gets .cpp + _space.json + compiled artifact dir."""
    model = tmp_path / "model.mlir"
    model.write_text(TWOCHAIN_MLIR)

    work = tmp_path / "work"
    res = subprocess.run(
        [
            "python3", str(RUNNER),
            "--input-linalg", str(model),
            "--inputs", "/dev/null",
            "--expected", "/dev/null",
            "--workdir", str(work),
            "--max-phase", "2",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"

    for kid in ("kernel_group0", "kernel_group1"):
        cpp = work / f"{kid}.cpp"
        space = work / f"{kid}_space.json"
        artifact_dir = work / "artifacts" / kid

        assert cpp.exists(), f"missing {cpp}"
        assert space.exists(), f"missing {space}"
        assert artifact_dir.is_dir(), f"missing artifact dir {artifact_dir}"

        # Artifact dir must contain the compiled binary
        bins = list(artifact_dir.glob("*.bin"))
        assert bins, f"no .bin found under {artifact_dir}, contents: {list(artifact_dir.iterdir())}"
