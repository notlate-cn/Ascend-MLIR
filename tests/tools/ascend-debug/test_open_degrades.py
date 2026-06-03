import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))

from ascend_debug import layout, open_view


def _make_min_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    stages_dir = run_dir / "stages"
    stages_dir.mkdir(parents=True)
    # two tiny but valid MLIR-ish stage artifacts
    (stages_dir / "010-normalize-out.mlir").write_text(
        'func.func @f(%a: tensor<4xf32>) -> tensor<4xf32> {\n'
        '  return %a : tensor<4xf32>\n}\n'
    )
    (stages_dir / "020-kernelize-out.mlir").write_text(
        'func.func @f(%a: tensor<4xf32>) -> tensor<4xf32> '
        '{ return %a : tensor<4xf32> }\n'
    )
    stages = (
        layout.StageArtifact(order=1, name="normalize",
                             path="stages/010-normalize-out.mlir", step="normalize"),
        layout.StageArtifact(order=2, name="kernelize",
                             path="stages/020-kernelize-out.mlir", step="kernelize"),
    )
    layout.write_manifest(
        run_dir, mode="develop-codegen", preset="", pipeline="auto-fuse-codegen",
        stages=stages, version="0.1", commands=[], reports=[], graphs=[],
    )
    return run_dir


def test_open_renders_from_collect_only_run_dir(tmp_path):
    run_dir = _make_min_run_dir(tmp_path)
    manifest = open_view.load_manifest(run_dir)
    index_path = open_view.render_index(run_dir, manifest)
    assert index_path.exists()
    assert index_path.suffix == ".html"
    assert "<html" in index_path.read_text(encoding="utf-8").lower()
