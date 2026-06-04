import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))
from ascend_debug import fusion_graph


def _setup(tmp_path):
    wd = tmp_path / "wd"
    (wd / "groups").mkdir(parents=True)
    (wd / "groups" / "kernel_group0.mlir").write_text(
        'func.func @kernel_group0(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {\n'
        '  %0 = linalg.generic ins(%a : tensor<4x4xf32>) outs(%a : tensor<4x4xf32>) {\n'
        '  ^bb0(%x: f32, %y: f32): linalg.yield %x : f32 } -> tensor<4x4xf32>\n'
        '  return %0 : tensor<4x4xf32>\n}\n'
    )
    (wd / "kernel_group0_best.json").write_text('{"XBLOCK": 8}')
    network = {"kernels": [
        {"id": "kernel_group0", "kind": "ascendc",
         "args": [{"from": "input", "name": "a"}],
         "results": [{"name": "r", "shape": [4, 4], "dtype": "f32"}]},
        {"id": "kernel_group1", "kind": "aclnn", "op": "MatMul",
         "args": [{"from": "kernel", "kernel": "kernel_group0", "result": 0}],
         "results": [{"name": "r2", "shape": [4, 4], "dtype": "f32"}]},
    ]}
    prov = {"kernels": [
        {"kernel_id": "kernel_group0", "fused_ops_summary": "add", "source_ops": [{"id": "op_000"}]},
        {"kernel_id": "kernel_group1", "fused_ops_summary": "matmul", "source_ops": [{"id": "op_001"}]},
    ]}
    return wd, network, prov


def test_group_nodes_and_children(tmp_path):
    wd, network, prov = _setup(tmp_path)
    g = fusion_graph.build_fusion_graph(network, prov, wd)
    nodes = {e["data"]["id"]: e["data"] for e in g["elements"] if "source" not in e["data"]}
    assert nodes["kernel_group0"]["type"] == "group"
    assert nodes["kernel_group0"]["kind"] == "ascendc"
    assert nodes["kernel_group0"]["shape"] == [4, 4]
    assert nodes["kernel_group0"]["label"] == "add"
    assert nodes["kernel_group0"]["drill"] == "kernels/kernel_group0/index.html"
    assert nodes["kernel_group0"]["tiling"]["best"] == {"XBLOCK": 8}
    assert nodes["kernel_group1"]["label"] == "matmul"
    children = [d for d in nodes.values() if d.get("parent") == "kernel_group0"]
    assert children and all(d["type"] == "op" for d in children)


def test_inter_group_edges(tmp_path):
    wd, network, prov = _setup(tmp_path)
    g = fusion_graph.build_fusion_graph(network, prov, wd)
    edges = [e["data"] for e in g["elements"] if "source" in e["data"]]
    inter = [e for e in edges if e["source"] == "kernel_group0" and e["target"] == "kernel_group1"]
    assert inter
