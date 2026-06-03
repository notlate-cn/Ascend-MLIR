import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))

from ascend_debug import stage_graph


def test_kernel_id_from_auto_fuse_group_id():
    op_text = 'linalg.generic {auto_fuse.group_id = 3 : i64} ...'
    attrs = stage_graph._build_semantic_attrs("linalg.generic", op_text)
    assert attrs["kernel"]["id"] == "3"


def test_role_from_auto_fuse_kind():
    op_text = 'func.func @k(...) attributes {auto_fuse.kind = "vec"}'
    attrs = stage_graph._build_semantic_attrs("func.func", op_text)
    assert attrs["kernel"]["role"] == "vec"


def test_aclnn_op_surfaces_as_role_when_no_kind():
    op_text = 'func.call @aclnn ... {aclnn.op = "Matmul"}'
    attrs = stage_graph._build_semantic_attrs("func.call", op_text)
    assert attrs["kernel"]["role"] == "Matmul"


def test_no_ascend_namespace_attrs_referenced():
    src = (ROOT / "tools" / "ascend-debug" / "ascend_debug" / "stage_graph.py").read_text()
    assert "ascend.schedule" not in src
    assert "ascend.op_role" not in src
    assert "ascend.kernel" not in src


def test_func_node_emitted_with_kind_and_tiling_badges():
    text = (
        'module {\n'
        '  func.func @k(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> '
        'attributes {ascendc.kernel_kind = "vec", '
        'afir.block_dim_expr = "ceil((d0*d1)/XBLOCK)", '
        'afir.axis_extent_expr = "(d0*d1)"} {\n'
        '    return %arg0 : tensor<?x?xf32>\n'
        '  }\n'
        '}\n'
    )
    graph = stage_graph.parse_stage_mlir({"order": 1, "name": "schedule", "path": "p"}, text)
    func_nodes = [n for n in graph["nodes"] if n["op_name"] == "func.func"]
    assert len(func_nodes) == 1
    fn = func_nodes[0]
    assert fn["function"] == "@k"
    assert fn["semantic_attrs"]["kernel"]["role"] == "vec"
    assert fn["semantic_attrs"]["schedule"]["block_dim"] == "ceil((d0*d1)/XBLOCK)"
    assert fn["semantic_attrs"]["schedule"]["axis_extent"] == "(d0*d1)"
    assert fn["badges"]


def test_arg_node_carries_tile_and_shape_badges():
    text = (
        'module {\n'
        '  func.func @k(%arg0: tensor<?x?xf32> {afir.symbolic_shape = "s0,s1"}, '
        '%arg1: index {auto_fuse.default_tile_size = 128 : i64}) -> tensor<?x?xf32> {\n'
        '    return %arg0 : tensor<?x?xf32>\n'
        '  }\n'
        '}\n'
    )
    graph = stage_graph.parse_stage_mlir({"order": 1, "name": "schedule", "path": "p"}, text)
    arg_nodes = {n["label"]: n for n in graph["nodes"] if n["op_name"] == "func.arg"}
    assert arg_nodes["%arg0"]["semantic_attrs"]["kernel"].get("symbolic_shape") == "s0,s1" \
        or any("s0,s1" in b for b in arg_nodes["%arg0"]["badges"])
    assert arg_nodes["%arg1"]["semantic_attrs"]["schedule"].get("default_tile_size") == "128" \
        or any("128" in b for b in arg_nodes["%arg1"]["badges"])
