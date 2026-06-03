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
