import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))

from ascend_debug import network_dag


def _net():
    return {
        "function": "f",
        "inputs": [{"name": "arg0", "shape": [4, 4], "dtype": "f16"}],
        "kernels": [
            {"id": "k0", "kind": "ascendc",
             "args": [{"from": "input", "name": "arg0"}],
             "results": [{"name": "k0_r0", "shape": [4, 4], "dtype": "f16"}]},
            {"id": "k1", "kind": "ascendc",
             "args": [{"from": "input", "name": "arg0"}],
             "results": [{"name": "k1_r0", "shape": [4, 4], "dtype": "f16"}]},
            {"id": "k2", "kind": "aclnn", "op": "MatMul",
             "args": [{"from": "kernel", "kernel": "k0", "result": 0},
                      {"from": "kernel", "kernel": "k1", "result": 0}],
             "results": [{"name": "k2_r0", "shape": [4, 4], "dtype": "f16"}]},
        ],
        "outputs": [{"from": "kernel", "kernel": "k2", "result": 0}],
    }


def _prov():
    return {"kernels": [
        {"kernel_id": "k0", "fused_ops_summary": "add", "source_ops": [{"id": "op_000"}]},
        {"kernel_id": "k1", "fused_ops_summary": "mul", "source_ops": [{"id": "op_001"}]},
        {"kernel_id": "k2", "fused_ops_summary": "matmul", "source_ops": [{"id": "op_002"}]},
    ]}


def test_nodes_keyed_by_kernel_id():
    s = network_dag.build_kernel_dag_summary(_net(), _prov())
    assert isinstance(s["nodes"], dict)
    assert set(s["nodes"]) == {"k0", "k1", "k2"}
    assert s["kernel_count"] == 3


def test_edges_from_kernel_args():
    s = network_dag.build_kernel_dag_summary(_net(), _prov())
    pairs = {(e["from"], e["to"]) for e in s["edges"]}
    assert ("k0", "k2") in pairs and ("k1", "k2") in pairs
    assert s["graph_edges"] == len(s["edges"])


def test_depth_and_critical_path():
    s = network_dag.build_kernel_dag_summary(_net(), _prov())
    assert s["nodes"]["k0"]["depth"] == 0
    assert s["nodes"]["k1"]["depth"] == 0
    assert s["nodes"]["k2"]["depth"] == 1
    assert s["critical_path_depth"] == 1


def test_label_from_provenance_and_shape():
    s = network_dag.build_kernel_dag_summary(_net(), _prov())
    assert s["nodes"]["k2"]["ops"][0]["label"] == "matmul"
    assert s["nodes"]["k2"]["aclnn_op"] == "MatMul"
    assert s["nodes"]["k0"]["output_shape"] == [4, 4]


def test_no_provenance_still_builds():
    s = network_dag.build_kernel_dag_summary(_net(), None)
    assert set(s["nodes"]) == {"k0", "k1", "k2"}
    assert s["nodes"]["k2"]["ops"][0]["label"] in ("MatMul", "aclnn")


def test_cycle_guard_terminates():
    net = {"kernels": [
        {"id": "a", "kind": "ascendc", "args": [{"from": "kernel", "kernel": "b"}], "results": []},
        {"id": "b", "kind": "ascendc", "args": [{"from": "kernel", "kernel": "a"}], "results": []},
    ]}
    s = network_dag.build_kernel_dag_summary(net, None)  # must not hang
    assert set(s["nodes"]) == {"a", "b"}
