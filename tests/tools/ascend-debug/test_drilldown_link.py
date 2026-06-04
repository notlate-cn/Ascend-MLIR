import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "ascend-debug"))
from ascend_debug import debug_graph

def test_detail_link_redirects_to_subdashboard(tmp_path):
    # kernel_summary with one real kernel id
    ks = {"nodes": {"kernel_group0": {"id": "kernel_group0", "kind": "ascendc"}}}
    # no sub-dashboard yet → default path
    m1 = debug_graph._kernel_detail_views(tmp_path, [], ks)
    assert m1["kernel_group0"] == "views/kernels/kernel_group0.html"
    # create the sub-dashboard index → redirect
    sub = tmp_path / "kernels" / "kernel_group0"
    sub.mkdir(parents=True)
    (sub / "index.html").write_text("<html></html>")
    m2 = debug_graph._kernel_detail_views(tmp_path, [], ks)
    assert m2["kernel_group0"] == "kernels/kernel_group0/index.html"
