#!/usr/bin/env bash
set -euo pipefail

INPUT_MLIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ascend-debug-cli.XXXXXX")"
TMP_REAL="$(cd "${TMP_DIR}" && pwd -P)"
SERVE_PID=""
cleanup() {
  if [[ -n "${SERVE_PID}" ]]; then
    kill "${SERVE_PID}" >/dev/null 2>&1 || true
    wait "${SERVE_PID}" >/dev/null 2>&1 || true
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

ascend-debug --help >"${TMP_DIR}/ascend-debug-help.txt" 2>&1
grep -Fq 'collect' "${TMP_DIR}/ascend-debug-help.txt"
grep -Fq 'open' "${TMP_DIR}/ascend-debug-help.txt"
grep -Fq 'serve' "${TMP_DIR}/ascend-debug-help.txt"
grep -Fq 'diff' "${TMP_DIR}/ascend-debug-help.txt"
grep -Fq 'locate' "${TMP_DIR}/ascend-debug-help.txt"
grep -Fq 'run' "${TMP_DIR}/ascend-debug-help.txt"
ascend-debug collect --help >"${TMP_DIR}/ascend-debug-collect-help.txt" 2>&1
grep -Fq -- '--mode {quick,deep}' "${TMP_DIR}/ascend-debug-collect-help.txt"
if grep -Fq -- '--preset' "${TMP_DIR}/ascend-debug-collect-help.txt"; then
  echo "collect help should not expose legacy --preset" >&2
  exit 1
fi
if grep -Fq -- '--pipeline' "${TMP_DIR}/ascend-debug-collect-help.txt"; then
  echo "collect help should not expose legacy --pipeline" >&2
  exit 1
fi
echo "ascend_debug.help=ok"

PYTHONPATH="${REPO_ROOT}/tools/ascend-debug${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
from ascend_debug import stage_graph

mlir = """module {
  func.func @copy_view(%arg0: memref<?xf16>, %arg1: memref<?xf16>) -> memref<?xf16> {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg1, %c0 : memref<?xf16>
    linalg.generic {indexing_maps = [], iterator_types = []} outs(%arg1 : memref<?xf16>) {
    ^bb0(%out: f16):
      linalg.yield %out : f16
    }
    %subview = memref.subview %arg1[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
    memref.copy %arg0, %subview : memref<?xf16> to memref<?xf16, strided<[1]>>
    return %arg1 : memref<?xf16>
  }
}
"""
graph = stage_graph.parse_stage_mlir({"order": 1, "name": "copy-view", "path": "stages/copy-view.mlir"}, mlir)
linalg_nodes = [node for node in graph["nodes"] if node["op_name"] == "linalg.generic"]
assert len(linalg_nodes) == 1, [node["op_name"] for node in graph["nodes"]]
copy_nodes = [node for node in graph["nodes"] if node["op_name"] == "memref.copy"]
assert len(copy_nodes) == 1, [node["op_name"] for node in graph["nodes"]]
copy_node = copy_nodes[0]
assert "%arg0" in copy_node["input_values"]
assert "%subview" in copy_node["input_values"]
incoming_values = {
    edge["value"]
    for edge in graph["edges"]
    if edge["to"] == copy_node["id"]
}
assert {"%arg0", "%subview"}.issubset(incoming_values), incoming_values
return_nodes = [node for node in graph["nodes"] if node["op_name"] == "func.return"]
assert len(return_nodes) == 1, [node["op_name"] for node in graph["nodes"]]
copy_effects = [
    edge
    for edge in graph["edges"]
    if edge["from"] == copy_node["id"]
    and edge["to"] == return_nodes[0]["id"]
    and edge.get("kind") == "memory_effect"
]
assert len(copy_effects) == 1, graph["edges"]
assert copy_effects[0]["value"] == "%arg1", copy_effects[0]

chain_mlir = """module {
  func.func @copy_chain(%arg0: memref<?xf16>, %arg1: memref<?xf16>) -> memref<?xf16> {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg1, %c0 : memref<?xf16>
    memref.copy %arg0, %arg1 : memref<?xf16> to memref<?xf16>
    %subview = memref.subview %arg1[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
    memref.copy %arg0, %subview : memref<?xf16> to memref<?xf16, strided<[1]>>
    return %arg1 : memref<?xf16>
  }
}
"""
chain_graph = stage_graph.parse_stage_mlir({"order": 1, "name": "copy-chain", "path": "stages/copy-chain.mlir"}, chain_mlir)
chain_copies = [node for node in chain_graph["nodes"] if node["op_name"] == "memref.copy"]
assert len(chain_copies) == 2, [node["op_name"] for node in chain_graph["nodes"]]
chain_subview = [node for node in chain_graph["nodes"] if node["op_name"] == "memref.subview"][0]
chain_return = [node for node in chain_graph["nodes"] if node["op_name"] == "func.return"][0]
chain_effects = [
    edge
    for edge in chain_graph["edges"]
    if edge.get("kind") == "memory_effect"
]
assert any(edge["from"] == chain_copies[0]["id"] and edge["to"] == chain_copies[1]["id"] for edge in chain_effects), chain_effects
assert any(edge["from"] == chain_copies[0]["id"] and edge["to"] == chain_return["id"] for edge in chain_effects), chain_effects
assert any(edge["from"] == chain_copies[1]["id"] and edge["to"] == chain_return["id"] for edge in chain_effects), chain_effects
assert not any(edge["from"] == chain_copies[0]["id"] and edge["to"] == chain_subview["id"] for edge in chain_effects), chain_effects

multiline_linalg_mlir = """module {
  func.func @multiline_linalg(
      %arg0: tensor<70x128xf16>,
      %arg1: tensor<70x128xf16>) -> tensor<70x128xf16> {
    %empty0 = tensor.empty() : tensor<70x128xf16>
    %add0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %arg1 : tensor<70x128xf16>, tensor<70x128xf16>)
      outs(%empty0 : tensor<70x128xf16>) {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<70x128xf16>
    return %add0 : tensor<70x128xf16>
  }
}
"""
multiline_graph = stage_graph.parse_stage_mlir({"order": 1, "name": "multiline-linalg", "path": "stages/multiline-linalg.mlir"}, multiline_linalg_mlir)
args = [node for node in multiline_graph["nodes"] if node["op_name"] == "func.arg"]
empty = next(node for node in multiline_graph["nodes"] if node["op_name"] == "tensor.empty")
generic = next(node for node in multiline_graph["nodes"] if node["op_name"] == "linalg.generic")
assert [arg["label"] for arg in args] == ["%arg0", "%arg1"], multiline_graph["nodes"]
assert {"%arg0", "%arg1", "%empty0"}.issubset(set(generic["input_values"])), generic
assert "%empty0" in generic["input_values"], generic
assert any(edge["from"] == args[0]["id"] and edge["to"] == generic["id"] and edge["value"] == "%arg0" for edge in multiline_graph["edges"]), multiline_graph["edges"]
assert any(edge["from"] == args[1]["id"] and edge["to"] == generic["id"] and edge["value"] == "%arg1" for edge in multiline_graph["edges"]), multiline_graph["edges"]
assert any(edge["from"] == empty["id"] and edge["to"] == generic["id"] and edge["value"] == "%empty0" for edge in multiline_graph["edges"]), multiline_graph["edges"]
assert multiline_graph["connectivity"]["suspicious_isolated_count"] == 0, multiline_graph["connectivity"]

tile_mlir = """module {
  func.func @tile_attrs(%arg0: tensor<70x128xf16>, %arg1: tensor<70x128xf16>) -> tensor<70x128xf16> {
    %empty0 = tensor.empty() : tensor<70x128xf16>
    %add0 = linalg.generic {indexing_maps = [], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<70x128xf16>, tensor<70x128xf16>) outs(%empty0 : tensor<70x128xf16>) attrs = {
      ascend.schedule.tile_binding = "symbolic",
      ascend.schedule.tile_params = [
        {axis = 0 : i64, axis_kind = "parallel", binding = "runtime", default = 70 : i64, extent = 70 : i64, name = "TB_M", primitive_uses = ["data_copy", "vector_compute", "write_back"], roles = ["bind_core", "kernel_loop"], upper_bound = 70 : i64},
        {axis = 1 : i64, axis_kind = "parallel", binding = "runtime", default = 70 : i64, extent = 128 : i64, name = "TB_N", primitive_uses = ["data_copy", "vector_compute", "write_back"], roles = ["kernel_loop", "vectorize"], upper_bound = 70 : i64}
      ],
      ascend.schedule.target_tile_policy = "target_ub_70"
    } {
    ^bb0(%x: f16, %y: f16, %o: f16):
      %v = arith.addf %x, %y : f16
      linalg.yield %v : f16
    } -> tensor<70x128xf16>
    return %add0 : tensor<70x128xf16>
  }
}
"""
tile_graph = stage_graph.parse_stage_mlir({"order": 32, "name": "tile-attrs", "path": "stages/tile-attrs.mlir"}, tile_mlir)
tile_node = next(node for node in tile_graph["nodes"] if node["op_name"] == "linalg.generic")
schedule = tile_node["semantic_attrs"]["schedule"]
assert schedule["tile_binding"] == "symbolic", schedule
assert [param["name"] for param in schedule["tile_params"]] == ["TB_M", "TB_N"], schedule
assert schedule["tile_params"][0]["default"] == 70, schedule
assert schedule["tile_params"][1]["extent"] == 128, schedule
assert "tile TB_M/TB_N" in tile_node["badges"], tile_node["badges"]

resource_mlir = """module {
  func.func @resource_chain(%pipe: i32, %src: i32, %bytes: index) {
    %c1 = arith.constant 1 : i32
    %queue = arith.constant 0 : i32
    %global = arith.constant 0 : i32
    %local = ascendc.que_bind.alloc_tensor %queue : i32
    %sum = ascendc.que_bind.alloc_tensor %queue : i32
    ascendc.pipe.init_queue %pipe, %queue, %c1, %bytes : i32, i32, i32, index
    ascendc.global_tensor.set_global_buffer %global, %src, %bytes : i32, i32, index
    emitasc.verbatim %local, %global, %bytes : i32, i32, index
    ascendc.add_l2 %sum, %local, %local, %bytes : i32, i32, i32, index
    ascendc.pipe_barrier pipe_all
    ascendc.que_bind.enque_tensor %queue, %sum : i32, i32
    return
  }
}
"""
resource_graph = stage_graph.parse_stage_mlir({"order": 60, "name": "resource-chain", "path": "stages/resource-chain.mlir"}, resource_mlir)
resource_nodes = {node["op_name"]: node for node in resource_graph["nodes"]}
barrier = resource_nodes["ascendc.pipe_barrier"]
set_global = resource_nodes["ascendc.global_tensor.set_global_buffer"]
verbatim = resource_nodes["emitasc.verbatim"]
add_l2 = resource_nodes["ascendc.add_l2"]
enque = resource_nodes["ascendc.que_bind.enque_tensor"]
resource_edges = [edge for edge in resource_graph["edges"] if edge.get("kind") == "resource_effect"]
control_edges = [edge for edge in resource_graph["edges"] if edge.get("kind") == "control"]
assert any(edge["from"] == set_global["id"] and edge["to"] == verbatim["id"] for edge in resource_edges), resource_edges
assert any(edge["from"] == verbatim["id"] and edge["to"] == add_l2["id"] for edge in resource_edges), resource_edges
assert any(edge["from"] == add_l2["id"] and edge["to"] == enque["id"] for edge in resource_edges), resource_edges
assert any(edge["to"] == barrier["id"] for edge in control_edges), control_edges
assert any(edge["from"] == barrier["id"] and edge["to"] == enque["id"] for edge in control_edges), control_edges
assert resource_graph["connectivity"]["dangling_effect_count"] == 0, resource_graph["connectivity"]

emit_mlir = """module {
  emitasc.declare_py_struct @Tiling
  func.func @emit_stage(%arg0: i32, %arg1: i32) {
    %member = emitasc.member %arg0 : i32
    emitasc.verbatim %arg1, %member : i32, i32
    scf.if %arg0 {
      emitasc.verbatim %arg1, %member : i32, i32
    }
    return
  }
}
"""
emit_graph = stage_graph.parse_stage_mlir({"order": 80, "name": "emit-stage", "path": "stages/emit-stage.mlir"}, emit_mlir)
connectivity = emit_graph["connectivity"]
assert connectivity["isolated_count"] >= 1, connectivity
assert connectivity["suspicious_isolated_count"] == 0, connectivity
assert connectivity["dangling_effect_count"] == 0, connectivity
terminal_ops = {
    item["op_name"]
    for item in connectivity["allowed_terminal_nodes"]
}
assert "emitasc.declare_py_struct" in terminal_ops, connectivity
assert "func.return" in terminal_ops, connectivity
assert connectivity["edge_kind_counts"].get("resource_effect", 0) >= 1, connectivity

constructor_mlir = """module {
  func.func @resource_constructors() {
    %pipe = ascendc.pipe
    %queue = ascendc.queue
    %tensor = ascendc.global_tensor
    %cast = emitasc.reinterpret_cast %tensor : i32
    return
  }
}
"""
constructor_graph = stage_graph.parse_stage_mlir({"order": 60, "name": "constructors", "path": "stages/constructors.mlir"}, constructor_mlir)
assert constructor_graph["connectivity"]["suspicious_isolated_count"] == 0, constructor_graph["connectivity"]
assert constructor_graph["connectivity"]["dangling_effect_count"] == 0, constructor_graph["connectivity"]
PY
echo "ascend_debug.stage_graph_resultless_ops=ok"

cat >"${TMP_DIR}/typed-stage-graph.mlir" <<'MLIR'
module {
  func.func @typed_graph(%arg0: memref<?xf16>, %arg1: memref<?xf16>) -> memref<?xf16> {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg1, %c0 : memref<?xf16>
    %subview = memref.subview %arg1[0] [%dim] [1] : memref<?xf16> to memref<?xf16, strided<[1]>>
    memref.copy %arg0, %subview : memref<?xf16> to memref<?xf16, strided<[1]>>
    return %arg1 : memref<?xf16>
  }
}
MLIR
STAGE_GRAPH_TOOL="${ASCEND_STAGE_GRAPH_TOOL:-build/bin/ascend-stage-graph}"
if [[ -x "${STAGE_GRAPH_TOOL}" ]]; then
  "${STAGE_GRAPH_TOOL}" \
    "${TMP_DIR}/typed-stage-graph.mlir" \
    --stage-order 1 \
    --stage-name typed-stage-graph \
    --stage-path stages/typed-stage-graph.mlir \
    --output "${TMP_DIR}/typed-stage-graph.json"
  python3 - "${TMP_DIR}/typed-stage-graph.json" <<'PY'
import json
import pathlib
import sys

graph = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert graph["stage"]["order"] == 1
assert graph["stage"]["name"] == "typed-stage-graph"
assert graph["nodes"]
assert graph["edges"]
assert any(edge.get("kind") == "value" for edge in graph["edges"]), graph["edges"]
assert any(edge.get("kind") == "memory_effect" for edge in graph["edges"]), graph["edges"]
PY
  echo "ascend_stage_graph.basic=ok"
else
  echo "ascend_stage_graph.basic=skipped"
fi

mkdir -p "${TMP_DIR}/fake-runtime-session"
cat >"${TMP_DIR}/fake-runtime-session/runtime-session" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${ASCEND_DEBUG_FAKE_RUNTIME_LOG}"
if [[ "$#" -eq 4 && "$1" == "--case" && "$3" == "--emit-run-manifest" ]]; then
  mkdir -p "$(dirname "$4")"
  printf '{"backend":"sim","tasks":[]}\n' >"$4"
  printf 'run_manifest.path=%s\n' "$4"
  exit 0
fi
if [[ "$#" -eq 3 && "$1" == "--run-manifest" && "$3" == "--run" ]]; then
  test -f "$2"
  printf 'session.backend=sim\n'
  printf 'session.result=success\n'
  exit 0
fi
if [[ "$#" -eq 4 && "$1" == "--compare-tensors" && "$3" == "--emit-validation-summary" ]]; then
  test -f "$2"
  mkdir -p "$(dirname "$4")"
  printf '{"schema_version":1,"tool":"runtime-session","status":"pass","comparison_count":1,"failed_count":0,"comparisons":[]}\n' >"$4"
  printf 'validation.status=pass\n'
  exit 0
fi
printf 'unexpected runtime-session invocation: %s\n' "$*" >&2
exit 2
SH
chmod +x "${TMP_DIR}/fake-runtime-session/runtime-session"
cat >"${TMP_DIR}/case.json" <<'JSON'
{
  "schema_version": 1,
  "artifact": {
    "root": "/tmp/artifact",
    "manifest": "/tmp/artifact_manifest.json"
  },
  "backend": {
    "kind": "sim"
  },
  "inputs": []
}
JSON
ASCEND_DEBUG_FAKE_RUNTIME_LOG="${TMP_DIR}/fake-runtime-session.log" \
  PATH="${TMP_DIR}/fake-runtime-session:${PATH}" \
  ascend-debug run "${TMP_DIR}/case.json" --out "${TMP_DIR}/debug-case-run"
test -f "${TMP_DIR}/debug-case-run/run_manifest.json"
grep -Fq -- "--case ${TMP_REAL}/case.json --emit-run-manifest ${TMP_REAL}/debug-case-run/run_manifest.json" \
  "${TMP_DIR}/fake-runtime-session.log"
grep -Fq -- "--run-manifest ${TMP_REAL}/debug-case-run/run_manifest.json --run" \
  "${TMP_DIR}/fake-runtime-session.log"
grep -Fq "run_manifest.path=${TMP_REAL}/debug-case-run/run_manifest.json" \
  "${TMP_DIR}/debug-case-run/runtime-session.prepare.log"
grep -Fq "session.result=success" \
  "${TMP_DIR}/debug-case-run/runtime-session.run.log"
echo "ascend_debug.run_case=ok"

ASCEND_DEBUG_FAKE_RUNTIME_LOG="${TMP_DIR}/fake-runtime-session-prepare-only.log" \
  PATH="${TMP_DIR}/fake-runtime-session:${PATH}" \
  ascend-debug run "${TMP_DIR}/case.json" \
    --out "${TMP_DIR}/debug-case-prepare-only" \
    --prepare-runtime-artifacts
test -f "${TMP_DIR}/debug-case-prepare-only/run_manifest.json"
grep -Fq -- "--case ${TMP_REAL}/case.json --emit-run-manifest ${TMP_REAL}/debug-case-prepare-only/run_manifest.json" \
  "${TMP_DIR}/fake-runtime-session-prepare-only.log"
if grep -Fq -- "--run-manifest ${TMP_REAL}/debug-case-prepare-only/run_manifest.json --run" \
    "${TMP_DIR}/fake-runtime-session-prepare-only.log"; then
  echo "prepare-only ascend-debug run unexpectedly executed runtime-session --run" >&2
  exit 1
fi
test ! -f "${TMP_DIR}/debug-case-prepare-only/runtime-session.run.log"
echo "ascend_debug.run_case_prepare_only=ok"

mkdir -p "${TMP_DIR}/fake-runtime-session-fail"
cat >"${TMP_DIR}/fake-runtime-session-fail/runtime-session" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${ASCEND_DEBUG_FAKE_RUNTIME_FAIL_LOG}"
if [[ "$#" -eq 4 && "$1" == "--case" && "$3" == "--emit-run-manifest" ]]; then
  mkdir -p "$(dirname "$4")"
  printf '{"backend":"sim","tasks":[]}\n' >"$4"
  printf 'run_manifest.path=%s\n' "$4"
  exit 0
fi
if [[ "$#" -eq 3 && "$1" == "--run-manifest" && "$3" == "--run" ]]; then
  printf 'runtime-session: error: simulated launch failure in test fixture before session.result\n' >&2
  exit 88
fi
printf 'unexpected runtime-session invocation: %s\n' "$*" >&2
exit 2
SH
chmod +x "${TMP_DIR}/fake-runtime-session-fail/runtime-session"
if ASCEND_DEBUG_FAKE_RUNTIME_FAIL_LOG="${TMP_DIR}/fake-runtime-session-fail.log" \
  PATH="${TMP_DIR}/fake-runtime-session-fail:${PATH}" \
  ascend-debug run "${TMP_DIR}/case.json" --out "${TMP_DIR}/debug-case-run-fail" \
  >"${TMP_DIR}/debug-case-run-fail.stdout" \
  2>"${TMP_DIR}/debug-case-run-fail.stderr"; then
  echo "expected ascend-debug run failure case to return non-zero" >&2
  exit 1
fi
test -f "${TMP_DIR}/debug-case-run-fail/manifest.json"
test -f "${TMP_DIR}/debug-case-run-fail/run_status.json"
test -f "${TMP_DIR}/debug-case-run-fail/run_manifest.json"
test -f "${TMP_DIR}/debug-case-run-fail/runtime-session.prepare.log"
test -f "${TMP_DIR}/debug-case-run-fail/reports/runtime-session.run.stderr.txt"
grep -Fq 'runtime-session: error: simulated launch failure in test fixture before session.result' \
  "${TMP_DIR}/debug-case-run-fail/reports/runtime-session.run.stderr.txt"
python3 - "${TMP_DIR}/debug-case-run-fail/manifest.json" "${TMP_DIR}/debug-case-run-fail/run_status.json" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
status = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert manifest["status"] == "failed", manifest
assert manifest["failed_stage"] == "runtime-run", manifest
assert manifest["failed_phase"] == "runtime", manifest
assert manifest["failure_status"] == "run_status.json", manifest
assert manifest["backend"] == "runtime", manifest
commands = manifest["commands"]
assert [command["status"] for command in commands] == ["success", "failed"], commands
assert commands[-1]["exit_code"] == 88, commands[-1]
assert status["status"] == "failed", status
assert status["failure"]["stage"] == "runtime-run", status
assert status["failure"]["phase"] == "runtime", status
assert status["failure"]["command"]["exit_code"] == 88, status
PY
ascend-debug open "${TMP_DIR}/debug-case-run-fail" --no-browser >"${TMP_DIR}/ascend-debug-open-run-fail.txt"
grep -Fq '<h2>Run Status</h2>' "${TMP_DIR}/debug-case-run-fail/index.html"
grep -Fq 'runtime-run' "${TMP_DIR}/debug-case-run-fail/index.html"
grep -Fq 'runtime-session: error: simulated launch failure in test fixture before session.result' "${TMP_DIR}/debug-case-run-fail/index.html"
echo "ascend_debug.run_case_failure_workspace=ok"

mkdir -p "${TMP_DIR}/fake-source-tools"
cat >"${TMP_DIR}/fake-source-tools/ascend-mlir-opt" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'ascend-mlir-opt %s\n' "$*" >>"${ASCEND_DEBUG_FAKE_SOURCE_LOG}"
cat "$1"
SH
chmod +x "${TMP_DIR}/fake-source-tools/ascend-mlir-opt"
cat >"${TMP_DIR}/fake-source-tools/ascend-mlir-translate" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'ascend-mlir-translate %s\n' "$*" >>"${ASCEND_DEBUG_FAKE_SOURCE_LOG}"
tiling=""
manifest=""
host_tiling=""
kernel=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiling-space-out=*) tiling="${1#--tiling-space-out=}" ;;
    --artifact-manifest-out=*) manifest="${1#--artifact-manifest-out=}" ;;
    --host-tiling-out=*) host_tiling="${1#--host-tiling-out=}" ;;
    -o)
      kernel="$2"
      shift
      ;;
  esac
  shift
done
printf '{"schema_version":1}\n' >"${tiling}"
printf '{"kernel_entries":[],"kernelGraph":{"nodes":[],"edges":[]}}\n' >"${manifest}"
printf 'extern "C" int source_tiling() { return 0; }\n' >"${host_tiling}"
printf 'extern "C" void source_kernel() {}\n' >"${kernel}"
SH
chmod +x "${TMP_DIR}/fake-source-tools/ascend-mlir-translate"
cat >"${TMP_DIR}/fake-source-tools/runtime-session" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'runtime-session %s\n' "$*" >>"${ASCEND_DEBUG_FAKE_SOURCE_LOG}"
if [[ "$1" == "--kernel" ]]; then
  output=""
  name=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output)
        output="$2"
        shift
        ;;
      --name)
        name="$2"
        shift
        ;;
    esac
    shift
  done
  mkdir -p "${output}/out"
  printf 'fake artifact for %s\n' "${name}" >"${output}/out/manifest.txt"
  printf 'artifact.kernel_name=%s\n' "${name}"
  printf 'artifact.root=%s\n' "${output}"
  printf 'artifact.manifest=%s\n' "${output}/out/manifest.txt"
  printf 'artifact.soc=Ascend910B1\n'
  exit 0
fi
if [[ "$#" -eq 4 && "$1" == "--case" && "$3" == "--emit-run-manifest" ]]; then
  mkdir -p "$(dirname "$4")"
  printf '{"backend":"sim","tasks":[]}\n' >"$4"
  printf 'run_manifest.path=%s\n' "$4"
  exit 0
fi
if [[ "$#" -eq 3 && "$1" == "--run-manifest" && "$3" == "--run" ]]; then
  test -f "$2"
  printf 'session.backend=sim\n'
  printf 'session.result=success\n'
  exit 0
fi
printf 'unexpected runtime-session invocation: %s\n' "$*" >&2
exit 2
SH
chmod +x "${TMP_DIR}/fake-source-tools/runtime-session"
cat >"${TMP_DIR}/fake-source-tools/c++" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'c++ %s\n' "$*" >>"${ASCEND_DEBUG_FAKE_SOURCE_LOG}"
out=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-o" ]]; then
    out="$2"
    shift
  fi
  shift
done
mkdir -p "$(dirname "${out}")"
printf 'fake host tiling so\n' >"${out}"
SH
chmod +x "${TMP_DIR}/fake-source-tools/c++"
cat >"${TMP_DIR}/source.mlir" <<'MLIR'
module {
  func.func @source_kernel(%arg0: tensor<?x?xf16>) -> tensor<?x?xf16> {
    return %arg0 : tensor<?x?xf16>
  }
}
MLIR
touch "${TMP_DIR}/input.raw" "${TMP_DIR}/expected.raw"
cat >"${TMP_DIR}/source-case.json" <<'JSON'
{
  "schema_version": 1,
  "source": {
    "mlir": "source.mlir"
  },
  "backend": {
    "kind": "sim"
  },
  "inputs": [
    {
      "name": "arg0",
      "path": "input.raw",
      "shape": [4, 8],
      "datatype": "f16"
    }
  ],
  "outputs": [
    {
      "name": "out0",
      "path": "actual.npy"
    }
  ],
  "expected_outputs": [
    {
      "name": "out0",
      "path": "expected.raw",
      "shape": [4, 8],
      "datatype": "f16"
    }
  ],
  "validation": {
    "atol": 0.01,
    "rtol": 0.02
  }
}
JSON
ASCEND_DEBUG_FAKE_SOURCE_LOG="${TMP_DIR}/fake-source-tools.log" \
  CANN_ROOT="${TMP_DIR}/fake-cann-root" \
  PATH="${TMP_DIR}/fake-source-tools:${PATH}" \
  ascend-debug run "${TMP_DIR}/source-case.json" --out "${TMP_DIR}/debug-source-run"
test -f "${TMP_DIR}/debug-source-run/step1_fused.mlir"
test -f "${TMP_DIR}/debug-source-run/step10_kernel.cpp"
test -f "${TMP_DIR}/debug-source-run/phase5_artifact_manifest.json"
test -f "${TMP_DIR}/debug-source-run/artifact/host_tiling.so"
test -f "${TMP_DIR}/debug-source-run/case.artifact.json"
python3 - "${TMP_DIR}/debug-source-run/case.artifact.json" <<'PY'
import json
import pathlib
import sys

case = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert "source" not in case
assert case["artifact"]["root"].endswith("/debug-source-run/artifact")
assert case["artifact"]["manifest"].endswith("/debug-source-run/phase5_artifact_manifest.json")
assert case["shape_args"] == {"arg0": [4, 8]}
assert case["inputs"][0]["dtype"] == "f16"
assert case["expected_outputs"][0]["dtype"] == "f16"
PY
grep -Fq -- 'ascend-mlir-opt' "${TMP_DIR}/fake-source-tools.log"
grep -Fq -- 'ascend-mlir-opt '"${TMP_REAL}"'/debug-source-run/step8_kernel_ir.mlir --ascend-canonicalize-cann-signature --canonicalize --cse' \
  "${TMP_DIR}/fake-source-tools.log"
grep -Fq -- 'ascend-mlir-translate' "${TMP_DIR}/fake-source-tools.log"
grep -Fq -- 'runtime-session --kernel' "${TMP_DIR}/fake-source-tools.log"
grep -Fq -- "--case ${TMP_REAL}/debug-source-run/case.artifact.json --emit-run-manifest ${TMP_REAL}/debug-source-run/run_manifest.json" \
  "${TMP_DIR}/fake-source-tools.log"
grep -Fq "session.result=success" \
  "${TMP_DIR}/debug-source-run/runtime-session.run.log"
echo "ascend_debug.run_source_case=ok"

ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run" \
  --preset quick
test -f "${TMP_DIR}/debug-run/stages/000-source.mlir"
test -f "${TMP_DIR}/debug-run/stages/010-normalize-in.mlir"
test -f "${TMP_DIR}/debug-run/stages/019-normalize-out.mlir"
test -f "${TMP_DIR}/debug-run/stages/020-kernelize-in.mlir"
cmp -s "${TMP_DIR}/debug-run/stages/019-normalize-out.mlir" "${TMP_DIR}/debug-run/stages/020-kernelize-in.mlir"
test -f "${TMP_DIR}/debug-run/stages/029-kernelize-out.mlir"
grep -Fq 'ascend.kernel' "${TMP_DIR}/debug-run/stages/029-kernelize-out.mlir"
test -f "${TMP_DIR}/debug-run/manifest.json"
test -f "${TMP_DIR}/debug-run/provenance.json"
echo "ascend_debug.collect=ok"

if ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run-memory-detail-negative" \
  --mode deep \
  --memory-detail >"${TMP_DIR}/ascend-debug-memory-detail-negative.txt" 2>"${TMP_DIR}/ascend-debug-memory-detail-negative.err"; then
  echo "expected deep memory-detail collect to fail" >&2
  exit 1
fi
grep -Fq -- '--memory-detail is only supported by --mode quick' "${TMP_DIR}/ascend-debug-memory-detail-negative.err"
echo "ascend_debug.collect_memory_detail_negative=ok"

ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run-deep" \
  --mode quick
test -f "${TMP_DIR}/debug-run-deep/stages/000-source.mlir"
test -f "${TMP_DIR}/debug-run-deep/stages/010-normalize-in.mlir"
test -f "${TMP_DIR}/debug-run-deep/stages/019-normalize-out.mlir"
test -f "${TMP_DIR}/debug-run-deep/stages/020-kernelize-in.mlir"
test -f "${TMP_DIR}/debug-run-deep/stages/029-kernelize-out.mlir"
test -f "${TMP_DIR}/debug-run-deep/stages/030-schedule-in.mlir"
test -f "${TMP_DIR}/debug-run-deep/stages/039-schedule-out.mlir"
test -f "${TMP_DIR}/debug-run-deep/stages/040-realize-in.mlir"
test -f "${TMP_DIR}/debug-run-deep/stages/049-realize-out.mlir"
test -f "${TMP_DIR}/debug-run-deep/reports/010-normalize.report.txt"
test -f "${TMP_DIR}/debug-run-deep/reports/020-kernelize.report.txt"
test -f "${TMP_DIR}/debug-run-deep/reports/030-schedule.report.txt"
test -f "${TMP_DIR}/debug-run-deep/reports/040-realize.report.txt"
echo "ascend_debug.collect_deep=ok"

mkdir -p "${TMP_DIR}/fake-collect-fail-tools"
cat >"${TMP_DIR}/fake-collect-fail-tools/ascend-mlir-opt" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'ascend-mlir-opt %s\n' "$*" >>"${ASCEND_DEBUG_FAKE_COLLECT_FAIL_LOG}"
if [[ "$*" == *"--ascend-kernelize"* ]]; then
  printf 'stages/020-kernelize-in.mlir:10:3: error: ascend-kernelize test fixture rejected unsupported op\n' >&2
  exit 77
fi
cat "$1"
SH
chmod +x "${TMP_DIR}/fake-collect-fail-tools/ascend-mlir-opt"
if ASCEND_DEBUG_FAKE_COLLECT_FAIL_LOG="${TMP_DIR}/fake-collect-fail-tools.log" \
  PATH="${TMP_DIR}/fake-collect-fail-tools:${PATH}" \
  ascend-debug collect "${INPUT_MLIR}" \
    --out "${TMP_DIR}/debug-collect-fail" \
    --mode quick \
    >"${TMP_DIR}/debug-collect-fail.stdout" \
    2>"${TMP_DIR}/debug-collect-fail.stderr"; then
  echo "expected ascend-debug collect failure case to return non-zero" >&2
  exit 1
fi
test -f "${TMP_DIR}/debug-collect-fail/manifest.json"
test -f "${TMP_DIR}/debug-collect-fail/run_status.json"
test -f "${TMP_DIR}/debug-collect-fail/stages/000-source.mlir"
test -f "${TMP_DIR}/debug-collect-fail/stages/019-normalize-out.mlir"
test -f "${TMP_DIR}/debug-collect-fail/stages/020-kernelize-in.mlir"
test ! -f "${TMP_DIR}/debug-collect-fail/stages/029-kernelize-out.mlir"
grep -Fq 'ascend-kernelize test fixture rejected unsupported op' "${TMP_DIR}/debug-collect-fail/reports/020-kernelize.report.txt"
python3 - "${TMP_DIR}/debug-collect-fail/manifest.json" "${TMP_DIR}/debug-collect-fail/run_status.json" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
status = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert manifest["status"] == "failed", manifest
assert manifest["failed_stage"] == "kernelize", manifest
assert manifest["failed_phase"] == "compile", manifest
assert manifest["failure_status"] == "run_status.json", manifest
assert [stage["name"] for stage in manifest["stages"]] == [
    "source",
    "normalize-in",
    "normalize-out",
    "kernelize-in",
], manifest["stages"]
commands = manifest["commands"]
assert [command["stage"] for command in commands] == ["normalize", "kernelize"], commands
assert [command["status"] for command in commands] == ["success", "failed"], commands
assert commands[-1]["exit_code"] == 77, commands[-1]
assert status["status"] == "failed", status
assert status["failure"]["stage"] == "kernelize", status
assert status["failure"]["phase"] == "compile", status
assert status["failure"]["command"]["stderr"] == "reports/020-kernelize.report.txt", status
PY
ascend-debug open "${TMP_DIR}/debug-collect-fail" --no-browser >"${TMP_DIR}/ascend-debug-open-collect-fail.txt"
grep -Fq '<h2>Run Status</h2>' "${TMP_DIR}/debug-collect-fail/index.html"
grep -Fq 'kernelize' "${TMP_DIR}/debug-collect-fail/index.html"
grep -Fq 'ascend-kernelize test fixture rejected unsupported op' "${TMP_DIR}/debug-collect-fail/index.html"
echo "ascend_debug.collect_failure_workspace=ok"

mkdir -p "${TMP_DIR}/fake-full-codegen-tools" "${TMP_DIR}/fake-full-codegen-cann"
cat >"${TMP_DIR}/fake-full-codegen-tools/ascend-mlir-opt" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'ascend-mlir-opt %s\n' "$*" >>"${ASCEND_DEBUG_FAKE_FULL_CODEGEN_LOG}"
input="$1"
dump_dir=""
for arg in "$@"; do
  if [[ "${arg}" == *debug-dump-dir=* ]]; then
    dump_dir="${arg#*debug-dump-dir=}"
    dump_dir="${dump_dir%% *}"
  fi
done
write_checkpoint() {
  local name="$1"
  mkdir -p "${dump_dir}"
  {
    printf '// checkpoint: %s\n' "${name}"
    cat "${input}"
  } >"${dump_dir}/${name}.mlir"
}
if [[ -n "${dump_dir}" ]]; then
  if [[ "$*" == *"--ascend-kernelize"* ]]; then
    write_checkpoint "021-kernelize-structured-ops"
    write_checkpoint "022-kernelize-structural-marking"
    write_checkpoint "023-kernelize-role-classification"
    write_checkpoint "024-kernelize-final-patterns"
  elif [[ "$*" == *"--ascend-schedule"* ]]; then
    write_checkpoint "031-schedule-cleared"
    write_checkpoint "032-schedule-decisions"
    write_checkpoint "033-schedule-final"
  elif [[ "$*" == *"--ascend-realize"* ]]; then
    write_checkpoint "041-realize-planned"
    write_checkpoint "042-realize-bufferized"
    write_checkpoint "043-realize-memory-space-annotated"
  fi
fi
cat "$1"
SH
chmod +x "${TMP_DIR}/fake-full-codegen-tools/ascend-mlir-opt"
ASCEND_DEBUG_FAKE_FULL_CODEGEN_LOG="${TMP_DIR}/fake-full-codegen-tools.log" \
  CANN_ROOT="${TMP_DIR}/fake-full-codegen-cann" \
  ASCEND_SOC_VERSION="SyntheticSoC" \
  PATH="${TMP_DIR}/fake-full-codegen-tools:${PATH}" \
  ascend-debug collect "${INPUT_MLIR}" \
    --out "${TMP_DIR}/debug-run-full-codegen" \
    --mode deep
test -f "${TMP_DIR}/debug-run-full-codegen/stages/010-normalize-prep-out.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/020-normalize-out.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/021-kernelize-structured-ops.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/022-kernelize-structural-marking.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/023-kernelize-role-classification.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/024-kernelize-final-patterns.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/030-kernelize-out.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/031-schedule-cleared.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/032-schedule-decisions.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/033-schedule-final.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/040-schedule-out.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/041-realize-planned.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/042-realize-bufferized.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/043-realize-memory-space-annotated.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/050-realize-out.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/060-compute-lower-out.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/070-parallelize-out.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/080-prepare-for-emit-out.mlir"
test -f "${TMP_DIR}/debug-run-full-codegen/stages/090-cann-signature-out.mlir"
grep -Fq -- 'debug-dump-dir=' "${TMP_DIR}/fake-full-codegen-tools.log"
grep -Fq -- '--ascend-schedule=target-tile-policy=target-aware' "${TMP_DIR}/fake-full-codegen-tools.log"
grep -Fq -- '--ascend-realize=materialization-mode=memory-space-annotate' "${TMP_DIR}/fake-full-codegen-tools.log"
python3 - "${TMP_DIR}/debug-run-full-codegen/manifest.json" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert manifest["mode"] == "deep"
assert manifest["pipeline"] == "full-codegen"
stages = {stage["name"]: stage for stage in manifest["stages"]}
by_phase = {}
for stage in manifest["stages"]:
    phase = stage.get("phase")
    if phase:
        by_phase.setdefault(phase, []).append(stage["name"])
assert by_phase["Kernelize"] == [
    "021-kernelize-structured-ops",
    "022-kernelize-structural-marking",
    "023-kernelize-role-classification",
    "024-kernelize-final-patterns",
    "030-kernelize-out",
]
assert by_phase["Schedule"] == [
    "031-schedule-cleared",
    "032-schedule-decisions",
    "033-schedule-final",
    "040-schedule-out",
]
assert by_phase["Realize"] == [
    "041-realize-planned",
    "042-realize-bufferized",
    "043-realize-memory-space-annotated",
    "050-realize-out",
]
assert stages["021-kernelize-structured-ops"]["step"] == "structured-ops"
assert stages["021-kernelize-structured-ops"]["step_info"]["title"] == "识别可 Kernelize 的算子"
assert "structured/tensor/arith" in stages["021-kernelize-structured-ops"]["step_info"]["purpose"]
assert stages["033-schedule-final"]["step"] == "final"
assert stages["033-schedule-final"]["step_info"]["title"] == "挂载 Schedule 契约"
assert "tile_params" in stages["033-schedule-final"]["step_info"]["outputs"]
assert stages["043-realize-memory-space-annotated"]["step"] == "memory-space-annotated"
assert stages["043-realize-memory-space-annotated"]["step_info"]["title"] == "标注内存空间"
assert stages["060-compute-lower-out"]["phase"] == "Translate"
assert stages["060-compute-lower-out"]["step"] == "ascend-compute-lower"
assert stages["060-compute-lower-out"]["step_info"]["title"] == "Lower 到 AscendC IR"
assert stages["090-cann-signature-out"]["phase"] == "Translate"
assert stages["090-cann-signature-out"]["step"] == "ascend-canonicalize-cann-signature"
assert by_phase["Translate"] == [
    "060-compute-lower-out",
    "070-parallelize-out",
    "080-prepare-for-emit-out",
    "090-cann-signature-out",
]
PY
ascend-debug open "${TMP_DIR}/debug-run-full-codegen" --no-browser >"${TMP_DIR}/ascend-debug-open-full-codegen.txt"
grep -Fq '<thead><tr><th>Stage</th><th>Step / Per pass</th><th>View</th><th>Command</th><th>Report</th></tr></thead>' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '识别可 Kernelize 的算子' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '挂载 Schedule 契约' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '标注内存空间' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq 'Lower 到 AscendC IR' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<div class="step-id">Step ID: structured-ops</div>' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<div class="step-id">Step ID: ascend-kernelize</div>' "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq 'th { background: #f1f5f9; text-align: center; }' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '.view-links { display: inline-flex; gap: 1rem; align-items: center; }' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<td class="stage-group-cell" rowspan="5">Kernelize</td>' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<div class="step-file">Dump: 021-kernelize-structured-ops</div>' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<div class="step-file">Dump: 030-kernelize-out</div>' "${TMP_DIR}/debug-run-full-codegen/index.html"
if grep -Fq '<div class="step-title">021-kernelize-structured-ops</div>' "${TMP_DIR}/debug-run-full-codegen/index.html"; then
  echo "open view should not use dump names as Step titles" >&2
  exit 1
fi
test -f "${TMP_DIR}/debug-run-full-codegen/views/debug_graph.html"
grep -Fq '<span class="stage-child-title">structured-ops</span>' \
  "${TMP_DIR}/debug-run-full-codegen/views/debug_graph.html"
grep -Fq '<span class="stage-child-title">ascend-kernelize</span>' \
  "${TMP_DIR}/debug-run-full-codegen/views/debug_graph.html"
if grep -Fq '<span class="stage-child-title">识别可 Kernelize 的算子</span>' "${TMP_DIR}/debug-run-full-codegen/views/debug_graph.html"; then
  echo "stage sidebar child buttons should show Step ID, not the readable Step title" >&2
  exit 1
fi
if grep -Fq 'class="stage-child-file"' "${TMP_DIR}/debug-run-full-codegen/views/debug_graph.html"; then
  echo "stage sidebar child buttons should show only Step ID, not dump artifact names" >&2
  exit 1
fi
grep -Fq '<span class="muted">No standalone command</span>' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<td class="command-cell" rowspan="5"><details class="command-detail"><summary><code>ascend-mlir-opt kernelize</code></summary>' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<summary><code>ascend-mlir-opt kernelize</code></summary>' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<code class="command-full">ascend-mlir-opt stages/020-normalize-out.mlir &#x27;--ascend-kernelize=dump-report=true debug-stage=kernelize' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<td class="report-cell" rowspan="5"><a href="views/reports/030-kernelize.report.txt.html">reports/030-kernelize.report.txt</a></td>' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
grep -Fq '<a href="views/reports/030-kernelize.report.txt.html">reports/030-kernelize.report.txt</a>' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
test -f "${TMP_DIR}/debug-run-full-codegen/views/reports/030-kernelize.report.txt.html"
grep -Fq ':root { color-scheme: light; }' "${TMP_DIR}/debug-run-full-codegen/views/reports/030-kernelize.report.txt.html"
grep -Fq '<pre>command:' "${TMP_DIR}/debug-run-full-codegen/views/reports/030-kernelize.report.txt.html"
grep -Fq '<td class="stage-group-cell" rowspan="4">Translate</td>' \
  "${TMP_DIR}/debug-run-full-codegen/index.html"
if grep -Fq '<th>状态</th>' "${TMP_DIR}/debug-run-full-codegen/index.html"; then
  echo "Stage Timeline should not expose status column" >&2
  exit 1
fi
if grep -Fq '<th>Order</th>' "${TMP_DIR}/debug-run-full-codegen/index.html"; then
  echo "Stage Timeline should not expose order column" >&2
  exit 1
fi
if grep -Fq '<th>MLIR</th>' "${TMP_DIR}/debug-run-full-codegen/index.html"; then
  echo "Stage Timeline should merge MLIR into View column" >&2
  exit 1
fi
if grep -Fq '<th>Graph</th>' "${TMP_DIR}/debug-run-full-codegen/index.html"; then
  echo "Stage Timeline should merge Graph into View column" >&2
  exit 1
fi
if grep -Fq '<summary>高级信息：执行命令</summary>' "${TMP_DIR}/debug-run-full-codegen/index.html"; then
  echo "index should merge command details into Stage Timeline" >&2
  exit 1
fi
test -f "${TMP_DIR}/debug-run-full-codegen/graphs/stages/021-kernelize-structured-ops.graph.json"
test -f "${TMP_DIR}/debug-run-full-codegen/graphs/stages/060-compute-lower-out.graph.json"
test -f "${TMP_DIR}/debug-run-full-codegen/graphs/stages/090-cann-signature-out.graph.json"
python3 - "${TMP_DIR}/debug-run-full-codegen/summaries/debug_graph.json" <<'PY'
import json
import pathlib
import sys

graph = json.loads(pathlib.Path(sys.argv[1]).read_text())
phase_groups = [group for group in graph["stage_groups"] if group["kind"] == "phase"]
assert [group["label"] for group in phase_groups] == [
    "Normalize",
    "Kernelize",
    "Schedule",
    "Realize",
    "Translate",
]
translate = next(group for group in phase_groups if group["label"] == "Translate")
kernelize = next(group for group in phase_groups if group["label"] == "Kernelize")
schedule = next(group for group in phase_groups if group["label"] == "Schedule")
realize = next(group for group in phase_groups if group["label"] == "Realize")
assert [step["name"] for step in kernelize["steps"]] == [
    "021-kernelize-structured-ops",
    "022-kernelize-structural-marking",
    "023-kernelize-role-classification",
    "024-kernelize-final-patterns",
    "030-kernelize-out",
]
assert [step["name"] for step in schedule["steps"]] == [
    "031-schedule-cleared",
    "032-schedule-decisions",
    "033-schedule-final",
    "040-schedule-out",
]
assert schedule["steps"][0]["step_info"]["title"] == "清理旧 Schedule 元数据"
assert schedule["steps"][1]["step_info"]["title"] == "选择 tile 和 tail 方案"
assert schedule["steps"][2]["step_info"]["title"] == "挂载 Schedule 契约"
assert schedule["steps"][3]["step_info"]["title"] == "Schedule 输出边界"
assert schedule["steps"][3]["same_as_previous"]["stage"] == "033-schedule-final"
assert [step["name"] for step in realize["steps"]] == [
    "041-realize-planned",
    "042-realize-bufferized",
    "043-realize-memory-space-annotated",
    "050-realize-out",
]
assert [step["name"] for step in translate["steps"]] == [
    "060-compute-lower-out",
    "070-parallelize-out",
    "080-prepare-for-emit-out",
    "090-cann-signature-out",
]
assert "compute-lower" not in [group["name"] for group in phase_groups]
PY
echo "ascend_debug.collect_full_codegen=ok"

mkdir -p "${TMP_DIR}/fake-memory-detail-opt" "${TMP_DIR}/fake-cann-root"
FAKE_CANN_ROOT="$(cd "${TMP_DIR}/fake-cann-root" && pwd -P)"
cat >"${TMP_DIR}/fake-memory-detail-opt/ascend-mlir-opt" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
input="$1"
pass_arg="${2:-}"
cat "${input}"
if [[ "${pass_arg}" == --ascend-realize* ]]; then
  cat >&2 <<'TEXT'
Ascend realize report (ascend-realize)
Realize report
  kernels = 1
StaticMemoryPlan:
  kernel = kernel_0
  mode = "workspace_layout"
  tracked_places = 5
  local_buffers = 2
  live_intervals = 2
  workspace_slots = 2
  peak_usage_known = true
  peak_usage_units = 1
  peak_usage_bytes_known = true
  local_buffer_bytes = 128
  workspace_bytes = 128
  peak_usage_bytes = 128
  capacity_check_deferred = false
  live_interval[0] = value_id=20 start=0 end=1 place=VECIN byte_size=128
  live_interval[1] = value_id=21 start=1 end=2 place=VECIN byte_size=128
  workspace_slot[0] = slot_id=0 value_id=20 offset=0 place=VECIN byte_size=128
  workspace_slot[1] = slot_id=1 value_id=21 offset=0 place=VECIN byte_size=128
MovementPlan:
  kernel = kernel_0
  mode = "movement_planning"
  movement_step[0] = step_id=0 value_id=20 slot_id=0 src=GM dst=VECIN path_selected=true byte_size=128
MemoryRealizationPlan:
  kernel = kernel_0
  mode = "memory_space_materialize"
TEXT
else
  printf 'fake report for %s\n' "${pass_arg}" >&2
fi
SH
chmod +x "${TMP_DIR}/fake-memory-detail-opt/ascend-mlir-opt"
PATH="${TMP_DIR}/fake-memory-detail-opt:${PATH}" \
  ASCEND_HOME_PATH="${FAKE_CANN_ROOT}" \
  ASCEND_SOC_VERSION="SyntheticSoC" \
  ascend-debug collect "${INPUT_MLIR}" \
    --out "${TMP_DIR}/debug-run-memory-detail" \
    --mode quick \
    --memory-detail
grep -Fq 'placement-mode=target-aware' "${TMP_DIR}/debug-run-memory-detail/manifest.json"
grep -Fq 'materialization-mode=plan-only' "${TMP_DIR}/debug-run-memory-detail/manifest.json"
grep -Fq "cann-root=${FAKE_CANN_ROOT}" "${TMP_DIR}/debug-run-memory-detail/manifest.json"
grep -Fq 'soc=SyntheticSoC' "${TMP_DIR}/debug-run-memory-detail/manifest.json"
grep -Fq 'workspace_slot[0]' "${TMP_DIR}/debug-run-memory-detail/reports/040-realize.report.txt"
ascend-debug open "${TMP_DIR}/debug-run-memory-detail" --no-browser >"${TMP_DIR}/ascend-debug-open-memory-detail.txt"
test -f "${TMP_DIR}/debug-run-memory-detail/summaries/memory.json"
test -f "${TMP_DIR}/debug-run-memory-detail/views/summaries/memory.json.html"
grep -Fq '<h1>Memory 摘要</h1>' "${TMP_DIR}/debug-run-memory-detail/views/summaries/memory.json.html"
grep -Fq 'class="ub-allocation-svg"' "${TMP_DIR}/debug-run-memory-detail/views/summaries/memory.json.html"
grep -Fq 'slot 0 value 20' "${TMP_DIR}/debug-run-memory-detail/views/summaries/memory.json.html"
echo "ascend_debug.collect_memory_detail=ok"

cat >"${TMP_DIR}/artifact_manifest.json" <<'JSON'
{
  "kernel_entries": [
    {
      "kernel_id": "kernel_0",
      "kernelKind": "vec",
      "scheduleEntries": [
      ],
      "workspaceSizeBytes": 4096
    }
  ],
  "kernelGraph": {
    "nodes": [
      {"name": "kernel_0"}
    ],
    "edges": []
  }
}
JSON

cat >"${TMP_DIR}/run_manifest.json" <<'JSON'
{
  "backend": "sim",
  "tasks": [
    {
      "task_id": "kernel_0",
      "inputs": [{"name": "input", "path": "input.npy"}],
      "outputs": [{"name": "out0", "shape": [-1, -1], "dtype": "f16"}],
      "workspace_size": 4096
    }
  ]
}
JSON

if ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run-legacy-manifest-alias" \
  --preset deep \
  --pipeline normalize-kernelize \
  --runtime-manifest "${TMP_DIR}/artifact_manifest.json" \
  >"${TMP_DIR}/legacy-manifest-alias.txt" 2>&1; then
  echo "ascend_debug.legacy_manifest_alias=unexpected_success"
  exit 1
fi
grep -Fq -- "--runtime-manifest" "${TMP_DIR}/legacy-manifest-alias.txt"
echo "ascend_debug.legacy_manifest_alias=rejected"

make_npy_pair() {
  local case_dir="$1"
  local rhs_last="$2"
  mkdir -p "${case_dir}/tensors/cpu" "${case_dir}/tensors/npu"
  python3 - "${case_dir}/tensors/cpu/output0.npy" "${case_dir}/tensors/npu/output0.npy" "${rhs_last}" <<'PY'
import pathlib
import struct
import sys

def write_npy(path, values):
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (3,), }"
    header_bytes = header.encode("latin1")
    padding = 16 - ((10 + len(header_bytes) + 1) % 16)
    header_bytes += b" " * padding + b"\n"
    payload = struct.pack("<3f", *values)
    pathlib.Path(path).write_bytes(
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + payload
    )

write_npy(sys.argv[1], [1.0, 2.0, 3.0])
write_npy(sys.argv[2], [1.0, 2.001, float(sys.argv[3])])
PY
}

ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run-graph" \
  --mode quick \
  --artifact-manifest "${TMP_DIR}/artifact_manifest.json" \
  --run-manifest "${TMP_DIR}/run_manifest.json"
test -f "${TMP_DIR}/debug-run-graph/graphs/artifact_manifest.json"
test -f "${TMP_DIR}/debug-run-graph/graphs/run_manifest.json"
test -f "${TMP_DIR}/debug-run-graph/graphs/kernelized.mlir"
test -f "${TMP_DIR}/debug-run-graph/graphs/kernel_dag.svg"
test -f "${TMP_DIR}/debug-run-graph/graphs/kernel_dag.summary.json"
test -f "${TMP_DIR}/debug-run-graph/reports/050-kernel-dag.report.txt"
grep -Fq 'ascend_debug.kernel_dag.kernel_count=1' "${TMP_DIR}/debug-run-graph/reports/050-kernel-dag.report.txt"
echo "ascend_debug.collect_graph=ok"

cat >"${TMP_DIR}/artifact_manifest_kernel_dag.json" <<'JSON'
{
  "kernel_entries": [
    {
      "kernel_id": "kernel_0",
      "kernelKind": "vec",
      "scheduleEntries": [
      ],
      "workspaceSizeBytes": 0
    },
    {
      "kernel_id": "kernel_1",
      "kernelKind": "vec",
      "scheduleEntries": [
      ],
      "workspaceSizeBytes": 0
    },
    {
      "kernel_id": "kernel_2",
      "kernelKind": "mix",
      "scheduleEntries": [
      ],
      "workspaceSizeBytes": 4096
    },
    {
      "kernel_id": "kernel_3",
      "kernelKind": "vec",
      "scheduleEntries": [
      ],
      "workspaceSizeBytes": 0
    },
    {
      "kernel_id": "kernel_4",
      "kernelKind": "vec",
      "scheduleEntries": [
      ],
      "workspaceSizeBytes": 0
    }
  ],
  "kernelGraph": {
    "nodes": [
      {"name": "kernel_0"},
      {"name": "kernel_1"},
      {"name": "kernel_2"},
      {"name": "kernel_3"},
      {"name": "kernel_4"}
    ],
    "edges": [
      {"from": "kernel_0", "to": "kernel_2", "carriedBuffers": ["k0_to_k2"]},
      {"from": "kernel_1", "to": "kernel_2", "carriedBuffers": ["k1_to_k2"]},
      {"from": "kernel_2", "to": "kernel_3", "carriedBuffers": ["k2_to_k3"]},
      {"from": "kernel_3", "to": "kernel_4", "carriedBuffers": ["k3_to_k4"]}
    ]
  }
}
JSON

cat >"${TMP_DIR}/run_manifest_kernel_dag.json" <<'JSON'
{
  "backend": "sim",
  "tasks": [
    {
      "task_id": "kernel_0",
      "inputs": [{"name": "arg0", "path": "input.npy"}],
      "outputs": [{"name": "out0", "shape": [1, 4, 128], "dtype": "f32"}],
      "workspace_size": 0
    },
    {
      "task_id": "kernel_1",
      "inputs": [],
      "outputs": [{"name": "out0", "shape": [128, 384], "dtype": "f32"}],
      "workspace_size": 0
    },
    {
      "task_id": "kernel_2",
      "inputs": [{"name": "arg0"}, {"name": "arg1"}],
      "outputs": [{"name": "out0", "shape": [1, 4, 384], "dtype": "f32"}],
      "workspace_size": 4096
    },
    {
      "task_id": "kernel_3",
      "inputs": [{"name": "arg0"}],
      "outputs": [{"name": "out0", "shape": [1, 4, 128], "dtype": "f32"}],
      "workspace_size": 0
    },
    {
      "task_id": "kernel_4",
      "inputs": [{"name": "arg0"}],
      "outputs": [{"name": "out0", "shape": [1, 4, 128], "dtype": "f32"}],
      "workspace_size": 0
    }
  ]
}
JSON

cat >"${TMP_DIR}/kernelized_kernel_dag.mlir" <<'MLIR'
module {
  func.func @main(%arg0: tensor<1x4x128xf32>) -> tensor<1x4x128xf32> {
    %empty0 = tensor.empty() : tensor<1x4x128xf32>
    %0 = linalg.transpose ins(%arg0 : tensor<1x4x128xf32>) outs(%empty0 : tensor<1x4x128xf32>) permutation = [0, 1, 2] {ascend.kernel = "kernel_0", ascend.op_role = "vector"} : tensor<1x4x128xf32> to tensor<1x4x128xf32>
    %empty1 = tensor.empty() : tensor<128x384xf32>
    %1 = linalg.transpose ins(%empty1 : tensor<128x384xf32>) outs(%empty1 : tensor<128x384xf32>) permutation = [1, 0] {ascend.kernel = "kernel_1", ascend.op_role = "vector"} : tensor<128x384xf32> to tensor<128x384xf32>
    %empty2 = tensor.empty() : tensor<1x4x384xf32>
    %2 = linalg.batch_matmul ins(%0, %1 : tensor<1x4x128xf32>, tensor<128x384xf32>) outs(%empty2 : tensor<1x4x384xf32>) {ascend.kernel = "kernel_2", ascend.op_role = "cube"} -> tensor<1x4x384xf32>
    %empty3 = tensor.empty() : tensor<1x4x128xf32>
    %3 = linalg.generic {iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<1x4x128xf32>) outs(%empty3 : tensor<1x4x128xf32>) attrs = {ascend.kernel = "kernel_3", ascend.op_role = "vector"} {
    ^bb0(%in: f32, %out: f32):
      %exp = math.exp %in : f32
      linalg.yield %exp : f32
    } -> tensor<1x4x128xf32>
    %4 = linalg.generic {iterator_types = ["parallel", "parallel", "parallel"]} ins(%3 : tensor<1x4x128xf32>) outs(%empty3 : tensor<1x4x128xf32>) attrs = {ascend.kernel = "kernel_4", ascend.op_role = "vector"} {
    ^bb0(%in: f32, %out: f32):
      %div = arith.divf %in, %in : f32
      linalg.yield %div : f32
    } -> tensor<1x4x128xf32>
    return %4 : tensor<1x4x128xf32>
  }
}
MLIR

ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run-kernel-dag" \
  --mode quick \
  --artifact-manifest "${TMP_DIR}/artifact_manifest_kernel_dag.json" \
  --run-manifest "${TMP_DIR}/run_manifest_kernel_dag.json" \
  --kernelized-ir "${TMP_DIR}/kernelized_kernel_dag.mlir"
grep -Fq 'ascend_debug.kernel_dag.kernel_count=5' "${TMP_DIR}/debug-run-kernel-dag/reports/050-kernel-dag.report.txt"
python3 - "${TMP_DIR}/debug-run-kernel-dag/graphs/kernel_dag.svg" "${TMP_DIR}/debug-run-kernel-dag/graphs/kernel_dag.summary.json" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

svg_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
ET.parse(svg_path)
svg_text = svg_path.read_text(encoding="utf-8")
summary = json.loads(summary_path.read_text(encoding="utf-8"))

assert summary["kernel_count"] == 5
assert summary["graph_edges"] == 4
assert summary["kind_counts"]["vec"] == 4
assert summary["kind_counts"]["mix"] == 1
assert summary["root_tasks"] == 2
assert summary["prepack_candidate_roots"] == 1
assert summary["critical_path_depth"] == 4
assert len(summary["simple_fusion_edges"]) == 1
assert "kernel_2" in svg_text
assert "batch_matmul" in svg_text
assert "1x4x128" in svg_text
PY
echo "ascend_debug.kernel_dag_internal=ok"

cat >"${TMP_DIR}/debug-run-graph/tensors/manifest.json" <<'JSON'
{
  "schema_version": 1,
  "comparisons": [
    {
      "id": "checkpoint/kernel_0",
      "kernel_id": "kernel_0",
      "task_id": "kernel_0",
      "lhs": "tensors/cpu/output0.npy",
      "rhs": "tensors/npu/output0.npy",
      "atol": 0.01,
      "rtol": 0.01
    }
  ]
}
JSON
make_npy_pair "${TMP_DIR}/debug-run-graph" "3.2"
if ascend-debug diff "${TMP_DIR}/debug-run-graph" >"${TMP_DIR}/ascend-debug-diff-graph.txt" 2>"${TMP_DIR}/ascend-debug-diff-graph.err"; then
  echo "expected ascend-debug diff to fail for graph checkpoint" >&2
  exit 1
fi
grep -Fq 'ascend_debug.diff.failed=1' "${TMP_DIR}/ascend-debug-diff-graph.txt"
ascend-debug locate "${TMP_DIR}/debug-run-graph" >"${TMP_DIR}/ascend-debug-locate-graph.txt"
grep -Fq 'ascend_debug.locate.status=fail' "${TMP_DIR}/ascend-debug-locate-graph.txt"
grep -Fq 'ascend_debug.locate.first_bad_kernel=kernel_0' "${TMP_DIR}/ascend-debug-locate-graph.txt"
grep -Fq 'ascend_debug.locate.first_bad_depth=1' "${TMP_DIR}/ascend-debug-locate-graph.txt"
grep -Fq 'ascend_debug.locate.first_bad_comparison=checkpoint/kernel_0' "${TMP_DIR}/ascend-debug-locate-graph.txt"
grep -Fq 'ascend_debug.locate.upstream_checked_passed=none' "${TMP_DIR}/ascend-debug-locate-graph.txt"
test -f "${TMP_DIR}/debug-run-graph/summaries/locate.json"
cat >"${TMP_DIR}/debug-run-graph/reports/040-realize.report.txt" <<'TEXT'
Realize report
  kernels = 1
BufferizedKernelIR:
  kernel = kernel_0
  mode = "gm_only"
PlacementPlan:
  kernel = kernel_0
  mode = "target_aware"
StaticMemoryPlan:
  kernel = kernel_0
  mode = "workspace_layout"
  tracked_places = 5
  local_buffers = 3
  live_intervals = 3
  workspace_slots = 3
  peak_usage_known = true
  peak_usage_units = 2
  peak_usage_bytes_known = true
  local_buffer_bytes = 384
  workspace_bytes = 256
  peak_usage_bytes = 256
  capacity_check_deferred = false
  live_interval[0] = value_id=10 start=0 end=2 place=VECIN byte_size=128
  live_interval[1] = value_id=11 start=2 end=4 place=VECIN byte_size=128
  live_interval[2] = value_id=12 start=1 end=3 place=VECIN byte_size=128
  workspace_slot[0] = slot_id=0 value_id=10 offset=0 place=VECIN byte_size=128
  workspace_slot[1] = slot_id=1 value_id=11 offset=0 place=VECIN byte_size=128
  workspace_slot[2] = slot_id=2 value_id=12 offset=128 place=VECIN byte_size=128
MovementPlan:
  kernel = kernel_0
  mode = "movement_planning"
  movement_step[0] = step_id=0 value_id=10 slot_id=0 src=GM dst=VECIN path_selected=true byte_size=128
MemoryRealizationPlan:
  kernel = kernel_0
  mode = "memory_space_materialize"
TEXT
ascend-debug open "${TMP_DIR}/debug-run-graph" --no-browser >"${TMP_DIR}/ascend-debug-open-graph.txt"
grep -Fq '<a class="primary-debug-link" href="views/debug_graph.html">打开调试工作台</a>' "${TMP_DIR}/debug-run-graph/index.html"
grep -Fq '<h2>Stage Timeline</h2>' "${TMP_DIR}/debug-run-graph/index.html"
grep -Fq '<a href="views/debug_graph.html?stage=29">Graph</a>' "${TMP_DIR}/debug-run-graph/index.html"
test -f "${TMP_DIR}/debug-run-graph/summaries/debug_graph.json"
test -f "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '<h1>Ascend Debug 调试工作台</h1>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '<svg id="unified-debug-graph-svg"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '<button class="mode-tab active" data-mode="stage"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '<button class="mode-tab" data-mode="kernel"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="stage-list"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="stage-phase-controls"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="graph-search"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="graph-fit"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="graph-reset"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '<button id="graph-reset" class="graph-tool-button" type="button">Reset</button>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function defaultStageGraphViewState()' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function resetStageGraphViewState()' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'stageGraphViewState = defaultStageGraphViewState();' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="sidebar-toggle"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="inspector-resizer"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="source-reader-overlay"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="inspector-detail"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="stage-diff-panel"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="graph-workspace-data"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="panel-title-block"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="panel-tools-column"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="panel-control-strip"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.panel-header { display: grid; grid-template-columns: minmax(0, 1fr) minmax(17rem, 24rem);' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.panel-tools-column { align-self: stretch; justify-self: end; width: 100%; display: grid; grid-template-rows: auto minmax(0, 1fr);' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.panel-control-strip { min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr);' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.stage-graph-controls { order: 1;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.panel-control-strip .stage-phase-controls { order: 2; align-self: end;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.step-explanation-grid { display: grid; grid-template-columns: max-content minmax(0, 1fr);' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.step-explanation-grid dd { margin: 0; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '.panel-control-strip { grid-column: 1 / -1;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage graph controls should live in the right header column, not a full-width row" >&2
  exit 1
fi
if grep -Fq '.step-explanation-grid { display: grid; grid-template-columns: 4.8rem minmax(0, 1fr) 4.8rem minmax(0, 1fr);' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "step explanations should use one label/value pair per row, not two compressed pairs per row" >&2
  exit 1
fi
grep -Fq -- '--inspector-width' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="layout-resizer"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="source-code"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.source-code { margin: 0; white-space: pre; overflow: auto;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.graph-panel { min-width: 0; overflow: hidden; display: grid; grid-template-rows: auto minmax(0, 1fr);' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.graph-canvas-wrap { overflow: auto; min-height: 22rem; height: auto;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '.graph-canvas-wrap { overflow: auto; height: calc(100vh - 15rem);' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "graph canvas should fill the graph panel instead of using a shorter fixed viewport height" >&2
  exit 1
fi
grep -Fq 'class="source-expand-button"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="stage-tree-group"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="stage-button stage-group-parent' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="stage-button stage-child-button' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'data-stage-child-indices=' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '<span class="stage-group-main">Kernelize</span>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '<span class="stage-group-main"><span>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage group labels should not duplicate the numeric order prefix" >&2
  exit 1
fi
if grep -Eq '<span>[0-9]+</span>[0-9]{3}-' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage child labels should not duplicate order and numeric stage-name prefixes" >&2
  exit 1
fi
grep -Fq '.stage-group-parent.parent-active' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.sidebar-body.kernel-mode .stage-navigation' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'sidebarBody.classList.toggle("kernel-mode", mode === "kernel")' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function stageNavigationSequence' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'const paddedOrder = orderText.padStart(3, "0");' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'stageName.startsWith(`${paddedOrder}-`)' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '${escapeHtml(stageDiffTitle(diff.from_stage))} -> ${escapeHtml(stageDiffTitle(diff.to_stage))}' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'document.getElementById("graph-title").textContent = stageGraphHeaderTitle(stage);' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'document.getElementById("graph-subtitle").textContent = `${graph.node_count} 个节点，${graph.edge_count} 条边，${graph.kernel_count} 个 Kernel`;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '["Stage", stageStageTitle(stage)]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '["Artifact", stage.path]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'return info.title || (stage && stage.step) || "未命名 Step";' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function stageStepId(stage)' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function stageGraphHeaderTitle(stage)' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'return `${stageStageTitle(stage)} / ${stageStepId(stage)}`;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function stageDiffTitle(stage)' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'return `Dump: ${stageBriefLabel(stage)}`;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '${escapeHtml(diff.from_stage.order)} ${escapeHtml(diff.from_stage.name)} ->' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage diff summary should not duplicate order and numeric stage-name prefixes" >&2
  exit 1
fi
if grep -Fq 'return info.title || (stage && (stage.step || stage.name)) || "Stage";' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stageStepTitle should not fall back to dump artifact names" >&2
  exit 1
fi
if grep -Fq '["stage", stageBriefLabel(stage)]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage origin should show phase as Stage and keep dump identity in a separate row" >&2
  exit 1
fi
if grep -Fq '["Step", stageStepTitle(stage)]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage origin should not repeat the Step title shown in the graph header" >&2
  exit 1
fi
if grep -Fq '["Step ID", stage.step || "无"]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage origin should not repeat the Step ID shown in the graph header" >&2
  exit 1
fi
if grep -Fq '["Dump", stageBriefLabel(stage)]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage origin and node detail should avoid duplicate dump rows; artifact path is enough" >&2
  exit 1
fi
if grep -Fq '["phase/step", [stage.phase, stage.step].filter(Boolean).join(" / ")]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage origin should split phase and step into separate rows" >&2
  exit 1
fi
if grep -Fq '["Stage", `${stage.order} ${stage.name}`]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "node detail should label dumped stage identity as Dump, not Stage" >&2
  exit 1
fi
if grep -Fq '`Stage Graph: ${group.label} / ${stageLabel}`' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage graph title should use the Chinese step title without the English Stage Graph prefix or phase label" >&2
  exit 1
fi
if grep -Fq '`${stageBriefLabel(stage)} | ${graph.node_count} 个节点' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage graph subtitle should not repeat the dumped stage file name" >&2
  exit 1
fi
if grep -Fq '<small class="stage-child-file">{_cell(child.get("name"))}</small>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage sidebar should label dump artifact names explicitly" >&2
  exit 1
fi
if grep -Fq 'Dump: kernelize-in' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage sidebar should not show dump names when Step ID is unavailable" >&2
  exit 1
fi
grep -Fq '上一页' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '下一页' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'addNavButton("上一页", sequence[currentPosition - 1])' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'addNavButton("下一页", sequence[currentPosition + 1])' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'type="button">${label}</button>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq 'type="button">${label} ${escapeHtml(stageBriefLabel(item))}</button>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage pager buttons should show only 上一步/下一步 without target text" >&2
  exit 1
fi
grep -Fq 'Kernel DAG 为空' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '当前 run 未收集 Kernel DAG 产物' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq 'class="stage-boundary-details"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage navigation should be a single expanded tree, not a collapsed boundary drawer" >&2
  exit 1
fi
if grep -Fq '显示边界快照' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage navigation should merge boundary snapshots into the tree" >&2
  exit 1
fi
if grep -Fq 'class="stage-child-purpose"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "stage sidebar child buttons should omit long purpose text; main panel owns step explanations" >&2
  exit 1
fi
grep -Fq '输入同 19 normalize-out' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="function-frame"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function functionFramesForGraph' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="node-badge"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function renderSemanticAttrSections' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function resolveKernelDagEntry' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '属性分组' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq 'class="badge-list"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "node inspector should not duplicate module facts as a non-clickable badge list" >&2
  exit 1
fi
if grep -Fq '所属 Kernel' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "node inspector should merge Kernel DAG facts into the Kernel attribute group" >&2
  exit 1
fi
grep -Fq '.detail-grid { display: grid; grid-template-columns: 1fr;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.semantic-group .detail-grid { gap: 0.34rem; }' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '.semantic-group .detail-grid { grid-template-columns: minmax(7.8rem, 42%) minmax(0, 1fr); }' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "semantic attribute groups should use single-column rows to avoid narrow two-column wrapping" >&2
  exit 1
fi
grep -Fq 'class="detail-row"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '<div class="detail-row">' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.detail-row { border: 1px solid #e3e8ef;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.detail-label { color: #475569; font-weight: 800; font-size: 0.72rem;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.detail-value { color: #111827; min-width: 0; white-space: nowrap; overflow-x: auto;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '["op_role", kernel.role]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '["op_roles", kernel.roles]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '["output_shape", dagKernelNode.output_shape]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '["kernel_dag_id", dagKernelId]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function movementPhasesHtml(phases)' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="tile-param-list"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="tile-param-card"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.tile-param-field { min-width: 0; display: grid; grid-template-columns: 5.6rem minmax(0, 1fr);' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '.tile-param-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "tile param fields should use readable single-column rows in the narrow inspector" >&2
  exit 1
fi
grep -Fq 'class="phase-legend"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'GM/UB 间搬运输入、输出或中间值' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '${phase}: ${movementPhaseDescription(phase)}' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "movement legend value should not repeat the phase name already shown in the left column" >&2
  exit 1
fi
if grep -Fq 'phase-chip-list' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "movement group should not repeat phases as both chips and legend rows" >&2
  exit 1
fi
if grep -Fq '这些 phase 描述当前 op 在调度/搬运计划中的职责' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "movement group should avoid a redundant explanatory sentence above the phase rows" >&2
  exit 1
fi
grep -Fq 'movement phases' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq 'move data_copy/vector_compute/write_back' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "movement phase details should live in the Movement group, not in raw badge text" >&2
  exit 1
fi
if grep -Fq 'function tileParamText(params)' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "tile params should render as structured rows instead of a semicolon-joined long line" >&2
  exit 1
fi
grep -Fq '["position.kind", position.kind]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '["角色", kernel.role]' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "semantic attr labels should use raw field names, not Chinese display names" >&2
  exit 1
fi
grep -Fq 'function beginCanvasPan' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function isBlankCanvasPanTarget' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function installInspectorResize' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function setSidebarCollapsed' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function expandSourceReader' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function highlightMlir' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'localStorage.setItem("ascendDebugInspectorWidth"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'localStorage.setItem("ascendDebugSidebarCollapsed"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'event.button !== 0 && event.button !== 2' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '!isBlankCanvasPanTarget(event.target)' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'const GRAPH_CANVAS_PADDING = 160' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'data-canvas-padding' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function scrollGraphToDefaultOrigin' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'padding * scale - GRAPH_DEFAULT_MARGIN' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '<g class="graph-content" transform="translate(${GRAPH_CANVAS_PADDING},${GRAPH_CANVAS_PADDING})">' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function applyGraphScale' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function fitGraphToView' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function searchActiveGraph' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function applyStageNeighborhood' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function clearStageNeighborhood' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function collectReachableNeighborhood' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'const ancestorNeighborhood = ["upstream", "both", "direct"].includes(stageGraphViewState.highlightMode)' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'collectReachableNeighborhood(activeEdges, nodeId, "ancestors", highlightDepth)' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'collectReachableNeighborhood(activeEdges, nodeId, "descendants", highlightDepth)' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="highlight-mode-controls"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'data-highlight-mode="upstream"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="focus-toggle"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="highlight-depth"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'data-edge-kind-filter="resource_effect"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="fold-helper-toggle"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function renderPathSummary' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function renderProvenanceSection' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function directGraphContext' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function kernelRuntimeStatus' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function tensorDiffForKernel' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function renderKernelLineage' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '<h3>Provenance</h3>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'provenance-edge-chip' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function updateGraphUrlState' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'let pendingStageSelection = null;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'let requestedNodeConsumed = false;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function selectedStageNodeSignature' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function resolveStageNodeSelection' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'pendingStageSelection = selectedStageNodeSignature(activeStage(), selectedKey);' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'stageNeighborhoodActive = true;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function applyEdgeAndHelperFilters' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'data-edge-from=' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'data-edge-kind=' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'neighborhood-node' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'dimmed' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'id="step-explanation"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq 'function renderStepInspectorSection' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "node inspector should not carry a second step explanation renderer" >&2
  exit 1
fi
if grep -Fq '<h3>Step Explanation</h3>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "node inspector should not duplicate the center step explanation" >&2
  exit 1
fi
grep -Fq 'selectStageNode(stage, graph, preferred, {neighborhood: shouldRefreshNeighborhood, updateUrl: false})' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'addEventListener("contextmenu"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'addEventListener("wheel"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'canvas.classList.add("panning")' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'diff-added' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'Stage Diff' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'Graph Audit' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'suspicious_isolated' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'dangling_effect' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="audit-summary-strip' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="audit-metrics"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="audit-detail-group audit-issues"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'class="audit-detail-list"' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq '.audit-metric-label { min-width: 0; overflow: hidden; text-overflow: ellipsis;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '<strong>${escapeHtml(suspicious)}</strong>suspicious_isolated' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "Graph Audit should not render long audit keys inside cramped diff pills" >&2
  exit 1
fi
grep -Fq 'function renderKernelDag' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'function renderStagePhaseControls' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'Kernel DAG' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '<h2>Kernel DAG</h2>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "node inspector should not duplicate the global Kernel DAG table" >&2
  exit 1
fi
if grep -Fq '<h2>诊断叠加</h2>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "node inspector should not duplicate global diagnostic overlays" >&2
  exit 1
fi
grep -Fq 'Tensor Diff' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'Memory' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq 'tensor&lt;' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "debug graph embedded JSON should not show HTML-escaped MLIR types" >&2
  exit 1
fi
python3 - "${TMP_DIR}/debug-run-graph/summaries/debug_graph.json" <<'PY'
import json
import pathlib
import sys

graph = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert graph["schema_version"] == 1
assert graph["visual_kind"] == "unified-debug-workspace"
assert graph["primary_stage"]["name"] == "kernelize-out"
assert graph["primary_stage"]["stage_view_path"] == "views/stages/029-kernelize-out.mlir.html"
assert graph["stage_count"] == 9
assert len(graph["stage_groups"]) == 5
kernelize = next(item for item in graph["stage_groups"] if item["name"] == "kernelize")
assert kernelize["input_stage"]["name"] == "kernelize-in"
assert kernelize["output_stage"]["name"] == "kernelize-out"
assert kernelize["input_same_as_previous_output"] is True
assert kernelize["previous_output_stage"]["name"] == "normalize-out"
primary_graph = graph["primary_stage"]["graph"]
assert "connectivity" in primary_graph
assert primary_graph["connectivity"]["suspicious_isolated_count"] == 0
assert "stage_connectivity" in graph
assert len(graph["stage_connectivity"]) == graph["stage_count"]
assert all("isolated_count" in item for item in graph["stage_connectivity"])
assert all("suspicious_isolated_count" in item for item in graph["stage_connectivity"])
assert all("dangling_effect_count" in item for item in graph["stage_connectivity"])
assert primary_graph["functions"][0]["name"] == "elementwise"
assert primary_graph["functions"][0]["node_ids"] == [node["id"] for node in primary_graph["nodes"]]
schedule_stage = next(item for item in graph["stages"] if item["name"] == "schedule-out")
scheduled = next(
    node
    for node in schedule_stage["graph"]["nodes"]
    if node["op_name"] == "linalg.generic" and node.get("kernel_id") == "kernel_0"
)
semantic = scheduled["semantic_attrs"]
assert semantic["schedule"]["template"] == "single_tile_per_block"
assert semantic["schedule"]["schedule_contract"] == "generic_tiled_loop"
assert semantic["movement"]["phases"] == ["data_copy", "vector_compute", "write_back"]
assert "movement phases" in scheduled["badges"]
assert "move data_copy/vector_compute/write_back" not in scheduled["badges"]
assert len(graph["stage_diffs"]) == graph["stage_count"] - 1
assert all("added_count" in item for item in graph["stage_diffs"])
assert any(item["to_stage"]["name"] == "kernelize-out" for item in graph["stage_diffs"])
assert graph["kernel_dag"]["kernel_count"] == 1
assert graph["kernel_dag"]["nodes"]["kernel_0"]["output_shape"] == "?x?"
assert graph["overlays"]["tensor_diff"]["status"] == "fail"
assert graph["overlays"]["locate"]["first_bad_kernel"] == "kernel_0"
assert graph["overlays"]["memory"]["peak_workspace_bytes"] == 256
PY
for hidden_heading in '图数据' 'Kernel</h2>' 'Tensor Diff' 'Locate' 'Memory</h2>' '摘要' '报告' 'Stage 演进图'; do
  if grep -Fq "<h2>${hidden_heading}" "${TMP_DIR}/debug-run-graph/index.html"; then
    echo "index should not expose duplicated ${hidden_heading} section" >&2
    exit 1
  fi
done
grep -Fq '节点详情' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'Region Body' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'node.region_body' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq 'Body 摘要' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "node inspector should show concrete body instead of body summary" >&2
  exit 1
fi
if grep -Fq '当前节点没有 region body' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "node inspector should not show no-region-body placeholder" >&2
  exit 1
fi
grep -Fq '查看完整 MLIR' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'Kernel 详情' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'stage.stage_view_path' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'URLSearchParams' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
if grep -Fq '<summary>原始产物</summary>' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "unexpected raw artifact drawer in node inspector" >&2
  exit 1
fi
if grep -Fq '原始 MLIR' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "unexpected duplicate raw MLIR action in node inspector" >&2
  exit 1
fi
if grep -Fq '独立 Stage Graph' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "unexpected duplicate stage graph action in node inspector" >&2
  exit 1
fi
if grep -Fq 'Stage View' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "unexpected duplicate Stage View action" >&2
  exit 1
fi
if grep -Fq 'views/graphs/stages/' "${TMP_DIR}/debug-run-graph/index.html"; then
  echo "index should link stage graph rows back to unified debug workspace" >&2
  exit 1
fi
if grep -Fq 'views/summaries/debug_graph.json.html' "${TMP_DIR}/debug-run-graph/index.html"; then
  echo "index should not expose debug graph summary JSON view" >&2
  exit 1
fi
grep -Fq '<a href="views/stages/029-kernelize-out.mlir.html">Text</a>' "${TMP_DIR}/debug-run-graph/index.html"
grep -Fq '../views/kernels/kernel_0.html' "${TMP_DIR}/debug-run-graph/graphs/kernel_dag.svg"
if grep -Fq -- '-1x-1' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"; then
  echo "dynamic kernel shapes should be shown as ?x?, not -1x-1" >&2
  exit 1
fi
test -f "${TMP_DIR}/debug-run-graph/views/stages/029-kernelize-out.mlir.html"
test -f "${TMP_DIR}/debug-run-graph/views/graphs/kernelized.mlir.html"
test ! -e "${TMP_DIR}/debug-run-graph/views/graphs/kernel_dag.summary.json.html"
test ! -e "${TMP_DIR}/debug-run-graph/views/graphs/stages/029-kernelize-out.graph.html"
test ! -e "${TMP_DIR}/debug-run-graph/views/summaries/debug_graph.json.html"
test -f "${TMP_DIR}/debug-run-graph/summaries/memory.json"
test -f "${TMP_DIR}/debug-run-graph/views/summaries/memory.json.html"
test -f "${TMP_DIR}/debug-run-graph/views/summaries/tensor_diff.json.html"
test -f "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"
grep -Fq '<input id="search"' "${TMP_DIR}/debug-run-graph/views/stages/029-kernelize-out.mlir.html"
grep -Fq 'class="code-table text-code-table"' "${TMP_DIR}/debug-run-graph/views/stages/029-kernelize-out.mlir.html"
grep -Fq 'color: #dbeafe' "${TMP_DIR}/debug-run-graph/views/stages/029-kernelize-out.mlir.html"
grep -Fq '<span class="line-number">1</span>' "${TMP_DIR}/debug-run-graph/views/stages/029-kernelize-out.mlir.html"
grep -Fq 'ascend.kernel' "${TMP_DIR}/debug-run-graph/views/stages/029-kernelize-out.mlir.html"
grep -Fq '<h1>kernel_0</h1>' "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"
grep -Fq 'workspace_size' "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"
grep -Fq 'workspace_size</th><td>4096' "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"
grep -Fq 'output_shape</th><td>?x?' "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"
grep -Fq 'MLIR Ops' "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"
grep -Fq '../graphs/kernelized.mlir.html#L' "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"
if grep -Fq 'DAG 摘要 JSON' "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"; then
  echo "kernel detail should not expose raw DAG JSON view" >&2
  exit 1
fi
grep -Fq '<h1>Memory 摘要</h1>' "${TMP_DIR}/debug-run-graph/views/summaries/memory.json.html"
grep -Fq 'class="memory-kernel-viz"' "${TMP_DIR}/debug-run-graph/views/summaries/memory.json.html"
grep -Fq 'class="ub-allocation-svg"' "${TMP_DIR}/debug-run-graph/views/summaries/memory.json.html"
grep -Fq 'id="kernel-kernel_0"' "${TMP_DIR}/debug-run-graph/views/summaries/memory.json.html"
grep -Fq 'reuse-group' "${TMP_DIR}/debug-run-graph/views/summaries/memory.json.html"
grep -Fq 'movement-edge' "${TMP_DIR}/debug-run-graph/views/summaries/memory.json.html"
grep -Fq 'value 10' "${TMP_DIR}/debug-run-graph/views/summaries/memory.json.html"
grep -Fq 'slot 0' "${TMP_DIR}/debug-run-graph/views/summaries/memory.json.html"
grep -Fq '<h1>Tensor Diff 摘要</h1>' "${TMP_DIR}/debug-run-graph/views/summaries/tensor_diff.json.html"
grep -Fq '<h1>Locate 摘要</h1>' "${TMP_DIR}/debug-run-graph/views/summaries/locate.json.html"
test ! -e "${TMP_DIR}/debug-run-graph/views/summaries/debug_graph.json.html"
grep -Fq '内存视图' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'summaries/memory.json.html#kernel-' "${TMP_DIR}/debug-run-graph/views/debug_graph.html"
grep -Fq 'UB 分配' "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"
grep -Fq '../summaries/memory.json.html#kernel-kernel_0' "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"
grep -Fq 'class="ub-allocation-svg"' "${TMP_DIR}/debug-run-graph/views/kernels/kernel_0.html"
cp -R "${TMP_DIR}/debug-run-graph" "${TMP_DIR}/debug-run-graph-physical-kernel"
rm -rf "${TMP_DIR}/debug-run-graph-physical-kernel/views/kernels"
python3 - "${TMP_DIR}/debug-run-graph-physical-kernel/graphs/kernel_dag.summary.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
summary = json.loads(path.read_text())
old = "kernel_0"
new = "physical_kernel"
summary["nodes"] = {new if key == old else key: value for key, value in summary["nodes"].items()}
for field in (
    "critical_path",
    "leaf_task_ids",
    "prepack_candidate_root_ids",
    "root_task_ids",
    "runtime_input_root_ids",
):
    if isinstance(summary.get(field), list):
        summary[field] = [new if item == old else item for item in summary[field]]
for edge in summary.get("edges", []):
    if edge.get("from") == old:
        edge["from"] = new
    if edge.get("to") == old:
        edge["to"] = new
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
ascend-debug open "${TMP_DIR}/debug-run-graph-physical-kernel" --no-browser >"${TMP_DIR}/ascend-debug-open-physical-kernel.txt"
test -f "${TMP_DIR}/debug-run-graph-physical-kernel/views/kernels/physical_kernel.html"
test -f "${TMP_DIR}/debug-run-graph-physical-kernel/views/kernels/kernel_0.html"
grep -Fq 'url=physical_kernel.html' "${TMP_DIR}/debug-run-graph-physical-kernel/views/kernels/kernel_0.html"
grep -Fq 'function kernelDetailHref' "${TMP_DIR}/debug-run-graph-physical-kernel/views/debug_graph.html"
grep -Fq 'function resolveKernelDagEntry' "${TMP_DIR}/debug-run-graph-physical-kernel/views/debug_graph.html"
grep -Fq 'candidateView === targetView' "${TMP_DIR}/debug-run-graph-physical-kernel/views/debug_graph.html"
python3 - "${TMP_DIR}/debug-run-graph-physical-kernel/summaries/debug_graph.json" <<'PY'
import json
import pathlib
import sys

graph = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert graph["kernel_detail_views"]["physical_kernel"] == "views/kernels/physical_kernel.html"
assert graph["kernel_detail_views"]["kernel_0"] == "views/kernels/physical_kernel.html"
PY
python3 - "${TMP_DIR}/debug-run-graph/summaries/memory.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert summary["analysis_level"] == "realize-memory-plan"
assert summary["source"] == "reports/040-realize.report.txt"
assert summary["peak_workspace_bytes"] == 256
assert summary["total_workspace_bytes"] == 256
assert summary["slot_reuse_group_count"] == 1
kernel = summary["kernels"][0]
assert kernel["kernel_id"] == "kernel_0"
assert kernel["reuse_groups"] == [
    {"place": "VECIN", "offset": 0, "slot_ids": [0, 1], "value_ids": [10, 11]}
]
assert kernel["peak_timeline"][1]["usage_bytes"] == 256
assert kernel["peak_timeline"][1]["active_values"] == [10, 12]
assert kernel["movement_edges"][0]["dst"] == "VECIN"
PY
mkdir -p "${TMP_DIR}/debug-run-memory-coverage/stages" \
  "${TMP_DIR}/debug-run-memory-coverage/graphs" \
  "${TMP_DIR}/debug-run-memory-coverage/reports"
printf 'module {}\n' >"${TMP_DIR}/debug-run-memory-coverage/stages/000-source.mlir"
cat >"${TMP_DIR}/debug-run-memory-coverage/manifest.json" <<'JSON'
{
  "schema_version": 1,
  "tool": "ascend-debug",
  "preset": "deep",
  "pipeline": "normalize-kernelize",
  "stages": [
    {"order": 0, "name": "source", "path": "stages/000-source.mlir"}
  ],
  "reports": [
    {"stage": "realize", "path": "reports/040-realize.report.txt"}
  ],
  "graphs": [
    {"kind": "kernel-dag-summary", "path": "graphs/kernel_dag.summary.json"}
  ]
}
JSON
cat >"${TMP_DIR}/debug-run-memory-coverage/graphs/kernel_dag.summary.json" <<'JSON'
{
  "schema_version": 1,
  "nodes": {
    "kernel_0": {"depth": 1, "kind": "vec", "workspace_size": 256},
    "kernel_1": {"depth": 2, "kind": "vec", "workspace_size": 0}
  },
  "edges": [
    {"from": "kernel_0", "to": "kernel_1"}
  ]
}
JSON
cat >"${TMP_DIR}/debug-run-memory-coverage/reports/040-realize.report.txt" <<'TEXT'
Realize report
  kernels = 1
StaticMemoryPlan:
  kernel = kernel_0
  mode = "workspace_layout"
  tracked_places = 3
  local_buffers = 1
  live_intervals = 1
  workspace_slots = 1
  peak_usage_known = true
  peak_usage_units = 1
  peak_usage_bytes_known = true
  local_buffer_bytes = 256
  workspace_bytes = 256
  peak_usage_bytes = 256
  capacity_check_deferred = false
  live_interval[0] = value_id=10 start=0 end=1 place=VECIN byte_size=256
  workspace_slot[0] = slot_id=0 value_id=10 offset=0 place=VECIN byte_size=256
TEXT
ascend-debug open "${TMP_DIR}/debug-run-memory-coverage" --no-browser >"${TMP_DIR}/ascend-debug-open-memory-coverage.txt"
test -f "${TMP_DIR}/debug-run-memory-coverage/views/summaries/memory.json.html"
grep -Fq '<h2>Kernel 覆盖</h2>' "${TMP_DIR}/debug-run-memory-coverage/views/summaries/memory.json.html"
grep -Fq '<td><a href="../kernels/kernel_0.html">kernel_0</a></td><td>1</td><td>256</td><td>realize-slot-plan</td>' "${TMP_DIR}/debug-run-memory-coverage/views/summaries/memory.json.html"
grep -Fq '<td><a href="../kernels/kernel_1.html">kernel_1</a></td><td>2</td><td>0</td><td>no-workspace</td>' "${TMP_DIR}/debug-run-memory-coverage/views/summaries/memory.json.html"
python3 - "${TMP_DIR}/debug-run-memory-coverage/summaries/memory.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert summary["kernel_count"] == 2
assert summary["detailed_kernel_count"] == 1
assert summary["unplanned_kernel_count"] == 1
assert summary["kernel_coverage"][0]["kernel_id"] == "kernel_0"
assert summary["kernel_coverage"][0]["memory_plan_status"] == "realize-slot-plan"
assert summary["kernel_coverage"][1]["kernel_id"] == "kernel_1"
assert summary["kernel_coverage"][1]["memory_plan_status"] == "no-workspace"
PY
if grep -Fq 'status=fail; first_bad_kernel=' "${TMP_DIR}/debug-run-graph/index.html"; then
  echo "locate dashboard should not render raw key-value status line" >&2
  exit 1
fi
echo "ascend_debug.open_kernel=ok"
echo "ascend_debug.open_graph=ok"

mkdir -p "${TMP_DIR}/debug-run-locate/summaries" "${TMP_DIR}/debug-run-locate/graphs"
cat >"${TMP_DIR}/debug-run-locate/summaries/tensor_diff.json" <<'JSON'
{
  "schema_version": 1,
  "tool": "ascend-debug",
  "status": "fail",
  "comparison_count": 3,
  "failed_count": 2,
  "comparisons": [
    {"id": "checkpoint/kernel_3", "status": "fail", "kernel_id": "kernel_3", "task_id": "kernel_3", "max_abs_error": 0.3},
    {"id": "checkpoint/kernel_1", "status": "fail", "kernel_id": "kernel_1", "task_id": "kernel_1", "max_abs_error": 0.1},
    {"id": "checkpoint/kernel_0", "status": "pass", "kernel_id": "kernel_0", "task_id": "kernel_0", "max_abs_error": 0.0}
  ]
}
JSON
cat >"${TMP_DIR}/debug-run-locate/graphs/kernel_dag.summary.json" <<'JSON'
{
  "schema_version": 1,
  "nodes": {
    "kernel_0": {"depth": 1, "kind": "vec"},
    "kernel_1": {"depth": 2, "kind": "vec"},
    "kernel_3": {"depth": 4, "kind": "vec"}
  },
  "edges": [
    {"from": "kernel_0", "to": "kernel_1"},
    {"from": "kernel_1", "to": "kernel_3"}
  ]
}
JSON
ascend-debug locate "${TMP_DIR}/debug-run-locate" >"${TMP_DIR}/ascend-debug-locate.txt"
grep -Fq 'ascend_debug.locate.status=fail' "${TMP_DIR}/ascend-debug-locate.txt"
grep -Fq 'ascend_debug.locate.failed_kernels=2' "${TMP_DIR}/ascend-debug-locate.txt"
grep -Fq 'ascend_debug.locate.first_bad_kernel=kernel_1' "${TMP_DIR}/ascend-debug-locate.txt"
grep -Fq 'ascend_debug.locate.first_bad_depth=2' "${TMP_DIR}/ascend-debug-locate.txt"
grep -Fq 'ascend_debug.locate.first_bad_comparison=checkpoint/kernel_1' "${TMP_DIR}/ascend-debug-locate.txt"
grep -Fq 'ascend_debug.locate.upstream_checked_passed=kernel_0' "${TMP_DIR}/ascend-debug-locate.txt"
python3 - "${TMP_DIR}/debug-run-locate/summaries/locate.json" <<'PY'
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert summary["schema_version"] == 1
assert summary["status"] == "fail"
assert summary["first_bad_kernel"] == "kernel_1"
assert summary["first_bad_depth"] == 2
assert summary["first_bad_comparison"]["id"] == "checkpoint/kernel_1"
assert summary["failed_kernel_ids"] == ["kernel_1", "kernel_3"]
assert summary["passed_kernel_ids"] == ["kernel_0"]
assert summary["first_bad_context"]["direct_upstream"] == ["kernel_0"]
assert summary["first_bad_context"]["direct_downstream"] == ["kernel_3"]
assert summary["first_bad_context"]["upstream_checked_passed"] == ["kernel_0"]
assert summary["first_bad_context"]["downstream_failed"] == ["kernel_3"]
assert summary["first_bad_context"]["unchecked_direct_upstream"] == []
PY
echo "ascend_debug.locate=ok"

cat >"${TMP_DIR}/tensor-manifest-pass.json" <<'JSON'
{
  "schema_version": 1,
  "comparisons": [
    {
      "id": "final/output0",
      "lhs": "tensors/cpu/output0.npy",
      "rhs": "tensors/npu/output0.npy",
      "atol": 0.01,
      "rtol": 0.01
    }
  ]
}
JSON

mkdir -p "${TMP_DIR}/debug-run-diff-pass/tensors"
cp "${TMP_DIR}/tensor-manifest-pass.json" "${TMP_DIR}/debug-run-diff-pass/tensors/manifest.json"
make_npy_pair "${TMP_DIR}/debug-run-diff-pass" "3.002"
ascend-debug diff "${TMP_DIR}/debug-run-diff-pass" >"${TMP_DIR}/ascend-debug-diff-pass.txt"
grep -Fq 'ascend_debug.diff.comparisons=1' "${TMP_DIR}/ascend-debug-diff-pass.txt"
grep -Fq 'ascend_debug.diff.failed=0' "${TMP_DIR}/ascend-debug-diff-pass.txt"
grep -Fq 'ascend_debug.diff.status=pass' "${TMP_DIR}/ascend-debug-diff-pass.txt"
test -f "${TMP_DIR}/debug-run-diff-pass/summaries/tensor_diff.json"
echo "ascend_debug.diff_pass=ok"

mkdir -p "${TMP_DIR}/debug-run-diff-fail/tensors"
cp "${TMP_DIR}/tensor-manifest-pass.json" "${TMP_DIR}/debug-run-diff-fail/tensors/manifest.json"
make_npy_pair "${TMP_DIR}/debug-run-diff-fail" "3.2"
if ascend-debug diff "${TMP_DIR}/debug-run-diff-fail" >"${TMP_DIR}/ascend-debug-diff-fail.txt" 2>"${TMP_DIR}/ascend-debug-diff-fail.err"; then
  echo "expected ascend-debug diff to fail for mismatched tensors" >&2
  exit 1
fi
grep -Fq 'ascend_debug.diff.comparisons=1' "${TMP_DIR}/ascend-debug-diff-fail.txt"
grep -Fq 'ascend_debug.diff.failed=1' "${TMP_DIR}/ascend-debug-diff-fail.txt"
grep -Fq 'ascend_debug.diff.status=fail' "${TMP_DIR}/ascend-debug-diff-fail.txt"
test -f "${TMP_DIR}/debug-run-diff-fail/summaries/tensor_diff.json"
if grep -Fq 'Traceback' "${TMP_DIR}/ascend-debug-diff-fail.err"; then
  echo "unexpected traceback for tensor diff mismatch" >&2
  exit 1
fi
echo "ascend_debug.diff_fail=ok"

ascend-debug open "${TMP_DIR}/debug-run-deep" --no-browser >"${TMP_DIR}/ascend-debug-open-deep.txt"
test -f "${TMP_DIR}/debug-run-deep/index.html"
grep -Fq '<span>mode</span><strong>quick</strong>' "${TMP_DIR}/debug-run-deep/index.html"
grep -Fq '<thead><tr><th>Stage</th><th>Step / Per pass</th><th>View</th><th>Command</th><th>Report</th></tr></thead>' \
  "${TMP_DIR}/debug-run-deep/index.html"
if grep -Fq '<th>状态</th>' "${TMP_DIR}/debug-run-deep/index.html"; then
  echo "Stage Timeline should not expose status column" >&2
  exit 1
fi
if grep -Fq '<th>Order</th>' "${TMP_DIR}/debug-run-deep/index.html"; then
  echo "Stage Timeline should not expose order column" >&2
  exit 1
fi
if grep -Fq '<summary>高级信息：执行命令</summary>' "${TMP_DIR}/debug-run-deep/index.html"; then
  echo "index should merge command details into Stage Timeline" >&2
  exit 1
fi
if grep -Fq '<h2>报告</h2>' "${TMP_DIR}/debug-run-deep/index.html"; then
  echo "index should not expose a separate report section" >&2
  exit 1
fi
grep -Fq '<a href="views/reports/030-schedule.report.txt.html">reports/030-schedule.report.txt</a>' "${TMP_DIR}/debug-run-deep/index.html"
grep -Fq '<a href="views/reports/040-realize.report.txt.html">reports/040-realize.report.txt</a>' "${TMP_DIR}/debug-run-deep/index.html"
test -f "${TMP_DIR}/debug-run-deep/views/reports/030-schedule.report.txt.html"
grep -Fq ':root { color-scheme: light; }' "${TMP_DIR}/debug-run-deep/views/reports/030-schedule.report.txt.html"
echo "ascend_debug.open_deep=ok"

mkdir -p "${TMP_DIR}/debug-run-stage-graph/stages"
cat >"${TMP_DIR}/debug-run-stage-graph/manifest.json" <<'JSON'
{
  "schema_version": 1,
  "tool": "ascend-debug",
  "preset": "quick",
  "pipeline": "normalize-kernelize",
  "stages": [
    {"order": 0, "name": "source", "path": "stages/000-source.mlir"},
    {"order": 29, "name": "kernelize-out", "path": "stages/029-kernelize-out.mlir"}
  ]
}
JSON
cat >"${TMP_DIR}/debug-run-stage-graph/stages/000-source.mlir" <<'MLIR'
func.func @elementwise(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %sum = arith.addf %x, %y : f16
    linalg.yield %sum : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}
MLIR
cat >"${TMP_DIR}/debug-run-stage-graph/stages/029-kernelize-out.mlir" <<'MLIR'
func.func @elementwise(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>) -> tensor<4x8xf16> {
  %empty = tensor.empty() : tensor<4x8xf16>
  %out = linalg.generic {ascend.kernel = "kernel_0", ascend.op_role = "vector", ascend.schedule.decision_id = "kernel_0.decision.0", iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<4x8xf16>, tensor<4x8xf16>)
    outs(%empty : tensor<4x8xf16>) {
  ^bb0(%x: f16, %y: f16, %o: f16):
    %sum = arith.addf %x, %y : f16
    linalg.yield %sum : f16
  } -> tensor<4x8xf16>
  return %out : tensor<4x8xf16>
}
MLIR
ascend-debug open "${TMP_DIR}/debug-run-stage-graph" --no-browser >"${TMP_DIR}/ascend-debug-open-stage-graph.txt"
grep -Fq '<h2>Stage Timeline</h2>' "${TMP_DIR}/debug-run-stage-graph/index.html"
grep -Fq '<a href="views/debug_graph.html?stage=0">Graph</a>' "${TMP_DIR}/debug-run-stage-graph/index.html"
grep -Fq '<a href="views/debug_graph.html?stage=29">Graph</a>' "${TMP_DIR}/debug-run-stage-graph/index.html"
test -f "${TMP_DIR}/debug-run-stage-graph/graphs/stages/000-source.graph.json"
test -f "${TMP_DIR}/debug-run-stage-graph/graphs/stages/029-kernelize-out.graph.json"
test ! -e "${TMP_DIR}/debug-run-stage-graph/views/graphs/stages/000-source.graph.html"
test ! -e "${TMP_DIR}/debug-run-stage-graph/views/graphs/stages/029-kernelize-out.graph.html"
python3 - "${TMP_DIR}/debug-run-stage-graph/graphs/stages/029-kernelize-out.graph.json" <<'PY'
import json
import pathlib
import sys

graph = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert graph["schema_version"] == 1
assert graph["stage"]["name"] == "kernelize-out"
assert graph["node_count"] >= 3
assert graph["edge_count"] >= 2
assert graph["kernel_count"] == 1
assert graph["layout"]["visual_kind"] == "svg-dag"
assert graph["layout"]["direction"] == "top-to-bottom"
assert len(graph["layout"]["nodes"]) == graph["node_count"]
assert len(graph["layout"]["edges"]) == graph["edge_count"]
for edge in graph["edges"]:
    source = graph["layout"]["nodes"][edge["from"]]
    target = graph["layout"]["nodes"][edge["to"]]
    assert source["y"] < target["y"]
nodes = graph["nodes"]
linalg_nodes = [node for node in nodes if node["op_name"] == "linalg.generic"]
assert linalg_nodes
assert linalg_nodes[0]["kernel_id"] == "kernel_0"
assert linalg_nodes[0]["body_summary"] == "arith.addf"
assert linalg_nodes[0]["body_ops"] == ["arith.addf", "linalg.yield"]
assert "arith.addf" in linalg_nodes[0]["region_body"]
assert "linalg.yield" in linalg_nodes[0]["region_body"]
assert "%arg0" in linalg_nodes[0]["input_values"]
assert "%out" in linalg_nodes[0]["result_values"]
PY
echo "ascend_debug.stage_graph=ok"

RESOLVED_RUN_DIR="$(python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "${TMP_DIR}/debug-run")"
ascend-debug open "${TMP_DIR}/debug-run" --no-browser >"${TMP_DIR}/ascend-debug-open.txt"
grep -Fq "ascend-debug.open.index=${RESOLVED_RUN_DIR}/index.html" "${TMP_DIR}/ascend-debug-open.txt"
test -f "${TMP_DIR}/debug-run/index.html"
grep -Fq '<h1>Ascend Debug</h1>' "${TMP_DIR}/debug-run/index.html"
grep -Fq '<h2>运行概览</h2>' "${TMP_DIR}/debug-run/index.html"
grep -Fq '<span>preset</span><strong>quick</strong>' "${TMP_DIR}/debug-run/index.html"
grep -Fq '<span>tool</span><strong>ascend-debug</strong>' "${TMP_DIR}/debug-run/index.html"
grep -Fq '<h2>Stage Timeline</h2>' "${TMP_DIR}/debug-run/index.html"
grep -Fq '<a href="views/stages/000-source.mlir.html">Text</a>' "${TMP_DIR}/debug-run/index.html"
grep -Fq '<a href="views/debug_graph.html?stage=0">Graph</a>' "${TMP_DIR}/debug-run/index.html"
grep -Fq '<a href="views/debug_graph.html?stage=29">Graph</a>' "${TMP_DIR}/debug-run/index.html"
if grep -Fq '查看 MLIR' "${TMP_DIR}/debug-run/index.html"; then
  echo "index should use Text instead of 查看 MLIR" >&2
  exit 1
fi
if grep -Fq '在工作台查看' "${TMP_DIR}/debug-run/index.html"; then
  echo "index should use Graph instead of 在工作台查看" >&2
  exit 1
fi
test -f "${TMP_DIR}/debug-run/views/stages/000-source.mlir.html"
test -f "${TMP_DIR}/debug-run/graphs/stages/000-source.graph.json"
test ! -e "${TMP_DIR}/debug-run/views/graphs/stages/000-source.graph.html"
python3 - "${TMP_DIR}/debug-run/index.html" <<'PY'
import pathlib
import sys

html = pathlib.Path(sys.argv[1]).read_text()
expected_rows = [
    ("source", '<td class="view-cell"><span class="view-links"><a href="views/stages/000-source.mlir.html">Text</a><a href="views/debug_graph.html?stage=0">Graph</a></span></td>'),
    ("normalize-in", '<td class="view-cell"><span class="view-links"><a href="views/stages/010-normalize-in.mlir.html">Text</a><a href="views/debug_graph.html?stage=10">Graph</a></span></td>'),
    ("normalize-out", '<td class="view-cell"><span class="view-links"><a href="views/stages/019-normalize-out.mlir.html">Text</a><a href="views/debug_graph.html?stage=19">Graph</a></span></td>'),
    ("kernelize-in", '<td class="view-cell"><span class="view-links"><a href="views/stages/020-kernelize-in.mlir.html">Text</a><a href="views/debug_graph.html?stage=20">Graph</a></span></td>'),
    ("kernelize-out", '<td class="view-cell"><span class="view-links"><a href="views/stages/029-kernelize-out.mlir.html">Text</a><a href="views/debug_graph.html?stage=29">Graph</a></span></td>'),
]
cursor = 0
for step, view_cell in expected_rows:
    view_position = html.find(view_cell, cursor)
    if view_position < 0:
        raise SystemExit(f"missing ordered view cell: {step!r}")
    cursor = view_position + len(view_cell)
PY
echo "ascend_debug.open=ok"

ascend-debug serve "${TMP_DIR}/debug-run-stage-graph" \
  --host 127.0.0.1 \
  --port 0 \
  --no-browser \
  >"${TMP_DIR}/ascend-debug-serve.txt" \
  2>"${TMP_DIR}/ascend-debug-serve.err" &
SERVE_PID="$!"
SERVE_URL=""
for _ in $(seq 1 100); do
  if ! kill -0 "${SERVE_PID}" >/dev/null 2>&1; then
    echo "ascend-debug serve exited early" >&2
    cat "${TMP_DIR}/ascend-debug-serve.err" >&2 || true
    exit 1
  fi
  SERVE_URL="$(python3 - "${TMP_DIR}/ascend-debug-serve.txt" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text() if pathlib.Path(sys.argv[1]).exists() else ""
match = re.search(r"ascend-debug\.serve\.url=(http://[^\s]+)", text)
print(match.group(1) if match else "")
PY
)"
  if [[ -n "${SERVE_URL}" ]]; then
    break
  fi
  sleep 0.05
done
if [[ -z "${SERVE_URL}" ]]; then
  echo "ascend-debug serve did not print a URL" >&2
  cat "${TMP_DIR}/ascend-debug-serve.err" >&2 || true
  exit 1
fi
python3 - "${SERVE_URL}/views/debug_graph.html" <<'PY'
import html
import json
import re
import sys
import urllib.request

page = urllib.request.urlopen(sys.argv[1], timeout=5).read().decode("utf-8")
payload = re.search(r'<script type="application/json" id="graph-workspace-data">(.*?)</script>', page, re.S)
if not payload:
    raise SystemExit("debug graph workspace JSON missing")
data = json.loads(html.unescape(payload.group(1)))
stage = next(item for item in data["stages"] if item["name"] == "kernelize-out")
node = next(item for item in stage["graph"]["nodes"] if item["op_name"] == "linalg.generic")
if node.get("body_summary") != "arith.addf":
    raise SystemExit(f"unexpected initial body summary: {node.get('body_summary')}")
PY
python3 - "${TMP_DIR}/debug-run-stage-graph/stages/029-kernelize-out.mlir" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
if "arith.addf" not in text:
    raise SystemExit("test fixture no longer contains arith.addf")
path.write_text(text.replace("arith.addf", "arith.subf"))
PY
python3 - "${SERVE_URL}/views/debug_graph.html" "${SERVE_URL}/views/stages/029-kernelize-out.mlir.html" <<'PY'
import html
import json
import re
import sys
import urllib.request

debug_page = urllib.request.urlopen(sys.argv[1], timeout=5).read().decode("utf-8")
payload = re.search(r'<script type="application/json" id="graph-workspace-data">(.*?)</script>', debug_page, re.S)
if not payload:
    raise SystemExit("debug graph workspace JSON missing after edit")
data = json.loads(html.unescape(payload.group(1)))
stage = next(item for item in data["stages"] if item["name"] == "kernelize-out")
node = next(item for item in stage["graph"]["nodes"] if item["op_name"] == "linalg.generic")
if node.get("body_summary") != "arith.subf":
    raise SystemExit(f"serve did not refresh graph from edited MLIR: {node.get('body_summary')}")
mlir_page = urllib.request.urlopen(sys.argv[2], timeout=5).read().decode("utf-8")
if "arith.subf" not in mlir_page:
    raise SystemExit("serve did not refresh MLIR HTML view from edited MLIR")
PY
kill "${SERVE_PID}" >/dev/null 2>&1 || true
wait "${SERVE_PID}" >/dev/null 2>&1 || true
SERVE_PID=""
echo "ascend_debug.serve=ok"

check_open_manifest_error() {
  local manifest="$1"
  local expected="$2"
  local case_dir="${TMP_DIR}/bad-${expected}"
  mkdir -p "${case_dir}"
  printf '%s\n' "${manifest}" >"${case_dir}/manifest.json"
  if ascend-debug open "${case_dir}" --no-browser >"${case_dir}/stdout.txt" 2>"${case_dir}/stderr.txt"; then
    echo "expected ascend-debug open to fail for ${expected}" >&2
    return 1
  fi
  grep -Fq 'ascend-debug: error:' "${case_dir}/stderr.txt"
  if grep -Fq 'Traceback' "${case_dir}/stderr.txt"; then
    echo "unexpected traceback for ${expected}" >&2
    return 1
  fi
}

check_open_manifest_error '{"schema_version": 1, "stages": [{"order": "0", "name": "source", "path": "stages/000-source.mlir"}]}' "bad-order"
check_open_manifest_error '{"schema_version": 1, "stages": [{"order": 0, "name": "source", "path": "../outside.mlir"}]}' "bad-path"
check_open_manifest_error '{"stages": []}' "missing-schema"
printf '{"schema_version": 1, "stages": [' >"${TMP_DIR}/corrupt-manifest.json"
check_open_manifest_error "$(cat "${TMP_DIR}/corrupt-manifest.json")" "corrupt-json"
DECODE_DIR="${TMP_DIR}/bad-decode"
mkdir -p "${DECODE_DIR}"
printf '\377' >"${DECODE_DIR}/manifest.json"
if ascend-debug open "${DECODE_DIR}" --no-browser >"${DECODE_DIR}/stdout.txt" 2>"${DECODE_DIR}/stderr.txt"; then
  echo "expected ascend-debug open to fail for decode" >&2
  exit 1
fi
grep -Fq 'ascend-debug: error:' "${DECODE_DIR}/stderr.txt"
if grep -Fq 'Traceback' "${DECODE_DIR}/stderr.txt"; then
  echo "unexpected traceback for decode" >&2
  exit 1
fi
echo "ascend_debug.open_negative=ok"

python3 - "${TMP_DIR}/debug-run/manifest.json" "${TMP_DIR}/debug-run/provenance.json" "${INPUT_MLIR}" <<'PY'
import json
import pathlib
import sys

def check(condition, message):
    if not condition:
        raise SystemExit(message)

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
provenance = json.loads(pathlib.Path(sys.argv[2]).read_text())
input_mlir = sys.argv[3]
stages = manifest["stages"]
print(f"ascend_debug.manifest.stage_count={len(stages)}")
for i, stage in enumerate(stages):
    print(f"ascend_debug.stage.{i}={pathlib.Path(stage['path']).name}")
expected_paths = [
    "stages/000-source.mlir",
    "stages/010-normalize-in.mlir",
    "stages/019-normalize-out.mlir",
    "stages/020-kernelize-in.mlir",
    "stages/029-kernelize-out.mlir",
]
check(manifest["schema_version"] == 1, "manifest schema_version must be 1")
check(manifest["tool"] == "ascend-debug", "manifest tool must be ascend-debug")
check(manifest["input"] == "stages/000-source.mlir", "manifest input must point to staged source")
check(manifest["mode"] == "quick", "manifest mode must be quick")
check(manifest["preset"] == "quick", "manifest preset must be quick")
check(manifest["backend"] == "compile", "manifest backend must be compile")
check(manifest["device_id"] is None, "manifest device_id must be null")
check(manifest["device_scope"] == "single_run_single_device", "manifest device_scope mismatch")
check([stage["order"] for stage in stages] == [0, 10, 19, 20, 29], "stage orders mismatch")
check([stage["name"] for stage in stages] == [
    "source",
    "normalize-in",
    "normalize-out",
    "kernelize-in",
    "kernelize-out",
], "stage names mismatch")
actual_paths = [stage["path"] for stage in stages]
check(actual_paths == expected_paths, f"stage paths mismatch: {actual_paths!r}")
absolute_paths = [path for path in actual_paths if pathlib.PurePosixPath(path).is_absolute()]
check(not absolute_paths, f"stage paths must be relative: {absolute_paths!r}")
check(provenance["schema_version"] == 1, "provenance schema_version must be 1")
check(provenance["tool"] == "ascend-debug", "provenance tool must be ascend-debug")
check(provenance["version"], "provenance version must be present")
check(provenance["original_input"] == input_mlir, "provenance original_input must preserve CLI input")
check(provenance["boundaries"] == [], "provenance boundaries must start empty")
check(provenance["kernels"] == [], "provenance kernels must start empty")
check(provenance["runtime_tasks"] == [], "provenance runtime_tasks must start empty")
PY

python3 - "${TMP_DIR}/debug-run-deep/manifest.json" <<'PY'
import json
import pathlib
import sys

def check(condition, message):
    if not condition:
        raise SystemExit(message)

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
stages = manifest["stages"]
commands = manifest.get("commands", [])
reports = manifest.get("reports", [])
print(f"ascend_debug.deep.stage_count={len(stages)}")
print(f"ascend_debug.deep.command_count={len(commands)}")
check(manifest["mode"] == "quick", "deep manifest mode must be quick")
check(manifest["preset"] == "deep", "deep manifest preset must be deep")
check([stage["order"] for stage in stages] == [0, 10, 19, 20, 29, 30, 39, 40, 49], "deep stage orders mismatch")
check([stage["name"] for stage in stages] == [
    "source",
    "normalize-in",
    "normalize-out",
    "kernelize-in",
    "kernelize-out",
    "schedule-in",
    "schedule-out",
    "realize-in",
    "realize-out",
], "deep stage names mismatch")
check([stage["path"] for stage in stages] == [
    "stages/000-source.mlir",
    "stages/010-normalize-in.mlir",
    "stages/019-normalize-out.mlir",
    "stages/020-kernelize-in.mlir",
    "stages/029-kernelize-out.mlir",
    "stages/030-schedule-in.mlir",
    "stages/039-schedule-out.mlir",
    "stages/040-realize-in.mlir",
    "stages/049-realize-out.mlir",
], "deep stage paths mismatch")
check(len(commands) == 4, "deep manifest must record four commands")
check([command["stage"] for command in commands] == ["normalize", "kernelize", "schedule", "realize"], "deep command stages mismatch")
check(all(command["status"] == "success" for command in commands), "deep commands must succeed")
check(all(command["tool"] == "ascend-mlir-opt" for command in commands), "deep commands must use ascend-mlir-opt")
check(len(reports) == 4, "deep manifest must record four reports")
check([report["path"] for report in reports] == [
    "reports/010-normalize.report.txt",
    "reports/020-kernelize.report.txt",
    "reports/030-schedule.report.txt",
    "reports/040-realize.report.txt",
], "deep report paths mismatch")
PY

python3 - "${TMP_DIR}/debug-run-graph/manifest.json" <<'PY'
import json
import pathlib
import sys

def check(condition, message):
    if not condition:
        raise SystemExit(message)

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
commands = manifest.get("commands", [])
reports = manifest.get("reports", [])
graphs = manifest.get("graphs", [])
print(f"ascend_debug.graph.command_count={len(commands)}")
print(f"ascend_debug.graph.artifact_count={len(graphs)}")
check(len(commands) == 5, "graph manifest must record five commands")
check(commands[-1]["stage"] == "kernel-dag", "graph command stage mismatch")
check(commands[-1]["tool"] == "ascend-debug", "graph command tool mismatch")
check(len(reports) == 5, "graph manifest must record five reports")
check(reports[-1]["path"] == "reports/050-kernel-dag.report.txt", "graph report path mismatch")
check([graph["path"] for graph in graphs] == [
    "graphs/artifact_manifest.json",
    "graphs/run_manifest.json",
    "graphs/kernelized.mlir",
    "graphs/kernel_dag.svg",
    "graphs/kernel_dag.summary.json",
], "graph artifact paths mismatch")
check([graph["kind"] for graph in graphs] == [
    "artifact-manifest",
    "run-manifest",
    "kernelized-ir",
    "kernel-dag-svg",
    "kernel-dag-summary",
], "graph artifact kinds mismatch")
PY

echo "ALL ASCEND DEBUG CLI TESTS PASSED"
