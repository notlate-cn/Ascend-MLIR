#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
TOOL="${REPO_ROOT}/test/tools/diagnostics/ascend_kernel_dag_viz.py"

echo "INFO: ascend kernel dag viz test entry"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ascend-kernel-dag-viz.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

cat >"${TMP_DIR}/artifact_manifest.json" <<'JSON'
{
  "kernel_entries": [
    {
      "kernel_id": "kernel_0",
      "kernelKind": "vec",
      "scheduleEntries": [
        {"tilingParams": {"selected_tile_shape": [1, 4, 128]}}
      ],
      "workspaceSizeBytes": 0
    },
    {
      "kernel_id": "kernel_1",
      "kernelKind": "vec",
      "scheduleEntries": [
        {"tilingParams": {"selected_tile_shape": [128, 384]}}
      ],
      "workspaceSizeBytes": 0
    },
    {
      "kernel_id": "kernel_2",
      "kernelKind": "mix",
      "scheduleEntries": [
        {"tilingParams": {"selected_tile_shape": [1, 4, 384, 128]}}
      ],
      "workspaceSizeBytes": 4096
    },
    {
      "kernel_id": "kernel_3",
      "kernelKind": "vec",
      "scheduleEntries": [
        {"tilingParams": {"selected_tile_shape": [1, 4, 128]}}
      ],
      "workspaceSizeBytes": 0
    },
    {
      "kernel_id": "kernel_4",
      "kernelKind": "vec",
      "scheduleEntries": [
        {"tilingParams": {"selected_tile_shape": [1, 4, 128]}}
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

cat >"${TMP_DIR}/run_manifest.json" <<'JSON'
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

cat >"${TMP_DIR}/step2_kernelized.mlir" <<'MLIR'
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

"${PYTHON}" "${TOOL}" \
  --artifact-manifest "${TMP_DIR}/artifact_manifest.json" \
  --run-manifest "${TMP_DIR}/run_manifest.json" \
  --kernelized-ir "${TMP_DIR}/step2_kernelized.mlir" \
  --svg-out "${TMP_DIR}/kernel_dag.svg" \
  --summary-out "${TMP_DIR}/kernel_dag_summary.json" \
  >"${TMP_DIR}/tool.stdout"

"${PYTHON}" "${TOOL}" \
  --runtime-manifest "${TMP_DIR}/artifact_manifest.json" \
  --summary-out "${TMP_DIR}/legacy_kernel_dag_summary.json" \
  >"${TMP_DIR}/legacy_tool.stdout"
grep -Fq 'ascend_kernel_dag_viz.kernel_count=5' "${TMP_DIR}/legacy_tool.stdout"

"${PYTHON}" - "${TMP_DIR}/kernel_dag.svg" "${TMP_DIR}/kernel_dag_summary.json" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

svg_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
ET.parse(svg_path)
svg_text = svg_path.read_text(encoding="utf-8")
summary = json.loads(summary_path.read_text(encoding="utf-8"))

print(f"ascend_kernel_dag_viz.kernel_count={summary['kernel_count']}")
print(f"ascend_kernel_dag_viz.graph_edges={summary['graph_edges']}")
print(f"ascend_kernel_dag_viz.kind.vec={summary['kind_counts']['vec']}")
print(f"ascend_kernel_dag_viz.kind.mix={summary['kind_counts']['mix']}")
print(f"ascend_kernel_dag_viz.root_tasks={summary['root_tasks']}")
print(
    "ascend_kernel_dag_viz.prepack_candidate_roots="
    f"{summary['prepack_candidate_roots']}"
)
print(
    "ascend_kernel_dag_viz.critical_path_depth="
    f"{summary['critical_path_depth']}"
)
print(
    "ascend_kernel_dag_viz.simple_fusion_edges="
    f"{len(summary['simple_fusion_edges'])}"
)
print(
    "ascend_kernel_dag_viz.svg.contains.kernel_2="
    f"{str('kernel_2' in svg_text).lower()}"
)
print(
    "ascend_kernel_dag_viz.svg.contains.batch_matmul="
    f"{str('batch_matmul' in svg_text).lower()}"
)
print(
    "ascend_kernel_dag_viz.svg.contains.1x4x128="
    f"{str('1x4x128' in svg_text).lower()}"
)
PY

echo "ALL ASCEND KERNEL DAG VIZ TESTS PASSED"
