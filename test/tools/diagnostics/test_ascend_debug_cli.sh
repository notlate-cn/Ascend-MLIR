#!/usr/bin/env bash
set -euo pipefail

INPUT_MLIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ascend-debug-cli.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

ascend-debug --help >"${TMP_DIR}/ascend-debug-help.txt" 2>&1
grep -Fq 'collect' "${TMP_DIR}/ascend-debug-help.txt"
grep -Fq 'open' "${TMP_DIR}/ascend-debug-help.txt"
grep -Fq 'diff' "${TMP_DIR}/ascend-debug-help.txt"
grep -Fq 'locate' "${TMP_DIR}/ascend-debug-help.txt"
echo "ascend_debug.help=ok"

ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run" \
  --pipeline normalize-kernelize
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

ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run-deep" \
  --preset deep \
  --pipeline normalize-kernelize
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

cat >"${TMP_DIR}/runtime_manifest.json" <<'JSON'
{
  "kernel_entries": [
    {
      "kernel_id": "kernel_0",
      "kernelKind": "vec",
      "scheduleEntries": [
        {"tilingParams": {"selected_tile_shape": [4, 8]}}
      ],
      "workspaceSizeBytes": 0
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
      "outputs": [{"name": "out0", "shape": [4, 8], "dtype": "f16"}],
      "workspace_size": 0
    }
  ]
}
JSON

ascend-debug collect "${INPUT_MLIR}" \
  --out "${TMP_DIR}/debug-run-graph" \
  --preset deep \
  --pipeline normalize-kernelize \
  --runtime-manifest "${TMP_DIR}/runtime_manifest.json" \
  --run-manifest "${TMP_DIR}/run_manifest.json" \
  --dag-viz "${SCRIPT_DIR}/ascend_kernel_dag_viz.py"
test -f "${TMP_DIR}/debug-run-graph/graphs/runtime_manifest.json"
test -f "${TMP_DIR}/debug-run-graph/graphs/run_manifest.json"
test -f "${TMP_DIR}/debug-run-graph/graphs/kernelized.mlir"
test -f "${TMP_DIR}/debug-run-graph/graphs/kernel_dag.svg"
test -f "${TMP_DIR}/debug-run-graph/graphs/kernel_dag.summary.json"
test -f "${TMP_DIR}/debug-run-graph/reports/050-kernel-dag-viz.report.txt"
grep -Fq 'ascend_kernel_dag_viz.kernel_count=1' "${TMP_DIR}/debug-run-graph/reports/050-kernel-dag-viz.report.txt"
echo "ascend_debug.collect_graph=ok"

ascend-debug open "${TMP_DIR}/debug-run-graph" --no-browser >"${TMP_DIR}/ascend-debug-open-graph.txt"
grep -Fq '<h2>Graphs</h2>' "${TMP_DIR}/debug-run-graph/index.html"
grep -Fq 'graphs/kernel_dag.svg' "${TMP_DIR}/debug-run-graph/index.html"
grep -Fq 'graphs/kernel_dag.summary.json' "${TMP_DIR}/debug-run-graph/index.html"
echo "ascend_debug.open_graph=ok"

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
grep -Fq '<dt>preset</dt><dd>deep</dd>' "${TMP_DIR}/debug-run-deep/index.html"
grep -Fq '<h2>Commands</h2>' "${TMP_DIR}/debug-run-deep/index.html"
grep -Fq '<h2>Reports</h2>' "${TMP_DIR}/debug-run-deep/index.html"
grep -Fq 'reports/030-schedule.report.txt' "${TMP_DIR}/debug-run-deep/index.html"
grep -Fq 'reports/040-realize.report.txt' "${TMP_DIR}/debug-run-deep/index.html"
echo "ascend_debug.open_deep=ok"

RESOLVED_RUN_DIR="$(python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "${TMP_DIR}/debug-run")"
ascend-debug open "${TMP_DIR}/debug-run" --no-browser >"${TMP_DIR}/ascend-debug-open.txt"
grep -Fq "ascend-debug.open.index=${RESOLVED_RUN_DIR}/index.html" "${TMP_DIR}/ascend-debug-open.txt"
test -f "${TMP_DIR}/debug-run/index.html"
grep -Fq '<h1>ascend-debug</h1>' "${TMP_DIR}/debug-run/index.html"
grep -Fq '<dt>schema_version</dt><dd>1</dd>' "${TMP_DIR}/debug-run/index.html"
grep -Fq '<dt>preset</dt><dd>quick</dd>' "${TMP_DIR}/debug-run/index.html"
grep -Fq '<dt>tool</dt><dd>ascend-debug</dd>' "${TMP_DIR}/debug-run/index.html"
python3 - "${TMP_DIR}/debug-run/index.html" <<'PY'
import pathlib
import sys

html = pathlib.Path(sys.argv[1]).read_text()
expected_rows = [
    ("0", "source", "stages/000-source.mlir", "present"),
    ("10", "normalize-in", "stages/010-normalize-in.mlir", "present"),
    ("19", "normalize-out", "stages/019-normalize-out.mlir", "present"),
    ("20", "kernelize-in", "stages/020-kernelize-in.mlir", "present"),
    ("29", "kernelize-out", "stages/029-kernelize-out.mlir", "present"),
]
cursor = 0
for row in expected_rows:
    needle = "".join(f"<td>{cell}</td>" for cell in row)
    position = html.find(needle, cursor)
    if position < 0:
        raise SystemExit(f"missing ordered stage row: {row!r}")
    cursor = position + len(needle)
PY
echo "ascend_debug.open=ok"

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
check(commands[-1]["stage"] == "kernel-dag-viz", "graph command stage mismatch")
check(commands[-1]["tool"] == "ascend_kernel_dag_viz.py", "graph command tool mismatch")
check(len(reports) == 5, "graph manifest must record five reports")
check(reports[-1]["path"] == "reports/050-kernel-dag-viz.report.txt", "graph report path mismatch")
check([graph["path"] for graph in graphs] == [
    "graphs/runtime_manifest.json",
    "graphs/run_manifest.json",
    "graphs/kernelized.mlir",
    "graphs/kernel_dag.svg",
    "graphs/kernel_dag.summary.json",
], "graph artifact paths mismatch")
check([graph["kind"] for graph in graphs] == [
    "runtime-manifest",
    "run-manifest",
    "kernelized-ir",
    "kernel-dag-svg",
    "kernel-dag-summary",
], "graph artifact kinds mismatch")
PY

echo "ALL ASCEND DEBUG CLI TESTS PASSED"
