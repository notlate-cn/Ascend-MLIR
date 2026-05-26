#!/usr/bin/env bash
set -euo pipefail

INPUT_MLIR="$1"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ascend-debug-cli.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

ascend-debug --help >"${TMP_DIR}/ascend-debug-help.txt" 2>&1
grep -q 'collect' "${TMP_DIR}/ascend-debug-help.txt"
grep -q 'open' "${TMP_DIR}/ascend-debug-help.txt"
grep -q 'diff' "${TMP_DIR}/ascend-debug-help.txt"
grep -q 'locate' "${TMP_DIR}/ascend-debug-help.txt"
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
grep -q 'ascend.kernel' "${TMP_DIR}/debug-run/stages/029-kernelize-out.mlir"
test -f "${TMP_DIR}/debug-run/manifest.json"
test -f "${TMP_DIR}/debug-run/provenance.json"
echo "ascend_debug.collect=ok"

RESOLVED_RUN_DIR="$(python3 -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' "${TMP_DIR}/debug-run")"
ascend-debug open "${TMP_DIR}/debug-run" --no-browser >"${TMP_DIR}/ascend-debug-open.txt"
grep -q "ascend-debug.open.index=${RESOLVED_RUN_DIR}/index.html" "${TMP_DIR}/ascend-debug-open.txt"
test -f "${TMP_DIR}/debug-run/index.html"
grep -q '<h1>ascend-debug</h1>' "${TMP_DIR}/debug-run/index.html"
grep -q '<dt>schema_version</dt><dd>1</dd>' "${TMP_DIR}/debug-run/index.html"
grep -q '<dt>preset</dt><dd>quick</dd>' "${TMP_DIR}/debug-run/index.html"
grep -q '<dt>tool</dt><dd>ascend-debug</dd>' "${TMP_DIR}/debug-run/index.html"
grep -q '<tr><td>0</td><td>source</td><td>stages/000-source.mlir</td><td>present</td></tr>' "${TMP_DIR}/debug-run/index.html"
grep -q '<tr><td>10</td><td>normalize-in</td><td>stages/010-normalize-in.mlir</td><td>present</td></tr>' "${TMP_DIR}/debug-run/index.html"
grep -q '<tr><td>19</td><td>normalize-out</td><td>stages/019-normalize-out.mlir</td><td>present</td></tr>' "${TMP_DIR}/debug-run/index.html"
grep -q '<tr><td>20</td><td>kernelize-in</td><td>stages/020-kernelize-in.mlir</td><td>present</td></tr>' "${TMP_DIR}/debug-run/index.html"
grep -q '<tr><td>29</td><td>kernelize-out</td><td>stages/029-kernelize-out.mlir</td><td>present</td></tr>' "${TMP_DIR}/debug-run/index.html"
echo "ascend_debug.open=ok"

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

echo "ALL ASCEND DEBUG CLI TESTS PASSED"
