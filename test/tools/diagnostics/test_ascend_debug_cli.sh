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
test -f "${TMP_DIR}/debug-run/stages/029-kernelize-out.mlir"
test -f "${TMP_DIR}/debug-run/manifest.json"
test -f "${TMP_DIR}/debug-run/provenance.json"
echo "ascend_debug.collect=ok"

ascend-debug open "${TMP_DIR}/debug-run" --no-browser >"${TMP_DIR}/ascend-debug-open.txt"
grep -q 'index.html' "${TMP_DIR}/ascend-debug-open.txt"
test -f "${TMP_DIR}/debug-run/index.html"
echo "ascend_debug.open=ok"

python3 - "${TMP_DIR}/debug-run/manifest.json" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
stages = manifest["stages"]
print(f"ascend_debug.manifest.stage_count={len(stages)}")
for i, stage in enumerate(stages):
    print(f"ascend_debug.stage.{i}={pathlib.Path(stage['path']).name}")
assert [stage["order"] for stage in stages] == [0, 10, 19, 20, 29]
assert manifest["tool"] == "ascend-debug"
assert manifest["preset"] == "quick"
assert manifest["device_scope"] == "single_run_single_device"
PY

echo "ALL ASCEND DEBUG CLI TESTS PASSED"
