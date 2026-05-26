#!/usr/bin/env bash
set -euo pipefail

INPUT_MLIR="$1"
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

echo "ALL ASCEND DEBUG CLI TESTS PASSED"
