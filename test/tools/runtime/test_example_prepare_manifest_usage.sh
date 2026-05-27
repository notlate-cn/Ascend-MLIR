#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_ROOT}"

grep -q -- '--artifact-manifest "$PHASE5_ARTIFACT_MANIFEST"' \
  examples/broadcast-add-reduce/run-mainline.sh
grep -q -- '--emit-run-manifest "$PREPARED_RUN_MANIFEST"' \
  examples/broadcast-add-reduce/run-mainline.sh
if grep -q 'cat > "$RUN_MANIFEST"' \
    examples/broadcast-add-reduce/run-mainline.sh; then
  echo "broadcast-add-reduce still hand-writes run_manifest.json" >&2
  exit 1
fi

test -f examples/relu-broadcast-transpose/case.json
grep -q -- 'CASE_JSON="$DIR/case.json"' \
  examples/relu-broadcast-transpose/run-mainline.sh
grep -q -- '"$ASCEND_DEBUG" run "$CASE_JSON"' \
  examples/relu-broadcast-transpose/run-mainline.sh
python3 - examples/relu-broadcast-transpose/case.json <<'PY'
import json
import sys
from pathlib import Path

case = json.loads(Path(sys.argv[1]).read_text())
assert case["schema_version"] == 1
assert "artifact" not in case
assert case["source"]["mlir"] == "step0_input.mlir"
assert case["backend"]["kind"] == "sim"
assert "shape_args" not in case
assert case["inputs"] == [
    {"name": "arg0", "path": "build_mainline/input_data0.npy", "dtype": "f16"},
    {"name": "arg1", "path": "build_mainline/input_data1.npy", "dtype": "f16"},
]
assert case["outputs"] == [
    {"name": "out0", "path": "build_mainline/output.npy", "dtype": "f16"},
]
assert case["expected_outputs"] == [
    {"name": "out0", "path": "build_mainline/output_expected.npy", "dtype": "f16"},
]
for generated in ("tiling", "block_dim", "workspace_size"):
    assert generated not in case
PY
if grep -q -- 'cat > "$CASE_JSON"' \
    examples/relu-broadcast-transpose/run-mainline.sh; then
  echo "relu-broadcast-transpose still generates case.json inside run-mainline.sh" >&2
  exit 1
fi
if grep -q -- '--artifact-manifest "$PHASE5_ARTIFACT_MANIFEST"' \
    examples/relu-broadcast-transpose/run-mainline.sh; then
  echo "relu-broadcast-transpose still uses low-level artifact-manifest CLI bindings" >&2
  exit 1
fi
if grep -q -- '--case "$CASE_JSON"' \
    examples/relu-broadcast-transpose/run-mainline.sh; then
  echo "relu-broadcast-transpose still calls runtime-session --case directly" >&2
  exit 1
fi
if grep -q -- '--shape-arg "arg0_dim0=$M"' \
    examples/relu-broadcast-transpose/run-mainline.sh; then
  echo "relu-broadcast-transpose still passes shape args through runtime-session CLI" >&2
  exit 1
fi
if grep -Eq -- '--block-dim|BLOCK_DIM|block_dim=auto' \
    examples/relu-broadcast-transpose/run-mainline.sh; then
  echo "relu-broadcast-transpose still exposes block-dim plumbing" >&2
  exit 1
fi
if grep -Eq -- 'mainline pipeline complete|build_mainline/step1_fused.mlir|build_mainline/case.artifact.json' \
    examples/relu-broadcast-transpose/run-mainline.sh; then
  echo "relu-broadcast-transpose still prints generated compile artifact inventory" >&2
  exit 1
fi

test -f examples/add-broadcast-concat/case.json
grep -q -- 'CASE_JSON="$DIR/case.json"' \
  examples/add-broadcast-concat/run-mainline.sh
grep -q -- '"$ASCEND_DEBUG" run "$CASE_JSON"' \
  examples/add-broadcast-concat/run-mainline.sh
python3 - examples/add-broadcast-concat/case.json <<'PY'
import json
import sys
from pathlib import Path

case = json.loads(Path(sys.argv[1]).read_text())
assert case["schema_version"] == 1
assert "artifact" not in case
assert case["source"]["mlir"] == "step0_input.mlir"
assert case["backend"]["kind"] == "sim"
assert "shape_args" not in case
assert case["inputs"] == [
    {"name": "arg0", "path": "build_mainline/input_a.npy", "dtype": "f16"},
    {"name": "arg1", "path": "build_mainline/input_b.npy", "dtype": "f16"},
    {"name": "arg2", "path": "build_mainline/input_c.npy", "dtype": "f16"},
    {"name": "arg3", "path": "build_mainline/input_d.npy", "dtype": "f16"},
]
assert case["outputs"] == [
    {"name": "out0", "path": "build_mainline/output_actual.npy", "dtype": "f16"},
]
assert case["expected_outputs"] == [
    {"name": "out0", "path": "build_mainline/output.npy", "dtype": "f16"},
]
for generated in ("tiling", "block_dim", "workspace_size"):
    assert generated not in case
PY
if grep -q -- 'cat > "$CASE_JSON"' \
    examples/add-broadcast-concat/run-mainline.sh; then
  echo "add-broadcast-concat still generates case.json inside run-mainline.sh" >&2
  exit 1
fi
if grep -q -- 'cat > "$RUN_MANIFEST"' \
    examples/add-broadcast-concat/run-mainline.sh; then
  echo "add-broadcast-concat still hand-writes run_manifest.json" >&2
  exit 1
fi
if grep -q -- '--artifact-manifest "$PHASE5_ARTIFACT_MANIFEST"' \
    examples/add-broadcast-concat/run-mainline.sh; then
  echo "add-broadcast-concat still uses low-level artifact-manifest CLI bindings" >&2
  exit 1
fi
if grep -q -- '--case "$CASE_JSON"' \
    examples/add-broadcast-concat/run-mainline.sh; then
  echo "add-broadcast-concat still calls runtime-session --case directly" >&2
  exit 1
fi
if grep -Eq -- '--block-dim|BLOCK_DIM|block_dim=auto' \
    examples/add-broadcast-concat/run-mainline.sh; then
  echo "add-broadcast-concat still exposes block-dim plumbing" >&2
  exit 1
fi
if grep -Eq -- 'mainline pipeline complete|build_mainline/step1_fused.mlir|build_mainline/case.artifact.json' \
    examples/add-broadcast-concat/run-mainline.sh; then
  echo "add-broadcast-concat still prints generated compile artifact inventory" >&2
  exit 1
fi

bash -n examples/broadcast-add-reduce/run-mainline.sh
bash -n examples/add-broadcast-concat/run-mainline.sh
bash -n examples/relu-broadcast-transpose/run-mainline.sh
