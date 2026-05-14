#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"

RUNS="${RUNS:-3}"
if [ "${1:-}" = "--runs" ]; then
  RUNS="${2:?missing value for --runs}"
fi

runtime_verify_setup_env
export LD_LIBRARY_PATH="$(runtime_verify_runtime_ld_library_path)"
runtime_verify_prepare_build_dir
runtime_verify_build_runtime_core
runtime_verify_build_example_toolchain
runtime_verify_build_mix_compiler

EXAMPLE="examples/matmul-add-leakyrelu/run.sh"
ARTIFACT_DIR="${PROJECT_ROOT}/examples/matmul-add-leakyrelu/build_mainline/artifact"
RESULTS="$(mktemp /tmp/runtime-mix-compile-timing.XXXXXX.jsonl)"
cleanup() {
  rm -f "${RESULTS}"
}
trap cleanup EXIT

run_one() {
  local index="$1"
  local log="/tmp/runtime-mix-compile-direct-source-${index}.log"

  bash "${EXAMPLE}" >"${log}" 2>&1 || {
    cat "${log}" >&2 || true
    echo "FAIL: direct-source run ${index} failed" >&2
    return 1
  }

  RUN_INDEX="${index}" TIMING_PATH="${ARTIFACT_DIR}/out/compile_timing.json" \
    MANIFEST_PATH="${ARTIFACT_DIR}/out/manifest.txt" ARTIFACT_DIR="${ARTIFACT_DIR}" \
    python3 - <<'PY' >>"${RESULTS}"
import json
import os
from pathlib import Path

timing_path = Path(os.environ["TIMING_PATH"])
timing = json.loads(timing_path.read_text())
manifest = {}
for raw in Path(os.environ["MANIFEST_PATH"]).read_text().splitlines():
    line = raw.strip()
    if not line or "=" not in line:
        continue
    key, value = line.split("=", 1)
    manifest[key] = value
metadata_path = Path(manifest["metadata_path"])
if not metadata_path.is_absolute():
    metadata_path = (Path(os.environ["ARTIFACT_DIR"]) / metadata_path).resolve()
metadata = json.loads(metadata_path.read_text())
helper_inputs = metadata.get("host_launch", {}).get("helper_inputs", {})
print(json.dumps({
    "run": int(os.environ["RUN_INDEX"]),
    "total_us": int(timing["total_elapsed_us"]),
    "tiling_backend": helper_inputs.get("tiling_backend", ""),
    "tiling_strategy": helper_inputs.get("tiling_strategy", ""),
    "host_launch_mode": metadata.get("host_launch", {}).get("mode", ""),
    "host_launch_helper_kind": metadata.get("host_launch", {}).get("helper_kind", ""),
    "stages": {
        stage["name"]: int(stage["elapsed_us"])
        for stage in timing.get("stages", [])
    },
}, sort_keys=True))
PY
}

for index in $(seq 1 "${RUNS}"); do
  run_one "${index}"
done

python3 - "${RESULTS}" <<'PY'
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
totals = []
stage_values = defaultdict(list)
backend_values = []
strategy_values = []
mode_values = []
helper_kind_values = []
for row in rows:
    totals.append(row["total_us"])
    backend_values.append(row.get("tiling_backend", ""))
    strategy_values.append(row.get("tiling_strategy", ""))
    mode_values.append(row.get("host_launch_mode", ""))
    helper_kind_values.append(row.get("host_launch_helper_kind", ""))
    for name, elapsed in row["stages"].items():
        stage_values[name].append(elapsed)

def mean_ms(values):
    return statistics.mean(values) / 1000.0

def unique(values):
    return sorted(set(values))

print("mix_compile_timing")
print(f"mode=direct-source runs={len(totals)} mean_ms={mean_ms(totals):.3f} min_ms={min(totals)/1000.0:.3f} max_ms={max(totals)/1000.0:.3f}")
print(
    "metadata "
    f"tiling_backend={unique(backend_values)} "
    f"tiling_strategy={unique(strategy_values)} "
    f"host_launch_mode={unique(mode_values)} "
    f"helper_kind={unique(helper_kind_values)}"
)
for name in sorted(stage_values):
    print(f"stage name={name} mean_ms={mean_ms(stage_values[name]):.3f}")
PY
