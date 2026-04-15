#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_verify_env.sh"

RUNS="${RUNS:-3}"
if [ "${1:-}" = "--runs" ]; then
  RUNS="${2:?missing value for --runs}"
fi

runtime_verify_setup_env
runtime_verify_prepare_build_dir
runtime_verify_build_runtime_core
runtime_verify_build_example_toolchain
runtime_verify_build_mix_compiler

EXAMPLE="examples/matmul-add-leakyrelu/run.sh"
ARTIFACT_DIR="${PROJECT_ROOT}/build/runtime-mix-matmul-add-leakyrelu"
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
    python3 - <<'PY' >>"${RESULTS}"
import json
import os
from pathlib import Path

timing_path = Path(os.environ["TIMING_PATH"])
timing = json.loads(timing_path.read_text())
print(json.dumps({
    "run": int(os.environ["RUN_INDEX"]),
    "total_us": int(timing["total_elapsed_us"]),
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
for row in rows:
    totals.append(row["total_us"])
    for name, elapsed in row["stages"].items():
        stage_values[name].append(elapsed)

def mean_ms(values):
    return statistics.mean(values) / 1000.0

print("mix_compile_timing")
print(f"mode=direct-source runs={len(totals)} mean_ms={mean_ms(totals):.3f} min_ms={min(totals)/1000.0:.3f} max_ms={max(totals)/1000.0:.3f}")
for name in sorted(stage_values):
    print(f"stage name={name} mean_ms={mean_ms(stage_values[name]):.3f}")
PY
