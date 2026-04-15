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
  local mode="$1"
  local index="$2"
  local log="/tmp/runtime-mix-compile-${mode}-${index}.log"

  ASCEND_MIX_CONTRACT_MODE="${mode}" bash "${EXAMPLE}" >"${log}" 2>&1 || {
    cat "${log}" >&2 || true
    echo "FAIL: ${mode} run ${index} failed" >&2
    return 1
  }

  MODE="${mode}" RUN_INDEX="${index}" TIMING_PATH="${ARTIFACT_DIR}/out/compile_timing.json" \
    python3 - <<'PY' >>"${RESULTS}"
import json
import os
from pathlib import Path

timing_path = Path(os.environ["TIMING_PATH"])
timing = json.loads(timing_path.read_text())
print(json.dumps({
    "mode": os.environ["MODE"],
    "run": int(os.environ["RUN_INDEX"]),
    "total_us": int(timing["total_elapsed_us"]),
    "stages": {
        stage["name"]: int(stage["elapsed_us"])
        for stage in timing.get("stages", [])
    },
}, sort_keys=True))
PY
}

for mode in direct-source legacy-preprocess; do
  for index in $(seq 1 "${RUNS}"); do
    run_one "${mode}" "${index}"
  done
done

python3 - "${RESULTS}" <<'PY'
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
by_mode = defaultdict(list)
stage_values = defaultdict(lambda: defaultdict(list))
for row in rows:
    by_mode[row["mode"]].append(row["total_us"])
    for name, elapsed in row["stages"].items():
        stage_values[row["mode"]][name].append(elapsed)

def mean_ms(values):
    return statistics.mean(values) / 1000.0

print("mix_compile_timing_compare")
for mode in ("direct-source", "legacy-preprocess"):
    values = by_mode[mode]
    print(f"mode={mode} runs={len(values)} mean_ms={mean_ms(values):.3f} min_ms={min(values)/1000.0:.3f} max_ms={max(values)/1000.0:.3f}")
    for name in sorted(stage_values[mode]):
        print(f"stage mode={mode} name={name} mean_ms={mean_ms(stage_values[mode][name]):.3f}")

if by_mode["direct-source"] and by_mode["legacy-preprocess"]:
    direct = mean_ms(by_mode["direct-source"])
    legacy = mean_ms(by_mode["legacy-preprocess"])
    saved = legacy - direct
    speedup = legacy / direct if direct else 0.0
    print(f"summary direct_mean_ms={direct:.3f} legacy_mean_ms={legacy:.3f} saved_ms={saved:.3f} speedup={speedup:.3f}x")
PY

