#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_SCRIPT="${REPO_ROOT}/examples/transformer/run-mainline.sh"
BUILD_DIR="${REPO_ROOT}/examples/transformer/build_mainline"

BATCH="${TRANSFORMER_CENSUS_BATCH:-1}"
SEQ="${TRANSFORMER_CENSUS_SEQ:-1}"
CASE_TIMEOUT="${TRANSFORMER_CENSUS_TIMEOUT:-900s}"
RUN_TIMEOUT="${RUN_TIMEOUT:-600s}"
USE_EXISTING="${TRANSFORMER_CENSUS_USE_EXISTING:-0}"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

require_tool() {
  command -v "$1" >/dev/null 2>&1 || fail "required tool '$1' is unavailable"
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    fail "${name} must be a positive integer"
  fi
}

echo "INFO: transformer kernel census entry"

[[ -f "${RUN_SCRIPT}" ]] || fail "transformer run script not found: ${RUN_SCRIPT}"
require_tool bash
require_tool python3
require_tool timeout
require_positive_int "TRANSFORMER_CENSUS_BATCH" "${BATCH}"
require_positive_int "TRANSFORMER_CENSUS_SEQ" "${SEQ}"

if [[ "${USE_EXISTING}" != "1" ]]; then
  log_file="$(mktemp "${TMPDIR:-/tmp}/afir-transformer-census.XXXXXX.log")"
  trap 'rm -f "${log_file}"' EXIT
  if ! RUN_TIMEOUT="${RUN_TIMEOUT}" timeout "${CASE_TIMEOUT}" bash "${RUN_SCRIPT}" \
      --runtime-e2e --batch "${BATCH}" --seq "${SEQ}" --log \
      >"${log_file}" 2>&1; then
    echo "FAIL [transformer-kernel-census batch=${BATCH} seq=${SEQ}]"
    tail -n 200 "${log_file}" || true
    exit 1
  fi
fi

python3 - "${BUILD_DIR}" "${BATCH}" "${SEQ}" <<'PY'
import collections
import json
import sys
from pathlib import Path

build_dir = Path(sys.argv[1])
batch = int(sys.argv[2])
seq = int(sys.argv[3])
artifact_manifest_path = build_dir / "artifact_manifest.json"
run_manifest_path = build_dir / "run_manifest.json"
runtime_log_path = build_dir / "runtime_session.log"

for path in (artifact_manifest_path, run_manifest_path, runtime_log_path):
    if not path.exists():
        raise SystemExit(f"missing transformer census input: {path}")

artifact_manifest = json.loads(artifact_manifest_path.read_text())
run_manifest = json.loads(run_manifest_path.read_text())
tasks = {task["task_id"]: task for task in run_manifest.get("tasks", [])}
entries = artifact_manifest.get("kernel_entries", [])
graph = artifact_manifest.get("kernelGraph", {})
edges = graph.get("edges", [])

valid_kinds = {"vec", "cube", "mix"}
kind_counts: collections.Counter[str] = collections.Counter()
for entry in entries:
    resources = entry.get("resources", {})
    kind = entry.get("kernelKind") or resources.get("kernelKind") or "vec"
    if kind not in valid_kinds:
        kind = "vec"
    kind_counts[kind] += 1

pred: dict[str, set[str]] = collections.defaultdict(set)
succ: dict[str, set[str]] = collections.defaultdict(set)
for edge in edges:
    src = edge.get("from")
    dst = edge.get("to")
    if not src or not dst:
        continue
    pred[dst].add(src)
    succ[src].add(dst)

roots = sorted(task_id for task_id in tasks if not pred[task_id])
leaves = sorted(task_id for task_id in tasks if not succ[task_id])

in_degree = {task_id: len(pred[task_id]) for task_id in tasks}
queue = collections.deque(task_id for task_id in tasks if in_degree[task_id] == 0)
depth = {task_id: 1 for task_id in tasks}
visited = []
while queue:
    task_id = queue.popleft()
    visited.append(task_id)
    for dst in succ[task_id]:
        depth[dst] = max(depth.get(dst, 1), depth[task_id] + 1)
        in_degree[dst] -= 1
        if in_degree[dst] == 0:
            queue.append(dst)
if len(visited) != len(tasks):
    raise SystemExit("kernel DAG contains a cycle or unknown task edge")

runtime_input_roots = [task_id for task_id in roots if task_id == "kernel_0"]
prepack_candidate_roots = [
    task_id for task_id in roots if task_id not in set(runtime_input_roots)
]

profile_count = None
for line in runtime_log_path.read_text(errors="replace").splitlines():
    if line.startswith("session.profile.count="):
        profile_count = int(line.split("=", 1)[1])
        break
if profile_count is None:
    raise SystemExit("missing session.profile.count in runtime session log")

print(f"transformer_census.batch={batch}")
print(f"transformer_census.seq={seq}")
print(f"transformer_census.kernel_count={len(entries)}")
print(f"transformer_census.task_count={len(tasks)}")
print(f"transformer_census.graph_edges={len(edges)}")
print(f"transformer_census.kind.vec={kind_counts['vec']}")
print(f"transformer_census.kind.cube={kind_counts['cube']}")
print(f"transformer_census.kind.mix={kind_counts['mix']}")
print(f"transformer_census.root_tasks={len(roots)}")
print(f"transformer_census.leaf_tasks={len(leaves)}")
print(f"transformer_census.critical_path_depth={max(depth.values(), default=0)}")
print(f"transformer_census.runtime_input_roots={len(runtime_input_roots)}")
print(f"transformer_census.prepack_candidate_roots={len(prepack_candidate_roots)}")
print("transformer_census.prepack_candidate_root_ids=" + ",".join(prepack_candidate_roots))
print(f"transformer_census.profile.count={profile_count}")
PY

echo "ALL TRANSFORMER KERNEL CENSUS PASSED"
