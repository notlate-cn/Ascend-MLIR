#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ ! -f "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh" ]]; then
  echo "scripts/resolve_ascend_env.sh not found" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/resolve_ascend_env.sh"
if ! resolve_ascend_home >/dev/null 2>&1; then
  echo "Ascend environment is unavailable; source CANN first" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/examples/env.sh"

log_file="$(mktemp "${TMPDIR:-/tmp}/broadcast-prepare-tiling.XXXXXX.log")"
trap 'rm -f "${log_file}"' EXIT

echo "--- Checking broadcast-add-reduce prepare-only tiling defaults ---"
if ! bash examples/broadcast-add-reduce/run.sh --prepare-runtime-artifacts --log \
    >"${log_file}" 2>&1; then
  echo "broadcast-add-reduce prepare-only failed" >&2
  tail -n 120 "${log_file}" >&2 || true
  exit 1
fi

run_manifest="${PROJECT_ROOT}/examples/broadcast-add-reduce/build_mainline/run_manifest.json"
if [[ ! -f "${run_manifest}" ]]; then
  echo "missing prepared run manifest: ${run_manifest}" >&2
  tail -n 120 "${log_file}" >&2 || true
  exit 1
fi
if [[ -f "${PROJECT_ROOT}/examples/broadcast-add-reduce/build_mainline/runtime_session.log" ]]; then
  echo "prepare-only path unexpectedly ran runtime-session --run" >&2
  exit 1
fi

python3 - "${run_manifest}" <<'PY'
import json
import struct
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
tasks = manifest.get("tasks") or [manifest]
task = next(
    (entry for entry in tasks if entry.get("task_id") == "broadcast_add_reducesum"),
    tasks[0],
)
binary = task.get("tiling", {}).get("binary")
if not binary:
    raise SystemExit("prepared task is missing tiling.binary")
binary_path = Path(binary)
if not binary_path.is_absolute():
    binary_path = manifest_path.parent / binary_path
if not binary_path.exists():
    raise SystemExit(f"missing tiling binary: {binary_path}")

raw = binary_path.read_bytes()
if len(raw) < 16:
    raise SystemExit(f"tiling binary too small: {binary_path} has {len(raw)} bytes")
tb_m, tb_n = struct.unpack_from("<qq", raw, 0)
if tb_m <= 0:
    raise SystemExit(f"TB_M must be positive in prepared tiling binary, got {tb_m}")
if tb_n <= 0:
    raise SystemExit(f"TB_N must be positive in prepared tiling binary, got {tb_n}")
print(f"broadcast-add-reduce prepared tiling: TB_M={tb_m} TB_N={tb_n}")
PY
