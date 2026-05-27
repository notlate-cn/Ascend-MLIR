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

bash -n examples/broadcast-add-reduce/run-mainline.sh
