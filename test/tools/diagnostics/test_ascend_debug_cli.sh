#!/usr/bin/env bash
# Smoke test: collect runs develop's 6-stage pipeline + open renders HTML.
set -euo pipefail

INPUT="$1"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

ascend-debug collect "$INPUT" --out "$WORK/run"

STAGES_OK="$(find "$WORK/run/stages" -name '0[1-6]0-*.mlir' | wc -l | tr -d ' ')"
echo "STAGES_OK=$STAGES_OK"

ascend-debug open "$WORK/run" --no-browser >/dev/null
if find "$WORK/run" -name '*.html' | grep -q .; then
  echo "OPEN_OK=1"
else
  echo "OPEN_OK=0"
fi
