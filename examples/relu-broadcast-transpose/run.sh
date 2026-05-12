#!/usr/bin/env bash
# Default example entry: run the Ascend mainline pipeline.
#
# The previous transform-interpreter based script is retained as run-legacy.sh
# for comparison while unsupported examples are migrated.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/run-mainline.sh" "$@"
