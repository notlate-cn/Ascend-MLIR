#!/usr/bin/env bash
# Default entry for the Ascend mainline pipeline.
#
# The previous transform-interpreter based script is retained as run-legacy.sh
# for comparison while the examples migrate to the mainline flow.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/run-mainline.sh" "$@"
