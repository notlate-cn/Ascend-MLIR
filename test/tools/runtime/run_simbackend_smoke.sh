#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_simbackend_examples.sh" \
  relu-broadcast-transpose \
  matmul-add-leakyrelu
