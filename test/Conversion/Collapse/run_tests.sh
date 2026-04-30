#!/usr/bin/env bash
# Usage:
#   ./run_tests.sh --all              # run all .mlir cases in this directory
#   ./run_tests.sh <file.mlir>        # run a single case

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../../build"
BIN_DIR="${BUILD_DIR}/bin"
FILECHECK="/usr/lib/llvm-18/bin/FileCheck"

# Colours
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass=0
fail=0

run_case() {
  local mlir_file="$1"
  local name
  name="$(basename "$mlir_file")"

  # Extract the first // RUN: line and substitute %s → file path
  local run_line
  run_line=$(grep -m1 '^// RUN:' "$mlir_file" | sed 's|^// RUN: ||')
  if [[ -z "$run_line" ]]; then
    echo -e "${RED}SKIP${NC}  $name  (no RUN line)"
    return
  fi

  local cmd
  cmd="${run_line//%s/$mlir_file}"

  # Prepend build/bin to PATH so afir-opt and FileCheck resolve correctly
  if PATH="${BIN_DIR}:/usr/lib/llvm-18/bin:${PATH}" eval "$cmd" 2>/tmp/run_tests_err; then
    echo -e "${GREEN}PASS${NC}  $name"
    ((pass++)) || true
  else
    echo -e "${RED}FAIL${NC}  $name"
    echo "     cmd : $cmd"
    echo "     err : $(cat /tmp/run_tests_err | head -5)"
    ((fail++)) || true
  fi
}

if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 --all | <file.mlir>"
  exit 0
fi

if [[ "${1}" == "--all" ]]; then
  for f in "${SCRIPT_DIR}"/*.mlir; do
    run_case "$f"
  done
  echo ""
  echo "Results: ${pass} passed, ${fail} failed"
  [[ $fail -eq 0 ]]
else
  # Single file — accept bare name or full path
  target="$1"
  [[ "$target" != /* ]] && target="${SCRIPT_DIR}/${target}"
  run_case "$target"
  [[ $fail -eq 0 ]]
fi
