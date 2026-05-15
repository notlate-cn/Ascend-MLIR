#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

check_phase_headers() {
  local phase="$1"
  shift
  local allowed
  local found=()
  local header

  while IFS= read -r header; do
    found+=("$header")
    local ok=false
    for allowed in "$@"; do
      if [[ "$header" == "$allowed" ]]; then
        ok=true
        break
      fi
    done
    if [[ "$ok" != true ]]; then
      echo "unexpected public Ascend ${phase} internal header: ${header}" >&2
      return 1
    fi
  done < <(find "include/Conversion/Ascend/${phase}" -maxdepth 1 -type f -name '*.h' | sort)

  for allowed in "$@"; do
    local seen=false
    for header in "${found[@]}"; do
      if [[ "$header" == "$allowed" ]]; then
        seen=true
        break
      fi
    done
    if [[ "$seen" != true ]]; then
      echo "missing public Ascend ${phase} header: ${allowed}" >&2
      return 1
    fi
  done
}

check_phase_headers Kernelize \
  "include/Conversion/Ascend/Kernelize/KernelizePass.h" \
  "include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h"
check_phase_headers Schedule "include/Conversion/Ascend/Schedule/SchedulePass.h"
check_phase_headers Realize "include/Conversion/Ascend/Realize/RealizePass.h"
