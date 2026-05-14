#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

check_phase_headers() {
  local phase="$1"
  local allowed="$2"
  local header

  while IFS= read -r header; do
    if [[ "$header" != "$allowed" ]]; then
      echo "unexpected public Ascend ${phase} internal header: ${header}" >&2
      return 1
    fi
  done < <(find "include/Conversion/Ascend/${phase}" -maxdepth 1 -type f -name '*.h' | sort)
}

check_phase_headers Kernelize "include/Conversion/Ascend/Kernelize/KernelizePass.h"
check_phase_headers Schedule "include/Conversion/Ascend/Schedule/SchedulePass.h"
check_phase_headers Realize "include/Conversion/Ascend/Realize/RealizePass.h"
