#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

pattern='AscendV2|ascend[.]v2|::v2([^[:alnum:]_]|$)|(^|[^[:alnum:]_])v2::|(^|[^[:alnum:]_])V2([^[:alnum:]_]|$)|ascend-v2-pipeline'
paths=(
  include/Conversion
  lib/Conversion
  tools/afir-opt
  test/Conversion
  test/unittests/Conversion
)

search_contents() {
  if command -v rg >/dev/null 2>&1; then
    rg -n "${pattern}" "${paths[@]}" --glob '!build/**'
  else
    find "${paths[@]}" -path '*/build/*' -prune -o -type f \
      -exec grep -nE "${pattern}" {} +
  fi
}

search_paths() {
  find "${paths[@]}" -path '*/build/*' -prune -o -print | grep -E "${pattern}"
}

found=0

if search_contents; then
  found=1
else
  result=$?
  if [[ ${result} -gt 1 ]]; then
    exit "${result}"
  fi
fi

if search_paths; then
  found=1
else
  result=$?
  if [[ ${result} -gt 1 ]]; then
    exit "${result}"
  fi
fi

if [[ ${found} -eq 1 ]]; then
  echo "found V2/v2 code naming in Ascend pipeline sources" >&2
  exit 1
fi
