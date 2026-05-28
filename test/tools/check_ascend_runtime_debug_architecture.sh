#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
MATCHES="$(mktemp "${TMPDIR:-/tmp}/ascend-runtime-debug-arch.XXXXXX")"
trap 'rm -f "${MATCHES}"' EXIT

fail() {
  echo "error: $*" >&2
  exit 1
}

search_pattern() {
  local pattern="$1"
  local path="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -n "$pattern" "$path"
  else
    grep -R -n -E "$pattern" "$path"
  fi
}

reject_pattern() {
  local path="$1"
  local pattern="$2"
  if search_pattern "$pattern" "$ROOT/$path" >"${MATCHES}"; then
    cat "${MATCHES}" >&2
    fail "unexpected pattern '$pattern' in $path"
  fi
}

reject_pattern "lib/Runtime" "[Aa][Ff][Ii][Rr]"
reject_pattern "tools/ascend-debug" "[Aa][Ff][Ii][Rr]"
