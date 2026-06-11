#!/usr/bin/env bash
# Remove CPU camodel simulation artifacts (dump/vcd/summary/profile files)
# left behind in example directories.
#
# Usage:
#   scripts/clean_camodel_artifacts.sh [-n|--dry-run] [PATH ...]
# Defaults to scanning ./examples relative to the repo root.

set -euo pipefail

dry_run=0
paths=()
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run) dry_run=1 ;;
    -h|--help)
      sed -n '2,9p' "$0"; exit 0 ;;
    *) paths+=("$arg") ;;
  esac
done

if [[ ${#paths[@]} -eq 0 ]]; then
  repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
  paths=("$repo_root/examples")
fi

# Patterns produced by the camodel simulator.
patterns=(
  'core*.dump'
  'core*_wave.vcd'
  'core*_summary_log'
  'profile_*_log*.toml'
  'npu_log*'
  'msprof*'
  'mcu_log.dump'
  'stars_log*.dump'
  'ffts_verify_log*.log'
)

total=0
total_bytes=0
for root in "${paths[@]}"; do
  [[ -e "$root" ]] || { echo "skip (not found): $root" >&2; continue; }
  for pat in "${patterns[@]}"; do
    while IFS= read -r -d '' f; do
      sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
      total=$((total + 1))
      total_bytes=$((total_bytes + sz))
      if [[ $dry_run -eq 1 ]]; then
        echo "would remove: $f ($sz B)"
      else
        rm -f "$f"
      fi
    done < <(find "$root" -type f -name "$pat" -print0)
  done
done

human=$(numfmt --to=iec --suffix=B "$total_bytes" 2>/dev/null || echo "${total_bytes}B")
if [[ $dry_run -eq 1 ]]; then
  echo "dry-run: $total files, $human"
else
  echo "removed: $total files, $human"
fi
