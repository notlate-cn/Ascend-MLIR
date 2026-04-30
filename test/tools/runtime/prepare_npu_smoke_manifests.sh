#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/npu_smoke"

usage() {
  cat <<'EOF' >&2
Usage:
  prepare_npu_smoke_manifests.sh vec <manifest> <artifact_root> <input0> <input1> <output> <expected>
  prepare_npu_smoke_manifests.sh mix <manifest> <artifact_root> <input0> <input1> <input2> <output> <expected>
EOF
  exit 2
}

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

render_manifest() {
  local template_path="$1"
  local output_path="$2"
  shift 2

  cp "${template_path}" "${output_path}"
  while [ "$#" -gt 1 ]; do
    local key="$1"
    local value
    value="$(escape_sed_replacement "$2")"
    sed -i.bak "s/@${key}@/${value}/g" "${output_path}"
    shift 2
  done
  rm -f "${output_path}.bak"
}

[ "$#" -ge 1 ] || usage
kind="$1"
shift

case "${kind}" in
  vec)
    [ "$#" -eq 6 ] || usage
    render_manifest \
      "${TEMPLATE_DIR}/vec_manifest.template.json" \
      "$1" \
      ARTIFACT_ROOT "$2" \
      INPUT0 "$3" \
      INPUT1 "$4" \
      OUTPUT "$5" \
      EXPECTED "$6"
    ;;
  mix)
    [ "$#" -eq 7 ] || usage
    render_manifest \
      "${TEMPLATE_DIR}/mix_manifest.template.json" \
      "$1" \
      ARTIFACT_ROOT "$2" \
      INPUT0 "$3" \
      INPUT1 "$4" \
      INPUT2 "$5" \
      OUTPUT "$6" \
      EXPECTED "$7"
    ;;
  *)
    usage
    ;;
esac
