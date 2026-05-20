#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: collect-plog.sh --out-dir DIR [--since-minutes N]

Collect lightweight real-NPU diagnostics into DIR.
EOF
}

OUT_DIR=""
SINCE_MINUTES="${ASCEND_MLIR_CI_PLOG_SINCE_MINUTES:-120}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --since-minutes)
      SINCE_MINUTES="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  echo "--out-dir is required" >&2
  usage >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

{
  date -Is
  echo "ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID:-}"
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info || true
  else
    echo "npu-smi not found"
  fi
} >"${OUT_DIR}/npu-smi.txt" 2>&1 || true

PLOG_PATHS=(
  /root/ascend/log/debug/plog
  /var/log/npu/slog
  /var/log/npu/plog
  /root/ascend/log
)

{
  echo "searched_at=$(date -Is)"
  echo "since_minutes=${SINCE_MINUTES}"
  for path in "${PLOG_PATHS[@]}"; do
    if [[ -e "${path}" ]]; then
      echo "== ${path} =="
      find "${path}" -type f -mmin "-${SINCE_MINUTES}" -print0 2>/dev/null |
        xargs -0 grep -Hn "errorStr" 2>/dev/null || true
    fi
  done
} >"${OUT_DIR}/plog-errorStr.txt" 2>&1 || true

{
  echo "searched_at=$(date -Is)"
  for path in "${PLOG_PATHS[@]}"; do
    if [[ -e "${path}" ]]; then
      echo "== ${path} =="
      find "${path}" -type f -mmin "-${SINCE_MINUTES}" -print 2>/dev/null |
        sort |
        tail -n 200
    fi
  done
} >"${OUT_DIR}/plog-files.txt" 2>&1 || true
