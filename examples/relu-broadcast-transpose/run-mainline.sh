#!/usr/bin/env bash
# Thin example wrapper: generate demo tensors, then run the user-facing case.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../mainline-target-env.sh
source "$DIR/../mainline-target-env.sh"
ASCEND_DEBUG="${ASCEND_DEBUG:-ascend-debug}"
PYTHON="${PYTHON:-python3}"

M=640
N=500
SEED=42
SOC="${SOC_VERSION:-Ascend910B1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --m)
      M="$2"
      shift 2
      ;;
    --n)
      N="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --soc)
      SOC="$2"
      shift 2
      ;;
    --log)
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

CASE_JSON="$DIR/case.json"
BUILD_DIR="$DIR/build_mainline"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

VALIDATION_LOG="$BUILD_DIR/runtime-session.run.log"

echo "relu-broadcast-transpose: generate data M=$M N=$N seed=$SEED"
"$PYTHON" "$DIR/gen_inputs.py" --m "$M" --n "$N" --seed "$SEED" \
  --out-dir "$BUILD_DIR"

echo "relu-broadcast-transpose: ascend-debug run $CASE_JSON"
SOC_VERSION="$SOC" "$ASCEND_DEBUG" run "$CASE_JSON" --out "$BUILD_DIR"
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"
grep -E '^session\.(backend|result|validation)=' "$VALIDATION_LOG"
