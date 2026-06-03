#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK="${WORK:-$SCRIPT_DIR/build_e2e}"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/home/gser/Ascend/cann}"
[ -f "$ASCEND_HOME_PATH/set_env.sh" ] && source "$ASCEND_HOME_PATH/set_env.sh"
source "$REPO/examples/env.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$REPO/build/bin:$PATH"

rm -rf "$WORK"
mkdir -p "$WORK"
# The model build + golden need torch. On a torch-less host (e.g. the real-NPU
# container) set FIXTURE_DIR to a dir holding pre-generated model.mlir + x.npy +
# expected.npy (staged from a torch host); otherwise build them here with gen.py.
if [ -n "${FIXTURE_DIR:-}" ] && [ -f "$FIXTURE_DIR/model.mlir" ]; then
  cp "$FIXTURE_DIR/model.mlir" "$FIXTURE_DIR/x.npy" "$FIXTURE_DIR/expected.npy" "$WORK/"
else
  PYTHONPATH="$REPO/python/torch" python3 "$SCRIPT_DIR/gen.py" \
    --out-dir "$WORK" --seq "${SEQ:-48}"
fi

cd "$REPO"
NETWORK_RUNNER_SKIP_AUTOTUNE=1 PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg "$WORK/model.mlir" \
  --inputs   "$WORK/x.npy" \
  --expected "$WORK/expected.npy" \
  --workdir  "$WORK" \
  --soc Ascend910B1 \
  --backend "${BACKEND:-sim}" \
  --atol 1e-2 --rtol 1e-2
