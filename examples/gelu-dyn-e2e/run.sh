#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK="${WORK:-$SCRIPT_DIR/build_e2e}"

# CANN toolkit + sim libs, arch-aware. Respects a pre-set ASCEND_HOME_PATH
# (e.g. the real-NPU container's reused CANN); falls back to the local dev path.
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/home/gser/Ascend/cann}"
[ -f "$ASCEND_HOME_PATH/set_env.sh" ] && source "$ASCEND_HOME_PATH/set_env.sh"
source "$REPO/examples/env.sh"
# env.sh assigns SCRIPT_DIR/PROJECT_ROOT (non-local) → restore ours.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$REPO/build/bin:$PATH"

rm -rf "$WORK"
mkdir -p "$WORK"
python3 "$SCRIPT_DIR/gen_inputs.py" --out-dir "$WORK" --seq "${SEQ:-48}"

cd "$REPO"
# Default tilings (no autotune): the dynamic seq extent is a ShapeDerived tiling
# param resolved from the input npy shape at launch, not a tunable.
NETWORK_RUNNER_SKIP_AUTOTUNE=1 PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg "$SCRIPT_DIR/model.mlir" \
  --inputs   "$WORK/x.npy" "$WORK/b.npy" \
  --expected "$WORK/expected.npy" \
  --workdir  "$WORK" \
  --soc Ascend910B1 \
  --backend "${BACKEND:-sim}" \
  --atol 1e-2 --rtol 1e-2
