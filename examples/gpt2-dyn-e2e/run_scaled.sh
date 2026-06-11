#!/usr/bin/env bash
# Run the real-GPT-2-dimension dynamic stack (gen_scaled.py) through
# network_runner. Weights are npy INPUTS, so there are many input_*.npy.
#   N_LAYER (default 12), SEQ (default 48), BACKEND (sim|npu).
#   FIXTURE_DIR: torch-less host (real-NPU container) — pre-staged model.mlir +
#                input_*.npy + expected.npy live there; otherwise built here.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK="${WORK:-$SCRIPT_DIR/build_scaled}"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/home/gser/Ascend/cann}"
[ -f "$ASCEND_HOME_PATH/set_env.sh" ] && source "$ASCEND_HOME_PATH/set_env.sh"
source "$REPO/examples/env.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$REPO/build/bin:$PATH"

rm -rf "$WORK"
mkdir -p "$WORK"
if [ -n "${FIXTURE_DIR:-}" ] && [ -f "$FIXTURE_DIR/model.mlir" ]; then
  cp "$FIXTURE_DIR"/model.mlir "$FIXTURE_DIR"/expected.npy "$WORK/"
  cp "$FIXTURE_DIR"/input_*.npy "$WORK/"
else
  PYTHONPATH="$REPO/python/torch" python3 "$SCRIPT_DIR/gen_scaled.py" \
    --out-dir "$WORK" --n-layer "${N_LAYER:-12}" --seq "${SEQ:-48}"
fi

NI=$(ls "$WORK"/input_*.npy | wc -l)
INPUTS=$(python3 -c "import sys; print(' '.join(f'$WORK/input_{i}.npy' for i in range(int(sys.argv[1]))))" "$NI")

cd "$REPO"
# shellcheck disable=SC2086
NETWORK_RUNNER_SKIP_AUTOTUNE=1 PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg "$WORK/model.mlir" \
  --inputs $INPUTS \
  --expected "$WORK/expected.npy" \
  --workdir  "$WORK" \
  --soc Ascend910B1 \
  --backend "${BACKEND:-sim}" \
  --atol 1e-2 --rtol 1e-2
