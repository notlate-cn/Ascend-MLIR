#!/usr/bin/env bash
# Run the GPT-2 from-hidden network through network_runner.
#   BACKEND=sim (default) or BACKEND=npu (phase-5 on device).
# Artifact (step0_linalg.mlir + input_*.npy + expected_0.npy) is NOT in git
# (485MB); point GPT2_ARTIFACT at where it was staged (default /data/gser).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/data/nyh/Ascend/latest}"
[ -f "$ASCEND_HOME_PATH/set_env.sh" ] && source "$ASCEND_HOME_PATH/set_env.sh"
source "$REPO/examples/env.sh"
# env.sh reassigns SCRIPT_DIR/PROJECT_ROOT (non-local) → restore ours.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$REPO/build/bin:$PATH"

ART="${GPT2_ARTIFACT:-/data/gser/gpt2-artifact}"
NI=$(( $(cat "$ART/hidden_input_index.txt") + 1 ))
INPUTS=$(python3 -c "print(' '.join(f'$ART/input_{i}.npy' for i in range($NI)))")

cd "$REPO"
NETWORK_RUNNER_SKIP_AUTOTUNE=1 PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg "$ART/step0_linalg.mlir" \
  --inputs $INPUTS \
  --expected "$ART/expected_0.npy" \
  --workdir "${GPT2_WORK:-/data/gser/gpt2_npu_work}" \
  --soc Ascend910B1 \
  --backend "${BACKEND:-sim}" \
  --atol 1e-2 --rtol 1e-2
