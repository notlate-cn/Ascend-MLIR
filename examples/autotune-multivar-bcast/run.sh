#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK="${WORK:-$SCRIPT_DIR/build_e2e}"

source /home/gser/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="/home/gser/Ascend/cann-9.0.0/x86_64-linux/simulator/Ascend910B1/lib:/home/gser/Ascend/cann-9.0.0/x86_64-linux/lib64:/home/gser/Ascend/cann-9.0.0/x86_64-linux/devlib/linux/x86_64:${LD_LIBRARY_PATH:-}"
export PATH="$REPO/build/bin:$PATH"

rm -rf "$WORK"; mkdir -p "$WORK"
python3 "$SCRIPT_DIR/gen_inputs.py" --out-dir "$WORK"

cd "$REPO"
PYTHONPATH=python python3 python/network_runner.py \
  --input-linalg "$SCRIPT_DIR/model.mlir" \
  --inputs   "$WORK/a.npy" "$WORK/b.npy" "$WORK/init.npy" \
  --expected "$WORK/expected.npy" \
  --workdir  "$WORK" \
  --soc Ascend910B1 \
  --atol 1e-2 --rtol 1e-2
