#!/usr/bin/env bash
# kg1-hang repro runner.
#
# Usage:
#   ./run.sh              # H4 baseline (full original behavior)
#   ./run.sh h1           # after manually patching network_host.cpp middle → memcpy
#   ./run.sh h3           # H3: skip aclInit via ASCEND_MLIR_FORCE_HOST_MODE
#
# Each variant is wrapped in `timeout` so hang shows as exit-code 124, not infinite wait.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

VARIANT="${1:-h4}"
WORK="${WORK:-$SCRIPT_DIR/build_${VARIANT}}"
TIMEOUT="${TIMEOUT:-120}"

source /home/gser/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="/home/gser/Ascend/cann-9.0.0/x86_64-linux/simulator/Ascend910B1/lib:/home/gser/Ascend/cann-9.0.0/x86_64-linux/lib64:/home/gser/Ascend/cann-9.0.0/x86_64-linux/devlib/linux/x86_64:${LD_LIBRARY_PATH:-}"
export PATH="$REPO/build/bin:$PATH"

# Pick variant-specific inputs.
case "$VARIANT" in
  h1) INPUT_VARIANT=h1 ;;
  *)  INPUT_VARIANT=h4 ;;
esac

# Force-host-mode flag for H3 (skip aclInit in generated network()).
if [[ "$VARIANT" == "h3" ]]; then
  export ASCEND_MLIR_FORCE_HOST_MODE=1
fi

rm -rf "$WORK"
mkdir -p "$WORK"
python3 "$SCRIPT_DIR/gen_inputs.py" --out-dir "$WORK" --variant "$INPUT_VARIANT"

cd "$REPO"
set +e
timeout "${TIMEOUT}" \
  env PYTHONPATH=python python3 python/network_runner.py \
    --input-network "$SCRIPT_DIR" \
    --inputs   "$WORK/q.npy"        "$WORK/scale.npy" "$WORK/bias.npy" \
               "$WORK/k.npy"        "$WORK/v.npy"     "$WORK/mask.npy" \
               "$WORK/init_pre.npy" "$WORK/init_fa.npy" \
               "$WORK/kg1_bias.npy" "$WORK/init_post.npy" \
    --expected "$WORK/expected.npy" \
    --workdir  "$WORK" \
    --soc Ascend910B1 \
    --atol 1e-2 --rtol 1e-2
rc=$?
set -e

if [[ $rc -eq 124 ]]; then
  echo "[run.sh] TIMEOUT after ${TIMEOUT}s — variant=$VARIANT (likely hang)"
elif [[ $rc -ne 0 ]]; then
  echo "[run.sh] FAILED rc=$rc — variant=$VARIANT"
else
  echo "[run.sh] OK — variant=$VARIANT"
fi
exit $rc
