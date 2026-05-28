#!/usr/bin/env bash
# Drive: export ResNet-18 -> network_runner pipeline up to --max-phase.
# Usage: examples/resnet18-e2e/run.sh [MAX_PHASE] [OUTDIR]
set -euo pipefail
MAX_PHASE="${1:-1}"
OUTDIR="${2:-/tmp/resnet18_e2e/r18}"

EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_ROOT="$(cd "$EX_DIR/../.." && pwd)"
source "${EX_DIR}/env_sibling.sh"

conda run -n torch-mlir python "${EX_DIR}/export_resnet18.py" \
  --batch 1 --size 224 --dtype fp32 --outdir "${OUTDIR}"

conda run -n torch-mlir python "${WT_ROOT}/python/network_runner.py" \
  --input-linalg "${OUTDIR}/step0_linalg.mlir" \
  --inputs "${OUTDIR}/input_0.npy" \
  --expected "${OUTDIR}/expected_0.npy" \
  --workdir "${OUTDIR}/work" \
  --max-phase "${MAX_PHASE}" \
  --atol 1e-2 --rtol 1e-2
