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

# ResNet has many inputs (60 BN buffers + image at the end) — enumerate by
# integer index so input_10.npy comes after input_9.npy (shell glob would sort
# lexically, putting 10 before 2). The export script writes them in the right
# order matching torch.export's named_buffers() traversal.
INPUTS=()
i=0
while [ -f "${OUTDIR}/input_${i}.npy" ]; do
  INPUTS+=( "${OUTDIR}/input_${i}.npy" )
  i=$((i + 1))
done

# network_runner.py uses argparse nargs="+" for --inputs, so all the npy
# paths follow a single --inputs flag.
conda run -n torch-mlir python "${WT_ROOT}/python/network_runner.py" \
  --input-linalg "${OUTDIR}/step0_linalg.mlir" \
  --inputs "${INPUTS[@]}" \
  --expected "${OUTDIR}/expected_0.npy" \
  --workdir "${OUTDIR}/work" \
  --max-phase "${MAX_PHASE}" \
  --atol 1e-2 --rtol 1e-2
