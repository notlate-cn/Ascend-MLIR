#!/usr/bin/env bash
# Drive: export tiny GPT-2 -> network_runner pipeline up to --max-phase.
# Usage: examples/gpt2-e2e/run.sh [MAX_PHASE] [OUTDIR]
set -euo pipefail
MAX_PHASE="${1:-1}"
OUTDIR="${2:-/tmp/gpt2_e2e/tiny}"

EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_ROOT="$(cd "$EX_DIR/../.." && pwd)"
source "${EX_DIR}/env_sibling.sh"

conda run -n torch-mlir python "${EX_DIR}/export_gpt2.py" \
  --n-layer 12 --n-head 2 --n-embd 64 --vocab 1024 --seq 8 \
  --dtype fp32 --outdir "${OUTDIR}"

# GPT-2 may have nn.Buffer-lifted inputs (causal mask in older transformers;
# none in 5.5+). Enumerate input_0..input_N by integer index — shell glob is
# lexical and would put input_10 before input_2.
INPUTS=()
i=0
while [ -f "${OUTDIR}/input_${i}.npy" ]; do
  INPUTS+=( "${OUTDIR}/input_${i}.npy" )
  i=$((i + 1))
done

conda run -n torch-mlir python "${WT_ROOT}/python/network_runner.py" \
  --input-linalg "${OUTDIR}/step0_linalg.mlir" \
  --inputs "${INPUTS[@]}" \
  --expected "${OUTDIR}/expected_0.npy" \
  --workdir "${OUTDIR}/work" \
  --max-phase "${MAX_PHASE}" \
  --atol 1e-2 --rtol 1e-2
