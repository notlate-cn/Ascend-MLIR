#!/usr/bin/env bash
# Drive: export tiny BERT -> network_runner pipeline up to --max-phase.
# Usage: examples/bert-e2e/run.sh [MAX_PHASE] [BACKEND] [OUTDIR]
set -euo pipefail
# Note: dev-network's network_runner is sim-only (no --backend flag).
MAX_PHASE="${1:-1}"
OUTDIR="${2:-/tmp/bert_e2e/tiny}"

# Capture our own dir before sourcing env (env.sh reassigns SCRIPT_DIR).
BERT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BERT_WT_ROOT="$(cd "$BERT_DIR/../.." && pwd)"
source "${BERT_DIR}/env_sibling.sh"

# fp32: recognize-attention emits an invalid f16->f32 collapse_shape on the
# f16-input/f32-accumulate matmul pattern (BERT). fp32 keeps matmuls all-f32.
conda run -n torch-mlir python "${BERT_DIR}/export_bert.py" \
  --hidden 64 --heads 1 --seq 8 --dtype fp32 --outdir "${OUTDIR}"

conda run -n torch-mlir python "${BERT_WT_ROOT}/python/network_runner.py" \
  --input-linalg "${OUTDIR}/step0_linalg.mlir" \
  --inputs "${OUTDIR}/input_0.npy" \
  --expected "${OUTDIR}/expected_0.npy" \
  --workdir "${OUTDIR}/work" \
  --max-phase "${MAX_PHASE}" \
  --atol 1e-2 --rtol 1e-2
