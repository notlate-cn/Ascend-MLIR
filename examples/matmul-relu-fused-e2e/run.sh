#!/usr/bin/env bash
# examples/matmul-relu-fused-e2e/run.sh
# CV-fusion (matmul + relu) end-to-end via vector-plan-codegen.
#
# This is the first CV-fusion gate that does NOT use the transform-interpreter
# path used by matmul-add-leakyrelu; it consumes raw `linalg.matmul + linalg.generic relu`
# and lets vector-plan handle group analysis, cube tile plan, and emission.
#
# Usage:
#   source examples/env.sh
#   bash examples/matmul-relu-fused-e2e/run.sh [--log]
#
# STAGE summary (compared to matmul-add-leakyrelu):
#   our STAGE 1-7: replaced by single `--vector-plan-codegen` invocation
#   our STAGE 8:   afir-translate -mlir-to-cann (unchanged)
#   STAGE 9/10:    sim run (TODO: mix-compiler integration for no-bias matmul)
set -euo pipefail
export ASCEND_DAV_SIM_VERSION=dav_3002

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"

VERBOSE=false
for arg in "$@"; do [[ $arg == "--log" ]] && VERBOSE=true; done
log() { $VERBOSE && echo "$@" || true; }

echo "=========================================================="
echo " matmul + relu CV-fusion via vector-plan-codegen"
echo "=========================================================="

# ── STAGE 1-7: vector-plan codegen (replaces transform+bufferize+place chain)
echo "=== [STAGE 1-7] vector-plan-codegen ==="
$AFIR_OPT --vector-plan-codegen \
  "$SCRIPT_DIR/step0_input.mlir" \
  -o "$SCRIPT_DIR/step7_cann.mlir"
log "  step7_cann.mlir done"

# ── STAGE 8: mlir → cann (.cpp)
echo "=== [STAGE 8] mlir-to-cann codegen ==="
$AFIR_TRANSLATE --mlir-to-cann \
  "$SCRIPT_DIR/step7_cann.mlir" \
  -o "$SCRIPT_DIR/step8_kernel.cpp"

# Validate the emitted kernel contains the cube + vector epilogue.
if ! grep -q 'mm.template IterateAll' "$SCRIPT_DIR/step8_kernel.cpp"; then
  echo "FAIL: emitted kernel missing IterateAll" >&2
  exit 2
fi
if ! grep -q 'Relu(' "$SCRIPT_DIR/step8_kernel.cpp"; then
  echo "FAIL: emitted kernel missing Relu epilogue" >&2
  exit 2
fi
echo "  step8_kernel.cpp emitted (cube + relu epilogue)"

# ── STAGE 9: test data
echo "=== [STAGE 9] gen test data ==="
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
mkdir -p "${DATA_DIR}"
python3 "$SCRIPT_DIR/gen_data.py" --out-dir "${DATA_DIR}"

# ── STAGE 10: sim run (TODO: mix-compiler needs no-bias matmul ABI support)
echo "=== [STAGE 10] sim run (deferred: needs mix-compiler ABI extension) ==="
echo "PASS [matmul-relu-fused-e2e codegen]"
