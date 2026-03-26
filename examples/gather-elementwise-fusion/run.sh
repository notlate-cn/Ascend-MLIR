#!/bin/bash
# ============================================================
# gather + elementwise fusion complete pipeline demo
#
# Usage:
#   source examples/env.sh
#   bash examples/gather-elementwise-fusion/run.sh [--log]
#
# Graph: relu -> index_select(dim=1) -> add
#   data[M,N] --relu--> relu_out[M,N]
#   relu_out + indices[K] --index_select--> gathered[M,K]
#   gathered + bias[K] --add--> out[M,K]
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"

VERBOSE=false
for arg in "$@"; do
  case $arg in --log) VERBOSE=true ;; esac
done

log() { if $VERBOSE; then echo "$@"; fi }

echo "========================================================"
echo " gather + elementwise fusion pipeline"
echo "========================================================"

echo ""
echo "==================== [STAGE 0] Parse ===================="
$AFIR_OPT "$DIR/step0_input.mlir" -o "$DIR/step0_input_out.mlir"
log "  ok: step0_input_out.mlir"

echo ""
echo "==================== [STAGE 1] --mark-structured-ops ===================="
$AFIR_OPT --mark-structured-ops \
  "$DIR/step0_input.mlir" \
  -o "$DIR/step1_marked.mlir"
log "  ok: step1_marked.mlir"
log "  [gather_dim]"
log "$(grep 'gather_dim' "$DIR/step1_marked.mlir" || echo '  (not found)')"

echo ""
echo "==================== [STAGE 2] --transform-interpreter ===================="
"$AFIR_OPT" "$DIR/step1_marked.mlir" \
  "--transform-preload-library=transform-library-paths=$DIR/step2_transform.mlir" \
  "--transform-interpreter=entry-point=__transform_main" \
  --canonicalize --cse \
  -o "$DIR/step2_tiled.mlir"
log "  ok: step2_tiled.mlir"

echo ""
echo "==================== [STAGE 3] --one-shot-bufferize ===================="
$AFIR_OPT \
  "--one-shot-bufferize=bufferize-function-boundaries=true allow-return-allocs-from-loops=true function-boundary-type-conversion=identity-layout-map" \
  "$DIR/step2_tiled.mlir" \
  --cse \
  -o "$DIR/step3_bufferized.mlir"
log "  ok: step3_bufferized.mlir"

echo ""
echo "==================== [STAGE 4] --ascendc-buffer-placement ===================="
$AFIR_OPT \
  --ascendc-buffer-placement \
  "$DIR/step3_bufferized.mlir" \
  -o "$DIR/step4_buffer_placement.mlir"
log "  ok: step4_buffer_placement.mlir"

echo ""
echo "==================== [STAGE 5] --linalg-to-ascendc ===================="
$AFIR_OPT \
  --linalg-to-ascendc \
  "$DIR/step4_buffer_placement.mlir" \
  --canonicalize --cse \
  -o "$DIR/step5_ascendc.mlir"
log "  ok: step5_ascendc.mlir"

echo ""
echo "==================== [STAGE 6] --ascendc-parallelize ===================="
$AFIR_OPT "$DIR/step5_ascendc.mlir" \
  --ascendc-parallelize \
  --canonicalize --cse \
  -o "$DIR/step6_parallelize.mlir"
log "  ok: step6_parallelize.mlir"

echo ""
echo "==================== [STAGE 7] --ascendc-prepare-for-emit ===================="
$AFIR_OPT "$DIR/step6_parallelize.mlir" \
  --ascendc-prepare-for-emit \
  --canonicalize --cse \
  -o "$DIR/step7_kernel.mlir"
log "  ok: step7_kernel.mlir"

echo ""
echo "==================== [STAGE 8] ascir-translate ===================="
ASCIR_TRANSLATE="${ASCIR_TRANSLATE:-ascir-translate}"
if command -v "$ASCIR_TRANSLATE" &>/dev/null; then
  python3 -c "
import re, sys
content = open('$DIR/step7_kernel.mlir').read()
content = content.replace('module attributes {transform.with_named_sequence}', 'module')
content = re.sub(r'  transform\.named_sequence.*?^  \}\n', '', content, flags=re.DOTALL|re.MULTILINE)
sys.stdout.write(content)
" > "$DIR/step8_no_transform.mlir"
  "$ASCIR_TRANSLATE" -mlir-to-ascendc "$DIR/step8_no_transform.mlir" \
    -o "$DIR/step8_kernel.cpp"
  log "  ok: step8_kernel.cpp"
else
  log "  (ascir-translate not found, skipping stage 8)"
fi

echo ""
echo "========================================================"
echo " Pipeline complete!"
echo "========================================================"
