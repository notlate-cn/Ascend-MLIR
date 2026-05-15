#!/usr/bin/env bash
# Ascend mainline smoke for transformer_dynamic.mlir.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
AFIR_OPT="${AFIR_OPT:-afir-opt}"

BUILD_DIR="$DIR/build_mainline"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

"$AFIR_OPT" "$DIR/transformer_dynamic.mlir" \
  --ascend-normalize \
  -o "$BUILD_DIR/step1_normalized.mlir"

"$AFIR_OPT" "$BUILD_DIR/step1_normalized.mlir" \
  --ascend-kernelize \
  -o "$BUILD_DIR/step2_kernelized.mlir"

if ! "$AFIR_OPT" "$BUILD_DIR/step2_kernelized.mlir" \
    --ascend-schedule='target-tile-policy=legacy-default' \
    --ascend-realize='materialization-mode=memory-space-annotate' \
    --ascend-compute-lower \
    -o "$BUILD_DIR/full_codegen.mlir" \
    2> "$BUILD_DIR/full_codegen.stderr"; then
  echo "transformer_dynamic.full_codegen=unexpected-gap" >&2
  cat "$BUILD_DIR/full_codegen.stderr" >&2
  exit 1
fi

echo "transformer_dynamic.mainline_prefix=pass"
echo "transformer_dynamic.transpose_kernelize_generalization=pass"
echo "transformer_dynamic.multi_kernel_func_metadata=per_kernel"
echo "transformer_dynamic.full_codegen=pass"
