#!/usr/bin/env bash
# Ascend mainline prefix smoke for transformer_dynamic.mlir.
#
# The full transformer graph is still outside the supported Phase 5 codegen
# closure. This script records the currently supported prefix and keeps the
# remaining gap explicit.

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

if "$AFIR_OPT" "$BUILD_DIR/step2_kernelized.mlir" \
  --ascend-compute-lower \
  -o "$BUILD_DIR/full_codegen_from_prefix_unexpected.mlir" \
  2> "$BUILD_DIR/full_codegen_from_prefix.stderr"; then
  echo "transformer_dynamic.full_codegen=unexpected-pass" >&2
  exit 1
fi

if ! grep -q "unsupported" "$BUILD_DIR/full_codegen_from_prefix.stderr"; then
  echo "transformer_dynamic.full_codegen=missing-unsupported-diagnostic" >&2
  cat "$BUILD_DIR/full_codegen_from_prefix.stderr" >&2
  exit 1
fi

echo "transformer_dynamic.mainline_prefix=pass"
echo "transformer_dynamic.full_codegen=deferred"
echo "transformer_dynamic.next_gap=unsupported_op_closure"
