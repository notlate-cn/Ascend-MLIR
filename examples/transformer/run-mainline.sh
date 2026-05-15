#!/usr/bin/env bash
# Ascend mainline smoke for transformer_dynamic.mlir.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../mainline-target-env.sh
source "$DIR/../mainline-target-env.sh"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
SOC="${SOC_VERSION:-Ascend910B1}"

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
    --ascend-schedule="target-tile-policy=target-aware cann-root=${CANN_ROOT} soc=${SOC}" \
    --ascend-realize='materialization-mode=memory-space-annotate' \
    --ascend-compute-lower \
    -o "$BUILD_DIR/full_codegen.mlir" \
    2> "$BUILD_DIR/full_codegen.stderr"; then
  echo "transformer_dynamic.full_codegen=unexpected-gap" >&2
  cat "$BUILD_DIR/full_codegen.stderr" >&2
  exit 1
fi

if ! "$AFIR_OPT" "$BUILD_DIR/full_codegen.mlir" \
    --ascend-parallelize \
    --ascend-prepare-for-emit \
    --ascend-canonicalize-cann-signature \
    -o "$BUILD_DIR/phase5_cann.mlir" \
    2> "$BUILD_DIR/phase5_backend.stderr"; then
  echo "transformer_dynamic.phase5_backend=unexpected-gap" >&2
  cat "$BUILD_DIR/phase5_backend.stderr" >&2
  exit 1
fi

if ! "$AFIR_TRANSLATE" -mlir-to-cann "$BUILD_DIR/phase5_cann.mlir" \
    --tiling-space-out="$BUILD_DIR/tiling.json" \
    --runtime-manifest-out="$BUILD_DIR/runtime_manifest.json" \
    --host-tiling-out="$BUILD_DIR/host_tiling.cpp" \
    --cann-soc="$SOC" \
    -o "$BUILD_DIR/kernel.cpp" \
    2> "$BUILD_DIR/phase5_translate.stderr"; then
  echo "transformer_dynamic.phase5_translate=unexpected-gap" >&2
  cat "$BUILD_DIR/phase5_translate.stderr" >&2
  exit 1
fi

echo "transformer_dynamic.mainline_prefix=pass"
echo "transformer_dynamic.transpose_kernelize_generalization=pass"
echo "transformer_dynamic.multi_kernel_func_metadata=per_kernel"
echo "transformer_dynamic.phase5_backend=pass"
echo "transformer_dynamic.phase5_translate=pass"
echo "transformer_dynamic.runtime_artifacts=pass"
echo "transformer_dynamic.full_codegen=pass"
