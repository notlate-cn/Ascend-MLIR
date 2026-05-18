#!/usr/bin/env bash
# Ascend mainline smoke for transformer_dynamic.mlir.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../mainline-target-env.sh
source "$DIR/../mainline-target-env.sh"
AFIR_OPT="${AFIR_OPT:-afir-opt}"
AFIR_TRANSLATE="${AFIR_TRANSLATE:-afir-translate}"
RUNTIME_SESSION="${RUNTIME_SESSION:-runtime-session}"
PYTHON="${PYTHON:-python3}"
SOC="${SOC_VERSION:-Ascend910B1}"
RUNTIME_E2E=false
BATCH=1
SEQ=1
BLOCK_DIM=1
VERBOSE=false

require_arg() {
  local opt="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "missing value for ${opt}" >&2
    exit 2
  fi
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "unsupported shape: ${name} must be >= 1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime-e2e)
      RUNTIME_E2E=true
      shift
      ;;
    --batch)
      require_arg "$1" "${2:-}"
      BATCH="$2"
      shift 2
      ;;
    --seq)
      require_arg "$1" "${2:-}"
      SEQ="$2"
      shift 2
      ;;
    --block-dim)
      require_arg "$1" "${2:-}"
      BLOCK_DIM="$2"
      shift 2
      ;;
    --soc)
      require_arg "$1" "${2:-}"
      SOC="$2"
      shift 2
      ;;
    --log)
      VERBOSE=true
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

require_positive_int "BATCH" "$BATCH"
require_positive_int "SEQ" "$SEQ"
require_positive_int "BLOCK_DIM" "$BLOCK_DIM"

log() {
  if $VERBOSE; then
    echo "$@"
  fi
}

BUILD_DIR="$DIR/build_mainline"
NPY_DIR="$BUILD_DIR/npy"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
RUN_MANIFEST="$BUILD_DIR/run_manifest.json"
ACTUAL_OUTPUT_DIR="$BUILD_DIR/outputs"
VALIDATION_LOG="$BUILD_DIR/runtime_session.log"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" "$NPY_DIR" "$ACTUAL_OUTPUT_DIR"

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

if ! $RUNTIME_E2E; then
  exit 0
fi

"$PYTHON" "$DIR/gen_data.py" \
  --batch "$BATCH" \
  --seq "$SEQ" \
  --out-dir "$NPY_DIR" \
  --artifact-root "$ARTIFACT_ROOT" \
  --tiling-schema "$BUILD_DIR/tiling.json" \
  --run-manifest "$RUN_MANIFEST" \
  --actual-output-dir "$ACTUAL_OUTPUT_DIR" \
  --block-dim "$BLOCK_DIM"

rm -rf "$ARTIFACT_ROOT"
"$RUNTIME_SESSION" \
  --kernel "$BUILD_DIR/kernel.cpp" \
  --kernel-kind vec \
  --output "$ARTIFACT_ROOT" \
  --name kernel
log "transformer_dynamic.artifact_root=$ARTIFACT_ROOT"

"$RUNTIME_SESSION" \
  --run-manifest "$RUN_MANIFEST" \
  --run >"$VALIDATION_LOG" 2>&1
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]\|^\[DRVSTUB_LOG\]\|^\[FuncCache\]\|^ \|^=\|^\[TmSim\]\|^>>>>' \
  "$VALIDATION_LOG" || true
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"

echo "transformer_dynamic.runtime_session=pass"
echo "transformer_dynamic.validation=pass"
