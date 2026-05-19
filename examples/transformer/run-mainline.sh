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
RUN_TIMEOUT="${RUN_TIMEOUT:-600s}"
export ASCEND_DAV_SIM_VERSION="${ASCEND_DAV_SIM_VERSION:-dav_3002}"

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

if [[ "$RUNTIME_SESSION" != */* ]]; then
  RUNTIME_SESSION="$(command -v "$RUNTIME_SESSION")"
fi
if [[ -z "${AFIR_MIX_TILING_HELPER:-}" ]] && command -v mix-tiling-helper >/dev/null 2>&1; then
  export AFIR_MIX_TILING_HELPER="$(command -v mix-tiling-helper)"
fi

log() {
  if $VERBOSE; then
    echo "$@"
  fi
}

BUILD_DIR="$DIR/build_mainline"
NPY_DIR="$BUILD_DIR/npy"
ARTIFACT_ROOT="$BUILD_DIR/artifact"
TILING_SCHEMA_DIR="$BUILD_DIR/tiling_schemas"
MIX_COMPILE_NPY_ROOT="$BUILD_DIR/mix_compile_npy"
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
    --ascend-kernel-split \
    --ascend-realize='materialization-mode=memory-space-annotate' \
    --annotate-mix-matmul-semantics \
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

KERNEL_FUNC_COUNT="$(grep -c 'func.func @kernel_' "$BUILD_DIR/phase5_cann.mlir" || true)"
KERNEL_CPP_COUNT="$(grep -c '__aicore__ void' "$BUILD_DIR/kernel.cpp" || true)"
if (( KERNEL_FUNC_COUNT < 2 || KERNEL_CPP_COUNT != KERNEL_FUNC_COUNT )); then
  echo "transformer_dynamic.kernel_split=unexpected-gap" >&2
  echo "phase5_func_count=${KERNEL_FUNC_COUNT} cpp_kernel_count=${KERNEL_CPP_COUNT}" >&2
  exit 1
fi

echo "transformer_dynamic.mainline_prefix=pass"
echo "transformer_dynamic.transpose_kernelize_generalization=pass"
echo "transformer_dynamic.kernel_split=pass"
echo "transformer_dynamic.kernel_count=${KERNEL_CPP_COUNT}"
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
  --compiler-runtime-manifest "$BUILD_DIR/runtime_manifest.json" \
  --cann-mlir "$BUILD_DIR/phase5_cann.mlir" \
  --tiling-schema-dir "$TILING_SCHEMA_DIR" \
  --mix-compile-npy-root "$MIX_COMPILE_NPY_ROOT" \
  --run-manifest "$RUN_MANIFEST" \
  --actual-output-dir "$ACTUAL_OUTPUT_DIR" \
  --block-dim "$BLOCK_DIM"

rm -rf "$ARTIFACT_ROOT"
echo "transformer_dynamic.artifact_compile=start"
while IFS=$'\t' read -r kernel_id kernel_kind; do
  [[ -n "$kernel_id" ]] || continue
  compile_args=(
    --kernel "$BUILD_DIR/kernel.cpp" \
    --kernel-kind "$kernel_kind" \
    --output "$ARTIFACT_ROOT/$kernel_id" \
    --name "$kernel_id"
  )
  if [[ "$kernel_kind" == "mix" ]]; then
    compile_args+=(
      --cann-mlir "$BUILD_DIR/phase5_cann.mlir"
      --npy-dir "$MIX_COMPILE_NPY_ROOT/$kernel_id"
    )
  fi
  if $VERBOSE; then
    echo "transformer_dynamic.artifact_compile.kernel=${kernel_id} kind=${kernel_kind}"
  fi
  "$RUNTIME_SESSION" "${compile_args[@]}"
done < <("$PYTHON" - "$BUILD_DIR/runtime_manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    root = json.load(f)
valid = {"vec", "cube", "mix"}
for entry in root.get("kernel_entries", []):
    resources = entry.get("resources", {})
    kind = entry.get("kernelKind") or resources.get("kernelKind") or "vec"
    if kind not in valid:
        kind = "vec"
    print(f"{entry['kernel_id']}\t{kind}")
PY
)
echo "transformer_dynamic.artifact_compile=pass"
log "transformer_dynamic.artifact_root=$ARTIFACT_ROOT"
if $VERBOSE; then
  "$PYTHON" - "$RUN_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    root = json.load(f)
tasks = root.get("tasks", [])
print(f"transformer_dynamic.run_plan.tasks={len(tasks)}")
for index, task in enumerate(tasks):
    deps = task.get("dependencies") or []
    deps_text = ",".join(deps) if deps else "-"
    print(
        f"transformer_dynamic.run_plan[{index}]={task.get('task_id', '<missing>')} "
        f"deps={deps_text}"
    )
PY
fi

echo "transformer_dynamic.runtime_session=start"
if ! timeout "$RUN_TIMEOUT" "$RUNTIME_SESSION" \
    --run-manifest "$RUN_MANIFEST" \
    --run >"$VALIDATION_LOG" 2>&1; then
  echo "transformer_dynamic.runtime_session=timeout_or_fail"
  tail -n 200 "$VALIDATION_LOG" || true
  exit 1
fi
grep -v '^\[info\]\|^\[PEM_AIC_LOG\]\|^\[INFO\]\|^\[WARNING\]\|^\[DRVSTUB_LOG\]\|^\[FuncCache\]\|^ \|^=\|^\[TmSim\]\|^>>>>' \
  "$VALIDATION_LOG" || true
grep -q '^session.backend=sim$' "$VALIDATION_LOG"
grep -q '^session.result=success$' "$VALIDATION_LOG"
grep -q '^session.validation=pass$' "$VALIDATION_LOG"

echo "transformer_dynamic.runtime_session=pass"
echo "transformer_dynamic.validation=pass"
