#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

allowed_headers=(
  "include/Conversion/Ascend/Translate/KernelIR/Capabilities/BackendSupportMatrix.h"
  "include/Conversion/Ascend/Translate/KernelIR/ComputeLoweringPass.h"
  "include/Conversion/Ascend/Translate/KernelIR/Capabilities/ElementwiseBodyOpRegistry.h"
  "include/Conversion/Ascend/Translate/KernelIR/Capabilities/LinalgBodyClassifier.h"
  "include/Conversion/Ascend/Translate/KernelIR/KernelIRUtils.h"
  "include/Conversion/Ascend/Translate/PreEmit/PreEmitPublicPasses.h"
  "include/Conversion/Ascend/Common/Attributes.h"
  "include/Conversion/Ascend/Common/SymbolConstraints.h"
  "include/Conversion/Ascend/Debug/DebugOptions.h"
  "include/Conversion/Ascend/Kernelize/KernelizeExternalModels.h"
  "include/Conversion/Ascend/Kernelize/KernelizeOpInterface.h"
  "include/Conversion/Ascend/Kernelize/KernelizePass.h"
  "include/Conversion/Ascend/Normalize/NormalizePass.h"
  "include/Conversion/Ascend/Passes.h"
  "include/Conversion/Ascend/Realize/RealizePass.h"
  "include/Conversion/Ascend/Schedule/SchedulePass.h"
)

is_allowed() {
  local header="$1"
  local allowed
  for allowed in "${allowed_headers[@]}"; do
    if [[ "$header" == "$allowed" ]]; then
      return 0
    fi
  done
  return 1
}

seen_headers=()
while IFS= read -r header; do
  seen_headers+=("$header")
  if ! is_allowed "$header"; then
    echo "unexpected public Ascend header: ${header}" >&2
    exit 1
  fi

  ifndef_guard="$(awk '/^#ifndef / {print $2; exit}' "$header")"
  define_guard="$(awk '/^#define / {print $2; exit}' "$header")"
  if [[ -z "$ifndef_guard" || "$ifndef_guard" != "$define_guard" ]]; then
    echo "invalid Ascend header guard in ${header}" >&2
    exit 1
  fi
  case "$ifndef_guard" in
    ASCEND_MLIR_CONVERSION_ASCEND_*) ;;
    *)
      echo "Ascend header guard must start with ASCEND_MLIR_CONVERSION_ASCEND_: ${header}: ${ifndef_guard}" >&2
      exit 1
      ;;
  esac
done < <(find "include/Conversion/Ascend" -type f -name '*.h' | sort)

for allowed in "${allowed_headers[@]}"; do
  seen=false
  for header in "${seen_headers[@]}"; do
    if [[ "$header" == "$allowed" ]]; then
      seen=true
      break
    fi
  done
  if [[ "$seen" != true ]]; then
    echo "missing public Ascend header: ${allowed}" >&2
    exit 1
  fi
done
