#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
ASCEND_ROOT="$ROOT/lib/Conversion/Ascend"
KERNELIR_ROOT="$ASCEND_ROOT/Translate/KernelIR"
COMPUTE_ROOT="$KERNELIR_ROOT/Compute"

fail() {
  echo "error: $*" >&2
  exit 1
}

search_pattern() {
  local pattern="$1"
  local path="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -n "$pattern" "$path"
  else
    grep -R -n -E "$pattern" "$path"
  fi
}

require_file() {
  local path="$1"
  [[ -f "$ROOT/$path" ]] || fail "missing expected architecture file: $path"
}

reject_pattern() {
  local path="$1"
  local pattern="$2"
  if search_pattern "$pattern" "$ROOT/$path" >/tmp/ascend_arch_check_match.txt; then
    cat /tmp/ascend_arch_check_match.txt >&2
    fail "unexpected pattern '$pattern' in $path"
  fi
}

require_pattern() {
  local path="$1"
  local pattern="$2"
  search_pattern "$pattern" "$ROOT/$path" >/dev/null ||
    fail "missing pattern '$pattern' in $path"
}

require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeLoweringInternal.h"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeLoweringPipeline.cpp"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeElementwiseLowering.cpp"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeFillLowering.cpp"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeMatmulLowering.cpp"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeLocalFallbackLowering.cpp"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeLoweringPreconditions.cpp"

reject_pattern \
  "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeLoweringPass.cpp" \
  "KernelizeInternalPasses|PreEmitInternalPasses"
require_pattern \
  "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeLoweringPass.cpp" \
  "prepareComputeLoweringPreconditions"

require_pattern \
  "include/Conversion/Ascend/Translate/KernelIR/Capabilities/BackendSupportMatrix.h" \
  "class BackendCapabilityProvider"
require_file \
  "lib/Conversion/Ascend/Translate/KernelIR/Capabilities/DefaultBackendCapabilityProvider.cpp"

require_pattern \
  "lib/Conversion/Ascend/Realize/TranslateMemoryBridge.h" \
  "class TranslateMemoryBridge"
require_pattern \
  "lib/Conversion/Ascend/Realize/TranslateMemoryBridge.cpp" \
  "DefaultTranslateMemoryBridge"

reject_pattern "lib/Conversion/Ascend/Realize/RealizePass.cpp" "buildMVPRealizePlans"
reject_pattern "include/Conversion/Ascend/Passes.td" "MVP|Phase 3B"

line_count=$(wc -l < "$COMPUTE_ROOT/ComputeOpConversion.cpp")
if [[ "$line_count" -ge 2500 ]]; then
  fail "ComputeOpConversion.cpp remains too large: $line_count lines"
fi
