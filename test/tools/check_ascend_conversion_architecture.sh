#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
ASCEND_ROOT="$ROOT/lib/Conversion/Ascend"
KERNELIZE_ROOT="$ASCEND_ROOT/Kernelize"
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

require_dir() {
  local path="$1"
  [[ -d "$ROOT/$path" ]] || fail "missing expected architecture directory: $path"
}

reject_file() {
  local path="$1"
  [[ ! -e "$ROOT/$path" ]] || fail "unexpected flat architecture file: $path"
}

require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeLoweringInternal.h"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeLoweringPipeline.cpp"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeLoweringContext.cpp"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeTransposeLowering.cpp"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeReductionLowering.cpp"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeParallelGenericLowering.cpp"
require_file "lib/Conversion/Ascend/Translate/KernelIR/Compute/ComputeGatherLowering.cpp"
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
if [[ "$line_count" -ge 1000 ]]; then
  fail "ComputeOpConversion.cpp remains too large: $line_count lines"
fi

require_dir "lib/Conversion/Ascend/Kernelize/Semantic"
require_dir "lib/Conversion/Ascend/Kernelize/Preprocess"
require_dir "lib/Conversion/Ascend/Kernelize/Analysis"
require_dir "lib/Conversion/Ascend/Kernelize/Candidate"
require_dir "lib/Conversion/Ascend/Kernelize/Pattern"
require_dir "lib/Conversion/Ascend/Kernelize/Split"

require_file "lib/Conversion/Ascend/Kernelize/Semantic/KernelizeSemanticUtils.h"
require_file "lib/Conversion/Ascend/Kernelize/Semantic/KernelizeOpRegistry.h"
require_file "lib/Conversion/Ascend/Kernelize/Semantic/KernelizeExternalModels.cpp"
require_file "lib/Conversion/Ascend/Kernelize/Semantic/KernelizeOpInterface.cpp"
require_file "lib/Conversion/Ascend/Kernelize/Preprocess/MixMatmulSemantics.cpp"
require_file "lib/Conversion/Ascend/Kernelize/Preprocess/GatherElementwiseFusion.cpp"
require_file "lib/Conversion/Ascend/Kernelize/Preprocess/StructuredOpSemanticMarking.cpp"
require_file "lib/Conversion/Ascend/Kernelize/Analysis/DependencyAnalysis.h"
require_file "lib/Conversion/Ascend/Kernelize/Analysis/OpRoleClassification.h"
require_file "lib/Conversion/Ascend/Kernelize/Analysis/StructuralMarking.h"
require_file "lib/Conversion/Ascend/Kernelize/Candidate/CandidateClosure.h"
require_file "lib/Conversion/Ascend/Kernelize/Candidate/CandidateMergeAnalysis.h"
require_file "lib/Conversion/Ascend/Kernelize/Candidate/FusionCandidateAnalysis.h"
require_file "lib/Conversion/Ascend/Kernelize/Candidate/HorizontalFusionAnalysis.h"
require_file "lib/Conversion/Ascend/Kernelize/Candidate/KernelizeFamilyResolver.h"
require_file "lib/Conversion/Ascend/Kernelize/Pattern/HandwrittenContractRegistry.h"
require_file "lib/Conversion/Ascend/Kernelize/Pattern/KernelPattern.h"
require_file "lib/Conversion/Ascend/Kernelize/Split/KernelSplitPass.cpp"

reject_file "lib/Conversion/Ascend/Kernelize/CandidateClosure.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/CandidateClosure.h"
reject_file "lib/Conversion/Ascend/Kernelize/CandidateMergeAnalysis.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/CandidateMergeAnalysis.h"
reject_file "lib/Conversion/Ascend/Kernelize/DependencyAnalysis.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/DependencyAnalysis.h"
reject_file "lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/FusionCandidateAnalysis.h"
reject_file "lib/Conversion/Ascend/Kernelize/GatherElementwiseFusion.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/HandwrittenContractRegistry.h"
reject_file "lib/Conversion/Ascend/Kernelize/HorizontalFusionAnalysis.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/HorizontalFusionAnalysis.h"
reject_file "lib/Conversion/Ascend/Kernelize/KernelizeExternalModels.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/KernelizeFamilyResolver.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/KernelizeFamilyResolver.h"
reject_file "lib/Conversion/Ascend/Kernelize/KernelizeOpInterface.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/KernelizeOpRegistry.h"
reject_file "lib/Conversion/Ascend/Kernelize/KernelizeSemanticUtils.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/KernelizeSemanticUtils.h"
reject_file "lib/Conversion/Ascend/Kernelize/MixMatmulSemantics.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/OpRoleClassification.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/OpRoleClassification.h"
reject_file "lib/Conversion/Ascend/Kernelize/StructuralMarking.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/StructuralMarking.h"
reject_file "lib/Conversion/Ascend/Kernelize/StructuredOpSemanticMarking.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/KernelPattern.cpp"
reject_file "lib/Conversion/Ascend/Kernelize/KernelPattern.h"
reject_file "lib/Conversion/Ascend/Kernelize/KernelSplitPass.cpp"
