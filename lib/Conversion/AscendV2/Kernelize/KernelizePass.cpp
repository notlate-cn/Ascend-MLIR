//===- KernelizePass.cpp - Ascend V2 kernelize pass -----------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Kernelize/KernelizePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "Conversion/AscendV2/Kernelize/CandidateMergeAnalysis.h"
#include "Conversion/AscendV2/Kernelize/DependencyAnalysis.h"
#include "Conversion/AscendV2/Kernelize/FusionCandidateAnalysis.h"
#include "Conversion/AscendV2/Kernelize/HorizontalFusionAnalysis.h"
#include "Conversion/AscendV2/Kernelize/KernelPattern.h"
#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "Conversion/AscendV2/Kernelize/OpRoleClassification.h"
#include "Conversion/AscendV2/Kernelize/StructuralMarking.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

#define GEN_PASS_DECL_ASCENDKERNELIZEPASS
#define GEN_PASS_DEF_ASCENDKERNELIZEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::afir::ascend::v2::kernelize;

namespace {

struct KernelizeReportEntry {
  std::string opRole;
  std::string kernelPattern;
  unsigned primaryOpCount = 0;
};

bool isFuncOp(Operation *op) {
  return op->getName().getStringRef() == "func.func";
}

void clearOwnedKernelizeAttrs(ModuleOp module) {
  auto clearAttrs = [](Operation *op) {
    op->removeAttr(kOpRolesAttr);
    op->removeAttr(kOpRoleAttr);
    op->removeAttr(kKernelAttr);
    op->removeAttr(kPrimaryAttr);
  };

  clearAttrs(module.getOperation());
  module.walk(clearAttrs);
}

void emitKernelizeReport(ArrayRef<KernelizeReportEntry> entries) {
  llvm::errs() << "Kernelize report\n";
  for (const KernelizeReportEntry &entry : entries) {
    llvm::errs() << "  op_role = \"" << entry.opRole << "\"\n";
    llvm::errs() << "  kernel_pattern = \"" << entry.kernelPattern << "\"\n";
    llvm::errs() << "  primary_ops = " << entry.primaryOpCount << "\n";
  }
}

SmallVector<KernelizeReportEntry>
buildKernelizeReportEntries(ArrayRef<KernelPattern> patterns) {
  SmallVector<KernelizeReportEntry> entries;
  for (const KernelPattern &pattern : patterns) {
    Operation *roleOp = !pattern.primaryOps.empty()
                            ? pattern.primaryOps.front()
                            : (!pattern.internalOps.empty()
                                   ? pattern.internalOps.front()
                                   : nullptr);
    if (!roleOp)
      continue;

    auto role = roleOp->getAttrOfType<StringAttr>(kOpRoleAttr);
    StringRef roleName = role ? role.getValue() : StringRef("unsupported");
    entries.push_back(KernelizeReportEntry{
        roleName.str(), pattern.kernelName,
        static_cast<unsigned>(pattern.primaryOps.size())});
  }
  return entries;
}

} // namespace

namespace mlir::afir {

struct AscendKernelizePass
    : public ::impl::AscendKernelizePassBase<AscendKernelizePass> {
  using AscendKernelizePassBase::AscendKernelizePassBase;

  void runOnOperation() override {
    ::mlir::ascend::v2::DebugOptions options{
        ::mlir::ascend::v2::parseDebugStage(debugStage), dumpReport};
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      ::mlir::ascend::v2::emitStageHeader(
          llvm::errs(), ::mlir::ascend::v2::DebugStage::Kernelize,
          getArgument());

    ModuleOp module = getOperation();
    if (module
            .walk([&](Operation *op) {
              if (!isFuncOp(op))
                return WalkResult::advance();

              auto normalized = op->getAttrOfType<BoolAttr>(kNormalizedAttr);
              if (normalized && normalized.getValue())
                return WalkResult::advance();

              op->emitError() << "requires ascend.v2.normalized";
              return WalkResult::interrupt();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    FailureOr<DependencyAnalysisResult> depResult =
        DependencyAnalyzer().analyze(module);
    if (failed(depResult)) {
      signalPassFailure();
      return;
    }

    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      emitDependencyAnalysisReport(llvm::errs(), *depResult);

    if (failed(StructuralMarker().mark(module, *depResult))) {
      signalPassFailure();
      return;
    }
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      emitStructuralMarkingReport(llvm::errs(), *depResult);

    clearOwnedKernelizeAttrs(module);

    FailureOr<OpRoleMap> roleMap = OpRoleClassifier().classify(*depResult);
    if (failed(roleMap)) {
      signalPassFailure();
      return;
    }
    attachRoleAttributes(module, *roleMap);
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      emitOpRoleClassificationReport(llvm::errs(), *depResult, *roleMap);

    KernelizeConfig config;
    SmallVector<FusionCandidate> fusionCandidates =
        FusionCandidateAnalyzer().analyze(*depResult, *roleMap, config);
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      emitFusionCandidateReport(llvm::errs(), fusionCandidates,
                                depResult->index);

    SmallVector<MergedCandidate> mergedCandidates =
        CandidateMergeAnalyzer().analyze(fusionCandidates, *depResult, config);
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      emitCandidateMergeReport(llvm::errs(), mergedCandidates,
                               depResult->index);

    SmallVector<HorizontalFusionCandidate> horizontalCandidates =
        HorizontalFusionAnalyzer().analyze(fusionCandidates, mergedCandidates,
                                           *depResult, config);
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      emitHorizontalFusionReport(llvm::errs(), horizontalCandidates);

    KernelPatternGraph graph = KernelPatternBuilder().build(
        fusionCandidates, mergedCandidates, horizontalCandidates, *depResult);
    SmallVector<KernelPattern> patterns =
        KernelPartitioner().partition(graph, *depResult);
    attachKernelPatternAttributes(module, patterns);

    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize)) {
      emitKernelPatternGraphReport(llvm::errs(), graph, depResult->index);
      emitKernelPartitionReport(llvm::errs(), patterns, depResult->index);
    }

    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      emitKernelizeReport(buildKernelizeReportEntries(patterns));
  }
};

std::unique_ptr<Pass> createAscendKernelizePass() {
  return std::make_unique<AscendKernelizePass>();
}

} // namespace mlir::afir
