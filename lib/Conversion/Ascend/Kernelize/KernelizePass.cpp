//===- KernelizePass.cpp - Ascend kernelize pass -----------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Kernelize/KernelizePass.h"

#include "Conversion/Ascend/Debug/DebugOptions.h"
#include "Conversion/Ascend/Kernelize/KernelizeExternalModels.h"
#include "Conversion/Ascend/Kernelize/Candidate/CandidateMergeAnalysis.h"
#include "Conversion/Ascend/Kernelize/Analysis/DependencyAnalysis.h"
#include "Conversion/Ascend/Kernelize/Candidate/FusionCandidateAnalysis.h"
#include "Conversion/Ascend/Kernelize/Candidate/HorizontalFusionAnalysis.h"
#include "Conversion/Ascend/Kernelize/KernelizeInternalPasses.h"
#include "Conversion/Ascend/Kernelize/Pattern/KernelPattern.h"
#include "Conversion/Ascend/Kernelize/KernelizeTypes.h"
#include "Conversion/Ascend/Kernelize/Analysis/OpRoleClassification.h"
#include "Conversion/Ascend/Kernelize/Analysis/StructuralMarking.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "Conversion/Ascend/Passes.h.inc"

using namespace mlir;
using namespace mlir::ascend::kernelize;

namespace {

struct KernelizeReportEntry {
  std::string opRole;
  std::string kernelPattern;
  unsigned primaryOpCount = 0;
};

bool isFuncOp(Operation *op) {
  return isa<func::FuncOp>(op);
}

void clearOwnedKernelizeAttrs(ModuleOp module) {
  auto clearAttrs = [](Operation *op) {
    op->removeAttr(kOpRolesAttr);
    op->removeAttr(kOpRoleAttr);
    op->removeAttr(kKernelAttr);
    op->removeAttr(kPrimaryAttr);
    op->removeAttr(kKernelizeHandwrittenKindAttr);
    op->removeAttr(kKernelizeTemplateFamiliesAttr);
    op->removeAttr(kKernelGraphEdgesAttr);
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
    StringRef roleName =
        role ? role.getValue()
             : StringRef(::mlir::ascend::kOpRoleUnsupported);
    entries.push_back(KernelizeReportEntry{
        roleName.str(), pattern.kernelName,
        static_cast<unsigned>(pattern.primaryOps.size())});
  }
  return entries;
}

} // namespace

namespace mlir::ascend {

struct AscendKernelizePass
    : public ::impl::AscendKernelizePassBase<AscendKernelizePass> {
  using AscendKernelizePassBase::AscendKernelizePassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    AscendKernelizePassBase::getDependentDialects(registry);
    ::mlir::ascend::kernelize::registerKernelizeExternalModels(registry);
  }

  void runOnOperation() override {
    ::mlir::ascend::debug::DebugOptions options{
        ::mlir::ascend::debug::parseDebugStage(debugStage), dumpReport};
    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Kernelize))
      ::mlir::ascend::debug::emitStageHeader(
          llvm::errs(), ::mlir::ascend::debug::DebugStage::Kernelize,
          getArgument());

    ModuleOp module = getOperation();
    if (module
            .walk([&](Operation *op) {
              if (!isFuncOp(op))
                return WalkResult::advance();

              auto normalized = op->getAttrOfType<BoolAttr>(kNormalizedAttr);
              if (normalized && normalized.getValue())
                return WalkResult::advance();

              op->emitError() << "requires ascend.normalized";
              return WalkResult::interrupt();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      if (failed(markStructuredOps(funcOp)) ||
          failed(fuseGatherElementwise(funcOp))) {
        signalPassFailure();
        return;
      }
    }

    FailureOr<DependencyAnalysisResult> depResult =
        DependencyAnalyzer().analyze(module);
    if (failed(depResult)) {
      signalPassFailure();
      return;
    }

    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Kernelize))
      emitDependencyAnalysisReport(llvm::errs(), *depResult);

    if (failed(StructuralMarker().mark(module, *depResult))) {
      signalPassFailure();
      return;
    }
    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Kernelize))
      emitStructuralMarkingReport(llvm::errs(), *depResult);

    clearOwnedKernelizeAttrs(module);

    FailureOr<OpRoleMap> roleMap = OpRoleClassifier().classify(*depResult);
    if (failed(roleMap)) {
      signalPassFailure();
      return;
    }
    attachRoleAttributes(module, *roleMap);
    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Kernelize))
      emitOpRoleClassificationReport(llvm::errs(), *depResult, *roleMap);

    KernelizeConfig config;
    SmallVector<FusionCandidate> fusionCandidates =
        FusionCandidateAnalyzer().analyze(*depResult, *roleMap, config);
    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Kernelize))
      emitFusionCandidateReport(llvm::errs(), fusionCandidates,
                                depResult->index);

    SmallVector<MergedCandidate> mergedCandidates =
        CandidateMergeAnalyzer().analyze(fusionCandidates, *depResult, config);
    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Kernelize))
      emitCandidateMergeReport(llvm::errs(), mergedCandidates,
                               depResult->index);

    SmallVector<HorizontalFusionCandidate> horizontalCandidates =
        HorizontalFusionAnalyzer().analyze(fusionCandidates, mergedCandidates,
                                           *depResult, config);
    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Kernelize))
      emitHorizontalFusionReport(llvm::errs(), horizontalCandidates);

    KernelPatternGraph graph = KernelPatternBuilder().build(
        fusionCandidates, mergedCandidates, horizontalCandidates, *depResult);
    SmallVector<KernelPattern> patterns =
        KernelPartitioner().partition(graph, *depResult);
    attachKernelPatternAttributes(module, patterns);

    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Kernelize)) {
      emitKernelPatternGraphReport(llvm::errs(), graph, depResult->index);
      emitKernelPartitionReport(llvm::errs(), patterns, depResult->index);
    }

    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Kernelize))
      emitKernelizeReport(buildKernelizeReportEntries(patterns));
  }
};

std::unique_ptr<Pass> createAscendKernelizePass() {
  return std::make_unique<AscendKernelizePass>();
}

} // namespace mlir::ascend
