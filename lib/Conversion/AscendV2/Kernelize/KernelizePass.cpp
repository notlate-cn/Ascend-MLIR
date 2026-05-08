//===- KernelizePass.cpp - Ascend V2 kernelize pass -----------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Kernelize/KernelizePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "Conversion/AscendV2/Kernelize/DependencyAnalysis.h"
#include "Conversion/AscendV2/Kernelize/FusionCandidateAnalysis.h"
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
#include "llvm/ADT/Twine.h"
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
    llvm::errs() << "  primary_ops = 1\n";
  }
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

    MLIRContext *context = module.getContext();
    SmallVector<KernelizeReportEntry> reportEntries;
    unsigned nextKernelId = 0;
    for (Operation *op : depResult->index.orderedOps) {
      auto role = op->getAttrOfType<StringAttr>(kOpRoleAttr);
      StringRef roleName = role ? role.getValue() : StringRef("unsupported");
      if (roleName == "unsupported")
        continue;

      std::string kernelId =
          (llvm::Twine("kernel_") + llvm::Twine(nextKernelId++)).str();
      op->setAttr(kKernelAttr, StringAttr::get(context, kernelId));
      op->setAttr(kPrimaryAttr, BoolAttr::get(context, true));
      reportEntries.push_back(KernelizeReportEntry{roleName.str(), kernelId});
    }

    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Kernelize))
      emitKernelizeReport(reportEntries);
  }
};

std::unique_ptr<Pass> createAscendKernelizePass() {
  return std::make_unique<AscendKernelizePass>();
}

} // namespace mlir::afir
