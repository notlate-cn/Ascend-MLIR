//===- RealizePass.cpp - Ascend V2 realize pass --------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Realize/RealizePass.h"

#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "Conversion/AscendV2/Realize/RealizeReport.h"
#include "Conversion/AscendV2/Realize/RealizeTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

#define GEN_PASS_DECL_ASCENDREALIZEPASS
#define GEN_PASS_DEF_ASCENDREALIZEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::afir::ascend::v2::realize;

namespace {

static FailureOr<SmallVector<RealizePlanBundle, 4>>
buildMVPRealizePlans(ModuleOp module, bool &emittedError) {
  DenseMap<StringRef, unsigned> scheduledOpsByKernel;
  DenseMap<StringRef, std::string> decisionByKernel;
  DenseMap<StringRef, std::string> skeletonByKernel;

  WalkResult walkResult = module.walk([&](Operation *op) {
    auto kernelAttr = op->getAttrOfType<StringAttr>(kKernelAttr);
    auto decisionAttr = op->getAttrOfType<StringAttr>(kScheduleDecisionIdAttr);
    auto skeletonAttr = op->getAttrOfType<StringAttr>(kStructuredLoweringAttr);
    if (!kernelAttr && !decisionAttr && !skeletonAttr)
      return WalkResult::advance();
    if (!kernelAttr || !decisionAttr || !skeletonAttr) {
      op->emitError()
          << "ascend-realize requires complete scheduled structured "
             "lowering attributes";
      emittedError = true;
      return WalkResult::interrupt();
    }
    StringRef kernel = kernelAttr.getValue();
    auto decisionIt = decisionByKernel.find(kernel);
    if (decisionIt != decisionByKernel.end() &&
        decisionIt->second != decisionAttr.getValue().str()) {
      op->emitError()
          << "ascend-realize requires consistent schedule attributes per "
             "kernel";
      emittedError = true;
      return WalkResult::interrupt();
    }
    auto skeletonIt = skeletonByKernel.find(kernel);
    if (skeletonIt != skeletonByKernel.end() &&
        skeletonIt->second != skeletonAttr.getValue().str()) {
      op->emitError()
          << "ascend-realize requires consistent schedule attributes per "
             "kernel";
      emittedError = true;
      return WalkResult::interrupt();
    }
    ++scheduledOpsByKernel[kernel];
    decisionByKernel[kernel] = decisionAttr.getValue().str();
    skeletonByKernel[kernel] = skeletonAttr.getValue().str();
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return failure();

  if (scheduledOpsByKernel.empty())
    return failure();

  SmallVector<StringRef> kernels;
  for (const auto &entry : scheduledOpsByKernel)
    kernels.push_back(entry.first);
  llvm::sort(kernels);

  SmallVector<RealizePlanBundle, 4> bundles;
  for (StringRef kernel : kernels) {
    RealizePlanBundle bundle;
    bundle.kernel.kernelId = kernel.str();
    bundle.kernel.decisionId = decisionByKernel[kernel];
    bundle.kernel.structuredLowering = skeletonByKernel[kernel];
    bundle.kernel.scheduledOps = scheduledOpsByKernel[kernel];
    bundle.bufferizedIR.kernelId = bundle.kernel.kernelId;
    bundle.placement.kernelId = bundle.kernel.kernelId;
    bundle.staticMemory.kernelId = bundle.kernel.kernelId;
    bundle.movement.kernelId = bundle.kernel.kernelId;
    bundle.realization.kernelId = bundle.kernel.kernelId;
    bundle.realization.frozen = true;
    bundles.push_back(std::move(bundle));
  }
  return bundles;
}

} // namespace

namespace mlir::afir {

struct AscendRealizePass
    : public ::impl::AscendRealizePassBase<AscendRealizePass> {
  using AscendRealizePassBase::AscendRealizePassBase;

  void runOnOperation() override {
    ::mlir::ascend::v2::DebugOptions options{
        ::mlir::ascend::v2::parseDebugStage(debugStage), dumpReport};
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Realize))
      ::mlir::ascend::v2::emitStageHeader(
          llvm::errs(), ::mlir::ascend::v2::DebugStage::Realize,
          getArgument());

    bool emittedError = false;
    FailureOr<SmallVector<RealizePlanBundle, 4>> bundles =
        buildMVPRealizePlans(getOperation(), emittedError);
    if (failed(bundles)) {
      if (!emittedError)
        getOperation()->emitError()
            << "ascend-realize requires scheduled structured lowering "
               "attributes";
      signalPassFailure();
      return;
    }

    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Realize))
      printRealizeReport(*bundles, llvm::errs());
  }
};

std::unique_ptr<Pass> createAscendRealizePass() {
  return std::make_unique<AscendRealizePass>();
}

} // namespace mlir::afir
