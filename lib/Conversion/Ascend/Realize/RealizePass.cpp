//===- RealizePass.cpp - Ascend realize pass --------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/RealizePass.h"

#include "Conversion/Ascend/Debug/DebugOptions.h"
#include "Conversion/Ascend/Realize/BufferizationDriver.h"
#include "Conversion/Ascend/Realize/MemoryRealizationDriver.h"
#include "Conversion/Ascend/Realize/MovementPlanner.h"
#include "Conversion/Ascend/Realize/PlacementPlanner.h"
#include "Conversion/Ascend/Realize/RealizeReport.h"
#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "Conversion/Ascend/Realize/StaticMemoryPlanner.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

#define GEN_PASS_DECL_ASCENDREALIZEPASS
#define GEN_PASS_DEF_ASCENDREALIZEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::afir::ascend::realize;

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

  if (scheduledOpsByKernel.empty()) {
    module.emitError("ascend-realize requires at least one op with scheduled "
                     "structured lowering attributes");
    emittedError = true;
    return failure();
  }

  BufferizationDriver bufferizationDriver;
  FailureOr<SmallVector<BufferizedKernelIR, 4>> bufferized =
      bufferizationDriver.collectTensorFacts(module);
  if (failed(bufferized))
    return failure();

  llvm::StringMap<BufferizedKernelIR> bufferizedByKernel;
  for (BufferizedKernelIR &ir : *bufferized) {
    std::string kernelId = ir.kernelId;
    bufferizedByKernel[kernelId] = std::move(ir);
  }

  SmallVector<StringRef> kernels;
  for (const auto &entry : scheduledOpsByKernel)
    kernels.push_back(entry.first);
  llvm::sort(kernels);

  SmallVector<RealizePlanBundle, 4> bundles;
  PlacementPlanner placementPlanner;
  StaticMemoryPlanner staticMemoryPlanner;
  MovementPlanner movementPlanner;
  MemoryRealizationDriver memoryRealizationDriver;
  for (StringRef kernel : kernels) {
    RealizePlanBundle bundle;
    bundle.kernel.kernelId = kernel.str();
    bundle.kernel.decisionId = decisionByKernel[kernel];
    bundle.kernel.structuredLowering = skeletonByKernel[kernel];
    bundle.kernel.scheduledOps = scheduledOpsByKernel[kernel];
    auto bufferizedIt = bufferizedByKernel.find(bundle.kernel.kernelId);
    if (bufferizedIt != bufferizedByKernel.end())
      bundle.bufferizedIR = std::move(bufferizedIt->second);
    else
      bundle.bufferizedIR.kernelId = bundle.kernel.kernelId;
    FailureOr<PlacementPlan> placement =
        placementPlanner.build(bundle.bufferizedIR);
    if (failed(placement)) {
      module.emitError("ascend-realize failed to build placement plan");
      emittedError = true;
      return failure();
    }
    bundle.placement = std::move(*placement);
    FailureOr<StaticMemoryPlan> staticMemory =
        staticMemoryPlanner.build(bundle.placement);
    if (failed(staticMemory)) {
      module.emitError("ascend-realize failed to build static memory plan");
      emittedError = true;
      return failure();
    }
    bundle.staticMemory = std::move(*staticMemory);
    FailureOr<MovementPlan> movement =
        movementPlanner.build(bundle.placement, bundle.staticMemory);
    if (failed(movement)) {
      module.emitError("ascend-realize failed to build movement plan");
      emittedError = true;
      return failure();
    }
    bundle.movement = std::move(*movement);
    FailureOr<MemoryRealizationPlan> realization =
        memoryRealizationDriver.materialize(bundle.placement,
                                            bundle.staticMemory,
                                            bundle.movement);
    if (failed(realization)) {
      module.emitError(
          "ascend-realize failed to materialize memory realization plan");
      emittedError = true;
      return failure();
    }
    bundle.realization = std::move(*realization);
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
    ::mlir::afir::ascend::debug::DebugOptions options{
        ::mlir::afir::ascend::debug::parseDebugStage(debugStage), dumpReport};
    if (::mlir::afir::ascend::debug::shouldDump(
            options, ::mlir::afir::ascend::debug::DebugStage::Realize))
      ::mlir::afir::ascend::debug::emitStageHeader(
          llvm::errs(), ::mlir::afir::ascend::debug::DebugStage::Realize,
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

    if (::mlir::afir::ascend::debug::shouldDump(
            options, ::mlir::afir::ascend::debug::DebugStage::Realize))
      printRealizeReport(*bundles, llvm::errs());
  }
};

std::unique_ptr<Pass> createAscendRealizePass() {
  return std::make_unique<AscendRealizePass>();
}

} // namespace mlir::afir
