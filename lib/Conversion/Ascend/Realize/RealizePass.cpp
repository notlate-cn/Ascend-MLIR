//===- RealizePass.cpp - Ascend realize pass --------------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/RealizePass.h"

#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Debug/DebugOptions.h"
#include "BufferizationDriver.h"
#include "MemoryRealizationDriver.h"
#include "TilingRealizationDriver.h"
#include "MovementPlanner.h"
#include "PlacementPlanner.h"
#include "RealizeReport.h"
#include "RealizeTypes.h"
#include "StaticMemoryPlanner.h"
#include "Target/Ascend/CannTargetProfileLoader.h"
#include "Target/Ascend/TargetMemoryModel.h"
#include "Target/Ascend/TargetProfile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>

#define GEN_PASS_DECL_ASCENDREALIZEPASS
#define GEN_PASS_DEF_ASCENDREALIZEPASS
#include "Conversion/Ascend/Passes.h.inc"

using namespace mlir;
using namespace mlir::ascend::realize;

namespace {

constexpr llvm::StringLiteral kPlanOnlyMaterializationMode = "plan-only";
constexpr llvm::StringLiteral kOneShotBufferizeMaterializationMode =
    "one-shot-bufferize";
constexpr llvm::StringLiteral kMemorySpaceAnnotateMaterializationMode =
    "memory-space-annotate";
constexpr llvm::StringLiteral kTiledLinalgMaterializationMode = "tiled-linalg";
constexpr llvm::StringLiteral kFullRealizeMaterializationMode = "full-realize";
constexpr llvm::StringLiteral kGmDefaultPlacementMode = "gm-default";
constexpr llvm::StringLiteral kTargetAwarePlacementMode = "target-aware";

static bool isSupportedMaterializationMode(StringRef mode) {
  return mode == kPlanOnlyMaterializationMode ||
         mode == kOneShotBufferizeMaterializationMode ||
         mode == kMemorySpaceAnnotateMaterializationMode ||
         mode == kTiledLinalgMaterializationMode ||
         mode == kFullRealizeMaterializationMode;
}

static bool isSupportedPlacementMode(StringRef mode) {
  return mode == kGmDefaultPlacementMode || mode == kTargetAwarePlacementMode;
}

static void appendWorkspaceExprTerm(std::string &expr, StringRef term) {
  if (term.empty() || term == "0")
    return;
  if (!expr.empty())
    expr += " + ";
  expr += term.str();
}

static LogicalResult
stampWorkspaceSizeAttrs(ModuleOp module,
                        ArrayRef<RealizePlanBundle> bundles) {
  llvm::StringMap<const StaticMemoryPlan *> staticMemoryByKernel;
  for (const RealizePlanBundle &bundle : bundles) {
    if (!bundle.kernel.kernelId.empty())
      staticMemoryByKernel[bundle.kernel.kernelId] = &bundle.staticMemory;
  }

  Builder builder(module.getContext());
  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    llvm::StringSet<> seenKernels;
    uint64_t workspaceByteCount = 0;
    std::string workspaceSizeExpr;
    bool hasScheduledKernel = false;
    bool hasKnownWorkspace = false;
    bool hasDynamicWorkspaceExpr = false;
    bool workspaceByteCountOverflow = false;

    funcOp.walk([&](Operation *op) {
      auto kernelAttr = op->getAttrOfType<StringAttr>(kKernelAttr);
      if (!kernelAttr)
        return;
      hasScheduledKernel = true;
      StringRef kernelId = kernelAttr.getValue();
      if (!seenKernels.insert(kernelId).second)
        return;
      const StaticMemoryPlan *staticMemory =
          staticMemoryByKernel.lookup(kernelId);
      if (!staticMemory)
        return;
      if (staticMemory->workspaceSizeExprKnown) {
        hasKnownWorkspace = true;
        hasDynamicWorkspaceExpr = true;
        appendWorkspaceExprTerm(workspaceSizeExpr,
                                staticMemory->workspaceSizeExpr);
        return;
      }
      if (!staticMemory->peakUsageBytesKnown)
        return;
      hasKnownWorkspace = true;
      uint64_t maxSigned =
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
      if (staticMemory->workspaceByteCount > maxSigned ||
          workspaceByteCount > maxSigned - staticMemory->workspaceByteCount) {
        workspaceByteCountOverflow = true;
        return;
      }
      workspaceByteCount += staticMemory->workspaceByteCount;
    });

    if (!hasScheduledKernel)
      continue;

    if (workspaceByteCountOverflow)
      return funcOp.emitError()
             << "computed "
             << ::mlir::ascend::kCannWorkspaceSizeBytesAttr
             << " exceeds signed 64-bit range";

    funcOp->removeAttr(::mlir::ascend::kCannWorkspaceSizeBytesAttr);
    funcOp->removeAttr(::mlir::ascend::kCannWorkspaceSizeExprAttr);
    if (!hasKnownWorkspace)
      continue;

    if (hasDynamicWorkspaceExpr) {
      if (workspaceByteCount != 0)
        appendWorkspaceExprTerm(workspaceSizeExpr,
                                std::to_string(workspaceByteCount));
      if (!workspaceSizeExpr.empty())
        funcOp->setAttr(::mlir::ascend::kCannWorkspaceSizeExprAttr,
                        builder.getStringAttr(workspaceSizeExpr));
      continue;
    }

    if (workspaceByteCount != 0)
      funcOp->setAttr(::mlir::ascend::kCannWorkspaceSizeBytesAttr,
                      builder.getI64IntegerAttr(
                          static_cast<int64_t>(workspaceByteCount)));
  }

  return success();
}

static FailureOr<SmallVector<RealizePlanBundle, 4>>
buildRealizePlanBundles(ModuleOp module,
                        const ::mlir::ascend::TargetMemoryModel *memoryModel,
                        bool &emittedError) {
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
        memoryModel ? placementPlanner.build(bundle.bufferizedIR, *memoryModel)
                    : placementPlanner.build(bundle.bufferizedIR);
    if (failed(placement)) {
      module.emitError("ascend-realize failed to build placement plan");
      emittedError = true;
      return failure();
    }
    bundle.placement = std::move(*placement);
    FailureOr<StaticMemoryPlan> staticMemory =
        memoryModel
            ? staticMemoryPlanner.build(bundle.placement, bundle.bufferizedIR,
                                        *memoryModel)
            : staticMemoryPlanner.build(bundle.placement,
                                        bundle.bufferizedIR);
    if (failed(staticMemory)) {
      module.emitError("ascend-realize failed to build static memory plan");
      emittedError = true;
      return failure();
    }
    bundle.staticMemory = std::move(*staticMemory);
    FailureOr<MovementPlan> movement =
        memoryModel
            ? movementPlanner.build(bundle.placement, bundle.staticMemory,
                                    *memoryModel)
            : movementPlanner.build(bundle.placement, bundle.staticMemory);
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

namespace mlir::ascend {

struct AscendRealizePass
    : public ::impl::AscendRealizePassBase<AscendRealizePass> {
  using AscendRealizePassBase::AscendRealizePassBase;

  void runOnOperation() override {
    if (!isSupportedPlacementMode(placementMode)) {
      getOperation()->emitError()
          << "unsupported ascend-realize placement-mode \"" << placementMode
          << "\"";
      signalPassFailure();
      return;
    }

    if (!isSupportedMaterializationMode(materializationMode)) {
      getOperation()->emitError()
          << "unsupported ascend-realize materialization-mode \""
          << materializationMode << "\"";
      signalPassFailure();
      return;
    }

    ::mlir::ascend::debug::DebugOptions options{
        ::mlir::ascend::debug::parseDebugStage(debugStage), dumpReport,
        StringRef(debugDumpDir).str()};
    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Realize))
      ::mlir::ascend::debug::emitStageHeader(
          llvm::errs(), ::mlir::ascend::debug::DebugStage::Realize,
          getArgument());

    std::optional<::mlir::ascend::TargetMemoryModel> targetMemoryModel;
    if (placementMode == kTargetAwarePlacementMode) {
      FailureOr<::mlir::ascend::TargetProfile> profile =
          ::mlir::ascend::CannTargetProfileLoader::load(cannRoot, soc);
      if (failed(profile)) {
        getOperation()->emitError()
            << "ascend-realize target-aware placement failed to load target "
               "profile";
        signalPassFailure();
        return;
      }

      FailureOr<::mlir::ascend::TargetMemoryModel> builtMemoryModel =
          ::mlir::ascend::TargetMemoryModelBuilder().build(*profile,
                                                           llvm::errs());
      if (failed(builtMemoryModel)) {
        getOperation()->emitError()
            << "ascend-realize target-aware placement failed to build target "
               "memory model";
        signalPassFailure();
        return;
      }
      targetMemoryModel = std::move(*builtMemoryModel);
    }

    bool emittedError = false;
    FailureOr<SmallVector<RealizePlanBundle, 4>> bundles =
        buildRealizePlanBundles(
            getOperation(),
            targetMemoryModel ? &*targetMemoryModel : nullptr, emittedError);
    if (failed(bundles)) {
      if (!emittedError)
        getOperation()->emitError()
            << "ascend-realize requires scheduled structured lowering "
               "attributes";
      signalPassFailure();
      return;
    }

    if (failed(stampWorkspaceSizeAttrs(getOperation(), *bundles))) {
      signalPassFailure();
      return;
    }
    if (failed(::mlir::ascend::debug::dumpCheckpoint(
            getOperation(), options, ::mlir::ascend::debug::DebugStage::Realize,
            "041-realize-planned"))) {
      signalPassFailure();
      return;
    }

    if (materializationMode == kTiledLinalgMaterializationMode ||
        materializationMode == kFullRealizeMaterializationMode) {
      TilingRealizationDriver tilingDriver;
      if (failed(tilingDriver.tileModule(getOperation()))) {
        getOperation()->emitError()
            << "ascend-realize tiled-linalg: tiling failed";
        signalPassFailure();
        return;
      }
    }

    if (materializationMode == kOneShotBufferizeMaterializationMode ||
        materializationMode == kMemorySpaceAnnotateMaterializationMode ||
        materializationMode == kFullRealizeMaterializationMode) {
      BufferizationDriver bufferizationDriver;
      if (failed(bufferizationDriver.runOneShotBufferize(getOperation()))) {
        getOperation()->emitError()
            << "ascend-realize failed to bufferize";
        signalPassFailure();
        return;
      }
      if (failed(::mlir::ascend::debug::dumpCheckpoint(
              getOperation(), options,
              ::mlir::ascend::debug::DebugStage::Realize,
              "042-realize-bufferized"))) {
        signalPassFailure();
        return;
      }
    }

    if (materializationMode == kMemorySpaceAnnotateMaterializationMode ||
        materializationMode == kFullRealizeMaterializationMode) {
      MemoryRealizationDriver memoryRealizationDriver;
      if (failed(memoryRealizationDriver.materialize(
              getOperation(), *bundles,
              MemoryRealizationMode::MemorySpaceAnnotate))) {
        getOperation()->emitError()
            << "ascend-realize failed to materialize memory realization plan";
        signalPassFailure();
        return;
      }
      if (failed(::mlir::ascend::debug::dumpCheckpoint(
              getOperation(), options,
              ::mlir::ascend::debug::DebugStage::Realize,
              "043-realize-memory-space-annotated"))) {
        signalPassFailure();
        return;
      }
    }

    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Realize))
      printRealizeReport(*bundles, llvm::errs());
  }
};

std::unique_ptr<Pass> createAscendRealizePass() {
  return std::make_unique<AscendRealizePass>();
}

} // namespace mlir::ascend
