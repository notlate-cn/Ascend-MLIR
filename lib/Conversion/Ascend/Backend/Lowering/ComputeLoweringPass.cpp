//===- ComputeLoweringPass.cpp - Ascend compute lowering wrapper ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/Lowering/ComputeLoweringPass.h"
#include "Conversion/Ascend/Backend/Lowering/BackendSupportMatrix.h"
#include "Conversion/Ascend/Backend/Lowering/LinalgBodyClassifier.h"
#include "Conversion/Ascend/Backend/Lowering/LinalgToAscendCUtils.h"
#include "../Codegen/CodegenPasses.h"
#include "../../Kernelize/KernelizeInternalPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

#define GEN_PASS_DEF_ASCENDCOMPUTELOWERPASS
#include "Conversion/Ascend/Passes.h.inc"

namespace mlir::afir {
namespace {

namespace backend = ascend::backend;

using backend::AscendBackendSupportMatrix;
using backend::ComputeKind;
using backend::MemorySpace;

MemorySpace memorySpaceOf(Type type) {
  return backend::parseMemorySpace(getMemorySpace(type));
}

LogicalResult verifySupportedInputs(func::FuncOp funcOp,
                                    const AscendBackendSupportMatrix &matrix) {
  WalkResult result = funcOp.walk([&](memref::CopyOp copyOp) {
    MemorySpace src = memorySpaceOf(copyOp.getSource().getType());
    MemorySpace dst = memorySpaceOf(copyOp.getTarget().getType());
    if (matrix.isSupportedMovementPath(src, dst))
      return WalkResult::advance();
    backend::UnsupportedReason reason = matrix.explainMovementPath(src, dst);
    copyOp.emitError(reason.detail);
    return WalkResult::interrupt();
  });
  if (result.wasInterrupted())
    return failure();

  result = funcOp.walk([&](Operation *op) {
    if (!isa<linalg::LinalgOp>(op))
      return WalkResult::advance();
    ComputeKind kind = backend::classifyLinalgComputeKind(op, matrix);
    if (matrix.isSupportedComputeKind(kind))
      return WalkResult::advance();
    backend::UnsupportedReason reason = matrix.explainComputeKind(kind);
    op->emitError(reason.detail);
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult verifyNoResidualLowerableOps(func::FuncOp funcOp) {
  WalkResult result = funcOp.walk([&](Operation *op) {
    if (isa<memref::CopyOp>(op) || isa<linalg::LinalgOp>(op)) {
      op->emitError("ascend-compute-lower left a lowerable operation behind");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

struct AscendComputeLowerPass
    : public ::impl::AscendComputeLowerPassBase<AscendComputeLowerPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    AscendBackendSupportMatrix matrix;

    if (failed(annotateAscendKernelKind(funcOp)) ||
        failed(annotateMixMatmulSemantics(funcOp))) {
      signalPassFailure();
      return;
    }
    if (failed(verifySupportedInputs(funcOp, matrix))) {
      signalPassFailure();
      return;
    }
    if (failed(lowerLinalgToAscendC(funcOp))) {
      signalPassFailure();
      return;
    }
    if (failed(verifyNoResidualLowerableOps(funcOp))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAscendComputeLowerPass() {
  return std::make_unique<AscendComputeLowerPass>();
}

} // namespace mlir::afir
