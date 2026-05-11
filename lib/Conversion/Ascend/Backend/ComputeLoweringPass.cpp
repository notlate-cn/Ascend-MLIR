//===- ComputeLoweringPass.cpp - Ascend compute lowering wrapper ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/ComputeLoweringPass.h"
#include "Conversion/Ascend/Backend/BackendSupportMatrix.h"
#include "Conversion/LinalgToAscendC/LinalgToAscendCUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

#include "ascir/Dialect/Asc/IR/Asc.h"

#define GEN_PASS_DEF_ASCENDCOMPUTELOWERPASS
#include "Conversion/Passes.h.inc"

namespace mlir::afir {
namespace {

namespace backend = ascend::backend;

using backend::AscendBackendSupportMatrix;
using backend::ComputeKind;
using backend::MemorySpace;

MemorySpace memorySpaceOf(Type type) {
  return backend::parseMemorySpace(getMemorySpace(type));
}

bool isSupportedAddReductionBody(linalg::GenericOp generic) {
  if (!llvm::is_contained(generic.getIteratorTypesArray(),
                          utils::IteratorType::reduction))
    return false;

  Block *body = generic.getBody();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;

  if (!yieldOp.getOperand(0).getDefiningOp<arith::AddFOp>())
    return false;

  for (Operation &bodyOp : body->without_terminator()) {
    if (!isa<arith::AddFOp, arith::ConstantOp>(bodyOp))
      return false;
  }

  return true;
}

ComputeKind classifyLinalgOp(Operation *op) {
  if (isa<linalg::MatmulOp>(op))
    return ComputeKind::Matmul;
  if (isa<linalg::FillOp>(op))
    return ComputeKind::Fill;
  if (auto elementwise = dyn_cast<linalg::ElementwiseOp>(op)) {
    auto kind = elementwise.getKind();
    if (kind == linalg::ElementwiseKind::add)
      return ComputeKind::ElementwiseAdd;
    if (kind == linalg::ElementwiseKind::max_signed)
      return ComputeKind::ElementwiseMax;
  }
  if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
    if (isSupportedAddReductionBody(generic))
      return ComputeKind::ReductionAdd;
  }
  return ComputeKind::Unknown;
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
    ComputeKind kind = classifyLinalgOp(op);
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
