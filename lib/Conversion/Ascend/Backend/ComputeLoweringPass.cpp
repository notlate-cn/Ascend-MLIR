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
#include "llvm/ADT/SmallSet.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"

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

bool isSupportedFusedElementwiseBody(linalg::GenericOp generic) {
  if (!llvm::all_of(generic.getIteratorTypesArray(), [](utils::IteratorType it) {
        return it == utils::IteratorType::parallel;
      }))
    return false;

  Block *body = generic.getBody();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;

  Operation *lastArithOp = nullptr;
  Value previousResult;
  for (Operation &bodyOp : body->without_terminator()) {
    if (!isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp,
             arith::ConstantOp>(bodyOp))
      return false;

    if (isa<arith::ConstantOp>(bodyOp))
      continue;

    if (bodyOp.getNumOperands() != 2 || bodyOp.getNumResults() != 1)
      return false;

    auto isAvailableOperand = [&](Value value) {
      if (isa<BlockArgument>(value))
        return true;
      if (value.getDefiningOp<arith::ConstantOp>())
        return true;
      return previousResult && value == previousResult;
    };
    if (!llvm::all_of(bodyOp.getOperands(), isAvailableOperand))
      return false;

    previousResult = bodyOp.getResult(0);
    lastArithOp = &bodyOp;
  }

  return lastArithOp && yieldOp.getOperand(0) == lastArithOp->getResult(0);
}

bool isSupportedVectorGatherBody(linalg::GenericOp generic) {
  if (!generic->hasAttr("gather_dim"))
    return false;
  if (!llvm::all_of(generic.getIteratorTypesArray(),
                    [](utils::IteratorType it) {
                      return it == utils::IteratorType::parallel;
                    }))
    return false;

  bool sawLoad = false;
  Value previousResult;
  for (Operation &bodyOp : generic.getBody()->without_terminator()) {
    if (isa<linalg::IndexOp, arith::IndexCastOp>(bodyOp))
      continue;
    if (isa<memref::LoadOp>(bodyOp)) {
      if (sawLoad)
        return false;
      sawLoad = true;
      previousResult = bodyOp.getResult(0);
      continue;
    }
    if (isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp>(bodyOp)) {
      if (!sawLoad)
        return false;
      if (bodyOp.getNumOperands() != 2 || bodyOp.getNumResults() != 1)
        return false;
      auto isAvailableOperand = [&](Value value) {
        if (isa<BlockArgument>(value))
          return true;
        if (value.getDefiningOp<arith::ConstantOp>())
          return true;
        return previousResult && value == previousResult;
      };
      if (!llvm::all_of(bodyOp.getOperands(), isAvailableOperand))
        return false;
      previousResult = bodyOp.getResult(0);
      continue;
    }
    return false;
  }

  auto yieldOp = dyn_cast<linalg::YieldOp>(generic.getBody()->getTerminator());
  return sawLoad && previousResult && yieldOp && yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperand(0) == previousResult;
}

bool hasOnChipOutput(linalg::LinalgOp linalgOp) {
  if (linalgOp.getNumDpsInits() == 0)
    return false;
  MemorySpace memorySpace =
      memorySpaceOf(linalgOp.getDpsInitOperand(0)->get().getType());
  return memorySpace != MemorySpace::GM && memorySpace != MemorySpace::Unknown;
}

bool isRank2SwapPermutation(ArrayRef<int64_t> permutation) {
  return permutation.size() == 2 && permutation[0] == 1 &&
         permutation[1] == 0;
}

bool isSupportedRank2Transpose(linalg::TransposeOp transpose) {
  return hasOnChipOutput(transpose) &&
         isRank2SwapPermutation(transpose.getPermutation());
}

bool isStandaloneTransposeGeneric(linalg::GenericOp generic) {
  if (!hasOnChipOutput(generic) || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1)
    return false;
  auto maps = generic.getIndexingMapsArray();
  if (maps.size() != 2)
    return false;
  AffineMap inMap = maps[0];
  AffineMap outMap = maps[1];
  unsigned rank = generic.getIteratorTypesArray().size();
  if (rank != 2 || !outMap.isIdentity() || inMap.getNumResults() != rank)
    return false;

  llvm::SmallSet<unsigned, 4> seen;
  SmallVector<int64_t, 2> permutation;
  for (unsigned r = 0; r < rank; ++r) {
    auto dimExpr = dyn_cast<AffineDimExpr>(inMap.getResult(r));
    if (!dimExpr)
      return false;
    unsigned position = dimExpr.getPosition();
    if (position >= rank || !seen.insert(position).second)
      return false;
    permutation.push_back(position);
  }
  if (!isRank2SwapPermutation(permutation))
    return false;

  Block &body = *generic.getBody();
  if (body.getOperations().size() != 1)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(&body.front());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return false;
  auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
  return blockArg && blockArg.getArgNumber() == 0;
}

ComputeKind classifyLinalgOp(Operation *op) {
  if (isa<linalg::MatmulOp>(op))
    return ComputeKind::Matmul;
  if (isa<linalg::FillOp>(op))
    return ComputeKind::Fill;
  if (auto transpose = dyn_cast<linalg::TransposeOp>(op))
    if (isSupportedRank2Transpose(transpose))
      return ComputeKind::Transpose;
  if (auto elementwise = dyn_cast<linalg::ElementwiseOp>(op)) {
    auto kind = elementwise.getKind();
    if (kind == linalg::ElementwiseKind::add)
      return ComputeKind::ElementwiseAdd;
    if (kind == linalg::ElementwiseKind::mul)
      return ComputeKind::ElementwiseMul;
    if (kind == linalg::ElementwiseKind::max_signed)
      return ComputeKind::ElementwiseMax;
  }
  if (auto generic = dyn_cast<linalg::GenericOp>(op)) {
    if (isStandaloneTransposeGeneric(generic))
      return ComputeKind::Transpose;
    if (isSupportedVectorGatherBody(generic))
      return ComputeKind::VectorGather;
    if (isSupportedAddReductionBody(generic))
      return ComputeKind::ReductionAdd;
    if (isSupportedFusedElementwiseBody(generic))
      return ComputeKind::FusedElementwise;
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
