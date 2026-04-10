//===- AFIRAddAxis.cpp - AFIR shape inference pass -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <functional>
namespace mlir {
namespace afir {

#define GEN_PASS_DEF_AFIRADDAXISPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

void getNeedToAddAxis(Operation *op, llvm::DenseMap<Value, int64_t> &needToAddAxis) {
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (auto opType = dyn_cast<RankedTensorType>(operand.getType())) {
      Value addOperand = operand;
      if (opType.getRank() != resultType.getRank()) {
        while (addOperand.getDefiningOp() && isa<afir::LoadOp>(addOperand.getDefiningOp())) {
          addOperand = dyn_cast<afir::LoadOp>(addOperand.getDefiningOp()).getInput();
        }
        needToAddAxis[addOperand] = std::max(needToAddAxis.lookup_or(addOperand, 0), resultType.getRank());
      }
    }
  }
}

void addAxis(llvm::DenseMap<Value, int64_t> &needToAddAxis, IRRewriter &rewriter) {
  for (auto &[val, rank] : needToAddAxis) {
    auto originType = dyn_cast<RankedTensorType>(val.getType());
    SmallVector<int64_t> newDims(rank - originType.getRank(), 1);
    newDims.insert(newDims.end(), originType.getShape().begin(), originType.getShape().end());
    auto newType = RankedTensorType::get(newDims, originType.getElementType(), originType.getEncoding());
    val.setType(newType);
    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      if (auto func = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
        auto originFuncType = func.getFunctionType();
        SmallVector<Type> inputs(originFuncType.getInputs());
        inputs[blockArg.getArgNumber()] = newType;
        func.setFunctionType(FunctionType::get(func.getContext(), inputs, func.getResultTypes()));
      }
    }
    for (auto user : val.getUsers()) {
      if (auto loadOp = dyn_cast<afir::LoadOp>(user)) {
        loadOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr({rewriter.getMultiDimIdentityMap(rank)}));
        loadOp.getResult().setType(newType);
      }
    }
  }
}

void addBroadcast(Operation *op, IRRewriter &rewriter) {
  rewriter.setInsertionPoint(op);
  SmallVector<Value> newOperands;
  bool needToAdd = false;
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  SmallVector<AffineMap> newIndexingMaps(
      dyn_cast<ArrayAttr>(op->getAttr("indexing_maps")).getAsValueRange<AffineMapAttr>());
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (!isa<RankedTensorType>(operand.getType())) {
      newOperands.push_back(operand);
    } else {
      auto opType = cast<RankedTensorType>(operand.getType());
      Value addOperand = operand;
      if (newIndexingMaps[idx].getNumResults() < resultType.getRank()) {
        newIndexingMaps[idx] = rewriter.getMultiDimIdentityMap(resultType.getRank());
      }
      if (addOperand.getType() != resultType) {
        needToAdd = true;
        auto newMap = rewriter.getMultiDimIdentityMap(resultType.getRank());
        auto originMap = rewriter.getMultiDimIdentityMap(resultType.getRank());
        for (int pos = 0; pos < resultType.getRank(); pos++) {
          if (opType.getShape()[pos] != resultType.getShape()[pos]) {
            originMap = originMap.replace(rewriter.getAffineDimExpr(pos), rewriter.getAffineConstantExpr(0),
                                          resultType.getRank(), 0);
          }
        }
        SmallVector<AffineMap> broadcastMap{originMap, newMap};
        ArrayAttr broadAttr = rewriter.getAffineMapArrayAttr(broadcastMap);
        addOperand = rewriter.create<afir::BroadcastOp>(op->getLoc(), resultType, addOperand, broadAttr, 0,
                                                        rewriter.getDictionaryAttr({}), rewriter.getArrayAttr({}),
                                                        rewriter.getArrayAttr({}));
      }
      newOperands.push_back(addOperand);
    }
  }
  if (needToAdd) {
    IRMapping mapper;
    mapper.map(op->getOperands(), newOperands);
    auto newOp = rewriter.clone(*op, mapper);
    rewriter.replaceOp(op, newOp);
    newIndexingMaps.back() = rewriter.getMultiDimIdentityMap(resultType.getRank());
    newOp->setAttr("indexing_maps", rewriter.getAffineMapArrayAttr(newIndexingMaps));
  }
}

struct AFIRAddAxisPass : public impl::AFIRAddAxisPassBase<AFIRAddAxisPass> {
  void runOnOperation() override {
    SmallVector<Operation *> opList;
    llvm::DenseMap<Value, int64_t> needToAddAxis;
    getOperation().walk([&](Operation *op) {
      if (op->getDialect()->getNamespace() == AFIRDialect::getDialectNamespace() && op->getOperands().size() > 1 &&
          !isa<afir::ConcatOp, afir::MaxOp>(op)) {
        opList.push_back(op);
      }
    });
    IRRewriter rewriter(&getContext());
    do {
      needToAddAxis.clear();
      for (auto op : opList) {
        getNeedToAddAxis(op, needToAddAxis);
      }
      addAxis(needToAddAxis, rewriter);
    } while (needToAddAxis.size());
    for (auto op : opList) {
      addBroadcast(op, rewriter);
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRAddAxisPass() {
  return std::make_unique<AFIRAddAxisPass>();
}

}  // namespace afir
}  // namespace mlir
