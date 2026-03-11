//===- LinalgInferShape.cpp - AFIR shape inference pass -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/AFIR.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <functional>
namespace mlir {

#define GEN_PASS_DEF_LINALGINFERSHAPEPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

struct LinalgInferShapePass : public impl::LinalgInferShapePassBase<LinalgInferShapePass> {
  void runOnOperation() override {
    OpBuilder builder(&getContext());

    getOperation().walk([&](linalg::LinalgOp op) {
      builder.setInsertionPoint(op);
      ReifiedRankedShapedTypeDims a;
      op.reifyResultShapes(builder, a);
      for (auto [num, dim] : llvm::enumerate(a)) {
        SmallVector<int64_t> newDims;
        for (auto d : dim) {
          auto optionalDim = getConstantIntValue(d);
          newDims.push_back(optionalDim.value_or(ShapedType::kDynamic));
        }
        auto originType = dyn_cast<ShapedType>(op.getDpsInitOperand(num)->get().getType());
        auto newType = originType.cloneWith(newDims, originType.getElementType());
        if (newType != originType) {
          op.getDpsInitOperand(num)->get().setType(newType);
          if (op->getNumResults() != 0) {
            op->getResult(num).setType(newType);
          }

          if (auto arg = dyn_cast<BlockArgument>(op.getDpsInitOperand(num)->get())) {
            if (auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())) {
              auto originFuncType = func.getFunctionType();
              SmallVector<Type> inputs(originFuncType.getInputs());
              inputs[arg.getArgNumber()] = newType;
              func.setFunctionType(FunctionType::get(func.getContext(), inputs, func.getResultTypes()));
            }
          }
        }
      }
    });
    getOperation().walk([&](func::FuncOp func) {
      SmallVector<func::ReturnOp> returnOps;
      func.walk([&](func::ReturnOp returnOp) { returnOps.push_back(returnOp); });

      if (returnOps.empty()) return;

      // Get the types of the values being returned
      auto firstReturn = returnOps[0];
      SmallVector<Type> newResultTypes;
      for (Value operand : firstReturn.getOperands()) {
        newResultTypes.push_back(operand.getType());
      }

      // Update function type
      auto funcType = func.getFunctionType();
      auto newFuncType = FunctionType::get(func.getContext(), funcType.getInputs(), newResultTypes);
      func.setFunctionType(newFuncType);
    });
  }
};
}  // namespace

std::unique_ptr<Pass> createLinalgInferShapePass() {
  return std::make_unique<LinalgInferShapePass>();
}

}  // namespace mlir
