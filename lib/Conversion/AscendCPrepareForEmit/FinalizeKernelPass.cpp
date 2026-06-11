#include "Conversion/AscendCPrepareForEmit/AscendCPrepareForEmitPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_ASCENDCFINALIZEKERNELPASS
#define GEN_PASS_DEF_ASCENDCFINALIZEKERNELPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

static void finalizeFunc(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();

  // 1. Set kernel attributes.
  func->setAttr("ascendc.aicore", UnitAttr::get(ctx));
  func->setAttr("ascendc.global", UnitAttr::get(ctx));

  // 2. Strip return operands (kernel returns void).
  func.walk([&](func::ReturnOp ret) {
    if (ret.getNumOperands() > 0) {
      OpBuilder b(ret);
      b.create<func::ReturnOp>(ret.getLoc());
      ret.erase();
    }
  });

  // Update function type to match (no results).
  SmallVector<Type> argTypes;
  for (BlockArgument arg : func.getBody().front().getArguments())
    argTypes.push_back(arg.getType());
  func.setFunctionType(FunctionType::get(ctx, argTypes, /*results=*/{}));

  // 3. Lower affine.min -> arith.minsi.
  SmallVector<affine::AffineMinOp> minOps;
  func.walk([&](affine::AffineMinOp op) { minOps.push_back(op); });
  for (affine::AffineMinOp minOp : minOps) {
    OpBuilder b(minOp);
    Location loc = minOp.getLoc();
    AffineMap map = minOp.getAffineMap();
    ValueRange operands = minOp.getOperands();
    SmallVector<Value> results;
    for (AffineExpr expr : map.getResults())
      results.push_back(mlir::affine::expandAffineExpr(
          b, loc, expr,
          operands.take_front(map.getNumDims()),
          operands.drop_front(map.getNumDims())));
    Value minVal = results[0];
    for (unsigned i = 1; i < results.size(); ++i)
      minVal = b.create<arith::MinSIOp>(loc, minVal, results[i]);
    minOp.replaceAllUsesWith(minVal);
    minOp.erase();
  }
}

struct AscendCFinalizeKernelPass
    : public ::impl::AscendCFinalizeKernelPassBase<AscendCFinalizeKernelPass> {
  using AscendCFinalizeKernelPassBase::AscendCFinalizeKernelPassBase;
  void runOnOperation() override { finalizeFunc(getOperation()); }
};

std::unique_ptr<Pass> createAscendCFinalizeKernelPass() {
  return std::make_unique<AscendCFinalizeKernelPass>();
}

} // namespace mlir::afir
