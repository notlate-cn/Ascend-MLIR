//===- ElementwiseOpsDemo.cpp - Demo for creating AFIR elementwise ops -*- C++ -*-===//
//
// This file demonstrates how to create AFIR elementwise operations (add, sub, mul, div)
// using MLIR C++ API with AFIRTensor types.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "Dialect/AFIR/AFIRDialect.h"
#include "Dialect/AFIR/AFIROps.h"

using namespace mlir;
using namespace mlir::afir;

AffineMap createCustomIndexedIdentityMap(
    MLIRContext *context,
    ArrayRef<unsigned> usedDimIndices
) {
  unsigned numUsedDims = usedDimIndices.size();
  unsigned maxDimIndex = *std::max_element(usedDimIndices.begin(), usedDimIndices.end());
  
  SmallVector<AffineExpr, 4> resultExprs;
  for (unsigned i = 0; i < numUsedDims; ++i) {
    resultExprs.push_back(getAffineDimExpr(usedDimIndices[i], context));
  }
  
  return AffineMap::get(maxDimIndex + 1, 0, resultExprs, context);
}

int main() {
  MLIRContext context;

  context.loadDialect<AFIRDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::linalg::LinalgDialect>();

  OpBuilder builder(&context);

  OwningOpRef<ModuleOp> module(ModuleOp::create(builder.getUnknownLoc()));

  {
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(module->getBody());

    auto posAttr0 = PositionInfoAttr::get(&context, 1, Position::GM, 0, 1, false);
    auto posAttr1 = PositionInfoAttr::get(&context, 2, Position::VEC_IN, 2, 1, true);
    auto posAttrResult = PositionInfoAttr::get(&context, 3, Position::VEC_OUT, 2, 1, true);

    auto tensorType0 = AFIRTensorType::get(&context, {4, 4}, builder.getF32Type(), 1, posAttr0);
    auto tensorType1 = AFIRTensorType::get(&context, {4, 4}, builder.getF32Type(), 2, posAttr1);
    auto tensorTypeResult = AFIRTensorType::get(&context, {4, 4}, builder.getF32Type(), 3, posAttrResult);

    AffineMap map_d4_d7 = createCustomIndexedIdentityMap(&context, {4, 7});
    auto indexingMapsAttr = builder.getAffineMapArrayAttr({map_d4_d7, map_d4_d7, map_d4_d7});

    auto funcType = builder.getFunctionType({tensorType0, tensorType1}, tensorTypeResult);

    auto funcOp = builder.create<func::FuncOp>(
      builder.getUnknownLoc(),
      "elementwise_operations",
      funcType
    );

    auto *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto arg0 = entryBlock->getArgument(0);
    auto arg1 = entryBlock->getArgument(1);

    SmallVector<int64_t> transposeVec = {1, 0};
    auto linalgAdd = builder.create<linalg::TransposeOp>(
      builder.getUnknownLoc(),
      arg0, arg1,
      transposeVec
    );

    auto addOp = builder.create<AddOp>(
      builder.getUnknownLoc(),
      tensorTypeResult,
      arg0,
      arg1,
      indexingMapsAttr
    );
    addOp->setAttr("compute_hint", builder.getI64IntegerAttr(1));
    addOp->setAttr("enable_optimization", builder.getBoolAttr(true));

    auto subOp = builder.create<SubOp>(
      builder.getUnknownLoc(),
      tensorTypeResult,
      arg0,
      arg1,
      indexingMapsAttr
    );
    subOp->setAttr("compute_hint", builder.getI64IntegerAttr(2));

    auto mulOp = builder.create<MulOp>(
      builder.getUnknownLoc(),
      tensorTypeResult,
      arg0,
      arg1,
      indexingMapsAttr
    );
    mulOp->setAttr("compute_hint", builder.getI64IntegerAttr(3));

    auto divOp = builder.create<DivOp>(
      builder.getUnknownLoc(),
      tensorTypeResult,
      arg0,
      arg1,
      indexingMapsAttr
    );
    divOp->setAttr("compute_hint", builder.getI64IntegerAttr(4));

    builder.create<func::ReturnOp>(builder.getUnknownLoc(), divOp.getResult());
  }

  if (failed(verify(*module))) {
    module->dump();
    llvm::errs() << "Verification failed\n";
    return 1;
  }

  llvm::outs() << "=== Generated MLIR Module ===\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  return 0;
}
