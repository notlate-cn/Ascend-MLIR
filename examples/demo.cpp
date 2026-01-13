//===- demo.cpp - Demo for creating AFIR elementwise ops -*- C++ -*-===//
//
// This file demonstrates how to create AFIR elementwise operations (add, sub, mul, div)
// using MLIR C++ API with standard tensor types.
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
#include "Dialect/AFIR/AFIR.h"

using namespace mlir;
using namespace mlir::afir;

AffineMap createCustomIndexedIdentityMap(MLIRContext *context, ArrayRef<unsigned> usedDimIndices) {
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

    auto tensorType = RankedTensorType::get({4, 4}, builder.getF32Type());

    AffineMap map = AffineMap::getMultiDimIdentityMap(2, &context);
    auto indexingMapsAttr = builder.getAffineMapArrayAttr({map, map, map});

    auto funcType = builder.getFunctionType({tensorType, tensorType}, tensorType);

    auto funcOp = builder.create<func::FuncOp>(builder.getUnknownLoc(), "elementwise_operations", funcType);

    auto *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto arg0 = entryBlock->getArgument(0);
    auto arg1 = entryBlock->getArgument(1);

    auto outputsAttr =
        AscTensorGroupsAttr::get(&context, {1, 2},                                              // vectorized_axis
                                 {builder.getI64IntegerAttr(4), builder.getI64IntegerAttr(8)},  // vectorized_strides
                                 42,                                                            // tensor_id
                                 10,                                                            // reuse_id
                                 PositionConfigAttr::get(&context, Position::VECTOR_IN, 2, true),
                                 5  // position_id
        );

    auto outputsAttrArray = builder.getArrayAttr(outputsAttr);

    llvm::outs() << "=== Debug outputsAttr ===\n";
    llvm::outs() << "vectorized_axis: ";
    for (auto val : outputsAttr.getVectorizedAxis()) {
      llvm::outs() << val << " ";
    }
    llvm::outs() << "\n";
    llvm::outs() << "tensor_id: " << outputsAttr.getTensorId() << "\n";
    llvm::outs() << "reuse_id: " << outputsAttr.getReuseId() << "\n";
    llvm::outs() << "position_id: " << outputsAttr.getPositionId() << "\n";
    llvm::outs() << "depth: " << outputsAttr.getPositionConfig().getDepth() << "\n";
    llvm::outs() << "is_double_buffer: " << outputsAttr.getPositionConfig().getIsDoubleBuffer() << "\n";
    llvm::outs() << "========================\n\n";

    auto irAttrDef =
        builder.getDictionaryAttr({builder.getNamedAttr("compute_hint", builder.getI64IntegerAttr(1)),
                                   builder.getNamedAttr("custom_option_1", builder.getI32IntegerAttr(100)),
                                   builder.getNamedAttr("custom_option_2", builder.getStringAttr("example_value"))});

    auto tmpBufDesc1 = TmpBufDescAttr::get(&context, builder.getStringAttr("(d0 * 128)"), 2);
    auto tmpBufDesc2 = TmpBufDescAttr::get(&context, builder.getStringAttr("(d0 * 64 + d1 * 8)"), 3);
    auto tmpBuffersAttr = builder.getArrayAttr({tmpBufDesc1, tmpBufDesc2});

    llvm::outs() << "=== Debug AxisAttr ===\n";
    auto axis1 = AxisAttr::get(&context, 0, builder.getStringAttr("z0"), AxisType::Original, false,
                               builder.getStringAttr("20"), builder.getStringAttr("1"), {}, -1);
    auto axis2 = AxisAttr::get(&context, 1, builder.getStringAttr("z1"), AxisType::Original, false,
                               builder.getStringAttr("31"), builder.getStringAttr("1"), {}, -1);

    auto axis1Attr = builder.getArrayAttr({axis1});
    auto axis2Attr = builder.getArrayAttr({axis2});
    auto graphAttr = AscGraphAttrGroupsAttr::get(&context, -1, {axis1, axis2}, AscGraphType::Compute, {});

    llvm::outs() << "AxisAttr axis1: " << axis1 << "\n";
    llvm::outs() << "AxisAttr axis2: " << axis2 << "\n";
    llvm::outs() << "AscGraphAttrGroupsAttr: " << graphAttr << "\n";
    llvm::outs() << "========================\n\n";

    auto addOp = builder.create<AddOp>(builder.getUnknownLoc(), tensorType, arg0, arg1, indexingMapsAttr,
                                       builder.getI32IntegerAttr(1), irAttrDef, tmpBuffersAttr, outputsAttrArray);

    auto subOpIrAttrDef =
        builder.getDictionaryAttr({builder.getNamedAttr("compute_hint", builder.getI64IntegerAttr(2)),
                                   builder.getNamedAttr("custom_option_1", builder.getI32IntegerAttr(200))});

    auto subOp = builder.create<SubOp>(builder.getUnknownLoc(), tensorType, arg0, arg1, indexingMapsAttr, nullptr,
                                       subOpIrAttrDef, nullptr, outputsAttrArray);

    auto mulOpIrAttrDef =
        builder.getDictionaryAttr({builder.getNamedAttr("compute_hint", builder.getI64IntegerAttr(3)),
                                   builder.getNamedAttr("custom_option_1", builder.getI32IntegerAttr(300))});

    auto mulOp = builder.create<MulOp>(builder.getUnknownLoc(), tensorType, arg0, arg1, indexingMapsAttr, nullptr,
                                       mulOpIrAttrDef, nullptr, outputsAttrArray);

    auto divOpIrAttrDef =
        builder.getDictionaryAttr({builder.getNamedAttr("compute_hint", builder.getI64IntegerAttr(4)),
                                   builder.getNamedAttr("custom_option_1", builder.getI32IntegerAttr(400))});

    auto divOp = builder.create<DivOp>(builder.getUnknownLoc(), tensorType, arg0, arg1, indexingMapsAttr, nullptr,
                                       divOpIrAttrDef, nullptr, outputsAttrArray);

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
