//===- Elementwise.cpp - AFIR Elementwise ops to ASC-IR conversion --*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// This file implements the conversion patterns for AFIR elementwise operations
// to ASC-IR dialect.
//
//===----------------------------------------------------------------------===//

#include "Conversion/AFIRToASCIR/Math/Elementwise.h"
#include "Conversion/AFIRToASCIR/DialectBuilder.h"
#include "Dialect/AFIR/AFIROps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

using namespace mlir;
using namespace mlir::afir;

namespace {

template <typename AFIRBinaryOp, typename ASCBinaryOp>
struct ConvertAFIRBinaryElementwiseOpToASCIR : public ConversionPattern {
  ConvertAFIRBinaryElementwiseOpToASCIR(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, AFIRBinaryOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    auto binaryOp = cast<AFIRBinaryOp>(op);
    auto resultType = op->getResult(0).getType();

    auto bufferTy = ascendc::TBufType::get(op->getContext(), ascendc::TPosition::VECCALC);
    auto shape = SmallVector<int64_t>(mlir::cast<ShapedType>(resultType).getShape());
    Value tbuf = ::mlir::ascendc::TBufOp::create(rewriter, op->getLoc(), bufferTy);
    auto localTType = ascendc::LocalTensorType::get(shape, mlir::cast<ShapedType>(resultType).getElementType());
    Value dst = ascendc::TBufGetTensorOp::create(rewriter, op->getLoc(), localTType, tbuf);

    Value lhs = operands[0];
    Value rhs = operands[1];

    ASCBinaryOp::create(rewriter, op->getLoc(), dst, lhs, rhs);
    rewriter.replaceOp(op, dst);
    return success();
  }
};

}  // namespace

void mlir::afir::populateLoweringAFIRElementwiseOpToASCIRPattern(RewritePatternSet &patterns, MLIRContext *ctx,
                                                                  TypeConverter &typeConverter) {
  patterns.add<ConvertAFIRBinaryElementwiseOpToASCIR<afir::AddOp, ascendc::AddL3Op>>(typeConverter, ctx);
  patterns.add<ConvertAFIRBinaryElementwiseOpToASCIR<afir::SubOp, ascendc::SubL3Op>>(typeConverter, ctx);
  patterns.add<ConvertAFIRBinaryElementwiseOpToASCIR<afir::MulOp, ascendc::MulL3Op>>(typeConverter, ctx);
  patterns.add<ConvertAFIRBinaryElementwiseOpToASCIR<afir::DivOp, ascendc::DivL3Op>>(typeConverter, ctx);
}
