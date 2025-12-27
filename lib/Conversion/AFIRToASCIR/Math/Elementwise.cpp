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

//===----------------------------------------------------------------------===//
// Elementwise Binary Operation Conversion Pattern
//===----------------------------------------------------------------------===//

template <typename AFIRBinaryOp, typename ASCBinaryOp>
struct ConvertAFIRBinaryElementwiseOpToASCIR
    : public OpRewritePattern<AFIRBinaryOp> {
  using OpRewritePattern<AFIRBinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AFIRBinaryOp op,
                               PatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();

    auto bufferTy = ascendc::TBufType::get(op.getContext(), ascendc::TPosition::VECCALC);
    auto shape = SmallVector<int64_t>(mlir::cast<ShapedType>(resultType).getShape());
    Value tbuf1 = ::mlir::ascendc::TBufOp::create(rewriter, op.getLoc(), bufferTy);
    auto localTType = ascendc::LocalTensorType::get(shape, mlir::cast<ShapedType>(resultType).getElementType());
    Value dst = ascendc::TBufGetTensorOp::create(rewriter, op.getLoc(), localTType, tbuf1);

    ASCBinaryOp::create(rewriter, op.getLoc(), dst, op.getLhs(), op.getRhs());
    rewriter.replaceOp(op, dst);
    return success();
  }
};

using ConvertAFIRAddOpToASCIR =
    ConvertAFIRBinaryElementwiseOpToASCIR<afir::AddOp, ascendc::AddL3Op>;
using ConvertAFIRSubOpToASCIR =
    ConvertAFIRBinaryElementwiseOpToASCIR<afir::SubOp, ascendc::SubL3Op>;
using ConvertAFIRMulOpToASCIR =
    ConvertAFIRBinaryElementwiseOpToASCIR<afir::MulOp, ascendc::MulL3Op>;
using ConvertAFIRDivOpToASCIR =
    ConvertAFIRBinaryElementwiseOpToASCIR<afir::DivOp, ascendc::DivL3Op>;

} // namespace

void mlir::afir::populateLoweringAFIRElementwiseOpToASCIRPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<ConvertAFIRAddOpToASCIR>(ctx);
  patterns.add<ConvertAFIRSubOpToASCIR>(ctx);
  patterns.add<ConvertAFIRMulOpToASCIR>(ctx);
  patterns.add<ConvertAFIRDivOpToASCIR>(ctx);
}
