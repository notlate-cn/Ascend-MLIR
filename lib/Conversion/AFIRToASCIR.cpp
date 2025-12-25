//===- AFIRToASCIR.cpp - AFIR to ASC-IR conversion --------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Passes.h"
#include "Dialect/AFIR/AFIROps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

namespace mlir {
namespace afir {

namespace {
//===----------------------------------------------------------------------===//
// 转换模式：将afir.add替换为ascir.add
//===----------------------------------------------------------------------===//
struct ConvertAFIRAddOpToASCIRAddOp : public OpRewritePattern<afir::AddOp> {
  using OpRewritePattern<afir::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(afir::AddOp op, PatternRewriter &rewriter) const override {
    // 获取原操作的结果类型
    auto resultType = op.getResult().getType();

    // 创建一个未初始化的目标张量
    auto bufferTy = ascendc::TBufType::get(op.getContext(), ascendc::TPosition::VECCALC);
    auto shape = SmallVector<int64_t>(mlir::cast<ShapedType>(resultType).getShape());
    Value tbuf1 = ::mlir::ascendc::TBufOp::create(rewriter, op.getLoc(), bufferTy);
    tbuf1.dump();
    auto localTType = ascendc::LocalTensorType::get(shape, mlir::cast<ShapedType>(resultType).getElementType());
    Value dst = ascendc::TBufGetTensorOp::create(rewriter, op.getLoc(), localTType, tbuf1);
    dst.dump();


    Value tbuf2 = ::mlir::ascendc::TBufOp::create(rewriter, op.getLoc(), bufferTy);
    Value src0 = ascendc::TBufGetTensorOp::create(rewriter, op.getLoc(), localTType, tbuf2);

    Value tbuf3 = ::mlir::ascendc::TBufOp::create(rewriter, op.getLoc(), bufferTy);
    Value src1 = ascendc::TBufGetTensorOp::create(rewriter, op.getLoc(), localTType, tbuf3);

    // Value dst = ascendc::LocalTensorOp::create(rewriter, op.getLoc(), resultType);
    // dst.dump();
    // Value src0 = ascendc::LocalTensorOp::create(rewriter, op.getLoc(), resultType);
    // src0.dump();
    // Value src1 = ascendc::LocalTensorOp::create(rewriter, op.getLoc(), resultType);
    // src1.dump();
    ascendc::AddL3Op newOp = ascendc::AddL3Op::create(rewriter, op.getLoc(), dst, src0, src1);
    newOp.dump();
    // 替换原操作的结果
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
struct ConvertAFIRToASCIRPass
    : public PassWrapper<ConvertAFIRToASCIRPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAFIRToASCIRPass)

  StringRef getArgument() const override { return "convert-afir-to-ascir"; }
  StringRef getDescription() const override {
    return "Convert AFIR dialect to ASC-IR dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AFIRDialect>();
    registry.insert<ascendc::AscendCDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addIllegalDialect<AFIRDialect>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertAFIRAddOpToASCIRAddOp>(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertAFIRToASCIRPass() {
  return std::make_unique<ConvertAFIRToASCIRPass>();
}

} // namespace afir
} // namespace mlir
