//===- AFIRToASCIR.cpp - AFIR to ASC-IR conversion --------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AFIRToASCIR/AFIRToASCIR.h"
#include "Conversion/AFIRToASCIR/Math/Elementwise.h"
#include "Dialect/AFIR/Ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

namespace mlir {
namespace afir {

namespace {

struct FuncReturnOpTypeConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

}  // namespace

class AFIRToASCIRTypeConverter : public TypeConverter {
 public:
  AFIRToASCIRTypeConverter() {
    addConversion([](Type type) -> std::optional<Type> {
      if (auto tensorType = dyn_cast<TensorType>(type)) {
        auto shape = tensorType.getShape();
        auto elementType = tensorType.getElementType();
        return ascendc::LocalTensorType::get(shape, elementType);
      }
      return type;
    });
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
struct ConvertAFIRToASCIRPass : public PassWrapper<ConvertAFIRToASCIRPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAFIRToASCIRPass)

  StringRef getArgument() const override {
    return "convert-afir-to-ascir";
  }
  StringRef getDescription() const override {
    return "Convert AFIR dialect to ASC-IR dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ascendc::AscendCDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target.addIllegalDialect<AFIRDialect>();
    target.addLegalDialect<ascendc::AscendCDialect>();

    AFIRToASCIRTypeConverter typeConverter;
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op.getOperandTypes()); });

    RewritePatternSet patterns(context);
    populateLoweringAFIRElementwiseOpToASCIRPattern(patterns, context, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, typeConverter);
    patterns.add<FuncReturnOpTypeConversion>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvertAFIRToASCIRPass() {
  return std::make_unique<ConvertAFIRToASCIRPass>();
}

}  // namespace afir
}  // namespace mlir
