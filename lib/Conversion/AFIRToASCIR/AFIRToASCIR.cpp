//===- AFIRToASCIR.cpp - AFIR to ASC-IR conversion --------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AFIRToASCIR/AFIRToASCIR.h"
#include "Conversion/AFIRToASCIR/Math/Elementwise.h"
#include "Dialect/AFIR/AFIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

#define GEN_PASS_DECL_CONVERTAFIRTOASCIRPASS
#define GEN_PASS_DEF_CONVERTAFIRTOASCIRPASS
#include "Conversion/Passes.h.inc"

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
struct ConvertAFIRToASCIRPass : public ::impl::ConvertAFIRToASCIRPassBase<ConvertAFIRToASCIRPass> {
  using Base = ::impl::ConvertAFIRToASCIRPassBase<ConvertAFIRToASCIRPass>;
  using Base::Base;

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

    std::string ascirText;
    llvm::raw_string_ostream os(ascirText);
    module.print(os);
    os.flush();

    if (!ascirPath.empty()) {
      std::error_code ec;
      llvm::ToolOutputFile file(ascirPath, ec, llvm::sys::fs::OF_None);
      if (ec) {
        llvm::errs() << "Error opening file: " << ec.message() << "\n";
        signalPassFailure();
        return;
      }
      file.os() << ascirText;
      file.keep();
    } else {
      llvm::errs() << ascirText;
    }
  }
};

std::unique_ptr<Pass> createConvertAFIRToASCIRPass() {
  return std::unique_ptr<Pass>(new ConvertAFIRToASCIRPass());
}

}  // namespace afir
}  // namespace mlir
