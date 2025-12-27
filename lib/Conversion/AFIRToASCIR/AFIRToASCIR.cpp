//===- AFIRToASCIR.cpp - AFIR to ASC-IR conversion --------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AFIRToASCIR/AFIRToASCIR.h"
#include "Conversion/AFIRToASCIR/Math/Elementwise.h"
#include "Dialect/AFIR/AFIROps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ascir/Dialect/Asc/IR/Asc.h"

namespace mlir {
namespace afir {

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

    RewritePatternSet patterns(context);
    populateLoweringAFIRElementwiseOpToASCIRPattern(patterns, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
  }
};

std::unique_ptr<Pass> createConvertAFIRToASCIRPass() {
  return std::make_unique<ConvertAFIRToASCIRPass>();
}

}  // namespace afir
}  // namespace mlir
