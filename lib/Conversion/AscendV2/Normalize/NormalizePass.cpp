//===- NormalizePass.cpp - Ascend V2 normalize pass -----------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Normalize/NormalizePass.h"

#include "Conversion/AscendV2/Common/Attributes.h"
#include "Conversion/AscendV2/Debug/DebugOptions.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DECL_ASCENDNORMALIZEPASS
#define GEN_PASS_DEF_ASCENDNORMALIZEPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace {

bool isAllowedInputDialect(StringRef dialectNamespace) {
  return llvm::StringSwitch<bool>(dialectNamespace)
      .Case("builtin", true)
      .Case("func", true)
      .Case("tensor", true)
      .Case("linalg", true)
      .Case("arith", true)
      .Case("math", true)
      .Default(false);
}

} // namespace

namespace mlir::afir {

struct AscendNormalizePass
    : public ::impl::AscendNormalizePassBase<AscendNormalizePass> {
  using AscendNormalizePassBase::AscendNormalizePassBase;

  void runOnOperation() override {
    ::mlir::ascend::v2::DebugOptions options{
        ::mlir::ascend::v2::parseDebugStage(debugStage), dumpReport};
    if (::mlir::ascend::v2::shouldDump(
            options, ::mlir::ascend::v2::DebugStage::Normalize))
      ::mlir::ascend::v2::emitStageHeader(
          llvm::errs(), ::mlir::ascend::v2::DebugStage::Normalize,
          getArgument());

    ModuleOp module = getOperation();
    if (module
            .walk([&](Operation *op) {
              StringRef dialectNamespace =
                  op->getName().getDialectNamespace();
              if (isAllowedInputDialect(dialectNamespace))
                return WalkResult::advance();

              op->emitError() << "unsupported dialect before Kernelize";
              return WalkResult::interrupt();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = module.getContext();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "func.func")
        op->setAttr(::mlir::afir::ascend::v2::kNormalizedAttr,
                    BoolAttr::get(context, true));
    });
  }
};

std::unique_ptr<Pass> createAscendNormalizePass() {
  return std::make_unique<AscendNormalizePass>();
}

} // namespace mlir::afir
