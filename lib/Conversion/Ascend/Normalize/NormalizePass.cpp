//===- NormalizePass.cpp - Ascend normalize pass -----------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Normalize/NormalizePass.h"

#include "SymbolEquivalenceAnalysis.h"
#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Common/SymbolConstraints.h"
#include "Conversion/Ascend/Debug/DebugOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DECL_ASCENDNORMALIZEPASS
#define GEN_PASS_DEF_ASCENDNORMALIZEPASS
#include "Conversion/Ascend/Passes.h.inc"

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

bool isAllowedInputOperation(Operation *op) {
  // Frontends emit cf.assert for dynamic shape guards. It is a guard carrier,
  // not a control-flow region that participates in kernel partitioning.
  if (isa<cf::AssertOp>(op))
    return true;

  return isAllowedInputDialect(op->getName().getDialectNamespace());
}

} // namespace

namespace mlir::ascend {

struct AscendNormalizePass
    : public ::impl::AscendNormalizePassBase<AscendNormalizePass> {
  using AscendNormalizePassBase::AscendNormalizePassBase;

  void runOnOperation() override {
    ::mlir::ascend::debug::DebugOptions options{
        ::mlir::ascend::debug::parseDebugStage(debugStage), dumpReport};
    if (::mlir::ascend::debug::shouldDump(
            options, ::mlir::ascend::debug::DebugStage::Normalize))
      ::mlir::ascend::debug::emitStageHeader(
          llvm::errs(), ::mlir::ascend::debug::DebugStage::Normalize,
          getArgument());

    ModuleOp module = getOperation();
    if (module
            .walk([&](Operation *op) {
              if (isAllowedInputOperation(op))
                return WalkResult::advance();

              op->emitError() << "unsupported dialect before Kernelize";
              return WalkResult::interrupt();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = module.getContext();
    if (module
            .walk([&](func::FuncOp func) {
              if (func->getAttr(::mlir::ascend::kSymbolConstraintsAttr) &&
                  failed(::mlir::ascend::symbol::verifySymbolConstraintAttr(
                      func)))
                return WalkResult::interrupt();

              FailureOr<::mlir::ascend::normalize::SymbolEquivalenceResult>
                  constraints =
                      ::mlir::ascend::normalize::analyzeSymbolEquivalence(
                          func);
              if (failed(constraints))
                return WalkResult::interrupt();

              func->setAttr(::mlir::ascend::kSymbolConstraintsAttr,
                            constraints->attr);
              func->setAttr(::mlir::ascend::kNormalizedAttr,
                            BoolAttr::get(context, true));
              return WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<Pass> createAscendNormalizePass() {
  return std::make_unique<AscendNormalizePass>();
}

} // namespace mlir::ascend
