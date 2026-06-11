//===- EliminateCfAssertPass.cpp - Remove cf.assert ops ----------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//
//
// torch-mlir inserts cf.assert ops for dynamic shape broadcast validation,
// e.g.:
//
//   %0 = arith.cmpi eq, %dim, %dim_0 : index
//   cf.assert %0, "mismatched size for broadcast"
//
// These are unnecessary in our pipeline (shape compatibility is guaranteed by
// the calling framework) and afir-translate does not support cf dialect.
// This pass removes them early in the torch-mlir post-processing stage.
//
//===----------------------------------------------------------------------===//

#include "Conversion/EliminateCfAssert/EliminateCfAssertPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DECL_ELIMINATECFASSERTPASS
#define GEN_PASS_DEF_ELIMINATECFASSERTPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

struct EliminateCfAssertPass
    : public ::impl::EliminateCfAssertPassBase<EliminateCfAssertPass> {
  using EliminateCfAssertPassBase::EliminateCfAssertPassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Collect all cf.assert ops.
    SmallVector<cf::AssertOp> assertOps;
    func.walk([&](cf::AssertOp op) { assertOps.push_back(op); });

    if (assertOps.empty())
      return;

    // Erase each assert and DCE its condition if unused.
    for (cf::AssertOp assertOp : assertOps) {
      Value cond = assertOp.getArg();
      assertOp.erase();

      // Simple DCE: if the condition's defining op has no remaining users,
      // erase it too (typically an arith.cmpi).
      if (Operation *defOp = cond.getDefiningOp()) {
        if (defOp->use_empty())
          defOp->erase();
      }
    }
  }
};

std::unique_ptr<Pass> createEliminateCfAssertPass() {
  return std::make_unique<EliminateCfAssertPass>();
}

} // namespace mlir::afir