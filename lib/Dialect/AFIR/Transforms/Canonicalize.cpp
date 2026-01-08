//===- AFIRCanonicalize.cpp - AFIR canonicalization pass --------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Dialect/AFIR/Ops.h"
#include "Dialect/AFIR/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace afir {

#define GEN_PASS_DEF_AFIRCANONICALIZEPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

// Pattern: add(x, 0) -> x (when 0 is a constant)
// Pattern: mul(x, 1) -> x (when 1 is a constant)
// These patterns can be expanded based on needs

struct AFIRCanonicalizePass : public impl::AFIRCanonicalizePassBase<AFIRCanonicalizePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());

    // Add canonicalization patterns here
    // patterns.add<...>(patterns.getContext());

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAFIRCanonicalizePass() {
  return std::make_unique<AFIRCanonicalizePass>();
}

}  // namespace afir
}  // namespace mlir
