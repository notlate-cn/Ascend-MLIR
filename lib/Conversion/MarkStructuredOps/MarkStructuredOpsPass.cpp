//===- MarkStructuredOpsPass.cpp - Mark structured linalg ops -------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/MarkStructuredOps/MarkStructuredOpsPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GEN_PASS_DECL_MARKSTRUCTUREDOPSPASS
#define GEN_PASS_DEF_MARKSTRUCTUREDOPSPASS
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir::afir {

namespace {

struct MarkStructuredOpsPass
    : public ::impl::MarkStructuredOpsPassBase<MarkStructuredOpsPass> {
  void runOnOperation() override;
};

}  // namespace

void MarkStructuredOpsPass::runOnOperation() {
  // Detection logic implemented in next task.
}

std::unique_ptr<Pass> createMarkStructuredOpsPass() {
  return std::make_unique<MarkStructuredOpsPass>();
}

}  // namespace mlir::afir
