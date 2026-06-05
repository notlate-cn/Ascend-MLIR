//===- AFIRSymbolicDimCSE.cpp - symbol-keyed tensor.dim CSE ------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// Merges `tensor.dim` ops that denote the same dynamic-shape symbol into one
// canonical `tensor.dim` on the symbol's root block argument.  Reads the attrs
// produced by --afir-symbolize-shapes (afir.dim_symbols on the func,
// afir.symbolic_shape on args, afir.symbolic_shapes on ops).  See
// docs/superpowers/specs/2026-06-05-symbol-aware-dim-cse-design.md.
//
//===----------------------------------------------------------------------===//

#include "Analysis/SymbolicShape/SymExpr.h"
#include "Dialect/AFIR/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_AFIRSYMBOLICDIMCSEPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::afir::symshape;

namespace {
struct AFIRSymbolicDimCSEPass
    : public impl::AFIRSymbolicDimCSEPassBase<AFIRSymbolicDimCSEPass> {
  void runOnOperation() override {}
};
} // namespace

std::unique_ptr<Pass> mlir::createAFIRSymbolicDimCSEPass() {
  return std::make_unique<AFIRSymbolicDimCSEPass>();
}
