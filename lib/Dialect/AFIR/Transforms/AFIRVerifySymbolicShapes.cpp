//===- AFIRVerifySymbolicShapes.cpp ----------------------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// Checks the well-formedness of `afir.symbolic_shapes` / `afir.dim_symbols`.
// Intended for lit tests (`-verify-diagnostics`).
//
//===----------------------------------------------------------------------===//

#include "Analysis/SymbolicShape/SymExpr.h"
#include "Dialect/AFIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_AFIRVERIFYSYMBOLICSHAPESPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::afir::symshape;

namespace {

struct AFIRVerifySymbolicShapesPass
    : public impl::AFIRVerifySymbolicShapesPassBase<
          AFIRVerifySymbolicShapesPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto dimSyms = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
    unsigned numSyms = dimSyms ? dimSyms.size() : 0;
    bool bad = false;

    func.walk([&](Operation *op) {
      auto attr = op->getAttrOfType<ArrayAttr>("afir.symbolic_shapes");
      if (!attr)
        return;
      if (attr.size() != op->getNumResults()) {
        op->emitError("afir.symbolic_shapes: length ")
            << attr.size() << " != #results " << op->getNumResults();
        bad = true;
        return;
      }
      for (auto [r, a] : llvm::enumerate(attr)) {
        auto str = dyn_cast<StringAttr>(a);
        if (!str) {
          op->emitError("afir.symbolic_shapes[") << r << "] is not a string";
          bad = true;
          return;
        }
        auto list = parseSymExprList(str.getValue());
        if (!list) {
          op->emitError("afir.symbolic_shapes[")
              << r << "] = \"" << str.getValue() << "\" failed to parse";
          bad = true;
          return;
        }
        auto st = dyn_cast<ShapedType>(op->getResult(r).getType());
        if (!st || !st.hasRank()) {
          op->emitError("afir.symbolic_shapes[")
              << r << "] on a non-ranked result";
          bad = true;
          return;
        }
        if ((int64_t)list->size() != st.getRank()) {
          op->emitError("afir.symbolic_shapes[")
              << r << "]: " << list->size() << " dims, type rank "
              << st.getRank();
          bad = true;
          return;
        }
        for (int64_t d = 0; d < st.getRank(); ++d) {
          bool oob = false;
          (*list)[d].walkSymbols([&](SymId id) {
            if (id >= numSyms)
              oob = true;
          });
          if (oob) {
            op->emitError("afir.symbolic_shapes[")
                << r << "][" << d << "]: symbol id out of range (have "
                << numSyms << " symbols)";
            bad = true;
            return;
          }
          if (!st.isDynamicDim(d)) {
            auto c = (*list)[d].getConst();
            if (!c || *c != st.getDimSize(d)) {
              op->emitError("afir.symbolic_shapes[")
                  << r << "][" << d << "]: static dim is " << st.getDimSize(d)
                  << " but symbolic shape is " << (*list)[d].str();
              bad = true;
              return;
            }
          }
        }
      }
    });
    if (bad)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createAFIRVerifySymbolicShapesPass() {
  return std::make_unique<AFIRVerifySymbolicShapesPass>();
}
