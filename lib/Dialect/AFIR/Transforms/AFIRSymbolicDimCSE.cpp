//===- AFIRSymbolicDimCSE.cpp - symbol-keyed tensor.dim CSE ------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// When two or more `tensor.dim` ops denote the same dynamic-shape symbol,
// replaces them with one canonical `tensor.dim` on the symbol's root block
// argument.  A lone `tensor.dim` for a symbol is left untouched (nothing to
// merge).  Reads the attrs
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

struct Root {
  unsigned arg;
  unsigned dim;
};

/// Resolves the serialized symbol id that `tensor.dim v, idx` denotes, or
/// nullopt if v has no symbolic shape here, idx is out of range, or the dim is
/// not a bare symbol (constant or compound expr).
static std::optional<SymId> resolveSymbol(Value v, int64_t idx,
                                          func::FuncOp func) {
  StringRef serialized;
  if (auto barg = dyn_cast<BlockArgument>(v)) {
    if (barg.getOwner() != &func.getBody().front())
      return std::nullopt;
    auto attr = func.getArgAttrOfType<StringAttr>(barg.getArgNumber(),
                                                  "afir.symbolic_shape");
    if (!attr)
      return std::nullopt;
    serialized = attr.getValue();
  } else {
    // A non-block-argument value is necessarily an OpResult, so it always has a
    // defining op.
    Operation *def = cast<OpResult>(v).getOwner();
    auto arr = def->getAttrOfType<ArrayAttr>("afir.symbolic_shapes");
    if (!arr)
      return std::nullopt;
    unsigned resNo = cast<OpResult>(v).getResultNumber();
    if (resNo >= arr.size())
      return std::nullopt;
    auto s = dyn_cast<StringAttr>(arr[resNo]);
    if (!s)
      return std::nullopt;
    serialized = s.getValue();
  }
  auto list = parseSymExprList(serialized);
  if (!list || idx < 0 || (size_t)idx >= list->size())
    return std::nullopt;
  const SymExpr &e = (*list)[idx];
  if (e.getKind() != SymExpr::Kind::Sym)
    return std::nullopt;
  return e.getSym();
}

struct AFIRSymbolicDimCSEPass
    : public impl::AFIRSymbolicDimCSEPassBase<AFIRSymbolicDimCSEPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal() || func.getBody().empty())
      return;
    auto dimSymbols = func->getAttrOfType<ArrayAttr>("afir.dim_symbols");
    if (!dimSymbols)
      return; // not symbolized / fully static.

    // id -> root (arg, dim).
    DenseMap<SymId, Root> idToRoot;
    for (Attribute a : dimSymbols) {
      auto d = cast<DictionaryAttr>(a);
      auto id = cast<IntegerAttr>(d.get("id")).getInt();
      auto arg = cast<IntegerAttr>(d.get("arg")).getInt();
      auto dim = cast<IntegerAttr>(d.get("dim")).getInt();
      idToRoot[static_cast<SymId>(id)] = {static_cast<unsigned>(arg),
                                          static_cast<unsigned>(dim)};
    }

    // Group resolvable tensor.dim ops by symbol id.
    DenseMap<SymId, SmallVector<tensor::DimOp>> classes;
    func.walk([&](tensor::DimOp d) {
      std::optional<int64_t> idx = d.getConstantIndex();
      if (!idx)
        return;
      std::optional<SymId> sym = resolveSymbol(d.getSource(), *idx, func);
      if (!sym || !idToRoot.count(*sym))
        return;
      classes[*sym].push_back(d);
    });

    // Deterministic order: sort symbol ids.
    SmallVector<SymId> keys;
    for (auto &kv : classes)
      keys.push_back(kv.first);
    llvm::sort(keys);

    Block &entry = func.getBody().front();
    OpBuilder b(&getContext());
    DenseMap<int64_t, Value> idxConsts; // dedup index constants we create.
    auto getIdx = [&](int64_t v) -> Value {
      Value &slot = idxConsts[v];
      if (!slot) {
        b.setInsertionPointToStart(&entry);
        slot = b.create<arith::ConstantIndexOp>(func.getLoc(), v);
      }
      return slot;
    };

    for (SymId sym : keys) {
      SmallVector<tensor::DimOp> &members = classes[sym];
      if (members.size() < 2)
        continue; // nothing to merge.
      Root root = idToRoot[sym];
      Value rootArg = entry.getArgument(root.arg);
      Value cidx = getIdx(static_cast<int64_t>(root.dim));
      b.setInsertionPointAfter(cidx.getDefiningOp());
      auto canon = b.create<tensor::DimOp>(members.front().getLoc(), rootArg, cidx);
      for (tensor::DimOp m : members) {
        m.getResult().replaceAllUsesWith(canon.getResult());
        m.erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createAFIRSymbolicDimCSEPass() {
  return std::make_unique<AFIRSymbolicDimCSEPass>();
}
