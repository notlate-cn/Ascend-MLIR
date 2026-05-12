//===- AFIRSymbolizeShapes.cpp - linalg-level shape symbolization *- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
// Mints a symbol for every dynamic `?` dimension of the kernel function's block
// arguments and propagates a SymExpr per result dim through the op set we care
// about (linalg structured ops, a few tensor reshape/alloc ops).  Writes the
// result onto each op as `afir.symbolic_shapes` (ArrayAttr of StringAttr, one
// per result, each a comma-separated SymExpr list) and onto the function as
// `afir.dim_symbols`.  Ops we cannot symbolize get no attr, and the "unknown"
// propagates forward (downstream transfers reading a missing shape also bail).
//
// See docs/superpowers/specs/2026-05-12-mlir-shape-symbolization-design.md.
//
//===----------------------------------------------------------------------===//

#include "Analysis/SymbolicShape/SymExpr.h"
#include "Analysis/SymbolicShape/DimSymbolTable.h"
#include "Dialect/AFIR/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_AFIRSYMBOLIZESHAPESPASS
#include "Dialect/AFIR/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::afir::symshape;

namespace {

using ShapeVec = SmallVector<SymExpr, 4>;

/// Picks the tighter of two SymExprs known to denote the same integer; unions
/// symbols in the table so equal dynamic dims collapse to one root.
static SymExpr unify(DimSymbolTable &table, SymExpr a, SymExpr b) {
  if (a == b)
    return a;
  auto ca = a.getConst(), cb = b.getConst();
  if (ca && cb)
    return a; // both constant: if they differ the IR is ill-defined; keep one.
  if (ca)
    return a; // constant is tighter than a symbolic expr.
  if (cb)
    return b;
  if (a.getKind() == SymExpr::Kind::Sym && b.getKind() == SymExpr::Kind::Sym) {
    table.alias(a.getSym(), b.getSym());
    return a;
  }
  return a; // incompatible exprs: keep the first.
}

struct AFIRSymbolizeShapesPass
    : public impl::AFIRSymbolizeShapesPassBase<AFIRSymbolizeShapesPass> {
  void runOnOperation() override;

private:
  // Per-run state.
  DimSymbolTable table;
  DenseMap<Value, ShapeVec> shapeMap;
  // op -> per-result symbolic shapes (un-serialized), filled during the walk;
  // attrs are written after the walk so the table is final.
  SmallVector<std::pair<Operation *, SmallVector<ShapeVec>>> pending;
  // linalg op -> per-iteration-dim extent SymExpr (un-serialized).  Emitted as
  // `afir.iter_extents` so consumers (Collapse) need not re-derive it.
  DenseMap<Operation *, ShapeVec> iterExtents;

  const ShapeVec *getShape(Value v) const {
    auto it = shapeMap.find(v);
    return it == shapeMap.end() ? nullptr : &it->second;
  }

  // Transfer functions: return std::nullopt to bail (no attr for this op).
  std::optional<SmallVector<ShapeVec>> transfer(Operation *op);
  std::optional<SmallVector<ShapeVec>> transferLinalg(linalg::LinalgOp op);
  std::optional<SymExpr> interpretSize(Value v);
};

//===----------------------------------------------------------------------===//
// transfer functions
//===----------------------------------------------------------------------===//

std::optional<SymExpr> AFIRSymbolizeShapesPass::interpretSize(Value v) {
  if (auto cst = getConstantIntValue(v))
    return SymExpr::constant(*cst);
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto dim = dyn_cast<tensor::DimOp>(def)) {
    auto idx = dim.getConstantIndex();
    if (!idx)
      return std::nullopt;
    const ShapeVec *src = getShape(dim.getSource());
    if (!src || *idx < 0 || (size_t)*idx >= src->size())
      return std::nullopt;
    return (*src)[*idx];
  }
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    auto a = interpretSize(mul.getLhs()), b = interpretSize(mul.getRhs());
    if (!a || !b)
      return std::nullopt;
    return SymExpr::mul(*a, *b);
  }
  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto a = interpretSize(add.getLhs()), b = interpretSize(add.getRhs());
    if (!a || !b)
      return std::nullopt;
    return SymExpr::add(*a, *b);
  }
  if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    auto a = interpretSize(sub.getLhs()), b = interpretSize(sub.getRhs());
    if (!a || !b)
      return std::nullopt;
    return SymExpr::sub(*a, *b);
  }
  return std::nullopt;
}

std::optional<SmallVector<ShapeVec>>
AFIRSymbolizeShapesPass::transferLinalg(linalg::LinalgOp op) {
  auto maps = op.getIndexingMapsArray();
  unsigned numLoops = op.getNumLoops();
  if (maps.size() != op->getNumOperands())
    return std::nullopt;

  SmallVector<SymExpr> iterSize(numLoops);
  for (auto [i, opnd] : llvm::enumerate(op->getOpOperands())) {
    const ShapeVec *vShape = getShape(opnd.get());
    if (!vShape)
      return std::nullopt;
    AffineMap m = maps[i];
    if (!m.isProjectedPermutation())
      return std::nullopt;
    if (m.getNumResults() != vShape->size())
      return std::nullopt;
    for (unsigned pos = 0; pos < m.getNumResults(); ++pos) {
      auto d = dyn_cast<AffineDimExpr>(m.getResult(pos));
      if (!d)
        return std::nullopt;
      unsigned dim = d.getPosition();
      SymExpr pin = (*vShape)[pos];
      iterSize[dim] = iterSize[dim].isValid() ? unify(table, iterSize[dim], pin)
                                              : pin;
    }
  }
  for (unsigned d = 0; d < numLoops; ++d)
    if (!iterSize[d].isValid())
      return std::nullopt;

  SmallVector<ShapeVec> results;
  unsigned firstInit = op.getNumDpsInputs();
  for (unsigned r = 0; r < op->getNumResults(); ++r) {
    AffineMap m = maps[firstInit + r];
    if (!m.isProjectedPermutation())
      return std::nullopt;
    ShapeVec rs;
    for (unsigned pos = 0; pos < m.getNumResults(); ++pos) {
      auto d = dyn_cast<AffineDimExpr>(m.getResult(pos));
      if (!d)
        return std::nullopt;
      rs.push_back(iterSize[d.getPosition()]);
    }
    results.push_back(std::move(rs));
  }
  iterExtents[op.getOperation()] = ShapeVec(iterSize.begin(), iterSize.end());
  return results;
}

std::optional<SmallVector<ShapeVec>>
AFIRSymbolizeShapesPass::transfer(Operation *op) {
  // linalg structured ops (generic, fill, generalized matmul/reduce/...).
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
    return transferLinalg(linalgOp);

  if (auto cs = dyn_cast<tensor::CollapseShapeOp>(op)) {
    const ShapeVec *src = getShape(cs.getSrc());
    if (!src)
      return std::nullopt;
    ShapeVec rs;
    for (const auto &group : cs.getReassociationIndices()) {
      SymExpr acc = SymExpr::constant(1);
      for (int64_t srcDim : group) {
        if (srcDim < 0 || (size_t)srcDim >= src->size())
          return std::nullopt;
        acc = SymExpr::mul(acc, (*src)[srcDim]);
      }
      rs.push_back(acc);
    }
    return SmallVector<ShapeVec>{rs};
  }

  if (auto es = dyn_cast<tensor::ExpandShapeOp>(op)) {
    const ShapeVec *src = getShape(es.getSrc());
    if (!src)
      return std::nullopt;
    auto resTy = cast<RankedTensorType>(es.getResult().getType());
    ShapeVec rs(resTy.getRank(), SymExpr());
    auto reassoc = es.getReassociationIndices();
    if (reassoc.size() != src->size())
      return std::nullopt;
    for (auto [srcDim, group] : llvm::enumerate(reassoc)) {
      if (group.size() == 1) {
        rs[group[0]] = (*src)[srcDim]; // identity dim, pass through
        continue;
      }
      // A genuine split: we only model splitting a statically-known dim.
      for (int64_t g : group) {
        if (resTy.isDynamicDim(g))
          return std::nullopt;
        rs[g] = SymExpr::constant(resTy.getDimSize(g));
      }
    }
    for (const SymExpr &e : rs)
      if (!e.isValid())
        return std::nullopt;
    return SmallVector<ShapeVec>{rs};
  }

  if (auto pad = dyn_cast<tensor::PadOp>(op)) {
    const ShapeVec *src = getShape(pad.getSource());
    if (!src)
      return std::nullopt;
    SmallVector<OpFoldResult> lo = pad.getMixedLowPad();
    SmallVector<OpFoldResult> hi = pad.getMixedHighPad();
    if (lo.size() != src->size() || hi.size() != src->size())
      return std::nullopt;
    ShapeVec rs;
    for (size_t i = 0; i < src->size(); ++i) {
      auto cl = getConstantIntValue(lo[i]), ch = getConstantIntValue(hi[i]);
      if (!cl || !ch)
        return std::nullopt;
      rs.push_back(SymExpr::add((*src)[i], SymExpr::constant(*cl + *ch)));
    }
    return SmallVector<ShapeVec>{rs};
  }

  if (auto cast0 = dyn_cast<tensor::CastOp>(op)) {
    const ShapeVec *src = getShape(cast0.getSource());
    if (!src)
      return std::nullopt;
    return SmallVector<ShapeVec>{*src};
  }

  if (auto ins = dyn_cast<tensor::InsertSliceOp>(op)) {
    const ShapeVec *dest = getShape(ins.getDest());
    if (!dest)
      return std::nullopt;
    return SmallVector<ShapeVec>{*dest};
  }

  // tensor.empty / bufferization.alloc_tensor / tensor.extract_slice and any
  // other op with results: build from the dynamic size operands if we can.
  auto buildFromDynamicSizes = [&](RankedTensorType ty,
                                   ValueRange dynSizes) -> std::optional<ShapeVec> {
    ShapeVec rs;
    unsigned di = 0;
    for (int64_t d = 0; d < ty.getRank(); ++d) {
      if (!ty.isDynamicDim(d)) {
        rs.push_back(SymExpr::constant(ty.getDimSize(d)));
        continue;
      }
      if (di >= dynSizes.size())
        return std::nullopt;
      auto e = interpretSize(dynSizes[di++]);
      if (!e)
        return std::nullopt;
      rs.push_back(*e);
    }
    return rs;
  };
  if (auto empty = dyn_cast<tensor::EmptyOp>(op)) {
    auto ty = cast<RankedTensorType>(empty.getResult().getType());
    auto rs = buildFromDynamicSizes(ty, empty.getDynamicSizes());
    if (!rs)
      return std::nullopt;
    return SmallVector<ShapeVec>{*rs};
  }
  if (auto alloc = dyn_cast<bufferization::AllocTensorOp>(op)) {
    auto ty = cast<RankedTensorType>(alloc.getResult().getType());
    auto rs = buildFromDynamicSizes(ty, alloc.getDynamicSizes());
    if (!rs)
      return std::nullopt;
    return SmallVector<ShapeVec>{*rs};
  }
  if (auto es = dyn_cast<tensor::ExtractSliceOp>(op)) {
    auto ty = cast<RankedTensorType>(es.getResult().getType());
    if (!ty.hasStaticShape())
      return std::nullopt; // rank-reduced / dynamic slices: not modeled.
    ShapeVec rs;
    for (int64_t d = 0; d < ty.getRank(); ++d)
      rs.push_back(SymExpr::constant(ty.getDimSize(d)));
    return SmallVector<ShapeVec>{rs};
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// pass driver
//===----------------------------------------------------------------------===//

void AFIRSymbolizeShapesPass::runOnOperation() {
  func::FuncOp func = getOperation();
  table = DimSymbolTable();
  shapeMap.clear();
  pending.clear();
  iterExtents.clear();

  if (func.isExternal() || func.getBody().empty())
    return;

  // Only do anything if the signature has a dynamic dim -- otherwise every
  // shape is a constant and there is nothing to symbolize.
  bool anyDynamic = false;
  for (Type t : func.getArgumentTypes())
    if (auto st = dyn_cast<ShapedType>(t); st && !st.hasStaticShape())
      anyDynamic = true;
  if (!anyDynamic)
    return;

  Block &entry = func.getBody().front();

  // 1. arg symbols.
  for (auto [argIdx, arg] : llvm::enumerate(entry.getArguments())) {
    auto st = dyn_cast<ShapedType>(arg.getType());
    if (!st || !st.hasRank())
      continue;
    ShapeVec sv;
    for (int64_t d = 0; d < st.getRank(); ++d) {
      if (st.isDynamicDim(d))
        sv.push_back(SymExpr::sym(table.getOrCreateForArgDim(argIdx, d)));
      else
        sv.push_back(SymExpr::constant(st.getDimSize(d)));
    }
    shapeMap[arg] = std::move(sv);
  }

  // 2. topological op walk (entry block is already in def-before-use order).
  for (Operation &op : entry) {
    if (op.getNumResults() == 0)
      continue;
    auto res = transfer(&op);
    if (!res || res->size() != op.getNumResults())
      continue; // bail: no attr, results stay out of shapeMap.
    // Clamp statically-known dims to their constant -- the type is ground truth.
    bool bad = false;
    for (auto [r, sv] : llvm::enumerate(*res)) {
      auto st = dyn_cast<ShapedType>(op.getResult(r).getType());
      if (!st || !st.hasRank() || (int64_t)sv.size() != st.getRank()) {
        bad = true;
        break;
      }
      for (int64_t d = 0; d < st.getRank(); ++d)
        if (!st.isDynamicDim(d))
          sv[d] = SymExpr::constant(st.getDimSize(d));
    }
    if (bad)
      continue;
    for (auto [r, sv] : llvm::enumerate(*res))
      shapeMap[op.getResult(r)] = sv;
    pending.push_back({&op, std::move(*res)});
  }

  // 3. table is final -- serialize and write attrs.
  Builder b(&getContext());
  auto serializedStr = [&](ArrayRef<SymExpr> sv) {
    SmallVector<SymExpr, 4> ser;
    for (const SymExpr &e : sv)
      ser.push_back(table.toSerialized(e));
    return b.getStringAttr(symExprListStr(ser));
  };
  for (auto &[op, resShapes] : pending) {
    SmallVector<Attribute> perResult;
    for (ShapeVec &sv : resShapes)
      perResult.push_back(serializedStr(sv));
    op->setAttr("afir.symbolic_shapes", b.getArrayAttr(perResult));
    auto it = iterExtents.find(op);
    if (it != iterExtents.end())
      op->setAttr("afir.iter_extents", serializedStr(it->second));
  }
  func->setAttr("afir.dim_symbols", table.toAttr(&getContext()));
}

} // namespace

std::unique_ptr<Pass> mlir::createAFIRSymbolizeShapesPass() {
  return std::make_unique<AFIRSymbolizeShapesPass>();
}
