#include "TileFuseUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

using namespace mlir;
using namespace mlir::vector_plan;

namespace mlir::afir {

bool resultUsedOnlyByGroupMembers(linalg::LinalgOp op,
                                   const CollapsedGroupInfo &info) {
  llvm::SmallPtrSet<Operation *, 8> members;
  for (linalg::LinalgOp m : info.topoMembers)
    members.insert(m.getOperation());
  for (Value r : op->getResults())
    for (Operation *user : r.getUsers())
      if (!members.contains(user))
        return false;
  return true;
}

// If `op` is a standalone transpose (a `linalg.generic` with one input read
// through a non-identity permutation, an identity-mapped output, and a
// yield-only body), return that permutation as `perm[inPos] = iteration dim
// occupying input-layout position inPos`; else nullopt.  (`linalg.transpose`
// becomes exactly this shape after `--linalg-generalize-named-ops`.)
static std::optional<SmallVector<int64_t>> transposePerm(linalg::LinalgOp op) {
  auto gen = dyn_cast<linalg::GenericOp>(op.getOperation());
  if (!gen || gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1)
    return std::nullopt;
  auto maps = gen.getIndexingMapsArray();
  if (maps.size() != 2 || !maps[1].isIdentity())
    return std::nullopt;
  unsigned rank = gen.getNumLoops();
  if (maps[0].getNumResults() != rank)
    return std::nullopt;
  SmallVector<int64_t> perm;
  llvm::SmallDenseSet<int64_t> seen;
  for (AffineExpr e : maps[0].getResults()) {
    auto de = dyn_cast<AffineDimExpr>(e);
    if (!de)
      return std::nullopt;
    int64_t p = (int64_t)de.getPosition();
    if (p < 0 || p >= (int64_t)rank || !seen.insert(p).second)
      return std::nullopt;
    perm.push_back(p);
  }
  bool ident = true;
  for (unsigned i = 0; i < rank; ++i)
    if (perm[i] != (int64_t)i) { ident = false; break; }
  if (ident)
    return std::nullopt;
  Block &body = *gen.getBody();
  if (body.getOperations().size() != 1)
    return std::nullopt;
  auto yieldOp = dyn_cast<linalg::YieldOp>(&body.front());
  if (!yieldOp || yieldOp.getNumOperands() != 1)
    return std::nullopt;
  auto ba = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
  if (!ba || ba.getArgNumber() != 0)
    return std::nullopt;
  return perm;
}

AxisGrouping classifyAxes(const CollapsedGroupInfo &info) {
  AxisGrouping g;
  int rank = (int)info.collapsedAxes.size();
  g.axes.resize(rank);
  for (int i = 0; i < rank; ++i)
    g.axes[i].origPos = i;

  // --- transpose group (preserve template) -------------------------------
  std::optional<SmallVector<int64_t>> permOr;
  for (linalg::LinalgOp m : info.topoMembers)
    if ((permOr = transposePerm(m)))
      break;
  if (permOr) {
    const SmallVector<int64_t> &perm = *permOr; // perm[inPos] = iter dim
    DenseSet<int> classified;
    // 1. trailing axes with input-pos == output-pos → N (the output map is
    //    identity, so the axis at output-position i is iter dim i; at
    //    input-position i it is perm[i]).
    int i = rank - 1;
    for (; i >= 0 && perm[i] == i; --i) {
      g.axes[i].kind = AxisKind::N;
      g.axes[i].bindMultiCore = false;
      g.nAxes.insert(g.nAxes.begin(), i);
      classified.insert(i);
    }
    // 2. from the first differing position backward: input-side → X (unless
    //    already in Y), output-side → Y (unless already in X); once both are,
    //    dump the remaining (still-unclassified, in order) into Y and stop.
    auto inX = [&](int d) { return llvm::is_contained(g.xAxes, d); };
    auto inY = [&](int d) { return llvm::is_contained(g.yAxes, d); };
    for (; i >= 0; --i) {
      int inDim = (int)perm[i], outDim = i;
      if (!inY(inDim)) { g.xAxes.insert(g.xAxes.begin(), inDim); classified.insert(inDim); }
      if (!inX(outDim)) { g.yAxes.insert(g.yAxes.begin(), outDim); classified.insert(outDim); }
      if (inY(inDim) && inX(outDim)) {
        for (int j = i; j >= 0; --j)
          if (!classified.count(j)) {
            g.yAxes.insert(g.yAxes.begin(), j);
            classified.insert(j);
          }
        break;
      }
    }
    for (int d = 0; d < rank; ++d)
      if (!classified.count(d)) { g.yAxes.push_back(d); classified.insert(d); }
    for (int d : g.xAxes) { g.axes[d].kind = AxisKind::X; g.axes[d].bindMultiCore = false; }
    for (int d : g.yAxes) { g.axes[d].kind = AxisKind::Y; g.axes[d].bindMultiCore = true; }
    for (int d : g.xAxes) g.axesOrder.push_back(d);
    for (int d : g.yAxes) g.axesOrder.push_back(d);
    for (int d : g.nAxes) g.axesOrder.push_back(d);
    return g;
  }

  // --- elementwise / reduce ----------------------------------------------
  DenseSet<int> bcastSet(info.broadcastAxes.begin(), info.broadcastAxes.end());
  for (int i = 0; i < rank; ++i) {
    AxisClass &ax = g.axes[i];
    g.axesOrder.push_back(i);
    if (info.collapsedAxes[i].role == AxisRole::Reduction) {
      ax.kind = AxisKind::R;
      ax.isReduceSplit = true;
      g.rAxes.push_back(i);
    } else {
      ax.kind = AxisKind::Y;
      // A broadcast axis (some operand is constant along it) is an ordinary
      // parallel axis to the scheduler — only the lowering treats it specially
      // (replicating the projecting operand).  Keep it out of pickBlockAxis
      // (bindMultiCore=false) so the block-axis choice is unaffected.
      ax.isBroadcastConst = bcastSet.count(i);
      ax.bindMultiCore = !ax.isBroadcastConst;
      g.yAxes.push_back(i);
    }
  }
  return g;
}

Value castToIndex(OpBuilder &b, Location loc, Value v) {
  if (v.getType().isIndex()) return v;
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

Value getAxisExtentValue(OpBuilder &b, Location loc,
                          const CollapsedGroupInfo &info, int axisIdx) {
  int64_t staticSize = info.collapsedAxes[axisIdx].staticSize;
  if (staticSize != ShapedType::kDynamic)
    return b.create<arith::ConstantIndexOp>(loc, staticSize);

  // Dynamic: search operands of the post-collapse generics for a value that
  // maps to axisIdx and dominates the current insertion point.
  //
  // After multi-op collapse, operands are tensor.collapse_shape results, not
  // function arguments. We must trace through collapse_shape to its source
  // (which should be a block argument) and compute the product of original
  // dimensions to recover the collapsed axis extent.
  for (linalg::LinalgOp op : info.topoMembers) {
    auto maps     = op.getIndexingMapsArray();
    auto operands = op->getOperands();
    for (auto [operand, map] : llvm::zip(operands, maps)) {
      if (!isa<RankedTensorType>(operand.getType())) continue;
      for (auto [dimPos, expr] : llvm::enumerate(map.getResults())) {
        auto d = dyn_cast<AffineDimExpr>(expr);
        if (!d || (int)d.getPosition() != axisIdx) continue;

        // Prefer block arguments: they dominate everywhere.
        if (isa<BlockArgument>(operand))
          return b.create<tensor::DimOp>(loc, operand, (int64_t)dimPos);

        // If operand is tensor.collapse_shape whose src is a block argument,
        // recover the collapsed extent as the product of the original dims.
        if (auto colOp = operand.getDefiningOp<tensor::CollapseShapeOp>()) {
          Value src = colOp.getSrc();
          if (isa<BlockArgument>(src)) {
            auto grps = colOp.getReassociationIndices();
            if ((size_t)dimPos < grps.size()) {
              Value prod = b.create<arith::ConstantIndexOp>(loc, 1);
              for (int64_t origDim : grps[dimPos])
                prod = b.create<arith::MulIOp>(
                    loc, prod,
                    b.create<tensor::DimOp>(loc, src, origDim).getResult());
              return prod;
            }
          }
        }
        // Operand does not dominate the insertion point; keep searching.
      }
    }
  }
  llvm_unreachable("axis not found in any operand map");
}

} // namespace mlir::afir
