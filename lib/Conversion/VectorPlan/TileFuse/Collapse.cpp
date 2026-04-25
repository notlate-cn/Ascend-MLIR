#include "Collapse.h"
#include "../GroupAnalysis/AxisLattice.h"
#include "Conversion/VectorPlan/GroupInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::vector_plan;
using namespace mlir::linalg;

namespace mlir::afir {

//===----------------------------------------------------------------------===//
// Static size extraction
//===----------------------------------------------------------------------===//

/// Get the static size for axis `axisIdx` from a linalg op's operands.
static int64_t getStaticAxisSize(LinalgOp op, int axisIdx) {
  for (auto [operand, map] :
       llvm::zip(op->getOperands(), op.getIndexingMapsArray())) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type) continue;
    for (auto [dimPos, expr] : llvm::enumerate(map.getResults())) {
      auto d = dyn_cast<AffineDimExpr>(expr);
      if (d && (int)d.getPosition() == axisIdx) {
        int64_t sz = type.getDimSize((int)dimPos);
        if (sz != ShapedType::kDynamic) return sz;
      }
    }
  }
  return ShapedType::kDynamic;
}

/// Fill static sizes into a vector of AxisInfos by querying the ops.
static void fillStaticSizes(SmallVector<AxisInfo> &axes,
                             ArrayRef<LinalgOp> members) {
  for (auto [i, ax] : llvm::enumerate(axes)) {
    if (ax.staticSize != ShapedType::kDynamic) continue;
    for (LinalgOp op : members) {
      int64_t sz = getStaticAxisSize(op, (int)i);
      if (sz != ShapedType::kDynamic) { ax.staticSize = sz; break; }
    }
  }
}

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

static SmallVector<SmallVector<int>>
findCandidateGroups(ArrayRef<AxisInfo> axes) {
  SmallVector<SmallVector<int>> groups;
  SmallVector<int> current;
  for (auto [i, ax] : llvm::enumerate(axes)) {
    if (!current.empty() && ax.role != axes[current.back()].role) {
      if (current.size() > 1) groups.push_back(current);
      current.clear();
    }
    current.push_back((int)i);
  }
  if (current.size() > 1) groups.push_back(current);
  return groups;
}

/// BCast = axes in G where some boundary input has partial G coverage
/// (at least one G-axis present AND the axis itself absent).
/// Case A (all G-axes absent) does NOT contribute.
static DenseSet<int>
computeBCast(ArrayRef<int> G, ArrayRef<LinalgOp> members,
             ArrayRef<Value> boundaryIn) {
  DenseSet<Value> bInSet(boundaryIn.begin(), boundaryIn.end());
  DenseSet<int> gSet(G.begin(), G.end());
  DenseSet<int> bcast;
  for (LinalgOp op : members) {
    auto inputs = op.getDpsInputs();
    auto maps   = op.getIndexingMapsArray();
    for (auto [operand, map] : llvm::zip(inputs, maps)) {
      if (!bInSet.count(operand)) continue;
      DenseSet<int> presentInG;
      for (AffineExpr expr : map.getResults())
        if (auto d = dyn_cast<AffineDimExpr>(expr))
          if (gSet.count((int)d.getPosition()))
            presentInG.insert((int)d.getPosition());
      if (presentInG.empty() || (int)presentInG.size() == (int)G.size())
        continue; // Case A or full coverage — no BCast contribution
      for (int g : G)
        if (!presentInG.count(g)) bcast.insert(g);
    }
  }
  return bcast;
}
static SmallVector<SmallVector<int>>
pruneBCastAxes(ArrayRef<int> G, const DenseSet<int> &bcast) {
  SmallVector<SmallVector<int>> out;
  SmallVector<int> current;
  for (int d : G) {
    if (bcast.count(d)) {
      if (current.size() >= 2) out.push_back(current);
      current.clear();
    } else {
      current.push_back(d);
    }
  }
  if (current.size() >= 2) out.push_back(current);
  return out;
}

enum class InputClass { A, B2, C };

static InputClass classifyInput(AffineMap map, ArrayRef<int> G) {
  DenseSet<int> gSet(G.begin(), G.end());
  DenseSet<int> presentInG;
  for (AffineExpr expr : map.getResults())
    if (auto d = dyn_cast<AffineDimExpr>(expr))
      if (gSet.count((int)d.getPosition()))
        presentInG.insert((int)d.getPosition());
  if (presentInG.empty()) return InputClass::A;
  // After pruning, all G-axes should be present (no partial); assert that.
  assert((int)presentInG.size() == (int)G.size() &&
         "B1 must be pruned before classifyInput");
  // Check if G-axes appear in order in the map results.
  SmallVector<int> positions;
  for (int g : G)
    for (auto [i, expr] : llvm::enumerate(map.getResults()))
      if (auto d = dyn_cast<AffineDimExpr>(expr))
        if ((int)d.getPosition() == g) { positions.push_back((int)i); break; }
  for (int k = 1; k < (int)positions.size(); ++k)
    if (positions[k] != positions[k-1]+1) return InputClass::B2;
  return InputClass::C;
}

static bool hasAnyB2(ArrayRef<int> G, ArrayRef<LinalgOp> members,
                     ArrayRef<Value> boundaryIn) {
  DenseSet<Value> bInSet(boundaryIn.begin(), boundaryIn.end());
  for (LinalgOp op : members) {
    auto inputs = op.getDpsInputs();
    auto maps   = op.getIndexingMapsArray();
    for (auto [operand, map] : llvm::zip(inputs, maps))
      if (bInSet.count(operand) && classifyInput(map, G) == InputClass::B2)
        return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// axisMap / collapsedAxes construction
//===----------------------------------------------------------------------===//

static void buildAxisMap(ArrayRef<AxisInfo> canonAxes,
                          ArrayRef<int> collapseGroup,
                          SmallVector<int> &axisMapOut,
                          SmallVector<AxisInfo> &collapsedAxesOut) {
  DenseSet<int> cgSet(collapseGroup.begin(), collapseGroup.end());
  int numOrig = (int)canonAxes.size();
  axisMapOut.resize(numOrig);
  int postIdx = 0;
  for (int origIdx = 0; origIdx < numOrig; ++origIdx) {
    if (!cgSet.count(origIdx)) {
      axisMapOut[origIdx] = postIdx++;
      collapsedAxesOut.push_back(canonAxes[origIdx]);
    } else if (origIdx == collapseGroup.front()) {
      int64_t prod = 1;
      for (int g : collapseGroup) {
        int64_t s = canonAxes[g].staticSize;
        prod = (s == ShapedType::kDynamic || prod == ShapedType::kDynamic)
                   ? ShapedType::kDynamic : prod * s;
      }
      axisMapOut[origIdx] = postIdx++;
      collapsedAxesOut.push_back(AxisInfo{"", prod, canonAxes[origIdx].role});
    } else {
      axisMapOut[origIdx] = -1; // absorbed
    }
  }
}

//===----------------------------------------------------------------------===//
// IR transformation helpers
//===----------------------------------------------------------------------===//

static AffineMap rewriteMap(AffineMap map, ArrayRef<int> axisMap,
                              int numPostDims) {
  SmallVector<AffineExpr> newResults;
  DenseSet<int> seen;
  for (AffineExpr expr : map.getResults()) {
    auto d = dyn_cast<AffineDimExpr>(expr);
    if (!d) { newResults.push_back(expr); continue; }
    int origIdx = (int)d.getPosition();
    int postIdx = (origIdx < (int)axisMap.size()) ? axisMap[origIdx] : origIdx;
    if (postIdx < 0) continue; // absorbed
    if (!seen.count(postIdx)) {
      newResults.push_back(getAffineDimExpr(postIdx, map.getContext()));
      seen.insert(postIdx);
    }
  }
  return AffineMap::get(numPostDims, 0, newResults, map.getContext());
}

/// Build tensor.collapse_shape / expand_shape reassociation.
/// Groups consecutive map result positions that share the same post-collapse axis.
static SmallVector<ReassociationIndices>
buildReassociation(AffineMap map, ArrayRef<int> axisMap) {
  SmallVector<ReassociationIndices> reassoc;
  int prevPost = -99;
  for (auto [i, expr] : llvm::enumerate(map.getResults())) {
    auto d = dyn_cast<AffineDimExpr>(expr);
    if (!d) continue;
    int origIdx = (int)d.getPosition();
    int postIdx = (origIdx < (int)axisMap.size()) ? axisMap[origIdx] : origIdx;
    if (postIdx < 0) {
      // Absorbed: append to the last group (same post-axis as leader).
      assert(!reassoc.empty());
      reassoc.back().push_back((int64_t)i);
    } else if (postIdx == prevPost) {
      reassoc.back().push_back((int64_t)i);
    } else {
      reassoc.push_back({(int64_t)i});
      prevPost = postIdx;
    }
  }
  return reassoc;
}

static RankedTensorType collapseType(RankedTensorType orig,
                                      ArrayRef<ReassociationIndices> reassoc) {
  SmallVector<int64_t> shape;
  for (auto &grp : reassoc) {
    int64_t sz = 1;
    for (int64_t dimPos : grp) {
      int64_t d = orig.getDimSize((int)dimPos);
      sz = (d == ShapedType::kDynamic || sz == ShapedType::kDynamic)
               ? ShapedType::kDynamic : sz * d;
    }
    shape.push_back(sz);
  }
  return RankedTensorType::get(shape, orig.getElementType());
}

static Value emitCollapseIfC(OpBuilder &builder, Location loc, Value operand,
                               AffineMap map, ArrayRef<int> collapseGroup,
                               ArrayRef<int> axisMap) {
  if (classifyInput(map, collapseGroup) != InputClass::C) return operand;
  auto reassoc = buildReassociation(map, axisMap);
  auto newType = collapseType(cast<RankedTensorType>(operand.getType()), reassoc);
  return builder.create<tensor::CollapseShapeOp>(loc, newType, operand, reassoc);
}

static void applyIRTransform(OpBuilder &builder, GenericOp lop,
                               ArrayRef<int> collapseGroup,
                               ArrayRef<int> axisMap, int numPostDims) {
  Location loc = lop.getLoc();
  builder.setInsertionPoint(lop);

  SmallVector<Value> newInputs;
  SmallVector<AffineMap> newMaps;
  auto allMaps    = lop.getIndexingMapsArray();
  int numInputs   = lop.getNumDpsInputs();
  auto inputVals  = lop.getDpsInputs();
  for (auto [operand, map] : llvm::zip(inputVals, llvm::ArrayRef(allMaps).take_front(numInputs))) {
    newInputs.push_back(emitCollapseIfC(builder, loc, operand, map,
                                         collapseGroup, axisMap));
    newMaps.push_back(rewriteMap(map, axisMap, numPostDims));
  }

  SmallVector<Value> newOuts;
  SmallVector<AffineMap> outMaps;
  auto outMapSlice = llvm::ArrayRef(allMaps).drop_front(numInputs);
  auto outsRange   = lop.getDpsInits();
  SmallVector<Value> outsVec(outsRange.begin(), outsRange.end());
  for (auto [outVal, outMap] : llvm::zip(outsVec, outMapSlice)) {
    newOuts.push_back(emitCollapseIfC(builder, loc, outVal, outMap,
                                       collapseGroup, axisMap));
    AffineMap rewritten = rewriteMap(outMap, axisMap, numPostDims);
    newMaps.push_back(rewritten);
    outMaps.push_back(outMap); // keep original for expand_shape
  }

  // Build new iterator_types dropping absorbed axes.
  SmallVector<utils::IteratorType> newIterTypes;
  for (auto [origIdx, it] : llvm::enumerate(lop.getIteratorTypesArray())) {
    int pm = ((int)origIdx < (int)axisMap.size()) ? axisMap[origIdx] : (int)origIdx;
    if (pm >= 0) newIterTypes.push_back(it);
  }

  SmallVector<Type> resultTypes;
  for (Value out : newOuts) resultTypes.push_back(out.getType());

  auto newGeneric = builder.create<GenericOp>(
      loc, resultTypes, newInputs, newOuts, newMaps, newIterTypes);

  // Clone body. The GenericOp was created without a bodyBuild, so its region is
  // empty (no blocks). cloneRegionBefore inserts at the given iterator position.
  builder.cloneRegionBefore(lop.getRegion(), newGeneric.getRegion(),
                             newGeneric.getRegion().begin());

  // Replace old results with expand_shape → restore original return types.
  builder.setInsertionPointAfter(newGeneric);
  for (auto [oldRes, newRes, outMap] :
       llvm::zip(lop.getResults(), newGeneric.getResults(), outMaps)) {
    auto reassoc = buildReassociation(outMap, axisMap);
    auto origType = cast<RankedTensorType>(oldRes.getType());
    Value expanded = builder.create<tensor::ExpandShapeOp>(
        loc, origType, newRes, reassoc);
    oldRes.replaceAllUsesWith(expanded);
  }
  lop.erase();
}

//===----------------------------------------------------------------------===//
// collapseGroup — public entry point
//===----------------------------------------------------------------------===//

CollapsedGroupInfo collapseGroup(OpBuilder &builder, func::FuncOp func) {
  SmallVector<LinalgOp> members;
  func.walk([&](LinalgOp op) { members.push_back(op); });

  CollapsedGroupInfo result;
  result.kind = GroupInfo::Kind::Vector;
  if (members.empty()) return result;

  auto canonAxes = computeCanonicalAxes(llvm::ArrayRef(members));
  // Fill in static sizes from op shapes (computeCanonicalAxes leaves them kDynamic).
  fillStaticSizes(canonAxes, members);

  result.canonicalAxes = canonAxes;
  result.topoMembers   = members;

  SmallVector<Value> boundaryIn;
  for (Value arg : func.getArguments())
    if (!arg.use_empty()) boundaryIn.push_back(arg);
  result.boundaryIn = boundaryIn;

  auto retOp = cast<func::ReturnOp>(func.getBody().front().getTerminator());
  result.boundaryOut = SmallVector<Value>(retOp.getOperands());

  // Identity default.
  result.collapsedAxes = canonAxes;
  for (int i = 0; i < (int)canonAxes.size(); ++i) result.axisMap.push_back(i);

  auto candidates = findCandidateGroups(canonAxes);
  SmallVector<int> chosenGroup;
  DenseSet<int> chosenBCast;
  for (auto &cand : candidates) {
    auto bcast  = computeBCast(cand, members, boundaryIn);
    auto pruned = pruneBCastAxes(cand, bcast);
    if (!pruned.empty()) { chosenGroup = pruned.front(); chosenBCast = bcast; break; }
  }
  if (chosenGroup.empty()) return result;

  if (hasAnyB2(chosenGroup, members, boundaryIn)) {
    result.hasB2 = true; result.noCollapse = true; return result;
  }

  SmallVector<int> newAxisMap;
  SmallVector<AxisInfo> newCollapsedAxes;
  buildAxisMap(canonAxes, chosenGroup, newAxisMap, newCollapsedAxes);
  result.axisMap       = newAxisMap;
  result.collapsedAxes = newCollapsedAxes;
  int numPost = (int)newCollapsedAxes.size();

  for (int origIdx : chosenBCast) {
    int pm = (origIdx < (int)newAxisMap.size()) ? newAxisMap[origIdx] : origIdx;
    if (pm >= 0 && !llvm::is_contained(result.broadcastAxes, pm))
      result.broadcastAxes.push_back(pm);
  }
  llvm::sort(result.broadcastAxes);

  if (members.size() == 1)
    if (auto lop = dyn_cast<GenericOp>(members.front().getOperation()))
      applyIRTransform(builder, lop, chosenGroup, newAxisMap, numPost);

  return result;
}

} // namespace mlir::afir
