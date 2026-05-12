#include "Collapse.h"
#include "TileFuseUtils.h"
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
      if (current.size() >= 2) groups.push_back(current);
      current.clear();
    }
    current.push_back((int)i);
  }
  if (current.size() >= 2) groups.push_back(current);
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
  if ((int)presentInG.size() != (int)G.size())
    return InputClass::B2;  // partial coverage (B1) → treat as B2 for safety
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

// Multi-op IR collapse: apply tensor.collapse_shape to all boundary inputs and
// intermediate empties, rewrite each generic's maps, and restore shapes with
// tensor.expand_shape at boundary outputs. Runs in topo order so each op's
// inputs are already in the collapsed domain when we reach it.
static void applyMultiOpIRTransform(OpBuilder &builder,
                                     ArrayRef<LinalgOp> members,
                                     ArrayRef<int> axisMap, int numPost) {
  if (members.size() < 2) return;
  Location loc = members.front()->getLoc();

  DenseSet<Operation *> memberSet;
  for (LinalgOp m : members)
    memberSet.insert(m.getOperation());

  // Collapse `v` using its indexing map. Returns v if already at post-collapse
  // rank (already processed by a previous op in the chain) or if the type is
  // unchanged (Class-A operand with no G-axes).
  auto resolveCollapsed = [&](Value v, AffineMap vMap) -> Value {
    auto mrt = dyn_cast<RankedTensorType>(v.getType());
    if (!mrt || mrt.getRank() == numPost)
      return v;
    auto reassoc = buildReassociation(vMap, axisMap);
    auto colType = collapseType(mrt, reassoc);
    if (colType == mrt)
      return v; // Class A: no G-axes, no shape change
    return builder.create<tensor::CollapseShapeOp>(loc, colType, v, reassoc);
  };

  for (LinalgOp op : members) {
    builder.setInsertionPoint(op);
    auto maps    = op.getIndexingMapsArray();
    auto operands = op->getOperands();
    int numIns  = op.getNumDpsInputs();
    int numOuts = op.getNumDpsInits();

    // Capture original output values before collapse for shape queries in expand_shape.
    SmallVector<Value> origOuts;
    for (int i = 0; i < numOuts; ++i)
      origOuts.push_back(operands[numIns + i]);

    SmallVector<Value> newInputs, newOuts;
    for (int i = 0; i < numIns; ++i)
      newInputs.push_back(resolveCollapsed(operands[i], maps[i]));
    for (int i = 0; i < numOuts; ++i)
      newOuts.push_back(resolveCollapsed(origOuts[i], maps[numIns + i]));

    SmallVector<AffineMap> newMaps;
    for (AffineMap m : maps)
      newMaps.push_back(rewriteMap(m, axisMap, numPost));

    SmallVector<utils::IteratorType> newIters;
    for (auto [i, it] : llvm::enumerate(op.getIteratorTypesArray())) {
      int pm = ((int)i < (int)axisMap.size()) ? axisMap[i] : (int)i;
      if (pm >= 0) newIters.push_back(it);
    }

    SmallVector<Type> resultTypes;
    for (Value out : newOuts) resultTypes.push_back(out.getType());
    auto newGeneric = builder.create<GenericOp>(
        loc, resultTypes, newInputs, newOuts, newMaps, newIters);
    builder.cloneRegionBefore(op->getRegion(0), newGeneric.getRegion(),
                               newGeneric.getRegion().begin());

    SmallVector<AffineMap> outMaps(maps.begin() + numIns, maps.end());
    builder.setInsertionPointAfter(newGeneric);
    int resIdx = 0;
    for (auto [oldRes, newRes, outMap] :
         llvm::zip(op->getResults(), newGeneric.getResults(), outMaps)) {
      // In-group uses consume the collapsed result directly.
      // Out-group uses (boundary outputs) get expand_shape to restore shape.
      SmallVector<OpOperand *> inUses, outUses;
      for (OpOperand &use : oldRes.getUses()) {
        if (memberSet.count(use.getOwner()))
          inUses.push_back(&use);
        else
          outUses.push_back(&use);
      }
      for (OpOperand *use : inUses)
        use->set(newRes);
      if (!outUses.empty()) {
        Value replacement = newRes;
        if (newRes.getType() != oldRes.getType()) {
          auto reassoc = buildReassociation(outMap, axisMap);
          // Build dynamic-dim operands from the original (pre-collapse) output.
          Value origOut = origOuts[resIdx];
          auto origOutTy = cast<RankedTensorType>(origOut.getType());
          auto resultTy = cast<RankedTensorType>(oldRes.getType());
          SmallVector<int64_t> staticShape;
          SmallVector<Value> dynamicDims;
          for (int d = 0; d < origOutTy.getRank(); ++d) {
            if (!origOutTy.isDynamicDim(d)) {
              staticShape.push_back(origOutTy.getDimSize(d));
            } else {
              staticShape.push_back(ShapedType::kDynamic);
              dynamicDims.push_back(
                  builder.create<tensor::DimOp>(loc, origOut, (int64_t)d).getResult());
            }
          }
          replacement = builder.create<tensor::ExpandShapeOp>(
              loc, resultTy, newRes,
              getReassociationIndicesAttribute(builder, reassoc),
              dynamicDims, staticShape);
        }
        for (OpOperand *use : outUses)
          use->set(replacement);
      }
      ++resIdx;
    }
    op.erase();
  }
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
  // Skip expand_shape for Class-A outputs (no G-axes collapsed) where
  // the new result type already matches the original type.
  builder.setInsertionPointAfter(newGeneric);
  int outIdx = 0;
  for (auto [oldRes, newRes, outMap] :
       llvm::zip(lop.getResults(), newGeneric.getResults(), outMaps)) {
    if (newRes.getType() == oldRes.getType()) {
      oldRes.replaceAllUsesWith(newRes);
      ++outIdx;
      continue;
    }
    auto reassoc = buildReassociation(outMap, axisMap);
    Value origOut = outsVec[outIdx];
    auto origOutTy = cast<RankedTensorType>(origOut.getType());
    SmallVector<int64_t> staticShape;
    SmallVector<Value> dynamicDims;
    for (int d = 0; d < origOutTy.getRank(); ++d) {
      if (!origOutTy.isDynamicDim(d)) {
        staticShape.push_back(origOutTy.getDimSize(d));
      } else {
        staticShape.push_back(ShapedType::kDynamic);
        dynamicDims.push_back(
            builder.create<tensor::DimOp>(loc, origOut, (int64_t)d).getResult());
      }
    }
    Value expanded = builder.create<tensor::ExpandShapeOp>(
        loc, cast<RankedTensorType>(oldRes.getType()), newRes,
        getReassociationIndicesAttribute(builder, reassoc),
        dynamicDims, staticShape);
    oldRes.replaceAllUsesWith(expanded);
    ++outIdx;
  }
  lop.erase();
}

//===----------------------------------------------------------------------===//
// collapseGroup — public entry point
//===----------------------------------------------------------------------===//

static CollapsedGroupInfo collapseGroupImpl(OpBuilder &builder,
                                            func::FuncOp func) {
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
    if (!pruned.empty()) {
      chosenGroup = pruned.front();  // v1: collapse only the first valid sub-group
      chosenBCast = bcast;
      break;
    }
  }
  if (chosenGroup.empty()) {
    // No collapsable sub-group (e.g. broadcast axes split every candidate group
    // to sub-groups of size < 2).  Still record broadcast axes so TilePlanGen
    // emits BCAST loops instead of treating them as regular parallel axes.
    for (auto &cand : candidates) {
      auto bcast = computeBCast(cand, members, boundaryIn);
      for (int b : bcast)
        if (!llvm::is_contained(result.broadcastAxes, b))
          result.broadcastAxes.push_back(b);
    }
    llvm::sort(result.broadcastAxes);
    return result;
  }

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

  // IR transformation: collapse the iteration space in the actual IR.
  if (members.size() == 1) {
    if (auto lop = dyn_cast<GenericOp>(members.front().getOperation()))
      applyIRTransform(builder, lop, chosenGroup, newAxisMap, numPost);
  } else {
    applyMultiOpIRTransform(builder, members, newAxisMap, numPost);
  }

  // Re-sync topoMembers and boundaryOut so Phase 3 sees valid ops/values.
  result.topoMembers.clear();
  func.walk([&](LinalgOp op) { result.topoMembers.push_back(op); });
  result.boundaryOut = SmallVector<Value>(retOp.getOperands());

  return result;
}

CollapsedGroupInfo collapseGroup(OpBuilder &builder, func::FuncOp func) {
  CollapsedGroupInfo result = collapseGroupImpl(builder, func);
  // ≈ AF GenTilingGroup/NormGroup — classify the post-collapse iteration axes
  // once here; TilePlanGen consumes result.grouping directly.
  result.grouping = classifyAxes(result);
  return result;
}

} // namespace mlir::afir
