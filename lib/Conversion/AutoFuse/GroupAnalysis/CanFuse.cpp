#include "CanFuse.h"
#include "AxisLattice.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

using namespace mlir;
using namespace mlir::auto_fuse;
using namespace mlir::linalg;

namespace mlir::afir {

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

/// Return all SSA results produced by members of a group.
static llvm::DenseSet<Value> getMemberResults(const FusionGroup &g) {
  llvm::DenseSet<Value> results;
  for (auto op : g.members)
    for (auto res : op.getOperation()->getResults())
      results.insert(res);
  return results;
}

/// Return all SSA operands consumed by members of a group.
static llvm::DenseSet<Value> getMemberOperands(const FusionGroup &g) {
  llvm::DenseSet<Value> operands;
  for (auto op : g.members)
    for (auto operand : op.getOperation()->getOperands())
      operands.insert(operand);
  return operands;
}

/// Return "link tensors": values produced by g1 that are consumed by g2.
static llvm::SmallVector<Value> getLinkTensors(const FusionGroup &g1,
                                                const FusionGroup &g2) {
  auto g1Results = getMemberResults(g1);
  auto g2Operands = getMemberOperands(g2);

  llvm::SmallVector<Value> links;
  for (auto v : g1Results)
    if (g2Operands.contains(v))
      links.push_back(v);
  return links;
}

/// Check if g1 produces a value consumed by g2 or vice versa.
static bool hasSSAEdge(const FusionGroup &g1, const FusionGroup &g2) {
  return !getLinkTensors(g1, g2).empty() ||
         !getLinkTensors(g2, g1).empty();
}

//===----------------------------------------------------------------------===//
// getFusionKind
//===----------------------------------------------------------------------===//

FusionKind getFusionKind(const FusionGroup &g1, const FusionGroup &g2) {
  // Vertical: SSA edge exists between the two groups
  if (hasSSAEdge(g1, g2))
    return FusionKind::Vertical;

  // Horizontal: share at least one boundaryIn *tensor* value.  A shared scalar
  // (e.g. the constant fill value feeding linalg.fill) carries no shared tensor
  // data and yields no fusion benefit, so it must not count.  Otherwise every
  // linalg.fill in a network shares the zero constant and collapses into one
  // scattered, cyclic pseudo-group that crashes --auto-fuse-group-outline.
  for (auto v : g1.boundaryIn)
    if (isa<ShapedType>(v.getType()) && g2.boundaryIn.contains(v))
      return FusionKind::Horizontal;

  return FusionKind::None;
}

//===----------------------------------------------------------------------===//
// Cycle check helper
//===----------------------------------------------------------------------===//

/// Check if merging g1 and g2 would create a cycle in the group dependency
/// graph.
///
/// Dependency edges are GLUE-AWARE: a value's "source groups" are traced
/// transitively back through non-group ops (tensor.collapse_shape /
/// expand_shape / extract_slice, ...).  A linalg-only view would miss edges
/// routed through reshape glue (ubiquitous in real networks) and approve merges
/// that produce cyclic, non-outlinable groups — the transformer-encoder crash.
///
/// We unify g1/g2 to a sentinel "merged" node and report a cycle iff the merged
/// node can reach itself via >= 1 intermediate group (direct g1<->g2 edges
/// become self-loops and are excluded).
bool wouldCreateCycle(const FusionGroup &g1, const FusionGroup &g2,
                      llvm::ArrayRef<FusionGroup> allGroups) {
  // Sentinel group id for the hypothetical merged group (distinct from any real
  // id and from the -1 "removed" marker).
  constexpr int32_t kMerged = -2;

  // Map every group-member op -> its group id, unifying g1/g2 to kMerged.
  llvm::DenseMap<Operation *, int32_t> opGroup;
  for (const auto &g : allGroups) {
    if (g.id < 0)
      continue;
    int32_t id = (g.id == g1.id || g.id == g2.id) ? kMerged : g.id;
    for (auto op : g.members)
      opGroup[op.getOperation()] = id;
  }

  // sourceGroups(v): group ids that transitively PRODUCE v, tracing back
  // through non-group glue ops.  Memoized; SSA is acyclic so it terminates.
  llvm::DenseMap<Value, llvm::SmallDenseSet<int32_t, 2>> memo;
  std::function<llvm::SmallDenseSet<int32_t, 2>(Value)> sourceGroups =
      [&](Value v) -> llvm::SmallDenseSet<int32_t, 2> {
    auto it = memo.find(v);
    if (it != memo.end())
      return it->second;
    llvm::SmallDenseSet<int32_t, 2> out;
    if (Operation *def = v.getDefiningOp()) {
      auto git = opGroup.find(def);
      if (git != opGroup.end()) {
        out.insert(git->second); // produced directly by a group member
      } else {
        for (Value operand : def->getOperands())
          for (int32_t s : sourceGroups(operand))
            out.insert(s);
      }
    }
    memo[v] = out; // store after recursion (return-by-value keeps refs safe)
    return out;
  };

  // Build glue-aware group dependency edges (producer -> consumer).
  llvm::DenseMap<int32_t, llvm::DenseSet<int32_t>> edges;
  for (const auto &g : allGroups) {
    if (g.id < 0)
      continue;
    int32_t cons = (g.id == g1.id || g.id == g2.id) ? kMerged : g.id;
    for (auto op : g.members)
      for (Value operand : op.getOperation()->getOperands())
        for (int32_t src : sourceGroups(operand))
          if (src != cons) // skip self-edges (incl. direct g1<->g2)
            edges[src].insert(cons);
  }

  // The merge creates a cycle iff the merged node can reach itself.
  llvm::DenseSet<int32_t> seen;
  llvm::SmallVector<int32_t> work{kMerged};
  while (!work.empty()) {
    int32_t cur = work.pop_back_val();
    for (int32_t nxt : edges.lookup(cur)) {
      if (nxt == kMerged)
        return true;
      if (seen.insert(nxt).second)
        work.push_back(nxt);
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// canFuseVector
//===----------------------------------------------------------------------===//

bool canFuseVector(const FusionGroup &g1, const FusionGroup &g2,
                   FusionKind kind,
                   llvm::ArrayRef<FusionGroup> allGroups,
                   const CanFuseOptions &opts) {
  // Rule 1: kind must not be None
  if (kind == FusionKind::None)
    return false;

  // Rule 2: Cycle check
  if (wouldCreateCycle(g1, g2, allGroups))
    return false;

  // Rule 3: Axis compatibility - merged canonical axes must have at least one
  // Parallel axis
  llvm::SmallVector<linalg::LinalgOp> allMembers;
  allMembers.append(g1.members.begin(), g1.members.end());
  allMembers.append(g2.members.begin(), g2.members.end());
  auto mergedAxes = computeCanonicalAxes(allMembers);

  bool hasParallel = llvm::any_of(mergedAxes, [](const AxisInfo &ax) {
    return ax.role == AxisRole::Parallel;
  });
  if (!hasParallel)
    return false;

  // Rule 4 (Vertical only): Link tensor fan-out check
  // If g1 produces T used by g2, all linalg users of T must be in g1 or g2.
  if (kind == FusionKind::Vertical) {
    // Build set of all member ops
    llvm::DenseSet<Operation *> allMemberOps;
    for (auto op : g1.members) allMemberOps.insert(op.getOperation());
    for (auto op : g2.members) allMemberOps.insert(op.getOperation());

    auto checkLinkFanOut = [&](const FusionGroup &producer,
                               const FusionGroup &consumer) -> bool {
      auto links = getLinkTensors(producer, consumer);
      for (auto v : links) {
        for (auto *user : v.getUsers()) {
          if (isa<linalg::LinalgOp>(user) && !allMemberOps.contains(user))
            return false;
        }
      }
      return true;
    };

    if (!checkLinkFanOut(g1, g2) || !checkLinkFanOut(g2, g1))
      return false;
  }

  // Rule 5: DPS init transparent (linalg.fill allowed; no extra check needed)

  // Rule 6: Epilogue reduction dependency
  // Find reduction ops in merged group
  llvm::DenseSet<Value> reduceResults;
  for (auto op : allMembers) {
    if (isa<linalg::ReduceOp>(op.getOperation())) {
      for (auto res : op.getOperation()->getResults())
        reduceResults.insert(res);
    }
    // Also check linalg.generic with reduction iterators
    bool hasReduction = llvm::any_of(
        op.getIteratorTypesArray(),
        [](utils::IteratorType t) { return t == utils::IteratorType::reduction; });
    if (hasReduction) {
      for (auto res : op.getOperation()->getResults())
        reduceResults.insert(res);
    }
  }

  if (!reduceResults.empty()) {
    // Find epilogue ops: all-parallel iterator_types that consume a reduce result
    int32_t epilogueCount = 0;
    for (auto op : allMembers) {
      bool allParallel = llvm::all_of(
          op.getIteratorTypesArray(),
          [](utils::IteratorType t) {
            return t == utils::IteratorType::parallel;
          });
      if (!allParallel)
        continue;

      bool consumesReduceResult = llvm::any_of(
          op.getOperation()->getOperands(),
          [&](Value v) { return reduceResults.contains(v); });
      if (consumesReduceResult)
        ++epilogueCount;
    }

    if (epilogueCount > opts.maxReduceEpilogueOps)
      return false;
  }

  // Horizontal extra inputs check (H2)
  if (kind == FusionKind::Horizontal) {
    // Compute new boundary inputs that would be added by the merge
    // A boundary input is "new" if it's in g2.boundaryIn but not in g1.boundaryIn
    // (and vice versa) after removing the link tensors
    int32_t extraInputs = 0;
    for (auto v : g2.boundaryIn)
      if (!g1.boundaryIn.contains(v))
        ++extraInputs;
    for (auto v : g1.boundaryIn)
      if (!g2.boundaryIn.contains(v))
        ++extraInputs;

    if (extraInputs > opts.maxHorizontalExtraInputs)
      return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// canFuseCubeEpilogue
//===----------------------------------------------------------------------===//

bool canFuseCubeEpilogue(const FusionGroup &cube, const FusionGroup &vec,
                         llvm::ArrayRef<FusionGroup> allGroups) {
  // E0: cycle check (mirrors canFuseVector Rule 2).  Without this, a cube+vec
  // epilogue merge whose two sides also have an intermediate group between them
  // (parallel dependency path) creates a cyclic group that cannot be outlined.
  if (wouldCreateCycle(cube, vec, allGroups))
    return false;

  // E1: all vec members have only parallel iterators
  for (auto op : vec.members) {
    bool allParallel = llvm::all_of(
        op.getIteratorTypesArray(),
        [](utils::IteratorType t) {
          return t == utils::IteratorType::parallel;
        });
    if (!allParallel)
      return false;
  }

  // E2: skip for v1 (no DPS aliasing check)

  // E3: simplified to E1 check (all parallel, already verified)

  // E4: skip for v1

  // E5: matmul result is used by exactly one VectorGroup
  // Find matmul ops in cube group and check their results
  for (auto op : cube.members) {
    if (!isa<linalg::MatmulOp>(op.getOperation()) &&
        !op.getOperation()->hasAttr("ascendc.unit"))
      continue;

    for (auto res : op.getOperation()->getResults()) {
      // Count linalg users not in cube
      llvm::DenseSet<Operation *> cubeOps;
      for (auto m : cube.members) cubeOps.insert(m.getOperation());

      int vecGroupUsers = 0;
      for (auto *user : res.getUsers()) {
        if (!isa<linalg::LinalgOp>(user)) continue;
        if (cubeOps.contains(user)) continue;

        // Check which group this user belongs to
        for (const auto &g : allGroups) {
          if (g.id < 0) continue;
          if (g.containsOp(user)) {
            ++vecGroupUsers;
            break;
          }
        }
      }

      if (vecGroupUsers > 1)
        return false;
    }
  }

  // Check that vec is not all shape ops (no pure reshapes)
  bool hasRealCompute = llvm::any_of(vec.members, [](linalg::LinalgOp op) {
    return !isa<linalg::TransposeOp>(op.getOperation()) &&
           !isa<linalg::BroadcastOp>(op.getOperation());
  });
  if (!hasRealCompute)
    return false;

  return true;
}

//===----------------------------------------------------------------------===//
// computeScore
//===----------------------------------------------------------------------===//

int64_t computeScore(const FusionGroup &g1, const FusionGroup &g2) {
  int64_t score = 0;

  auto accumLinks = [&](const FusionGroup &producer,
                        const FusionGroup &consumer) {
    auto links = getLinkTensors(producer, consumer);
    for (auto v : links) {
      auto type = dyn_cast<ShapedType>(v.getType());
      if (!type) continue;

      int64_t bytes = 1;
      for (auto dim : type.getShape()) {
        if (dim == ShapedType::kDynamic) {
          bytes = 1; // dynamic: use 1 as proxy
          break;
        }
        bytes *= dim;
      }
      // Multiply by element byte size
      auto elemType = type.getElementType();
      int64_t elemBytes = 1;
      if (elemType.isF16() || elemType.isBF16() || elemType.isInteger(16))
        elemBytes = 2;
      else if (elemType.isF32() || elemType.isInteger(32))
        elemBytes = 4;
      else if (elemType.isF64() || elemType.isInteger(64))
        elemBytes = 8;
      score += bytes * elemBytes;
    }
  };

  accumLinks(g1, g2);
  accumLinks(g2, g1);
  return score;
}

} // namespace mlir::afir
