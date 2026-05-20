#include "AxisLattice.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::auto_fuse;
using namespace mlir::linalg;

namespace mlir::afir {

llvm::SmallVector<AxisInfo>
computeCanonicalAxes(llvm::ArrayRef<linalg::LinalgOp> members) {
  if (members.empty())
    return {};

  // Find the maximum number of loops across all members
  unsigned maxLoops = 0;
  for (auto op : members)
    maxLoops = std::max(maxLoops, (unsigned)op.getNumLoops());

  // For each axis position, compute the join
  // Reduction > Parallel > Absent
  // Absent + Absent = skip (don't emit this axis)
  enum class AxisClass { Absent, Parallel, Reduction };

  llvm::SmallVector<AxisInfo> result;
  for (unsigned a = 0; a < maxLoops; ++a) {
    AxisClass cls = AxisClass::Absent;
    int64_t staticSize = ShapedType::kDynamic;

    for (auto op : members) {
      if (a >= (unsigned)op.getNumLoops())
        continue; // absent for this member

      auto iterTypes = op.getIteratorTypesArray();
      if (iterTypes[a] == utils::IteratorType::reduction) {
        cls = AxisClass::Reduction;
        // Try to determine static size from loop bounds
        // (skip for now; leave as dynamic)
      } else {
        // parallel
        if (cls == AxisClass::Absent)
          cls = AxisClass::Parallel;
        // Try to get static size from the shaped operands
        for (auto operand : op.getOperation()->getOperands()) {
          auto type = dyn_cast<ShapedType>(operand.getType());
          if (!type)
            continue;
          // Match axis a using indexing maps
          // For simplicity, just look for the first operand that has this dim
          // In practice this needs indexing map analysis, but for the canonical
          // size we use the first static size we find.
          if (staticSize == ShapedType::kDynamic) {
            // We don't have easy access to which dim maps to axis a without
            // analyzing indexing maps. Leave as dynamic for now.
          }
        }
      }
    }

    if (cls == AxisClass::Absent)
      continue; // skip this axis position

    AxisRole role = (cls == AxisClass::Reduction) ? AxisRole::Reduction
                                                   : AxisRole::Parallel;
    std::string name = "d" + std::to_string(a);
    // We store the name as a StringRef pointing into a stable string.
    // Since AxisInfo.name is StringRef (not owning), we'd have a lifetime issue
    // with a local string. For the canonical axes we use an empty name since
    // the consumer (CanFuse) only looks at role.
    result.push_back(AxisInfo{"", staticSize, role});
  }
  return result;
}

llvm::SmallVector<AxisInfo>
computeCanonicalAxes(const FusionGroup &g) {
  return computeCanonicalAxes(llvm::ArrayRef<linalg::LinalgOp>(g.members));
}

llvm::DenseSet<Value>
collectBoundaryIn(llvm::ArrayRef<linalg::LinalgOp> members,
                  func::FuncOp func) {
  // Build a set of all SSA values produced by the member ops
  llvm::DenseSet<Operation *> memberOps;
  for (auto op : members)
    memberOps.insert(op.getOperation());

  // Build a set of all values defined by member ops
  llvm::DenseSet<Value> memberResults;
  for (auto op : members)
    for (auto result : op.getOperation()->getResults())
      memberResults.insert(result);

  llvm::DenseSet<Value> boundary;
  for (auto op : members) {
    for (auto operand : op.getOperation()->getOperands()) {
      // If not defined by a member op, it's a boundary input
      if (!memberResults.contains(operand))
        boundary.insert(operand);
    }
  }
  return boundary;
}

} // namespace mlir::afir
