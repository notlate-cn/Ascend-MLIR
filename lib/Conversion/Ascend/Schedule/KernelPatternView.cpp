//===- KernelPatternView.cpp - Ascend schedule pattern view ------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/KernelPatternView.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <limits>
#include <utility>

using namespace mlir;

namespace mlir::afir::ascend::schedule {
namespace {

struct PatternGroup {
  KernelPatternView view;
  unsigned firstOrdinal = std::numeric_limits<unsigned>::max();
};

OpRole computeDominantRole(ArrayRef<PatternOpView> ops) {
  constexpr OpRole priority[] = {OpRole::Cube, OpRole::Reduction,
                                 OpRole::Vector, OpRole::Memory,
                                 OpRole::Unknown};
  for (OpRole role : priority) {
    if (llvm::any_of(ops, [&](const PatternOpView &opView) {
          return opView.role == role;
        }))
      return role;
  }
  return OpRole::Unknown;
}

LogicalResult finalizePatternGroup(PatternGroup &group) {
  KernelPatternView &view = group.view;
  llvm::sort(view.ops, [](const PatternOpView &lhs,
                          const PatternOpView &rhs) {
    return lhs.ordinal < rhs.ordinal;
  });

  view.primaryOps.clear();
  for (const PatternOpView &opView : view.ops) {
    if (opView.primary)
      view.primaryOps.push_back(opView.op);
  }

  if (view.primaryOps.empty()) {
    Operation *diagnosticOp = view.ops.empty() ? nullptr : view.ops.front().op;
    if (diagnosticOp)
      diagnosticOp->emitError()
          << "requires at least one " << kPrimaryAttr
          << " op in kernel pattern";
    return failure();
  }

  view.dominantRole = computeDominantRole(view.ops);
  return success();
}

} // namespace

FailureOr<SmallVector<KernelPatternView>>
buildKernelPatternViews(ModuleOp module) {
  SmallVector<PatternGroup> groups;
  llvm::StringMap<unsigned> groupIndex;
  unsigned nextOrdinal = 0;

  WalkResult walkResult = module.walk([&](func::FuncOp funcOp) {
    return funcOp.walk([&](linalg::LinalgOp linalgOp) {
      Operation *op = linalgOp.getOperation();
      auto kernelAttr = op->getAttrOfType<StringAttr>(kKernelAttr);
      if (!kernelAttr) {
        op->emitError() << "requires " << kKernelAttr;
        return WalkResult::interrupt();
      }

      StringRef kernelId = kernelAttr.getValue();
      auto groupIt = groupIndex.find(kernelId);
      if (groupIt == groupIndex.end()) {
        unsigned groupOrdinal = static_cast<unsigned>(groups.size());
        groupIt = groupIndex.try_emplace(kernelId, groupOrdinal).first;
        groups.push_back(PatternGroup{});
        PatternGroup &group = groups.back();
        group.view.kernelId = kernelId.str();
        group.firstOrdinal = nextOrdinal;
      }

      PatternGroup &group = groups[groupIt->second];
      auto roleAttr = op->getAttrOfType<StringAttr>(kOpRoleAttr);
      auto primaryAttr = op->getAttrOfType<BoolAttr>(kPrimaryAttr);
      group.view.ops.push_back(PatternOpView{
          op, nextOrdinal,
          roleAttr ? parseOpRole(roleAttr.getValue()) : OpRole::Unknown,
          primaryAttr && primaryAttr.getValue()});
      ++nextOrdinal;
      return WalkResult::advance();
    });
  });
  if (walkResult.wasInterrupted())
    return failure();

  for (PatternGroup &group : groups) {
    if (failed(finalizePatternGroup(group)))
      return failure();
  }

  llvm::sort(groups, [](const PatternGroup &lhs, const PatternGroup &rhs) {
    return lhs.firstOrdinal < rhs.firstOrdinal;
  });

  SmallVector<KernelPatternView> patterns;
  patterns.reserve(groups.size());
  for (PatternGroup &group : groups)
    patterns.push_back(std::move(group.view));
  return patterns;
}

const PatternOpView *selectDominantPrimaryOp(const KernelPatternView &pattern) {
  for (const PatternOpView &opView : pattern.ops) {
    if (opView.primary && opView.role == pattern.dominantRole)
      return &opView;
  }

  for (const PatternOpView &opView : pattern.ops) {
    if (opView.primary)
      return &opView;
  }

  return nullptr;
}

void printKernelPatternViews(ArrayRef<KernelPatternView> patterns,
                             llvm::raw_ostream &os) {
  for (const KernelPatternView &pattern : patterns) {
    os << "SchedulePatternView:\n";
    os << "  kernel = " << pattern.kernelId << "\n";
    os << "  ops = " << pattern.ops.size() << "\n";
    os << "  primary_ops = " << pattern.primaryOps.size() << "\n";
    os << "  dominant_role = " << stringifyOpRole(pattern.dominantRole)
       << "\n";
  }
}

} // namespace mlir::afir::ascend::schedule
