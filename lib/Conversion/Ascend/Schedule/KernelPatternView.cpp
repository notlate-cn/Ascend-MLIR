//===- KernelPatternView.cpp - Ascend schedule pattern view ------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "KernelPatternView.h"

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

void appendUniqueString(SmallVectorImpl<std::string> &values,
                        StringRef value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value.str());
}

void appendTemplateFamilies(Operation *op,
                            SmallVectorImpl<std::string> &families) {
  auto familyAttrs = op->getAttrOfType<ArrayAttr>(kKernelizeTemplateFamiliesAttr);
  if (!familyAttrs)
    return;
  for (Attribute attr : familyAttrs) {
    auto family = dyn_cast<StringAttr>(attr);
    if (!family)
      continue;
    appendUniqueString(families, family.getValue());
  }
}

LogicalResult mergeHandwrittenKind(Operation *op, KernelPatternView &view) {
  auto kind = op->getAttrOfType<StringAttr>(kKernelizeHandwrittenKindAttr);
  if (!kind)
    return success();
  if (view.handwrittenKind.empty()) {
    view.handwrittenKind = kind.getValue().str();
    return success();
  }
  if (view.handwrittenKind != kind.getValue()) {
    op->emitError() << "conflicting " << kKernelizeHandwrittenKindAttr
                    << " values inside kernel pattern";
    return failure();
  }
  return success();
}

bool hasReductionIterator(linalg::LinalgOp linalgOp) {
  return llvm::is_contained(linalgOp.getIteratorTypesArray(),
                            utils::IteratorType::reduction);
}

bool isReductionInitHelper(linalg::LinalgOp linalgOp) {
  auto fillOp = dyn_cast<linalg::FillOp>(linalgOp.getOperation());
  if (!fillOp)
    return false;

  bool hasReductionInitUser = false;
  for (Value result : fillOp->getResults()) {
    for (Operation *user : result.getUsers()) {
      auto userLinalgOp = dyn_cast<linalg::LinalgOp>(user);
      if (!userLinalgOp || !hasReductionIterator(userLinalgOp))
        return false;
      if (!llvm::is_contained(userLinalgOp.getDpsInits(), result))
        return false;
      hasReductionInitUser = true;
    }
  }

  return hasReductionInitUser;
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
        if (isReductionInitHelper(linalgOp))
          return WalkResult::advance();
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
      auto primaryAttr = op->getAttrOfType<BoolAttr>(kPrimaryAttr);
      group.view.ops.push_back(PatternOpView{
          op, nextOrdinal, deriveOpRole(op),
          primaryAttr && primaryAttr.getValue()});
      appendTemplateFamilies(op, group.view.templateFamilies);
      if (failed(mergeHandwrittenKind(op, group.view)))
        return WalkResult::interrupt();
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
    if (!pattern.templateFamilies.empty()) {
      os << "  template_families = [";
      llvm::interleaveComma(pattern.templateFamilies, os);
      os << "]\n";
    }
    if (!pattern.handwrittenKind.empty())
      os << "  handwritten_kind = \"" << pattern.handwrittenKind << "\"\n";
  }
}

} // namespace mlir::afir::ascend::schedule
