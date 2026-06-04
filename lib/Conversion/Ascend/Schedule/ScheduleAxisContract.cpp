//===- ScheduleAxisContract.cpp - Ascend schedule axis contract --------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "ScheduleAxisContract.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;

namespace mlir::ascend::schedule {
namespace {

std::string getStableAxisName(const LogicalAxisInfo &axis) {
  if (!axis.symbolName.empty())
    return axis.symbolName;
  return (llvm::Twine("axis") + llvm::Twine(axis.logicalAxisId)).str();
}

SymbolicAxisRef makeAxisRef(const LogicalAxisInfo &axis) {
  SymbolicAxisRef ref;
  ref.logicalAxisId = axis.logicalAxisId;
  ref.symbolName = getStableAxisName(axis);
  ref.kind = axis.kind;
  return ref;
}

const LogicalAxisInfo *lookupLogicalAxis(const CoalescedAxisInfo &axes,
                                         unsigned logicalAxisId) {
  for (const LogicalAxisInfo &axis : axes.logicalAxes)
    if (axis.logicalAxisId == logicalAxisId)
      return &axis;
  return nullptr;
}

bool hasAxisSymbol(ArrayRef<SymbolicAxisRef> refs, StringRef symbolName) {
  return llvm::any_of(refs, [&](const SymbolicAxisRef &ref) {
    return ref.symbolName == symbolName;
  });
}

void appendUniqueAxis(SmallVectorImpl<SymbolicAxisRef> &refs,
                      const LogicalAxisInfo &axis) {
  SymbolicAxisRef ref = makeAxisRef(axis);
  if (hasAxisSymbol(refs, ref.symbolName))
    return;
  refs.push_back(std::move(ref));
}

void removeTileableReductionSymbols(ScheduleAxisContract &contract) {
  llvm::StringSet<> reductionSymbols;
  for (const SymbolicAxisRef &axis : contract.requiredReductionAxes)
    reductionSymbols.insert(axis.symbolName);

  llvm::erase_if(contract.tileableAxes, [&](const SymbolicAxisRef &axis) {
    return reductionSymbols.contains(axis.symbolName);
  });
}

} // namespace

FailureOr<ScheduleAxisContract>
buildScheduleAxisContract(const KernelPatternView &pattern,
                          const CoalescedAxisInfo &axes) {
  (void)pattern;

  ScheduleAxisContract contract;
  llvm::SmallSet<unsigned, 4> requiredReductionIds;
  for (unsigned logicalAxisId : axes.reductionAxes) {
    const LogicalAxisInfo *axis = lookupLogicalAxis(axes, logicalAxisId);
    if (!axis)
      continue;
    appendUniqueAxis(contract.requiredReductionAxes, *axis);
    requiredReductionIds.insert(logicalAxisId);
  }

  for (unsigned logicalAxisId : axes.parallelAxes) {
    if (requiredReductionIds.contains(logicalAxisId))
      continue;
    const LogicalAxisInfo *axis = lookupLogicalAxis(axes, logicalAxisId);
    if (!axis)
      continue;
    appendUniqueAxis(contract.tileableAxes, *axis);
  }

  removeTileableReductionSymbols(contract);
  return contract;
}

void printScheduleAxisList(ArrayRef<SymbolicAxisRef> axes,
                           llvm::raw_ostream &os) {
  os << "[";
  llvm::interleaveComma(axes, os,
                        [&](const SymbolicAxisRef &axis) {
                          os << axis.symbolName;
                        });
  os << "]";
}

} // namespace mlir::ascend::schedule
