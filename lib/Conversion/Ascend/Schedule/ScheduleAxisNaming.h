//===- ScheduleAxisNaming.h - Schedule axis naming helpers -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEAXISNAMING_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEAXISNAMING_H

#include "ScheduleTypes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace mlir::ascend::schedule {

inline std::string getLogicalAxisLabel(unsigned logicalAxisId) {
  return (llvm::Twine("axis") + llvm::Twine(logicalAxisId)).str();
}

inline bool hasDuplicateAxisSymbol(ArrayRef<LogicalAxisInfo> axes,
                                   llvm::StringRef symbolName) {
  if (symbolName.empty())
    return false;

  unsigned matches = 0;
  for (const LogicalAxisInfo &axis : axes) {
    if (axis.symbolName != symbolName)
      continue;
    if (++matches > 1)
      return true;
  }
  return false;
}

inline std::string getAxisRefText(unsigned logicalAxisId,
                                  llvm::StringRef symbolName) {
  std::string label = getLogicalAxisLabel(logicalAxisId);
  if (symbolName.empty())
    return label;
  return (llvm::Twine(label) + "(sym=" + symbolName + ")").str();
}

inline std::string getAxisRefText(const SymbolicAxisRef &axis) {
  return getAxisRefText(axis.logicalAxisId, axis.symbolName);
}

inline std::string
getSymbolicTileParamName(ArrayRef<LogicalAxisInfo> axes,
                         const LogicalAxisInfo &axis) {
  if (axis.symbolName.empty())
    return "";

  if (hasDuplicateAxisSymbol(axes, axis.symbolName)) {
    return (llvm::Twine("T_") + getLogicalAxisLabel(axis.logicalAxisId) + "_" +
            axis.symbolName)
        .str();
  }
  return (llvm::Twine("T_") + axis.symbolName).str();
}

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULEAXISNAMING_H
