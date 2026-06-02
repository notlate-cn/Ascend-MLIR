//===- ScheduleTemplateImplementation.h - Schedule templates ----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULETEMPLATEIMPLEMENTATION_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULETEMPLATEIMPLEMENTATION_H

#include "ScheduleTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir::ascend::schedule {

class ScheduleTemplateImplementation {
public:
  virtual ~ScheduleTemplateImplementation() = default;

  virtual const ScheduleTemplate &metadata() const = 0;
  virtual llvm::StringRef implementationKind() const = 0;
  virtual llvm::StringRef description() const = 0;
  virtual SmallVector<TileShape>
  generateTileShapes(const ScheduleProblem &problem,
                     const ScheduleSearchOptions &options) const = 0;
};

std::unique_ptr<ScheduleTemplateImplementation>
createSingleTilePerBlockTemplateImplementation(llvm::StringRef family,
                                               llvm::StringRef roleTag,
                                               unsigned minRank,
                                               unsigned maxRank,
                                               unsigned priority);

std::unique_ptr<ScheduleTemplateImplementation>
createGroupedTilePerBlockTemplateImplementation(ScheduleTemplate tmpl);

std::unique_ptr<ScheduleTemplateImplementation>
createRoleDrivenTemplateImplementation(ScheduleTemplate tmpl,
                                       llvm::StringRef description);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_SCHEDULETEMPLATEIMPLEMENTATION_H
