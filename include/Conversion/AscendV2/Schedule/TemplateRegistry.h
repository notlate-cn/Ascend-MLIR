//===- TemplateRegistry.h - Ascend V2 schedule templates -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_TEMPLATEREGISTRY_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_TEMPLATEREGISTRY_H

#include "Conversion/AscendV2/Schedule/ScheduleTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::afir::ascend::v2::schedule {

SmallVector<ScheduleTemplate>
matchScheduleTemplates(const ScheduleProblem &problem);

void printTemplateRegistryReport(StringRef kernelId,
                                 ArrayRef<ScheduleTemplate> matches,
                                 llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::v2::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_SCHEDULE_TEMPLATEREGISTRY_H
