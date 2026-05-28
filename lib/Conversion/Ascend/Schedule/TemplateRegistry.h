//===- TemplateRegistry.h - Ascend schedule templates -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_TEMPLATEREGISTRY_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_TEMPLATEREGISTRY_H

#include "ScheduleTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::ascend::schedule {

/// Register a custom schedule template into the global registry.
/// Call before the first use of matchScheduleTemplates.
void registerTemplate(ScheduleTemplate tmpl);

/// Register the built-in default templates. Called automatically on first use;
/// exposed here for explicit initialization in tests.
void registerBuiltinTemplates();

SmallVector<ScheduleTemplate>
matchScheduleTemplates(const ScheduleProblem &problem);

void printTemplateRegistryReport(StringRef kernelId,
                                 ArrayRef<ScheduleTemplate> matches,
                                 llvm::raw_ostream &os);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_TEMPLATEREGISTRY_H
