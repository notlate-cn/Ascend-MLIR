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

#include <memory>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::ascend::schedule {

class ScheduleTemplateImplementation;

/// Register a custom schedule template into the global registry.
/// Call before the first use of matchScheduleTemplates.
void registerTemplate(ScheduleTemplate tmpl);

/// Register a custom schedule template implementation into the global registry.
/// Call before the first use of matchScheduleTemplateImplementations.
void registerTemplateImplementation(
    std::unique_ptr<ScheduleTemplateImplementation> implementation);

/// Register the built-in default templates. Called automatically on first use;
/// exposed here for explicit initialization in tests.
void registerBuiltinTemplates();

SmallVector<const ScheduleTemplateImplementation *>
matchScheduleTemplateImplementations(const ScheduleProblem &problem);

SmallVector<ScheduleTemplate>
matchScheduleTemplates(const ScheduleProblem &problem);

void printTemplateRegistryReport(StringRef kernelId,
                                 ArrayRef<ScheduleTemplate> matches,
                                 llvm::raw_ostream &os);

void printTemplateRegistryReport(
    StringRef kernelId,
    ArrayRef<const ScheduleTemplateImplementation *> matches,
    llvm::raw_ostream &os);

} // namespace mlir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_TEMPLATEREGISTRY_H
