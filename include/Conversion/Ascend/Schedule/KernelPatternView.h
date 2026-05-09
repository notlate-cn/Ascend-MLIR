//===- KernelPatternView.h - Ascend schedule pattern view -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_KERNELPATTERNVIEW_H
#define ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_KERNELPATTERNVIEW_H

#include "Conversion/Ascend/Schedule/ScheduleTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::schedule {

FailureOr<SmallVector<KernelPatternView>>
buildKernelPatternViews(ModuleOp module);

const PatternOpView *selectDominantPrimaryOp(const KernelPatternView &pattern);

void printKernelPatternViews(ArrayRef<KernelPatternView> patterns,
                             llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::schedule

#endif // ASCEND_MLIR_CONVERSION_ASCEND_SCHEDULE_KERNELPATTERNVIEW_H
