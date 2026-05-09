//===- RealizeReport.h - Ascend realize reports ------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_REALIZEREPORT_H
#define ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_REALIZEREPORT_H

#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::realize {

void printRealizeReport(llvm::ArrayRef<RealizePlanBundle> bundles,
                        llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::realize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_REALIZE_REALIZEREPORT_H
