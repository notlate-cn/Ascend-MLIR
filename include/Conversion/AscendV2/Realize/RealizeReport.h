//===- RealizeReport.h - Ascend V2 realize reports ------------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZEREPORT_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZEREPORT_H

#include "Conversion/AscendV2/Realize/RealizeTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::v2::realize {

void printRealizeReport(llvm::ArrayRef<RealizePlanBundle> bundles,
                        llvm::raw_ostream &os);

} // namespace mlir::afir::ascend::v2::realize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_REALIZE_REALIZEREPORT_H
