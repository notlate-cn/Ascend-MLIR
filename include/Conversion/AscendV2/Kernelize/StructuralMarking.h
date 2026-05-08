//===- StructuralMarking.h - Ascend V2 structural marking -------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_STRUCTURALMARKING_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_STRUCTURALMARKING_H

#include "Conversion/AscendV2/Kernelize/DependencyAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::v2::kernelize {

class StructuralMarker {
public:
  LogicalResult mark(ModuleOp module,
                     const DependencyAnalysisResult &deps) const;
};

void emitStructuralMarkingReport(raw_ostream &os,
                                 const DependencyAnalysisResult &deps);

} // namespace mlir::afir::ascend::v2::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_STRUCTURALMARKING_H
