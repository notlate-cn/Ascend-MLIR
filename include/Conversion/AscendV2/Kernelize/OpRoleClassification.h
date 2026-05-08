//===- OpRoleClassification.h - Ascend V2 op role classification -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_OPROLECLASSIFICATION_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_OPROLECLASSIFICATION_H

#include "Conversion/AscendV2/Kernelize/DependencyAnalysis.h"
#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::v2::kernelize {

using OpRoleList = SmallVector<OpRole, 4>;
using OpRoleMap = DenseMap<Operation *, OpRoleList>;

class OpRoleClassifier {
public:
  FailureOr<OpRoleMap> classify(const DependencyAnalysisResult &deps) const;
};

void attachRoleAttributes(ModuleOp module, const OpRoleMap &roleMap);
void emitOpRoleClassificationReport(raw_ostream &os,
                                     const DependencyAnalysisResult &deps,
                                     const OpRoleMap &roleMap);

} // namespace mlir::afir::ascend::v2::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_OPROLECLASSIFICATION_H
