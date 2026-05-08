//===- DependencyAnalysis.h - Ascend V2 kernel dependency analysis -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_DEPENDENCYANALYSIS_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_DEPENDENCYANALYSIS_H

#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::afir::ascend::v2::kernelize {

struct OpSemanticSummary {
  Operation *op = nullptr;
  OperationId opId;
  AccessPatternKind accessPattern = AccessPatternKind::Unknown;
  SmallVector<StringRef> iteratorTypes;
  unsigned resultRank = 0;
  bool hasReductionIterator = false;
  bool hasOnlyParallelIterators = false;
};

struct ProducerConsumerIndex {
  SmallVector<Operation *> orderedOps;
  DenseMap<Operation *, OperationId> opIds;
  DenseMap<Operation *, SmallVector<Operation *>> producers;
  DenseMap<Operation *, SmallVector<Operation *>> consumers;
};

struct DependencyAnalysisResult {
  ProducerConsumerIndex index;
  DenseMap<Operation *, OpSemanticSummary> summaries;
};

class DependencyAnalyzer {
public:
  FailureOr<DependencyAnalysisResult> analyze(ModuleOp module) const;
};

void emitDependencyAnalysisReport(raw_ostream &os,
                                  const DependencyAnalysisResult &result);

} // namespace mlir::afir::ascend::v2::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_DEPENDENCYANALYSIS_H
