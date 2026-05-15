//===- DependencyAnalysis.h - Ascend kernel dependency analysis -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_DEPENDENCYANALYSIS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_DEPENDENCYANALYSIS_H

#include "KernelizeTypes.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::afir::ascend::kernelize {

struct OpSemanticSummary {
  Operation *op = nullptr;
  OperationId opId;
  KernelizeParticipationKind participation =
      KernelizeParticipationKind::Unsupported;
  AccessPatternKind accessPattern = AccessPatternKind::Unknown;
  SmallVector<IteratorKind> iteratorTypes;
  SmallVector<AffineMap> indexingMaps;
  SmallVector<unsigned> resultRanks;
  SmallVector<KernelizeSemanticTrait> traits;
  std::string modelName = "unknown";
  std::string unsupportedReason;
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

struct UnsupportedProducerDiagnostic {
  Operation *producer = nullptr;
  Operation *consumer = nullptr;
  std::string reason;
};

struct DependencyAnalysisResult {
  ProducerConsumerIndex index;
  DenseMap<Operation *, OpSemanticSummary> summaries;
  SmallVector<UnsupportedProducerDiagnostic, 4> unsupportedProducers;
};

class DependencyAnalyzer {
public:
  FailureOr<DependencyAnalysisResult> analyze(ModuleOp module) const;
};

void emitDependencyAnalysisReport(raw_ostream &os,
                                  const DependencyAnalysisResult &result);

} // namespace mlir::afir::ascend::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_KERNELIZE_DEPENDENCYANALYSIS_H
