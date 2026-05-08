//===- CandidateClosure.h - Ascend V2 candidate closure model ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_CANDIDATECLOSURE_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_CANDIDATECLOSURE_H

#include "Conversion/AscendV2/Kernelize/DependencyAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::afir::ascend::v2::kernelize {

struct CandidateClosure {
  SmallVector<Operation *, 0> internalOps;
  SmallVector<Value, 0> externalInputs;
  SmallVector<Value, 0> externalOutputs;
  SmallVector<Value, 0> escapingValues;
  bool isClosed = false;
  std::string failureReason;
};

CandidateClosure computeCandidateClosure(ArrayRef<Operation *> internalOps,
                                          const ProducerConsumerIndex &index);

} // namespace mlir::afir::ascend::v2::kernelize

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_KERNELIZE_CANDIDATECLOSURE_H
