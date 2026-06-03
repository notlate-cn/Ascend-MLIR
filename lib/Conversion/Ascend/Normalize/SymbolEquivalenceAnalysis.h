//===- SymbolEquivalenceAnalysis.h - Normalize symbol facts -----*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_SYMBOLEQUIVALENCEANALYSIS_H
#define ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_SYMBOLEQUIVALENCEANALYSIS_H

#include "Conversion/Ascend/Common/SymbolConstraints.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::ascend::normalize {

struct SymbolEqualityProof {
  symbol::DimRef lhs;
  symbol::DimRef rhs;
  StringRef rule;
};

struct SymbolEquivalenceResult {
  ArrayAttr attr;
  SmallVector<SymbolEqualityProof, 16> proofs;
};

FailureOr<SymbolEquivalenceResult> analyzeSymbolEquivalence(func::FuncOp func);

} // namespace mlir::ascend::normalize

#endif // ASCEND_MLIR_CONVERSION_ASCEND_NORMALIZE_SYMBOLEQUIVALENCEANALYSIS_H
