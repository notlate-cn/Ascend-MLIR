//===- BackendSupportMatrix.h - Ascend backend support matrix ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_SUPPORT_MATRIX_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_SUPPORT_MATRIX_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace mlir::afir::ascend::backend {

enum class MemorySpace : int64_t {
  Unknown = -1,
  GM = 0,
  A1 = 1,
  A2 = 2,
  B1 = 3,
  B2 = 4,
  CO1 = 7,
  VECIN = 9,
  VECOUT = 10,
  VECCALC = 11,
};

enum class ComputeKind {
  Unknown,
  Matmul,
  Fill,
  ElementwiseAdd,
  ElementwiseMax,
  ReductionAdd,
};

struct UnsupportedReason {
  std::string category;
  std::string detail;
};

MemorySpace parseMemorySpace(int64_t value);
llvm::StringRef stringifyMemorySpace(MemorySpace space);
llvm::StringRef stringifyComputeKind(ComputeKind kind);

class AscendBackendSupportMatrix {
public:
  bool isSupportedMovementPath(MemorySpace source, MemorySpace target) const;
  UnsupportedReason explainMovementPath(MemorySpace source,
                                        MemorySpace target) const;

  bool isSupportedComputeKind(ComputeKind kind) const;
  UnsupportedReason explainComputeKind(ComputeKind kind) const;
};

} // namespace mlir::afir::ascend::backend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_SUPPORT_MATRIX_H
