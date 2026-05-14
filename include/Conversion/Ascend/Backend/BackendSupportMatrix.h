//===- BackendSupportMatrix.h - Ascend backend support matrix ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_SUPPORT_MATRIX_H
#define ASCEND_MLIR_CONVERSION_ASCEND_BACKEND_SUPPORT_MATRIX_H

#include "Target/Ascend/TargetProfile.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::afir::ascend::backend {

using MemorySpace = ::mlir::ascend::MemoryPlace;
inline constexpr MemorySpace kUnknownMemorySpace =
    static_cast<MemorySpace>(-1);

enum class ComputeKind {
  Unknown,
  Matmul,
  BatchMatmul,
  Fill,
  ElementwiseAdd,
  ElementwiseMul,
  ElementwiseMax,
  FusedElementwise,
  TensorCopy,
  ScalarGeneric,
  Transpose,
  VectorGather,
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
