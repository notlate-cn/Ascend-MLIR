//===- BackendSupportMatrix.h - Ascend KernelIR support matrix ---*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_CAPABILITIES_BACKEND_SUPPORT_MATRIX_H
#define ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_CAPABILITIES_BACKEND_SUPPORT_MATRIX_H

#include "Target/Ascend/TargetProfile.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir::ascend::backend {

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
  // elementwise ops
  ElementwiseSub,
  ElementwiseDiv,
  ElementwiseNeg,
  ElementwiseExp,
  ElementwiseExp2,
  ElementwiseLog,
  ElementwiseSqrt,
  ElementwiseRsqrt,
  ElementwiseTanh,
  ElementwiseErf,
  ElementwiseAbs,
  ElementwiseSin,
  ElementwiseCos,
  ElementwiseFma,
  ElementwiseReciprocal,
  ElementwiseRelu,
  ElementwiseSelect,
  ElementwiseMin,
  ElementwisePyAscMath,
  ElementwisePyAscBitwise,
  // reduction ops
  ReductionMax,
  ReductionMin,
  ReductionMul,
};

struct UnsupportedReason {
  std::string category;
  std::string detail;
};

MemorySpace parseMemorySpace(int64_t value);
llvm::StringRef stringifyMemorySpace(MemorySpace space);
llvm::StringRef stringifyComputeKind(ComputeKind kind);

class BackendCapabilityProvider {
public:
  virtual ~BackendCapabilityProvider() = default;

  virtual bool supportsMovementPath(MemorySpace source,
                                    MemorySpace target) const = 0;
  virtual bool supportsComputeKind(ComputeKind kind) const = 0;
  virtual bool supportsDtype(ComputeKind kind,
                             mlir::ArrayRef<mlir::Type> inputTypes,
                             mlir::ArrayRef<mlir::Type> outputTypes) const = 0;
};

const BackendCapabilityProvider &getDefaultBackendCapabilityProvider();
std::unique_ptr<BackendCapabilityProvider>
createTargetProfileBackendCapabilityProvider(
    const ::mlir::ascend::TargetProfile &profile);

class AscendBackendSupportMatrix {
public:
  explicit AscendBackendSupportMatrix(
      const BackendCapabilityProvider &provider =
          getDefaultBackendCapabilityProvider());

  bool isSupportedMovementPath(MemorySpace source, MemorySpace target) const;
  UnsupportedReason explainMovementPath(MemorySpace source,
                                        MemorySpace target) const;

  bool isSupportedComputeKind(ComputeKind kind) const;
  UnsupportedReason explainComputeKind(ComputeKind kind) const;

  bool isSupportedDtype(ComputeKind kind,
                        mlir::ArrayRef<mlir::Type> inputTypes,
                        mlir::ArrayRef<mlir::Type> outputTypes) const;
  UnsupportedReason explainDtype(ComputeKind kind,
                                 mlir::ArrayRef<mlir::Type> inputTypes,
                                 mlir::ArrayRef<mlir::Type> outputTypes) const;

private:
  const BackendCapabilityProvider &provider;
};

} // namespace mlir::ascend::backend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_CAPABILITIES_BACKEND_SUPPORT_MATRIX_H
