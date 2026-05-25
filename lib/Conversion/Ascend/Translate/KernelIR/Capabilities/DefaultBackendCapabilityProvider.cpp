//===- DefaultBackendCapabilityProvider.cpp - Default backend capabilities -===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/BackendSupportMatrix.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::afir::ascend::backend {
namespace {

bool isGenericFloatDtype(mlir::Type t) {
  return t.isF32() || t.isF16() || t.isBF16();
}

bool isCubeMatmulInputDtype(mlir::Type t) {
  return t.isF16() || t.isBF16();
}

bool isCubeMatmulOutputDtype(mlir::Type t) {
  return t.isF16() || t.isBF16() || t.isF32();
}

bool isMatmulKind(ComputeKind kind) {
  return kind == ComputeKind::Matmul || kind == ComputeKind::BatchMatmul;
}

class DefaultBackendCapabilityProvider final : public BackendCapabilityProvider {
public:
  bool supportsMovementPath(MemorySpace source,
                            MemorySpace target) const override {
    return (source == MemorySpace::GM && target == MemorySpace::GM) ||
           (source == MemorySpace::GM &&
            (target == MemorySpace::A1 || target == MemorySpace::B1 ||
             target == MemorySpace::VECIN)) ||
           (source == MemorySpace::A1 && target == MemorySpace::A2) ||
           (source == MemorySpace::B1 && target == MemorySpace::B2) ||
           (source == MemorySpace::CO1 && target == MemorySpace::VECIN) ||
           (source == MemorySpace::VECOUT && target == MemorySpace::GM);
  }

  bool supportsComputeKind(ComputeKind kind) const override {
    switch (kind) {
    case ComputeKind::Matmul:
    case ComputeKind::BatchMatmul:
    case ComputeKind::Fill:
    case ComputeKind::ElementwiseAdd:
    case ComputeKind::ElementwiseMul:
    case ComputeKind::ElementwiseMax:
    case ComputeKind::FusedElementwise:
    case ComputeKind::TensorCopy:
    case ComputeKind::ScalarGeneric:
    case ComputeKind::Transpose:
    case ComputeKind::VectorGather:
    case ComputeKind::ReductionAdd:
    case ComputeKind::ElementwiseSub:
    case ComputeKind::ElementwiseDiv:
    case ComputeKind::ElementwiseNeg:
    case ComputeKind::ElementwiseExp:
    case ComputeKind::ElementwiseLog:
    case ComputeKind::ElementwiseSqrt:
    case ComputeKind::ElementwiseRsqrt:
    case ComputeKind::ElementwiseAbs:
    case ComputeKind::ElementwiseMin:
    case ComputeKind::ReductionMax:
    case ComputeKind::ReductionMin:
    case ComputeKind::ReductionMul:
      return true;
    case ComputeKind::ElementwiseExp2:
    case ComputeKind::ElementwiseTanh:
    case ComputeKind::ElementwiseErf:
    case ComputeKind::ElementwiseSin:
    case ComputeKind::ElementwiseCos:
    case ComputeKind::ElementwiseFma:
    case ComputeKind::ElementwiseReciprocal:
    case ComputeKind::ElementwiseRelu:
    case ComputeKind::ElementwiseSelect:
    case ComputeKind::Unknown:
      return false;
    }
    return false;
  }

  bool supportsDtype(ComputeKind kind, mlir::ArrayRef<mlir::Type> inputTypes,
                     mlir::ArrayRef<mlir::Type> outputTypes) const override {
    auto inputOk = isMatmulKind(kind) ? isCubeMatmulInputDtype
                                      : isGenericFloatDtype;
    auto outputOk = isMatmulKind(kind) ? isCubeMatmulOutputDtype
                                       : isGenericFloatDtype;
    for (mlir::Type t : inputTypes)
      if (!inputOk(t))
        return false;
    for (mlir::Type t : outputTypes)
      if (!outputOk(t))
        return false;
    return true;
  }
};

} // namespace

const BackendCapabilityProvider &getDefaultBackendCapabilityProvider() {
  static const DefaultBackendCapabilityProvider provider;
  return provider;
}

} // namespace mlir::afir::ascend::backend
