//===- DefaultBackendCapabilityProvider.cpp - Default backend capabilities -===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/AscendCOpCapabilityRegistry.h"
#include "Conversion/Ascend/Translate/KernelIR/Capabilities/BackendSupportMatrix.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::ascend::backend {
namespace {

bool isGenericFloatDtype(mlir::Type t) {
  return t.isF32() || t.isF16() || t.isBF16();
}

bool isIntegerOrBoolDtype(mlir::Type t) {
  return llvm::isa<mlir::IntegerType>(t);
}

bool isMatmulKind(ComputeKind kind) {
  return kind == ComputeKind::Matmul || kind == ComputeKind::BatchMatmul;
}

bool supportsMmadDtypes(mlir::ArrayRef<mlir::Type> inputTypes,
                        mlir::ArrayRef<mlir::Type> outputTypes) {
  const AscendCOpCapability *capability =
      lookupAscendCOpCapability(AscendCOpKind::Mmad);
  if (!capability || inputTypes.size() != 2 || outputTypes.size() != 1 ||
      capability->dtypeOperands.size() < 3)
    return false;
  return isAscendCOperandDtypeAllowed(capability->dtypeOperands[0],
                                      outputTypes[0]) &&
         isAscendCOperandDtypeAllowed(capability->dtypeOperands[1],
                                      inputTypes[0]) &&
         isAscendCOperandDtypeAllowed(capability->dtypeOperands[2],
                                      inputTypes[1]);
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
           (source == MemorySpace::CO1 &&
            (target == MemorySpace::VECIN ||
             target == MemorySpace::VECOUT)) ||
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
    case ComputeKind::ElementwiseTanh:
    case ComputeKind::ElementwiseErf:
    case ComputeKind::ElementwiseAbs:
    case ComputeKind::ElementwiseSin:
    case ComputeKind::ElementwiseCos:
    case ComputeKind::ElementwiseMin:
    case ComputeKind::ElementwisePyAscMath:
    case ComputeKind::ElementwisePyAscBitwise:
    case ComputeKind::ReductionMax:
    case ComputeKind::ReductionMin:
    case ComputeKind::ReductionMul:
      return true;
    case ComputeKind::ElementwiseExp2:
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
    bool (*inputOk)(mlir::Type) = isGenericFloatDtype;
    bool (*outputOk)(mlir::Type) = isGenericFloatDtype;
    if (isMatmulKind(kind))
      return supportsMmadDtypes(inputTypes, outputTypes);
    if (kind == ComputeKind::ElementwisePyAscBitwise) {
      inputOk = isIntegerOrBoolDtype;
      outputOk = isIntegerOrBoolDtype;
    }
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

} // namespace mlir::ascend::backend
