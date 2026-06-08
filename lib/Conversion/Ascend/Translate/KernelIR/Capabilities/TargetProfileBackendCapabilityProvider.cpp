//===- TargetProfileBackendCapabilityProvider.cpp - Target capabilities ---===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/BackendSupportMatrix.h"

#include "Target/Ascend/TargetIntrinsicModel.h"
#include "mlir/IR/BuiltinTypes.h"
#include <memory>
#include <optional>

namespace mlir::ascend::backend {
namespace {

class TargetProfileBackendCapabilityProvider final
    : public BackendCapabilityProvider {
public:
  explicit TargetProfileBackendCapabilityProvider(
      const ::mlir::ascend::TargetProfile &profile)
      : profile(profile) {
    FailureOr<::mlir::ascend::TargetIntrinsicModel> builtModel =
        ::mlir::ascend::TargetIntrinsicModelBuilder().build(profile);
    if (succeeded(builtModel))
      intrinsicModel = std::move(*builtModel);
  }

  bool supportsMovementPath(MemorySpace source,
                            MemorySpace target) const override {
    if (!getDefaultBackendCapabilityProvider().supportsMovementPath(source,
                                                                    target))
      return false;
    return isAdvertisedMemoryPlace(source) && isAdvertisedMemoryPlace(target);
  }

  bool supportsComputeKind(ComputeKind kind) const override {
    if (!getDefaultBackendCapabilityProvider().supportsComputeKind(kind))
      return false;
    if (!intrinsicModel)
      return false;

    switch (kind) {
    case ComputeKind::Matmul:
    case ComputeKind::BatchMatmul:
      return hasTargetComputeIntrinsic(::mlir::ascend::ComputeKind::Matmul);
    case ComputeKind::ElementwiseAdd:
      return hasTargetComputeIntrinsic(::mlir::ascend::ComputeKind::VectorAdd);
    case ComputeKind::ElementwiseExp:
      return hasTargetComputeIntrinsic(::mlir::ascend::ComputeKind::VectorExp);
    case ComputeKind::Transpose:
      return hasTargetComputeIntrinsic(
          ::mlir::ascend::ComputeKind::VectorTranspose);
    case ComputeKind::VectorGather:
      return hasTargetComputeIntrinsic(
          ::mlir::ascend::ComputeKind::VectorGather);
    case ComputeKind::ReductionAdd:
    case ComputeKind::ReductionMax:
    case ComputeKind::ReductionMin:
    case ComputeKind::ReductionMul:
      return hasTargetComputeIntrinsic(
          ::mlir::ascend::ComputeKind::VectorReduce);
    case ComputeKind::Fill:
    case ComputeKind::ElementwiseMul:
    case ComputeKind::ElementwiseMax:
    case ComputeKind::FusedElementwise:
    case ComputeKind::TensorCopy:
    case ComputeKind::ScalarGeneric:
    case ComputeKind::ElementwiseSub:
    case ComputeKind::ElementwiseDiv:
    case ComputeKind::ElementwiseNeg:
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
      return !intrinsicModel->getIntrinsicsForUnit(
                                ::mlir::ascend::ExecutionUnit::Vector)
                  .empty();
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
    if (!getDefaultBackendCapabilityProvider().supportsDtype(kind, inputTypes,
                                                             outputTypes))
      return false;
    if (!profile.hardware.supportBF16) {
      for (mlir::Type type : inputTypes)
        if (type.isBF16())
          return false;
      for (mlir::Type type : outputTypes)
        if (type.isBF16())
          return false;
    }
    return true;
  }

private:
  bool isAdvertisedMemoryPlace(MemorySpace place) const {
    if (place == MemorySpace::GM || place == MemorySpace::GMFlat)
      return true;
    auto it = profile.capacityBytes.find(place);
    return it != profile.capacityBytes.end() && it->second > 0;
  }

  bool hasTargetComputeIntrinsic(::mlir::ascend::ComputeKind kind) const {
    return intrinsicModel &&
           !intrinsicModel->getIntrinsicsForComputeKind(kind).empty();
  }

  ::mlir::ascend::TargetProfile profile;
  std::optional<::mlir::ascend::TargetIntrinsicModel> intrinsicModel;
};

} // namespace

std::unique_ptr<BackendCapabilityProvider>
createTargetProfileBackendCapabilityProvider(
    const ::mlir::ascend::TargetProfile &profile) {
  return std::make_unique<TargetProfileBackendCapabilityProvider>(profile);
}

} // namespace mlir::ascend::backend
