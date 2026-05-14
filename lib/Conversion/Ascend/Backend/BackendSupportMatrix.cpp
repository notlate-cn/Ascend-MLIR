//===- BackendSupportMatrix.cpp - Ascend backend support matrix -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Backend/BackendSupportMatrix.h"

#include "llvm/Support/FormatVariadic.h"

namespace mlir::afir::ascend::backend {

MemorySpace parseMemorySpace(int64_t value) {
  switch (value) {
  case static_cast<int64_t>(MemorySpace::GM):
    return MemorySpace::GM;
  case static_cast<int64_t>(MemorySpace::A1):
    return MemorySpace::A1;
  case static_cast<int64_t>(MemorySpace::A2):
    return MemorySpace::A2;
  case static_cast<int64_t>(MemorySpace::B1):
    return MemorySpace::B1;
  case static_cast<int64_t>(MemorySpace::B2):
    return MemorySpace::B2;
  case static_cast<int64_t>(MemorySpace::CO1):
    return MemorySpace::CO1;
  case static_cast<int64_t>(MemorySpace::VECIN):
    return MemorySpace::VECIN;
  case static_cast<int64_t>(MemorySpace::VECOUT):
    return MemorySpace::VECOUT;
  case static_cast<int64_t>(MemorySpace::VECCALC):
    return MemorySpace::VECCALC;
  case static_cast<int64_t>(MemorySpace::GMFlat):
    return MemorySpace::GMFlat;
  default:
    return kUnknownMemorySpace;
  }
}

llvm::StringRef stringifyMemorySpace(MemorySpace space) {
  if (space == kUnknownMemorySpace)
    return "Unknown";

  switch (space) {
  case MemorySpace::GM:
    return "GM";
  case MemorySpace::A1:
    return "A1";
  case MemorySpace::A2:
    return "A2";
  case MemorySpace::B1:
    return "B1";
  case MemorySpace::B2:
    return "B2";
  case MemorySpace::CO1:
    return "CO1";
  case MemorySpace::VECIN:
    return "VECIN";
  case MemorySpace::VECOUT:
    return "VECOUT";
  case MemorySpace::VECCALC:
    return "VECCALC";
  case MemorySpace::GMFlat:
    return "GMFlat";
  }
  return "Unknown";
}

llvm::StringRef stringifyComputeKind(ComputeKind kind) {
  switch (kind) {
  case ComputeKind::Matmul:
    return "matmul";
  case ComputeKind::BatchMatmul:
    return "batch_matmul";
  case ComputeKind::Fill:
    return "fill";
  case ComputeKind::ElementwiseAdd:
    return "elementwise_add";
  case ComputeKind::ElementwiseMul:
    return "elementwise_mul";
  case ComputeKind::ElementwiseMax:
    return "elementwise_max";
  case ComputeKind::FusedElementwise:
    return "fused_elementwise";
  case ComputeKind::TensorCopy:
    return "tensor_copy";
  case ComputeKind::ScalarGeneric:
    return "scalar_generic";
  case ComputeKind::Transpose:
    return "transpose";
  case ComputeKind::VectorGather:
    return "vector_gather";
  case ComputeKind::ReductionAdd:
    return "reduction_add";
  case ComputeKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

bool AscendBackendSupportMatrix::isSupportedMovementPath(
    MemorySpace source, MemorySpace target) const {
  return (source == MemorySpace::GM && target == MemorySpace::GM) ||
         (source == MemorySpace::GM &&
          (target == MemorySpace::A1 || target == MemorySpace::B1 ||
           target == MemorySpace::VECIN)) ||
         (source == MemorySpace::A1 && target == MemorySpace::A2) ||
         (source == MemorySpace::B1 && target == MemorySpace::B2) ||
         (source == MemorySpace::CO1 && target == MemorySpace::VECIN) ||
         (source == MemorySpace::VECOUT && target == MemorySpace::GM);
}

UnsupportedReason AscendBackendSupportMatrix::explainMovementPath(
    MemorySpace source, MemorySpace target) const {
  if (isSupportedMovementPath(source, target))
    return {"movement", ""};
  return {"movement",
          llvm::formatv("unsupported movement path {0} -> {1}",
                        stringifyMemorySpace(source),
                        stringifyMemorySpace(target))
              .str()};
}

bool AscendBackendSupportMatrix::isSupportedComputeKind(
    ComputeKind kind) const {
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
    return true;
  case ComputeKind::Unknown:
    return false;
  }
  return false;
}

UnsupportedReason AscendBackendSupportMatrix::explainComputeKind(
    ComputeKind kind) const {
  if (isSupportedComputeKind(kind))
    return {"compute", ""};
  return {"compute",
          llvm::formatv("unsupported compute kind {0}",
                        stringifyComputeKind(kind))
              .str()};
}

} // namespace mlir::afir::ascend::backend
