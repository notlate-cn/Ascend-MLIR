//===- BackendSupportMatrix.cpp - Ascend backend support matrix -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/BackendSupportMatrix.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

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
  case ComputeKind::ElementwiseSub:        return "elementwise_sub";
  case ComputeKind::ElementwiseDiv:        return "elementwise_div";
  case ComputeKind::ElementwiseNeg:        return "elementwise_neg";
  case ComputeKind::ElementwiseExp:        return "elementwise_exp";
  case ComputeKind::ElementwiseExp2:       return "elementwise_exp2";
  case ComputeKind::ElementwiseLog:        return "elementwise_log";
  case ComputeKind::ElementwiseSqrt:       return "elementwise_sqrt";
  case ComputeKind::ElementwiseRsqrt:      return "elementwise_rsqrt";
  case ComputeKind::ElementwiseTanh:       return "elementwise_tanh";
  case ComputeKind::ElementwiseErf:        return "elementwise_erf";
  case ComputeKind::ElementwiseAbs:        return "elementwise_abs";
  case ComputeKind::ElementwiseSin:        return "elementwise_sin";
  case ComputeKind::ElementwiseCos:        return "elementwise_cos";
  case ComputeKind::ElementwiseFma:        return "elementwise_fma";
  case ComputeKind::ElementwiseReciprocal: return "elementwise_reciprocal";
  case ComputeKind::ElementwiseRelu:       return "elementwise_relu";
  case ComputeKind::ElementwiseSelect:     return "elementwise_select";
  case ComputeKind::ElementwiseMin:        return "elementwise_min";
  case ComputeKind::ReductionMax:          return "reduction_max";
  case ComputeKind::ReductionMin:          return "reduction_min";
  case ComputeKind::ReductionMul:          return "reduction_mul";
  case ComputeKind::Unknown:
    return "unknown";
  }
  return "unknown";
}

AscendBackendSupportMatrix::AscendBackendSupportMatrix(
    const BackendCapabilityProvider &provider)
    : provider(provider) {}

bool AscendBackendSupportMatrix::isSupportedMovementPath(
    MemorySpace source, MemorySpace target) const {
  return provider.supportsMovementPath(source, target);
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
  return provider.supportsComputeKind(kind);
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

bool AscendBackendSupportMatrix::isSupportedDtype(
    ComputeKind kind, mlir::ArrayRef<mlir::Type> inputTypes,
    mlir::ArrayRef<mlir::Type> outputTypes) const {
  return provider.supportsDtype(kind, inputTypes, outputTypes);
}

UnsupportedReason AscendBackendSupportMatrix::explainDtype(
    ComputeKind kind, mlir::ArrayRef<mlir::Type> inputTypes,
    mlir::ArrayRef<mlir::Type> outputTypes) const {
  if (isSupportedDtype(kind, inputTypes, outputTypes))
    return {"dtype", ""};
  bool isMatmulKind = kind == ComputeKind::Matmul ||
                      kind == ComputeKind::BatchMatmul;
  for (mlir::Type t : inputTypes) {
    bool inputOk = isMatmulKind ? (t.isF16() || t.isBF16())
                                : (t.isF32() || t.isF16() || t.isBF16());
    if (!inputOk) {
      std::string detail;
      llvm::raw_string_ostream os(detail);
      os << "unsupported input dtype for " << stringifyComputeKind(kind)
         << ": " << t;
      return {"dtype", os.str()};
    }
  }
  for (mlir::Type t : outputTypes) {
    bool outputOk = isMatmulKind ? (t.isF16() || t.isBF16() || t.isF32())
                                 : (t.isF32() || t.isF16() || t.isBF16());
    if (!outputOk) {
      std::string detail;
      llvm::raw_string_ostream os(detail);
      os << "unsupported output dtype for " << stringifyComputeKind(kind)
         << ": " << t;
      return {"dtype", os.str()};
    }
  }
  return {"dtype", "unsupported dtype combination"};
}

} // namespace mlir::afir::ascend::backend
