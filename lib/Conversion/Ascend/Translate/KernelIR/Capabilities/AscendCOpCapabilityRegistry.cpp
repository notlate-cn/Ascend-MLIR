//===- AscendCOpCapabilityRegistry.cpp - AscendC op capabilities ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/AscendCOpCapabilityRegistry.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::ascend::backend {
namespace {

bool matchesDtype(AscendCDtype dtype, Type type) {
  switch (dtype) {
  case AscendCDtype::F16:
    return type.isF16();
  case AscendCDtype::BF16:
    return type.isBF16();
  case AscendCDtype::F32:
    return type.isF32();
  }
  return false;
}

Type materializeDtype(AscendCDtype dtype, MLIRContext *context) {
  switch (dtype) {
  case AscendCDtype::F16:
    return Float16Type::get(context);
  case AscendCDtype::BF16:
    return BFloat16Type::get(context);
  case AscendCDtype::F32:
    return Float32Type::get(context);
  }
  return {};
}

const AscendCDtype kMmadDstAllowed[] = {AscendCDtype::F16, AscendCDtype::BF16,
                                        AscendCDtype::F32};
const AscendCDtype kMmadInputAllowed[] = {AscendCDtype::F16,
                                          AscendCDtype::BF16};
const AscendCDtype kF32ToMmadInputTargets[] = {AscendCDtype::BF16,
                                               AscendCDtype::F16};

const AscendCOperandDtypeRule kMmadOperands[] = {
    {"dst", AscendCOperandRole::OutputAccumulator, kMmadDstAllowed, {}},
    {"fm", AscendCOperandRole::Lhs, kMmadInputAllowed,
     kF32ToMmadInputTargets},
    {"filter", AscendCOperandRole::Rhs, kMmadInputAllowed,
     kF32ToMmadInputTargets},
};

const AscendCOpCapability kMmadCapability = {
    AscendCOpKind::Mmad, "ascendc.mmad", ComputeKind::Matmul,
    AscendCTargetUnit::Cube, kMmadOperands};

} // namespace

const AscendCOpCapability *lookupAscendCOpCapability(AscendCOpKind kind) {
  switch (kind) {
  case AscendCOpKind::Mmad:
    return &kMmadCapability;
  }
  return nullptr;
}

bool isAscendCOperandDtypeAllowed(const AscendCOperandDtypeRule &rule,
                                  Type type) {
  return llvm::any_of(rule.allowedDtypes, [&](AscendCDtype dtype) {
    return matchesDtype(dtype, type);
  });
}

void appendAscendCOperandLegalizationTargets(
    const AscendCOperandDtypeRule &rule, Type sourceType,
    SmallVectorImpl<Type> &targets) {
  if (!sourceType.isF32())
    return;
  MLIRContext *context = sourceType.getContext();
  for (AscendCDtype dtype : rule.f32LegalizationTargets)
    targets.push_back(materializeDtype(dtype, context));
}

} // namespace mlir::ascend::backend
