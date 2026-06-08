//===- AscendCOpCapabilityRegistry.h - AscendC op capabilities -*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_CAPABILITIES_ASCENDC_OP_CAPABILITY_REGISTRY_H
#define ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_CAPABILITIES_ASCENDC_OP_CAPABILITY_REGISTRY_H

#include "Conversion/Ascend/Translate/KernelIR/Capabilities/BackendSupportMatrix.h"

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::ascend::backend {

enum class AscendCOpKind {
  Mmad,
};

enum class AscendCTargetUnit {
  Vector,
  Cube,
};

enum class AscendCOperandRole {
  OutputAccumulator,
  Lhs,
  Rhs,
};

enum class AscendCDtype {
  F16,
  BF16,
  F32,
};

struct AscendCOperandDtypeRule {
  llvm::StringRef name;
  AscendCOperandRole role;
  llvm::ArrayRef<AscendCDtype> allowedDtypes;
  llvm::ArrayRef<AscendCDtype> f32LegalizationTargets;
};

struct AscendCOpCapability {
  AscendCOpKind kind;
  llvm::StringRef opName;
  ComputeKind computeKind;
  AscendCTargetUnit targetUnit;
  llvm::ArrayRef<AscendCOperandDtypeRule> dtypeOperands;
};

const AscendCOpCapability *lookupAscendCOpCapability(AscendCOpKind kind);

bool isAscendCOperandDtypeAllowed(const AscendCOperandDtypeRule &rule,
                                  mlir::Type type);

void appendAscendCOperandLegalizationTargets(
    const AscendCOperandDtypeRule &rule, mlir::Type sourceType,
    llvm::SmallVectorImpl<mlir::Type> &targets);

} // namespace mlir::ascend::backend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_TRANSLATE_KERNELIR_CAPABILITIES_ASCENDC_OP_CAPABILITY_REGISTRY_H
