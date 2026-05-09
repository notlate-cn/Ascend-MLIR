//===- Attributes.h - Ascend shared attribute names ---------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCEND_COMMON_ATTRIBUTES_H
#define ASCEND_MLIR_CONVERSION_ASCEND_COMMON_ATTRIBUTES_H

#include "llvm/ADT/StringRef.h"

namespace mlir::afir::ascend {

inline constexpr llvm::StringLiteral kNormalizedAttr = "ascend.normalized";
inline constexpr llvm::StringLiteral kOpRoleAttr = "ascend.op_role";
inline constexpr llvm::StringLiteral kOpRolesAttr = "ascend.op_roles";
inline constexpr llvm::StringLiteral kKernelAttr = "ascend.kernel";
inline constexpr llvm::StringLiteral kPrimaryAttr = "ascend.primary";
inline constexpr llvm::StringLiteral kScheduleDecisionIdAttr =
    "ascend.schedule.decision_id";
inline constexpr llvm::StringLiteral kStructuredLoweringAttr =
    "ascend.schedule.structured_lowering";

} // namespace mlir::afir::ascend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_COMMON_ATTRIBUTES_H
