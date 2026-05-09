//===- Attributes.h - Ascend V2 shared attribute names ---------*- C++ -*-===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MLIR_CONVERSION_ASCENDV2_COMMON_ATTRIBUTES_H
#define ASCEND_MLIR_CONVERSION_ASCENDV2_COMMON_ATTRIBUTES_H

#include "llvm/ADT/StringRef.h"

namespace mlir::afir::ascend::v2 {

inline constexpr llvm::StringLiteral kNormalizedAttr = "ascend.v2.normalized";
inline constexpr llvm::StringLiteral kOpRoleAttr = "ascend.v2.op_role";
inline constexpr llvm::StringLiteral kOpRolesAttr = "ascend.v2.op_roles";
inline constexpr llvm::StringLiteral kKernelAttr = "ascend.v2.kernel";
inline constexpr llvm::StringLiteral kPrimaryAttr = "ascend.v2.primary";
inline constexpr llvm::StringLiteral kScheduleDecisionIdAttr =
    "ascend.v2.schedule.decision_id";
inline constexpr llvm::StringLiteral kStructuredLoweringAttr =
    "ascend.v2.schedule.structured_lowering";

} // namespace mlir::afir::ascend::v2

#endif // ASCEND_MLIR_CONVERSION_ASCENDV2_COMMON_ATTRIBUTES_H
