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
inline constexpr llvm::StringLiteral kOpRoleVector = "vector";
inline constexpr llvm::StringLiteral kOpRoleCube = "cube";
inline constexpr llvm::StringLiteral kOpRoleReduction = "reduction";
inline constexpr llvm::StringLiteral kOpRoleMemory = "memory";
inline constexpr llvm::StringLiteral kOpRoleUnsupported = "unsupported";
inline constexpr llvm::StringLiteral kKernelizeOpRoleVector = "Vector";
inline constexpr llvm::StringLiteral kKernelizeOpRoleCube = "Cube";
inline constexpr llvm::StringLiteral kKernelAttr = "ascend.kernel";
inline constexpr llvm::StringLiteral kKernelGraphEdgesAttr =
    "ascend.kernel_graph.edges";
inline constexpr llvm::StringLiteral kKernelizeHandwrittenGroupAttr =
    "ascend.kernelize.handwritten_group";
inline constexpr llvm::StringLiteral kKernelizeHandwrittenKindAttr =
    "ascend.kernelize.handwritten_kind";
inline constexpr llvm::StringLiteral kKernelizeHandwrittenKindAttentionSdpa =
    "attention_sdpa";
inline constexpr llvm::StringLiteral kKernelizeMustCoLocateGroupAttr =
    "ascend.kernelize.must_colocate_group";
inline constexpr llvm::StringLiteral kKernelizeMustSeparateGroupAttr =
    "ascend.kernelize.must_separate_group";
inline constexpr llvm::StringLiteral kKernelizeTemplateFamiliesAttr =
    "ascend.kernelize.template_families";
inline constexpr llvm::StringLiteral kPrimaryAttr = "ascend.primary";
inline constexpr llvm::StringLiteral kScheduleDecisionIdAttr =
    "ascend.schedule.decision_id";
inline constexpr llvm::StringLiteral kStructuredLoweringAttr =
    "ascend.schedule.structured_lowering";
inline constexpr llvm::StringLiteral kScheduleSelectedTileShapeAttr =
    "ascend.schedule.selected_tile_shape";
inline constexpr llvm::StringLiteral kScheduleGuardMarkersAttr =
    "ascend.schedule.guard_markers";
inline constexpr llvm::StringLiteral kScheduleTailPoliciesAttr =
    "ascend.schedule.tail_policies";
inline constexpr llvm::StringLiteral kScheduleTailPlanAttr =
    "ascend.schedule.tail_plan";
inline constexpr llvm::StringLiteral kScheduleTailMarkersAttr =
    "ascend.schedule.tail_markers";
inline constexpr llvm::StringLiteral kScheduleTargetTilePolicyAttr =
    "ascend.schedule.target_tile_policy";
inline constexpr llvm::StringLiteral kScheduleKernelMetadataAttr =
    "ascend.schedule.kernel_metadata";
inline constexpr llvm::StringLiteral kAscendCUnitAttr = "ascendc.unit";
inline constexpr llvm::StringLiteral kAscendCUnitCube = "AiCore.Cube";
inline constexpr llvm::StringLiteral kAscendCUnitVector = "AiCore.Vector";
inline constexpr llvm::StringLiteral kAscendCKernelKindAttr =
    "ascendc.kernel_kind";
inline constexpr llvm::StringLiteral kAscendCKernelKindVec = "vec";
inline constexpr llvm::StringLiteral kAscendCKernelKindCube = "cube";
inline constexpr llvm::StringLiteral kAscendCKernelKindMix = "mix";
inline constexpr llvm::StringLiteral kGatherDimAttr = "gather_dim";
inline constexpr llvm::StringLiteral kEmbeddingDimAttr = "embedding_dim";

} // namespace mlir::afir::ascend

#endif // ASCEND_MLIR_CONVERSION_ASCEND_COMMON_ATTRIBUTES_H
