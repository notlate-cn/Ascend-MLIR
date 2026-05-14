//===- AscendCommonAttributesTest.cpp - Ascend attr tests -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Common/Attributes.h"
#include "Conversion/Ascend/Kernelize/KernelizeTypes.h"
#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "Conversion/Ascend/Schedule/ScheduleTypes.h"

#include "gtest/gtest.h"

namespace ascend = mlir::afir::ascend;

TEST(AscendCommonAttributesTest, SharedConstantsMatchLayerNamespaces) {
  EXPECT_EQ(ascend::kernelize::kNormalizedAttr, ascend::kNormalizedAttr);
  EXPECT_EQ(ascend::kernelize::kOpRoleAttr, ascend::kOpRoleAttr);
  EXPECT_EQ(ascend::schedule::kOpRoleAttr, ascend::kOpRoleAttr);
  EXPECT_EQ(ascend::kernelize::kPrimaryAttr, ascend::kPrimaryAttr);
  EXPECT_EQ(ascend::schedule::kPrimaryAttr, ascend::kPrimaryAttr);
  EXPECT_EQ(ascend::kernelize::kKernelAttr, ascend::kKernelAttr);
  EXPECT_EQ(ascend::schedule::kKernelAttr, ascend::kKernelAttr);
  EXPECT_EQ(ascend::realize::kKernelAttr, ascend::kKernelAttr);
  EXPECT_EQ(ascend::schedule::kScheduleDecisionIdAttr,
            ascend::kScheduleDecisionIdAttr);
  EXPECT_EQ(ascend::realize::kScheduleDecisionIdAttr,
            ascend::kScheduleDecisionIdAttr);
  EXPECT_EQ(ascend::schedule::kStructuredLoweringAttr,
            ascend::kStructuredLoweringAttr);
  EXPECT_EQ(ascend::realize::kStructuredLoweringAttr,
            ascend::kStructuredLoweringAttr);
  EXPECT_EQ(ascend::schedule::kScheduleSelectedTileShapeAttr,
            ascend::kScheduleSelectedTileShapeAttr);
  EXPECT_EQ(ascend::schedule::kScheduleTailPoliciesAttr,
            ascend::kScheduleTailPoliciesAttr);
  EXPECT_EQ(ascend::schedule::kScheduleTailPlanAttr,
            ascend::kScheduleTailPlanAttr);
}

TEST(AscendCommonAttributesTest, SharedBackendContractStringsAreCentralized) {
  EXPECT_EQ(ascend::kAscendCUnitAttr, "ascendc.unit");
  EXPECT_EQ(ascend::kAscendCUnitCube, "AiCore.Cube");
  EXPECT_EQ(ascend::kAscendCUnitVector, "AiCore.Vector");
  EXPECT_EQ(ascend::kAscendCKernelKindAttr, "ascendc.kernel_kind");
  EXPECT_EQ(ascend::kAscendCKernelKindVec, "vec");
  EXPECT_EQ(ascend::kAscendCKernelKindCube, "cube");
  EXPECT_EQ(ascend::kAscendCKernelKindMix, "mix");
  EXPECT_EQ(ascend::kOpRoleVector, "vector");
  EXPECT_EQ(ascend::kOpRoleCube, "cube");
  EXPECT_EQ(ascend::kOpRoleReduction, "reduction");
  EXPECT_EQ(ascend::kOpRoleMemory, "memory");
  EXPECT_EQ(ascend::kOpRoleUnsupported, "unsupported");
  EXPECT_EQ(ascend::kKernelizeOpRoleVector, "Vector");
  EXPECT_EQ(ascend::kKernelizeOpRoleCube, "Cube");
  EXPECT_EQ(ascend::kGatherDimAttr, "gather_dim");
  EXPECT_EQ(ascend::kEmbeddingDimAttr, "embedding_dim");
}
