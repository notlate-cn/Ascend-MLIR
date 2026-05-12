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
}
