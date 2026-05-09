//===- AscendV2CommonAttributesTest.cpp - Ascend V2 attr tests -----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/AscendV2/Common/Attributes.h"
#include "Conversion/AscendV2/Kernelize/KernelizeTypes.h"
#include "Conversion/AscendV2/Realize/RealizeTypes.h"
#include "Conversion/AscendV2/Schedule/ScheduleTypes.h"

#include "gtest/gtest.h"

namespace v2 = mlir::afir::ascend::v2;

TEST(AscendV2CommonAttributesTest, SharedConstantsMatchLayerNamespaces) {
  EXPECT_EQ(v2::kernelize::kNormalizedAttr, v2::kNormalizedAttr);
  EXPECT_EQ(v2::kernelize::kOpRoleAttr, v2::kOpRoleAttr);
  EXPECT_EQ(v2::schedule::kOpRoleAttr, v2::kOpRoleAttr);
  EXPECT_EQ(v2::kernelize::kPrimaryAttr, v2::kPrimaryAttr);
  EXPECT_EQ(v2::schedule::kPrimaryAttr, v2::kPrimaryAttr);
  EXPECT_EQ(v2::kernelize::kKernelAttr, v2::kKernelAttr);
  EXPECT_EQ(v2::schedule::kKernelAttr, v2::kKernelAttr);
  EXPECT_EQ(v2::realize::kKernelAttr, v2::kKernelAttr);
  EXPECT_EQ(v2::schedule::kScheduleDecisionIdAttr,
            v2::kScheduleDecisionIdAttr);
  EXPECT_EQ(v2::realize::kScheduleDecisionIdAttr,
            v2::kScheduleDecisionIdAttr);
  EXPECT_EQ(v2::schedule::kStructuredLoweringAttr,
            v2::kStructuredLoweringAttr);
  EXPECT_EQ(v2::realize::kStructuredLoweringAttr,
            v2::kStructuredLoweringAttr);
}
