//===- AscendScheduleTemplateRegistryTest.cpp - Schedule template tests ---===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/TemplateRegistry.h"
#include "Conversion/Ascend/Schedule/ScheduleTemplateImplementation.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::ascend::schedule;

TEST(AscendScheduleTemplateRegistryTest, RegistryReturnsTemplateImplementations) {
  ScheduleProblem problem;
  problem.kernelId = "kernel_0";
  problem.dominantRole = OpRole::Vector;
  problem.resultRank = 2;
  problem.resultShape = {70, 128};
  problem.templateTags.push_back(kOpRoleVector.str());

  LogicalAxisInfo axis0;
  axis0.logicalAxisId = 0;
  axis0.kind = AxisKind::Parallel;
  axis0.staticExtent = 70;
  problem.axes.logicalAxes.push_back(axis0);

  LogicalAxisInfo axis1;
  axis1.logicalAxisId = 1;
  axis1.kind = AxisKind::Parallel;
  axis1.staticExtent = 128;
  problem.axes.logicalAxes.push_back(axis1);

  AxisScheduleConstraint constraint0;
  constraint0.logicalAxisId = 0;
  constraint0.kind = AxisKind::Parallel;
  constraint0.allowedRoles.push_back(AxisExecutionRole::BindCoreCandidate);
  constraint0.allowedRoles.push_back(AxisExecutionRole::KernelLoopCandidate);
  problem.axes.axisScheduleConstraints.push_back(constraint0);

  AxisScheduleConstraint constraint1;
  constraint1.logicalAxisId = 1;
  constraint1.kind = AxisKind::Parallel;
  constraint1.allowedRoles.push_back(AxisExecutionRole::VectorizeCandidate);
  problem.axes.axisScheduleConstraints.push_back(constraint1);

  SmallVector<const ScheduleTemplateImplementation *> implementations =
      matchScheduleTemplateImplementations(problem);

  ASSERT_EQ(implementations.size(), 1u);
  const ScheduleTemplate &metadata = implementations.front()->metadata();
  EXPECT_EQ(metadata.family, kScheduleFamilyVectorGeneric);
  EXPECT_EQ(metadata.name, kScheduleTemplateSingleTilePerBlock);
  EXPECT_EQ(implementations.front()->implementationKind(),
            kScheduleTemplateSingleTilePerBlock);
  EXPECT_EQ(implementations.front()->description(),
            "one runtime tile per block");

  SmallVector<TileShape> tileShapes =
      implementations.front()->generateTileShapes(problem,
                                                  ScheduleSearchOptions{});
  ASSERT_FALSE(tileShapes.empty());
  ASSERT_EQ(tileShapes.front().tileSizes.size(), 2u);
  EXPECT_EQ(tileShapes.front().tileSizes[0], 32);
  EXPECT_EQ(tileShapes.front().tileSizes[1], 128);
}

TEST(AscendScheduleTemplateRegistryTest, HandwrittenAttentionUsesGroupedImplementation) {
  ScheduleProblem problem;
  problem.kernelId = "kernel_attention";
  problem.dominantRole = OpRole::Cube;
  problem.resultRank = 3;
  problem.resultShape = {2, 4, 8};
  problem.templateTags.push_back(kKernelizeHandwrittenKindAttentionSdpa.str());
  problem.templateTags.push_back(kOpRoleCube.str());

  SmallVector<const ScheduleTemplateImplementation *> implementations =
      matchScheduleTemplateImplementations(problem);

  const ScheduleTemplateImplementation *attention = nullptr;
  for (const ScheduleTemplateImplementation *implementation :
       implementations) {
    if (implementation->metadata().family ==
        kKernelizeHandwrittenKindAttentionSdpa) {
      attention = implementation;
      break;
    }
  }

  ASSERT_NE(attention, nullptr);
  EXPECT_EQ(attention->metadata().name, kScheduleTemplateGroupedTilePerBlock);
  EXPECT_EQ(attention->implementationKind(), kScheduleTemplateGroupedTilePerBlock);
  EXPECT_EQ(attention->description(), "grouped runtime tiles per block");
}
