//===- AscendScheduleDecisionTest.cpp - Schedule decision tests ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/ScheduleDecision.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::ascend::schedule;

TEST(AscendScheduleDecisionTest, TailPolicyPreferenceComesFromTargetPolicy) {
  ScheduleProblem problem;
  problem.kernelId = "kernel_0";
  problem.targetTilePolicy.tailPolicyPreference = {
      AxisTailPolicy::PadAndMask, AxisTailPolicy::MaskedTail,
      AxisTailPolicy::ScalarEpilogue, AxisTailPolicy::FullExtent,
      AxisTailPolicy::MustDivide};

  LogicalAxisInfo axis;
  axis.logicalAxisId = 0;
  axis.kind = AxisKind::Parallel;
  axis.staticExtent = 17;
  problem.axes.logicalAxes.push_back(axis);

  AxisScheduleConstraint constraint;
  constraint.logicalAxisId = 0;
  constraint.kind = AxisKind::Parallel;
  constraint.allowedTailPolicies.push_back(AxisTailPolicy::MaskedTail);
  constraint.allowedTailPolicies.push_back(AxisTailPolicy::PadAndMask);
  constraint.primitiveUses.push_back(PrimitiveAxisUseKind::GatherIndex);
  problem.axes.axisScheduleConstraints.push_back(std::move(constraint));

  ScheduleTemplate tmpl{kScheduleFamilyVectorGeneric.str(),
                        kScheduleTemplateSingleTilePerBlock.str(),
                        {"vector"}, 1, 8, 0};
  ScheduleInstance instance;
  instance.instanceId = "kernel_0.vector_generic.0";
  instance.tmpl = tmpl;
  instance.tileShape.tileSizes.push_back(8);

  ScheduleDecisionSet decisions = buildScheduleDecisionSet(problem, instance);
  ASSERT_EQ(decisions.decisions.size(), 1u);
  ASSERT_EQ(decisions.decisions.front().tailPlans.size(), 1u);
  EXPECT_EQ(decisions.decisions.front().tailPlans.front().selectedPolicy,
            AxisTailPolicy::PadAndMask);
}

TEST(AscendScheduleDecisionTest, RuntimeTileParamsDescribeSearchSpace) {
  ScheduleProblem problem;
  problem.kernelId = "kernel_memory";
  problem.dominantRole = OpRole::Memory;
  problem.targetTilePolicy.defaultParallelTile = 32;

  LogicalAxisInfo axis;
  axis.logicalAxisId = 0;
  axis.kind = AxisKind::Parallel;
  axis.staticExtent = 70;
  problem.axes.logicalAxes.push_back(axis);

  AxisScheduleConstraint constraint;
  constraint.logicalAxisId = 0;
  constraint.kind = AxisKind::Parallel;
  constraint.allowedRoles.push_back(AxisExecutionRole::BindCoreCandidate);
  constraint.allowedRoles.push_back(AxisExecutionRole::KernelLoopCandidate);
  constraint.primitiveUses.push_back(PrimitiveAxisUseKind::DataCopy);
  constraint.primitiveUses.push_back(PrimitiveAxisUseKind::VectorCompute);
  constraint.primitiveUses.push_back(PrimitiveAxisUseKind::WriteBack);
  problem.axes.axisScheduleConstraints.push_back(std::move(constraint));

  ScheduleTemplate tmpl{kScheduleFamilyMemoryCopy.str(),
                        kScheduleTemplateSingleTilePerBlock.str(),
                        {"memory"}, 1, 8, 0};
  ScheduleInstance instance;
  instance.instanceId = "kernel_memory.memory_copy.0";
  instance.tmpl = tmpl;
  instance.tileShape.tileSizes.push_back(64);

  ScheduleDecisionSet decisions = buildScheduleDecisionSet(problem, instance);
  ASSERT_EQ(decisions.decisions.size(), 1u);

  const auto &tileParams = decisions.decisions.front().tileParams;
  ASSERT_EQ(tileParams.size(), 1u);
  EXPECT_EQ(tileParams.front().name, "TB_M");
  EXPECT_EQ(tileParams.front().logicalAxisId, 0u);
  EXPECT_EQ(tileParams.front().binding, TileParamBinding::Runtime);
  EXPECT_EQ(tileParams.front().defaultValue, 32);
  EXPECT_EQ(tileParams.front().upperBound, 70);
  EXPECT_EQ(tileParams.front().extent, 70);

  ASSERT_EQ(tileParams.front().primitiveUses.size(), 2u);
  EXPECT_EQ(tileParams.front().primitiveUses[0],
            PrimitiveAxisUseKind::DataCopy);
  EXPECT_EQ(tileParams.front().primitiveUses[1],
            PrimitiveAxisUseKind::WriteBack);

  ASSERT_EQ(decisions.decisions.front().tailPlans.size(), 1u);
  EXPECT_TRUE(ShapedType::isDynamic(
      decisions.decisions.front().tailPlans.front().tileSize));
}
