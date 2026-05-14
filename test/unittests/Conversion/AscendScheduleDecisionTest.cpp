//===- AscendScheduleDecisionTest.cpp - Schedule decision tests ----------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Schedule/ScheduleDecision.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::afir::ascend::schedule;

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

  ScheduleTemplate tmpl{"vector_generic", "single_tile_per_block",
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
