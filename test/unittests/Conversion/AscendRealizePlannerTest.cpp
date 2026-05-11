//===- AscendRealizePlannerTest.cpp - Ascend realize planner tests ---===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/MemoryRealizationDriver.h"
#include "Conversion/Ascend/Realize/MovementPlanner.h"
#include "Conversion/Ascend/Realize/RealizeTypes.h"

#include "gtest/gtest.h"

using namespace mlir::afir::ascend::realize;

namespace {

PlacementPlan makePlacementPlan() {
  PlacementPlan plan;
  plan.kernelId = "kernel_0";
  plan.mode = "gm_default";
  plan.selectedPlaceCount = 4;
  plan.gmPlaceCount = 4;
  return plan;
}

StaticMemoryPlan makeStaticMemoryPlan() {
  StaticMemoryPlan plan;
  plan.kernelId = "kernel_0";
  plan.mode = "empty_workspace";
  plan.trackedPlaceCount = 4;
  plan.workspaceSlotCount = 0;
  return plan;
}

MovementPlan makeMovementPlan() {
  MovementPlan plan;
  plan.kernelId = "kernel_0";
  plan.mode = "gm_noop";
  return plan;
}

} // namespace

TEST(AscendRealizePlannerTest, MovementPlannerRejectsMismatchedKernelId) {
  PlacementPlan placement = makePlacementPlan();
  StaticMemoryPlan staticMemory = makeStaticMemoryPlan();
  staticMemory.kernelId = "kernel_1";

  MovementPlanner planner;
  EXPECT_TRUE(llvm::failed(planner.build(placement, staticMemory)));
}

TEST(AscendRealizePlannerTest, MovementPlannerRejectsMismatchedTrackedPlaces) {
  PlacementPlan placement = makePlacementPlan();
  StaticMemoryPlan staticMemory = makeStaticMemoryPlan();
  staticMemory.trackedPlaceCount = placement.selectedPlaceCount + 1;

  MovementPlanner planner;
  EXPECT_TRUE(llvm::failed(planner.build(placement, staticMemory)));
}

TEST(AscendRealizePlannerTest, MovementPlannerBuildsWithFutureStaticMode) {
  PlacementPlan placement = makePlacementPlan();
  StaticMemoryPlan staticMemory = makeStaticMemoryPlan();
  staticMemory.mode = "packed_workspace";

  MovementPlanner planner;
  auto plan = planner.build(placement, staticMemory);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->kernelId, "kernel_0");
  EXPECT_EQ(plan->mode, "gm_noop");
}

TEST(AscendRealizePlannerTest, MovementPlannerBuildsWithWorkspaceSlots) {
  PlacementPlan placement = makePlacementPlan();
  StaticMemoryPlan staticMemory = makeStaticMemoryPlan();
  staticMemory.workspaceSlotCount = 1;

  MovementPlanner planner;
  auto plan = planner.build(placement, staticMemory);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->kernelId, "kernel_0");
  EXPECT_EQ(plan->mode, "gm_noop");
}

TEST(AscendRealizePlannerTest, MovementPlannerBuildsWithKnownPeakUsage) {
  PlacementPlan placement = makePlacementPlan();
  StaticMemoryPlan staticMemory = makeStaticMemoryPlan();
  staticMemory.peakUsageKnown = true;

  MovementPlanner planner;
  auto plan = planner.build(placement, staticMemory);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->kernelId, "kernel_0");
  EXPECT_EQ(plan->mode, "gm_noop");
}

TEST(AscendRealizePlannerTest, MovementPlannerBuildsGmNoopPlan) {
  PlacementPlan placement = makePlacementPlan();
  StaticMemoryPlan staticMemory = makeStaticMemoryPlan();

  MovementPlanner planner;
  auto plan = planner.build(placement, staticMemory);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->kernelId, "kernel_0");
  EXPECT_EQ(plan->mode, "gm_noop");
  EXPECT_EQ(plan->crossPlaceEdgeCount, 0u);
  EXPECT_EQ(plan->movementCount, 0u);
  EXPECT_EQ(plan->redundantMovementCount, 0u);
}

TEST(AscendRealizePlannerTest, MemoryRealizationRejectsMismatchedKernelIds) {
  PlacementPlan placement = makePlacementPlan();
  StaticMemoryPlan staticMemory = makeStaticMemoryPlan();
  MovementPlan movement = makeMovementPlan();
  movement.kernelId = "kernel_1";

  MemoryRealizationDriver driver;
  EXPECT_TRUE(
      llvm::failed(driver.materialize(placement, staticMemory, movement)));
}

TEST(AscendRealizePlannerTest, MemoryRealizationBuildsReadOnlyFreezePlan) {
  PlacementPlan placement = makePlacementPlan();
  StaticMemoryPlan staticMemory = makeStaticMemoryPlan();
  MovementPlan movement = makeMovementPlan();

  MemoryRealizationDriver driver;
  auto plan = driver.materialize(placement, staticMemory, movement);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->kernelId, "kernel_0");
  EXPECT_EQ(plan->mode, "read_only_freeze");
  EXPECT_EQ(plan->verificationScope, "plan_identity_only");
  EXPECT_TRUE(plan->planIdsVerified);
  EXPECT_TRUE(plan->frozen);
  EXPECT_EQ(plan->materializedAllocCount, 0u);
  EXPECT_EQ(plan->materializedCopyCount, 0u);
}
