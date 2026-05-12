//===- AscendRealizePlannerTest.cpp - Ascend realize planner tests ---===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/MemoryRealizationDriver.h"
#include "Conversion/Ascend/Realize/MovementPlanner.h"
#include "Conversion/Ascend/Realize/PlacementPlanner.h"
#include "Conversion/Ascend/Realize/RealizeTypes.h"
#include "Conversion/Ascend/Realize/StaticMemoryPlanner.h"
#include "Target/Ascend/TargetMemoryModel.h"

#include "gtest/gtest.h"

using namespace mlir::afir::ascend::realize;

namespace {

mlir::ascend::TargetProfile makeCompleteTargetProfile() {
  mlir::ascend::TargetProfile profile;
  profile.identity.socVersion = "SyntheticSoC";
  profile.hardware.aiCoreCount = 1;
  profile.hardware.l1SizeBytes = 512 * 1024;
  profile.hardware.ubSizeBytes = 192 * 1024;
  profile.capacityBytes[mlir::ascend::MemoryPlace::GM] = 1024 * 1024 * 1024;
  profile.capacityBytes[mlir::ascend::MemoryPlace::A1] = 512 * 1024;
  profile.capacityBytes[mlir::ascend::MemoryPlace::B1] = 512 * 1024;
  profile.capacityBytes[mlir::ascend::MemoryPlace::A2] = 64 * 1024;
  profile.capacityBytes[mlir::ascend::MemoryPlace::B2] = 64 * 1024;
  profile.capacityBytes[mlir::ascend::MemoryPlace::CO1] = 128 * 1024;
  profile.capacityBytes[mlir::ascend::MemoryPlace::VECIN] = 192 * 1024;
  profile.capacityBytes[mlir::ascend::MemoryPlace::VECOUT] = 192 * 1024;
  profile.capacityBytes[mlir::ascend::MemoryPlace::VECCALC] = 192 * 1024;
  return profile;
}

BufferizedKernelIR makeBufferizedKernelIR() {
  BufferizedKernelIR ir;
  ir.kernelId = "kernel_0";
  ir.mode = "tensor_facts";
  ir.inputValueCount = 2;
  ir.outputValueCount = 1;
  ir.temporaryValueCount = 1;
  ir.vectorTemporaryValueCount = 1;
  ir.bufferValueCount = 4;
  return ir;
}

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

TEST(AscendRealizePlannerTest, PlacementPlannerBuildsTargetAwareVecCalcPlan) {
  llvm::raw_null_ostream os;
  auto memoryModel =
      mlir::ascend::TargetMemoryModelBuilder().build(makeCompleteTargetProfile(),
                                                     os);
  ASSERT_TRUE(llvm::succeeded(memoryModel));

  PlacementPlanner planner;
  auto plan = planner.build(makeBufferizedKernelIR(), *memoryModel);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->kernelId, "kernel_0");
  EXPECT_EQ(plan->mode, "target_aware");
  EXPECT_EQ(plan->selectedPlaceCount, 4u);
  EXPECT_EQ(plan->gmPlaceCount, 3u);
  EXPECT_EQ(plan->onChipPlaceCount, 1u);
  EXPECT_EQ(plan->deferredLocalPlaceCount, 0u);
}

TEST(AscendRealizePlannerTest, PlacementPlannerKeepsNonVectorTemporariesInGm) {
  llvm::raw_null_ostream os;
  auto memoryModel =
      mlir::ascend::TargetMemoryModelBuilder().build(makeCompleteTargetProfile(),
                                                     os);
  ASSERT_TRUE(llvm::succeeded(memoryModel));

  BufferizedKernelIR ir = makeBufferizedKernelIR();
  ir.vectorTemporaryValueCount = 0;

  PlacementPlanner planner;
  auto plan = planner.build(ir, *memoryModel);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->mode, "target_aware");
  EXPECT_EQ(plan->selectedPlaceCount, 4u);
  EXPECT_EQ(plan->gmPlaceCount, 4u);
  EXPECT_EQ(plan->onChipPlaceCount, 0u);
  EXPECT_EQ(plan->deferredLocalPlaceCount, 1u);
}

TEST(AscendRealizePlannerTest, PlacementPlannerFallsBackWhenVecCalcUnsupported) {
  mlir::ascend::TargetMemoryModel emptyMemoryModel;

  PlacementPlanner planner;
  auto plan = planner.build(makeBufferizedKernelIR(), emptyMemoryModel);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->mode, "gm_default");
  EXPECT_EQ(plan->selectedPlaceCount, 4u);
  EXPECT_EQ(plan->gmPlaceCount, 4u);
  EXPECT_EQ(plan->onChipPlaceCount, 0u);
  EXPECT_EQ(plan->deferredLocalPlaceCount, 1u);
}

TEST(AscendRealizePlannerTest,
     StaticMemoryPlannerBuildsWorkspaceLayoutForOnChipPlaces) {
  PlacementPlan placement = makePlacementPlan();
  placement.mode = "target_aware";
  placement.gmPlaceCount = 3;
  placement.onChipPlaceCount = 1;
  placement.deferredLocalPlaceCount = 0;

  StaticMemoryPlanner planner;
  auto plan = planner.build(placement);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->kernelId, "kernel_0");
  EXPECT_EQ(plan->mode, "workspace_layout");
  EXPECT_EQ(plan->trackedPlaceCount, 4u);
  EXPECT_EQ(plan->localBufferCount, 1u);
  EXPECT_EQ(plan->liveIntervalCount, 1u);
  EXPECT_EQ(plan->workspaceSlotCount, 1u);
  EXPECT_TRUE(plan->peakUsageKnown);
  EXPECT_EQ(plan->peakUsageUnitCount, 1u);
  EXPECT_TRUE(plan->capacityCheckDeferred);
}

TEST(AscendRealizePlannerTest, MovementPlannerBuildsPlanningForOnChipWorkspace) {
  PlacementPlan placement = makePlacementPlan();
  placement.mode = "target_aware";
  placement.gmPlaceCount = 3;
  placement.onChipPlaceCount = 1;
  placement.deferredLocalPlaceCount = 0;

  StaticMemoryPlan staticMemory = makeStaticMemoryPlan();
  staticMemory.mode = "workspace_layout";
  staticMemory.localBufferCount = 1;
  staticMemory.liveIntervalCount = 1;
  staticMemory.workspaceSlotCount = 1;
  staticMemory.peakUsageKnown = true;
  staticMemory.peakUsageUnitCount = 1;
  staticMemory.capacityCheckDeferred = true;

  MovementPlanner planner;
  auto plan = planner.build(placement, staticMemory);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->kernelId, "kernel_0");
  EXPECT_EQ(plan->mode, "movement_planning");
  EXPECT_EQ(plan->crossPlaceEdgeCount, 1u);
  EXPECT_EQ(plan->movementDemandCount, 1u);
  EXPECT_EQ(plan->selectedPathCount, 0u);
  EXPECT_EQ(plan->pathSelectionDeferredCount, 1u);
  EXPECT_EQ(plan->workspaceReuseCandidateCount, 1u);
  EXPECT_EQ(plan->movementCount, 0u);
  EXPECT_EQ(plan->redundantMovementCount, 0u);
  EXPECT_TRUE(plan->materializationDeferred);
}

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
  EXPECT_EQ(plan->movementDemandCount, 0u);
  EXPECT_EQ(plan->selectedPathCount, 0u);
  EXPECT_EQ(plan->pathSelectionDeferredCount, 0u);
  EXPECT_EQ(plan->workspaceReuseCandidateCount, 0u);
  EXPECT_FALSE(plan->materializationDeferred);
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

TEST(AscendRealizePlannerTest, MemoryRealizationAnnotatesPlanForMemorySpaces) {
  MemoryRealizationPlan plan;
  plan.kernelId = "kernel_0";
  MemoryRealizationDriver driver;
  Phase5BridgeMaterializationCounts materializationCounts;

  driver.markMemorySpaceMaterialized(plan, 1, materializationCounts);

  EXPECT_EQ(plan.kernelId, "kernel_0");
  EXPECT_EQ(plan.mode, "memory_space_annotate");
  EXPECT_EQ(plan.verificationScope, "memory_space_annotation");
  EXPECT_TRUE(plan.planIdsVerified);
  EXPECT_TRUE(plan.frozen);
  EXPECT_EQ(plan.memorySpaceAnnotationCount, 1u);
  EXPECT_EQ(plan.materializedAllocCount, 0u);
  EXPECT_EQ(plan.materializedCopyCount, 0u);
}
