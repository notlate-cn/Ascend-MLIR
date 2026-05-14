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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

using namespace mlir;
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

OwningOpRef<ModuleOp> parseRealizeModule(MLIRContext &context,
                                         StringRef moduleText) {
  context.loadDialect<arith::ArithDialect, func::FuncDialect,
                      linalg::LinalgDialect, memref::MemRefDialect>();
  return parseSourceString<ModuleOp>(moduleText, &context);
}

RealizePlanBundle makePlanBundle() {
  RealizePlanBundle bundle;
  bundle.kernel.kernelId = "kernel_0";
  bundle.kernel.decisionId = "kernel_0.decision.0";
  bundle.kernel.structuredLowering = "loop_skeleton_v0";
  bundle.kernel.scheduledOps = 1;
  bundle.bufferizedIR = makeBufferizedKernelIR();
  bundle.placement = makePlacementPlan();
  bundle.staticMemory = makeStaticMemoryPlan();
  bundle.movement = makeMovementPlan();
  MemoryRealizationDriver driver;
  auto realization =
      driver.materialize(bundle.placement, bundle.staticMemory,
                         bundle.movement);
  assert(succeeded(realization));
  bundle.realization = *realization;
  return bundle;
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

TEST(AscendRealizePlannerTest, RealizeMemoryPlaceUsesTargetProfileValues) {
  EXPECT_EQ(static_cast<int64_t>(MemoryPlace::GM),
            static_cast<int64_t>(mlir::ascend::MemoryPlace::GM));
  EXPECT_EQ(static_cast<int64_t>(MemoryPlace::A1),
            static_cast<int64_t>(mlir::ascend::MemoryPlace::A1));
  EXPECT_EQ(static_cast<int64_t>(MemoryPlace::A2),
            static_cast<int64_t>(mlir::ascend::MemoryPlace::A2));
  EXPECT_EQ(static_cast<int64_t>(MemoryPlace::B1),
            static_cast<int64_t>(mlir::ascend::MemoryPlace::B1));
  EXPECT_EQ(static_cast<int64_t>(MemoryPlace::B2),
            static_cast<int64_t>(mlir::ascend::MemoryPlace::B2));
  EXPECT_EQ(static_cast<int64_t>(MemoryPlace::CO1),
            static_cast<int64_t>(mlir::ascend::MemoryPlace::CO1));
  EXPECT_EQ(static_cast<int64_t>(MemoryPlace::VECIN),
            static_cast<int64_t>(mlir::ascend::MemoryPlace::VECIN));
  EXPECT_EQ(static_cast<int64_t>(MemoryPlace::VECOUT),
            static_cast<int64_t>(mlir::ascend::MemoryPlace::VECOUT));
  EXPECT_EQ(static_cast<int64_t>(MemoryPlace::VECCALC),
            static_cast<int64_t>(mlir::ascend::MemoryPlace::VECCALC));
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

TEST(AscendRealizePlannerTest, MemoryRealizationMaterializeMutatesIRAndPlan) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseRealizeModule(
      context, R"mlir(
module {
  func.func @f(%arg0: memref<4x8xf32>, %arg1: memref<4x8xf32>) -> memref<4x8xf32>
      attributes {ascend.normalized = true} {
    %out = memref.alloc() {alignment = 64 : i64} : memref<4x8xf32>
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<4x8xf32>, memref<4x8xf32>)
      outs(%out : memref<4x8xf32>)
      attrs = {ascend.kernel = "kernel_0",
               ascend.op_role = "vector",
               ascend.schedule.decision_id = "kernel_0.decision.0",
               ascend.schedule.structured_lowering = "loop_skeleton_v0"} {
    ^bb0(%lhs: f32, %rhs: f32, %old: f32):
      %0 = arith.addf %lhs, %rhs : f32
      linalg.yield %0 : f32
    }
    return %out : memref<4x8xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  SmallVector<RealizePlanBundle, 1> bundles;
  bundles.push_back(makePlanBundle());

  MemoryRealizationDriver driver;
  ASSERT_TRUE(succeeded(driver.materialize(
      *module, bundles, MemoryRealizationMode::MemorySpaceAnnotate)));

  ASSERT_EQ(bundles.size(), 1u);
  EXPECT_EQ(bundles[0].realization.mode, "memory_space_materialize");
  EXPECT_EQ(bundles[0].realization.verificationScope,
            "memory_space_materialization");
  EXPECT_EQ(bundles[0].realization.materializedAllocCount, 1u);
  EXPECT_EQ(bundles[0].realization.materializedCopyCount, 1u);

  unsigned vecOutAllocCount = 0;
  module->walk([&](memref::AllocOp allocOp) {
    auto type = cast<MemRefType>(allocOp.getType());
    auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
    if (space && space.getInt() ==
                     static_cast<int64_t>(mlir::ascend::MemoryPlace::VECOUT))
      ++vecOutAllocCount;
  });
  EXPECT_EQ(vecOutAllocCount, 1u);
}
