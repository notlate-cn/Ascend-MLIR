//===- AscendRealizePlannerTest.cpp - Ascend realize planner tests ---===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/BufferizationDriver.h"
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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
                      linalg::LinalgDialect, memref::MemRefDialect,
                      scf::SCFDialect>();
  return parseSourceString<ModuleOp>(moduleText, &context);
}

OwningOpRef<ModuleOp> parseVectorChainModule(MLIRContext &context) {
  return parseRealizeModule(
      context, R"mlir(
module {
  func.func @f(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>) -> tensor<64xf16>
      attributes {ascend.normalized = true} {
    %empty0 = tensor.empty() : tensor<64xf16>
    %mid = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0, %arg1 : tensor<64xf16>, tensor<64xf16>)
      outs(%empty0 : tensor<64xf16>)
      attrs = {ascend.kernel = "kernel_0",
               ascend.op_role = "vector",
               ascend.schedule.decision_id = "kernel_0.decision.0",
               ascend.schedule.structured_lowering = "loop_skeleton_v0"} {
    ^bb0(%lhs: f16, %rhs: f16, %old: f16):
      %sum = arith.addf %lhs, %rhs : f16
      linalg.yield %sum : f16
    } -> tensor<64xf16>
    %empty1 = tensor.empty() : tensor<64xf16>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%mid : tensor<64xf16>)
      outs(%empty1 : tensor<64xf16>)
      attrs = {ascend.kernel = "kernel_0",
               ascend.op_role = "vector",
               ascend.schedule.decision_id = "kernel_0.decision.0",
               ascend.schedule.structured_lowering = "loop_skeleton_v0"} {
    ^bb0(%x: f16, %old: f16):
      %neg = arith.negf %x : f16
      linalg.yield %neg : f16
    } -> tensor<64xf16>
    return %out : tensor<64xf16>
  }
}
)mlir");
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

TEST(AscendRealizePlannerTest,
     StaticMemoryPlannerComputesStaticBytePeakForVectorTemporary) {
  BufferizedKernelIR ir = makeBufferizedKernelIR();
  ir.staticByteSizeKnown = true;
  ir.vectorTemporaryByteCount = 128;

  PlacementPlan placement = makePlacementPlan();
  placement.mode = "target_aware";
  placement.gmPlaceCount = 3;
  placement.onChipPlaceCount = 1;
  placement.deferredLocalPlaceCount = 0;

  StaticMemoryPlanner planner;
  auto plan = planner.build(placement, ir);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->mode, "workspace_layout");
  EXPECT_TRUE(plan->peakUsageKnown);
  EXPECT_TRUE(plan->peakUsageBytesKnown);
  EXPECT_EQ(plan->localBufferByteCount, 128u);
  EXPECT_EQ(plan->workspaceByteCount, 128u);
  EXPECT_EQ(plan->peakUsageByteCount, 128u);
}

TEST(AscendRealizePlannerTest,
     StaticMemoryPlannerRejectsPeakUsageOverTargetCapacity) {
  mlir::ascend::TargetProfile profile = makeCompleteTargetProfile();
  profile.capacityBytes[mlir::ascend::MemoryPlace::VECCALC] = 64;
  llvm::raw_null_ostream os;
  auto memoryModel =
      mlir::ascend::TargetMemoryModelBuilder().build(profile, os);
  ASSERT_TRUE(llvm::succeeded(memoryModel));

  BufferizedKernelIR ir = makeBufferizedKernelIR();
  ir.staticByteSizeKnown = true;
  ir.vectorTemporaryByteCount = 128;

  PlacementPlan placement = makePlacementPlan();
  placement.mode = "target_aware";
  placement.gmPlaceCount = 3;
  placement.onChipPlaceCount = 1;
  placement.deferredLocalPlaceCount = 0;

  StaticMemoryPlanner planner;
  EXPECT_TRUE(failed(planner.build(placement, ir, *memoryModel)));
}

TEST(AscendRealizePlannerTest,
     BufferizationDriverCollectsStaticByteFacts) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseVectorChainModule(context);
  ASSERT_TRUE(module);

  BufferizationDriver driver;
  FailureOr<SmallVector<BufferizedKernelIR, 4>> facts =
      driver.collectTensorFacts(*module);
  ASSERT_TRUE(succeeded(facts));
  ASSERT_EQ(facts->size(), 1u);
  const BufferizedKernelIR &ir = facts->front();
  EXPECT_EQ(ir.kernelId, "kernel_0");
  EXPECT_TRUE(ir.staticByteSizeKnown);
  EXPECT_EQ(ir.inputByteCount, 256u);
  EXPECT_EQ(ir.outputByteCount, 128u);
  EXPECT_EQ(ir.temporaryByteCount, 128u);
  EXPECT_EQ(ir.vectorTemporaryByteCount, 128u);
}

TEST(AscendRealizePlannerTest, BufferizationDriverBuildsStableValueFacts) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseVectorChainModule(context);
  ASSERT_TRUE(module);

  BufferizationDriver driver;
  FailureOr<SmallVector<BufferizedKernelIR, 4>> facts =
      driver.collectTensorFacts(*module);
  ASSERT_TRUE(succeeded(facts));
  ASSERT_EQ(facts->size(), 1u);
  const BufferizedKernelIR &ir = facts->front();

  ASSERT_EQ(ir.valueFacts.size(), 4u);
  EXPECT_EQ(ir.valueFacts[0].valueId, 0u);
  EXPECT_EQ(ir.valueFacts[0].role, BufferizedValueRole::Input);
  EXPECT_FALSE(ir.valueFacts[0].isVectorTemporary);
  EXPECT_TRUE(ir.valueFacts[0].staticByteSizeKnown);
  EXPECT_EQ(ir.valueFacts[0].byteSize, 128u);

  EXPECT_EQ(ir.valueFacts[1].valueId, 1u);
  EXPECT_EQ(ir.valueFacts[1].role, BufferizedValueRole::Input);
  EXPECT_FALSE(ir.valueFacts[1].isVectorTemporary);
  EXPECT_TRUE(ir.valueFacts[1].staticByteSizeKnown);
  EXPECT_EQ(ir.valueFacts[1].byteSize, 128u);

  EXPECT_EQ(ir.valueFacts[2].valueId, 2u);
  EXPECT_EQ(ir.valueFacts[2].role, BufferizedValueRole::Temporary);
  EXPECT_TRUE(ir.valueFacts[2].isVectorTemporary);
  EXPECT_TRUE(ir.valueFacts[2].staticByteSizeKnown);
  EXPECT_EQ(ir.valueFacts[2].byteSize, 128u);

  EXPECT_EQ(ir.valueFacts[3].valueId, 3u);
  EXPECT_EQ(ir.valueFacts[3].role, BufferizedValueRole::Output);
  EXPECT_FALSE(ir.valueFacts[3].isVectorTemporary);
  EXPECT_TRUE(ir.valueFacts[3].staticByteSizeKnown);
  EXPECT_EQ(ir.valueFacts[3].byteSize, 128u);
}

TEST(AscendRealizePlannerTest,
     StaticMemoryPlannerBuildsValueLevelWorkspaceSlots) {
  BufferizedKernelIR ir = makeBufferizedKernelIR();
  ir.staticByteSizeKnown = true;
  ir.vectorTemporaryByteCount = 128;
  ir.valueFacts = {
      BufferizedValueFact{/*valueId=*/0, BufferizedValueRole::Input,
                          /*isVectorTemporary=*/false,
                          /*staticByteSizeKnown=*/true, /*byteSize=*/128},
      BufferizedValueFact{/*valueId=*/1, BufferizedValueRole::Temporary,
                          /*isVectorTemporary=*/true,
                          /*staticByteSizeKnown=*/true, /*byteSize=*/128},
      BufferizedValueFact{/*valueId=*/2, BufferizedValueRole::Output,
                          /*isVectorTemporary=*/false,
                          /*staticByteSizeKnown=*/true, /*byteSize=*/128}};

  PlacementPlan placement = makePlacementPlan();
  placement.mode = "target_aware";
  placement.gmPlaceCount = 3;
  placement.onChipPlaceCount = 1;
  placement.deferredLocalPlaceCount = 0;

  StaticMemoryPlanner planner;
  auto plan = planner.build(placement, ir);

  ASSERT_TRUE(llvm::succeeded(plan));
  ASSERT_EQ(plan->liveIntervals.size(), 1u);
  EXPECT_EQ(plan->liveIntervals[0].valueId, 1u);
  EXPECT_EQ(plan->liveIntervals[0].start, 0u);
  EXPECT_EQ(plan->liveIntervals[0].end, 1u);
  EXPECT_EQ(plan->liveIntervals[0].place, MemoryPlace::VECCALC);
  EXPECT_TRUE(plan->liveIntervals[0].staticByteSizeKnown);
  EXPECT_EQ(plan->liveIntervals[0].byteSize, 128u);

  ASSERT_EQ(plan->workspaceSlots.size(), 1u);
  EXPECT_EQ(plan->workspaceSlots[0].slotId, 0u);
  EXPECT_EQ(plan->workspaceSlots[0].valueId, 1u);
  EXPECT_EQ(plan->workspaceSlots[0].offset, 0u);
  EXPECT_EQ(plan->workspaceSlots[0].place, MemoryPlace::VECCALC);
  EXPECT_TRUE(plan->workspaceSlots[0].staticByteSizeKnown);
  EXPECT_EQ(plan->workspaceSlots[0].byteSize, 128u);
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

TEST(AscendRealizePlannerTest,
     MovementPlannerBuildsValueLevelStepsForWorkspaceSlots) {
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
  staticMemory.workspaceSlots = {StaticMemoryWorkspaceSlot{
      /*slotId=*/0, /*valueId=*/1, /*offset=*/0, MemoryPlace::VECCALC,
      /*staticByteSizeKnown=*/true, /*byteSize=*/128}};

  MovementPlanner planner;
  auto plan = planner.build(placement, staticMemory);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->movementDemandCount, 1u);
  EXPECT_EQ(plan->selectedPathCount, 0u);
  EXPECT_EQ(plan->pathSelectionDeferredCount, 1u);
  ASSERT_EQ(plan->movementSteps.size(), 1u);
  EXPECT_EQ(plan->movementSteps[0].stepId, 0u);
  EXPECT_EQ(plan->movementSteps[0].valueId, 1u);
  EXPECT_EQ(plan->movementSteps[0].slotId, 0u);
  EXPECT_EQ(plan->movementSteps[0].srcPlace, MemoryPlace::GM);
  EXPECT_EQ(plan->movementSteps[0].dstPlace, MemoryPlace::VECCALC);
  EXPECT_FALSE(plan->movementSteps[0].pathSelected);
  EXPECT_TRUE(plan->movementSteps[0].pathSelectionDeferred);
  EXPECT_TRUE(plan->movementSteps[0].staticByteSizeKnown);
  EXPECT_EQ(plan->movementSteps[0].byteSize, 128u);
}

TEST(AscendRealizePlannerTest,
     MovementPlannerSelectsDirectTargetPathForMovementSteps) {
  llvm::raw_null_ostream os;
  auto memoryModel =
      mlir::ascend::TargetMemoryModelBuilder().build(makeCompleteTargetProfile(),
                                                     os);
  ASSERT_TRUE(llvm::succeeded(memoryModel));

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
  staticMemory.workspaceSlots = {StaticMemoryWorkspaceSlot{
      /*slotId=*/0, /*valueId=*/1, /*offset=*/0, MemoryPlace::VECIN,
      /*staticByteSizeKnown=*/true, /*byteSize=*/128}};

  MovementPlanner planner;
  auto plan = planner.build(placement, staticMemory, *memoryModel);

  ASSERT_TRUE(llvm::succeeded(plan));
  EXPECT_EQ(plan->movementDemandCount, 1u);
  EXPECT_EQ(plan->selectedPathCount, 1u);
  EXPECT_EQ(plan->pathSelectionDeferredCount, 0u);
  ASSERT_EQ(plan->movementSteps.size(), 1u);
  EXPECT_EQ(plan->movementSteps[0].srcPlace, MemoryPlace::GM);
  EXPECT_EQ(plan->movementSteps[0].dstPlace, MemoryPlace::VECIN);
  EXPECT_TRUE(plan->movementSteps[0].pathSelected);
  EXPECT_FALSE(plan->movementSteps[0].pathSelectionDeferred);
  EXPECT_EQ(plan->movementSteps[0].pathVariant, 0u);
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

TEST(AscendRealizePlannerTest,
     MemoryRealizationMaterializesSelectedMovementSteps) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseRealizeModule(
      context, R"mlir(
module {
  func.func @f(%arg0: memref<4x8xf32>) attributes {ascend.normalized = true} {
    %out = memref.alloc() : memref<4x8xf32>
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : memref<4x8xf32>)
      outs(%out : memref<4x8xf32>)
      attrs = {ascend.kernel = "kernel_0",
               ascend.op_role = "vector",
               ascend.schedule.decision_id = "kernel_0.decision.0",
               ascend.schedule.structured_lowering = "loop_skeleton_v0"} {
    ^bb0(%input: f32, %old: f32):
      %0 = arith.negf %input : f32
      linalg.yield %0 : f32
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  SmallVector<RealizePlanBundle, 1> bundles;
  bundles.push_back(makePlanBundle());
  bundles[0].movement.mode = "movement_planning";
  bundles[0].movement.movementDemandCount = 1;
  bundles[0].movement.selectedPathCount = 1;
  bundles[0].movement.pathSelectionDeferredCount = 0;
  bundles[0].movement.materializationDeferred = true;

  MovementStep step;
  step.stepId = 0;
  step.valueId = 0;
  step.slotId = 0;
  step.srcPlace = MemoryPlace::GM;
  step.dstPlace = MemoryPlace::VECIN;
  step.pathSelected = true;
  step.pathSelectionDeferred = false;
  step.staticByteSizeKnown = true;
  step.byteSize = 128;
  bundles[0].movement.movementSteps.push_back(step);

  MemoryRealizationDriver driver;
  ASSERT_TRUE(succeeded(driver.materialize(
      *module, bundles, MemoryRealizationMode::MemorySpaceAnnotate)));

  EXPECT_EQ(bundles[0].realization.mode, "memory_space_materialize");
  EXPECT_EQ(bundles[0].realization.materializedAllocCount, 1u);
  EXPECT_EQ(bundles[0].realization.materializedCopyCount, 1u);

  unsigned vecInAllocCount = 0;
  unsigned copyCount = 0;
  unsigned vecInInputCount = 0;
  module->walk([&](memref::AllocOp allocOp) {
    auto type = cast<MemRefType>(allocOp.getType());
    auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
    if (space && space.getInt() ==
                     static_cast<int64_t>(mlir::ascend::MemoryPlace::VECIN))
      ++vecInAllocCount;
  });
  module->walk([&](memref::CopyOp) { ++copyCount; });
  module->walk([&](linalg::LinalgOp linalgOp) {
    for (OpOperand *input : linalgOp.getDpsInputOperands()) {
      auto type = dyn_cast<MemRefType>(input->get().getType());
      auto space = type ? dyn_cast_or_null<IntegerAttr>(type.getMemorySpace())
                        : IntegerAttr();
      if (space && space.getInt() ==
                       static_cast<int64_t>(mlir::ascend::MemoryPlace::VECIN))
        ++vecInInputCount;
    }
  });
  EXPECT_EQ(vecInAllocCount, 1u);
  EXPECT_EQ(copyCount, 1u);
  EXPECT_EQ(vecInInputCount, 1u);
}

TEST(AscendRealizePlannerTest,
     Phase5BridgeFailureDoesNotLeavePartialVecOutAlloc) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseRealizeModule(
      context, R"mlir(
module {
  func.func @f(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>,
               %concat: memref<?x?xf32>, %m: index, %n: index,
               %which_dim: index) -> memref<?x?xf32>
      attributes {ascend.normalized = true} {
    %out = memref.alloc(%m, %n) {alignment = 64 : i64} : memref<?x?xf32>
    %dim = memref.dim %out, %which_dim : memref<?x?xf32>
    %sub = memref.subview %concat[0, 0][%m, %n][1, 1]
      : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%out : memref<?x?xf32>)
      attrs = {ascend.kernel = "kernel_0",
               ascend.op_role = "vector",
               ascend.schedule.decision_id = "kernel_0.decision.0",
               ascend.schedule.structured_lowering = "loop_skeleton_v0"} {
    ^bb0(%lhs: f32, %rhs: f32, %old: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    }
    memref.copy %out, %sub
      : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
    return %concat : memref<?x?xf32>
  }
}
)mlir");
  ASSERT_TRUE(module);

  MemoryRealizationDriver driver;
  EXPECT_TRUE(failed(driver.materializePhase5Bridge(*module)));

  unsigned vecOutAllocCount = 0;
  module->walk([&](memref::AllocOp allocOp) {
    auto type = cast<MemRefType>(allocOp.getType());
    auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
    if (space && space.getInt() ==
                     static_cast<int64_t>(mlir::ascend::MemoryPlace::VECOUT))
      ++vecOutAllocCount;
  });
  EXPECT_EQ(vecOutAllocCount, 0u);
}

TEST(AscendRealizePlannerTest,
     Phase5CubeBridgeDominatesNestedVectorUse) {
  MLIRContext context;
  OwningOpRef<ModuleOp> module = parseRealizeModule(
      context, R"mlir(
module {
  func.func @f(%lhs: memref<4x4xf16>, %rhs: memref<4x4xf16>,
               %bias: memref<4x4xf32>, %cond: i1)
      attributes {ascend.normalized = true} {
    %mat = memref.alloc() : memref<4x4xf32>
    linalg.matmul {
      ascend.kernel = "kernel_0",
      ascend.op_role = "cube",
      ascend.schedule.decision_id = "kernel_0.decision.0",
      ascend.schedule.structured_lowering = "loop_skeleton_v0"
    }
      ins(%lhs, %rhs : memref<4x4xf16>, memref<4x4xf16>)
      outs(%mat : memref<4x4xf32>)
    scf.if %cond {
      %vec = memref.alloc() : memref<4x4xf32>
      linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%mat, %bias : memref<4x4xf32>, memref<4x4xf32>)
        outs(%vec : memref<4x4xf32>)
        attrs = {ascend.kernel = "kernel_0",
                 ascend.op_role = "vector",
                 ascend.schedule.decision_id = "kernel_0.decision.0",
                 ascend.schedule.structured_lowering = "loop_skeleton_v0"} {
      ^bb0(%x: f32, %bias_elem: f32, %old: f32):
        %sum = arith.addf %x, %bias_elem : f32
        linalg.yield %sum : f32
      }
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  MemoryRealizationDriver driver;
  FailureOr<llvm::StringMap<Phase5BridgeMaterializationCounts>> counts =
      driver.materializePhase5Bridge(*module);
  ASSERT_TRUE(succeeded(counts));
  ASSERT_EQ(counts->lookup("kernel_0").materializedAllocCount, 6u);
  ASSERT_EQ(counts->lookup("kernel_0").materializedCopyCount, 5u);

  unsigned vecInAllocCount = 0;
  module->walk([&](memref::AllocOp allocOp) {
    auto type = cast<MemRefType>(allocOp.getType());
    auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
    if (space && space.getInt() ==
                     static_cast<int64_t>(mlir::ascend::MemoryPlace::VECIN))
      ++vecInAllocCount;
  });
  EXPECT_EQ(vecInAllocCount, 1u);
}
