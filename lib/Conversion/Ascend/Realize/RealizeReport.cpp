//===- RealizeReport.cpp - Ascend realize reports ---------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "RealizeReport.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::ascend::realize {
namespace {

void printByteSize(llvm::raw_ostream &os, bool known, uint64_t byteSize) {
  if (known)
    os << " byte_size=" << byteSize;
  else
    os << " byte_size=unknown";
}

void printStaticMemoryDetails(const StaticMemoryPlan &plan,
                              llvm::raw_ostream &os) {
  for (auto [index, interval] : llvm::enumerate(plan.liveIntervals)) {
    os << "  live_interval[" << index << "] = value_id="
       << interval.valueId << " start=" << interval.start
       << " end=" << interval.end << " place="
       << ::mlir::ascend::stringifyMemoryPlace(interval.place);
    printByteSize(os, interval.staticByteSizeKnown, interval.byteSize);
    os << "\n";
  }
  for (auto [index, slot] : llvm::enumerate(plan.workspaceSlots)) {
    os << "  workspace_slot[" << index << "] = slot_id=" << slot.slotId
       << " value_id=" << slot.valueId << " offset=" << slot.offset
       << " place=" << ::mlir::ascend::stringifyMemoryPlace(slot.place);
    printByteSize(os, slot.staticByteSizeKnown, slot.byteSize);
    if (slot.byteSizeExprKnown)
      os << " byte_size_expr=\"" << slot.byteSizeExpr << "\"";
    os << "\n";
  }
}

void printMovementDetails(const MovementPlan &plan, llvm::raw_ostream &os) {
  for (auto [index, step] : llvm::enumerate(plan.movementSteps)) {
    os << "  movement_step[" << index << "] = step_id=" << step.stepId
       << " value_id=" << step.valueId << " slot_id=" << step.slotId
       << " src=" << ::mlir::ascend::stringifyMemoryPlace(step.srcPlace)
       << " dst=" << ::mlir::ascend::stringifyMemoryPlace(step.dstPlace)
       << " path_selected=" << (step.pathSelected ? "true" : "false")
       << " path_variant=" << step.pathVariant
       << " path_selection_deferred="
       << (step.pathSelectionDeferred ? "true" : "false");
    printByteSize(os, step.staticByteSizeKnown, step.byteSize);
    os << "\n";
  }
}

} // namespace

void printRealizeReport(llvm::ArrayRef<RealizePlanBundle> bundles,
                        llvm::raw_ostream &os) {
  os << "Realize report\n";
  os << "  kernels = " << bundles.size() << "\n";
  for (const RealizePlanBundle &bundle : bundles) {
    os << "BufferizedKernelIR:\n";
    os << "  kernel = " << bundle.bufferizedIR.kernelId << "\n";
    os << "  mode = \"" << bundle.bufferizedIR.mode << "\"\n";
    os << "  buffer_values = " << bundle.bufferizedIR.bufferValueCount << "\n";
    os << "  input_values = " << bundle.bufferizedIR.inputValueCount << "\n";
    os << "  output_values = " << bundle.bufferizedIR.outputValueCount << "\n";
    os << "  temporary_values = " << bundle.bufferizedIR.temporaryValueCount
       << "\n";
    os << "  vector_temporary_values = "
       << bundle.bufferizedIR.vectorTemporaryValueCount << "\n";
    os << "PlacementPlan:\n";
    os << "  kernel = " << bundle.placement.kernelId << "\n";
    os << "  mode = \"" << bundle.placement.mode << "\"\n";
    os << "  selected_places = " << bundle.placement.selectedPlaceCount
       << "\n";
    os << "  gm_places = " << bundle.placement.gmPlaceCount << "\n";
    os << "  on_chip_places = " << bundle.placement.onChipPlaceCount << "\n";
    os << "  deferred_local_places = "
       << bundle.placement.deferredLocalPlaceCount << "\n";
    os << "StaticMemoryPlan:\n";
    os << "  kernel = " << bundle.staticMemory.kernelId << "\n";
    os << "  mode = \"" << bundle.staticMemory.mode << "\"\n";
    os << "  tracked_places = " << bundle.staticMemory.trackedPlaceCount
       << "\n";
    os << "  local_buffers = " << bundle.staticMemory.localBufferCount
       << "\n";
    os << "  live_intervals = " << bundle.staticMemory.liveIntervalCount
       << "\n";
    os << "  workspace_slots = " << bundle.staticMemory.workspaceSlotCount
       << "\n";
    os << "  peak_usage_known = "
       << (bundle.staticMemory.peakUsageKnown ? "true" : "false") << "\n";
    os << "  peak_usage_units = " << bundle.staticMemory.peakUsageUnitCount
       << "\n";
    os << "  peak_usage_bytes_known = "
       << (bundle.staticMemory.peakUsageBytesKnown ? "true" : "false")
       << "\n";
    os << "  local_buffer_bytes = "
       << bundle.staticMemory.localBufferByteCount << "\n";
    os << "  workspace_bytes = " << bundle.staticMemory.workspaceByteCount
       << "\n";
    os << "  peak_usage_bytes = " << bundle.staticMemory.peakUsageByteCount
       << "\n";
    os << "  capacity_check_deferred = "
       << (bundle.staticMemory.capacityCheckDeferred ? "true" : "false")
       << "\n";
    os << "  workspace_size_expr_known = "
       << (bundle.staticMemory.workspaceSizeExprKnown ? "true" : "false")
       << "\n";
    if (bundle.staticMemory.workspaceSizeExprKnown)
      os << "  workspace_size_expr = \"" << bundle.staticMemory.workspaceSizeExpr
         << "\"\n";
    printStaticMemoryDetails(bundle.staticMemory, os);
    os << "MovementPlan:\n";
    os << "  kernel = " << bundle.movement.kernelId << "\n";
    os << "  mode = \"" << bundle.movement.mode << "\"\n";
    os << "  cross_place_edges = " << bundle.movement.crossPlaceEdgeCount
       << "\n";
    os << "  movements = " << bundle.movement.movementCount << "\n";
    os << "  redundant_movements = "
       << bundle.movement.redundantMovementCount << "\n";
    os << "  movement_demands = " << bundle.movement.movementDemandCount
       << "\n";
    os << "  selected_paths = " << bundle.movement.selectedPathCount << "\n";
    os << "  path_selection_deferred = "
       << bundle.movement.pathSelectionDeferredCount << "\n";
    os << "  workspace_reuse_candidates = "
       << bundle.movement.workspaceReuseCandidateCount << "\n";
    os << "  dynamic_view_chain_rewrites = "
       << bundle.movement.dynamicViewChainRewriteCount << "\n";
    os << "  deferred_view_chain_rewrites = "
       << bundle.movement.deferredViewChainRewriteCount << "\n";
    os << "  materialization_deferred = "
       << (bundle.movement.materializationDeferred ? "true" : "false")
       << "\n";
    printMovementDetails(bundle.movement, os);
    os << "MemoryRealizationPlan:\n";
    os << "  kernel = " << bundle.realization.kernelId << "\n";
    os << "  mode = \"" << bundle.realization.mode << "\"\n";
    os << "  frozen = " << (bundle.realization.frozen ? "true" : "false")
       << "\n";
    os << "  verification_scope = \""
       << bundle.realization.verificationScope << "\"\n";
    os << "  plan_ids_verified = "
       << (bundle.realization.planIdsVerified ? "true" : "false") << "\n";
    os << "  memory_space_annotations = "
       << bundle.realization.memorySpaceAnnotationCount << "\n";
    os << "  materialized_allocs = "
       << bundle.realization.materializedAllocCount << "\n";
    os << "  materialized_copies = "
       << bundle.realization.materializedCopyCount << "\n";
  }
}

} // namespace mlir::ascend::realize
