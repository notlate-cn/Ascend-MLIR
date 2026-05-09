//===- RealizeReport.cpp - Ascend realize reports ---------------------===//
//
// Part of the Ascend-MLIR Project
//
//===----------------------------------------------------------------------===//

#include "Conversion/Ascend/Realize/RealizeReport.h"

namespace mlir::afir::ascend::realize {

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
    os << "  workspace_slots = " << bundle.staticMemory.workspaceSlotCount
       << "\n";
    os << "MovementPlan:\n";
    os << "  kernel = " << bundle.movement.kernelId << "\n";
    os << "  movements = " << bundle.movement.movementCount << "\n";
    os << "MemoryRealizationPlan:\n";
    os << "  kernel = " << bundle.realization.kernelId << "\n";
    os << "  frozen = " << (bundle.realization.frozen ? "true" : "false")
       << "\n";
  }
}

} // namespace mlir::afir::ascend::realize
