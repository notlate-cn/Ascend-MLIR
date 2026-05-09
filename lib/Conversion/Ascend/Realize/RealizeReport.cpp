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
    os << "PlacementPlan:\n";
    os << "  kernel = " << bundle.placement.kernelId << "\n";
    os << "  selected_places = " << bundle.placement.selectedPlaceCount
       << "\n";
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
