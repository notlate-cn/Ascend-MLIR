#pragma once

#include "Runtime/Execution/TaskGraph.h"

#include "llvm/Support/Error.h"

namespace mlir::runtime {

llvm::Error
configureDynamicLibraryArtifactSimulationEnv(const KernelArtifact &artifact);

} // namespace mlir::runtime
