#pragma once

#include "Runtime/Artifact/RunManifest.h"

#include "llvm/Support/Error.h"

#include <string>

namespace mlir::runtime {

struct DebugCasePrepareRequest {
  std::string casePath;
  std::string outputRunManifestPath;
  std::string defaultOutputDirectory;
};

llvm::Error emitRunManifestFromDebugCase(const DebugCasePrepareRequest &request);

} // namespace mlir::runtime
