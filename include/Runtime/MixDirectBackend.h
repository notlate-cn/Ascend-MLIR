#pragma once

#include "Runtime/MixArtifact.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir::runtime {

struct MixDirectCompileConfig {
  std::string kernelSrc;
  std::string kernelName;
  std::string socVersion;
  std::string outputDir;
};

class MixDirectBackend {
public:
  llvm::Expected<MixArtifact> compile(const MixDirectCompileConfig& cfg);
};

} // namespace mlir::runtime
