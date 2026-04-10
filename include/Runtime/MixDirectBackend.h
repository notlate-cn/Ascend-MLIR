#pragma once

#include "Runtime/MixArtifact.h"
#include "llvm/Support/Error.h"
#include <optional>
#include <string>

namespace mlir::runtime {

struct MixDirectCompileConfig {
  std::string kernelSrc;
  std::string kernelName;
  std::string socVersion;
  std::string outputDir;
  std::optional<std::string> cannMlirPath;
  std::optional<std::string> npyDir;
};

class MixDirectBackend {
public:
  llvm::Expected<MixArtifact> compile(const MixDirectCompileConfig& cfg);
};

} // namespace mlir::runtime
