#pragma once

#include "Runtime/Mix/MixArtifact.h"
#include "Runtime/Execution/TaskGraph.h"
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

KernelArtifact normalizeMixArtifact(const MixArtifact &artifact,
                                    KernelKind kind,
                                    MixResourceType mixResourceType);

} // namespace mlir::runtime
