#pragma once

#include "Runtime/Compiler.h"
#include "Runtime/MixArtifact.h"
#include "Runtime/MixDirectBackend.h"
#include "Runtime/TaskGraph.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace mlir::runtime {

struct ArtifactCompileRequest {
  std::string kernelSource;
  std::string kernelName;
  KernelKind kernelKind = KernelKind::Vec;
  std::string socVersion;
  std::string outputDir;
  std::optional<std::string> cannMlirPath;
  std::optional<std::string> npyDir;
};

MixResourceType inferMixResourceTypeFromKernelKind(KernelKind kind);

class ArtifactCompiler {
public:
  llvm::Expected<KernelArtifact> compile(const ArtifactCompileRequest &req) const;
};

} // namespace mlir::runtime
