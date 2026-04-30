#pragma once

#include "Runtime/Artifact/ArtifactCompiler.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::runtime {

class VecCubeArtifactBackend {
public:
  llvm::Expected<KernelArtifact>
  compile(const ArtifactCompileRequest &req, llvm::StringRef resolvedSoc) const;
};

} // namespace mlir::runtime
