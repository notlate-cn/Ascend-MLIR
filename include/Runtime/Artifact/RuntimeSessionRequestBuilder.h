#pragma once

#include "Runtime/Artifact/ArtifactCompiler.h"
#include "Runtime/Execution/TaskGraph.h"
#include "Runtime/Support/Types.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>

namespace mlir::runtime {

struct RuntimeSessionArtifactRequest {
  std::string artifactRoot;
  std::string kernelSource;
  std::string kernelName;
  std::string outputDir;
  std::string socVersion;
  std::optional<std::string> cannMlirPath;
  std::optional<std::string> npyDir;
  KernelKind kernelKind = KernelKind::Mix;
};

llvm::Expected<KernelArtifact>
loadRuntimeSessionArtifactFromRoot(llvm::StringRef artifactRoot);

llvm::Expected<KernelArtifact>
prepareRuntimeSessionArtifact(const RuntimeSessionArtifactRequest &request);

llvm::Expected<TaskGraph>
buildRuntimeSessionSingleTaskGraph(const KernelArtifact &artifact,
                                   llvm::StringRef taskId);

llvm::Expected<std::pair<ExecutionBackendKind, TaskGraph>>
prepareRuntimeSessionGraphFromManifest(llvm::StringRef runManifestPath);

} // namespace mlir::runtime
