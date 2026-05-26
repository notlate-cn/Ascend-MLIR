#pragma once

#include "Runtime/Execution/TaskGraph.h"
#include "llvm/Support/Error.h"

#include <string>

namespace mlir::runtime {

struct RunTaskSpec {
  std::string taskId;
  std::string artifactRoot;
  std::vector<std::string> dependencies;
  ExecutionInvocation invocation;
};

struct RunManifestSpec {
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  std::vector<RunTaskSpec> tasks;
};

struct ArtifactManifestPrepareRequest {
  std::string artifactManifestPath;
  std::string artifactRoot;
  std::string outputRunManifestPath;
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
};

llvm::Expected<RunManifestSpec> loadRunManifest(const std::string &path);

llvm::Error
emitRunManifestFromArtifactManifest(const ArtifactManifestPrepareRequest &request);

} // namespace mlir::runtime
