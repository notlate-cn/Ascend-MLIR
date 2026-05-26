#pragma once

#include "Runtime/Execution/TaskGraph.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

struct ArtifactManifestBindingPath {
  std::string taskId;
  std::string bindingName;
  std::string path;
};

struct ArtifactManifestPrepareRequest {
  std::string artifactManifestPath;
  std::string artifactRoot;
  std::string outputRunManifestPath;
  std::vector<std::pair<std::string, int64_t>> shapeArgs;
  std::vector<ArtifactManifestBindingPath> inputPaths;
  std::vector<ArtifactManifestBindingPath> outputPaths;
  std::vector<ArtifactManifestBindingPath> expectedOutputPaths;
  std::optional<bool> enableProfiling;
  std::optional<double> atol;
  std::optional<double> rtol;
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
};

llvm::Expected<RunManifestSpec> loadRunManifest(const std::string &path);

llvm::Error
emitRunManifestFromArtifactManifest(const ArtifactManifestPrepareRequest &request);

} // namespace mlir::runtime
