#pragma once

#include "Runtime/TaskGraph.h"
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

llvm::Expected<RunManifestSpec> loadRunManifest(const std::string &path);

} // namespace mlir::runtime
