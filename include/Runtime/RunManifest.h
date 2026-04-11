#pragma once

#include "Runtime/TaskGraph.h"
#include "llvm/Support/Error.h"

#include <string>

namespace mlir::runtime {

struct RunManifestSpec {
  std::string taskId;
  std::string artifactRoot;
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  ExecutionInvocation invocation;
};

llvm::Expected<RunManifestSpec> loadRunManifest(const std::string &path);

} // namespace mlir::runtime
