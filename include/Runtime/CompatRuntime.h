#pragma once

#include "Runtime/ArtifactCompiler.h"
#include "Runtime/RunManifest.h"

#include <optional>

namespace mlir::runtime {

struct CompatCompileOptions {
  std::string kernelSource;
  std::string kernelName;
  KernelKind kernelKind = KernelKind::Vec;
  std::string socVersion;
  std::string outputDir;
  std::optional<std::string> cannMlirPath;
  std::optional<std::string> npyDir;
};

ArtifactCompileRequest
buildCompatCompileRequest(const CompatCompileOptions &options);

struct CompatSingleTaskManifestOptions {
  std::string artifactRoot;
  std::string taskId = "main";
  ExecutionBackendKind backendKind = ExecutionBackendKind::Simulation;
  std::vector<std::string> dependencies;
  ExecutionInvocation invocation;
};

RunManifestSpec
buildCompatSingleTaskRunManifest(const CompatSingleTaskManifestOptions &options);

} // namespace mlir::runtime
