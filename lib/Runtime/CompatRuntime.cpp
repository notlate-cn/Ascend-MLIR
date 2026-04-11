#include "Runtime/CompatRuntime.h"

namespace mlir::runtime {

ArtifactCompileRequest
buildCompatCompileRequest(const CompatCompileOptions &options) {
  ArtifactCompileRequest request;
  request.kernelSource = options.kernelSource;
  request.kernelName = options.kernelName;
  request.kernelKind = options.kernelKind;
  request.socVersion = options.socVersion;
  request.outputDir = options.outputDir;
  request.cannMlirPath = options.cannMlirPath;
  request.npyDir = options.npyDir;
  return request;
}

RunManifestSpec
buildCompatSingleTaskRunManifest(const CompatSingleTaskManifestOptions &options) {
  RunManifestSpec manifest;
  manifest.backendKind = options.backendKind;

  RunTaskSpec task;
  task.taskId = options.taskId;
  task.artifactRoot = options.artifactRoot;
  task.dependencies = options.dependencies;
  task.invocation = options.invocation;
  manifest.tasks.push_back(std::move(task));

  return manifest;
}

} // namespace mlir::runtime
