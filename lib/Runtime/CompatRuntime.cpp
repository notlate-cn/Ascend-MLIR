#include "Runtime/CompatRuntime.h"

namespace mlir::runtime {

ArtifactCompileRequest
buildCompatCompileRequest(const CompatCompileOptions &options) {
  ArtifactCompileRequest request;
  request.kernelSource = options.kernelSourcePath;
  request.kernelName = options.requestedKernelName;
  if (options.kernelType == "cube")
    request.kernelKind = KernelKind::Cube;
  else if (options.kernelType == "mix")
    request.kernelKind = KernelKind::Mix;
  else
    request.kernelKind = KernelKind::Vec;
  request.socVersion = options.socVersion;
  request.outputDir = options.outputRoot;
  return request;
}

RunManifestSpec
buildCompatSingleTaskRunManifest(const CompatValidateOptions &options) {
  RunManifestSpec manifest;
  manifest.backendKind = ExecutionBackendKind::Simulation;

  RunTaskSpec task;
  task.taskId = "main";
  task.artifactRoot = options.artifactRoot;

  TensorBinding input;
  input.sourceKind = BindingSourceKind::ExternalFile;
  for (size_t index = 0; index < options.inputPaths.size(); ++index) {
    TensorBinding binding = input;
    binding.name = "data" + std::to_string(index);
    binding.path = options.inputPaths[index];
    task.invocation.inputs.push_back(std::move(binding));
  }

  if (!options.actualOutputPath.empty()) {
    TensorBinding output;
    output.name = "out";
    output.sourceKind = BindingSourceKind::ExternalFile;
    output.path = options.actualOutputPath;
    task.invocation.outputs.push_back(std::move(output));
  }

  if (!options.expectedOutputPath.empty()) {
    TensorBinding expectedOutput;
    expectedOutput.name = "out";
    expectedOutput.sourceKind = BindingSourceKind::ExternalFile;
    expectedOutput.path = options.expectedOutputPath;
    task.invocation.expectedOutputs.push_back(std::move(expectedOutput));
  }

  if (!options.tilingBinaryPath.empty()) {
    TilingBinding tiling;
    tiling.binaryPath = options.tilingBinaryPath;
    if (!options.tilingSchemaPath.empty())
      tiling.schemaPath = options.tilingSchemaPath;
    if (!options.tilingParams.empty())
      tiling.params = options.tilingParams;
    task.invocation.tiling = std::move(tiling);
  } else if (!options.tilingSchemaPath.empty() || !options.tilingParams.empty()) {
    TilingBinding tiling;
    tiling.schemaPath = options.tilingSchemaPath;
    tiling.params = options.tilingParams;
    task.invocation.tiling = std::move(tiling);
  }

  task.invocation.blockDim = options.blockDim;
  task.invocation.atol = options.atol;
  task.invocation.rtol = options.rtol;
  manifest.tasks.push_back(std::move(task));

  return manifest;
}

} // namespace mlir::runtime
