#include "Runtime/CompatRuntime.h"
#include "Runtime/NpyIO.h"

#include <utility>

namespace mlir::runtime {

llvm::Expected<ArtifactCompileRequest>
buildCompatCompileRequest(const CompatCompileOptions &options) {
  if (options.kernelType != "vec" && options.kernelType != "cube" &&
      options.kernelType != "mix") {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported kernel type: %s",
                                   options.kernelType.c_str());
  }

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
  request.arch = options.arch;
  request.verbose = options.verbose;
  return request;
}

llvm::Expected<RunManifestSpec>
buildCompatSingleTaskRunManifest(const CompatValidateOptions &options) {
  const bool hasExpectedOutput = !options.expectedOutputPath.empty();
  const bool hasActualOutput = !options.actualOutputPath.empty();
  const bool hasExplicitOutputMetadata =
      options.actualOutputShape.has_value() && options.actualOutputDType.has_value();

  if (hasExpectedOutput && !hasActualOutput) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "expected output path requires an actual output path");
  }
  if (!hasExpectedOutput && hasActualOutput && !hasExplicitOutputMetadata) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "actual output path requires explicit output shape and dtype metadata");
  }

  RunManifestSpec manifest;
  manifest.backendKind = ExecutionBackendKind::Simulation;

  RunTaskSpec task;
  task.taskId = "main";
  task.artifactRoot = options.artifactRoot;

  for (size_t index = 0; index < options.inputPaths.size(); ++index) {
    TensorBinding binding;
    binding.name = "data" + std::to_string(index);
    binding.sourceKind = BindingSourceKind::ExternalFile;
    binding.path = options.inputPaths[index];
    task.invocation.inputs.push_back(std::move(binding));
  }

  if (hasExpectedOutput) {
    auto expectedArrOr = LoadNpy(options.expectedOutputPath);
    if (!expectedArrOr)
      return expectedArrOr.takeError();

    TensorBinding output;
    output.name = "out";
    output.sourceKind = BindingSourceKind::ExternalFile;
    output.path = options.actualOutputPath;
    output.shape = expectedArrOr->shape;
    output.dtype = expectedArrOr->dtype;
    task.invocation.outputs.push_back(std::move(output));

    TensorBinding expectedOutput;
    expectedOutput.name = "out";
    expectedOutput.sourceKind = BindingSourceKind::ExternalFile;
    expectedOutput.path = options.expectedOutputPath;
    task.invocation.expectedOutputs.push_back(std::move(expectedOutput));
  } else if (hasActualOutput) {
    TensorBinding output;
    output.name = "out";
    output.sourceKind = BindingSourceKind::ExternalFile;
    output.path = options.actualOutputPath;
    output.shape = options.actualOutputShape;
    output.dtype = options.actualOutputDType;
    task.invocation.outputs.push_back(std::move(output));
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
