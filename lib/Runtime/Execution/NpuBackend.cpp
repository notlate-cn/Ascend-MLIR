// lib/Runtime/NpuBackend.cpp
#include "Runtime/NpuBackend.h"

#include "Runtime/Execution/DefaultExecutionRunner.h"
#include "Runtime/NpyIO.h"
#include "Runtime/OutputComparator.h"
#include "Runtime/TilingPack.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <optional>
#include <utility>

namespace mlir::runtime {

namespace {

constexpr uint32_t kMagicElfAiVec = 0x41415246u;
constexpr uint32_t kMagicElfAiCube = 0x41494343u;

llvm::Error stageError(llvm::StringRef stage, llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "[npu:%s] %s", stage.str().c_str(),
                                 message.str().c_str());
}

llvm::Error stageError(llvm::StringRef stage, llvm::Error error) {
  return stageError(stage, llvm::toString(std::move(error)));
}

llvm::Expected<std::vector<NDArray>>
loadExpectedOutputs(const ExecutionInvocation &invocation) {
  std::vector<NDArray> expected;
  for (const TensorBinding &binding : invocation.expectedOutputs) {
    if (binding.sourceKind != BindingSourceKind::ExternalFile)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "expected output binding must be an external file: %s",
          binding.name.c_str());
    auto arrayOr = LoadNpy(binding.path);
    if (!arrayOr)
      return arrayOr.takeError();
    expected.push_back(std::move(*arrayOr));
  }
  return expected;
}

llvm::Expected<RunArgs> buildRunArgs(const ExecutionInvocation &invocation) {
  RunArgs args;
  args.block_dim = invocation.blockDim;
  args.workspace_size = invocation.workspaceSize;

  if (invocation.outputs.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "npu path requires at least one output binding");
  }

  auto tilingOr = packTilingBytes(invocation.tiling);
  if (!tilingOr)
    return tilingOr.takeError();
  args.tiling = std::move(*tilingOr);

  for (const TensorBinding &binding : invocation.inputs) {
    if (binding.sourceKind != BindingSourceKind::ExternalFile)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "input binding must be an external file: %s", binding.name.c_str());
    if (binding.path.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "input binding is missing path: %s",
                                     binding.name.c_str());
    auto arrayOr = LoadNpy(binding.path);
    if (!arrayOr)
      return arrayOr.takeError();
    args.inputs.push_back(std::move(*arrayOr));
  }

  auto expectedOutputsOr = loadExpectedOutputs(invocation);
  if (!expectedOutputsOr)
    return expectedOutputsOr.takeError();
  if (!invocation.outputs.empty() && !expectedOutputsOr->empty() &&
      invocation.outputs.size() != expectedOutputsOr->size()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "output binding count does not match expected output count");
  }

  if (!expectedOutputsOr->empty()) {
    for (const NDArray &expected : *expectedOutputsOr) {
      NDArray output;
      output.shape = expected.shape;
      output.dtype = expected.dtype;
      output.allocate();
      args.outputs.push_back(std::move(output));
    }
    return args;
  }

  for (const TensorBinding &binding : invocation.outputs) {
    if (binding.sourceKind != BindingSourceKind::ExternalFile)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "output binding must be an external file: %s", binding.name.c_str());
    if (binding.path.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "output binding is missing path: %s",
                                     binding.name.c_str());
    if (!binding.shape)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "output binding is missing shape metadata: %s",
          binding.name.c_str());
    if (!binding.dtype)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "output binding is missing dtype metadata: %s",
          binding.name.c_str());

    NDArray output;
    output.shape = *binding.shape;
    output.dtype = *binding.dtype;
    output.allocate();
    args.outputs.push_back(std::move(output));
  }

  return args;
}

llvm::Error writeActualOutputs(const ExecutionInvocation &invocation,
                               const RunArgs &args) {
  for (size_t index = 0; index < invocation.outputs.size(); ++index) {
    if (auto err = SaveNpy(invocation.outputs[index].path, args.outputs[index]))
      return err;
  }
  return llvm::Error::success();
}

uint32_t magicForKernelKind(KernelKind kind) {
  switch (kind) {
  case KernelKind::Vec:
  case KernelKind::Mix:
    return kMagicElfAiVec;
  case KernelKind::Cube:
    return kMagicElfAiCube;
  }
  return kMagicElfAiVec;
}

llvm::Expected<ExecutionResult> runWithExecutor(const ExecutionRequest &request) {
  if (request.task.artifact.kernelKind == KernelKind::Mix) {
    if (request.task.artifact.sharedLibraryPath.empty()) {
      return stageError("artifact",
                        "mix artifact is missing shared library path");
    }
  } else if (request.task.artifact.deviceBinaryPath.empty()) {
    return stageError("artifact", "artifact is missing device binary path");
  }

  auto argsOr = buildRunArgs(request.task.invocation);
  if (!argsOr)
    return stageError("bindings", argsOr.takeError());
  RunArgs args = std::move(*argsOr);

  auto expectedOutputsOr = loadExpectedOutputs(request.task.invocation);
  if (!expectedOutputsOr)
    return stageError("bindings", expectedOutputsOr.takeError());

  auto runnerOr = createDefaultExecutionRunner(ExecutionRunnerMode::RealDevice);
  if (!runnerOr)
    return stageError("executor_initialize", runnerOr.takeError());
  std::unique_ptr<ExecutionRunner> runner = std::move(*runnerOr);
  if (auto err = runner->initialize())
    return stageError("executor_initialize", std::move(err));

  if (request.task.artifact.kernelKind == KernelKind::Mix) {
    DynamicLibraryExecutionLaunch launch;
    launch.sharedLibraryPath = request.task.artifact.sharedLibraryPath;
    launch.symbolName = "aclrtlaunch_" + request.task.artifact.kernelName;
    if (auto err = runner->runDynamicLibraryArtifact(launch, args)) {
      return stageError("kernel_launch", std::move(err));
    }
  } else {
    FileExecutionLaunch launch;
    launch.binaryPath = request.task.artifact.deviceBinaryPath;
    launch.kernelName = request.task.artifact.kernelName;
    launch.magic = magicForKernelKind(request.task.artifact.kernelKind);
    if (auto err = runner->runFile(launch, args)) {
      return stageError("kernel_launch", std::move(err));
    }
  }

  if (!expectedOutputsOr->empty()) {
    auto validationOr =
        compareRuntimeOutputs(args.outputs, *expectedOutputsOr,
                              request.task.invocation.atol,
                              request.task.invocation.rtol);
    if (!validationOr) {
      return stageError("validate", validationOr.takeError());
    }
    const OutputComparisonResult &validation = *validationOr;
    if (!validation.passed) {
      return stageError(
          "validate",
          llvm::formatv("npu output mismatch: max_abs_diff={0:F} mean_abs_diff={1:F}",
                        validation.maxAbsDiff, validation.meanAbsDiff)
              .str());
    }
  }

  if (auto err = writeActualOutputs(request.task.invocation, args))
    return stageError("write_outputs", std::move(err));

  ExecutionResult result;
  result.taskId = request.task.taskId;
  for (const TensorBinding &binding : request.task.invocation.outputs)
    result.producedFiles.push_back(binding.path);
  return result;
}

} // namespace

NpuBackend::NpuBackend(std::shared_ptr<ExecutionBackendDriver> driver)
    : driver_(std::move(driver)) {}

ExecutionBackendKind NpuBackend::kind() const {
  return ExecutionBackendKind::Npu;
}

llvm::Expected<ExecutionResult>
NpuBackend::run(const ExecutionRequest &request) {
  if (driver_)
    return driver_->run(request);
  return runWithExecutor(request);
}

} // namespace mlir::runtime
