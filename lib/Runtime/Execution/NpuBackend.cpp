// lib/Runtime/NpuBackend.cpp
#include "Runtime/NpuBackend.h"

#include "Runtime/Execution/DefaultExecutionRunner.h"
#include "Runtime/NpyIO.h"
#include "Runtime/OutputComparator.h"
#include "Runtime/TilingPack.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include <cstdlib>
#include <limits>
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

llvm::Error validateExternalFileBinding(llvm::StringRef role,
                                        const TensorBinding &binding) {
  if (binding.sourceKind != BindingSourceKind::ExternalFile) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "%s binding must be an external file: %s",
                                   role.str().c_str(), binding.name.c_str());
  }
  if (binding.path.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "%s binding is missing path: %s",
                                   role.str().c_str(), binding.name.c_str());
  }
  return llvm::Error::success();
}

llvm::Error validateOutputBindingPathAndSource(const TensorBinding &binding) {
  if (binding.sourceKind != BindingSourceKind::ExternalFile &&
      binding.sourceKind != BindingSourceKind::InputAlias) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "output binding has unsupported source: %s", binding.name.c_str());
  }
  if (binding.sourceKind == BindingSourceKind::InputAlias &&
      binding.aliasedInputName.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "input-alias output binding is missing input: %s",
        binding.name.c_str());
  }
  if (binding.path.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output binding is missing path: %s",
                                   binding.name.c_str());
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<NDArray>>
loadExpectedOutputs(const ExecutionInvocation &invocation) {
  std::vector<NDArray> expected;
  for (const TensorBinding &binding : invocation.expectedOutputs) {
    if (auto err = validateExternalFileBinding("expected output", binding))
      return std::move(err);
    auto arrayOr = LoadNpy(binding.path);
    if (!arrayOr)
      return arrayOr.takeError();
    expected.push_back(std::move(*arrayOr));
  }
  return expected;
}

llvm::Error validateOutputBindingAgainstExpected(const TensorBinding &binding,
                                                 const NDArray &expected) {
  if (auto err = validateOutputBindingPathAndSource(binding))
    return err;
  if (binding.shape && *binding.shape != expected.shape) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "output binding metadata does not match expected output: %s shape",
        binding.name.c_str());
  }
  if (binding.dtype && *binding.dtype != expected.dtype) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "output binding metadata does not match expected output: %s dtype",
        binding.name.c_str());
  }
  return llvm::Error::success();
}

llvm::Error validateOutputBindingWithoutExpected(const TensorBinding &binding) {
  if (auto err = validateOutputBindingPathAndSource(binding))
    return err;
  if (!binding.shape) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "output binding is missing shape metadata: %s",
        binding.name.c_str());
  }
  if (!binding.dtype) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "output binding is missing dtype metadata: %s",
        binding.name.c_str());
  }
  return llvm::Error::success();
}

std::optional<size_t>
findExpectedOutputIndexForBinding(const ExecutionInvocation &invocation,
                                  llvm::StringRef outputName,
                                  size_t outputIndex,
                                  size_t expectedCount) {
  for (size_t index = 0; index < invocation.expectedOutputs.size(); ++index) {
    if (invocation.expectedOutputs[index].name == outputName)
      return index;
  }
  if (expectedCount == invocation.outputs.size() && outputIndex < expectedCount)
    return outputIndex;
  return std::nullopt;
}

std::optional<size_t>
findOutputIndexForExpectedBinding(const ExecutionInvocation &invocation,
                                  llvm::StringRef expectedName,
                                  size_t expectedIndex,
                                  size_t actualCount) {
  for (size_t index = 0; index < invocation.outputs.size(); ++index) {
    if (invocation.outputs[index].name == expectedName)
      return index;
  }
  if (invocation.expectedOutputs.size() == actualCount &&
      expectedIndex < actualCount)
    return expectedIndex;
  return std::nullopt;
}

std::optional<size_t>
findInputIndexByName(const ExecutionInvocation &invocation,
                     llvm::StringRef inputName) {
  for (size_t index = 0; index < invocation.inputs.size(); ++index)
    if (invocation.inputs[index].name == inputName)
      return index;
  return std::nullopt;
}

NDArray buildAllocatedOutputFromExpected(const NDArray &expected) {
  NDArray output;
  output.shape = expected.shape;
  output.dtype = expected.dtype;
  output.allocate();
  return output;
}

NDArray buildAllocatedOutputFromShapeAndType(std::vector<int64_t> shape,
                                             DType dtype) {
  NDArray output;
  output.shape = std::move(shape);
  output.dtype = dtype;
  output.allocate();
  return output;
}

NDArray buildAllocatedOutputFromBinding(const TensorBinding &binding) {
  NDArray output;
  output.shape = *binding.shape;
  output.dtype = *binding.dtype;
  output.allocate();
  return output;
}

llvm::Expected<std::vector<NDArray>>
validateInvocationBindings(const ExecutionInvocation &invocation) {
  if (invocation.outputs.empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "npu path requires at least one output binding");
  }

  for (const TensorBinding &binding : invocation.inputs) {
    if (auto err = validateExternalFileBinding("input", binding))
      return llvm::Expected<std::vector<NDArray>>(std::move(err));
  }

  auto expectedOutputsOr = loadExpectedOutputs(invocation);
  if (!expectedOutputsOr)
    return expectedOutputsOr.takeError();

  std::vector<bool> matchedExpected(expectedOutputsOr->size(), false);
  for (size_t outputIndex = 0; outputIndex < invocation.outputs.size();
       ++outputIndex) {
    const TensorBinding &binding = invocation.outputs[outputIndex];
    if (auto expectedIndex = findExpectedOutputIndexForBinding(
            invocation, binding.name, outputIndex, expectedOutputsOr->size())) {
      if (auto err = validateOutputBindingAgainstExpected(
              binding, (*expectedOutputsOr)[*expectedIndex])) {
        return llvm::Expected<std::vector<NDArray>>(std::move(err));
      }
      matchedExpected[*expectedIndex] = true;
      continue;
    }
    if (auto err = validateOutputBindingWithoutExpected(binding))
      return llvm::Expected<std::vector<NDArray>>(std::move(err));
  }

  for (size_t expectedIndex = 0; expectedIndex < matchedExpected.size();
       ++expectedIndex) {
    if (!matchedExpected[expectedIndex]) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "expected output binding has no matching output: %s",
          invocation.expectedOutputs[expectedIndex].name.c_str());
    }
  }

  return std::move(*expectedOutputsOr);
}

llvm::Expected<RunArgs>
buildRunArgs(const ExecutionInvocation &invocation,
             llvm::ArrayRef<NDArray> expectedOutputs) {
  RunArgs args;
  args.block_dim = invocation.blockDim;
  args.workspace_size = invocation.workspaceSize;

  auto tilingOr = packTilingBytes(invocation.tiling);
  if (!tilingOr)
    return tilingOr.takeError();
  args.tiling = std::move(*tilingOr);

  for (const TensorBinding &binding : invocation.inputs) {
    auto arrayOr = LoadNpy(binding.path);
    if (!arrayOr)
      return arrayOr.takeError();
    args.inputs.push_back(std::move(*arrayOr));
  }

  for (size_t outputIndex = 0; outputIndex < invocation.outputs.size();
       ++outputIndex) {
    const TensorBinding &binding = invocation.outputs[outputIndex];
    if (binding.sourceKind == BindingSourceKind::InputAlias) {
      std::optional<size_t> inputIndex =
          findInputIndexByName(invocation, binding.aliasedInputName);
      if (!inputIndex) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "input-alias output references unknown input: %s",
            binding.aliasedInputName.c_str());
      }
      const NDArray &input = args.inputs[*inputIndex];
      std::vector<int64_t> shape =
          binding.shape ? *binding.shape : input.shape;
      DType dtype = binding.dtype ? *binding.dtype : input.dtype;
      RunArgs::InPlaceOutput inPlace;
      inPlace.inputIndex = *inputIndex;
      inPlace.array =
          buildAllocatedOutputFromShapeAndType(std::move(shape), dtype);
      args.in_place_outputs.push_back(std::move(inPlace));
      continue;
    }
    if (binding.sourceKind != BindingSourceKind::ExternalFile) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "unsupported output binding source for npu: %s",
          binding.name.c_str());
    }
    if (auto expectedIndex = findExpectedOutputIndexForBinding(
            invocation, binding.name, outputIndex, expectedOutputs.size())) {
      args.outputs.push_back(
          buildAllocatedOutputFromExpected(expectedOutputs[*expectedIndex]));
      continue;
    }
    args.outputs.push_back(buildAllocatedOutputFromBinding(binding));
  }

  return args;
}

llvm::Expected<std::vector<NDArray>>
collectActualOutputsForInvocation(const ExecutionInvocation &invocation,
                                  RunArgs &args) {
  std::vector<NDArray> actualOutputs;
  actualOutputs.reserve(invocation.outputs.size());
  size_t outputIndex = 0;
  size_t inPlaceIndex = 0;

  for (const TensorBinding &binding : invocation.outputs) {
    NDArray view;
    if (binding.sourceKind == BindingSourceKind::InputAlias) {
      if (inPlaceIndex >= args.in_place_outputs.size()) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "input-alias output index is out of range: %s",
            binding.name.c_str());
      }
      NDArray &actual = args.in_place_outputs[inPlaceIndex++].array;
      view.shape = actual.shape;
      view.dtype = actual.dtype;
      view.setExternal(actual.data);
    } else {
      if (outputIndex >= args.outputs.size()) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "output binding index is out of range: %s",
            binding.name.c_str());
      }
      NDArray &actual = args.outputs[outputIndex++];
      view.shape = actual.shape;
      view.dtype = actual.dtype;
      view.setExternal(actual.data);
    }
    actualOutputs.push_back(std::move(view));
  }

  return actualOutputs;
}

llvm::Expected<std::vector<NDArray>>
selectActualOutputsForExpected(const ExecutionInvocation &invocation,
                               llvm::ArrayRef<NDArray> actualOutputs,
                               llvm::ArrayRef<NDArray> expectedOutputs) {
  std::vector<NDArray> selected;
  selected.reserve(expectedOutputs.size());
  for (size_t expectedIndex = 0; expectedIndex < expectedOutputs.size();
       ++expectedIndex) {
    const TensorBinding &expectedBinding =
        invocation.expectedOutputs[expectedIndex];
    std::optional<size_t> actualIndex = findOutputIndexForExpectedBinding(
        invocation, expectedBinding.name, expectedIndex, actualOutputs.size());
    if (!actualIndex) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "expected output binding has no matching output: %s",
          expectedBinding.name.c_str());
    }
    if (*actualIndex >= actualOutputs.size()) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "expected output binding index is out of range: %s",
          expectedBinding.name.c_str());
    }
    const NDArray &actual = actualOutputs[*actualIndex];
    NDArray view;
    view.shape = actual.shape;
    view.dtype = actual.dtype;
    view.setExternal(actual.data);
    selected.push_back(std::move(view));
  }
  return selected;
}

llvm::Error writeActualOutputs(const ExecutionInvocation &invocation,
                               RunArgs &args) {
  auto actualOutputsOr = collectActualOutputsForInvocation(invocation, args);
  if (!actualOutputsOr)
    return actualOutputsOr.takeError();
  for (size_t index = 0; index < invocation.outputs.size(); ++index) {
    if (auto err =
            SaveNpy(invocation.outputs[index].path, (*actualOutputsOr)[index]))
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

llvm::Expected<int> resolveNpuDeviceIdFromEnv() {
  const char *raw = std::getenv("ASCEND_DEVICE_ID");
  if (!raw || !*raw)
    return 0;

  llvm::StringRef text(raw);
  text = text.trim();
  int64_t deviceId = 0;
  if (text.empty() || text.getAsInteger(10, deviceId) || deviceId < 0 ||
      deviceId > std::numeric_limits<int32_t>::max()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "ASCEND_DEVICE_ID must be a non-negative integer, got '%s'", raw);
  }
  return static_cast<int>(deviceId);
}

llvm::Expected<ExecutionResult>
runWithExecutor(const ExecutionRequest &request,
                const NpuExecutionRunnerFactory &runnerFactory) {
  if (request.task.artifact.kernelKind == KernelKind::Mix) {
    if (request.task.artifact.sharedLibraryPath.empty()) {
      return stageError("artifact",
                        "mix artifact is missing shared library path");
    }
    if (request.task.artifact.sharedLibrarySymbol.empty()) {
      return stageError("artifact",
                        "mix artifact is missing shared library symbol");
    }
  } else if (request.task.artifact.deviceBinaryPath.empty()) {
    return stageError("artifact", "artifact is missing device binary path");
  }

  auto expectedOutputsOr = validateInvocationBindings(request.task.invocation);
  if (!expectedOutputsOr)
    return stageError("bindings", expectedOutputsOr.takeError());

  auto argsOr = buildRunArgs(request.task.invocation, *expectedOutputsOr);
  if (!argsOr)
    return stageError("bindings", argsOr.takeError());
  RunArgs args = std::move(*argsOr);

  auto deviceIdOr = resolveNpuDeviceIdFromEnv();
  if (!deviceIdOr)
    return stageError("executor_initialize", deviceIdOr.takeError());

  const NpuExecutionRunnerFactory &factory =
      runnerFactory ? runnerFactory : createDefaultExecutionRunner;
  auto runnerOr = factory(ExecutionRunnerMode::RealDevice);
  if (!runnerOr)
    return stageError("executor_initialize", runnerOr.takeError());
  std::unique_ptr<ExecutionRunner> runner = std::move(*runnerOr);
  if (auto err = runner->initialize(*deviceIdOr))
    return stageError("executor_initialize", std::move(err));

  if (request.task.artifact.kernelKind == KernelKind::Mix) {
    DynamicLibraryExecutionLaunch launch;
    launch.sharedLibraryPath = request.task.artifact.sharedLibraryPath;
    launch.symbolName = request.task.artifact.sharedLibrarySymbol;
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
    auto actualOutputsOr =
        collectActualOutputsForInvocation(request.task.invocation, args);
    if (!actualOutputsOr)
      return stageError("validate", actualOutputsOr.takeError());
    auto actualForExpectedOr = selectActualOutputsForExpected(
        request.task.invocation, *actualOutputsOr, *expectedOutputsOr);
    if (!actualForExpectedOr) {
      return stageError("validate", actualForExpectedOr.takeError());
    }
    auto validationOr = compareRuntimeOutputs(
        *actualForExpectedOr, *expectedOutputsOr, request.task.invocation.atol,
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

llvm::Expected<ExecutionResult> runWithDriver(
    const ExecutionRequest &request, ExecutionBackendDriver &driver) {
  auto expectedOutputsOr = validateInvocationBindings(request.task.invocation);
  if (!expectedOutputsOr)
    return stageError("bindings", expectedOutputsOr.takeError());

  auto resultOr = driver.run(request);
  if (!resultOr)
    return stageError("driver", resultOr.takeError());
  return std::move(*resultOr);
}

} // namespace

NpuBackend::NpuBackend(std::shared_ptr<ExecutionBackendDriver> driver)
    : NpuBackend(std::move(driver), createDefaultExecutionRunner) {}

NpuBackend::NpuBackend(std::shared_ptr<ExecutionBackendDriver> driver,
                       NpuExecutionRunnerFactory runnerFactory)
    : driver_(std::move(driver)), runnerFactory_(std::move(runnerFactory)) {}

ExecutionBackendKind NpuBackend::kind() const {
  return ExecutionBackendKind::Npu;
}

BackendCapabilities NpuBackend::capabilities() const {
  const std::optional<BackendCapabilities> driverCaps =
      driver_ ? std::optional<BackendCapabilities>(driver_->capabilities())
              : std::nullopt;
  BackendCapabilities caps = driverCaps.value_or(BackendCapabilities{});
  if (!driver_)
    caps.supportsConcurrentDispatch = true;
  if (!driver_)
    caps.supportsConcurrentExecution = true;
  caps.requiresSerializedLaunch = false;
  if (caps.maxConcurrentTasks == 0)
    caps.maxConcurrentTasks = 1;
  if (caps.maxConcurrentStreams == 0)
    caps.maxConcurrentStreams = 1;
  if (driverCaps)
    caps.requiresSerializedLaunch = driverCaps->requiresSerializedLaunch;
  return caps;
}

llvm::Expected<ExecutionResult>
NpuBackend::run(const ExecutionRequest &request) {
  if (driver_)
    return runWithDriver(request, *driver_);
  return runWithExecutor(request, runnerFactory_);
}

} // namespace mlir::runtime
