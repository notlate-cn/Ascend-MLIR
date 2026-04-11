// lib/Runtime/NpuBackend.cpp
#include "Runtime/NpuBackend.h"

#include "Runtime/Executor.h"
#include "Runtime/NpyIO.h"
#include "Runtime/SimValidator.h"
#include "Runtime/TilingPack.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>

namespace mlir::runtime {

namespace {

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
    return Executor::MAGIC_ELF_AIVEC;
  case KernelKind::Cube:
    return Executor::MAGIC_ELF_AICUBE;
  }
  return Executor::MAGIC_ELF_AIVEC;
}

llvm::Expected<ExecutionResult> runWithExecutor(const ExecutionRequest &request) {
  if (request.task.artifact.kernelKind == KernelKind::Mix) {
    if (request.task.artifact.packedSharedObjectPath.empty()) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "mix artifact is missing packed shared object path");
    }
  } else if (request.task.artifact.deviceBinaryPath.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "artifact is missing device binary path");
  }

  auto argsOr = buildRunArgs(request.task.invocation);
  if (!argsOr)
    return argsOr.takeError();
  RunArgs args = std::move(*argsOr);

  auto expectedOutputsOr = loadExpectedOutputs(request.task.invocation);
  if (!expectedOutputsOr)
    return expectedOutputsOr.takeError();

  Executor executor(BackendMode::RealDevice);
  if (auto err = executor.Initialize())
    return std::move(err);

  if (request.task.artifact.kernelKind == KernelKind::Mix) {
    if (auto err = executor.RunPackedMixFile(
            request.task.artifact.packedSharedObjectPath,
            request.task.artifact.kernelName, args)) {
      return std::move(err);
    }
  } else {
    if (auto err = executor.RunFile(request.task.artifact.deviceBinaryPath,
                                    request.task.artifact.kernelName, args,
                                    magicForKernelKind(
                                        request.task.artifact.kernelKind))) {
      return std::move(err);
    }
  }

  if (!expectedOutputsOr->empty()) {
    SimValidator validator;
    SimValidator::Result validation = validator.CompareOnly(
        args, *expectedOutputsOr, request.task.invocation.atol,
        request.task.invocation.rtol);
    if (!validation.error_msg.empty()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                     validation.error_msg.c_str());
    }
    if (!validation.passed) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "npu output mismatch: max_abs_diff=%f mean_abs_diff=%f",
          validation.max_abs_diff, validation.mean_abs_diff);
    }
  }

  if (auto err = writeActualOutputs(request.task.invocation, args))
    return std::move(err);

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
