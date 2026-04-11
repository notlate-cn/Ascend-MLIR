// lib/Runtime/SimBackend.cpp
#include "Runtime/SimBackend.h"
#include "Runtime/Executor.h"
#include "Runtime/NpyIO.h"
#include "Runtime/ProfileUtils.h"
#include "Runtime/SimValidator.h"
#include "Runtime/TilingPack.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <cstring>
#include <optional>
#include <utility>

namespace mlir::runtime {

namespace {

class WorkingDirectoryGuard {
public:
  static llvm::Expected<WorkingDirectoryGuard>
  enter(const std::string &newWorkingDirectory) {
    llvm::SmallString<256> previousDirectory;
    if (auto ec = llvm::sys::fs::current_path(previousDirectory))
      return llvm::createStringError(ec, "cannot read current working directory");
    if (auto ec = llvm::sys::fs::set_current_path(newWorkingDirectory))
      return llvm::createStringError(ec, "cannot enter working directory: %s",
                                     newWorkingDirectory.c_str());
    return WorkingDirectoryGuard(previousDirectory.str().str());
  }

  ~WorkingDirectoryGuard() {
    if (!previousDirectory_.empty())
      (void)llvm::sys::fs::set_current_path(previousDirectory_);
  }

  WorkingDirectoryGuard(WorkingDirectoryGuard &&other) noexcept
      : previousDirectory_(std::move(other.previousDirectory_)) {
    other.previousDirectory_.clear();
  }

  WorkingDirectoryGuard &operator=(WorkingDirectoryGuard &&other) noexcept {
    if (this != &other) {
      previousDirectory_ = std::move(other.previousDirectory_);
      other.previousDirectory_.clear();
    }
    return *this;
  }

  WorkingDirectoryGuard(const WorkingDirectoryGuard &) = delete;
  WorkingDirectoryGuard &operator=(const WorkingDirectoryGuard &) = delete;

private:
  explicit WorkingDirectoryGuard(std::string previousDirectory)
      : previousDirectory_(std::move(previousDirectory)) {}

  std::string previousDirectory_;
};

llvm::Expected<std::vector<NDArray>>
loadExpectedOutputs(const ExecutionInvocation &invocation) {
  std::vector<NDArray> expected;
  for (const TensorBinding &binding : invocation.expectedOutputs) {
    if (binding.sourceKind != BindingSourceKind::ExternalFile)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
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

  auto tilingOr = packTilingBytes(invocation.tiling);
  if (!tilingOr)
    return tilingOr.takeError();
  args.tiling = std::move(*tilingOr);

  for (const TensorBinding &binding : invocation.inputs) {
    if (binding.sourceKind != BindingSourceKind::ExternalFile)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "input binding must be an external file: %s",
                                     binding.name.c_str());
    auto arrayOr = LoadNpy(binding.path);
    if (!arrayOr)
      return arrayOr.takeError();
    args.inputs.push_back(std::move(*arrayOr));
  }

  auto expectedOutputsOr = loadExpectedOutputs(invocation);
  if (!expectedOutputsOr)
    return expectedOutputsOr.takeError();
  if (!invocation.outputs.empty() &&
      invocation.outputs.size() != expectedOutputsOr->size()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "output binding count does not match expected output count");
  }
  if (expectedOutputsOr->empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "simulation path currently requires expected_outputs to infer output buffers");
  }

  for (const NDArray &expected : *expectedOutputsOr) {
    NDArray output;
    output.shape = expected.shape;
    output.dtype = expected.dtype;
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

llvm::Expected<ExecutionResult>
runWithExecutor(const ExecutionRequest &request) {
  auto cwdGuardOr = WorkingDirectoryGuard::enter(request.workingDirectory);
  if (!cwdGuardOr)
    return cwdGuardOr.takeError();

  auto argsOr = buildRunArgs(request.task.invocation);
  if (!argsOr)
    return argsOr.takeError();
  RunArgs args = std::move(*argsOr);

  auto expectedOutputsOr = loadExpectedOutputs(request.task.invocation);
  if (!expectedOutputsOr)
    return expectedOutputsOr.takeError();

  Executor executor(BackendMode::Simulation);
  if (auto err = executor.Initialize())
    return std::move(err);

  if (request.task.artifact.kernelKind == KernelKind::Mix) {
    const std::string &sharedObjectPath =
        request.task.artifact.packedSharedObjectPath;
    if (sharedObjectPath.empty()) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "mix artifact is missing packed shared object path");
    }
    if (auto err = executor.RunPackedMixFile(sharedObjectPath,
                                             request.task.artifact.kernelName,
                                             args)) {
      return std::move(err);
    }
  } else {
    const std::string &binaryPath = request.task.artifact.deviceBinaryPath;
    if (binaryPath.empty()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "artifact is missing device binary path");
    }
    if (auto err = executor.RunFile(binaryPath, request.task.artifact.kernelName,
                                    args,
                                    magicForKernelKind(
                                        request.task.artifact.kernelKind))) {
      return std::move(err);
    }
  }

  SimValidator validator;
  SimValidator::Result validation = validator.CompareOnly(
      args, *expectedOutputsOr, /*atol=*/1.0, /*rtol=*/1e-2);
  if (!validation.error_msg.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                   validation.error_msg.c_str());
  }
  if (!validation.passed) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "simulation output mismatch: max_abs_diff=%f mean_abs_diff=%f",
        validation.max_abs_diff, validation.mean_abs_diff);
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

SimBackend::SimBackend(std::shared_ptr<ExecutionBackendDriver> driver)
    : driver_(std::move(driver)) {}

ExecutionBackendKind SimBackend::kind() const {
  return ExecutionBackendKind::Simulation;
}

llvm::Expected<ExecutionResult>
SimBackend::run(const ExecutionRequest &request) {
  llvm::Expected<ExecutionResult> resultOr =
      driver_ ? driver_->run(request) : runWithExecutor(request);
  if (!resultOr)
    return resultOr.takeError();

  const std::string &sessionId =
      request.sessionId.empty() ? request.task.taskId : request.sessionId;
  auto traceOr = normalizeSimulatorProfileTrace(sessionId,
                                                request.task.taskId,
                                                resultOr->producedFiles);
  if (traceOr && !resultOr->profileTrace)
    resultOr->profileTrace = std::move(*traceOr);

  return resultOr;
}

} // namespace mlir::runtime
