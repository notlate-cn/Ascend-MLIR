// lib/Runtime/SimBackend.cpp
#include "Runtime/SimBackend.h"
#include "Runtime/Executor.h"
#include "Runtime/NpyIO.h"
#include "Runtime/PathUtils.h"
#include "Runtime/ProfileUtils.h"
#include "Runtime/SimValidator.h"
#include "Runtime/TilingPack.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#include <cstring>
#include <cstdlib>
#include <optional>
#include <type_traits>
#include <utility>

namespace mlir::runtime {

namespace {

template <typename ErrorT>
llvm::Error stageError(llvm::StringRef stage, ErrorT &&errorLike) {
  std::string detail;
  if constexpr (std::is_same_v<std::decay_t<ErrorT>, llvm::Error>) {
    detail = llvm::toString(std::forward<ErrorT>(errorLike));
  } else {
    detail = std::forward<ErrorT>(errorLike);
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "[sim:%s] %s",
                                 stage.str().c_str(), detail.c_str());
}

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

  if (invocation.outputs.empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "simulation path requires at least one output binding");
  }

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
  if (!invocation.outputs.empty() && !expectedOutputsOr->empty() &&
      invocation.outputs.size() != expectedOutputsOr->size()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
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
    if (!binding.shape)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "output binding is missing shape metadata: %s", binding.name.c_str());
    if (!binding.dtype)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "output binding is missing dtype metadata: %s", binding.name.c_str());

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

void prependEnvPath(const char *name, const std::string &prefix) {
  if (prefix.empty())
    return;
  const char *current = std::getenv(name);
  std::string value = prefix;
  if (current && *current) {
    value.push_back(':');
    value += current;
  }
  ::setenv(name, value.c_str(), 1);
}

llvm::Error configurePackedMixEnvironment(const KernelArtifact &artifact) {
  const std::string ascendHome = findAscendHome();
  if (ascendHome.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Ascend toolkit root is not configured; set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME");

  const std::string resolvedSoc =
      resolveSocVersion(artifact.socVersion, "Ascend910B1");
  const std::string ascendLib64 = findAscendLib64Dir(ascendHome);
  const std::string simLibDir =
      findAscendSimulatorLibDir(ascendHome, resolvedSoc);
  auto davSimLibDirOr = requireAscendDavSimulatorLibDir(ascendHome);
  if (!davSimLibDirOr)
    return davSimLibDirOr.takeError();
  const std::string deviceLibDir = findAscendDeviceLibDir(ascendHome);

  prependEnvPath("LD_LIBRARY_PATH", artifact.artifactRoot + "/out");
  prependEnvPath("LD_LIBRARY_PATH", ascendLib64);
  prependEnvPath("LD_LIBRARY_PATH", simLibDir);
  prependEnvPath("LD_LIBRARY_PATH", *davSimLibDirOr);
  prependEnvPath("LD_LIBRARY_PATH", deviceLibDir);

  return llvm::Error::success();
}

llvm::Expected<ExecutionResult>
runWithExecutor(const ExecutionRequest &request) {
  auto cwdGuardOr = WorkingDirectoryGuard::enter(request.workingDirectory);
  if (!cwdGuardOr)
    return stageError("working_directory", cwdGuardOr.takeError());

  if (request.task.artifact.kernelKind == KernelKind::Mix) {
    if (request.task.artifact.packedSharedObjectPath.empty()) {
      return stageError("artifact",
                        "mix artifact is missing packed shared object path");
    }
    if (auto err = configurePackedMixEnvironment(request.task.artifact))
      return stageError("artifact", std::move(err));
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

  Executor executor(BackendMode::Simulation);
  if (auto err = executor.Initialize())
    return stageError("executor_initialize", std::move(err));

  if (request.task.artifact.kernelKind == KernelKind::Mix) {
    const std::string &sharedObjectPath =
        request.task.artifact.packedSharedObjectPath;
    if (auto err = executor.RunPackedMixFile(sharedObjectPath,
                                             request.task.artifact.kernelName,
                                             args)) {
      return stageError("kernel_launch", std::move(err));
    }
  } else {
    const std::string &binaryPath = request.task.artifact.deviceBinaryPath;
    if (auto err = executor.RunFile(binaryPath, request.task.artifact.kernelName,
                                    args,
                                    magicForKernelKind(
                                        request.task.artifact.kernelKind))) {
      return stageError("kernel_launch", std::move(err));
    }
  }

  if (!expectedOutputsOr->empty()) {
    SimValidator validator;
    SimValidator::Result validation = validator.CompareOnly(
        args, *expectedOutputsOr, request.task.invocation.atol,
        request.task.invocation.rtol);
    if (!validation.error_msg.empty()) {
      return stageError("validate", validation.error_msg);
    }
    if (!validation.passed) {
      return stageError(
          "validate",
          llvm::formatv("simulation output mismatch: max_abs_diff={0:F} "
                        "mean_abs_diff={1:F}",
                        validation.max_abs_diff, validation.mean_abs_diff)
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
