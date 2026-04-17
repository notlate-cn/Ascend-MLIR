// lib/Runtime/SimBackend.cpp
#include "Runtime/SimBackend.h"
#include "Runtime/Execution/DefaultExecutionRunner.h"
#include "Runtime/Execution/DynamicLibraryArtifactEnv.h"
#include "Runtime/NpyIO.h"
#include "Runtime/OutputComparator.h"
#include "Runtime/ProfileUtils.h"
#include "Runtime/TilingPack.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>

namespace mlir::runtime {

namespace {

constexpr uint32_t kMagicElfAiVec = 0x41415246u;
constexpr uint32_t kMagicElfAiCube = 0x41494343u;

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

class SimulatorDispatchQueue {
public:
  static SimulatorDispatchQueue &instance() {
    static SimulatorDispatchQueue queue;
    return queue;
  }

  std::optional<std::string>
  run(std::function<std::optional<std::string>(ExecutionRunner &)> fn) {
    auto task = std::make_shared<
        std::packaged_task<std::optional<std::string>()>>([this, fn = std::move(fn)]() mutable {
      auto runnerOr = getOrCreateRunner();
      if (!runnerOr)
        return std::optional<std::string>(llvm::toString(runnerOr.takeError()));
      return fn(**runnerOr);
    });
    auto future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      workQueue_.push_back([task]() mutable { (*task)(); });
    }
    cv_.notify_one();
    return future.get();
  }

private:
  SimulatorDispatchQueue() : worker_([this] { workerLoop(); }) {}

  ~SimulatorDispatchQueue() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable())
      worker_.join();
  }

  void workerLoop() {
    while (true) {
      std::function<void()> work;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stopping_ || !workQueue_.empty(); });
        if (stopping_ && workQueue_.empty()) {
          runner_.reset();
          return;
        }
        work = std::move(workQueue_.front());
        workQueue_.pop_front();
      }
      work();
    }
  }

  llvm::Expected<ExecutionRunner *> getOrCreateRunner() {
    if (!runner_) {
      auto runnerOr = createDefaultExecutionRunner(ExecutionRunnerMode::Simulation);
      if (!runnerOr)
        return runnerOr.takeError();
      runner_ = std::move(*runnerOr);
      if (auto err = runner_->initialize()) {
        runner_.reset();
        return std::move(err);
      }
    }
    return runner_.get();
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> workQueue_;
  bool stopping_ = false;
  std::thread worker_;
  std::unique_ptr<ExecutionRunner> runner_;
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

llvm::StringRef kernelKindToString(KernelKind kind) {
  switch (kind) {
  case KernelKind::Vec:
    return "vec";
  case KernelKind::Cube:
    return "cube";
  case KernelKind::Mix:
    return "mix";
  }
  return "vec";
}

llvm::StringRef dtypeToShortName(DType dtype) {
  switch (dtype) {
  case DType::F16:
    return "f16";
  case DType::BF16:
    return "bf16";
  case DType::F32:
    return "f32";
  case DType::INT8:
    return "i8";
  case DType::INT32:
    return "i32";
  case DType::INT64:
    return "i64";
  }
  return "f16";
}

llvm::json::Array toJsonShape(llvm::ArrayRef<int64_t> shape) {
  llvm::json::Array jsonShape;
  for (int64_t dim : shape)
    jsonShape.push_back(dim);
  return jsonShape;
}

llvm::Expected<llvm::json::Array>
buildProfileTensorArray(llvm::ArrayRef<TensorBinding> bindings,
                        llvm::ArrayRef<NDArray> arrays) {
  if (bindings.size() != arrays.size()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "profile tensor metadata count does not match runtime array count");
  }
  llvm::json::Array tensors;
  for (size_t index = 0; index < bindings.size(); ++index) {
    const TensorBinding &binding = bindings[index];
    const NDArray &array = arrays[index];
    llvm::json::Object tensor;
    tensor["name"] = binding.name;
    tensor["shape"] = toJsonShape(array.shape);
    tensor["dtype"] = dtypeToShortName(array.dtype);
    tensors.push_back(std::move(tensor));
  }
  return tensors;
}

llvm::Expected<llvm::json::Object>
buildProfileTilingObject(const ExecutionRequest &request,
                         llvm::ArrayRef<uint8_t> tilingBytes) {
  llvm::json::Object tilingObject;
  const std::optional<TilingBinding> &tiling = request.task.invocation.tiling;
  tilingObject["present"] = static_cast<bool>(tiling);

  std::string tilingPath;
  if (tiling) {
    if (!tiling->binaryPath.empty()) {
      tilingPath = tiling->binaryPath;
    } else {
      llvm::SmallString<256> materializedPath(request.workingDirectory);
      llvm::sys::path::append(materializedPath, "tiling.bin");
      std::error_code ec;
      llvm::raw_fd_ostream os(materializedPath, ec, llvm::sys::fs::OF_None);
      if (ec) {
        return llvm::createStringError(
            ec, "cannot write simulator tiling artifact: %s",
            materializedPath.c_str());
      }
      if (!tilingBytes.empty()) {
        os.write(reinterpret_cast<const char *>(tilingBytes.data()),
                 tilingBytes.size());
      }
      os.flush();
      if (os.has_error()) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "failed to flush simulator tiling artifact: %s",
            materializedPath.c_str());
      }
      tilingPath = materializedPath.str().str();
    }
  }
  tilingObject["binary_path"] = tilingPath;
  tilingObject["bytes"] = static_cast<int64_t>(tilingBytes.size());
  return tilingObject;
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

llvm::Expected<std::string>
materializeSimulatorProfileArtifact(const ExecutionRequest &request,
                                    const RunArgs &args,
                                    int64_t cycleCount) {
  llvm::SmallString<256> profileDir(request.workingDirectory);
  llvm::sys::path::append(profileDir, "opprof", "simulator");
  if (auto ec = llvm::sys::fs::create_directories(profileDir))
    return llvm::createStringError(ec,
                                   "cannot create simulator profile directory: %s",
                                   profileDir.c_str());

  llvm::SmallString<256> profilePath(profileDir);
  llvm::sys::path::append(profilePath, "trace.json");

  llvm::json::Object root;
  root["schema_version"] = 1;
  root["backend"] = "simulation";
  root["session_id"] = request.sessionId;
  root["task_id"] = request.task.taskId;
  root["kernel_name"] = request.task.artifact.kernelName;
  root["kernel_kind"] = std::string(kernelKindToString(request.task.artifact.kernelKind));
  root["soc_version"] = request.task.artifact.socVersion;
  root["block_dim"] = request.task.invocation.blockDim;
  root["workspace_size"] = static_cast<int64_t>(args.workspace_size);
  root["cycle_count"] = cycleCount;
  root["elapsed_us"] = cycleCount;
  root["score"] = cycleCount;
  root["validation_passed"] = true;
  root["artifact_root"] = request.task.artifact.artifactRoot;
  auto inputsOr = buildProfileTensorArray(request.task.invocation.inputs,
                                          args.inputs);
  if (!inputsOr)
    return inputsOr.takeError();
  root["inputs"] = std::move(*inputsOr);
  auto outputsOr = buildProfileTensorArray(request.task.invocation.outputs,
                                           args.outputs);
  if (!outputsOr)
    return outputsOr.takeError();
  root["outputs"] = std::move(*outputsOr);
  auto tilingObjectOr = buildProfileTilingObject(request, args.tiling);
  if (!tilingObjectOr)
    return tilingObjectOr.takeError();
  root["tiling"] = std::move(*tilingObjectOr);

  std::error_code ec;
  llvm::raw_fd_ostream os(profilePath, ec);
  if (ec)
    return llvm::createStringError(ec, "cannot write simulator profile artifact: %s",
                                   profilePath.c_str());
  llvm::json::OStream jos(os, /*IndentSize=*/2);
  jos.value(llvm::json::Value(std::move(root)));
  os << "\n";
  os.flush();
  if (os.has_error())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "failed to flush simulator profile artifact: %s", profilePath.c_str());

  return profilePath.str().str();
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

llvm::Expected<ExecutionResult>
runWithExecutor(const ExecutionRequest &request) {
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

  auto argsOr = buildRunArgs(request.task.invocation);
  if (!argsOr)
    return stageError("bindings", argsOr.takeError());
  RunArgs args = std::move(*argsOr);

  auto expectedOutputsOr = loadExpectedOutputs(request.task.invocation);
  if (!expectedOutputsOr)
    return stageError("bindings", expectedOutputsOr.takeError());

  auto runStart = std::chrono::steady_clock::now();

  auto launchError = SimulatorDispatchQueue::instance().run(
      [&](ExecutionRunner &runner) -> std::optional<std::string> {
    if (request.task.artifact.kernelKind == KernelKind::Mix) {
      if (auto err = configureDynamicLibraryArtifactSimulationEnv(
              request.task.artifact))
        return llvm::toString(stageError("artifact", std::move(err)));
    }

    if (request.task.artifact.kernelKind == KernelKind::Mix) {
      DynamicLibraryExecutionLaunch launch;
      launch.sharedLibraryPath = request.task.artifact.sharedLibraryPath;
      launch.symbolName = request.task.artifact.sharedLibrarySymbol;
      if (auto err = runner.runDynamicLibraryArtifact(launch, args))
        return llvm::toString(stageError("kernel_launch", std::move(err)));
    } else {
      FileExecutionLaunch launch;
      launch.binaryPath = request.task.artifact.deviceBinaryPath;
      launch.kernelName = request.task.artifact.kernelName;
      launch.magic = magicForKernelKind(request.task.artifact.kernelKind);
      if (auto err = runner.runFile(launch, args))
        return llvm::toString(stageError("kernel_launch", std::move(err)));
    }
    return std::nullopt;
  });
  if (launchError)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                   launchError->c_str());

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
          llvm::formatv("simulation output mismatch: max_abs_diff={0:F} "
                        "mean_abs_diff={1:F}",
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
  ProfileTrace runtimeTrace;
  runtimeTrace.sessionId = request.sessionId;
  runtimeTrace.setAttribute("simulator_launch_model", "dispatch_thread");
  runtimeTrace.addCounter("serialized_launch_count", 1);
  if (request.task.invocation.enableProfiling) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - runStart);
    const int64_t elapsedUs = elapsed.count();
    auto profilePathOr =
        materializeSimulatorProfileArtifact(request, args, elapsedUs);
    if (!profilePathOr)
      return stageError("profiling", profilePathOr.takeError());
    result.producedFiles.push_back(*profilePathOr);
    runtimeTrace.addProfileArtifact(request.task.taskId,
                                    ExecutionBackendKind::Simulation,
                                    *profilePathOr, elapsedUs, elapsedUs);
  }
  result.profileTrace = std::move(runtimeTrace);
  return result;
}

} // namespace

llvm::Expected<std::string>
materializeSimulatorProfileArtifactForTest(const ExecutionRequest &request,
                                           int64_t cycleCount) {
  auto argsOr = buildRunArgs(request.task.invocation);
  if (!argsOr)
    return argsOr.takeError();
  return materializeSimulatorProfileArtifact(request, *argsOr, cycleCount);
}

SimBackend::SimBackend(std::shared_ptr<ExecutionBackendDriver> driver)
    : driver_(std::move(driver)) {}

ExecutionBackendKind SimBackend::kind() const {
  return ExecutionBackendKind::Simulation;
}

bool SimBackend::allowsConcurrentTaskDispatch() const {
  return driver_ ? driver_->allowsConcurrentTaskDispatch() : true;
}

llvm::Expected<ExecutionResult>
SimBackend::run(const ExecutionRequest &request) {
  llvm::Expected<ExecutionResult> resultOr =
      driver_ ? driver_->run(request) : runWithExecutor(request);
  if (!resultOr)
    return resultOr.takeError();

  if (!resultOr->profileTrace) {
    const std::string &sessionId =
        request.sessionId.empty() ? request.task.taskId : request.sessionId;
    auto traceOr = normalizeSimulatorProfileTrace(sessionId,
                                                  request.task.taskId,
                                                  resultOr->producedFiles);
    if (traceOr)
      resultOr->profileTrace = std::move(*traceOr);
  }

  return resultOr;
}

} // namespace mlir::runtime
