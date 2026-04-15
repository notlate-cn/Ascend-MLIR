// test/tools/runtime/test_taskgraph_runtime.cpp
//
// Focused unit test for the runtime task graph / profile trace object model.
//
// Build (on xvm):
//   cd /home/niu/code/Codex-Ascend-MLIR
//   c++ -std=c++17 -I include -I /home/niu/code/llvm-project/build/include \
//       -I /home/niu/code/llvm-project/llvm/include \
//       test/tools/runtime/test_taskgraph_runtime.cpp \
//       build/lib/libAscendCRuntime.a \
//       $(/home/niu/code/llvm-project/build/bin/llvm-config --ldflags --libs support) \
//       -ldl -o /tmp/test_taskgraph_runtime
//   /tmp/test_taskgraph_runtime

#include "Runtime/ProfileTrace.h"
#include "Runtime/ProfileUtils.h"
#include "Runtime/MixAbi.h"
#include "Runtime/MixArtifact.h"
#include "Runtime/RunManifest.h"
#include "Runtime/ExecutionBackend.h"
#include "Runtime/Execution/DefaultExecutionRunner.h"
#include "Runtime/Execution/ExecutionRunner.h"
#include "Runtime/Execution/NativeExecutionRunner.h"
#include "Runtime/ExecutionSession.h"
#include "Runtime/NpuBackend.h"
#include "Runtime/NpyIO.h"
#include "Runtime/TilingPack.h"
#include "Runtime/TilingSchema.h"
#include "Runtime/TaskGraph.h"
#include "Runtime/ArtifactCompiler.h"
#include "Runtime/Execution/RuntimeFrontendCore.h"
#include "Runtime/VecCubeArtifactBackend.h"
#include "Runtime/RuntimeSessionRequestBuilder.h"
#include "Runtime/SimBackend.h"
#include "Runtime/OutputComparator.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <chrono>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace mlir::runtime;

namespace mlir::runtime {
llvm::Expected<std::string>
materializeSimulatorProfileArtifactForTest(const ExecutionRequest &request,
                                           int64_t cycleCount);
llvm::Expected<ProfileTrace>
retainProfileArtifactsForCli(const ProfileTrace &trace,
                             llvm::StringRef destinationRoot);
llvm::Expected<RetainedProfileCliArtifacts>
retainProfileArtifactsForCliRun(const ProfileTrace &trace,
                                llvm::StringRef destinationRoot);
llvm::Error pruneRetainedProfileDirectoriesForTest(llvm::StringRef root,
                                                   size_t keepCount);
size_t retainedProfilePruneKeepCountForNewSession(size_t sessionLimit);
llvm::Expected<std::string>
prepareRetainedProfileRunRootForCli(llvm::StringRef destinationRoot,
                                    size_t sessionLimit);
}

static int g_pass = 0;
static int g_fail = 0;

static std::string writeTempNpy(const std::string &stem,
                                const std::vector<int64_t> &shape,
                                DType dtype) {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() / (stem + ".npy");
  NDArray array;
  array.shape = shape;
  array.dtype = dtype;
  array.allocate();
  std::memset(array.data, 0, array.nbytes());
  auto err = SaveNpy(path.string(), array);
  if (err) {
    llvm::errs() << "FAIL: cannot write temp npy " << path.string() << "\n";
    llvm::consumeError(std::move(err));
    ++g_fail;
    return "";
  }
  return path.string();
}

static std::string readFileContents(const std::string &path);

static std::string readTextFile(const std::string &path) {
  std::ifstream is(path);
  if (!is) {
    llvm::errs() << "FAIL: cannot read temp file " << path << "\n";
    ++g_fail;
    return "";
  }
  return std::string((std::istreambuf_iterator<char>(is)),
                     std::istreambuf_iterator<char>());
}

static NDArray makeArray(std::vector<int64_t> shape, DType dtype,
                         std::vector<uint8_t> bytes) {
  NDArray array;
  array.shape = std::move(shape);
  array.dtype = dtype;
  array.allocate();
  std::memcpy(array.data, bytes.data(), bytes.size());
  return array;
}

static NDArray makeF32Array(std::vector<int64_t> shape,
                            std::vector<float> values) {
  NDArray array;
  array.shape = std::move(shape);
  array.dtype = DType::F32;
  array.allocate();
  std::memcpy(array.data, values.data(), values.size() * sizeof(float));
  return array;
}

class FakeExecutionRunner final : public ExecutionRunner {
public:
  explicit FakeExecutionRunner(ExecutionRunnerMode mode) : mode_(mode) {}

  ExecutionRunnerMode mode() const override { return mode_; }

  llvm::Error initialize(int deviceId = 0) override {
    initializeCalled = true;
    lastDeviceId = deviceId;
    return llvm::Error::success();
  }

  llvm::Error runFile(const FileExecutionLaunch &launch, RunArgs &) override {
    runFileCalled = true;
    lastFileLaunch = launch;
    return llvm::Error::success();
  }

  llvm::Error runPackedMixFile(const PackedMixExecutionLaunch &launch,
                               RunArgs &) override {
    runPackedMixCalled = true;
    lastPackedMixLaunch = launch;
    return llvm::Error::success();
  }

  bool initializeCalled = false;
  bool runFileCalled = false;
  bool runPackedMixCalled = false;
  int lastDeviceId = -1;
  FileExecutionLaunch lastFileLaunch;
  PackedMixExecutionLaunch lastPackedMixLaunch;

private:
  ExecutionRunnerMode mode_;
};

static std::filesystem::path makeTempDir(const std::string &stem) {
  static int uniqueCounter = 0;
  return std::filesystem::temp_directory_path() /
         (stem + "-" + std::to_string(++uniqueCounter));
}

static std::filesystem::path makeRuntimeSessionArtifactRoot(
    const std::string &stem) {
  std::filesystem::path root = makeTempDir(stem);
  std::filesystem::create_directories(root / "out");

  std::ofstream manifest(root / "out" / "manifest.txt");
  if (!manifest) {
    llvm::errs() << "FAIL: cannot write runtime session manifest "
                 << (root / "out" / "manifest.txt").string() << "\n";
    ++g_fail;
    return {};
  }

  manifest << "kernel_name=fake_kernel\n";
  manifest << "soc_version=Ascend910B1\n";
  manifest << "kernel_kind=mix\n";
  manifest << "mix_resource_type=mix_1c1v\n";
  manifest << "device_binary_path=fake.bin\n";

  std::ofstream binary(root / "fake.bin", std::ios::binary);
  if (!binary) {
    llvm::errs() << "FAIL: cannot write runtime session binary "
                 << (root / "fake.bin").string() << "\n";
    ++g_fail;
    return {};
  }
  binary.put('\0');
  return root;
}

static std::filesystem::path makeRuntimeSessionArtifactRootWithMetadata(
    const std::string &stem, bool writeMetadataFile) {
  std::filesystem::path root = makeTempDir(stem);
  std::filesystem::create_directories(root / "out");

  std::ofstream manifest(root / "out" / "manifest.txt");
  if (!manifest) {
    llvm::errs() << "FAIL: cannot write runtime session manifest "
                 << (root / "out" / "manifest.txt").string() << "\n";
    ++g_fail;
    return {};
  }

  manifest << "kernel_name=fake_kernel\n";
  manifest << "soc_version=Ascend910B1\n";
  manifest << "kernel_kind=mix\n";
  manifest << "mix_resource_type=mix_1c1v\n";
  manifest << "device_binary_path=fake.bin\n";
  manifest << "metadata_path=out/mix_metadata.json\n";

  std::ofstream binary(root / "fake.bin", std::ios::binary);
  if (!binary) {
    llvm::errs() << "FAIL: cannot write runtime session binary "
                 << (root / "fake.bin").string() << "\n";
    ++g_fail;
    return {};
  }
  binary.put('\0');

  if (writeMetadataFile) {
    std::ofstream metadata(root / "out" / "mix_metadata.json");
    if (!metadata) {
      llvm::errs() << "FAIL: cannot write runtime session metadata "
                   << (root / "out" / "mix_metadata.json").string() << "\n";
      ++g_fail;
      return {};
    }
    metadata << "{\n"
             << "  \"schema_version\": 1,\n"
             << "  \"kernel_kind\": \"mix\",\n"
             << "  \"kernel_name\": \"fake_kernel\",\n"
             << "  \"runtime_kernel_name\": \"fake_kernel\",\n"
             << "  \"soc_version\": \"Ascend910B1\",\n"
             << "  \"mix_kernel_type\": \"mix_aic_1_2\",\n"
             << "  \"launcher_symbol\": \"aclrtlaunch_fake_kernel\",\n"
             << "  \"entries\": { \"aic\": \"fake_kernel_0_mix_aic\", \"aiv\": \"fake_kernel_0_mix_aiv\" },\n"
             << "  \"generated\": { \"source_path\": \"work/generated/auto_gen_fake_kernel.cpp\" },\n"
             << "  \"device_compile\": {\n"
             << "    \"aic_arch\": \"dav-c220-cube\",\n"
             << "    \"aiv_arch\": \"dav-c220-vec\",\n"
             << "    \"aic_definitions\": [],\n"
             << "    \"aiv_definitions\": []\n"
             << "  },\n"
             << "  \"artifacts\": {\n"
             << "    \"device_object_path\": \"out/device.o\",\n"
             << "    \"packed_shared_object_path\": \"out/libfake_kernel_packed.so\",\n"
             << "    \"tiling_file_path\": \"out/tiling.bin\",\n"
             << "    \"launch_info_file_path\": \"out/launch_info.txt\"\n"
             << "  },\n"
             << "  \"abi\": {\n"
             << "    \"workspace_mode\": \"fixed\",\n"
             << "    \"workspace_bytes\": 16777216,\n"
             << "    \"tiling_mode\": \"generated_file\",\n"
             << "    \"tiling_source\": \"out/tiling.bin\",\n"
             << "    \"inputs\": [],\n"
             << "    \"outputs\": []\n"
             << "  },\n"
             << "  \"host_launch\": {\n"
             << "    \"mode\": \"helper\",\n"
             << "    \"helper_kind\": \"mix-tiling-helper\",\n"
             << "    \"helper_inputs\": {}\n"
             << "  }\n"
             << "}\n";
  }

  return root;
}

static std::filesystem::path makeRuntimeSessionMixArtifactRootWithAbiDefaults(
    const std::string &stem, bool writeMetadataFile) {
  std::filesystem::path root = makeTempDir(stem);
  std::filesystem::create_directories(root / "out");

  std::ofstream manifest(root / "out" / "manifest.txt");
  if (!manifest) {
    llvm::errs() << "FAIL: cannot write mix artifact manifest "
                 << (root / "out" / "manifest.txt").string() << "\n";
    ++g_fail;
    return {};
  }

  manifest << "kernel_name=fake_kernel\n";
  manifest << "requested_kernel_name=fake_kernel\n";
  manifest << "soc_version=Ascend910B1\n";
  manifest << "kernel_kind=mix\n";
  manifest << "mix_resource_type=mix_1c1v\n";
  manifest << "device_binary_path=fake.bin\n";
  manifest << "abi_input_count=2\n";
  manifest << "abi_input0_name=lhs\n";
  manifest << "abi_input0_file=fake_kernel.lhs.input.bin\n";
  manifest << "abi_input0_dtype=f16\n";
  manifest << "abi_input0_shape=16,16\n";
  manifest << "abi_input1_name=rhs\n";
  manifest << "abi_input1_file=fake_kernel.rhs.input.bin\n";
  manifest << "abi_input1_dtype=f16\n";
  manifest << "abi_input1_shape=16,16\n";
  manifest << "abi_output_count=1\n";
  manifest << "abi_output0_name=out\n";
  manifest << "abi_output0_file=fake_kernel.out.output.bin\n";
  manifest << "abi_output0_dtype=f16\n";
  manifest << "abi_output0_shape=16,16\n";
  manifest << "abi_output0_golden_file=fake_kernel.out.golden.bin\n";
  manifest << "abi_workspace_bytes=2048\n";
  manifest << "abi_block_dim=3\n";
  manifest << "abi_workspace_mode=fixed\n";
  manifest << "abi_tiling_mode=generated_file\n";
  manifest << "abi_tiling_source=out/legacy_tiling.bin\n";
  manifest << "abi_launcher_symbol=aclrtlaunch_fake_kernel_legacy\n";
  manifest << "abi_aic_entry=fake_kernel_legacy_aic\n";
  manifest << "abi_aiv_entry=fake_kernel_legacy_aiv\n";
  if (writeMetadataFile)
    manifest << "metadata_path=out/mix_metadata.json\n";

  std::ofstream binary(root / "fake.bin", std::ios::binary);
  if (!binary) {
    llvm::errs() << "FAIL: cannot write mix artifact binary "
                 << (root / "fake.bin").string() << "\n";
    ++g_fail;
    return {};
  }
  binary.put('\0');

  std::ofstream legacyTiling(root / "out" / "legacy_tiling.bin",
                             std::ios::binary);
  if (!legacyTiling) {
    llvm::errs() << "FAIL: cannot write legacy tiling binary "
                 << (root / "out" / "legacy_tiling.bin").string() << "\n";
    ++g_fail;
    return {};
  }
  legacyTiling.put('\0');

  if (writeMetadataFile) {
    std::ofstream tiling(root / "out" / "tiling.bin", std::ios::binary);
    if (!tiling) {
      llvm::errs() << "FAIL: cannot write metadata tiling binary "
                   << (root / "out" / "tiling.bin").string() << "\n";
      ++g_fail;
      return {};
    }
    tiling.put('\1');

    std::ofstream launchInfo(root / "out" / "launch_info.txt");
    if (!launchInfo) {
      llvm::errs() << "FAIL: cannot write metadata launch info "
                   << (root / "out" / "launch_info.txt").string() << "\n";
      ++g_fail;
      return {};
    }
    launchInfo << "block_dim=8\n";

    std::ofstream metadata(root / "out" / "mix_metadata.json");
    if (!metadata) {
      llvm::errs() << "FAIL: cannot write mix metadata "
                   << (root / "out" / "mix_metadata.json").string() << "\n";
      ++g_fail;
      return {};
    }
    metadata << "{\n"
             << "  \"schema_version\": 1,\n"
             << "  \"kernel_kind\": \"mix\",\n"
             << "  \"kernel_name\": \"fake_kernel\",\n"
             << "  \"runtime_kernel_name\": \"fake_kernel\",\n"
             << "  \"soc_version\": \"Ascend910B1\",\n"
             << "  \"mix_kernel_type\": \"mix_1c1v\",\n"
             << "  \"launcher_symbol\": \"aclrtlaunch_fake_kernel\",\n"
             << "  \"entries\": { \"aic\": \"fake_kernel_0_mix_aic\", \"aiv\": \"fake_kernel_0_mix_aiv\" },\n"
             << "  \"generated\": { \"source_path\": \"work/generated/auto_gen_fake_kernel.cpp\" },\n"
             << "  \"device_compile\": {\n"
             << "    \"aic_arch\": \"dav-c220-cube\",\n"
             << "    \"aiv_arch\": \"dav-c220-vec\",\n"
             << "    \"aic_definitions\": [],\n"
             << "    \"aiv_definitions\": []\n"
             << "  },\n"
             << "  \"artifacts\": {\n"
             << "    \"device_object_path\": \"out/device.o\",\n"
             << "    \"packed_shared_object_path\": \"out/libfake_kernel_packed.so\",\n"
             << "    \"tiling_file_path\": \"out/tiling.bin\",\n"
             << "    \"launch_info_file_path\": \"out/launch_info.txt\"\n"
             << "  },\n"
             << "  \"abi\": {\n"
             << "    \"workspace_mode\": \"fixed\",\n"
             << "    \"workspace_bytes\": 16777216,\n"
             << "    \"tiling_mode\": \"generated_file\",\n"
             << "    \"tiling_source\": \"out/tiling.bin\",\n"
             << "    \"workspace_arg_index\": 5,\n"
             << "    \"tiling_arg_index\": 6,\n"
             << "    \"inputs\": [\n"
             << "      { \"name\": \"lhs\", \"dtype\": \"f32\", \"shape\": [4, 8], \"runtime_file\": \"fake_kernel.lhs.input.bin\" },\n"
             << "      { \"name\": \"rhs\", \"dtype\": \"f32\", \"shape\": [8, 4], \"runtime_file\": \"fake_kernel.rhs.input.bin\" }\n"
             << "    ],\n"
             << "    \"outputs\": [\n"
             << "      { \"name\": \"out\", \"dtype\": \"f32\", \"shape\": [4, 8], \"runtime_file\": \"fake_kernel.out.output.bin\", \"golden_file\": \"fake_kernel.out.golden.bin\" }\n"
             << "    ]\n"
             << "  },\n"
             << "  \"host_launch\": {\n"
             << "    \"mode\": \"helper\",\n"
             << "    \"helper_kind\": \"mix-tiling-helper\",\n"
             << "    \"helper_inputs\": {}\n"
             << "  }\n"
             << "}\n";
  }

  return root;
}

static std::filesystem::path makeRuntimeSessionVecArtifactRoot(
    const std::string &stem) {
  std::filesystem::path root = makeTempDir(stem);
  std::filesystem::create_directories(root / "out");

  std::ofstream manifest(root / "out" / "manifest.txt");
  if (!manifest) {
    llvm::errs() << "FAIL: cannot write runtime session manifest "
                 << (root / "out" / "manifest.txt").string() << "\n";
    ++g_fail;
    return {};
  }

  manifest << "kernel_name=fake_vec\n";
  manifest << "soc_version=Ascend910B1\n";
  manifest << "kernel_kind=vec\n";
  manifest << "device_binary_path=fake.bin\n";

  std::ofstream binary(root / "fake.bin", std::ios::binary);
  if (!binary) {
    llvm::errs() << "FAIL: cannot write runtime session binary "
                 << (root / "fake.bin").string() << "\n";
    ++g_fail;
    return {};
  }
  binary.put('\0');
  return root;
}

static std::filesystem::path makeRuntimeSessionArtifactRootWithoutKernelKind(
    const std::string &stem) {
  std::filesystem::path root = makeTempDir(stem);
  std::filesystem::create_directories(root / "out");

  std::ofstream manifest(root / "out" / "manifest.txt");
  if (!manifest) {
    llvm::errs() << "FAIL: cannot write runtime session manifest "
                 << (root / "out" / "manifest.txt").string() << "\n";
    ++g_fail;
    return {};
  }

  manifest << "kernel_name=fake_kernel\n";
  manifest << "soc_version=Ascend910B1\n";
  manifest << "mix_resource_type=mix_1c1v\n";
  manifest << "device_binary_path=fake.bin\n";

  std::ofstream binary(root / "fake.bin", std::ios::binary);
  if (!binary) {
    llvm::errs() << "FAIL: cannot write runtime session binary "
                 << (root / "fake.bin").string() << "\n";
    ++g_fail;
    return {};
  }
  binary.put('\0');
  return root;
}

struct RuntimeSessionTempRoot {
  explicit RuntimeSessionTempRoot(std::filesystem::path p)
      : path(std::move(p)) {}

  ~RuntimeSessionTempRoot() {
    if (path.empty())
      return;
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }

  std::filesystem::path path;
};

class RecordingBackendDriver : public ExecutionBackendDriver {
public:
  llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &request) override {
    ++invocations;
    lastRequest = request;
    ExecutionResult result;
    result.taskId = "driver:" + request.task.taskId;
    result.producedFiles.push_back(request.workingDirectory + "/done");
    return result;
  }

  int invocations = 0;
  ExecutionRequest lastRequest;
};

class ProfileArtifactBackendDriver : public ExecutionBackendDriver {
public:
  llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &request) override {
    ++invocations;
    lastRequest = request;
    ExecutionResult result;
    result.taskId = "driver:" + request.task.taskId;
    result.producedFiles.push_back(request.workingDirectory + "/done");
    result.producedFiles.push_back(
        request.workingDirectory + "/opprof/simulator/trace.json");
    ProfileTrace trace;
    trace.sessionId = "driver-session";
    addProfileArtifact(trace, "driver-task", ExecutionBackendKind::Simulation,
                       "/tmp/existing/profile.json");
    result.profileTrace = std::move(trace);
    return result;
  }

  int invocations = 0;
  ExecutionRequest lastRequest;
};

class SynthesizingProfileArtifactBackendDriver : public ExecutionBackendDriver {
public:
  llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &request) override {
    ++invocations;
    lastRequest = request;
    ExecutionResult result;
    result.taskId = "driver:" + request.task.taskId;
    result.producedFiles.push_back(
        request.workingDirectory + "/opprof/simulator/trace.json");
    return result;
  }

  int invocations = 0;
  ExecutionRequest lastRequest;
};

class OrderedExecutionBackendDriver : public ExecutionBackendDriver {
public:
  llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &request) override {
    seenTaskIds.push_back(request.task.taskId);
    seenSessionIds.push_back(request.sessionId);
    seenWorkingDirectories.push_back(request.workingDirectory);

    ExecutionResult result;
    result.taskId = request.task.taskId;
    result.producedFiles.push_back(request.workingDirectory + "/" +
                                   request.task.taskId + ".done");

    ProfileTrace trace;
    trace.sessionId = request.sessionId;
    addProfileArtifact(trace, request.task.taskId,
                       ExecutionBackendKind::Simulation,
                       request.workingDirectory + "/" + request.task.taskId +
                           ".profile.json");
    result.profileTrace = std::move(trace);
    return result;
  }

  std::vector<std::string> seenTaskIds;
  std::vector<std::string> seenSessionIds;
  std::vector<std::string> seenWorkingDirectories;
};

class FailingExecutionBackendDriver : public ExecutionBackendDriver {
public:
  explicit FailingExecutionBackendDriver(std::string message)
      : message(std::move(message)) {}

  llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &) override {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                   message.c_str());
  }

  std::string message;
};

class CapturingExecutionBackendDriver : public ExecutionBackendDriver {
public:
  llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &request) override {
    requests.push_back(request);
    ExecutionResult result;
    result.taskId = request.task.taskId;
    for (const TensorBinding &binding : request.task.invocation.outputs)
      result.producedFiles.push_back(binding.path);
    return result;
  }

  std::vector<ExecutionRequest> requests;
};

static RuntimeTask makeGateTask(const std::string &taskId, KernelKind kind,
                                MixResourceType mixType) {
  RuntimeTask task;
  task.taskId = taskId;
  task.artifact.kernelName = taskId + "_kernel";
  task.artifact.kernelKind = kind;
  task.artifact.mixResourceType = mixType;
  task.artifact.socVersion = "Ascend910B1";
  return task;
}

class SuccessfulNpuBackendDriver : public ExecutionBackendDriver {
public:
  llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &request) override {
    ++invocations;
    lastRequest = request;
    ExecutionResult result;
    result.taskId = request.task.taskId;
    for (const TensorBinding &binding : request.task.invocation.outputs)
      result.producedFiles.push_back(binding.path);
    ProfileTrace trace;
    trace.sessionId = request.sessionId;
    trace.addProfileArtifact(request.task.taskId, ExecutionBackendKind::Npu,
                             request.workingDirectory + "/" +
                                 request.task.taskId + ".npu-profile.json");
    result.profileTrace = std::move(trace);
    return result;
  }

  int invocations = 0;
  ExecutionRequest lastRequest;
};

#define EXPECT(cond, msg)                                                     \
  do {                                                                        \
    if (cond) {                                                               \
      ++g_pass;                                                               \
    } else {                                                                  \
      llvm::errs() << "FAIL: " << (msg) << "\n";                            \
      ++g_fail;                                                               \
    }                                                                         \
  } while (0)

static void testTaskGraphBasics() {
  TaskGraph graph;

  KernelArtifact artifact;
  artifact.kernelName = "mix_add";
  artifact.kernelKind = KernelKind::Mix;
  artifact.mixResourceType = MixResourceType::Mix1C1V;

  RuntimeTask taskA;
  taskA.taskId = "task_a";
  taskA.artifact = artifact;

  RuntimeTask taskB;
  taskB.taskId = "task_b";
  taskB.artifact = artifact;
  taskB.dependencies = {"task_a"};

  auto addA = graph.addTask(taskA);
  EXPECT(!addA, "add task_a");
  auto addB = graph.addTask(taskB);
  EXPECT(!addB, "add task_b");

  auto orderedOr = graph.orderedTasks();
  EXPECT((bool)orderedOr, "orderedTasks succeeds");
  if (orderedOr) {
    EXPECT(orderedOr->size() == 2, "orderedTasks size");
    EXPECT((*orderedOr)[0].taskId == "task_a", "orderedTasks first task");
    EXPECT((*orderedOr)[1].taskId == "task_b", "orderedTasks second task");
    EXPECT((*orderedOr)[0].artifact.kernelName == "mix_add",
           "orderedTasks preserves payload");
  }

  auto orderOr = graph.topologicalOrder();
  EXPECT((bool)orderOr, "topologicalOrder succeeds");
  if (orderOr) {
    EXPECT(orderOr->size() == 2, "topologicalOrder size");
    EXPECT((*orderOr)[0] == "task_a", "topologicalOrder first id");
    EXPECT((*orderOr)[1] == "task_b", "topologicalOrder second id");
  }

  ProfileTrace trace;
  trace.sessionId = "sess0";
  ProfileEvent event;
  event.taskId = "task_a";
  event.backend = ExecutionBackendKind::Simulation;
  event.eventKind = "kernel_complete";
  event.artifact = "trace.json";
  trace.events.push_back(event);
  EXPECT(trace.sessionId == "sess0", "profile trace session id");
  EXPECT(trace.events.size() == 1, "profile trace stores events");
  EXPECT(trace.events[0].taskId == "task_a", "profile event task id");
}

static void testProfileTraceCollectsArtifactPaths() {
  ProfileTrace trace;
  trace.sessionId = "sess1";
  trace.addEvent(ProfileEvent{"task_a", ExecutionBackendKind::Simulation,
                              "kernel_complete", "/tmp/not-a-profile"});
  trace.addProfileArtifact("task_a", ExecutionBackendKind::Simulation,
                           "/tmp/profile_a.json");
  trace.addProfileArtifact("task_b", ExecutionBackendKind::Simulation,
                           "/tmp/profile_b.json");

  const std::vector<std::string> artifacts = trace.profileArtifactPaths();
  EXPECT(artifacts.size() == 2,
         "profile trace collects only profile artifact events");
  if (artifacts.size() == 2) {
    EXPECT(artifacts[0] == "/tmp/profile_a.json",
           "profile trace preserves first artifact path");
    EXPECT(artifacts[1] == "/tmp/profile_b.json",
           "profile trace preserves second artifact path");
  }
}

static void testDuplicateTaskIds() {
  TaskGraph graph;
  RuntimeTask task;
  task.taskId = "dup";

  auto add1 = graph.addTask(task);
  EXPECT(!add1, "first duplicate-test insert succeeds");
  auto add2 = graph.addTask(task);
  EXPECT((bool)add2, "duplicate task id rejected");
  if (add2)
    llvm::consumeError(std::move(add2));
}

static void testEmptyTaskId() {
  TaskGraph graph;
  RuntimeTask task;
  task.taskId = "";

  auto add = graph.addTask(task);
  EXPECT((bool)add, "empty task id rejected");
  if (add)
    llvm::consumeError(std::move(add));
}

static void testUnknownDependency() {
  TaskGraph graph;
  RuntimeTask task;
  task.taskId = "task_b";
  task.dependencies = {"task_a"};

  auto add = graph.addTask(task);
  EXPECT(!add, "insert task with missing dependency");
  auto orderOr = graph.topologicalOrder();
  EXPECT(!(bool)orderOr, "unknown dependency rejected");
  if (!orderOr)
    llvm::consumeError(orderOr.takeError());
}

static void testCycleDetection() {
  TaskGraph graph;

  RuntimeTask taskA;
  taskA.taskId = "task_a";
  taskA.dependencies = {"task_b"};

  RuntimeTask taskB;
  taskB.taskId = "task_b";
  taskB.dependencies = {"task_a"};

  auto addA = graph.addTask(taskA);
  EXPECT(!addA, "insert cycle task_a");
  auto addB = graph.addTask(taskB);
  EXPECT(!addB, "insert cycle task_b");

  auto orderOr = graph.topologicalOrder();
  EXPECT(!(bool)orderOr, "cycle rejected");
  if (!orderOr)
    llvm::consumeError(orderOr.takeError());
}

static void testKernelArtifactNormalization() {
  EXPECT(inferMixResourceTypeFromKernelKind(KernelKind::Mix) ==
             MixResourceType::Mix1C1V,
         "mix kernels default to Mix1C1V");
  EXPECT(inferMixResourceTypeFromKernelKind(KernelKind::Vec) ==
             MixResourceType::Unknown,
         "non-mix kernels do not infer a mix resource type");

  MixArtifact mixArtifact;
  mixArtifact.kernel_name = "demo_kernel";
  mixArtifact.soc_version = "Ascend910B1";
  mixArtifact.work_dir = "/tmp/mix/work";
  mixArtifact.kernel_so_path = "/tmp/mix/libdemo_kernel_packed.so";
  mixArtifact.device_object_path = "/tmp/mix/device.o";
  mixArtifact.manifest_path = "/tmp/mix/mix-artifact.txt";
  mixArtifact.metadata_path = "/tmp/mix/out/mix_metadata.json";

  KernelArtifact normalizedMix = normalizeMixArtifact(
      mixArtifact, KernelKind::Mix, MixResourceType::Mix1C1V);
  EXPECT(normalizedMix.kernelName == "demo_kernel",
         "normalized mix artifact keeps kernel name");
  EXPECT(normalizedMix.kernelKind == KernelKind::Mix,
         "normalized mix artifact keeps kernel kind");
  EXPECT(normalizedMix.mixResourceType == MixResourceType::Mix1C1V,
         "normalized mix artifact keeps resource type");
  EXPECT(normalizedMix.deviceBinaryPath == "/tmp/mix/device.o",
         "normalized mix artifact stores device object path");
  EXPECT(normalizedMix.packedSharedObjectPath ==
             "/tmp/mix/libdemo_kernel_packed.so",
         "normalized mix artifact stores packed shared object path");
  EXPECT(normalizedMix.manifestPath == "/tmp/mix/mix-artifact.txt",
         "normalized mix artifact stores manifest path");
  EXPECT(normalizedMix.metadataPath == "/tmp/mix/out/mix_metadata.json",
         "normalized mix artifact stores metadata path");
  EXPECT(normalizedMix.artifactRoot == "/tmp/mix",
         "normalized mix artifact stores compile root, not work dir");
}

static void testVecCubeArtifactBackendCompilesVecArtifact() {
  const std::filesystem::path rootPath = makeTempDir("vec-cube-backend-vec");
  RuntimeSessionTempRoot cleanup(rootPath);
  EXPECT(!cleanup.path.empty(), "vec cube backend vec fixture root created");
  if (cleanup.path.empty())
    return;

  ArtifactCompileRequest req;
  req.kernelSource = "examples/relu-broadcast-transpose/step8_kernel.cpp";
  req.kernelName = "relu_transpose_broadcast_add";
  req.kernelKind = KernelKind::Vec;
  req.outputDir = cleanup.path.string();

  VecCubeArtifactBackend backend;
  auto artifactOr = backend.compile(req, "Ascend910B1");
  EXPECT((bool)artifactOr, "vec cube backend compiles vec request");
  if (!artifactOr) {
    llvm::consumeError(artifactOr.takeError());
    return;
  }

  const std::filesystem::path expectedManifest =
      cleanup.path / "out" / "manifest.txt";
  EXPECT(artifactOr->kernelKind == KernelKind::Vec,
         "vec cube backend preserves vec kernel kind");
  EXPECT(artifactOr->mixResourceType == MixResourceType::Unknown,
         "vec cube backend keeps vec mix resource unknown");
  EXPECT(artifactOr->manifestPath == expectedManifest.string(),
         "vec cube backend writes vec manifest under out/manifest.txt");
  EXPECT(std::filesystem::exists(artifactOr->manifestPath),
         "vec cube backend manifest file exists");
  EXPECT(std::filesystem::path(artifactOr->manifestPath).parent_path().filename() ==
             "out",
         "vec cube backend manifest path parent directory is out");
  const std::string manifestContents =
      readFileContents(artifactOr->manifestPath);
  EXPECT(manifestContents.find("kernel_kind=vec") != std::string::npos,
         "vec cube backend manifest records vec kernel kind");
}

static void testVecCubeArtifactBackendCompilesCubeArtifact() {
  const std::filesystem::path rootPath = makeTempDir("vec-cube-backend-cube");
  RuntimeSessionTempRoot cleanup(rootPath);
  EXPECT(!cleanup.path.empty(), "vec cube backend cube fixture root created");
  if (cleanup.path.empty())
    return;

  ArtifactCompileRequest req;
  req.kernelSource = "examples/matmul-add-leakyrelu/step8_kernel.cpp";
  req.kernelName = "matmul_add_leakyrelu";
  req.kernelKind = KernelKind::Cube;
  req.outputDir = cleanup.path.string();

  VecCubeArtifactBackend backend;
  auto artifactOr = backend.compile(req, "Ascend910B1");
  EXPECT((bool)artifactOr, "vec cube backend compiles cube request");
  if (!artifactOr) {
    llvm::consumeError(artifactOr.takeError());
    return;
  }

  const std::filesystem::path expectedManifest =
      cleanup.path / "out" / "manifest.txt";
  EXPECT(artifactOr->kernelKind == KernelKind::Cube,
         "vec cube backend preserves cube kernel kind");
  EXPECT(artifactOr->mixResourceType == MixResourceType::Unknown,
         "vec cube backend keeps cube mix resource unknown");
  EXPECT(artifactOr->manifestPath == expectedManifest.string(),
         "vec cube backend writes cube manifest under out/manifest.txt");
  EXPECT(std::filesystem::exists(artifactOr->manifestPath),
         "vec cube backend manifest file exists");
  EXPECT(std::filesystem::path(artifactOr->manifestPath).parent_path().filename() ==
             "out",
         "vec cube backend manifest path parent directory is out");
  const std::string manifestContents =
      readFileContents(artifactOr->manifestPath);
  EXPECT(manifestContents.find("kernel_kind=cube") != std::string::npos,
         "vec cube backend manifest records cube kernel kind");
}

static void testRuntimeSessionRequestBuilderLoadsMixArtifactFromRoot() {
  const std::filesystem::path rootPath =
      makeRuntimeSessionArtifactRoot("runtime-session-builder-artifact");
  RuntimeSessionTempRoot cleanup(rootPath);
  EXPECT(!cleanup.path.empty(), "runtime session builder fixture root created");
  if (cleanup.path.empty())
    return;

  auto artifactOr = loadRuntimeSessionArtifactFromRoot(cleanup.path.string());
  EXPECT((bool)artifactOr, "runtime session builder loads artifact root");
  if (!artifactOr) {
    llvm::consumeError(artifactOr.takeError());
    return;
  }

  EXPECT(artifactOr->kernelName == "fake_kernel",
         "runtime session builder loads kernel name from manifest root");
  EXPECT(artifactOr->kernelKind == KernelKind::Mix,
         "runtime session builder loads kernel kind from manifest root");
  EXPECT(artifactOr->mixResourceType == MixResourceType::Mix1C1V,
         "runtime session builder loads mix resource type from manifest root");
  EXPECT(artifactOr->artifactRoot == cleanup.path.string(),
         "runtime session builder keeps artifact root");
  EXPECT(artifactOr->manifestPath ==
             (cleanup.path / "out" / "manifest.txt").string(),
         "runtime session builder keeps manifest path");
}

static void testRuntimeSessionRequestBuilderLoadsMixArtifactMetadataPath() {
  const std::filesystem::path rootPath =
      makeRuntimeSessionArtifactRootWithMetadata(
          "runtime-session-builder-artifact-metadata", true);
  RuntimeSessionTempRoot cleanup(rootPath);
  EXPECT(!cleanup.path.empty(),
         "runtime session builder metadata fixture root created");
  if (cleanup.path.empty())
    return;

  auto artifactOr = loadRuntimeSessionArtifactFromRoot(cleanup.path.string());
  EXPECT((bool)artifactOr,
         "runtime session builder loads artifact root with metadata");
  if (!artifactOr) {
    llvm::consumeError(artifactOr.takeError());
    return;
  }

  EXPECT(artifactOr->metadataPath ==
             (cleanup.path / "out" / "mix_metadata.json").string(),
         "runtime session builder keeps metadata path");
}

static void testRuntimeSessionRequestBuilderRejectsMissingMixArtifactMetadata() {
  const std::filesystem::path rootPath =
      makeRuntimeSessionArtifactRootWithMetadata(
          "runtime-session-builder-artifact-metadata-missing", false);
  RuntimeSessionTempRoot cleanup(rootPath);
  EXPECT(!cleanup.path.empty(),
         "runtime session builder missing metadata fixture root created");
  if (cleanup.path.empty())
    return;

  auto artifactOr = loadRuntimeSessionArtifactFromRoot(cleanup.path.string());
  EXPECT(!(bool)artifactOr,
         "runtime session builder rejects advertised missing metadata path");
  if (!artifactOr)
    llvm::consumeError(artifactOr.takeError());
}

static void testPrepareRuntimeSessionGraphUsesMixMetadataDefaults() {
  const std::filesystem::path rootPath =
      makeRuntimeSessionMixArtifactRootWithAbiDefaults(
          "runtime-session-builder-mix-metadata-defaults", true);
  RuntimeSessionTempRoot cleanup(rootPath);
  EXPECT(!cleanup.path.empty(),
         "runtime session builder mix metadata-default fixture root created");
  if (cleanup.path.empty())
    return;

  const std::filesystem::path manifestPath =
      makeTempDir("runtime-session-builder-mix-metadata-run-manifest") /
      "run-manifest.json";
  std::filesystem::create_directories(manifestPath.parent_path());
  {
    std::ofstream os(manifestPath);
    os << "{\n"
       << "  \"task_id\": \"main\",\n"
       << "  \"backend\": \"sim\",\n"
       << "  \"artifact_root\": \"" << cleanup.path.string() << "\",\n"
       << "  \"inputs\": [\n"
       << "    { \"name\": \"lhs\", \"path\": \"/tmp/lhs.npy\" },\n"
       << "    { \"name\": \"rhs\", \"path\": \"/tmp/rhs.npy\" }\n"
       << "  ],\n"
       << "  \"outputs\": [\n"
       << "    { \"name\": \"out\", \"path\": \"/tmp/out.npy\" }\n"
       << "  ],\n"
       << "  \"expected_outputs\": [\n"
       << "    { \"name\": \"out\", \"path\": \"/tmp/out.golden.npy\" }\n"
       << "  ]\n"
       << "}\n";
  }

  auto graphOr = prepareRuntimeSessionGraphFromManifest(manifestPath.string());
  EXPECT((bool)graphOr,
         "runtime session builder applies mix metadata defaults");
  if (!graphOr) {
    llvm::consumeError(graphOr.takeError());
    return;
  }

  auto orderedOr = graphOr->second.orderedTasks();
  EXPECT((bool)orderedOr,
         "runtime session builder orders mix metadata-default graph");
  if (!orderedOr) {
    llvm::consumeError(orderedOr.takeError());
    return;
  }
  EXPECT(orderedOr->size() == 1,
         "runtime session builder metadata-default graph has one task");
  if (orderedOr->size() != 1)
    return;

  const RuntimeTask &task = orderedOr->front();
  EXPECT(task.invocation.blockDim == 8,
         "runtime session builder prefers metadata block dim");
  EXPECT(task.invocation.workspaceSize == 16777216,
         "runtime session builder prefers metadata workspace size");
  EXPECT(task.invocation.tiling.has_value(),
         "runtime session builder materializes metadata tiling binding");
  if (task.invocation.tiling) {
    EXPECT(task.invocation.tiling->binaryPath ==
               (cleanup.path / "out" / "tiling.bin").string(),
           "runtime session builder prefers metadata tiling path");
  }
  EXPECT(task.invocation.outputs.size() == 1,
         "runtime session builder keeps metadata-default output count");
  if (task.invocation.outputs.size() == 1) {
    EXPECT(task.invocation.outputs[0].shape.has_value(),
           "runtime session builder fills output shape from metadata");
    EXPECT(task.invocation.outputs[0].dtype.has_value(),
           "runtime session builder fills output dtype from metadata");
    if (task.invocation.outputs[0].shape) {
      EXPECT(task.invocation.outputs[0].shape->size() == 2 &&
                 (*task.invocation.outputs[0].shape)[0] == 4 &&
                 (*task.invocation.outputs[0].shape)[1] == 8,
             "runtime session builder prefers metadata output shape");
    }
    if (task.invocation.outputs[0].dtype) {
      EXPECT(*task.invocation.outputs[0].dtype == DType::F32,
             "runtime session builder prefers metadata output dtype");
    }
  }
  EXPECT(task.invocation.expectedOutputs.size() == 1,
         "runtime session builder keeps metadata-default expected output count");
  if (task.invocation.expectedOutputs.size() == 1) {
    EXPECT(task.invocation.expectedOutputs[0].shape.has_value(),
           "runtime session builder fills expected output shape from metadata");
    EXPECT(task.invocation.expectedOutputs[0].dtype.has_value(),
           "runtime session builder fills expected output dtype from metadata");
    if (task.invocation.expectedOutputs[0].dtype) {
      EXPECT(*task.invocation.expectedOutputs[0].dtype == DType::F32,
             "runtime session builder prefers metadata expected output dtype");
    }
  }
  EXPECT(task.invocation.inputs.size() == 2,
         "runtime session builder keeps metadata-default input count");
  if (task.invocation.inputs.size() == 2) {
    EXPECT(task.invocation.inputs[0].shape.has_value(),
           "runtime session builder fills input shape from metadata");
    EXPECT(task.invocation.inputs[0].dtype.has_value(),
           "runtime session builder fills input dtype from metadata");
    if (task.invocation.inputs[0].dtype) {
      EXPECT(*task.invocation.inputs[0].dtype == DType::F32,
             "runtime session builder prefers metadata input dtype");
    }
  }
}

static void testPrepareRuntimeSessionGraphFallsBackToManifestAbiDefaults() {
  const std::filesystem::path rootPath =
      makeRuntimeSessionMixArtifactRootWithAbiDefaults(
          "runtime-session-builder-mix-manifest-defaults", false);
  RuntimeSessionTempRoot cleanup(rootPath);
  EXPECT(!cleanup.path.empty(),
         "runtime session builder mix manifest-default fixture root created");
  if (cleanup.path.empty())
    return;

  const std::filesystem::path manifestPath =
      makeTempDir("runtime-session-builder-mix-manifest-run-manifest") /
      "run-manifest.json";
  std::filesystem::create_directories(manifestPath.parent_path());
  {
    std::ofstream os(manifestPath);
    os << "{\n"
       << "  \"task_id\": \"main\",\n"
       << "  \"backend\": \"sim\",\n"
       << "  \"artifact_root\": \"" << cleanup.path.string() << "\",\n"
       << "  \"inputs\": [\n"
       << "    { \"name\": \"lhs\", \"path\": \"/tmp/lhs.npy\" },\n"
       << "    { \"name\": \"rhs\", \"path\": \"/tmp/rhs.npy\" }\n"
       << "  ],\n"
       << "  \"outputs\": [\n"
       << "    { \"name\": \"out\", \"path\": \"/tmp/out.npy\" }\n"
       << "  ]\n"
       << "}\n";
  }

  auto graphOr = prepareRuntimeSessionGraphFromManifest(manifestPath.string());
  EXPECT((bool)graphOr,
         "runtime session builder falls back to manifest ABI defaults");
  if (!graphOr) {
    llvm::consumeError(graphOr.takeError());
    return;
  }

  auto orderedOr = graphOr->second.orderedTasks();
  EXPECT((bool)orderedOr,
         "runtime session builder orders mix manifest-default graph");
  if (!orderedOr) {
    llvm::consumeError(orderedOr.takeError());
    return;
  }
  EXPECT(orderedOr->size() == 1,
         "runtime session builder manifest-default graph has one task");
  if (orderedOr->size() != 1)
    return;

  const RuntimeTask &task = orderedOr->front();
  EXPECT(task.invocation.blockDim == 3,
         "runtime session builder falls back to manifest block dim");
  EXPECT(task.invocation.workspaceSize == 2048,
         "runtime session builder falls back to manifest workspace size");
  EXPECT(task.invocation.tiling.has_value(),
         "runtime session builder materializes manifest tiling binding");
  if (task.invocation.tiling) {
    EXPECT(task.invocation.tiling->binaryPath ==
               (cleanup.path / "out" / "legacy_tiling.bin").string(),
           "runtime session builder falls back to manifest tiling path");
  }
  EXPECT(task.invocation.outputs.size() == 1,
         "runtime session builder keeps manifest-default output count");
  if (task.invocation.outputs.size() == 1) {
    EXPECT(task.invocation.outputs[0].shape.has_value(),
           "runtime session builder fills output shape from manifest ABI");
    EXPECT(task.invocation.outputs[0].dtype.has_value(),
           "runtime session builder fills output dtype from manifest ABI");
    if (task.invocation.outputs[0].shape) {
      EXPECT(task.invocation.outputs[0].shape->size() == 2 &&
                 (*task.invocation.outputs[0].shape)[0] == 16 &&
                 (*task.invocation.outputs[0].shape)[1] == 16,
             "runtime session builder falls back to manifest output shape");
    }
    if (task.invocation.outputs[0].dtype) {
      EXPECT(*task.invocation.outputs[0].dtype == DType::F16,
             "runtime session builder falls back to manifest output dtype");
    }
  }
}

static void testRuntimeSessionRequestBuilderLoadsVecArtifactFromRoot() {
  const std::filesystem::path rootPath =
      makeRuntimeSessionVecArtifactRoot("runtime-session-builder-vec");
  RuntimeSessionTempRoot cleanup(rootPath);
  EXPECT(!cleanup.path.empty(), "runtime session builder vec fixture root created");
  if (cleanup.path.empty())
    return;

  auto artifactOr = loadRuntimeSessionArtifactFromRoot(cleanup.path.string());
  EXPECT((bool)artifactOr, "runtime session builder loads vec artifact root");
  if (!artifactOr) {
    llvm::consumeError(artifactOr.takeError());
    return;
  }

  EXPECT(artifactOr->kernelKind == KernelKind::Vec,
         "runtime session builder preserves vec kernel kind");
  EXPECT(artifactOr->mixResourceType == MixResourceType::Unknown,
         "runtime session builder keeps vec mix resource type unknown");
  EXPECT(artifactOr->artifactRoot == cleanup.path.string(),
         "runtime session builder keeps vec artifact root");
}

static void testRuntimeSessionRequestBuilderRejectsUnsupportedKernelKind() {
  RuntimeSessionArtifactRequest request;
  request.kernelSource = "/tmp/runtime-session-builder-invalid.cpp";
  request.kernelName = "invalid_kernel";
  request.outputDir = "/tmp/runtime-session-builder-output";
  request.socVersion = "Ascend910B1";
  request.kernelKind = static_cast<KernelKind>(123);

  auto artifactOr = prepareRuntimeSessionArtifact(request);
  EXPECT(!artifactOr,
         "runtime session builder rejects unsupported kernel kind values");
  if (!artifactOr) {
    std::string message = llvm::toString(artifactOr.takeError());
    EXPECT(message.find("unsupported kernel kind") != std::string::npos,
           "unsupported kernel kind failure reports a clear error");
  }
}

static void testRuntimeSessionRequestBuilderRejectsMissingKernelKind() {
  const std::filesystem::path rootPath =
      makeRuntimeSessionArtifactRootWithoutKernelKind(
          "runtime-session-builder-missing-kind");
  RuntimeSessionTempRoot cleanup(rootPath);
  EXPECT(!cleanup.path.empty(),
         "runtime session builder missing-kind fixture root created");
  if (cleanup.path.empty())
    return;

  auto artifactOr = loadRuntimeSessionArtifactFromRoot(cleanup.path.string());
  EXPECT(!artifactOr, "runtime session builder rejects missing kernel_kind");
  if (!artifactOr) {
    std::string message = llvm::toString(artifactOr.takeError());
    EXPECT(message.find("missing required field: kernel_kind") !=
               std::string::npos,
           "missing kernel_kind failure reports a clear error");
  }
}

static void testRuntimeSessionRequestBuilderBuildsSingleTaskGraph() {
  const std::filesystem::path rootPath =
      makeRuntimeSessionArtifactRoot("runtime-session-builder-graph");
  RuntimeSessionTempRoot cleanup(rootPath);
  EXPECT(!cleanup.path.empty(),
         "runtime session builder graph fixture root created");
  if (cleanup.path.empty())
    return;

  KernelArtifact artifact;
  artifact.kernelName = "fake_kernel";
  artifact.kernelKind = KernelKind::Mix;
  artifact.mixResourceType = MixResourceType::Mix1C1V;
  artifact.socVersion = "Ascend910B1";
  artifact.artifactRoot = cleanup.path.string();
  artifact.manifestPath = (cleanup.path / "out" / "manifest.txt").string();
  artifact.deviceBinaryPath = (cleanup.path / "fake.bin").string();

  auto graphOr = buildRuntimeSessionSingleTaskGraph(artifact, "main");
  EXPECT((bool)graphOr, "runtime session builder creates a task graph");
  if (!graphOr) {
    llvm::consumeError(graphOr.takeError());
    return;
  }

  auto orderedOr = graphOr->orderedTasks();
  EXPECT((bool)orderedOr, "runtime session builder orders single task graph");
  if (!orderedOr) {
    llvm::consumeError(orderedOr.takeError());
    return;
  }

  EXPECT(orderedOr->size() == 1,
         "runtime session builder creates a single task graph");
  if (orderedOr->size() == 1) {
    EXPECT(orderedOr->front().taskId == "main",
           "runtime session builder uses provided task id");
    EXPECT(orderedOr->front().artifact.mixResourceType ==
               MixResourceType::Mix1C1V,
           "runtime session builder preserves mix resource type");
  }
}

static void testMixValidationCanBeRepresentedAsRuntimeTask() {
  MixAbiMetadata abi;
  abi.logicalKernelName = "mix_add";
  abi.runtimeKernelName = "mix_add_runtime";
  abi.workspaceBytes = 4096;
  abi.blockDim = 8;
  abi.tilingMode = "generated_file";
  abi.tilingSource = "out/tiling.bin";
  abi.inputs = {
      {"lhs", buildCanonicalInputFileName(abi.runtimeKernelName, "lhs"), "",
       DType::F16, {16}},
      {"rhs", buildCanonicalInputFileName(abi.runtimeKernelName, "rhs"), "",
       DType::F16, {16}},
  };
  abi.outputs = {
      {"out", buildCanonicalOutputFileName(abi.runtimeKernelName, "out"),
       buildCanonicalGoldenFileName(abi.runtimeKernelName, "out"), DType::F16,
       {16}},
  };

  RunManifestSpec manifest;
  manifest.backendKind = ExecutionBackendKind::Simulation;

  RunTaskSpec task;
  task.taskId = "main";
  task.artifactRoot = "/tmp/mix-artifact";
  TensorBinding lhsBinding;
  lhsBinding.name = "lhs";
  lhsBinding.sourceKind = BindingSourceKind::ExternalFile;
  lhsBinding.path = "/tmp/input/" + abi.inputs[0].runtimeFile;
  task.invocation.inputs.push_back(lhsBinding);

  TensorBinding rhsBinding;
  rhsBinding.name = "rhs";
  rhsBinding.sourceKind = BindingSourceKind::ExternalFile;
  rhsBinding.path = "/tmp/input/" + abi.inputs[1].runtimeFile;
  task.invocation.inputs.push_back(rhsBinding);

  TensorBinding outputBinding;
  outputBinding.name = "out";
  outputBinding.sourceKind = BindingSourceKind::ExternalFile;
  outputBinding.path = "/tmp/mix-artifact/" + abi.outputs[0].runtimeFile;
  outputBinding.shape = abi.outputs[0].shape;
  outputBinding.dtype = abi.outputs[0].dtype;
  task.invocation.outputs.push_back(outputBinding);

  TilingBinding tilingBinding;
  tilingBinding.binaryPath = "/tmp/mix-artifact/out/tiling.bin";
  task.invocation.tiling = tilingBinding;
  task.invocation.blockDim = abi.blockDim;
  task.invocation.workspaceSize = abi.workspaceBytes;
  manifest.tasks.push_back(task);

  EXPECT(manifest.backendKind == ExecutionBackendKind::Simulation,
         "mix validation manifest uses simulation backend");
  EXPECT(manifest.tasks.size() == 1,
         "mix validation manifest carries one runtime task");
  if (manifest.tasks.size() != 1)
    return;

  const RunTaskSpec &taskSpec = manifest.tasks[0];
  EXPECT(taskSpec.artifactRoot == "/tmp/mix-artifact",
         "mix validation manifest preserves artifact root");
  EXPECT(taskSpec.invocation.inputs.size() == 2,
         "mix validation manifest preserves two inputs");
  EXPECT(taskSpec.invocation.inputs[0].path ==
             "/tmp/input/" + abi.inputs[0].runtimeFile,
         "mix validation manifest preserves first input path");
  EXPECT(taskSpec.invocation.outputs.size() == 1,
         "mix validation manifest preserves one output");
  EXPECT(taskSpec.invocation.outputs[0].path ==
             "/tmp/mix-artifact/" + abi.outputs[0].runtimeFile,
         "mix validation manifest preserves runtime output path");
  EXPECT(taskSpec.invocation.tiling.has_value(),
         "mix validation manifest carries tiling binding");
  if (taskSpec.invocation.tiling) {
    EXPECT(taskSpec.invocation.tiling->binaryPath ==
               "/tmp/mix-artifact/out/tiling.bin",
           "mix validation manifest preserves tiling bytes path");
  }

  KernelArtifact artifact;
  artifact.kernelName = abi.runtimeKernelName;
  artifact.kernelKind = KernelKind::Mix;
  artifact.mixResourceType = MixResourceType::Mix1C1V;
  artifact.socVersion = "Ascend910B1";
  artifact.artifactRoot = taskSpec.artifactRoot;
  artifact.manifestPath = "/tmp/mix-artifact/out/manifest.txt";
  artifact.packedSharedObjectPath = "/tmp/mix/libmix_add_runtime_packed.so";
  artifact.deviceBinaryPath = artifact.packedSharedObjectPath;

  RuntimeTask runtimeTask;
  runtimeTask.taskId = taskSpec.taskId;
  runtimeTask.artifact = artifact;
  runtimeTask.invocation = taskSpec.invocation;

  TaskGraph graph;
  auto addErr = graph.addTask(runtimeTask);
  EXPECT(!addErr, "mix validation runtime graph adds task");
  if (addErr) {
    llvm::consumeError(std::move(addErr));
    return;
  }

  auto orderedOr = graph.orderedTasks();
  EXPECT((bool)orderedOr, "mix validation runtime graph orders task");
  if (!orderedOr)
    return;
  EXPECT(orderedOr->size() == 1,
         "mix validation runtime graph remains a single runtime task");
  if (orderedOr->size() == 1) {
    EXPECT((*orderedOr)[0].artifact.kernelKind == KernelKind::Mix,
           "mix validation runtime task keeps mix kernel kind");
    EXPECT((*orderedOr)[0].artifact.packedSharedObjectPath ==
               "/tmp/mix/libmix_add_runtime_packed.so",
           "mix validation runtime task keeps packed shared object path");
    EXPECT((*orderedOr)[0].invocation.outputs[0].shape.has_value(),
           "mix validation runtime task carries output shape metadata");
    EXPECT((*orderedOr)[0].invocation.outputs[0].dtype.has_value(),
           "mix validation runtime task carries output dtype metadata");
  }

  auto driver = std::make_shared<CapturingExecutionBackendDriver>();
  CapturingExecutionBackendDriver *driverPtr = driver.get();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);
  auto traceOr = session.run(graph);
  EXPECT((bool)traceOr, "mix validation runtime session runs");
  if (!traceOr) {
    llvm::consumeError(traceOr.takeError());
    return;
  }

  EXPECT(driverPtr->requests.size() == 1,
         "mix validation runtime session sends one backend request");
  if (driverPtr->requests.size() != 1)
    return;

  const ExecutionRequest &request = driverPtr->requests[0];
  EXPECT(request.sessionId == traceOr->sessionId,
         "mix validation runtime session propagates session id");
  EXPECT(request.task.taskId == "main",
         "mix validation runtime session preserves task id");
  EXPECT(request.task.artifact.kernelKind == KernelKind::Mix,
         "mix validation runtime session preserves mix kernel kind");
  EXPECT(request.task.artifact.artifactRoot == "/tmp/mix-artifact",
         "mix validation runtime session preserves artifact root");
  EXPECT(request.task.artifact.manifestPath ==
             "/tmp/mix-artifact/out/manifest.txt",
         "mix validation runtime session preserves manifest path");
  EXPECT(request.task.artifact.packedSharedObjectPath ==
             "/tmp/mix/libmix_add_runtime_packed.so",
         "mix validation runtime session preserves packed shared object path");
  EXPECT(request.task.invocation.inputs.size() == 2,
         "mix validation runtime session preserves input count");
  EXPECT(request.task.invocation.outputs.size() == 1,
         "mix validation runtime session preserves output count");
  if (request.task.invocation.outputs.size() == 1) {
    EXPECT(request.task.invocation.outputs[0].path ==
               "/tmp/mix-artifact/" + abi.outputs[0].runtimeFile,
           "mix validation runtime session preserves output path");
    EXPECT(request.task.invocation.outputs[0].shape.has_value(),
           "mix validation runtime session preserves output shape metadata");
    EXPECT(request.task.invocation.outputs[0].dtype.has_value(),
           "mix validation runtime session preserves output dtype metadata");
  }
  EXPECT(request.task.invocation.tiling.has_value(),
         "mix validation runtime session carries tiling binding");
  if (request.task.invocation.tiling) {
    EXPECT(request.task.invocation.tiling->binaryPath ==
               "/tmp/mix-artifact/out/tiling.bin",
           "mix validation runtime session preserves tiling bytes path");
  }
  EXPECT(request.task.invocation.blockDim == abi.blockDim,
         "mix validation runtime session preserves block dim");
  EXPECT(request.task.invocation.workspaceSize == abi.workspaceBytes,
         "mix validation runtime session preserves workspace size");
}

static void testOutputComparatorExactMatchPasses() {
  NDArray actual = makeF32Array({2}, {1.0f, 2.0f});
  NDArray expected = makeF32Array({2}, {1.0f, 2.0f});
  std::vector<NDArray> actuals;
  actuals.push_back(std::move(actual));
  std::vector<NDArray> expecteds;
  expecteds.push_back(std::move(expected));

  auto resultOr = compareRuntimeOutputs(actuals, expecteds, 0.0, 0.0);
  EXPECT((bool)resultOr, "output comparator returns a result for exact match");
  if (!resultOr)
    return;

  EXPECT(resultOr->passed, "output comparator marks exact match as passing");
  EXPECT(resultOr->maxAbsDiff == 0.0,
         "output comparator reports zero max abs diff for exact match");
  EXPECT(resultOr->meanAbsDiff == 0.0,
         "output comparator reports zero mean abs diff for exact match");
  EXPECT(resultOr->errorMessage.empty(),
         "output comparator leaves error message empty for exact match");
}

static void testOutputComparatorMismatchReturnsDetailedFailure() {
  NDArray actual = makeF32Array({2}, {1.0f, 2.25f});
  NDArray expected = makeF32Array({2}, {1.0f, 2.0f});
  std::vector<NDArray> actuals;
  actuals.push_back(std::move(actual));
  std::vector<NDArray> expecteds;
  expecteds.push_back(std::move(expected));

  auto resultOr = compareRuntimeOutputs(actuals, expecteds, 1e-4, 1e-4);
  EXPECT((bool)resultOr,
         "output comparator returns a result for numeric mismatch");
  if (!resultOr)
    return;

  EXPECT(!resultOr->passed,
         "output comparator marks a clear mismatch as failing");
  EXPECT(resultOr->maxAbsDiff > 0.0,
         "output comparator reports non-zero max abs diff for mismatch");
  EXPECT(resultOr->meanAbsDiff > 0.0,
         "output comparator reports non-zero mean abs diff for mismatch");
  EXPECT(!resultOr->errorMessage.empty(),
         "output comparator returns a non-empty error message for mismatch");
}

static void testOutputComparatorStructuralMismatchReturnsError() {
  NDArray actualCountMismatch = makeF32Array({2}, {1.0f, 2.0f});
  std::vector<NDArray> countActuals;
  countActuals.push_back(std::move(actualCountMismatch));
  std::vector<NDArray> countExpecteds;
  auto countOr = compareRuntimeOutputs(countActuals, countExpecteds, 0.0, 0.0);
  EXPECT(!countOr,
         "output comparator rejects mismatched output counts with llvm::Error");
  if (!countOr) {
    std::string message = llvm::toString(countOr.takeError());
    EXPECT(!message.empty(),
           "output comparator count mismatch returns a diagnostic");
  }

  NDArray actualShapeMismatch = makeF32Array({2}, {1.0f, 2.0f});
  NDArray expectedShapeMismatch = makeF32Array({1, 2}, {1.0f, 2.0f});
  std::vector<NDArray> shapeActuals;
  shapeActuals.push_back(std::move(actualShapeMismatch));
  std::vector<NDArray> shapeExpecteds;
  shapeExpecteds.push_back(std::move(expectedShapeMismatch));
  auto shapeOr = compareRuntimeOutputs(shapeActuals, shapeExpecteds, 0.0, 0.0);
  EXPECT(!shapeOr,
         "output comparator rejects mismatched shapes with llvm::Error");
  if (!shapeOr) {
    std::string message = llvm::toString(shapeOr.takeError());
    EXPECT(!message.empty(),
           "output comparator shape mismatch returns a diagnostic");
  }

  NDArray actualDtypeMismatch = makeF32Array({2}, {1.0f, 2.0f});
  NDArray expectedDtypeMismatch = makeArray({2}, DType::INT32,
                                            {1, 0, 0, 0, 2, 0, 0, 0});
  std::vector<NDArray> dtypeActuals;
  dtypeActuals.push_back(std::move(actualDtypeMismatch));
  std::vector<NDArray> dtypeExpecteds;
  dtypeExpecteds.push_back(std::move(expectedDtypeMismatch));
  auto dtypeOr = compareRuntimeOutputs(dtypeActuals, dtypeExpecteds, 0.0, 0.0);
  EXPECT(!dtypeOr,
         "output comparator rejects mismatched dtypes with llvm::Error");
  if (!dtypeOr) {
    std::string message = llvm::toString(dtypeOr.takeError());
    EXPECT(!message.empty(),
           "output comparator dtype mismatch returns a diagnostic");
  }
}

static void testExecutionRunnerContractSupportsSimulationAndRealDeviceModes() {
  FakeExecutionRunner simRunner(ExecutionRunnerMode::Simulation);
  FakeExecutionRunner npuRunner(ExecutionRunnerMode::RealDevice);

  EXPECT(simRunner.mode() == ExecutionRunnerMode::Simulation,
         "execution runner contract preserves simulation mode");
  EXPECT(npuRunner.mode() == ExecutionRunnerMode::RealDevice,
         "execution runner contract preserves real-device mode");

  EXPECT(!simRunner.initialize(3),
         "execution runner fake initialize succeeds for simulation mode");
  EXPECT(simRunner.initializeCalled && simRunner.lastDeviceId == 3,
         "execution runner fake records initialize device id");
}

static void testExecutionRunnerContractSupportsFileLaunches() {
  FakeExecutionRunner runner(ExecutionRunnerMode::Simulation);
  RunArgs args;
  FileExecutionLaunch launch;
  launch.binaryPath = "/tmp/fake.bin";
  launch.kernelName = "fake_kernel";
  launch.magic = 0x41415246u;

  EXPECT(!runner.runFile(launch, args),
         "execution runner contract accepts vec or cube file launch");
  EXPECT(runner.runFileCalled,
         "execution runner fake records file launch invocation");
  EXPECT(runner.lastFileLaunch.binaryPath == "/tmp/fake.bin",
         "execution runner fake keeps file launch binary path");
  EXPECT(runner.lastFileLaunch.kernelName == "fake_kernel",
         "execution runner fake keeps file launch kernel name");
  EXPECT(runner.lastFileLaunch.magic == 0x41415246u,
         "execution runner fake keeps file launch magic");
}

static void testExecutionRunnerContractSupportsPackedMixLaunches() {
  FakeExecutionRunner runner(ExecutionRunnerMode::Simulation);
  RunArgs args;
  PackedMixExecutionLaunch launch;
  launch.sharedLibraryPath = "/tmp/libfake_packed.so";
  launch.kernelName = "fake_mix_kernel";

  EXPECT(!runner.runPackedMixFile(launch, args),
         "execution runner contract accepts packed mix launch");
  EXPECT(runner.runPackedMixCalled,
         "execution runner fake records packed mix launch invocation");
  EXPECT(runner.lastPackedMixLaunch.sharedLibraryPath ==
             "/tmp/libfake_packed.so",
         "execution runner fake keeps packed mix library path");
  EXPECT(runner.lastPackedMixLaunch.kernelName == "fake_mix_kernel",
         "execution runner fake keeps packed mix kernel name");
}

static void testNativeExecutionRunnerCompileCoverage() {
  NativeExecutionRunner runner(ExecutionRunnerMode::Simulation);

  EXPECT(runner.mode() == ExecutionRunnerMode::Simulation,
         "native execution runner preserves construction mode");
}

static void testSimulationBackendReportsMissingVecBinaryLaunchFailure() {
#ifdef _WIN32
  EXPECT(true, "missing vec binary launch-failure contract is covered on xvm");
#else
  std::filesystem::path workingDir =
      makeTempDir("taskgraph-runtime-sim-launch-failure");
  std::filesystem::create_directories(workingDir);
  std::filesystem::path outputPath = workingDir / "out.npy";
  std::filesystem::path errorPath = workingDir / "error.txt";
  std::filesystem::path missingBinaryPath =
      workingDir / "definitely-missing.vec.bin";

  pid_t pid = fork();
  EXPECT(pid >= 0, "fork for simulation launch-failure contract succeeds");
  if (pid < 0) {
    std::filesystem::remove_all(workingDir);
    return;
  }

  if (pid == 0) {
    auto simOr = createExecutionBackend(ExecutionBackendKind::Simulation);
    if (!simOr) {
      std::ofstream os(errorPath);
      os << llvm::toString(simOr.takeError());
      os.flush();
      std::_Exit(0);
    }

    ExecutionRequest request;
    request.workingDirectory = workingDir.string();
    request.task.taskId = "task_sim_vec_missing_binary";
    request.task.artifact.kernelName = "vec_kernel";
    request.task.artifact.kernelKind = KernelKind::Vec;
    request.task.artifact.deviceBinaryPath = missingBinaryPath.string();
    TensorBinding output;
    output.name = "out";
    output.sourceKind = BindingSourceKind::ExternalFile;
    output.path = outputPath.string();
    output.shape = std::vector<int64_t>{1};
    output.dtype = DType::F32;
    request.task.invocation.outputs.push_back(output);

    auto resultOr = (*simOr)->run(request);
    std::ofstream os(errorPath);
    if (resultOr) {
      os << "unexpected-success";
    } else {
      os << llvm::toString(resultOr.takeError());
    }
    os.flush();
    std::_Exit(0);
  }

  int status = 0;
  EXPECT(waitpid(pid, &status, 0) == pid,
         "waitpid for simulation launch-failure contract succeeds");
  EXPECT(WIFEXITED(status), "simulation launch-failure child exits cleanly");

  std::ifstream is(errorPath);
  std::string message((std::istreambuf_iterator<char>(is)),
                      std::istreambuf_iterator<char>());
  EXPECT(!message.empty(),
         "simulation launch-failure contract records an error message");
  EXPECT(message.find("[sim:kernel_launch]") != std::string::npos,
         "simulation backend preserves kernel_launch stage for missing vec binary");
  std::filesystem::remove_all(workingDir);
#endif
}

static void testArtifactCompilerRequestValidation() {
  ArtifactCompiler compiler;

  ArtifactCompileRequest missingSource;
  missingSource.kernelName = "demo_kernel";
  missingSource.outputDir = "/tmp/taskgraph-artifact-validation";
  auto srcErr = compiler.compile(missingSource);
  EXPECT(!(bool)srcErr, "missing source is rejected");
  if (!srcErr)
    llvm::consumeError(srcErr.takeError());

  ArtifactCompileRequest missingOutputDir;
  missingOutputDir.kernelSource = "/tmp/demo.cpp";
  missingOutputDir.kernelName = "demo_kernel";
  auto outErr = compiler.compile(missingOutputDir);
  EXPECT(!(bool)outErr, "missing output dir is rejected");
  if (!outErr)
    llvm::consumeError(outErr.takeError());

  ArtifactCompileRequest missingKernelName;
  missingKernelName.kernelSource = "/tmp/demo.cpp";
  missingKernelName.outputDir = "/tmp/taskgraph-artifact-validation";
  auto nameErr = compiler.compile(missingKernelName);
  EXPECT(!(bool)nameErr, "missing kernel name is rejected");
  if (!nameErr)
    llvm::consumeError(nameErr.takeError());
}

static void testFrontendCompileRequestBuilderPreservesFields() {
  FrontendCompileInput input;
  input.kernelSource = "/tmp/frontend-kernel.cpp";
  input.outputDir = "/tmp/frontend-out";
  input.kernelName = "frontend_kernel";
  input.socVersion = "Ascend910B1";
  input.arch = "dav-c220-vec";
  input.kernelKind = KernelKind::Vec;
  input.optLevel = 2;
  input.cannMlirPath = "/tmp/step7_cann.mlir";
  input.npyDir = "/tmp/npy-dir";

  auto requestOr = buildFrontendCompileRequest(input);
  EXPECT((bool)requestOr, "frontend compile request builds");
  if (!requestOr)
    return;

  EXPECT(requestOr->kernelSource == input.kernelSource,
         "frontend compile request preserves kernel source");
  EXPECT(requestOr->outputDir == input.outputDir,
         "frontend compile request preserves output dir");
  EXPECT(requestOr->kernelName == input.kernelName,
         "frontend compile request preserves kernel name");
  EXPECT(requestOr->socVersion == input.socVersion,
         "frontend compile request preserves soc version");
  EXPECT(requestOr->arch == input.arch,
         "frontend compile request preserves arch");
  EXPECT(requestOr->kernelKind == KernelKind::Vec,
         "frontend compile request preserves kernel kind");
  EXPECT(requestOr->optLevel == 2,
         "frontend compile request preserves opt level");
  EXPECT(requestOr->cannMlirPath && *requestOr->cannMlirPath == "/tmp/step7_cann.mlir",
         "frontend compile request preserves cann mlir path");
  EXPECT(requestOr->npyDir && *requestOr->npyDir == "/tmp/npy-dir",
         "frontend compile request preserves npy dir");
}

static void testFrontendSingleTaskRunPreparationAndSummary() {
  KernelArtifact artifact;
  artifact.kernelName = "frontend_vec";
  artifact.kernelKind = KernelKind::Vec;
  artifact.artifactRoot = "/tmp/frontend-artifact";
  artifact.deviceBinaryPath = "/tmp/frontend-artifact/frontend_vec.bin";

  ExecutionInvocation invocation;
  invocation.blockDim = 8;
  invocation.atol = 1.5;
  invocation.rtol = 0.05;
  invocation.outputs.push_back(TensorBinding{
      "out", BindingSourceKind::ExternalFile, "/tmp/frontend-out.npy", "", "",
      std::vector<int64_t>{4}, DType::F16});
  invocation.expectedOutputs.push_back(TensorBinding{
      "out", BindingSourceKind::ExternalFile, "/tmp/frontend-golden.npy", "",
      "", std::vector<int64_t>{4}, DType::F16});

  FrontendSingleTaskRunRequest request;
  request.backendKind = ExecutionBackendKind::Simulation;
  request.taskId = "main";
  request.artifact = artifact;
  request.invocation = invocation;

  auto preparedOr = prepareFrontendSingleTaskRun(request);
  EXPECT((bool)preparedOr, "frontend single-task run prepares");
  if (!preparedOr)
    return;

  EXPECT(preparedOr->backendKind == ExecutionBackendKind::Simulation,
         "frontend single-task run preserves backend");
  auto tasksOr = preparedOr->graph.orderedTasks();
  EXPECT((bool)tasksOr, "frontend single-task run graph orders tasks");
  if (!tasksOr || tasksOr->size() != 1)
    return;
  EXPECT((*tasksOr)[0].taskId == "main",
         "frontend single-task run preserves task id");
  EXPECT((*tasksOr)[0].artifact.kernelName == "frontend_vec",
         "frontend single-task run preserves artifact");
  EXPECT((*tasksOr)[0].invocation.outputs.size() == 1,
         "frontend single-task run preserves outputs");
  EXPECT((*tasksOr)[0].invocation.expectedOutputs.size() == 1,
         "frontend single-task run preserves expected outputs");

  ProfileTrace trace;
  trace.sessionId = "session-frontend";
  trace.addProfileArtifact("main", ExecutionBackendKind::Simulation,
                           "/tmp/frontend-profile.json");

  FrontendRunSummary successSummary = summarizeFrontendRunSuccess(
      ExecutionBackendKind::Simulation, /*validationRan=*/true, trace,
      "/tmp/session_summary.json");
  EXPECT(successSummary.success,
         "frontend run summary marks success");
  EXPECT(successSummary.validationStatus == FrontendValidationStatus::Passed,
         "frontend run summary marks validation pass");
  EXPECT(successSummary.profileTrace.profileArtifactPaths().size() == 1,
         "frontend run summary preserves profile artifacts");
  EXPECT(successSummary.profileArtifactPaths.size() == 1,
         "frontend run summary surfaces artifact path list");
  EXPECT(successSummary.profileArtifactPaths[0] == "/tmp/frontend-profile.json",
         "frontend run summary preserves surfaced artifact path");
  EXPECT(successSummary.retainedSummaryPath == "/tmp/session_summary.json",
         "frontend run summary preserves retained summary path");

  FrontendRunSummary errorSummary = summarizeFrontendRunError(
      ExecutionBackendKind::Simulation, /*validationRan=*/true,
      "[sim:validate] simulation output mismatch: expected 1 got 2");
  EXPECT(!errorSummary.success,
         "frontend run summary marks error");
  EXPECT(errorSummary.validationStatus == FrontendValidationStatus::Failed,
         "frontend run summary marks validation failure");
  EXPECT(errorSummary.rawErrorMessage ==
             "[sim:validate] simulation output mismatch: expected 1 got 2",
         "frontend run summary preserves raw error message");
  EXPECT(errorSummary.errorStage == "validate",
         "frontend run summary extracts error stage");
  EXPECT(errorSummary.errorMessage == "simulation output mismatch: expected 1 got 2",
         "frontend run summary strips stage prefix");

  FrontendRunSummary nonValidationError = summarizeFrontendRunError(
      ExecutionBackendKind::Simulation, /*validationRan=*/false,
      "[sim:kernel_launch] launch failed");
  EXPECT(nonValidationError.validationStatus ==
             FrontendValidationStatus::NotRun,
         "frontend run summary leaves validation unset when validation did not run");
}

static void testFrontendRunExecutionUsesNormalizedContract() {
  KernelArtifact artifact;
  artifact.kernelName = "frontend_vec";
  artifact.kernelKind = KernelKind::Vec;
  artifact.deviceBinaryPath = "/tmp/frontend.bin";

  TensorBinding output;
  output.name = "out";
  output.path = "/tmp/frontend-out.npy";

  ExecutionInvocation invocation;
  invocation.outputs.push_back(output);
  invocation.expectedOutputs.push_back(output);

  FrontendSingleTaskRunRequest request;
  request.backendKind = ExecutionBackendKind::Simulation;
  request.taskId = "main";
  request.artifact = artifact;
  request.invocation = invocation;

  auto preparedOr = prepareFrontendSingleTaskRun(request);
  EXPECT((bool)preparedOr, "frontend execute contract prepares graph");
  if (!preparedOr)
    return;

  auto successDriver = std::make_shared<OrderedExecutionBackendDriver>();
  ExecutionSession successSession(ExecutionBackendKind::Simulation, successDriver);
  FrontendRunSummary successSummary =
      executeFrontendPreparedRun(successSession, *preparedOr);
  EXPECT(successSummary.success,
         "frontend execute contract marks successful run");
  EXPECT(successSummary.validationStatus == FrontendValidationStatus::Passed,
         "frontend execute contract preserves validation pass");
  EXPECT(successSummary.profileArtifactPaths.size() == 1,
         "frontend execute contract surfaces trace artifact path");
  if (!successSummary.profileArtifactPaths.empty()) {
    EXPECT(successSummary.profileArtifactPaths[0].find(".profile.json") !=
               std::string::npos,
           "frontend execute contract forwards profile path from trace");
  }

  auto failingDriver = std::make_shared<FailingExecutionBackendDriver>(
      "[sim:validate] simulation output mismatch");
  ExecutionSession failingSession(ExecutionBackendKind::Simulation, failingDriver);
  FrontendRunSummary failureSummary =
      executeFrontendPreparedRun(failingSession, *preparedOr);
  EXPECT(!failureSummary.success,
         "frontend execute contract marks failed run");
  EXPECT(failureSummary.validationStatus == FrontendValidationStatus::Failed,
         "frontend execute contract marks validation failure");
  EXPECT(failureSummary.errorStage == "validate",
         "frontend execute contract extracts error stage");
  EXPECT(failureSummary.rawErrorMessage ==
             "[sim:validate] simulation output mismatch",
         "frontend execute contract preserves raw error");

  FrontendSingleTaskRunRequest noValidationRequest = request;
  noValidationRequest.invocation.expectedOutputs.clear();
  auto noValidationPreparedOr = prepareFrontendSingleTaskRun(noValidationRequest);
  EXPECT((bool)noValidationPreparedOr,
         "frontend execute contract prepares non-validating graph");
  if (!noValidationPreparedOr)
    return;
  ExecutionSession noValidationSession(ExecutionBackendKind::Simulation,
                                       successDriver);
  FrontendRunSummary noValidationSummary =
      executeFrontendPreparedRun(noValidationSession, *noValidationPreparedOr);
  EXPECT(noValidationSummary.validationStatus ==
             FrontendValidationStatus::NotRun,
         "frontend execute contract leaves validation unset when no golden outputs exist");
}

static std::string readFileContents(const std::string &path) {
  std::ifstream is(path);
  return std::string((std::istreambuf_iterator<char>(is)),
                     std::istreambuf_iterator<char>());
}

static void testBackendSelection() {
  auto driver = std::make_shared<RecordingBackendDriver>();
  auto simOr = createExecutionBackend(ExecutionBackendKind::Simulation, driver);
  auto npuOr = createExecutionBackend(ExecutionBackendKind::Npu, driver);
  EXPECT((bool)simOr, "simulation backend factory succeeds");
  EXPECT((bool)npuOr, "npu backend factory succeeds");
  if (simOr)
    EXPECT((*simOr)->kind() == ExecutionBackendKind::Simulation,
           "simulation backend reports its kind");
  if (npuOr)
    EXPECT((*npuOr)->kind() == ExecutionBackendKind::Npu,
           "npu backend reports its kind");

  ExecutionRequest request;
  request.task.taskId = "single";
  request.task.artifact.kernelName = "mix_add";
  request.task.artifact.kernelKind = KernelKind::Mix;
  request.task.artifact.mixResourceType = MixResourceType::Mix1C1V;
  request.workingDirectory = "/tmp/taskgraph-runtime";
  EXPECT(request.task.taskId == "single", "execution request stores task");
}

static void testDefaultBackendRequiresDriver() {
  auto simOr = createExecutionBackend(ExecutionBackendKind::Simulation);
  auto npuOr = createExecutionBackend(ExecutionBackendKind::Npu);
  EXPECT((bool)simOr, "simulation backend factory without driver succeeds");
  EXPECT((bool)npuOr, "npu backend factory without driver succeeds");
  if (!simOr || !npuOr)
    return;

  ExecutionRequest request;
  request.task.taskId = "task_a";

  auto simResult = (*simOr)->run(request);
  EXPECT(!(bool)simResult, "simulation backend without driver fails");
  if (!simResult)
    llvm::consumeError(simResult.takeError());

  auto npuResult = (*npuOr)->run(request);
  EXPECT(!(bool)npuResult, "npu backend without driver fails");
  if (!npuResult)
    llvm::consumeError(npuResult.takeError());
}

static void testInvalidBackendSelection() {
  auto bad = createExecutionBackend(static_cast<ExecutionBackendKind>(99));
  EXPECT(!(bool)bad, "invalid backend selection is rejected");
  if (!bad)
    llvm::consumeError(bad.takeError());
}

static void testBackendDelegatesToDriver() {
  auto driver = std::make_shared<RecordingBackendDriver>();
  auto simOr = createExecutionBackend(ExecutionBackendKind::Simulation, driver);
  auto npuOr = createExecutionBackend(ExecutionBackendKind::Npu, driver);
  EXPECT((bool)simOr, "simulation backend factory with driver succeeds");
  EXPECT((bool)npuOr, "npu backend factory with driver succeeds");
  if (!simOr || !npuOr)
    return;

  ExecutionRequest request;
  request.sessionId = "session-a";
  request.task.taskId = "task_a";
  request.workingDirectory = "/tmp/taskgraph-runtime";

  auto simResult = (*simOr)->run(request);
  EXPECT((bool)simResult, "simulation backend delegates");
  if (simResult) {
    EXPECT(simResult->taskId == "driver:task_a",
           "simulation backend returns driver result");
    EXPECT(simResult->producedFiles.size() == 1,
           "simulation backend preserves produced files");
    EXPECT(!(bool)simResult->profileTrace,
           "non-profile files do not surface a profile trace");
  }

  auto npuResult = (*npuOr)->run(request);
  EXPECT((bool)npuResult, "npu backend delegates");
  if (npuResult)
    EXPECT(npuResult->taskId == "driver:task_a",
           "npu backend returns driver result");

  EXPECT(driver->invocations == 2, "shared driver sees both invocations");
  EXPECT(driver->lastRequest.task.taskId == "task_a",
         "driver receives the original request");
}

static void testNpuBackendRejectsMissingDeviceBinaryPath() {
  auto npuOr = createExecutionBackend(ExecutionBackendKind::Npu);
  EXPECT((bool)npuOr, "npu backend factory without driver succeeds for validation");
  if (!npuOr)
    return;

  ExecutionRequest request;
  request.task.taskId = "task_npu_vec";
  request.task.artifact.kernelName = "vec_kernel";
  request.task.artifact.kernelKind = KernelKind::Vec;
  request.task.invocation.outputs.push_back(
      TensorBinding{"out", BindingSourceKind::ExternalFile, "/tmp/task_npu_vec.npy",
                    "", "", std::vector<int64_t>{4}, DType::F16});

  auto resultOr = (*npuOr)->run(request);
  EXPECT(!(bool)resultOr, "npu backend rejects missing device binary path");
  if (!resultOr) {
    const std::string message = llvm::toString(resultOr.takeError());
    EXPECT(message.find("[npu:artifact]") != std::string::npos,
           "npu backend reports artifact stage for missing device binary");
    EXPECT(message.find("artifact is missing device binary path") != std::string::npos,
           "npu backend reports missing device binary path");
  }
}

static void testNpuBackendRejectsMissingMixSharedObjectPath() {
  auto npuOr = createExecutionBackend(ExecutionBackendKind::Npu);
  EXPECT((bool)npuOr, "npu backend factory without driver succeeds for mix validation");
  if (!npuOr)
    return;

  ExecutionRequest request;
  request.task.taskId = "task_npu_mix";
  request.task.artifact.kernelName = "mix_kernel";
  request.task.artifact.kernelKind = KernelKind::Mix;
  request.task.invocation.outputs.push_back(
      TensorBinding{"out", BindingSourceKind::ExternalFile, "/tmp/task_npu_mix.npy",
                    "", "", std::vector<int64_t>{4}, DType::F16});

  auto resultOr = (*npuOr)->run(request);
  EXPECT(!(bool)resultOr, "npu backend rejects missing packed mix shared object");
  if (!resultOr) {
    const std::string message = llvm::toString(resultOr.takeError());
    EXPECT(message.find("[npu:artifact]") != std::string::npos,
           "npu backend reports artifact stage for missing mix shared object");
    EXPECT(message.find("mix artifact is missing packed shared object path") !=
               std::string::npos,
           "npu backend reports missing packed mix shared object path");
  }
}

static void testNpuBackendReachesRealDeviceModePath() {
  auto npuOr = createExecutionBackend(ExecutionBackendKind::Npu);
  EXPECT((bool)npuOr, "npu backend factory without driver succeeds for real-device path");
  if (!npuOr)
    return;

  const std::string inputPath =
      writeTempNpy("taskgraph-runtime-npu-input", {4}, DType::F16);
  if (inputPath.empty())
    return;

  ExecutionRequest request;
  request.task.taskId = "task_npu_real";
  request.task.artifact.kernelName = "vec_kernel";
  request.task.artifact.kernelKind = KernelKind::Vec;
  request.task.artifact.deviceBinaryPath = "/tmp/fake_npu_kernel.bin";
  request.task.invocation.inputs.push_back(
      TensorBinding{"in", BindingSourceKind::ExternalFile, inputPath});
  request.task.invocation.outputs.push_back(
      TensorBinding{"out", BindingSourceKind::ExternalFile, "/tmp/task_npu_real.npy",
                    "", "", std::vector<int64_t>{4}, DType::F16});

  auto resultOr = (*npuOr)->run(request);
  EXPECT(!(bool)resultOr, "npu backend without real device still fails explicitly");
  if (!resultOr) {
    const std::string message = llvm::toString(resultOr.takeError());
    EXPECT(message.find("[npu:executor_initialize]") != std::string::npos,
           "npu backend reports executor initialize stage");
  }
}

static void testExecutionSessionSupportsNpuSuccessDriver() {
  auto driver = std::make_shared<SuccessfulNpuBackendDriver>();
  SuccessfulNpuBackendDriver *driverPtr = driver.get();
  ExecutionSession session(ExecutionBackendKind::Npu, driver);

  RuntimeTask task;
  task.taskId = "npu_task";
  task.artifact.kernelName = "npu_kernel";
  task.artifact.kernelKind = KernelKind::Vec;
  task.artifact.deviceBinaryPath = "/tmp/fake_npu_kernel.bin";
  task.invocation.outputs.push_back(
      TensorBinding{"out", BindingSourceKind::ExternalFile, "/tmp/npu_task.npy",
                    "", "", std::vector<int64_t>{4}, DType::F16});

  TaskGraph graph;
  auto addErr = graph.addTask(task);
  EXPECT(!addErr, "npu success driver graph add task");
  if (addErr) {
    llvm::consumeError(std::move(addErr));
    return;
  }

  auto traceOr = session.run(graph);
  EXPECT((bool)traceOr, "execution session runs with npu success driver");
  if (traceOr) {
    EXPECT(traceOr->events.size() == 1,
           "npu success driver yields one profile event");
    if (!traceOr->events.empty()) {
      EXPECT(traceOr->events.front().backend == ExecutionBackendKind::Npu,
             "npu success driver profile event backend");
      EXPECT(traceOr->events.front().eventKind == "profile_artifact",
             "npu success driver profile event kind");
    }
  } else {
    llvm::consumeError(traceOr.takeError());
  }

  EXPECT(driverPtr->invocations == 1, "npu success driver invoked once");
  EXPECT(driverPtr->lastRequest.task.taskId == "npu_task",
         "npu success driver receives task");
}

static void testSimulatorProfileNormalization() {
  std::vector<std::string> producedFiles = {
      "/tmp/taskgraph-runtime/done",
      "/tmp/taskgraph-runtime/opprof/simulator/trace.json",
      "/tmp/taskgraph-runtime/opprof/simulator/notes.txt"};

  EXPECT(!isSimulatorProfileArtifact(producedFiles[0]),
         "regular output is not treated as a simulator profile artifact");
  EXPECT(isSimulatorProfileArtifact(producedFiles[1]),
         "simulator trace is recognized as a profile artifact");
  EXPECT(!isSimulatorProfileArtifact(producedFiles[2]),
         "non-trace simulator file is ignored");
  EXPECT(!isSimulatorProfileArtifact(
             "/tmp/taskgraph-runtime/opprof/simulator/foo-trace.json"),
         "near-match trace filename is ignored");
  EXPECT(!isSimulatorProfileArtifact(
             "/tmp/taskgraph-runtime/opprof/simulator/subdir/trace.json"),
         "trace in subdirectory is ignored");

  auto traceOr = normalizeSimulatorProfileTrace("sess0", "task0",
                                                producedFiles);
  EXPECT((bool)traceOr, "profile normalization finds simulator trace");
  if (traceOr) {
    EXPECT(traceOr->sessionId == "sess0",
           "normalized trace keeps session id");
    EXPECT(traceOr->events.size() == 1,
           "normalized trace filters to profile artifacts only");
    if (!traceOr->events.empty()) {
      EXPECT(traceOr->events.front().taskId == "task0",
             "normalized event keeps task mapping");
      EXPECT(traceOr->events.front().backend ==
                 ExecutionBackendKind::Simulation,
             "normalized event keeps backend kind");
      EXPECT(traceOr->events.front().artifact ==
                 "/tmp/taskgraph-runtime/opprof/simulator/trace.json",
             "normalized event stores profile artifact path");
    }
  }

  std::vector<std::string> nonProfileFiles = {
      "/tmp/taskgraph-runtime/done",
      "/tmp/taskgraph-runtime/log.txt"};
  auto emptyTraceOr = normalizeSimulatorProfileTrace("sess1", "task1",
                                                     nonProfileFiles);
  EXPECT(!(bool)emptyTraceOr,
         "normalization returns no trace when no profile artifacts exist");
}

static void testSimulatorProfileNormalizationExtractsMetrics() {
  const std::filesystem::path sourceRoot =
      makeTempDir("profile-normalize-metrics-src");
  std::filesystem::create_directories(sourceRoot / "opprof" / "simulator");

  const std::filesystem::path tracePath =
      sourceRoot / "opprof" / "simulator" / "trace.json";
  {
    std::ofstream os(tracePath);
    os << R"({
  "schema_version": 1,
  "backend": "simulation",
  "session_id": "sess-metrics",
  "task_id": "task-metrics",
  "score": 4242,
  "cycle_count": 4242
})";
  }

  std::vector<std::string> producedFiles = {tracePath.string()};
  auto traceOr = normalizeSimulatorProfileTrace("sess-metrics", "task-metrics",
                                                producedFiles);
  EXPECT((bool)traceOr,
         "profile normalization succeeds for schema v1 simulator trace");
  if (traceOr) {
    EXPECT(traceOr->events.size() == 1,
           "profile normalization emits one event for schema v1 trace");
    if (!traceOr->events.empty()) {
      EXPECT(traceOr->events.front().score &&
                 *traceOr->events.front().score == 4242,
             "normalized simulator profile extracts score");
      EXPECT(traceOr->events.front().cycleCount &&
                 *traceOr->events.front().cycleCount == 4242,
             "normalized simulator profile extracts cycle_count");
    }
  }

  std::error_code ec;
  std::filesystem::remove_all(sourceRoot, ec);
}

static void testAddProfileArtifactHelper() {
  ProfileTrace trace;
  trace.sessionId = "sess_helper";

  addProfileArtifact(trace, "task_helper", ExecutionBackendKind::Simulation,
                     "/tmp/taskgraph-runtime/opprof/simulator/helper.json");

  EXPECT(trace.events.size() == 1,
         "addProfileArtifact appends one profile event");
  if (!trace.events.empty()) {
    EXPECT(trace.events.front().taskId == "task_helper",
           "addProfileArtifact preserves task id");
    EXPECT(trace.events.front().backend == ExecutionBackendKind::Simulation,
           "addProfileArtifact preserves backend");
    EXPECT(trace.events.front().eventKind == "profile_artifact",
           "addProfileArtifact sets profile artifact event kind");
    EXPECT(trace.events.front().artifact ==
               "/tmp/taskgraph-runtime/opprof/simulator/helper.json",
           "addProfileArtifact preserves artifact path");
  }
}

static void testRetainProfileArtifactsForCli() {
  const std::filesystem::path sourceRoot = makeTempDir("profile-retain-src");
  const std::filesystem::path destRoot = makeTempDir("profile-retain-dst");
  std::filesystem::create_directories(sourceRoot / "opprof" / "simulator");
  std::filesystem::create_directories(destRoot);

  const std::filesystem::path tracePath =
      sourceRoot / "opprof" / "simulator" / "trace.json";
  {
    std::ofstream os(tracePath);
    os << "{\"score\": 123}";
  }

  ProfileTrace trace;
  trace.sessionId = "sess-retain";
  addProfileArtifact(trace, "task_main", ExecutionBackendKind::Simulation,
                     tracePath.string());

  auto retainedOr = retainProfileArtifactsForCli(trace, destRoot.string());
  EXPECT((bool)retainedOr,
         "retainProfileArtifactsForCli copies simulator profile artifacts");
  if (retainedOr) {
    const std::vector<std::string> artifacts =
        retainedOr->profileArtifactPaths();
    EXPECT(artifacts.size() == 1,
           "retained trace keeps exactly one profile artifact");
    if (!artifacts.empty()) {
      EXPECT(artifacts.front() != tracePath.string(),
             "retained trace rewrites artifact path away from session dir");
      EXPECT(std::filesystem::exists(artifacts.front()),
             "retained trace points to a copied profile artifact");
      EXPECT(readTextFile(artifacts.front()) == "{\"score\": 123}",
             "retained profile artifact preserves contents");
    }
  }

  std::error_code ec;
  std::filesystem::remove_all(sourceRoot, ec);
  std::filesystem::remove_all(destRoot, ec);
}

static std::filesystem::path
prepareProfileSummaryRetentionFixture(ProfileTrace &trace,
                                      const std::filesystem::path &sourceRoot,
                                      const std::filesystem::path &destRoot) {
  std::filesystem::create_directories(sourceRoot / "work");
  std::filesystem::create_directories(destRoot);

  const std::filesystem::path mainPath = sourceRoot / "work" / "main.json";
  const std::filesystem::path consumerPath =
      sourceRoot / "work" / "consumer.json";
  {
    std::ofstream os(mainPath);
    os << R"({"score":10,"cycle_count":10})";
  }
  {
    std::ofstream os(consumerPath);
    os << R"({"score":20,"cycle_count":20})";
  }

  trace.sessionId = "runtime-session--summary";
  addProfileArtifact(trace, "main", ExecutionBackendKind::Simulation,
                     mainPath.string());
  addProfileArtifact(trace, "consumer", ExecutionBackendKind::Simulation,
                     consumerPath.string());

  return destRoot / trace.sessionId;
}

struct ProfileSummaryRetentionFixture {
  std::filesystem::path sourceRoot;
  std::filesystem::path destRoot;
  std::filesystem::path retainedSessionDir;
  std::filesystem::path summaryPath;
  std::filesystem::path mainPath;
  std::filesystem::path consumerPath;
};

static ProfileSummaryRetentionFixture makeProfileSummaryRetentionFixture(
    ProfileTrace &trace, llvm::StringRef stem) {
  ProfileSummaryRetentionFixture fixture;
  fixture.sourceRoot = makeTempDir((stem + "-src").str());
  fixture.destRoot = makeTempDir((stem + "-dst").str());
  fixture.retainedSessionDir =
      prepareProfileSummaryRetentionFixture(trace, fixture.sourceRoot,
                                            fixture.destRoot);
  fixture.summaryPath = fixture.retainedSessionDir / "session_summary.json";
  fixture.mainPath = fixture.retainedSessionDir / "tasks" / "main.json";
  fixture.consumerPath = fixture.retainedSessionDir / "tasks" / "consumer.json";
  return fixture;
}

static void cleanupProfileSummaryRetentionFixture(
    const ProfileSummaryRetentionFixture &fixture) {
  std::error_code ec;
  std::filesystem::remove_all(fixture.sourceRoot, ec);
  std::filesystem::remove_all(fixture.destRoot, ec);
}

static void testRetainProfileArtifactsCreatesSessionSummary() {
  ProfileTrace trace;
  const ProfileSummaryRetentionFixture fixture =
      makeProfileSummaryRetentionFixture(trace, "profile-retain-summary");

  auto retainedOr = retainProfileArtifactsForCli(trace, fixture.destRoot.string());
  EXPECT((bool)retainedOr,
         "retainProfileArtifactsForCli retains summary session artifacts");
  if (retainedOr) {
    const std::vector<std::string> artifacts =
        retainedOr->profileArtifactPaths();

    EXPECT(std::filesystem::exists(fixture.summaryPath),
           "retained profiles include session_summary.json");
    EXPECT(std::filesystem::exists(fixture.mainPath),
           "retained profiles include tasks/main.json");
    EXPECT(std::filesystem::exists(fixture.consumerPath),
           "retained profiles include tasks/consumer.json");
    EXPECT(artifacts.size() == 2,
           "retained trace keeps both profile artifact paths");
    if (artifacts.size() == 2) {
      EXPECT(artifacts[0] == fixture.mainPath.string(),
             "retained trace points first artifact at tasks/main.json");
      EXPECT(artifacts[1] == fixture.consumerPath.string(),
             "retained trace points second artifact at tasks/consumer.json");
    }
  }

  cleanupProfileSummaryRetentionFixture(fixture);
}

static void testRetainedSessionSummaryContents() {
  ProfileTrace trace;
  const ProfileSummaryRetentionFixture fixture =
      makeProfileSummaryRetentionFixture(trace, "profile-retain-summary-json");

  auto retainedOr = retainProfileArtifactsForCli(trace, fixture.destRoot.string());
  EXPECT((bool)retainedOr,
         "retainProfileArtifactsForCli retains summary json artifacts");
  if (retainedOr) {
    EXPECT(std::filesystem::exists(fixture.summaryPath),
           "retained session summary json exists");
    if (!std::filesystem::exists(fixture.summaryPath)) {
      cleanupProfileSummaryRetentionFixture(fixture);
      return;
    }

    const std::string summaryText = readTextFile(fixture.summaryPath.string());
    auto jsonOr = llvm::json::parse(summaryText);
    EXPECT((bool)jsonOr, "retained session summary parses as json");
    if (!jsonOr) {
      cleanupProfileSummaryRetentionFixture(fixture);
      llvm::consumeError(jsonOr.takeError());
      return;
    }

    const auto *object = jsonOr->getAsObject();
    EXPECT(object != nullptr, "retained session summary is a json object");
    if (object) {
      auto schemaVersion = object->getInteger("schema_version");
      auto sessionId = object->getString("session_id");
      auto backend = object->getString("backend");
      auto taskCount = object->getInteger("task_count");
      auto successfulTaskCount = object->getInteger("successful_task_count");
      auto failedTaskCount = object->getInteger("failed_task_count");
      auto totalScore = object->getInteger("total_score");
      auto totalCycleCount = object->getInteger("total_cycle_count");
      EXPECT(schemaVersion && *schemaVersion == 1,
             "retained session summary has schema_version=1");
      EXPECT(sessionId && *sessionId == "runtime-session--summary",
             "retained session summary preserves session_id");
      EXPECT(backend && *backend == "simulation",
             "retained session summary preserves backend");
      EXPECT(taskCount && *taskCount == 2,
             "retained session summary counts two tasks");
      EXPECT(successfulTaskCount && *successfulTaskCount == 2,
             "retained session summary counts successful tasks");
      EXPECT(failedTaskCount && *failedTaskCount == 0,
             "retained session summary counts failed tasks");
      EXPECT(totalScore && *totalScore == 30,
             "retained session summary totals score");
      EXPECT(totalCycleCount && *totalCycleCount == 30,
             "retained session summary totals cycle count");

      const auto *tasks = object->getArray("tasks");
      EXPECT(tasks && tasks->size() == 2,
             "retained session summary emits two task entries");
      if (tasks && tasks->size() == 2) {
        const auto *task0 = (*tasks)[0].getAsObject();
        auto profilePath0 =
            task0 ? task0->getString("profile_path") : decltype(task0->getString("profile_path")){};
        EXPECT(profilePath0 && *profilePath0 == fixture.mainPath.string(),
               "retained session summary task0 points to tasks/main.json");
        const auto *task1 = (*tasks)[1].getAsObject();
        auto profilePath1 =
            task1 ? task1->getString("profile_path") : decltype(task1->getString("profile_path")){};
        EXPECT(profilePath1 && *profilePath1 == fixture.consumerPath.string(),
               "retained session summary task1 points to tasks/consumer.json");
      }
    }
  }

  cleanupProfileSummaryRetentionFixture(fixture);
}

static void testRetainedSessionSummaryFallsBackBetweenScoreAndCycleCount() {
  ProfileTrace trace;
  const std::filesystem::path sourceRoot =
      makeTempDir("profile-retain-summary-fallback-src");
  const std::filesystem::path destRoot =
      makeTempDir("profile-retain-summary-fallback-dst");
  std::filesystem::create_directories(sourceRoot / "work");
  std::filesystem::create_directories(destRoot);

  const std::filesystem::path mainPath = sourceRoot / "work" / "main.json";
  const std::filesystem::path consumerPath =
      sourceRoot / "work" / "consumer.json";
  {
    std::ofstream os(mainPath);
    os << R"({"score":11})";
  }
  {
    std::ofstream os(consumerPath);
    os << R"({"cycle_count":19})";
  }

  trace.sessionId = "runtime-session--summary-fallback";
  addProfileArtifact(trace, "main", ExecutionBackendKind::Simulation,
                     mainPath.string());
  addProfileArtifact(trace, "consumer", ExecutionBackendKind::Simulation,
                     consumerPath.string());

  const std::filesystem::path summaryPath =
      destRoot / trace.sessionId / "session_summary.json";
  auto retainedOr = retainProfileArtifactsForCli(trace, destRoot.string());
  EXPECT((bool)retainedOr,
         "retainProfileArtifactsForCli falls back between score and cycle_count");
  if (retainedOr) {
    auto parsed = llvm::json::parse(readTextFile(summaryPath.string()));
    EXPECT((bool)parsed,
           "retained session summary fallback fixture parses as json");
    if (parsed) {
      const auto *object = parsed->getAsObject();
      EXPECT(object != nullptr,
             "retained session summary fallback fixture is a json object");
      if (object) {
        auto totalScore = object->getInteger("total_score");
        auto totalCycleCount = object->getInteger("total_cycle_count");
        EXPECT(totalScore && *totalScore == 30,
               "retained session summary falls back missing score values");
        EXPECT(totalCycleCount && *totalCycleCount == 30,
               "retained session summary falls back missing cycle_count values");
      }
    } else {
      llvm::consumeError(parsed.takeError());
    }
  }

  std::error_code ec;
  std::filesystem::remove_all(sourceRoot, ec);
  std::filesystem::remove_all(destRoot, ec);
}

static void testRetainProfileArtifactsUsesEventMetricsWithoutParsingJson() {
  const std::filesystem::path sourceRoot =
      makeTempDir("profile-retain-summary-event-metrics-src");
  const std::filesystem::path destRoot =
      makeTempDir("profile-retain-summary-event-metrics-dst");
  std::filesystem::create_directories(sourceRoot / "work");
  std::filesystem::create_directories(destRoot);

  const std::filesystem::path mainPath = sourceRoot / "work" / "main.json";
  {
    std::ofstream os(mainPath);
    os << "this is not valid json";
  }

  ProfileTrace trace;
  trace.sessionId = "runtime-session--summary-event-metrics";
  trace.addEvent(ProfileEvent{
      "main",
      ExecutionBackendKind::Simulation,
      "profile_artifact",
      mainPath.string(),
      77,
      77,
  });

  const std::filesystem::path summaryPath =
      destRoot / trace.sessionId / "session_summary.json";
  auto retainedOr = retainProfileArtifactsForCli(trace, destRoot.string());
  EXPECT((bool)retainedOr,
         "retainProfileArtifactsForCli uses in-memory event metrics without reparsing task json");
  if (retainedOr) {
    auto parsed = llvm::json::parse(readTextFile(summaryPath.string()));
    EXPECT((bool)parsed,
           "retained session summary from event metrics parses as json");
    if (parsed) {
      const auto *object = parsed->getAsObject();
      EXPECT(object != nullptr,
             "retained session summary from event metrics is a json object");
      if (object) {
        auto totalScore = object->getInteger("total_score");
        auto totalCycleCount = object->getInteger("total_cycle_count");
        EXPECT(totalScore && *totalScore == 77,
               "retained session summary uses score from event metrics");
        EXPECT(totalCycleCount && *totalCycleCount == 77,
               "retained session summary uses cycle_count from event metrics");
      }
    } else {
      llvm::consumeError(parsed.takeError());
    }
  }

  std::error_code ec;
  std::filesystem::remove_all(sourceRoot, ec);
  std::filesystem::remove_all(destRoot, ec);
}

static void testRetainProfileArtifactsFailsOnDuplicateTaskIds() {
  ProfileTrace trace;
  const std::filesystem::path sourceRoot =
      makeTempDir("profile-retain-summary-duplicate-src");
  const std::filesystem::path destRoot =
      makeTempDir("profile-retain-summary-duplicate-dst");
  std::filesystem::create_directories(sourceRoot / "work");
  std::filesystem::create_directories(destRoot);

  const std::filesystem::path firstPath = sourceRoot / "work" / "first.json";
  const std::filesystem::path secondPath = sourceRoot / "work" / "second.json";
  {
    std::ofstream os(firstPath);
    os << R"({"score":1,"cycle_count":1})";
  }
  {
    std::ofstream os(secondPath);
    os << R"({"score":2,"cycle_count":2})";
  }

  trace.sessionId = "runtime-session--summary-duplicate";
  addProfileArtifact(trace, "main", ExecutionBackendKind::Simulation,
                     firstPath.string());
  addProfileArtifact(trace, "main", ExecutionBackendKind::Simulation,
                     secondPath.string());

  auto retainedOr = retainProfileArtifactsForCli(trace, destRoot.string());
  EXPECT(!retainedOr,
         "retainProfileArtifactsForCli fails on duplicate retained task ids");
  if (!retainedOr) {
    std::string message = llvm::toString(retainedOr.takeError());
    EXPECT(message.find("duplicate retained profile task id") !=
               std::string::npos,
           "duplicate retained task failure reports a clear error");
  }

  std::error_code ec;
  std::filesystem::remove_all(sourceRoot, ec);
  std::filesystem::remove_all(destRoot, ec);
}

static void testRetainProfileArtifactsPrunesOldSessions() {
  const std::filesystem::path retainRoot = makeTempDir("profile-retain-root");
  std::filesystem::create_directories(retainRoot);

  for (int i = 0; i < 3; ++i) {
    const std::filesystem::path dir =
        retainRoot / ("runtime-session--old" + std::to_string(i));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "trace.json") << i;
    std::filesystem::last_write_time(
        dir, std::filesystem::file_time_type::clock::now() +
                 std::chrono::seconds(i));
  }

  auto err = pruneRetainedProfileDirectoriesForTest(retainRoot.string(), 2);
  EXPECT(!err, "retention pruning succeeds");
  if (err)
    llvm::consumeError(std::move(err));

  EXPECT(std::filesystem::exists(retainRoot / "runtime-session--old1"),
         "retention keeps second-newest directory");
  EXPECT(std::filesystem::exists(retainRoot / "runtime-session--old2"),
         "retention keeps newest directory");
  EXPECT(!std::filesystem::exists(retainRoot / "runtime-session--old0"),
         "retention prunes oldest directory");

  std::error_code ec;
  std::filesystem::remove_all(retainRoot, ec);
}

static void testRetainProfileArtifactsIgnoresNonDirectories() {
  const std::filesystem::path retainRoot = makeTempDir("profile-retain-files");
  std::filesystem::create_directories(retainRoot);
  std::ofstream(retainRoot / "README.txt") << "keep me";
  std::filesystem::create_directories(retainRoot / "runtime-session--keep");

  auto err = pruneRetainedProfileDirectoriesForTest(retainRoot.string(), 1);
  EXPECT(!err, "retention pruning ignores non-directories");
  if (err)
    llvm::consumeError(std::move(err));

  EXPECT(std::filesystem::exists(retainRoot / "README.txt"),
         "retention does not touch non-directory files");
  EXPECT(std::filesystem::exists(retainRoot / "runtime-session--keep"),
         "retention keeps the single retained session directory");

  std::error_code ec;
  std::filesystem::remove_all(retainRoot, ec);
}

static void testRetainedProfilePruneKeepCountForNewSession() {
  EXPECT(retainedProfilePruneKeepCountForNewSession(0) == 0,
         "pre-run retained profile pruning keeps zero directories when limit is zero");
  EXPECT(retainedProfilePruneKeepCountForNewSession(1) == 0,
         "pre-run retained profile pruning reserves one slot for the new session");
  EXPECT(retainedProfilePruneKeepCountForNewSession(20) == 19,
         "pre-run retained profile pruning keeps limit minus one existing sessions");
}

static void testPrepareRetainedProfileRunRootForCliPrunesBeforeNewSession() {
  const std::filesystem::path retainRoot =
      makeTempDir("profile-retain-cli-prepare");
  std::filesystem::create_directories(retainRoot);

  for (int i = 0; i < 3; ++i) {
    const std::filesystem::path dir =
        retainRoot / ("runtime-session--old" + std::to_string(i));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "trace.json") << i;
    std::filesystem::last_write_time(
        dir, std::filesystem::file_time_type::clock::now() +
                 std::chrono::seconds(i));
  }

  auto preparedOr = prepareRetainedProfileRunRootForCli(retainRoot.string(), 2);
  EXPECT((bool)preparedOr,
         "prepareRetainedProfileRunRootForCli succeeds");
  if (preparedOr) {
    EXPECT(*preparedOr == retainRoot.string(),
           "prepareRetainedProfileRunRootForCli keeps the original retain root");
    EXPECT(std::filesystem::exists(retainRoot / "runtime-session--old2"),
           "prepareRetainedProfileRunRootForCli keeps the newest existing session");
    EXPECT(!std::filesystem::exists(retainRoot / "runtime-session--old1"),
           "prepareRetainedProfileRunRootForCli prunes sessions beyond the reserved slot");
    EXPECT(!std::filesystem::exists(retainRoot / "runtime-session--old0"),
           "prepareRetainedProfileRunRootForCli prunes the oldest session");
  }

  std::error_code ec;
  std::filesystem::remove_all(retainRoot, ec);
}

static void testRuntimeSessionWorkdirPruneKeepCountForNewRun() {
  EXPECT(runtimeSessionWorkdirPruneKeepCountForNewRun(0) == 0,
         "pre-run runtime session pruning keeps zero directories when limit is zero");
  EXPECT(runtimeSessionWorkdirPruneKeepCountForNewRun(1) == 0,
         "pre-run runtime session pruning reserves one slot for the new session");
  EXPECT(runtimeSessionWorkdirPruneKeepCountForNewRun(20) == 19,
         "pre-run runtime session pruning keeps limit minus one existing sessions");
}

static void testPrepareRuntimeSessionWorkdirRootForCliPrunesBeforeNewRun() {
  const std::filesystem::path retainRoot =
      makeTempDir("runtime-session-workdir-prepare");
  std::filesystem::create_directories(retainRoot);

  for (int i = 0; i < 3; ++i) {
    const std::filesystem::path dir =
        retainRoot / ("runtime-session--old" + std::to_string(i));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "trace.txt") << i;
    std::filesystem::last_write_time(
        dir, std::filesystem::file_time_type::clock::now() +
                 std::chrono::seconds(i));
  }

  auto preparedOr =
      prepareRuntimeSessionWorkdirRootForCli(retainRoot.string(), 2);
  EXPECT((bool)preparedOr,
         "prepareRuntimeSessionWorkdirRootForCli succeeds");
  if (preparedOr) {
    EXPECT(*preparedOr == retainRoot.string(),
           "prepareRuntimeSessionWorkdirRootForCli keeps the original root");
    EXPECT(std::filesystem::exists(retainRoot / "runtime-session--old2"),
           "prepareRuntimeSessionWorkdirRootForCli keeps the newest existing session");
    EXPECT(!std::filesystem::exists(retainRoot / "runtime-session--old1"),
           "prepareRuntimeSessionWorkdirRootForCli prunes sessions beyond the reserved slot");
    EXPECT(!std::filesystem::exists(retainRoot / "runtime-session--old0"),
           "prepareRuntimeSessionWorkdirRootForCli prunes the oldest session");
  }

  std::error_code ec;
  std::filesystem::remove_all(retainRoot, ec);
}

static void testRetainProfileArtifactsForCliRunReturnsSummaryPath() {
  ProfileTrace trace;
  const ProfileSummaryRetentionFixture fixture =
      makeProfileSummaryRetentionFixture(trace, "profile-retain-cli-run");

  auto retainedOr = retainProfileArtifactsForCliRun(trace, fixture.destRoot.string());
  EXPECT((bool)retainedOr,
         "retainProfileArtifactsForCliRun succeeds");
  if (retainedOr) {
    EXPECT(retainedOr->summaryPath == fixture.summaryPath.string(),
           "retainProfileArtifactsForCliRun returns retained session summary path");
    EXPECT(std::filesystem::exists(retainedOr->summaryPath),
           "retainProfileArtifactsForCliRun materializes summary path");
    EXPECT(retainedOr->trace.profileArtifactPaths().size() == 2,
           "retainProfileArtifactsForCliRun returns retained trace");
  }

  cleanupProfileSummaryRetentionFixture(fixture);
}

static void testBackendSurfacesProfileTrace() {
  auto driver = std::make_shared<SynthesizingProfileArtifactBackendDriver>();
  auto simOr = createExecutionBackend(ExecutionBackendKind::Simulation, driver);
  EXPECT((bool)simOr, "simulation backend factory with profile driver succeeds");
  if (!simOr)
    return;

  ExecutionRequest request;
  request.sessionId = "session-profile";
  request.task.taskId = "task_profile";
  request.workingDirectory = "/tmp/taskgraph-runtime";

  auto simResult = (*simOr)->run(request);
  EXPECT((bool)simResult, "simulation backend run succeeds");
  if (simResult) {
    EXPECT(simResult->producedFiles.size() == 1,
           "driver produced files are preserved");
    EXPECT((bool)simResult->profileTrace,
           "profile trace is synthesized");
    if (simResult->profileTrace) {
      EXPECT(simResult->profileTrace->sessionId == "session-profile",
             "synthesized trace uses explicit session id");
      EXPECT(simResult->profileTrace->events.size() == 1,
             "synthesized trace contains one event");
      if (!simResult->profileTrace->events.empty()) {
        EXPECT(simResult->profileTrace->events.front().taskId ==
                   "task_profile",
               "synthesized trace event task id is correct");
        EXPECT(simResult->profileTrace->events.front().backend ==
                   ExecutionBackendKind::Simulation,
               "synthesized trace event backend is correct");
        EXPECT(simResult->profileTrace->events.front().eventKind ==
                   "profile_artifact",
               "synthesized trace event kind is correct");
        EXPECT(simResult->profileTrace->events.front().artifact ==
                   "/tmp/taskgraph-runtime/opprof/simulator/trace.json",
               "synthesized trace artifact path is correct");
      }
    }
  }
}

static void testBackendPreservesExistingProfileTrace() {
  auto driver = std::make_shared<ProfileArtifactBackendDriver>();
  auto simOr = createExecutionBackend(ExecutionBackendKind::Simulation, driver);
  EXPECT((bool)simOr, "simulation backend factory with preserving driver succeeds");
  if (!simOr)
    return;

  ExecutionRequest request;
  request.sessionId = "session-existing";
  request.task.taskId = "task_profile";
  request.workingDirectory = "/tmp/taskgraph-runtime";

  auto simResult = (*simOr)->run(request);
  EXPECT((bool)simResult, "simulation backend run with existing trace succeeds");
  if (simResult) {
    EXPECT((bool)simResult->profileTrace,
           "existing trace is preserved");
    if (simResult->profileTrace) {
      EXPECT(simResult->profileTrace->sessionId == "driver-session",
             "existing trace session id is preserved");
      EXPECT(simResult->profileTrace->events.size() == 1,
             "existing trace event count is preserved");
      if (!simResult->profileTrace->events.empty()) {
        EXPECT(simResult->profileTrace->events.front().taskId == "driver-task",
               "existing trace task id is preserved");
        EXPECT(simResult->profileTrace->events.front().artifact ==
                   "/tmp/existing/profile.json",
               "existing trace artifact is preserved");
      }
    }
  }
}

static void testSimulatorProfileSchemaV1Artifact() {
  const std::filesystem::path runtimeDir = makeTempDir("taskgraph-profile-run");
  std::filesystem::create_directories(runtimeDir);

  const std::string inputPath =
      writeTempNpy("taskgraph-profile-input", {4, 2}, DType::F16);
  if (inputPath.empty())
    return;
  const std::string expectedOutputPath =
      writeTempNpy("taskgraph-profile-expected", {2, 4}, DType::F32);
  if (expectedOutputPath.empty())
    return;

  ExecutionRequest request;
  request.sessionId = "session-profile";
  request.workingDirectory = runtimeDir.string();
  request.task.taskId = "task0";
  request.task.artifact.kernelName = "kernel0";
  request.task.artifact.kernelKind = KernelKind::Vec;
  request.task.artifact.socVersion = "Ascend910B1";
  request.task.artifact.artifactRoot = "/tmp/artifact-root";
  request.task.invocation.blockDim = 8;
  request.task.invocation.workspaceSize = 8192;
  request.task.invocation.enableProfiling = true;
  request.task.invocation.inputs.push_back(TensorBinding{
      "input0", BindingSourceKind::ExternalFile, inputPath});
  request.task.invocation.outputs.push_back(TensorBinding{
      "output0", BindingSourceKind::ExternalFile,
      (runtimeDir / "actual.npy").string()});
  request.task.invocation.expectedOutputs.push_back(TensorBinding{
      "output0", BindingSourceKind::ExternalFile, expectedOutputPath});
  request.task.invocation.tiling = TilingBinding{};
  request.task.invocation.tiling->schemaPath =
      (std::filesystem::current_path() / "examples" /
       "relu-broadcast-transpose" / "tiling_space.json")
          .string();
  request.task.invocation.tiling->params =
      "TB_M=64,TB_N=64,dim_arg0_0=640,dim_arg1_0=500,dim_arg0_1=1,dim_arg1_1=640";

  auto profilePathOr =
      materializeSimulatorProfileArtifactForTest(request, 1498485);
  EXPECT((bool)profilePathOr, "sim profile artifact materialization succeeds");
  if (!profilePathOr) {
    llvm::consumeError(profilePathOr.takeError());
    return;
  }

  const std::filesystem::path profilePath = *profilePathOr;

  EXPECT(std::filesystem::exists(profilePath),
         "sim profile trace artifact exists");

  const std::string profileText = readTextFile(profilePath.string());
  if (profileText.empty())
    return;

  auto jsonOr = llvm::json::parse(profileText);
  EXPECT((bool)jsonOr, "sim profile trace parses as json");
  if (!jsonOr) {
    llvm::consumeError(jsonOr.takeError());
    return;
  }

  const auto *object = jsonOr->getAsObject();
  EXPECT(object != nullptr, "sim profile trace is a json object");
  if (!object)
    return;

  EXPECT(object->getInteger("schema_version") &&
             *object->getInteger("schema_version") == 1,
         "sim profile trace has schema_version=1");
  EXPECT(object->getString("backend") &&
             *object->getString("backend") == "simulation",
         "sim profile trace carries backend");
  EXPECT(object->getString("session_id") &&
             *object->getString("session_id") == "session-profile",
         "sim profile trace carries session_id");
  EXPECT(object->getString("task_id") &&
             *object->getString("task_id") == "task0",
         "sim profile trace carries task_id");
  EXPECT(object->getString("kernel_name") &&
             *object->getString("kernel_name") == "kernel0",
         "sim profile trace carries kernel_name");
  EXPECT(object->getString("kernel_kind") &&
             *object->getString("kernel_kind") == "vec",
         "sim profile trace carries kernel_kind");
  EXPECT(object->getString("soc_version") &&
             *object->getString("soc_version") == "Ascend910B1",
         "sim profile trace carries soc_version");
  EXPECT(object->getInteger("block_dim") &&
             *object->getInteger("block_dim") == 8,
         "sim profile trace carries block_dim");
  EXPECT(object->getInteger("workspace_size") &&
             *object->getInteger("workspace_size") == 8192,
         "sim profile trace carries workspace_size");
  EXPECT(object->getInteger("cycle_count") &&
             *object->getInteger("cycle_count") == 1498485,
         "sim profile trace carries cycle_count");
  EXPECT(object->getInteger("elapsed_us") &&
             *object->getInteger("elapsed_us") == 1498485,
         "sim profile trace carries elapsed_us");
  EXPECT(object->getInteger("score") &&
             *object->getInteger("score") == 1498485,
         "sim profile trace carries score");
  EXPECT(object->getBoolean("validation_passed") &&
             *object->getBoolean("validation_passed"),
         "sim profile trace marks validation_passed");
  EXPECT(object->getString("artifact_root") &&
             *object->getString("artifact_root") == "/tmp/artifact-root",
         "sim profile trace carries artifact_root");

  auto *inputs = object->getArray("inputs");
  EXPECT(inputs && inputs->size() == 1, "sim profile trace emits one input");
  if (inputs && inputs->size() == 1) {
    const auto *input0 = (*inputs)[0].getAsObject();
    EXPECT(input0 && input0->getString("name") &&
               *input0->getString("name") == "input0",
           "sim profile trace input0 name");
    EXPECT(input0 && input0->getString("dtype") &&
               *input0->getString("dtype") == "f16",
           "sim profile trace input0 dtype");
    EXPECT(input0 && input0->getArray("shape") &&
               input0->getArray("shape")->size() == 2 &&
               input0->getArray("shape")->front().getAsInteger() &&
               *input0->getArray("shape")->front().getAsInteger() == 4,
           "sim profile trace input0 shape");
  }

  auto *outputs = object->getArray("outputs");
  EXPECT(outputs && outputs->size() == 1, "sim profile trace emits one output");
  if (outputs && outputs->size() == 1) {
    const auto *output0 = (*outputs)[0].getAsObject();
    EXPECT(output0 && output0->getString("name") &&
               *output0->getString("name") == "output0",
           "sim profile trace output name");
    EXPECT(output0 && output0->getString("dtype") &&
               *output0->getString("dtype") == "f32",
           "sim profile trace output dtype");
    EXPECT(output0 && output0->getArray("shape") &&
               output0->getArray("shape")->size() == 2 &&
               output0->getArray("shape")->front().getAsInteger() &&
               *output0->getArray("shape")->front().getAsInteger() == 2,
           "sim profile trace output shape");
  }

  const auto *tiling = object->getObject("tiling");
  EXPECT(tiling != nullptr, "sim profile trace emits tiling object");
  if (tiling) {
    EXPECT(tiling->getBoolean("present") &&
               *tiling->getBoolean("present"),
           "sim profile trace marks tiling present");
    EXPECT(tiling->getString("binary_path") &&
               *tiling->getString("binary_path") ==
                   (runtimeDir / "tiling.bin").string(),
           "sim profile trace materializes tiling binary path");
    const auto expectedTilingBytes =
        static_cast<int64_t>(std::filesystem::file_size(runtimeDir / "tiling.bin"));
    EXPECT(tiling->getInteger("bytes") &&
               *tiling->getInteger("bytes") == expectedTilingBytes,
           "sim profile trace records tiling byte size");
  }
}

static void testExecutionSessionPlansTopologicalOrder() {
  TaskGraph graph;

  RuntimeTask taskA;
  taskA.taskId = "task_a";

  RuntimeTask taskB;
  taskB.taskId = "task_b";
  taskB.dependencies = {"task_a"};

  RuntimeTask taskC;
  taskC.taskId = "task_c";
  taskC.dependencies = {"task_b"};

  auto addC = graph.addTask(taskC);
  EXPECT(!addC, "execution session add task_c");
  auto addA = graph.addTask(taskA);
  EXPECT(!addA, "execution session add task_a");
  auto addB = graph.addTask(taskB);
  EXPECT(!addB, "execution session add task_b");

  ExecutionSession session(ExecutionBackendKind::Simulation);
  auto planOr = session.plan(graph);
  EXPECT((bool)planOr, "execution session plan succeeds");
  if (planOr) {
    EXPECT(planOr->orderedTaskIds.size() == 3,
           "execution session plan size");
    EXPECT(planOr->readyTaskIds.size() == 1,
           "execution session plan ready task count");
    EXPECT(planOr->blockedTaskCount == 2,
           "execution session plan blocked task count");
    EXPECT(planOr->orderedTaskIds[0] == "task_a",
           "execution session plan first task");
    EXPECT(planOr->orderedTaskIds[1] == "task_b",
           "execution session plan second task");
    EXPECT(planOr->orderedTaskIds[2] == "task_c",
           "execution session plan third task");
    EXPECT(planOr->readyTaskIds[0] == "task_a",
           "execution session plan ready task");
  }
}

static void testExecutionSessionPlanTracksMultipleReadyRoots() {
  TaskGraph graph;

  RuntimeTask taskA;
  taskA.taskId = "task_a";
  RuntimeTask taskB;
  taskB.taskId = "task_b";
  RuntimeTask taskC;
  taskC.taskId = "task_c";
  taskC.dependencies = {"task_a", "task_b"};

  auto addA = graph.addTask(taskA);
  EXPECT(!addA, "execution session multi-root add task_a");
  auto addC = graph.addTask(taskC);
  EXPECT(!addC, "execution session multi-root add task_c");
  auto addB = graph.addTask(taskB);
  EXPECT(!addB, "execution session multi-root add task_b");

  ExecutionSession session(ExecutionBackendKind::Simulation);
  auto planOr = session.plan(graph);
  EXPECT((bool)planOr, "execution session multi-root plan succeeds");
  if (planOr) {
    EXPECT(planOr->readyTaskIds.size() == 2,
           "execution session multi-root ready count");
    EXPECT(planOr->blockedTaskCount == 1,
           "execution session multi-root blocked count");
    EXPECT(planOr->readyTaskIds[0] == "task_a",
           "execution session multi-root first ready task");
    EXPECT(planOr->readyTaskIds[1] == "task_b",
           "execution session multi-root second ready task");
  }
}

static void testExecutionSessionRunsTasksInTopologicalOrder() {
  TaskGraph graph;

  RuntimeTask taskA;
  taskA.taskId = "task_a";

  RuntimeTask taskB;
  taskB.taskId = "task_b";
  taskB.dependencies = {"task_a"};

  RuntimeTask taskC;
  taskC.taskId = "task_c";
  taskC.dependencies = {"task_b"};

  auto addC = graph.addTask(taskC);
  EXPECT(!addC, "execution session run add task_c");
  auto addA = graph.addTask(taskA);
  EXPECT(!addA, "execution session run add task_a");
  auto addB = graph.addTask(taskB);
  EXPECT(!addB, "execution session run add task_b");

  auto executionOrderOr = graph.executionOrder();
  EXPECT((bool)executionOrderOr, "task graph execution order succeeds");
  if (executionOrderOr) {
    EXPECT(executionOrderOr->size() == 3, "task graph execution order size");
    EXPECT((*executionOrderOr)[0].taskId == "task_a",
           "task graph execution order first task");
    EXPECT((*executionOrderOr)[1].taskId == "task_b",
           "task graph execution order second task");
    EXPECT((*executionOrderOr)[2].taskId == "task_c",
           "task graph execution order third task");
  }

  auto driver = std::make_shared<OrderedExecutionBackendDriver>();
  OrderedExecutionBackendDriver *driverPtr = driver.get();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  auto traceOr = session.run(graph);
  EXPECT((bool)traceOr, "execution session run succeeds");
  if (traceOr) {
    EXPECT(traceOr->events.size() == 3,
           "execution session aggregates profile events");
    EXPECT(traceOr->events[0].taskId == "task_a",
           "execution session profile event order first");
    EXPECT(traceOr->events[1].taskId == "task_b",
           "execution session profile event order second");
    EXPECT(traceOr->events[2].taskId == "task_c",
           "execution session profile event order third");
  }

  EXPECT(driverPtr->seenTaskIds.size() == 3,
         "execution session backend invocation count");
  if (driverPtr->seenTaskIds.size() == 3) {
    EXPECT(traceOr && traceOr->sessionId == driverPtr->seenSessionIds.front(),
           "execution session trace keeps session id");
    EXPECT(driverPtr->seenTaskIds[0] == "task_a",
           "execution session backend sees first task");
    EXPECT(driverPtr->seenTaskIds[1] == "task_b",
           "execution session backend sees second task");
    EXPECT(driverPtr->seenTaskIds[2] == "task_c",
           "execution session backend sees third task");

    EXPECT(driverPtr->seenSessionIds[0] == driverPtr->seenSessionIds[1] &&
               driverPtr->seenSessionIds[1] == driverPtr->seenSessionIds[2],
           "execution session reuses one session id");
    EXPECT(driverPtr->seenWorkingDirectories[0] ==
               driverPtr->seenWorkingDirectories[1] &&
               driverPtr->seenWorkingDirectories[1] ==
                   driverPtr->seenWorkingDirectories[2],
           "execution session reuses one working directory");
    EXPECT(std::filesystem::exists(driverPtr->seenWorkingDirectories[0]),
           "execution session working directory exists");
  }

  auto secondTraceOr = session.run(graph);
  EXPECT((bool)secondTraceOr, "execution session second run succeeds");
  EXPECT(driverPtr->seenSessionIds.size() == 6,
         "execution session second run adds three more invocations");
  if (driverPtr->seenSessionIds.size() == 6) {
    EXPECT(driverPtr->seenSessionIds[0] != driverPtr->seenSessionIds[3],
           "execution session generates a fresh session id per run");
    EXPECT(driverPtr->seenWorkingDirectories[0] !=
               driverPtr->seenWorkingDirectories[3],
           "execution session generates a fresh working directory per run");
    EXPECT(driverPtr->seenSessionIds[3] == driverPtr->seenSessionIds[4] &&
               driverPtr->seenSessionIds[4] == driverPtr->seenSessionIds[5],
           "execution session second run reuses one session id");
    EXPECT(driverPtr->seenWorkingDirectories[3] ==
               driverPtr->seenWorkingDirectories[4] &&
               driverPtr->seenWorkingDirectories[4] ==
                   driverPtr->seenWorkingDirectories[5],
           "execution session second run reuses one working directory");
  }
  if (traceOr && secondTraceOr) {
    EXPECT(traceOr->sessionId != secondTraceOr->sessionId,
           "execution session returns a fresh trace session id per run");
  }
}

static void testExecutionSessionCanReleaseWorkingDirectoriesForProcessExit() {
  TaskGraph graph;

  RuntimeTask task;
  task.taskId = "task_release";

  auto addTaskErr = graph.addTask(task);
  EXPECT(!addTaskErr, "execution session release add task");

  auto driver = std::make_shared<OrderedExecutionBackendDriver>();
  OrderedExecutionBackendDriver *driverPtr = driver.get();
  auto session =
      std::make_unique<ExecutionSession>(ExecutionBackendKind::Simulation, driver);

  auto traceOr = session->run(graph);
  EXPECT((bool)traceOr, "execution session release run succeeds");
  if (!(bool)traceOr || driverPtr->seenWorkingDirectories.empty()) {
    if (!traceOr)
      llvm::consumeError(traceOr.takeError());
    return;
  }

  const std::string workingDirectory = driverPtr->seenWorkingDirectories.front();
  EXPECT(std::filesystem::exists(workingDirectory),
         "execution session release sees existing working directory");

  session->releaseWorkingDirectoriesForProcessExit();
  session.reset();

  EXPECT(std::filesystem::exists(workingDirectory),
         "execution session release skips destructor cleanup for process-exit path");

  std::error_code ec;
  std::filesystem::remove_all(workingDirectory, ec);
}

static void testExecutionSessionCarriesInvocationBindings() {
  RuntimeTask task;
  task.taskId = "main";
  task.invocation.blockDim = 8;
  task.invocation.workspaceSize = 4096;
  task.invocation.atol = 3.5;
  task.invocation.rtol = 0.125;

  TensorBinding input;
  input.name = "input0";
  input.path = "/tmp/input0.npy";
  task.invocation.inputs.push_back(input);

  TensorBinding output;
  output.name = "output0";
  output.path = "/tmp/output0.npy";
  output.shape = std::vector<int64_t>{4, 8};
  output.dtype = DType::F32;
  task.invocation.outputs.push_back(output);

  TaskGraph graph;
  auto addTaskErr = graph.addTask(task);
  EXPECT(!addTaskErr, "execution session invocation add task");

  auto driver = std::make_shared<RecordingBackendDriver>();
  RecordingBackendDriver *driverPtr = driver.get();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  auto traceOr = session.run(graph);
  EXPECT((bool)traceOr, "execution session invocation run succeeds");
  EXPECT(driverPtr->invocations == 1,
         "execution session invocation backend invoked exactly once");
  if (driverPtr->invocations == 1) {
    EXPECT(driverPtr->lastRequest.task.invocation.blockDim == 8,
           "execution session preserves invocation block dim");
    EXPECT(driverPtr->lastRequest.task.invocation.workspaceSize == 4096,
           "execution session preserves invocation workspace size");
    EXPECT(driverPtr->lastRequest.task.invocation.atol == 3.5,
           "execution session preserves invocation atol");
    EXPECT(driverPtr->lastRequest.task.invocation.rtol == 0.125,
           "execution session preserves invocation rtol");
    EXPECT(driverPtr->lastRequest.task.invocation.inputs.size() == 1,
           "execution session preserves invocation inputs");
    EXPECT(driverPtr->lastRequest.task.invocation.outputs.size() == 1,
           "execution session preserves invocation outputs");
    if (driverPtr->lastRequest.task.invocation.inputs.size() == 1) {
      EXPECT(driverPtr->lastRequest.task.invocation.inputs[0].name == "input0",
             "execution session preserves invocation input name");
      EXPECT(driverPtr->lastRequest.task.invocation.inputs[0].path ==
                 "/tmp/input0.npy",
             "execution session preserves invocation input path");
    }
    if (driverPtr->lastRequest.task.invocation.outputs.size() == 1) {
      EXPECT(driverPtr->lastRequest.task.invocation.outputs[0].name ==
                 "output0",
             "execution session preserves invocation output name");
      EXPECT(driverPtr->lastRequest.task.invocation.outputs[0].path ==
                 "/tmp/output0.npy",
             "execution session preserves invocation output path");
      EXPECT(driverPtr->lastRequest.task.invocation.outputs[0].shape.has_value(),
             "execution session preserves invocation output shape metadata");
      EXPECT(driverPtr->lastRequest.task.invocation.outputs[0].dtype.has_value(),
             "execution session preserves invocation output dtype metadata");
      if (driverPtr->lastRequest.task.invocation.outputs[0].shape) {
        EXPECT(driverPtr->lastRequest.task.invocation.outputs[0].shape->size() ==
                   2 &&
                   (*driverPtr->lastRequest.task.invocation.outputs[0].shape)[0] == 4 &&
                   (*driverPtr->lastRequest.task.invocation.outputs[0].shape)[1] == 8,
               "execution session preserves invocation output shape values");
      }
      if (driverPtr->lastRequest.task.invocation.outputs[0].dtype) {
        EXPECT(*driverPtr->lastRequest.task.invocation.outputs[0].dtype ==
                   DType::F32,
               "execution session preserves invocation output dtype value");
      }
    }
  }
}

static void testExecutionSessionResolvesTaskOutputBindings() {
  TaskGraph graph;

  RuntimeTask producer;
  producer.taskId = "producer";
  TensorBinding produced;
  produced.name = "mid";
  produced.shape = std::vector<int64_t>{16};
  produced.dtype = DType::F16;
  producer.invocation.outputs.push_back(produced);

  RuntimeTask consumer;
  consumer.taskId = "consumer";
  consumer.dependencies = {"producer"};
  TensorBinding consumed;
  consumed.name = "mid";
  consumed.sourceKind = BindingSourceKind::TaskOutput;
  consumed.upstreamTaskId = "producer";
  consumed.upstreamOutputName = "mid";
  consumer.invocation.inputs.push_back(consumed);
  TensorBinding finalOutput;
  finalOutput.name = "out";
  finalOutput.shape = std::vector<int64_t>{16};
  finalOutput.dtype = DType::F16;
  consumer.invocation.outputs.push_back(finalOutput);

  auto addProducer = graph.addTask(producer);
  EXPECT(!addProducer, "task output binding add producer");
  auto addConsumer = graph.addTask(consumer);
  EXPECT(!addConsumer, "task output binding add consumer");

  auto driver = std::make_shared<CapturingExecutionBackendDriver>();
  CapturingExecutionBackendDriver *driverPtr = driver.get();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  auto traceOr = session.run(graph);
  EXPECT((bool)traceOr, "task output binding run succeeds");
  EXPECT(driverPtr->requests.size() == 2,
         "task output binding invokes both tasks");
  if (driverPtr->requests.size() == 2) {
    const auto &producerRequest = driverPtr->requests[0];
    const auto &consumerRequest = driverPtr->requests[1];
    EXPECT(producerRequest.task.invocation.outputs.size() == 1,
           "task output binding producer output count");
    EXPECT(!producerRequest.task.invocation.outputs[0].path.empty(),
           "task output binding producer output path is materialized");
    EXPECT(consumerRequest.task.invocation.inputs.size() == 1,
           "task output binding consumer input count");
    EXPECT(consumerRequest.task.invocation.inputs[0].sourceKind ==
               BindingSourceKind::ExternalFile,
           "task output binding consumer input is resolved to external file");
    EXPECT(consumerRequest.task.invocation.inputs[0].path ==
               producerRequest.task.invocation.outputs[0].path,
           "task output binding consumer reuses producer materialized path");
    EXPECT(consumerRequest.task.invocation.inputs[0].shape.has_value(),
           "task output binding propagates shape metadata");
    EXPECT(consumerRequest.task.invocation.inputs[0].dtype.has_value(),
           "task output binding propagates dtype metadata");
    if (consumerRequest.task.invocation.inputs[0].shape) {
      EXPECT(consumerRequest.task.invocation.inputs[0].shape->size() == 1 &&
                 (*consumerRequest.task.invocation.inputs[0].shape)[0] == 16,
             "task output binding propagated shape value");
    }
    if (consumerRequest.task.invocation.inputs[0].dtype) {
      EXPECT(*consumerRequest.task.invocation.inputs[0].dtype == DType::F16,
             "task output binding propagated dtype value");
    }
    EXPECT(!consumerRequest.task.invocation.outputs[0].path.empty(),
           "task output binding downstream output path is materialized");
  }
}

class RejectingTaskBDriver : public ExecutionBackendDriver {
public:
  llvm::Expected<ExecutionResult>
  run(const ExecutionRequest &request) override {
    seenTaskIds.push_back(request.task.taskId);
    if (request.task.taskId == "task_b") {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "[test:gate] rejected task_b");
    }
    ExecutionResult result;
    result.taskId = request.task.taskId;
    return result;
  }

  std::vector<std::string> seenTaskIds;
};

static void testExecutionSessionStopsAtGateRejectedTask() {
  TaskGraph graph;

  RuntimeTask taskA;
  taskA.taskId = "task_a";

  RuntimeTask taskB;
  taskB.taskId = "task_b";

  RuntimeTask taskC;
  taskC.taskId = "task_c";
  taskC.dependencies = {"task_a"};

  auto addA = graph.addTask(taskA);
  EXPECT(!addA, "execution session gate add task_a");
  auto addB = graph.addTask(taskB);
  EXPECT(!addB, "execution session gate add task_b");
  auto addC = graph.addTask(taskC);
  EXPECT(!addC, "execution session gate add task_c");

  auto driver = std::make_shared<RejectingTaskBDriver>();
  RejectingTaskBDriver *driverPtr = driver.get();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  auto traceOr = session.run(graph);
  EXPECT(!(bool)traceOr, "execution session gate rejection fails run");
  if (!traceOr) {
    const std::string message = llvm::toString(traceOr.takeError());
    EXPECT(message.find("[test:gate] rejected task_b") != std::string::npos,
           "execution session gate surfaces rejection message");
  }
  EXPECT(driverPtr->seenTaskIds.size() == 2,
         "execution session gate stops after rejected ready task");
  if (driverPtr->seenTaskIds.size() == 2) {
    EXPECT(driverPtr->seenTaskIds[0] == "task_a",
           "execution session gate runs first ready task");
    EXPECT(driverPtr->seenTaskIds[1] == "task_b",
           "execution session gate runs second ready task before stopping");
  }
}

static void testExecutionSessionRejectsUnknownMixResourceType() {
  auto driver = std::make_shared<RecordingBackendDriver>();
  RecordingBackendDriver *driverPtr = driver.get();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  TaskGraph graph;
  auto addTaskErr = graph.addTask(
      makeGateTask("mix_unknown", KernelKind::Mix, MixResourceType::Unknown));
  EXPECT(!addTaskErr, "scheduler gate unknown mix add task");

  auto traceOr = session.run(graph);
  EXPECT(!(bool)traceOr, "scheduler gate rejects unknown mix resource type");
  EXPECT(driverPtr->invocations == 0,
         "scheduler gate does not invoke backend on rejection");
  if (!traceOr) {
    const std::string message = llvm::toString(traceOr.takeError());
    EXPECT(message.find("unsupported mix resource type") != std::string::npos,
           "scheduler gate error mentions unsupported mix resource type");
    EXPECT(message.find("mix_unknown") != std::string::npos,
           "scheduler gate error mentions task id");
  }
}

static void testExecutionSessionAcceptsSupportedMixResourceTypes() {
  auto driver = std::make_shared<RecordingBackendDriver>();
  RecordingBackendDriver *driverPtr = driver.get();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  TaskGraph graph;
  auto add1 = graph.addTask(
      makeGateTask("mix_1c1v", KernelKind::Mix, MixResourceType::Mix1C1V));
  EXPECT(!add1, "scheduler gate add mix_1c1v");
  auto add2 = graph.addTask(
      makeGateTask("mix_1c2v", KernelKind::Mix, MixResourceType::Mix1C2V));
  EXPECT(!add2, "scheduler gate add mix_1c2v");

  auto traceOr = session.run(graph);
  EXPECT((bool)traceOr,
         "scheduler gate accepts supported mix resource types");
  EXPECT(driverPtr->invocations == 2,
         "scheduler gate invokes backend for both supported mix tasks");
}

static void testExecutionSessionAcceptsVecAndCubeTasks() {
  auto driver = std::make_shared<RecordingBackendDriver>();
  RecordingBackendDriver *driverPtr = driver.get();
  ExecutionSession session(ExecutionBackendKind::Simulation, driver);

  TaskGraph graph;
  auto addVec =
      graph.addTask(makeGateTask("vec_task", KernelKind::Vec,
                                 MixResourceType::Unknown));
  EXPECT(!addVec, "scheduler gate add vec task");
  auto addCube =
      graph.addTask(makeGateTask("cube_task", KernelKind::Cube,
                                 MixResourceType::Unknown));
  EXPECT(!addCube, "scheduler gate add cube task");

  auto traceOr = session.run(graph);
  EXPECT((bool)traceOr, "scheduler gate accepts vec and cube tasks");
  EXPECT(driverPtr->invocations == 2,
         "scheduler gate invokes backend for vec and cube tasks");
}

static void testRunManifestParsesVecSimulationSpec() {
  const std::string manifestPath = "/tmp/runtime_run_manifest.json";
  {
    std::ofstream os(manifestPath);
    os << R"JSON({
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "/tmp/artifact",
  "inputs": [
    { "name": "data0", "path": "/tmp/in0.npy" },
    { "name": "data1", "path": "/tmp/in1.npy" }
  ],
  "outputs": [
    { "name": "out", "path": "/tmp/actual.npy", "shape": [4, 8], "dtype": "f32" }
  ],
  "expected_outputs": [
    { "name": "out", "path": "/tmp/expected.npy" }
  ],
  "tiling": {
    "schema": "/tmp/tiling_space.json",
    "params": "TB_M=64,TB_N=64"
  },
  "block_dim": 8,
  "workspace_size": 16384,
  "profiling": true,
  "atol": 2.5,
  "rtol": 0.05
})JSON";
  }

  auto specOr = loadRunManifest(manifestPath);
  EXPECT((bool)specOr, "run manifest parse succeeds");
  if (specOr) {
    EXPECT(specOr->backendKind == ExecutionBackendKind::Simulation,
           "run manifest backend kind");
    EXPECT(specOr->tasks.size() == 1,
           "run manifest single-task compatibility preserves one task");
    if (specOr->tasks.size() == 1) {
      const RunTaskSpec &task = specOr->tasks[0];
      EXPECT(task.taskId == "main", "run manifest task id");
      EXPECT(task.artifactRoot == "/tmp/artifact",
             "run manifest artifact root");
      EXPECT(task.invocation.inputs.size() == 2,
           "run manifest input count");
      EXPECT(task.invocation.outputs.size() == 1,
           "run manifest output count");
      EXPECT(task.invocation.expectedOutputs.size() == 1,
           "run manifest expected output count");
      EXPECT(task.invocation.blockDim == 8,
           "run manifest block dim");
      EXPECT(task.invocation.workspaceSize == 16384,
           "run manifest workspace size");
      EXPECT(task.invocation.enableProfiling,
             "run manifest profiling flag");
      EXPECT(task.invocation.atol == 2.5,
             "run manifest atol");
      EXPECT(task.invocation.rtol == 0.05,
             "run manifest rtol");
      EXPECT(task.invocation.tiling.has_value(),
           "run manifest tiling present");
      EXPECT(task.invocation.outputs[0].shape.has_value(),
           "run manifest output shape metadata present");
      EXPECT(task.invocation.outputs[0].dtype.has_value(),
           "run manifest output dtype metadata present");
      if (task.invocation.outputs[0].shape) {
        EXPECT(task.invocation.outputs[0].shape->size() == 2 &&
                   (*task.invocation.outputs[0].shape)[0] == 4 &&
                   (*task.invocation.outputs[0].shape)[1] == 8,
             "run manifest output shape metadata values");
      }
      if (task.invocation.outputs[0].dtype) {
        EXPECT(*task.invocation.outputs[0].dtype == DType::F32,
             "run manifest output dtype metadata value");
      }
      if (task.invocation.tiling) {
        EXPECT(task.invocation.tiling->schemaPath == "/tmp/tiling_space.json",
             "run manifest tiling schema path");
        EXPECT(task.invocation.tiling->params == "TB_M=64,TB_N=64",
             "run manifest tiling params");
      }
    }
  }
}

static void testRunManifestParsesOutputMetadataWithoutExpectedOutputs() {
  const std::string manifestPath = "/tmp/runtime_run_manifest_no_expected.json";
  {
    std::ofstream os(manifestPath);
    os << R"JSON({
  "task_id": "main",
  "backend": "sim",
  "artifact_root": "/tmp/artifact",
  "inputs": [
    { "name": "data0", "path": "/tmp/in0.npy" }
  ],
  "outputs": [
    { "name": "out", "path": "/tmp/actual.npy", "shape": [32], "dtype": "f16" }
  ]
})JSON";
  }

  auto specOr = loadRunManifest(manifestPath);
  EXPECT((bool)specOr, "run manifest without expected outputs parses");
  if (specOr) {
    EXPECT(specOr->tasks.size() == 1,
           "run manifest without expected outputs keeps one task");
    if (specOr->tasks.size() == 1) {
      const RunTaskSpec &task = specOr->tasks[0];
      EXPECT(task.invocation.expectedOutputs.empty(),
           "run manifest without expected outputs leaves golden bindings empty");
      EXPECT(task.invocation.outputs.size() == 1,
           "run manifest without expected outputs keeps output bindings");
      if (task.invocation.outputs.size() == 1) {
        EXPECT(task.invocation.outputs[0].shape.has_value(),
             "run manifest without expected outputs carries output shape");
        EXPECT(task.invocation.outputs[0].dtype.has_value(),
             "run manifest without expected outputs carries output dtype");
        if (task.invocation.outputs[0].shape) {
          EXPECT(task.invocation.outputs[0].shape->size() == 1 &&
                     (*task.invocation.outputs[0].shape)[0] == 32,
               "run manifest without expected outputs shape value");
        }
        if (task.invocation.outputs[0].dtype) {
          EXPECT(*task.invocation.outputs[0].dtype == DType::F16,
               "run manifest without expected outputs dtype value");
        }
      }
    }
  }
}

static void testRunManifestParsesTaskOutputBinding() {
  const std::string manifestPath = "/tmp/runtime_run_manifest_task_output.json";
  {
    std::ofstream os(manifestPath);
    os << R"JSON({
  "task_id": "consumer",
  "backend": "sim",
  "artifact_root": "/tmp/artifact",
  "inputs": [
    {
      "name": "mid",
      "source": "task_output",
      "upstream_task": "producer",
      "upstream_output": "mid"
    }
  ],
  "outputs": [
    { "name": "out", "path": "/tmp/out.npy", "shape": [16], "dtype": "f16" }
  ]
})JSON";
  }

  auto specOr = loadRunManifest(manifestPath);
  EXPECT((bool)specOr, "run manifest task output binding parses");
  if (specOr) {
    EXPECT(specOr->tasks.size() == 1,
           "run manifest task output keeps one task");
    if (specOr->tasks.size() == 1) {
      const RunTaskSpec &task = specOr->tasks[0];
      EXPECT(task.invocation.inputs.size() == 1,
           "run manifest task output input count");
      if (task.invocation.inputs.size() == 1) {
        EXPECT(task.invocation.inputs[0].sourceKind ==
                 BindingSourceKind::TaskOutput,
             "run manifest task output source kind");
        EXPECT(task.invocation.inputs[0].upstreamTaskId == "producer",
             "run manifest task output upstream task");
        EXPECT(task.invocation.inputs[0].upstreamOutputName == "mid",
             "run manifest task output upstream output");
        EXPECT(task.invocation.inputs[0].path.empty(),
             "run manifest task output does not require path");
      }
    }
  }
}

static void testRunManifestParsesDagSpec() {
  const std::string manifestPath = "/tmp/runtime_run_manifest_dag.json";
  {
    std::ofstream os(manifestPath);
    os << R"JSON({
  "backend": "sim",
  "artifact_root": "/tmp/shared-artifact",
  "tasks": [
    {
      "task_id": "producer",
      "inputs": [
        { "name": "data0", "path": "/tmp/in0.npy" }
      ],
      "outputs": [
        { "name": "mid", "shape": [16], "dtype": "f16", "path": "/tmp/mid.npy" }
      ]
    },
    {
      "task_id": "consumer",
      "dependencies": ["producer"],
      "inputs": [
        { "name": "mid", "source": "task_output", "upstream_task": "producer", "upstream_output": "mid" }
      ],
      "outputs": [
        { "name": "out", "shape": [16], "dtype": "f16", "path": "/tmp/out.npy" }
      ]
    }
  ]
})JSON";
  }

  auto specOr = loadRunManifest(manifestPath);
  EXPECT((bool)specOr, "run manifest dag parses");
  if (specOr) {
    EXPECT(specOr->backendKind == ExecutionBackendKind::Simulation,
           "run manifest dag backend kind");
    EXPECT(specOr->tasks.size() == 2,
           "run manifest dag task count");
    if (specOr->tasks.size() == 2) {
      EXPECT(specOr->tasks[0].artifactRoot == "/tmp/shared-artifact",
             "run manifest dag task 0 inherits top-level artifact root");
      EXPECT(specOr->tasks[1].artifactRoot == "/tmp/shared-artifact",
             "run manifest dag task 1 inherits top-level artifact root");
      EXPECT(specOr->tasks[1].dependencies.size() == 1 &&
                 specOr->tasks[1].dependencies[0] == "producer",
             "run manifest dag dependencies");
      EXPECT(specOr->tasks[1].invocation.inputs.size() == 1 &&
                 specOr->tasks[1].invocation.inputs[0].sourceKind ==
                     BindingSourceKind::TaskOutput,
             "run manifest dag task output input");
    }
  }
}

static void testRunManifestParsesDagArtifactRootOverride() {
  const std::string manifestPath =
      "/tmp/runtime_run_manifest_dag_override.json";
  {
    std::ofstream os(manifestPath);
    os << R"JSON({
  "backend": "sim",
  "artifact_root": "/tmp/shared-artifact",
  "tasks": [
    {
      "task_id": "producer",
      "artifact_root": "/tmp/producer-artifact",
      "outputs": [
        { "name": "mid", "shape": [16], "dtype": "f16" }
      ]
    },
    {
      "task_id": "consumer",
      "dependencies": ["producer"],
      "artifact_root": "/tmp/consumer-artifact",
      "inputs": [
        { "name": "mid", "source": "task_output", "upstream_task": "producer", "upstream_output": "mid" }
      ],
      "outputs": [
        { "name": "out", "shape": [16], "dtype": "f16" }
      ]
    }
  ]
})JSON";
  }

  auto specOr = loadRunManifest(manifestPath);
  EXPECT((bool)specOr, "run manifest dag artifact-root override parses");
  if (specOr) {
    EXPECT(specOr->tasks.size() == 2,
           "run manifest dag artifact-root override task count");
    if (specOr->tasks.size() == 2) {
      EXPECT(specOr->tasks[0].artifactRoot == "/tmp/producer-artifact",
             "run manifest dag task 0 artifact root override");
      EXPECT(specOr->tasks[1].artifactRoot == "/tmp/consumer-artifact",
             "run manifest dag task 1 artifact root override");
    }
  }
}

int main() {
  testTaskGraphBasics();
  testProfileTraceCollectsArtifactPaths();
  testDuplicateTaskIds();
  testEmptyTaskId();
  testUnknownDependency();
  testCycleDetection();
  testKernelArtifactNormalization();
  testVecCubeArtifactBackendCompilesVecArtifact();
  testVecCubeArtifactBackendCompilesCubeArtifact();
  testRuntimeSessionRequestBuilderLoadsMixArtifactFromRoot();
  testRuntimeSessionRequestBuilderLoadsMixArtifactMetadataPath();
  testRuntimeSessionRequestBuilderRejectsMissingMixArtifactMetadata();
  testPrepareRuntimeSessionGraphUsesMixMetadataDefaults();
  testPrepareRuntimeSessionGraphFallsBackToManifestAbiDefaults();
  testRuntimeSessionRequestBuilderLoadsVecArtifactFromRoot();
  testRuntimeSessionRequestBuilderRejectsUnsupportedKernelKind();
  testRuntimeSessionRequestBuilderRejectsMissingKernelKind();
  testRuntimeSessionRequestBuilderBuildsSingleTaskGraph();
  testMixValidationCanBeRepresentedAsRuntimeTask();
  testOutputComparatorExactMatchPasses();
  testOutputComparatorMismatchReturnsDetailedFailure();
  testOutputComparatorStructuralMismatchReturnsError();
  testExecutionRunnerContractSupportsSimulationAndRealDeviceModes();
  testExecutionRunnerContractSupportsFileLaunches();
  testExecutionRunnerContractSupportsPackedMixLaunches();
  testNativeExecutionRunnerCompileCoverage();
  testSimulationBackendReportsMissingVecBinaryLaunchFailure();
  testArtifactCompilerRequestValidation();
  testFrontendCompileRequestBuilderPreservesFields();
  testFrontendSingleTaskRunPreparationAndSummary();
  testFrontendRunExecutionUsesNormalizedContract();
  testBackendSelection();
  testDefaultBackendRequiresDriver();
  testInvalidBackendSelection();
  testBackendDelegatesToDriver();
  testNpuBackendRejectsMissingDeviceBinaryPath();
  testNpuBackendRejectsMissingMixSharedObjectPath();
  testNpuBackendReachesRealDeviceModePath();
  testExecutionSessionSupportsNpuSuccessDriver();
  testSimulatorProfileNormalization();
  testSimulatorProfileNormalizationExtractsMetrics();
  testAddProfileArtifactHelper();
  testRetainProfileArtifactsForCli();
  testRetainProfileArtifactsCreatesSessionSummary();
  testRetainedSessionSummaryContents();
  testRetainedSessionSummaryFallsBackBetweenScoreAndCycleCount();
  testRetainProfileArtifactsUsesEventMetricsWithoutParsingJson();
  testRetainProfileArtifactsFailsOnDuplicateTaskIds();
  testRetainProfileArtifactsPrunesOldSessions();
  testRetainProfileArtifactsIgnoresNonDirectories();
  testRetainedProfilePruneKeepCountForNewSession();
  testPrepareRetainedProfileRunRootForCliPrunesBeforeNewSession();
  testRuntimeSessionWorkdirPruneKeepCountForNewRun();
  testPrepareRuntimeSessionWorkdirRootForCliPrunesBeforeNewRun();
  testRetainProfileArtifactsForCliRunReturnsSummaryPath();
  testBackendSurfacesProfileTrace();
  testBackendPreservesExistingProfileTrace();
  testSimulatorProfileSchemaV1Artifact();
  testExecutionSessionPlansTopologicalOrder();
  testExecutionSessionPlanTracksMultipleReadyRoots();
  testExecutionSessionRunsTasksInTopologicalOrder();
  testExecutionSessionCanReleaseWorkingDirectoriesForProcessExit();
  testExecutionSessionCarriesInvocationBindings();
  testExecutionSessionResolvesTaskOutputBindings();
  testExecutionSessionStopsAtGateRejectedTask();
  testExecutionSessionRejectsUnknownMixResourceType();
  testExecutionSessionAcceptsSupportedMixResourceTypes();
  testExecutionSessionAcceptsVecAndCubeTasks();
  testRunManifestParsesVecSimulationSpec();
  testRunManifestParsesOutputMetadataWithoutExpectedOutputs();
  testRunManifestParsesTaskOutputBinding();
  testRunManifestParsesDagSpec();
  testRunManifestParsesDagArtifactRootOverride();

  llvm::outs() << g_pass << " passed, " << g_fail << " failed\n";
  return g_fail ? 1 : 0;
}
