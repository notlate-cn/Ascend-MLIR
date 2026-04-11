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
#include "Runtime/CompatRuntime.h"
#include "Runtime/RunManifest.h"
#include "Runtime/ExecutionBackend.h"
#include "Runtime/ExecutionSession.h"
#include "Runtime/NpuBackend.h"
#include "Runtime/NpyIO.h"
#include "Runtime/TaskGraph.h"
#include "Runtime/ArtifactCompiler.h"
#include "Runtime/SimBackend.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <fstream>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace mlir::runtime;

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

  KernelArtifact vecArtifact =
      normalizeCompiledArtifact("/tmp/demo_kernel.bin", "demo_kernel",
                                KernelKind::Vec, "Ascend910B1",
                                "/tmp/demo-out");
  EXPECT(vecArtifact.kernelName == "demo_kernel",
         "normalized vec artifact keeps kernel name");
  EXPECT(vecArtifact.kernelKind == KernelKind::Vec,
         "normalized vec artifact keeps kernel kind");
  EXPECT(vecArtifact.deviceBinaryPath == "/tmp/demo_kernel.bin",
         "normalized vec artifact stores device binary path");
  EXPECT(vecArtifact.artifactRoot == "/tmp/demo-out",
         "normalized vec artifact stores artifact root");

  MixArtifact mixArtifact;
  mixArtifact.kernel_name = "demo_kernel";
  mixArtifact.soc_version = "Ascend910B1";
  mixArtifact.work_dir = "/tmp/mix/work";
  mixArtifact.kernel_so_path = "/tmp/mix/libdemo_kernel_packed.so";
  mixArtifact.device_object_path = "/tmp/mix/device.o";
  mixArtifact.manifest_path = "/tmp/mix/mix-artifact.txt";

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
  EXPECT(normalizedMix.artifactRoot == "/tmp/mix",
         "normalized mix artifact stores compile root, not work dir");
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

static void testCompatCompileRequestPreservesFields() {
  CompatCompileOptions options;
  options.kernelSourcePath = "/tmp/demo.cpp";
  options.outputRoot = "/tmp/taskgraph-compat-out";
  options.requestedKernelName = "legacy_name";
  options.socVersion = "Ascend910B1";
  options.arch = "dav-c220-vec";
  options.kernelType = "mix";
  options.verbose = true;

  auto requestOr = buildCompatCompileRequest(options);
  EXPECT((bool)requestOr, "compat compile request builds");
  if (!requestOr)
    return;
  ArtifactCompileRequest request = *requestOr;
  EXPECT(request.kernelSource == options.kernelSourcePath,
         "compat compile request preserves kernel source");
  EXPECT(request.kernelName == options.requestedKernelName,
         "compat compile request preserves kernel name");
  EXPECT(request.kernelKind == KernelKind::Mix,
         "compat compile request maps kernel type to kernel kind");
  EXPECT(request.socVersion == options.socVersion,
         "compat compile request preserves soc version");
  EXPECT(request.outputDir == options.outputRoot,
         "compat compile request preserves output dir");
  EXPECT(request.arch == options.arch,
         "compat compile request preserves arch");
  EXPECT(request.verbose == options.verbose,
         "compat compile request preserves verbose flag");
}

static void testCompatCompileRequestRejectsUnknownKernelType() {
  CompatCompileOptions options;
  options.kernelSourcePath = "/tmp/demo.cpp";
  options.outputRoot = "/tmp/taskgraph-compat-out";
  options.requestedKernelName = "legacy_name";
  options.socVersion = "Ascend910B1";
  options.kernelType = "unsupported";

  auto requestOr = buildCompatCompileRequest(options);
  EXPECT(!(bool)requestOr, "compat compile request rejects unknown kernel type");
  if (!requestOr)
    llvm::consumeError(requestOr.takeError());
}

static void testCompatSingleTaskRunManifestBuildsExpectedBackedTask() {
  CompatValidateOptions options;
  options.artifactRoot = "/tmp/artifact";
  options.inputPaths = {"/tmp/in0.npy", "/tmp/in1.npy"};
  options.expectedOutputPath = "/tmp/golden.npy";
  options.actualOutputPath = "/tmp/out.npy";
  options.blockDim = 8;
  options.atol = 2.5;
  options.rtol = 0.05;

  NDArray expectedArray;
  expectedArray.shape = {4, 8};
  expectedArray.dtype = DType::F32;
  expectedArray.allocate();
  std::memset(expectedArray.data, 0, expectedArray.nbytes());
  auto expectedWrite = SaveNpy(options.expectedOutputPath, expectedArray);
  EXPECT(!expectedWrite, "compat expected-backed manifest writes fixture");
  if (expectedWrite)
    llvm::consumeError(std::move(expectedWrite));

  auto manifestOr = buildCompatSingleTaskRunManifest(options);
  EXPECT((bool)manifestOr, "compat expected-backed manifest builds");
  if (!manifestOr)
    return;
  RunManifestSpec manifest = *manifestOr;
  EXPECT(manifest.backendKind == ExecutionBackendKind::Simulation,
         "compat run manifest uses simulation backend");
  EXPECT(manifest.tasks.size() == 1,
         "compat run manifest builds one task");
  if (manifest.tasks.size() == 1) {
    const RunTaskSpec &task = manifest.tasks[0];
    EXPECT(task.taskId == "main",
           "compat run manifest synthesizes task id");
    EXPECT(task.artifactRoot == options.artifactRoot,
           "compat run manifest preserves artifact root");
    EXPECT(task.invocation.inputs.size() == 2,
           "compat run manifest builds input bindings");
    EXPECT(task.invocation.inputs[0].sourceKind ==
               BindingSourceKind::ExternalFile,
           "compat run manifest uses file-backed input bindings");
    EXPECT(task.invocation.inputs[0].path == "/tmp/in0.npy",
           "compat run manifest preserves first input path");
    EXPECT(task.invocation.inputs[1].path == "/tmp/in1.npy",
           "compat run manifest preserves second input path");
    EXPECT(task.invocation.outputs.size() == 1,
           "compat run manifest builds actual output binding");
    EXPECT(task.invocation.outputs[0].path == options.actualOutputPath,
           "compat run manifest preserves actual output path");
    EXPECT(task.invocation.outputs[0].shape.has_value(),
           "compat run manifest derives output shape from expected output");
    EXPECT(task.invocation.outputs[0].dtype.has_value(),
           "compat run manifest derives output dtype from expected output");
    if (task.invocation.outputs[0].shape) {
      EXPECT(task.invocation.outputs[0].shape->size() == 2 &&
                 (*task.invocation.outputs[0].shape)[0] == 4 &&
                 (*task.invocation.outputs[0].shape)[1] == 8,
             "compat run manifest preserves expected output shape");
    }
    if (task.invocation.outputs[0].dtype) {
      EXPECT(*task.invocation.outputs[0].dtype == DType::F32,
             "compat run manifest preserves expected output dtype");
    }
    EXPECT(task.invocation.expectedOutputs.size() == 1,
           "compat run manifest builds expected output binding");
    EXPECT(task.invocation.expectedOutputs[0].path == options.expectedOutputPath,
           "compat run manifest preserves expected output path");
    EXPECT(task.invocation.blockDim == 8,
           "compat run manifest preserves block dim");
    EXPECT(task.invocation.atol == 2.5,
           "compat run manifest preserves atol");
    EXPECT(task.invocation.rtol == 0.05,
           "compat run manifest preserves rtol");
  }
}

static void testCompatSingleTaskRunManifestBuildsMetadataBackedTask() {
  CompatValidateOptions options;
  options.artifactRoot = "/tmp/artifact";
  options.inputPaths = {"/tmp/in0.npy"};
  options.actualOutputPath = "/tmp/out.npy";
  options.actualOutputShape = std::vector<int64_t>{16};
  options.actualOutputDType = DType::F16;
  options.blockDim = 4;
  options.atol = 1.5;
  options.rtol = 0.02;

  auto manifestOr = buildCompatSingleTaskRunManifest(options);
  EXPECT((bool)manifestOr, "compat metadata-backed manifest builds");
  if (!manifestOr)
    return;
  RunManifestSpec manifest = *manifestOr;
  EXPECT(manifest.tasks.size() == 1,
         "compat metadata-backed manifest builds one task");
  if (manifest.tasks.size() == 1) {
    const RunTaskSpec &task = manifest.tasks[0];
    EXPECT(task.invocation.outputs.size() == 1,
           "compat metadata-backed manifest builds actual output binding");
    if (task.invocation.outputs.size() == 1) {
      EXPECT(task.invocation.outputs[0].path == options.actualOutputPath,
             "compat metadata-backed manifest preserves actual output path");
      EXPECT(task.invocation.outputs[0].shape.has_value(),
             "compat metadata-backed manifest preserves output shape");
      EXPECT(task.invocation.outputs[0].dtype.has_value(),
             "compat metadata-backed manifest preserves output dtype");
      if (task.invocation.outputs[0].shape) {
        EXPECT(task.invocation.outputs[0].shape->size() == 1 &&
                   (*task.invocation.outputs[0].shape)[0] == 16,
               "compat metadata-backed manifest preserves output shape value");
      }
      if (task.invocation.outputs[0].dtype) {
        EXPECT(*task.invocation.outputs[0].dtype == DType::F16,
               "compat metadata-backed manifest preserves output dtype value");
      }
    }
    EXPECT(task.invocation.expectedOutputs.empty(),
           "compat metadata-backed manifest omits expected outputs");
    EXPECT(task.invocation.blockDim == 4,
           "compat metadata-backed manifest preserves block dim");
    EXPECT(task.invocation.atol == 1.5,
           "compat metadata-backed manifest preserves atol");
    EXPECT(task.invocation.rtol == 0.02,
           "compat metadata-backed manifest preserves rtol");
  }
}

static void testCompatSingleTaskRunManifestRejectsInvalidCombination() {
  CompatValidateOptions expectedWithoutActual;
  expectedWithoutActual.artifactRoot = "/tmp/artifact";
  expectedWithoutActual.expectedOutputPath = "/tmp/golden.npy";
  auto missingActualOr =
      buildCompatSingleTaskRunManifest(expectedWithoutActual);
  EXPECT(!(bool)missingActualOr,
         "compat manifest rejects expected output without actual output");
  if (!missingActualOr)
    llvm::consumeError(missingActualOr.takeError());

  CompatValidateOptions actualWithoutMetadata;
  actualWithoutMetadata.artifactRoot = "/tmp/artifact";
  actualWithoutMetadata.actualOutputPath = "/tmp/out.npy";
  auto missingMetadataOr =
      buildCompatSingleTaskRunManifest(actualWithoutMetadata);
  EXPECT(!(bool)missingMetadataOr,
         "compat manifest rejects actual output without metadata");
  if (!missingMetadataOr)
    llvm::consumeError(missingMetadataOr.takeError());
}

static void testVecCompileCreatesOutputDir() {
  std::error_code ec;
  const std::string outputDir = "/tmp/taskgraph-artifact-out-created";
  std::filesystem::remove_all(outputDir, ec);
  EXPECT(!std::filesystem::exists(outputDir),
         "precondition: output dir does not exist");

  auto dirErr = prepareCompileOutputDir(outputDir);
  EXPECT(!dirErr, "prepareCompileOutputDir succeeds");
  if (dirErr)
    llvm::consumeError(std::move(dirErr));
  EXPECT(std::filesystem::exists(outputDir),
         "prepareCompileOutputDir creates the directory");
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
  testArtifactCompilerRequestValidation();
  testCompatCompileRequestPreservesFields();
  testCompatCompileRequestRejectsUnknownKernelType();
  testCompatSingleTaskRunManifestBuildsExpectedBackedTask();
  testCompatSingleTaskRunManifestBuildsMetadataBackedTask();
  testCompatSingleTaskRunManifestRejectsInvalidCombination();
  testVecCompileCreatesOutputDir();
  testBackendSelection();
  testDefaultBackendRequiresDriver();
  testInvalidBackendSelection();
  testBackendDelegatesToDriver();
  testNpuBackendRejectsMissingDeviceBinaryPath();
  testNpuBackendRejectsMissingMixSharedObjectPath();
  testNpuBackendReachesRealDeviceModePath();
  testExecutionSessionSupportsNpuSuccessDriver();
  testSimulatorProfileNormalization();
  testAddProfileArtifactHelper();
  testBackendSurfacesProfileTrace();
  testBackendPreservesExistingProfileTrace();
  testExecutionSessionPlansTopologicalOrder();
  testExecutionSessionPlanTracksMultipleReadyRoots();
  testExecutionSessionRunsTasksInTopologicalOrder();
  testExecutionSessionCarriesInvocationBindings();
  testExecutionSessionResolvesTaskOutputBindings();
  testExecutionSessionStopsAtGateRejectedTask();
  testRunManifestParsesVecSimulationSpec();
  testRunManifestParsesOutputMetadataWithoutExpectedOutputs();
  testRunManifestParsesTaskOutputBinding();
  testRunManifestParsesDagSpec();
  testRunManifestParsesDagArtifactRootOverride();

  llvm::outs() << g_pass << " passed, " << g_fail << " failed\n";
  return g_fail ? 1 : 0;
}
