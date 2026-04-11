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
#include "Runtime/RunManifest.h"
#include "Runtime/ExecutionBackend.h"
#include "Runtime/ExecutionSession.h"
#include "Runtime/NpuBackend.h"
#include "Runtime/TaskGraph.h"
#include "Runtime/ArtifactCompiler.h"
#include "Runtime/SimBackend.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace mlir::runtime;

static int g_pass = 0;
static int g_fail = 0;

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
    EXPECT(planOr->orderedTaskIds[0] == "task_a",
           "execution session plan first task");
    EXPECT(planOr->orderedTaskIds[1] == "task_b",
           "execution session plan second task");
    EXPECT(planOr->orderedTaskIds[2] == "task_c",
           "execution session plan third task");
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
  "profiling": true
})JSON";
  }

  auto specOr = loadRunManifest(manifestPath);
  EXPECT((bool)specOr, "run manifest parse succeeds");
  if (specOr) {
    EXPECT(specOr->taskId == "main", "run manifest task id");
    EXPECT(specOr->backendKind == ExecutionBackendKind::Simulation,
           "run manifest backend kind");
    EXPECT(specOr->artifactRoot == "/tmp/artifact",
           "run manifest artifact root");
    EXPECT(specOr->invocation.inputs.size() == 2,
           "run manifest input count");
    EXPECT(specOr->invocation.outputs.size() == 1,
           "run manifest output count");
    EXPECT(specOr->invocation.expectedOutputs.size() == 1,
           "run manifest expected output count");
    EXPECT(specOr->invocation.blockDim == 8,
           "run manifest block dim");
    EXPECT(specOr->invocation.workspaceSize == 16384,
           "run manifest workspace size");
    EXPECT(specOr->invocation.enableProfiling,
           "run manifest profiling flag");
    EXPECT(specOr->invocation.tiling.has_value(),
           "run manifest tiling present");
    EXPECT(specOr->invocation.outputs[0].shape.has_value(),
           "run manifest output shape metadata present");
    EXPECT(specOr->invocation.outputs[0].dtype.has_value(),
           "run manifest output dtype metadata present");
    if (specOr->invocation.outputs[0].shape) {
      EXPECT(specOr->invocation.outputs[0].shape->size() == 2 &&
                 (*specOr->invocation.outputs[0].shape)[0] == 4 &&
                 (*specOr->invocation.outputs[0].shape)[1] == 8,
             "run manifest output shape metadata values");
    }
    if (specOr->invocation.outputs[0].dtype) {
      EXPECT(*specOr->invocation.outputs[0].dtype == DType::F32,
             "run manifest output dtype metadata value");
    }
    if (specOr->invocation.tiling) {
      EXPECT(specOr->invocation.tiling->schemaPath == "/tmp/tiling_space.json",
             "run manifest tiling schema path");
      EXPECT(specOr->invocation.tiling->params == "TB_M=64,TB_N=64",
             "run manifest tiling params");
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
    EXPECT(specOr->invocation.expectedOutputs.empty(),
           "run manifest without expected outputs leaves golden bindings empty");
    EXPECT(specOr->invocation.outputs.size() == 1,
           "run manifest without expected outputs keeps output bindings");
    if (specOr->invocation.outputs.size() == 1) {
      EXPECT(specOr->invocation.outputs[0].shape.has_value(),
             "run manifest without expected outputs carries output shape");
      EXPECT(specOr->invocation.outputs[0].dtype.has_value(),
             "run manifest without expected outputs carries output dtype");
      if (specOr->invocation.outputs[0].shape) {
        EXPECT(specOr->invocation.outputs[0].shape->size() == 1 &&
                   (*specOr->invocation.outputs[0].shape)[0] == 32,
               "run manifest without expected outputs shape value");
      }
      if (specOr->invocation.outputs[0].dtype) {
        EXPECT(*specOr->invocation.outputs[0].dtype == DType::F16,
               "run manifest without expected outputs dtype value");
      }
    }
  }
}

int main() {
  testTaskGraphBasics();
  testDuplicateTaskIds();
  testEmptyTaskId();
  testUnknownDependency();
  testCycleDetection();
  testKernelArtifactNormalization();
  testArtifactCompilerRequestValidation();
  testVecCompileCreatesOutputDir();
  testBackendSelection();
  testDefaultBackendRequiresDriver();
  testInvalidBackendSelection();
  testBackendDelegatesToDriver();
  testSimulatorProfileNormalization();
  testAddProfileArtifactHelper();
  testBackendSurfacesProfileTrace();
  testBackendPreservesExistingProfileTrace();
  testExecutionSessionPlansTopologicalOrder();
  testExecutionSessionRunsTasksInTopologicalOrder();
  testExecutionSessionCarriesInvocationBindings();
  testRunManifestParsesVecSimulationSpec();
  testRunManifestParsesOutputMetadataWithoutExpectedOutputs();

  llvm::outs() << g_pass << " passed, " << g_fail << " failed\n";
  return g_fail ? 1 : 0;
}
