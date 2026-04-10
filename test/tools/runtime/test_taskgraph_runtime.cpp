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
#include "Runtime/TaskGraph.h"
#include "Runtime/ArtifactCompiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <string>
#include <vector>

using namespace mlir::runtime;

static int g_pass = 0;
static int g_fail = 0;

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

int main() {
  testTaskGraphBasics();
  testDuplicateTaskIds();
  testEmptyTaskId();
  testUnknownDependency();
  testCycleDetection();
  testKernelArtifactNormalization();
  testArtifactCompilerRequestValidation();
  testVecCompileCreatesOutputDir();

  llvm::outs() << g_pass << " passed, " << g_fail << " failed\n";
  return g_fail ? 1 : 0;
}
