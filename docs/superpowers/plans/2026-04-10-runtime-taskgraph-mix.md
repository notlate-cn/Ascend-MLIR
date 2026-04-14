# Runtime TaskGraph Mix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-center `lib/Runtime` around task-graph runtime abstractions while preserving existing AscendC/mix compile and execution flows as backend implementations, and deliver compile, simulator+profiling, and NPU execution paths for single-task and conservative DAG execution.

**Architecture:** Introduce stable runtime objects (`KernelArtifact`, `RuntimeTask`, `TaskGraph`, `ExecutionSession`, `ExecutionBackend`, `ProfileTrace`) as the public center of `lib/Runtime`. Normalize existing `Compiler`, `MixDirectBackend`, `Executor`, host runner, and validator logic behind compile/runtime backend boundaries. Phase one supports dependency-correct DAG execution with serial or limited parallel scheduling and explicit mix resource metadata (`AIVOnly`, `AICOnly`, `Mix1C1V`, `Mix1C2V`).

**Tech Stack:** C++17, LLVM Support, existing `AscendCRuntime` library, CMake, simulator/NPU CANN runtime libraries, existing runtime tests and runtime/mix CLI tools.

---

### Task 1: Add Core Runtime Data Model

**Files:**
- Create: `include/Runtime/TaskGraph.h`
- Create: `lib/Runtime/TaskGraph.cpp`
- Create: `include/Runtime/ProfileTrace.h`
- Create: `lib/Runtime/ProfileTrace.cpp`
- Modify: `lib/Runtime/CMakeLists.txt`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing test for graph/task/profile object behavior**

```cpp
#include "Runtime/ProfileTrace.h"
#include "Runtime/TaskGraph.h"
#include "llvm/Support/Error.h"
#include <string>

using namespace mlir::runtime;

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

  auto errA = graph.addTask(taskA);
  if (errA)
    llvm::report_fatal_error(llvm::toString(std::move(errA)));
  auto errB = graph.addTask(taskB);
  if (errB)
    llvm::report_fatal_error(llvm::toString(std::move(errB)));

  auto orderOr = graph.topologicalOrder();
  if (!orderOr)
    llvm::report_fatal_error(llvm::toString(orderOr.takeError()));

  if (orderOr->size() != 2 || (*orderOr)[0] != "task_a" || (*orderOr)[1] != "task_b")
    llvm::report_fatal_error("unexpected task order");

  ProfileTrace trace;
  trace.sessionId = "sess0";
  trace.events.push_back(ProfileEvent{
      .taskId = "task_a",
      .backend = ExecutionBackendKind::Simulation,
      .eventKind = "kernel_complete",
      .artifact = "trace.json"});
  if (trace.events.size() != 1)
    llvm::report_fatal_error("expected one event");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build && source examples/env.sh && c++ -std=c++17 -I include -I /home/niu/code/llvm-project/build/include -I /home/niu/code/llvm-project/llvm/include test/tools/runtime/test_taskgraph_runtime.cpp build/lib/libAscendCRuntime.a $(/home/niu/code/llvm-project/build/bin/llvm-config --ldflags --libs support) -ldl -o /tmp/test_taskgraph_runtime && /tmp/test_taskgraph_runtime'`
Expected: FAIL with missing `Runtime/TaskGraph.h` or undefined runtime task graph symbols.

- [ ] **Step 3: Write minimal runtime object model**

```cpp
enum class KernelKind { Vec, Cube, Mix };
enum class MixResourceType { Unknown, AIVOnly, AICOnly, Mix1C1V, Mix1C2V };
enum class ExecutionBackendKind { Simulation, Npu };

struct KernelArtifact {
  std::string kernelName;
  KernelKind kernelKind = KernelKind::Vec;
  MixResourceType mixResourceType = MixResourceType::Unknown;
  std::string socVersion;
  std::string artifactRoot;
  std::string deviceBinaryPath;
  std::string packedSharedObjectPath;
  std::string manifestPath;
};

struct RuntimeTask {
  std::string taskId;
  KernelArtifact artifact;
  std::vector<std::string> dependencies;
};

class TaskGraph {
public:
  llvm::Error addTask(const RuntimeTask &task);
  llvm::Expected<std::vector<std::string>> topologicalOrder() const;
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build && source examples/env.sh && c++ -std=c++17 -I include -I /home/niu/code/llvm-project/build/include -I /home/niu/code/llvm-project/llvm/include test/tools/runtime/test_taskgraph_runtime.cpp build/lib/libAscendCRuntime.a $(/home/niu/code/llvm-project/build/bin/llvm-config --ldflags --libs support) -ldl -o /tmp/test_taskgraph_runtime && /tmp/test_taskgraph_runtime'`
Expected: PASS with task graph creation and topological ordering succeeding.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/TaskGraph.h lib/Runtime/TaskGraph.cpp include/Runtime/ProfileTrace.h lib/Runtime/ProfileTrace.cpp lib/Runtime/CMakeLists.txt test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: add runtime task graph core model"
```

### Task 2: Normalize Compile Outputs Into `KernelArtifact`

**Files:**
- Create: `include/Runtime/ArtifactCompiler.h`
- Create: `lib/Runtime/ArtifactCompiler.cpp`
- Modify: `include/Runtime/MixArtifact.h`
- Modify: `include/Runtime/Compiler.h`
- Modify: `lib/Runtime/Compiler.cpp`
- Modify: `include/Runtime/MixDirectBackend.h`
- Modify: `lib/Runtime/MixDirectBackend.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing compile-normalization test**

```cpp
static void testKernelArtifactNormalization() {
  ArtifactCompileRequest req;
  req.kernelName = "demo_kernel";
  req.kernelSource = "/tmp/demo_kernel.cpp";
  req.kernelKind = KernelKind::Mix;
  req.socVersion = "Ascend910B1";

  auto kind = inferMixResourceTypeFromKernelKind(req.kernelKind);
  if (kind != MixResourceType::Mix1C1V)
    llvm::report_fatal_error("expected default mix resource type");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build'`
Expected: FAIL because `ArtifactCompileRequest` and compile normalization helpers do not exist.

- [ ] **Step 3: Add `ArtifactCompiler` that wraps existing compile backends**

```cpp
struct ArtifactCompileRequest {
  std::string kernelSource;
  std::string kernelName;
  KernelKind kernelKind = KernelKind::Vec;
  std::string socVersion;
  std::string outputDir;
  std::optional<std::string> cannMlirPath;
  std::optional<std::string> npyDir;
};

class ArtifactCompiler {
public:
  llvm::Expected<KernelArtifact> compile(const ArtifactCompileRequest &req);
};

llvm::Expected<KernelArtifact> ArtifactCompiler::compile(const ArtifactCompileRequest &req) {
  if (req.kernelKind == KernelKind::Mix) {
    MixDirectCompileConfig cfg;
    cfg.kernelSrc = req.kernelSource;
    cfg.kernelName = req.kernelName;
    cfg.socVersion = req.socVersion;
    cfg.outputDir = req.outputDir;
    cfg.cannMlirPath = req.cannMlirPath;
    cfg.npyDir = req.npyDir;
    auto mixOr = MixDirectBackend().compile(cfg);
    if (!mixOr)
      return mixOr.takeError();

    KernelArtifact artifact;
    artifact.kernelName = mixOr->kernel_name;
    artifact.kernelKind = KernelKind::Mix;
    artifact.mixResourceType = MixResourceType::Mix1C1V;
    artifact.socVersion = mixOr->soc_version;
    artifact.artifactRoot = mixOr->work_dir;
    artifact.packedSharedObjectPath = mixOr->kernel_so_path;
    artifact.manifestPath = mixOr->manifest_path;
    return artifact;
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "vec/cube normalization not implemented");
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build && source examples/env.sh && ctest --test-dir build --output-on-failure -R runtime'`
Expected: PASS for updated runtime normalization tests, with compile wrappers building successfully.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/ArtifactCompiler.h lib/Runtime/ArtifactCompiler.cpp include/Runtime/MixArtifact.h include/Runtime/Compiler.h lib/Runtime/Compiler.cpp include/Runtime/MixDirectBackend.h lib/Runtime/MixDirectBackend.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: normalize runtime compile outputs into kernel artifacts"
```

### Task 3: Introduce Unified Execution Backend Interface

**Files:**
- Create: `include/Runtime/ExecutionBackend.h`
- Create: `lib/Runtime/ExecutionBackend.cpp`
- Create: `include/Runtime/SimBackend.h`
- Create: `lib/Runtime/SimBackend.cpp`
- Create: `include/Runtime/NpuBackend.h`
- Create: `lib/Runtime/NpuBackend.cpp`
- Modify: `include/Runtime/Executor.h`
- Modify: `lib/Runtime/Executor.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing backend selection test**

```cpp
static void testBackendSelection() {
  RuntimeTask task;
  task.taskId = "single";
  task.artifact.kernelName = "mix_add";
  task.artifact.kernelKind = KernelKind::Mix;
  task.artifact.mixResourceType = MixResourceType::Mix1C1V;

  auto sim = createExecutionBackend(ExecutionBackendKind::Simulation);
  auto npu = createExecutionBackend(ExecutionBackendKind::Npu);
  if (!sim || !npu)
    llvm::report_fatal_error("expected both backends");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build'`
Expected: FAIL because backend factory and concrete backend classes do not exist.

- [ ] **Step 3: Add common execution backend interface and adapter backends**

```cpp
struct ExecutionRequest {
  RuntimeTask task;
  std::string workingDirectory;
};

struct ExecutionResult {
  std::string taskId;
  std::vector<std::string> producedFiles;
};

class ExecutionBackend {
public:
  virtual ~ExecutionBackend() = default;
  virtual ExecutionBackendKind kind() const = 0;
  virtual llvm::Expected<ExecutionResult> run(const ExecutionRequest &request) = 0;
};

std::unique_ptr<ExecutionBackend> createExecutionBackend(ExecutionBackendKind kind);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build && source examples/env.sh && ctest --test-dir build --output-on-failure -R runtime'`
Expected: PASS with backend construction succeeding and runtime library linking the new backend units.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/ExecutionBackend.h lib/Runtime/ExecutionBackend.cpp include/Runtime/SimBackend.h lib/Runtime/SimBackend.cpp include/Runtime/NpuBackend.h lib/Runtime/NpuBackend.cpp include/Runtime/Executor.h lib/Runtime/Executor.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: add unified runtime execution backends"
```

### Task 4: Implement Simulator Profiling Integration

**Files:**
- Modify: `include/Runtime/ProfileTrace.h`
- Modify: `lib/Runtime/ProfileTrace.cpp`
- Modify: `lib/Runtime/SimBackend.cpp`
- Create: `include/Runtime/ProfileUtils.h`
- Create: `lib/Runtime/ProfileUtils.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing simulator profiling test**

```cpp
static void testProfileTraceMapping() {
  ProfileTrace trace;
  trace.sessionId = "sess0";
  addProfileArtifact(trace, "task0", ExecutionBackendKind::Simulation,
                     "/tmp/opprof/simulator/trace.json");
  if (trace.events.empty() || trace.events.front().taskId != "task0")
    llvm::report_fatal_error("expected task-mapped profile event");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build'`
Expected: FAIL because normalized profile mapping helpers do not exist.

- [ ] **Step 3: Add simulator profile normalization helpers**

```cpp
struct ProfileEvent {
  std::string taskId;
  ExecutionBackendKind backend = ExecutionBackendKind::Simulation;
  std::string eventKind;
  std::string artifact;
};

void addProfileArtifact(ProfileTrace &trace, llvm::StringRef taskId,
                        ExecutionBackendKind backend, llvm::StringRef artifactPath) {
  trace.events.push_back(ProfileEvent{
      .taskId = taskId.str(),
      .backend = backend,
      .eventKind = "profile_artifact",
      .artifact = artifactPath.str()});
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build && source examples/env.sh && ctest --test-dir build --output-on-failure -R runtime'`
Expected: PASS with task-level profile event mapping working in unit tests.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/ProfileTrace.h lib/Runtime/ProfileTrace.cpp include/Runtime/ProfileUtils.h lib/Runtime/ProfileUtils.cpp lib/Runtime/SimBackend.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: normalize simulator profiling traces"
```

### Task 5: Add `ExecutionSession` and Conservative DAG Execution

**Files:**
- Create: `include/Runtime/ExecutionSession.h`
- Create: `lib/Runtime/ExecutionSession.cpp`
- Modify: `include/Runtime/TaskGraph.h`
- Modify: `lib/Runtime/TaskGraph.cpp`
- Test: `test/tools/runtime/test_taskgraph_runtime.cpp`

- [ ] **Step 1: Write the failing execution-session test**

```cpp
static void testExecutionSessionTopoRun() {
  TaskGraph graph;
  RuntimeTask taskA;
  taskA.taskId = "A";
  RuntimeTask taskB;
  taskB.taskId = "B";
  taskB.dependencies = {"A"};

  if (auto err = graph.addTask(taskA))
    llvm::report_fatal_error(llvm::toString(std::move(err)));
  if (auto err = graph.addTask(taskB))
    llvm::report_fatal_error(llvm::toString(std::move(err)));

  ExecutionSession session(ExecutionBackendKind::Simulation);
  auto resultOr = session.plan(graph);
  if (!resultOr)
    llvm::report_fatal_error(llvm::toString(resultOr.takeError()));
  if (resultOr->orderedTaskIds.size() != 2 || resultOr->orderedTaskIds[0] != "A")
    llvm::report_fatal_error("unexpected session plan");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build'`
Expected: FAIL because `ExecutionSession` does not exist.

- [ ] **Step 3: Add conservative execution session**

```cpp
struct SessionPlan {
  std::vector<std::string> orderedTaskIds;
};

class ExecutionSession {
public:
  explicit ExecutionSession(ExecutionBackendKind backendKind);
  llvm::Expected<SessionPlan> plan(const TaskGraph &graph) const;
  llvm::Expected<ProfileTrace> run(const TaskGraph &graph);

private:
  ExecutionBackendKind backendKind_;
};

llvm::Expected<SessionPlan> ExecutionSession::plan(const TaskGraph &graph) const {
  auto orderOr = graph.topologicalOrder();
  if (!orderOr)
    return orderOr.takeError();
  return SessionPlan{.orderedTaskIds = *orderOr};
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build && source examples/env.sh && ctest --test-dir build --output-on-failure -R runtime'`
Expected: PASS with DAG planning and ordered execution semantics validated.

- [ ] **Step 5: Commit**

```bash
git add include/Runtime/ExecutionSession.h lib/Runtime/ExecutionSession.cpp include/Runtime/TaskGraph.h lib/Runtime/TaskGraph.cpp test/tools/runtime/test_taskgraph_runtime.cpp
git commit -m "feat: add conservative DAG execution session"
```

### Task 6: Rework CLI Entry Points Around Artifacts and Sessions

**Files:**
- Modify: `tools/mix-compiler/mix_compiler_main.cpp`
- Modify: `tools/mix-validator/mix_validator_main.cpp`
- Create: `tools/runtime-session/runtime_session_main.cpp`
- Create: `tools/runtime-session/CMakeLists.txt`
- Modify: `tools/CMakeLists.txt`
- Test: `test/tools/runtime/run_runtime.sh`

- [ ] **Step 1: Write the failing CLI integration check**

```bash
test -x build/bin/runtime-session
build/bin/runtime-session --help | grep -q "task graph runtime"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build && source examples/env.sh && test -x build/bin/runtime-session'`
Expected: FAIL because the new CLI does not exist.

- [ ] **Step 3: Add artifact/session-oriented CLI wiring**

```cpp
int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "task graph runtime\n");

  ArtifactCompiler compiler;
  auto artifactOr = compiler.compile(req);
  if (!artifactOr) {
    llvm::errs() << llvm::toString(artifactOr.takeError()) << "\n";
    return 2;
  }

  TaskGraph graph;
  RuntimeTask task;
  task.taskId = "root";
  task.artifact = *artifactOr;
  if (auto err = graph.addTask(task)) {
    llvm::errs() << llvm::toString(std::move(err)) << "\n";
    return 3;
  }

  ExecutionSession session(ExecutionBackendKind::Simulation);
  auto profileOr = session.run(graph);
  if (!profileOr) {
    llvm::errs() << llvm::toString(profileOr.takeError()) << "\n";
    return 4;
  }
  return 0;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `ssh xvm@orb 'cd /home/niu/code/Ascend-MLIR && ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build && source examples/env.sh && build/bin/runtime-session --help >/tmp/runtime_session_help.txt && grep -q "task graph runtime" /tmp/runtime_session_help.txt'`
Expected: PASS with the new CLI built and exposing task-graph runtime help text.

- [ ] **Step 5: Commit**

```bash
git add tools/mix-compiler/mix_compiler_main.cpp tools/mix-validator/mix_validator_main.cpp tools/runtime-session/runtime_session_main.cpp tools/runtime-session/CMakeLists.txt tools/CMakeLists.txt test/tools/runtime/run_runtime.sh
git commit -m "feat: add task graph runtime cli"
```

### Task 7: End-to-End Verification In xvm

**Files:**
- Modify: `test/tools/runtime/run_runtime.sh`
- Modify: `examples/dev-env.md`
- Test: `test/tools/runtime/run_runtime.sh`

- [ ] **Step 1: Write the failing end-to-end verification flow**

```bash
#!/usr/bin/env bash
set -euo pipefail
ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR
  ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build
  source examples/env.sh
  build/bin/runtime-session --help
'
```

- [ ] **Step 2: Run it to verify it fails before doc/script updates**

Run: `bash test/tools/runtime/run_runtime.sh`
Expected: FAIL if the runtime-session flow is not yet wired into the documented dev path.

- [ ] **Step 3: Update verification script and dev documentation**

```bash
ssh xvm@orb '
  cd /home/niu/code/Ascend-MLIR
  ./scripts/build.sh --build-project --llvm-build-dir /home/niu/code/llvm-project/build
  source examples/env.sh
  ctest --test-dir build --output-on-failure -R runtime
  build/bin/runtime-session --help
'
```

- [ ] **Step 4: Run it to verify it passes**

Run: `bash test/tools/runtime/run_runtime.sh`
Expected: PASS with build, runtime tests, and task-graph runtime CLI all succeeding in `xvm`.

- [ ] **Step 5: Commit**

```bash
git add test/tools/runtime/run_runtime.sh examples/dev-env.md
git commit -m "test: add xvm verification flow for task graph runtime"
```

## Self-Review

### Spec coverage

- compile AscendC kernel: covered by Task 2
- CPU simulation execution: covered by Task 3 and Task 5
- profiling collection: covered by Task 4
- NPU execution path: covered by Task 3
- task-graph runtime center: covered by Task 1 and Task 5
- future multi-task DAG orchestration path: enabled by Task 1, Task 3, and Task 5
- xvm-based development workflow: covered by Task 7

### Placeholder scan

- No `TODO` or `TBD` placeholders remain in task steps
- Each task includes concrete files, concrete commands, and concrete target code snippets

### Type consistency

- shared enums and object names are consistently referenced as `KernelKind`, `MixResourceType`, `ExecutionBackendKind`, `KernelArtifact`, `RuntimeTask`, `TaskGraph`, `ExecutionSession`, and `ProfileTrace`
- compile path consistently normalizes into `KernelArtifact`
- runtime execution consistently consumes `RuntimeTask` through `ExecutionBackend`
