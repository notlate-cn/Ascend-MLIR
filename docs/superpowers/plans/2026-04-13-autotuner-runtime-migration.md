# Autotuner Runtime Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `tools/autotuner`'s legacy compile/execute/validate orchestration with a runtime-native implementation built on `ArtifactCompiler`, `ExecutionSession`, `SimBackend`, and runtime profiling outputs.

**Architecture:** The migrated autotuner becomes a single-task tiling search CLI. It compiles or loads one `KernelArtifact`, enumerates tiling candidates, executes each candidate through runtime-native simulation with validation enabled, parses runtime-produced profiling artifacts into a scalar score, and writes a JSON best-config summary.

**Tech Stack:** C++17, LLVM support libraries, existing runtime libraries (`ArtifactCompiler`, `ExecutionSession`, `RunManifest`, `ProfileTrace`), xvm simulator verification.

---

### Task 1: Define runtime-native autotuner data flow in `autotuner_main.cpp`

**Files:**
- Modify: `tools/autotuner/autotuner_main.cpp`
- Test: `tools/autotuner/autotuner_main.cpp` compile-only smoke on xvm

- [ ] **Step 1: Remove legacy runtime-centric includes and add runtime-native includes**

Replace the top include block so `autotuner_main.cpp` no longer depends on:

```cpp
#include "Runtime/Compiler.h"
#include "Runtime/Executor.h"
#include "Runtime/SimValidator.h"
#include "Runtime/HostRunnerGen.h"
```

and instead depends on:

```cpp
#include "Runtime/ArtifactCompiler.h"
#include "Runtime/ExecutionSession.h"
#include "Runtime/RunManifest.h"
#include "Runtime/TaskGraph.h"
#include "Runtime/ProfileTrace.h"
#include "Runtime/ProfileUtils.h"
#include "Runtime/NpyIO.h"
#include "Runtime/PathUtils.h"
```

- [ ] **Step 2: Add small focused structs for the new autotuner flow**

Add local structs near the existing `TilingSpace` declarations:

```cpp
struct SearchInputs {
  std::vector<std::string> inputFiles;
  std::string expectedFile;
  std::map<std::string, int64_t> shape;
  double atol = 1.0;
  double rtol = 1e-2;
};

struct CandidateExecutionSpec {
  std::vector<std::pair<std::string, int64_t>> params;
  int64_t blockDim = 1;
  std::string actualOutputPath;
  std::string profileOutputDir;
};
```

- [ ] **Step 3: Run a compile-only check to confirm the file still parses after include/struct edits**

Run on xvm after the first mechanical edit:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh >/dev/null
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
cmake --build build --target autotuner -j4
```

Expected: build reaches `autotuner` compile stage; it may still fail on missing yet-to-be-implemented symbols, but not on removed include paths or missing old headers.

- [ ] **Step 4: Commit**

```bash
git add tools/autotuner/autotuner_main.cpp
git commit -m "refactor: prepare autotuner for runtime-native flow"
```

### Task 2: Replace artifact preparation with `ArtifactCompiler` / artifact-root loading

**Files:**
- Modify: `tools/autotuner/autotuner_main.cpp`
- Reuse: `tools/runtime-session/runtime_session_main.cpp`
- Test: xvm `autotuner` build

- [ ] **Step 1: Add new CLI options for runtime-native entry**

In `autotuner_main.cpp`, keep `--space`, `--inputs`, `--expected`, `--shape`, `--atol`, `--rtol`, `--output`, but replace legacy compile assumptions with:

```cpp
static cl::opt<std::string> ArtifactRoot(
    "artifact-root",
    cl::desc("Existing artifact root to search without recompiling"),
    cl::init(""));
static cl::opt<std::string> KernelKindName(
    "kernel-kind",
    cl::desc("Kernel kind when compiling: vec, cube, or mix"),
    cl::init("vec"));
static cl::opt<std::string> BestConfigOut(
    "output",
    cl::desc("Best-config JSON output path"),
    cl::init("best_config.json"));
static cl::opt<std::string> ProfileOutDir(
    "profile-out",
    cl::desc("Directory for retained profiling artifacts"),
    cl::init(""));
```

- [ ] **Step 2: Port the artifact loading helpers from `runtime-session`**

Add local helper functions modeled on `tools/runtime-session/runtime_session_main.cpp`:

```cpp
static llvm::Expected<KernelKind> parseKernelKind(llvm::StringRef name);
static llvm::Expected<std::string> locateManifestPath(llvm::StringRef artifactRoot);
static llvm::Expected<KernelArtifact> loadArtifactFromRoot(llvm::StringRef artifactRoot);
```

Use the same manifest resolution rules as `runtime-session`:

- `out/manifest.txt`
- `mix-artifact.txt`
- `out/mix-artifact.txt`

- [ ] **Step 3: Add a single artifact preparation entry point**

Implement:

```cpp
static llvm::Expected<KernelArtifact>
prepareArtifact(const TilingSpace &space);
```

Behavior:

- if `--artifact-root` is provided, load existing artifact
- otherwise compile via `ArtifactCompiler`
- compilation uses:
  - `KernelFile` override first
  - else `tiling_space.json` `kernel_file`
  - `SocVersion` override first
  - else `tiling_space.json` `soc`
  - `KernelKindName` override first
  - else `tiling_space.json` `kernel_type`

- [ ] **Step 4: Replace old `Compiler` setup with `prepareArtifact()`**

Delete direct `Compiler::Config`, `Compiler compiler(...)`, and one-off build directory logic from the search path. Search should now receive a prepared `KernelArtifact`.

- [ ] **Step 5: Rebuild `autotuner` on xvm**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh >/dev/null
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
cmake --build build --target autotuner -j4
```

Expected: `autotuner` links or fails only on the next not-yet-migrated execution path, not on artifact preparation.

- [ ] **Step 6: Commit**

```bash
git add tools/autotuner/autotuner_main.cpp
git commit -m "refactor: move autotuner artifact setup to runtime compiler"
```

### Task 3: Replace candidate execution with `ExecutionSession(SimBackend)`

**Files:**
- Modify: `tools/autotuner/autotuner_main.cpp`
- Reuse: `include/Runtime/TaskGraph.h`
- Reuse: `include/Runtime/ExecutionSession.h`
- Test: xvm `autotuner` build and smoke

- [ ] **Step 1: Add helpers to build runtime bindings for a candidate**

Implement local helpers:

```cpp
static llvm::Expected<std::vector<TensorBinding>>
loadInputBindings(const std::vector<std::string> &inputFiles);

static llvm::Expected<TensorBinding>
buildOutputBinding(const std::string &expectedFile,
                   const std::string &actualOutputPath);

static llvm::Expected<TensorBinding>
buildExpectedOutputBinding(const std::string &expectedFile);

static std::optional<TilingBinding>
buildTilingBinding(const std::vector<uint8_t> &tilingBytes,
                   llvm::StringRef workingDir);
```
```

- [ ] **Step 2: Build a runtime-native candidate graph**

Implement:

```cpp
static llvm::Expected<TaskGraph>
buildCandidateGraph(const KernelArtifact &artifact,
                    const SearchInputs &inputs,
                    const CandidateExecutionSpec &candidate);
```

The graph should contain a single `RuntimeTask` with:

- external-file input bindings from `--inputs`
- one output binding written to the candidate output path
- one expected-output binding from `--expected`
- packed tiling bytes written into a per-candidate temporary binary file
- `enableProfiling = true`
- candidate-specific `blockDim`
- default workspace carried over from current autotuner behavior

- [ ] **Step 3: Replace legacy `Executor` / `SimValidator` execution with `ExecutionSession`**

Replace the old search loop execution body with:

```cpp
ExecutionSession session(ExecutionBackendKind::Simulation);
auto graphOr = buildCandidateGraph(artifact, searchInputs, candidateSpec);
auto traceOr = session.run(*graphOr);
```

Candidate policy:

- success only if `session.run()` succeeds
- runtime validation failure excludes the candidate
- runtime output file should be left on disk only for the best config

- [ ] **Step 4: Remove legacy execution objects**

Delete:

- `Executor executor;`
- `executor.Initialize()`
- direct `Run` / `RunFile`
- direct `SimValidator validator;`
- direct compare logic in the candidate loop

- [ ] **Step 5: Build and run a minimal autotuner smoke invocation on xvm**

Use a real generated vec kernel:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh >/dev/null
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
cmake --build build --target autotuner -j4
build/bin/autotuner \
  --space examples/relu-broadcast-transpose/tiling_space.json \
  --kernel examples/relu-broadcast-transpose/step8_kernel.cpp \
  --kernel-kind vec \
  --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy \
  --expected examples/relu-broadcast-transpose/output_expected.npy \
  --shape M=640,N=500 \
  --output /tmp/autotuner-best.json
```

Expected: the command runs candidates through runtime simulation and writes `/tmp/autotuner-best.json`; profiling-based ranking may still be stubbed at this point.

- [ ] **Step 6: Commit**

```bash
git add tools/autotuner/autotuner_main.cpp
git commit -m "refactor: run autotuner candidates through execution session"
```

### Task 4: Add runtime profiling score extraction for autotuner

**Files:**
- Modify: `tools/autotuner/autotuner_main.cpp`
- Optionally Modify: `lib/Runtime/ProfileUtils.cpp`
- Optionally Modify: `include/Runtime/ProfileUtils.h`
- Test: xvm autotuner smoke with profile-backed scoring

- [ ] **Step 1: Inspect runtime profiling artifact format from a real sim run**

Run on xvm:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh >/dev/null
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
find /tmp/ascendc-runtime -name '*.json' | tail -n 10
```

Expected: identify the simulator profile JSON path and the field carrying the cycle-like score.

- [ ] **Step 2: Add a focused parser helper**

Implement one reusable helper, either local to `autotuner_main.cpp` or in `ProfileUtils`, with this shape:

```cpp
static llvm::Expected<int64_t>
extractRuntimeScore(const ProfileTrace &trace);
```

Rules:

- use the first profile artifact path in `trace.profileArtifactPaths()`
- parse the simulator profiling JSON
- extract a single integer score
- return an error if no profile artifact exists or the expected field is missing

- [ ] **Step 3: Use runtime profile score in best-config selection**

Replace old cycle-count collection with:

```cpp
auto scoreOr = extractRuntimeScore(*traceOr);
if (!scoreOr) {
  // mark candidate failed with profiling parse error
}
result.cycle_count = *scoreOr;
result.passed = true;
```

Best-config selection remains:

- only passed candidates compete
- lower score wins

- [ ] **Step 4: Re-run the xvm autotuner smoke and inspect JSON output**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh >/dev/null
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
build/bin/autotuner \
  --space examples/relu-broadcast-transpose/tiling_space.json \
  --kernel examples/relu-broadcast-transpose/step8_kernel.cpp \
  --kernel-kind vec \
  --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy \
  --expected examples/relu-broadcast-transpose/output_expected.npy \
  --shape M=640,N=500 \
  --output /tmp/autotuner-best.json
cat /tmp/autotuner-best.json
```

Expected: the JSON contains a best config with a non-negative score derived from runtime profiling.

- [ ] **Step 5: Commit**

```bash
git add tools/autotuner/autotuner_main.cpp include/Runtime/ProfileUtils.h lib/Runtime/ProfileUtils.cpp
git commit -m "feat: score autotuner candidates from runtime profiling"
```

### Task 5: Remove obsolete autotuner-only legacy code and legacy flags

**Files:**
- Modify: `tools/autotuner/autotuner_main.cpp`
- Test: xvm autotuner build and smoke

- [ ] **Step 1: Delete runner/perf-report legacy branches**

Remove:

- `PerfReport`
- `MsprofPath`
- `PerfReportOutDir`
- `HostRunnerGen` usage
- any code that shells out to `msprof`
- any post-search legacy runner generation

- [ ] **Step 2: Remove no-longer-valid CLI help text and dead helpers**

Delete helpers that only existed for the old direct execution path, keeping:

- search-space parsing
- shape parsing
- block-dim expression evaluation
- tiling byte packing

- [ ] **Step 3: Rebuild and rerun the xvm autotuner smoke**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh >/dev/null
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
cmake --build build --target autotuner -j4
build/bin/autotuner --help
build/bin/autotuner \
  --space examples/relu-broadcast-transpose/tiling_space.json \
  --kernel examples/relu-broadcast-transpose/step8_kernel.cpp \
  --kernel-kind vec \
  --inputs examples/relu-broadcast-transpose/input_data0.npy,examples/relu-broadcast-transpose/input_data1.npy \
  --expected examples/relu-broadcast-transpose/output_expected.npy \
  --shape M=640,N=500 \
  --output /tmp/autotuner-best.json
```

Expected: `--help` no longer advertises legacy runner/perf-report flow, and the smoke invocation still succeeds.

- [ ] **Step 4: Commit**

```bash
git add tools/autotuner/autotuner_main.cpp
git commit -m "refactor: drop legacy autotuner execution paths"
```

### Task 6: Add focused verification and keep runtime/examples green

**Files:**
- Modify: `test/tools/runtime/run_runtime.sh` only if an autotuner smoke belongs there
- Optionally Create: `test/tools/runtime/run_autotuner_smoke.sh`
- Test: xvm runtime/examples regressions

- [ ] **Step 1: Add one focused autotuner smoke entry**

Prefer a dedicated script:

```bash
test/tools/runtime/run_autotuner_smoke.sh
```

It should:

- source `examples/env.sh`
- require `autotuner`, `runtime-session`, `afir-opt`, `afir-translate`
- run a real vec autotuner smoke on xvm
- assert best-config JSON exists and contains `"score"`

- [ ] **Step 2: Run runtime and examples regressions after the migration**

Run on xvm:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
source examples/env.sh >/dev/null
export LLVM_BUILD_DIR=/home/niu/code/llvm-project/llvm/build
bash test/tools/runtime/run_runtime.sh
bash test/tools/examples/example_pipelines.sh
bash test/tools/runtime/run_autotuner_smoke.sh
```

Expected:

- `run_runtime.sh` passes
- `example_pipelines.sh` passes
- autotuner smoke passes

- [ ] **Step 3: Commit**

```bash
git add test/tools/runtime/run_autotuner_smoke.sh test/tools/runtime/run_runtime.sh
git commit -m "test: add runtime-native autotuner smoke"
```

### Task 7: Final cleanup review of old autotuner dependencies

**Files:**
- Modify: `tools/autotuner/autotuner_main.cpp` if needed
- Review: `tools/autotuner/CMakeLists.txt`

- [ ] **Step 1: Audit includes and direct legacy dependencies**

Confirm `tools/autotuner/autotuner_main.cpp` no longer includes:

```cpp
#include "Runtime/Compiler.h"
#include "Runtime/Executor.h"
#include "Runtime/SimValidator.h"
#include "Runtime/HostRunnerGen.h"
```

- [ ] **Step 2: Audit runtime behavior assumptions**

Confirm the autotuner now depends on:

- `ArtifactCompiler`
- artifact loading helpers
- `ExecutionSession`
- runtime profiling extraction

and does not shell out to separate profiler tooling.

- [ ] **Step 3: Run a final grep-based audit**

Run:

```bash
cd /Users/niu/Code/Codex-Ascend-MLIR
rg -n "Compiler.h|Executor.h|SimValidator.h|HostRunnerGen.h|msprof|perf-report" tools/autotuner
```

Expected: no matches that reflect active autotuner implementation.

- [ ] **Step 4: Commit**

```bash
git add tools/autotuner/autotuner_main.cpp tools/autotuner/CMakeLists.txt
git commit -m "chore: finalize autotuner runtime migration"
```
